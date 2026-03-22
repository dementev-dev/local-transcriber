"""Бэкенд транскрипции на основе faster-whisper (CTranslate2)."""

from __future__ import annotations

import gc
import io
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

# CUDA bootstrap — должен быть ДО импорта faster_whisper / ctranslate2
from local_transcriber._cuda_bootstrap import ensure_cublas_loadable

ensure_cublas_loadable()

from faster_whisper import WhisperModel  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from huggingface_hub.errors import LocalEntryNotFoundError  # noqa: E402

from local_transcriber.types import Segment, TranscribeResult  # noqa: E402

MODEL_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
}

MODEL_ALLOW_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]

MODEL_REQUIRED_FILES = [
    "config.json",
    "model.bin",
    "tokenizer.json",
]


class FasterWhisperBackend:
    """Бэкенд транскрипции через faster-whisper (CTranslate2)."""

    def __init__(self):
        self.actual_compute_type: str | None = None

    def ensure_model_available(
        self,
        model_name: str,
        compute_type: str,
        on_status: Callable[[str], None] | None = None,
    ) -> str:
        """Резолвит alias модели в repo_id и гарантирует наличие файлов."""
        self.actual_compute_type = compute_type
        local_path = Path(model_name).expanduser()
        if local_path.is_dir():
            _validate_model_dir(local_path)
            return str(local_path)

        repo_id = _resolve_model_repo(model_name)

        try:
            _notify(on_status, f"Проверяю кэш модели {model_name}...")
            cached_path = Path(_snapshot_download(repo_id, local_files_only=True))
            _validate_model_dir(cached_path)
            return str(cached_path)
        except LocalEntryNotFoundError:
            pass
        except ValueError:
            _notify(on_status, f"Кэш модели {model_name} неполный, докачиваю...")

        _notify(on_status, f"Скачиваю модель {model_name} из Hugging Face...")
        downloaded_path = Path(_snapshot_download(repo_id, local_files_only=False))
        _validate_model_dir(downloaded_path)
        return str(downloaded_path)

    def create_model(
        self,
        model_path: str,
        device: str,
        compute_type: str,
    ) -> Any:
        """Создаёт WhisperModel."""
        try:
            return WhisperModel(model_path, device=device, compute_type=compute_type)
        except ImportError as exc:
            if _is_missing_socksio_error(exc):
                raise RuntimeError(
                    "Обнаружен SOCKS proxy, но не установлена зависимость `socksio`, "
                    "нужная для загрузки модели из Hugging Face через proxy. "
                    "Обновите окружение: `uv sync`."
                ) from exc
            raise

    def transcribe(
        self,
        model: Any,
        file_path: Path,
        language: str | None,
        on_segment: Callable[[Segment], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> TranscribeResult:
        """Транскрибирует файл через faster-whisper."""
        segment_generator, info = model.transcribe(
            str(file_path), language=language,
        )
        total_duration = info.duration
        segments: list[Segment] = []
        for raw_seg in segment_generator:
            seg = Segment(start=raw_seg.start, end=raw_seg.end, text=raw_seg.text)
            if on_segment is not None:
                on_segment(seg)
            segments.append(seg)
            _notify(
                on_status,
                f"Транскрибирую... {_fmt_time(seg.end)} / {_fmt_time(total_duration)}"
                f"  [{len(segments)} сегм.]",
            )

        return TranscribeResult(
            segments=segments,
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
            device_used="",  # оркестратор проставит actual_device
        )


def _notify(on_status: Callable[[str], None] | None, message: str) -> None:
    if on_status is not None:
        on_status(message)


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _resolve_model_repo(model_name: str) -> str:
    if "/" in model_name:
        return model_name
    repo_id = MODEL_REPOS.get(model_name)
    if repo_id is None:
        expected = ", ".join(MODEL_REPOS)
        raise ValueError(f"Неподдерживаемая модель '{model_name}'. Ожидалось одно из: {expected}")
    return repo_id


def _snapshot_download(repo_id: str, local_files_only: bool) -> str:
    try:
        return snapshot_download(
            repo_id,
            local_files_only=local_files_only,
            allow_patterns=MODEL_ALLOW_PATTERNS,
        )
    except ImportError as exc:
        if _is_missing_socksio_error(exc):
            raise RuntimeError(
                "Обнаружен SOCKS proxy, но не установлена зависимость `socksio`, "
                "нужная для загрузки модели из Hugging Face через proxy. "
                "Обновите окружение: `uv sync`."
            ) from exc
        raise


def _validate_model_dir(model_dir: Path) -> None:
    missing = [
        filename for filename in MODEL_REQUIRED_FILES if not (model_dir / filename).exists()
    ]
    if not any(model_dir.glob("vocabulary.*")):
        missing.append("vocabulary.*")
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Неполная локальная модель в '{model_dir}': отсутствуют {missing_str}")


def _is_missing_socksio_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "socks proxy" in msg and "socksio" in msg
