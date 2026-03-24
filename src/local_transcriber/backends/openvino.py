"""Бэкенд транскрипции на основе OpenVINO GenAI."""

from __future__ import annotations

import threading
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from local_transcriber.types import Segment, TranscribeResult

# (model_alias, compute_type) → HF repo
MODEL_REPOS: dict[tuple[str, str], str] = {
    ("tiny", "int8"): "OpenVINO/whisper-tiny-int8-ov",
    ("base", "fp16"): "OpenVINO/whisper-base-fp16-ov",
    ("small", "int8"): "OpenVINO/whisper-small-int8-ov",
    ("medium", "int8"): "OpenVINO/whisper-medium-int8-ov",
    ("medium", "fp16"): "OpenVINO/whisper-medium-fp16-ov",
    ("large-v3", "int8"): "OpenVINO/whisper-large-v3-int8-ov",
    ("large-v3", "fp16"): "OpenVINO/whisper-large-v3-fp16-ov",
}

# Fallback: если точная пара не найдена, пробуем альтернативный compute_type
_COMPUTE_TYPE_FALLBACKS: dict[str, list[str]] = {
    "float32": ["fp16", "int8"],
    "float16": ["fp16", "int8"],
    "fp16": ["fp16", "int8"],
    "int8": ["int8", "fp16"],
}

# large-v3: при неявном compute_type предпочитаем fp16 (стабильнее по качеству)
_IMPLICIT_COMPUTE_TYPE_OVERRIDES: dict[str, str] = {
    "large-v3": "fp16",
}

MODEL_REQUIRED_FILES = [
    "openvino_encoder_model.xml",
    "openvino_decoder_model.xml",
]


class OpenVINOBackend:
    """Бэкенд транскрипции через openvino-genai WhisperPipeline."""

    def __init__(
        self,
        ov_device: str = "openvino-cpu",
        compute_type_explicit: bool = True,
    ):
        """ov_device: "openvino", "openvino-gpu" или "openvino-cpu".
        compute_type_explicit=False означает, что compute_type пришёл из дефолтов.
        """
        self._ov_device = ov_device
        self._compute_type_explicit = compute_type_explicit
        self.actual_compute_type: str | None = None
        self.actual_ov_device: str | None = None

    def ensure_model_available(
        self,
        model_name: str,
        compute_type: str,
        on_status: Callable[[str], None] | None = None,
    ) -> str:
        """Скачивает/находит OpenVINO модель нужной квантизации."""
        repo_id, resolved_ct = self._resolve_repo(model_name, compute_type)
        self.actual_compute_type = resolved_ct

        try:
            _notify(on_status, f"Проверяю кэш модели {model_name} (OpenVINO)...")
            cached_path = Path(snapshot_download(repo_id, local_files_only=True))
            _validate_model_dir(cached_path)
            return str(cached_path)
        except LocalEntryNotFoundError:
            pass
        except ValueError:
            _notify(on_status, f"Кэш модели {model_name} неполный, докачиваю...")

        _notify(on_status, f"Скачиваю модель {model_name} (OpenVINO) из Hugging Face...")
        downloaded_path = Path(snapshot_download(repo_id, local_files_only=False))
        _validate_model_dir(downloaded_path)
        return str(downloaded_path)

    def _resolve_ov_device(self) -> str:
        """Резолвит device string в OpenVINO device: ``"GPU"`` или ``"CPU"``."""
        if self._ov_device == "openvino-gpu":
            return "GPU"
        if self._ov_device == "openvino-cpu":
            return "CPU"
        # "openvino" → авто-детект
        try:
            from openvino import Core

            if "GPU" in Core().available_devices:
                return "GPU"
        except Exception:
            pass
        return "CPU"

    def create_model(
        self,
        model_path: str,
        device: str,
        compute_type: str,
    ) -> Any:
        """Создаёт WhisperPipeline."""
        import openvino_genai as ov_genai

        ov_dev = self._resolve_ov_device()
        self.actual_ov_device = ov_dev
        return ov_genai.WhisperPipeline(model_path, ov_dev)

    def transcribe(
        self,
        model: Any,
        file_path: Path,
        language: str | None,
        on_segment: Callable[[Segment], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> TranscribeResult:
        """Транскрибирует файл через OpenVINO GenAI."""
        from faster_whisper import decode_audio

        _notify(on_status, "Загружаю аудио...")
        raw_speech = decode_audio(str(file_path), sampling_rate=16000)
        duration = len(raw_speech) / 16000.0

        kwargs: dict[str, Any] = {"return_timestamps": True}
        if language:
            kwargs["language"] = f"<|{language}|>"

        dur_min = int(duration // 60)
        duration_str = f"{dur_min} мин" if dur_min > 0 else f"{int(duration)} сек"
        pcm_list = raw_speech.tolist()
        result = _generate_with_progress(model, pcm_list, kwargs, duration_str, on_status)

        segments: list[Segment] = []
        if hasattr(result, "chunks") and result.chunks:
            for chunk in result.chunks:
                start = max(0.0, chunk.start_ts)
                end = max(start, chunk.end_ts)
                seg = Segment(
                    start=start,
                    end=end,
                    text=chunk.text,
                )
                if on_segment is not None:
                    on_segment(seg)
                segments.append(seg)
                _notify(
                    on_status,
                    f"Транскрибирую (OpenVINO)... [{len(segments)} сегм.]",
                )

        detected_language = language or "auto"
        language_probability = 1.0 if language else 0.0

        return TranscribeResult(
            segments=segments,
            language=detected_language,
            language_probability=language_probability,
            duration=duration,
            device_used="",  # оркестратор проставит
        )

    def _resolve_repo(self, model_name: str, compute_type: str) -> tuple[str, str]:
        """Находит HF repo для пары (model, compute_type) с fallback.

        Возвращает (repo_id, actual_compute_type).
        """
        # Для неявного compute_type: override для конкретных моделей
        if not self._compute_type_explicit and model_name in _IMPLICIT_COMPUTE_TYPE_OVERRIDES:
            compute_type = _IMPLICIT_COMPUTE_TYPE_OVERRIDES[model_name]

        # Точное совпадение
        repo = MODEL_REPOS.get((model_name, compute_type))
        if repo:
            return repo, compute_type

        # Fallback только для неявного compute_type
        if not self._compute_type_explicit:
            fallbacks = _COMPUTE_TYPE_FALLBACKS.get(compute_type, [])
            for fallback_ct in fallbacks:
                repo = MODEL_REPOS.get((model_name, fallback_ct))
                if repo:
                    return repo, fallback_ct

        # Явный --compute-type с несуществующей парой → ошибка
        available = [ct for (m, ct) in MODEL_REPOS if m == model_name]
        if available:
            raise ValueError(
                f"Модель '{model_name}' недоступна с compute_type='{compute_type}' для OpenVINO. "
                f"Доступные варианты: {', '.join(sorted(set(available)))}"
            )

        all_models = sorted({m for m, _ in MODEL_REPOS})
        raise ValueError(
            f"Модель '{model_name}' не найдена для OpenVINO. "
            f"Доступные модели: {', '.join(all_models)}"
        )


def _generate_with_progress(
    model: Any,
    pcm_list: list[float],
    kwargs: dict[str, Any],
    duration_str: str,
    on_status: Callable[[str], None] | None,
) -> Any:
    """Запускает model.generate() в потоке, обновляя статус с elapsed time."""
    result_box: list[Any] = [None]
    error_box: list[BaseException | None] = [None]

    def run() -> None:
        try:
            result_box[0] = model.generate(pcm_list, **kwargs)
        except BaseException as exc:
            error_box[0] = exc

    thread = threading.Thread(target=run)
    start = time.monotonic()
    thread.start()

    while thread.is_alive():
        elapsed = int(time.monotonic() - start)
        elapsed_str = f"{elapsed // 60:02d}:{elapsed % 60:02d}"
        _notify(on_status, f"Транскрибирую {duration_str} аудио (OpenVINO)... прошло {elapsed_str}")
        thread.join(timeout=1.0)

    if error_box[0] is not None:
        raise error_box[0]

    return result_box[0]


def _notify(on_status: Callable[[str], None] | None, message: str) -> None:
    if on_status is not None:
        on_status(message)


def _validate_model_dir(model_dir: Path) -> None:
    missing = [f for f in MODEL_REQUIRED_FILES if not (model_dir / f).exists()]
    if missing:
        raise ValueError(
            f"Неполная OpenVINO модель в '{model_dir}': отсутствуют {', '.join(missing)}"
        )
