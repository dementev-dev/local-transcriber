import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Должен быть ДО импорта faster_whisper / ctranslate2
from local_transcriber._cuda_bootstrap import ensure_cublas_loadable

ensure_cublas_loadable()

from faster_whisper import WhisperModel  # noqa: E402
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

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
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
]


@dataclass
class Segment:
    start: float  # seconds
    end: float  # seconds
    text: str


@dataclass
class TranscribeResult:
    segments: list[Segment]
    language: str
    language_probability: float
    duration: float  # seconds
    device_used: str  # "cpu" / "cuda"


def transcribe(
    file_path: Path,
    model_name: str = "large-v3",
    device: str = "auto",
    compute_type: str = "int8",
    language: str | None = None,
    on_segment: Callable[[Segment], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    strict_device: bool = False,
) -> TranscribeResult:
    actual_device = device
    lang_arg = language if language and language != "auto" else None

    try:
        _notify_status(on_status, f"Загружаю модель на {device}...")
        model = _create_model(model_name, device, compute_type)
    except (RuntimeError, ValueError) as exc:
        if device != "cpu" and _is_cuda_error(exc):
            if strict_device:
                raise
            warnings.warn(
                f"Не удалось загрузить модель на {device}: {exc}. "
                "Переключение на CPU.",
                stacklevel=2,
            )
            actual_device = "cpu"
            _notify_status(on_status, "Загружаю модель на cpu...")
            model = _create_model(model_name, "cpu", compute_type)
        else:
            raise

    try:
        _notify_status(on_status, "Транскрибирую...")
        segments, info = _run_transcription(model, file_path, lang_arg, on_segment)
    except (RuntimeError, ValueError) as exc:
        if actual_device != "cpu" and _is_cuda_error(exc):
            if strict_device:
                raise
            warnings.warn(
                f"CUDA ошибка при транскрипции: {exc}. "
                "Переключение на CPU и повтор.",
                stacklevel=2,
            )
            actual_device = "cpu"
            _notify_status(on_status, "Загружаю модель на cpu...")
            model = _create_model(model_name, "cpu", compute_type)
            _notify_status(on_status, "Транскрибирую...")
            segments, info = _run_transcription(model, file_path, lang_arg, on_segment)
        else:
            raise

    return TranscribeResult(
        segments=segments,
        language=info.language,
        language_probability=info.language_probability,
        duration=info.duration,
        device_used=actual_device,
    )


def ensure_model_available(
    model_name: str,
    on_status: Callable[[str], None] | None = None,
) -> str:
    local_path = Path(model_name).expanduser()
    if local_path.is_dir():
        _validate_model_dir(local_path)
        return str(local_path)

    repo_id = _resolve_model_repo(model_name)

    try:
        _notify_status(on_status, f"Проверяю кэш модели {model_name}...")
        cached_path = Path(_snapshot_download(repo_id, local_files_only=True))
        _validate_model_dir(cached_path)
        return str(cached_path)
    except LocalEntryNotFoundError:
        pass
    except ValueError:
        _notify_status(on_status, f"Кэш модели {model_name} неполный, докачиваю...")

    _notify_status(on_status, f"Скачиваю модель {model_name} из Hugging Face...")
    downloaded_path = Path(_snapshot_download(repo_id, local_files_only=False))
    _validate_model_dir(downloaded_path)
    return str(downloaded_path)


def _run_transcription(model, file_path, lang_arg, on_segment):
    """Run model.transcribe and iterate segments. Returns (segments, info)."""
    segment_generator, info = model.transcribe(str(file_path), language=lang_arg)
    segments: list[Segment] = []
    for raw_seg in segment_generator:
        seg = Segment(start=raw_seg.start, end=raw_seg.end, text=raw_seg.text)
        if on_segment is not None:
            on_segment(seg)
        segments.append(seg)
    return segments, info


def _create_model(model_name: str, device: str, compute_type: str):
    try:
        return WhisperModel(model_name, device=device, compute_type=compute_type)
    except ImportError as exc:
        if _is_missing_socksio_error(exc):
            raise RuntimeError(
                "Обнаружен SOCKS proxy, но не установлена зависимость `socksio`, "
                "нужная для загрузки модели из Hugging Face через proxy. "
                "Обновите окружение: `uv sync`."
            ) from exc
        raise


def _is_cuda_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "cuda" in msg or "out of memory" in msg


def _is_missing_socksio_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "socks proxy" in msg and "socksio" in msg


def _notify_status(on_status: Callable[[str], None] | None, message: str) -> None:
    if on_status is not None:
        on_status(message)


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
