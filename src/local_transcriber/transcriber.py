import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel


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
) -> TranscribeResult:
    actual_device = device
    lang_arg = language if language and language != "auto" else None

    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    except (RuntimeError, ValueError) as exc:
        if device != "cpu" and _is_cuda_error(exc):
            warnings.warn(
                f"Не удалось загрузить модель на {device}: {exc}. "
                "Переключение на CPU.",
                stacklevel=2,
            )
            actual_device = "cpu"
            model = WhisperModel(model_name, device="cpu", compute_type=compute_type)
        else:
            raise

    try:
        segments, info = _run_transcription(model, file_path, lang_arg, on_segment)
    except (RuntimeError, ValueError) as exc:
        if actual_device != "cpu" and _is_cuda_error(exc):
            warnings.warn(
                f"CUDA ошибка при транскрипции: {exc}. "
                "Переключение на CPU и повтор.",
                stacklevel=2,
            )
            actual_device = "cpu"
            model = WhisperModel(model_name, device="cpu", compute_type=compute_type)
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


def _is_cuda_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "cuda" in msg or "out of memory" in msg
