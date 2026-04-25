"""Бэкенд транскрипции на основе onnx-asr (GigaAM, Parakeet, FastConformer)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from local_transcriber.types import Segment, TranscribeResult

MODEL_ALIASES: dict[str, str] = {
    "gigaam-v3": "gigaam-v3-ctc",
    "parakeet-v3": "nemo-parakeet-tdt-0.6b-v3",
}

SUPPORTED_ALIASES = ", ".join(MODEL_ALIASES)


class OnnxAsrBackend:
    """Бэкенд транскрипции через onnx-asr (ONNX Runtime)."""

    def __init__(self):
        self.actual_compute_type: str | None = None
        self._resolved_model_id: str | None = None
        self._vad: Any = None

    def ensure_model_available(
        self,
        model_name: str,
        compute_type: str,
        on_status: Callable[[str], None] | None = None,
    ) -> str:
        """Resolves model alias and returns the onnx-asr model identifier.

        onnx-asr downloads models automatically via load_model(),
        so this just validates the alias and returns the identifier string.
        """
        self.actual_compute_type = compute_type
        self._resolved_model_id = self._resolve_model(model_name)
        return self._resolved_model_id

    def create_model(
        self,
        model_path: str,
        device: str,
        compute_type: str,
        cpu_threads: int = 0,
    ) -> Any:
        """Creates onnx-asr model with VAD.

        model_path: onnx-asr model identifier (e.g. "gigaam-v3-ctc").
        compute_type: "int8", "fp16", or "float32" — passed as quantization.
        cpu_threads: not used by onnx-asr (onnxruntime manages threads internally).
        """
        import onnx_asr

        ct = compute_type if compute_type in ("int8", "fp16", "float32") else "int8"

        model = onnx_asr.load_model(
            model=model_path,
            quantization=ct,
        )
        vad = onnx_asr.load_vad("silero")
        self._vad = vad
        return model.with_vad(vad)

    def transcribe(
        self,
        model: Any,
        file_path: Path,
        language: str | None,
        on_segment: Callable[[Segment], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> TranscribeResult:
        """Transcribes audio file using onnx-asr model with VAD.

        model: result of create_model() — a SegmentResultsAsrAdapter.
        file_path: path to audio/video file (any format supported by faster-whisper decode).
        language: language code (e.g. "ru", "en") — only meaningful for multilingual models.
        """
        from faster_whisper import decode_audio

        _notify(on_status, "Загружаю аудио...")
        audio_array = decode_audio(str(file_path), sampling_rate=16000)
        duration = len(audio_array) / 16000.0

        _notify(on_status, "Транскрибирую (onnx-asr)...")
        segments: list[Segment] = []
        detected_language = language or "unknown"

        for vad_seg in model.recognize(audio_array, sample_rate=16000, language=language):
            seg = Segment(
                start=max(0.0, vad_seg.start),
                end=max(0.0, vad_seg.end),
                text=vad_seg.text,
            )
            if on_segment is not None:
                on_segment(seg)
            segments.append(seg)
            _notify(
                on_status,
                f"Транскрибирую (onnx-asr)... [{len(segments)} сегм.]",
            )

        return TranscribeResult(
            segments=segments,
            language=detected_language,
            language_probability=1.0 if language else 0.0,
            duration=duration,
            device_used="",  # оркестратор проставит
        )

    def _resolve_model(self, model_name: str) -> str:
        """Resolve alias to onnx-asr model name. Raw names pass through."""
        if model_name in MODEL_ALIASES:
            return MODEL_ALIASES[model_name]
        if "/" in model_name or model_name.count("-") >= 2:
            # Looks like a raw onnx-asr name — allow passthrough
            return model_name
        raise ValueError(
            f"Неподдерживаемая модель '{model_name}'. "
            f"Доступные алиасы: {SUPPORTED_ALIASES}. "
            f"Либо укажите полное имя модели onnx-asr."
        )


def _notify(on_status: Callable[[str], None] | None, message: str) -> None:
    if on_status is not None:
        on_status(message)
