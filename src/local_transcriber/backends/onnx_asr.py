"""Бэкенд транскрипции на основе onnx-asr (GigaAM, Parakeet, FastConformer)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_transcriber.types import Segment, TranscribeResult


@dataclass(frozen=True)
class OnnxModelSpec:
    """Имя onnx-asr и опубликованные варианты квантизации модели."""

    model_id: str
    quantizations: frozenset[str | None]


_INT8_AND_FLOAT32 = frozenset({"int8", None})

MODEL_CATALOG: dict[str, OnnxModelSpec] = {
    "gigaam-v3": OnnxModelSpec("gigaam-v3-ctc", _INT8_AND_FLOAT32),
    "parakeet-v3": OnnxModelSpec(
        "nemo-parakeet-tdt-0.6b-v3",
        _INT8_AND_FLOAT32,
    ),
    "gigaam-multilingual-ctc": OnnxModelSpec(
        "gigaam-multilingual-ctc",
        _INT8_AND_FLOAT32,
    ),
    "gigaam-multilingual-large-ctc": OnnxModelSpec(
        "gigaam-multilingual-large-ctc",
        _INT8_AND_FLOAT32,
    ),
    "gigaam-v3-e2e-ctc": OnnxModelSpec(
        "gigaam-v3-e2e-ctc",
        _INT8_AND_FLOAT32,
    ),
    "gigaam-v3-e2e-rnnt": OnnxModelSpec(
        "gigaam-v3-e2e-rnnt",
        _INT8_AND_FLOAT32,
    ),
}

# Оставлено как совместимое представление публичного каталога алиасов.
MODEL_ALIASES: dict[str, str] = {
    alias: spec.model_id for alias, spec in MODEL_CATALOG.items()
}

SUPPORTED_ALIASES = ", ".join(MODEL_ALIASES)

# compute_type проекта → onnx-asr quantization (file suffix; None = unquantized).
_QUANTIZATION_MAP: dict[str, str | None] = {
    "int8": "int8",
    "fp16": "fp16",
    "float16": "fp16",
    "float32": None,
    "fp32": None,
}


def _normalize_quantization(compute_type: str) -> str | None:
    """Маппит compute_type проекта в значение onnx-asr ``quantization``.

    onnx-asr использует ``quantization`` как суффикс имени файла модели:
    ``int8``/``fp16`` подгружают квантизованные веса, ``None`` — unquantized
    (float32). Передача ``"float32"`` строкой пытается найти несуществующий
    файл с суффиксом ``_float32`` и приводит к ошибке загрузки.
    """
    if compute_type not in _QUANTIZATION_MAP:
        supported = ", ".join(sorted(_QUANTIZATION_MAP))
        raise ValueError(
            f"Неподдерживаемый compute_type '{compute_type}' для onnx-asr. "
            f"Допустимо: {supported}."
        )
    return _QUANTIZATION_MAP[compute_type]


class OnnxAsrBackend:
    """Бэкенд транскрипции через onnx-asr (ONNX Runtime)."""

    def __init__(self, compute_type_explicit: bool = True):
        self._compute_type_explicit = compute_type_explicit
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
        spec = MODEL_CATALOG.get(model_name)
        quantization = _normalize_quantization(compute_type)
        if spec is not None and quantization not in spec.quantizations:
            if self._compute_type_explicit:
                available = _format_compute_types(spec.quantizations)
                raise ValueError(
                    f"Модель '{model_name}' недоступна с compute_type='{compute_type}' "
                    f"для onnx-asr. Доступные варианты: {available}"
                )

            resolved_compute_type = _preferred_compute_type(spec.quantizations)
            _notify(
                on_status,
                f"Модель {model_name} недоступна с compute_type={compute_type}; "
                f"использую {resolved_compute_type}.",
            )
        else:
            resolved_compute_type = _compute_type_for_quantization(quantization)

        self.actual_compute_type = resolved_compute_type
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

        compute_type маппится в onnx-asr ``quantization`` — это суффикс файла
        модели; для unquantized (float32/fp32) нужно None, не строку.
        """
        import onnx_asr

        actual_compute_type = self.actual_compute_type or compute_type
        quantization = _normalize_quantization(actual_compute_type)

        model = onnx_asr.load_model(
            model=model_path,
            quantization=quantization,
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

        for vad_seg in model.recognize(
            audio_array, sample_rate=16000, language=language
        ):
            start = max(0.0, vad_seg.start)
            end = max(0.0, vad_seg.end)
            if end <= start:
                continue
            seg = Segment(
                start=start,
                end=end,
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


def _format_compute_types(quantizations: frozenset[str | None]) -> str:
    values = [_compute_type_for_quantization(value) for value in quantizations]
    return ", ".join(sorted(values))


def _compute_type_for_quantization(quantization: str | None) -> str:
    return "float32" if quantization is None else quantization


def _preferred_compute_type(quantizations: frozenset[str | None]) -> str:
    for quantization in ("int8", None, "fp16"):
        if quantization in quantizations:
            return _compute_type_for_quantization(quantization)
    raise ValueError("Для ONNX-модели не указаны доступные квантизации")


def _notify(on_status: Callable[[str], None] | None, message: str) -> None:
    if on_status is not None:
        on_status(message)
