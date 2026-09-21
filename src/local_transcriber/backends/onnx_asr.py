"""Бэкенд транскрипции на основе onnx-asr (GigaAM, Parakeet, FastConformer)."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_transcriber.types import (
    UNKNOWN_LANGUAGE,
    Segment,
    TranscribeResult,
    Word,
    WordTimestampsUnavailableError,
)
from local_transcriber.utils import package_version


@dataclass(frozen=True)
class OnnxModelSpec:
    """Имя onnx-asr, варианты квантизации и поддерживаемые языки."""

    model_id: str
    quantizations: frozenset[str | None]
    supported_languages: frozenset[str]


_INT8_AND_FLOAT32 = frozenset({"int8", None})
_RUSSIAN_ONLY = frozenset({"ru"})
_GIGAAM_MULTILINGUAL_LANGUAGES = frozenset({"ru", "en", "kk", "ky", "uz"})
_PARAKEET_V3_LANGUAGES = frozenset(
    {
        "bg",
        "hr",
        "cs",
        "da",
        "nl",
        "en",
        "et",
        "fi",
        "fr",
        "de",
        "el",
        "hu",
        "it",
        "lv",
        "lt",
        "mt",
        "pl",
        "pt",
        "ro",
        "sk",
        "sl",
        "es",
        "sv",
        "ru",
        "uk",
    }
)
_WHISPER_MODEL_NAMES = frozenset(
    {"tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"}
)
# faster-whisper не знает turbo, поэтому для cpu и cuda подсказываем medium.
# Источник правды — MODEL_REPOS в backends/faster_whisper.py и backends/openvino.py
_OPENVINO_ONLY_WHISPER_MODELS = frozenset({"large-v3-turbo"})

MODEL_CATALOG: dict[str, OnnxModelSpec] = {
    "gigaam-v3": OnnxModelSpec("gigaam-v3-ctc", _INT8_AND_FLOAT32, _RUSSIAN_ONLY),
    "parakeet-v3": OnnxModelSpec(
        "nemo-parakeet-tdt-0.6b-v3",
        _INT8_AND_FLOAT32,
        _PARAKEET_V3_LANGUAGES,
    ),
    "gigaam-multilingual-ctc": OnnxModelSpec(
        "gigaam-multilingual-ctc",
        _INT8_AND_FLOAT32,
        _GIGAAM_MULTILINGUAL_LANGUAGES,
    ),
    "gigaam-multilingual-large-ctc": OnnxModelSpec(
        "gigaam-multilingual-large-ctc",
        _INT8_AND_FLOAT32,
        _GIGAAM_MULTILINGUAL_LANGUAGES,
    ),
    "gigaam-v3-e2e-ctc": OnnxModelSpec(
        "gigaam-v3-e2e-ctc",
        _INT8_AND_FLOAT32,
        _RUSSIAN_ONLY,
    ),
    "gigaam-v3-e2e-rnnt": OnnxModelSpec(
        "gigaam-v3-e2e-rnnt",
        _INT8_AND_FLOAT32,
        _RUSSIAN_ONLY,
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
        self._model_name: str | None = None
        self._model_spec: OnnxModelSpec | None = None
        self._vad: Any = None
        self._providers: list[str] = []
        self._quantization: str | None = None
        self._cpu_threads = 0

    @property
    def word_timestamps_available(self) -> bool:
        """Каталожные модели проверены; произвольный raw id отклоняется."""
        return self._model_spec is not None

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
        self._model_name = model_name
        self._model_spec = spec
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
        cpu_threads > 0 задаёт intra_op_num_threads сессиям ASR и VAD;
        0 оставляет настройки потоков onnxruntime.
        """
        import onnx_asr

        actual_compute_type = self.actual_compute_type or compute_type
        quantization = _normalize_quantization(actual_compute_type)

        providers = ["CPUExecutionProvider"]
        session_kwargs: dict[str, Any] = {"providers": providers}
        if cpu_threads > 0:
            session_kwargs["sess_options"] = _session_options(cpu_threads)
        model = onnx_asr.load_model(
            model=model_path,
            quantization=quantization,
            **session_kwargs,
        )
        vad = onnx_asr.load_vad("silero", **session_kwargs)
        self._vad = vad
        self._providers = providers
        self._quantization = quantization
        self._cpu_threads = cpu_threads
        return model.with_vad(vad).with_timestamps()

    def runtime_info(self) -> dict[str, str]:
        """Версии ONNX Runtime/onnx-asr и providers, заданные сессиям ASR и VAD.

        Доступность provider в wheel не означает, что граф выполняется на нём:
        сессии получают только явно заданный список.
        """
        import onnxruntime

        configured = ", ".join(self._providers)
        return {
            "onnxruntime": package_version("onnxruntime"),
            "onnx_asr": package_version("onnx-asr"),
            "available_providers": ", ".join(onnxruntime.get_available_providers()),
            "asr_providers": configured,
            "vad_providers": configured,
            "quantization": self._quantization or "float32",
            "intra_op_threads": str(self._cpu_threads) if self._cpu_threads else "по умолчанию",
        }

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

        self._warn_if_language_unsupported(language)
        _notify(on_status, "Загружаю аудио...")
        audio_array = decode_audio(str(file_path), sampling_rate=16000)
        if isinstance(audio_array, tuple):
            raise TypeError("Декодер неожиданно вернул раздельные стереоканалы")
        duration = len(audio_array) / 16000.0

        _notify(on_status, "Транскрибирую (onnx-asr)...")
        segments: list[Segment] = []
        words: list[Word] = []
        result_language = (
            language or _model_language(self._model_spec) or UNKNOWN_LANGUAGE
        )

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
            segment_words = _timestamped_segment_words(vad_seg, start, end)
            if vad_seg.text.strip() and not segment_words:
                raise WordTimestampsUnavailableError(
                    "ONNX-ASR не вернул пословные таймкоды для распознанного текста"
                )
            words.extend(segment_words)
            if on_segment is not None:
                on_segment(seg)
            segments.append(seg)
            _notify(
                on_status,
                f"Транскрибирую (onnx-asr)... [{len(segments)} сегм.]",
            )

        return TranscribeResult(
            segments=segments,
            language=result_language,
            language_probability=1.0 if language else 0.0,
            duration=duration,
            device_used="",  # оркестратор проставит
            words=words,
        )

    def _resolve_model(self, model_name: str) -> str:
        """Resolve alias to onnx-asr model name. Raw names pass through."""
        if model_name in MODEL_ALIASES:
            return MODEL_ALIASES[model_name]
        if model_name in _WHISPER_MODEL_NAMES:
            fallback = _whisper_fallback_model(model_name)
            raise ValueError(
                f"Модель '{model_name}' относится к Whisper и не поддерживается "
                "ONNX-бэкендом. --device auto всегда выбирает ONNX CPU; "
                f"укажите --device openvino-cpu --model {model_name} на x86, "
                f"--device cpu --model {fallback} на любой платформе "
                f"или --device cuda --model {fallback} при NVIDIA GPU."
            )
        if "/" in model_name or model_name.count("-") >= 2:
            # Looks like a raw onnx-asr name — allow passthrough
            return model_name
        raise ValueError(
            f"Неподдерживаемая модель '{model_name}'. "
            f"Доступные алиасы: {SUPPORTED_ALIASES}. "
            f"Либо укажите полное имя модели onnx-asr."
        )

    def _warn_if_language_unsupported(self, language: str | None) -> None:
        if (
            language is None
            or self._model_spec is None
            or language in self._model_spec.supported_languages
        ):
            return

        supported = ", ".join(sorted(self._model_spec.supported_languages))
        warnings.warn(
            f"Язык '{language}' не поддерживается моделью '{self._model_name}' "
            f"(поддерживаются: {supported}). Результат может быть некорректным. "
            "Для других языков возьмите Whisper: "
            "--device openvino-cpu --model medium на x86, "
            "--device cpu --model medium на любой платформе "
            "или --device cuda --model medium при NVIDIA GPU.",
            UserWarning,
            stacklevel=2,
        )


def _session_options(cpu_threads: int) -> Any:
    """SessionOptions с бюджетом потоков; общий объект для ASR и VAD."""
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = cpu_threads
    return options


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


def _whisper_fallback_model(model_name: str) -> str:
    """Модель для подсказки про faster-whisper: turbo там недоступен."""
    if model_name in _OPENVINO_ONLY_WHISPER_MODELS:
        return "medium"
    return model_name


def _model_language(spec: OnnxModelSpec | None) -> str | None:
    if spec is not None and len(spec.supported_languages) == 1:
        return next(iter(spec.supported_languages))
    return None


def _notify(on_status: Callable[[str], None] | None, message: str) -> None:
    if on_status is not None:
        on_status(message)


def _timestamped_segment_words(
    vad_segment: Any,
    segment_start: float,
    segment_end: float,
) -> list[Word]:
    tokens = getattr(vad_segment, "tokens", None)
    timestamps = getattr(vad_segment, "timestamps", None)
    if not tokens or not timestamps or len(tokens) != len(timestamps):
        return []

    grouped: list[tuple[float, str]] = []
    current_start = float(timestamps[0])
    current_tokens: list[str] = []
    for token, timestamp in zip(tokens, timestamps, strict=True):
        if token[:1].isspace() and current_tokens:
            grouped.append((current_start, "".join(current_tokens)))
            current_start = float(timestamp)
            current_tokens = []
        current_tokens.append(token)
    grouped.append((current_start, "".join(current_tokens)))

    words: list[Word] = []
    for index, (relative_start, text) in enumerate(grouped):
        start = min(
            segment_end,
            max(segment_start, segment_start + relative_start),
        )
        next_start = next(
            (
                candidate_start
                for candidate_start, _ in grouped[index + 1 :]
                if candidate_start > relative_start
            ),
            None,
        )
        end = max(
            start,
            min(segment_end, segment_start + next_start)
            if next_start is not None
            else segment_end,
        )
        words.append(Word(start=start, end=end, text=text))
    return words
