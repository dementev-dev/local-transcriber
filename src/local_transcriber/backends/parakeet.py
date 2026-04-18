"""Бэкенд транскрипции на основе NVIDIA Parakeet TDT (onnx-asr + Silero VAD)."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
import onnx_asr

from local_transcriber.types import Segment, TranscribeResult

# HF-репозиторий с уже сконвертированными ONNX-файлами (int8 + fp32).
# Официальный `nvidia/parakeet-tdt-0.6b-v3` содержит только .nemo/.safetensors и
# onnx-asr loader его не поддерживает.
MODEL_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"
ONNX_ASR_MODEL_ALIAS = "nemo-parakeet-tdt-0.6b-v3"

SUPPORTED_MODEL_NAMES: set[str] = {"parakeet-tdt-0.6b-v3", "parakeet"}
SUPPORTED_COMPUTE_TYPES: set[str] = {"int8", "float32"}

MODEL_REQUIRED_FILES: list[str] = [
    "config.json",
]

# Паттерны для snapshot_download: качаем только то, что нужно под выбранный
# compute_type — иначе HF тянет и int8, и fp32 (~5 GB вместо ~670 MB / ~2 GB).
_INT8_PATTERNS: list[str] = [
    "config.json",
    "vocab.txt",
    "nemo128.onnx",
    "encoder-model.int8.onnx",
    "decoder_joint-model.int8.onnx",
]
_FLOAT32_PATTERNS: list[str] = [
    "config.json",
    "vocab.txt",
    "nemo128.onnx",
    "encoder-model.onnx",
    "encoder-model.onnx.data",
    "decoder_joint-model.onnx",
]


class ParakeetBackend:
    """Бэкенд Parakeet TDT v3 через onnx-asr (CPU execution provider)."""

    def __init__(self, compute_type_explicit: bool = True):
        self._compute_type_explicit = compute_type_explicit
        self.actual_compute_type: str | None = None
        self.effective_model_name: str = "parakeet-tdt-0.6b-v3"

    def ensure_model_available(
        self,
        model_name: str,
        compute_type: str,
        on_status: Callable[[str], None] | None = None,
    ) -> str:
        if model_name not in SUPPORTED_MODEL_NAMES:
            allowed = ", ".join(sorted(SUPPORTED_MODEL_NAMES))
            raise ValueError(
                f"Parakeet поддерживает только модели: {allowed}. "
                f"Получено: '{model_name}'. Передайте --model parakeet или уберите "
                f"'model' из .transcriber.toml."
            )
        if compute_type not in SUPPORTED_COMPUTE_TYPES:
            raise ValueError(
                f"Parakeet поддерживает только compute_type: int8 или float32. "
                f"Получено: '{compute_type}'."
            )

        self.actual_compute_type = compute_type
        patterns = _INT8_PATTERNS if compute_type == "int8" else _FLOAT32_PATTERNS

        # Cache-first: сначала проверяем локальный кэш (как faster_whisper / openvino бэкенды)
        try:
            _notify(on_status, f"Проверяю кэш модели Parakeet ({MODEL_REPO})...")
            cached_path = Path(
                snapshot_download(MODEL_REPO, local_files_only=True, allow_patterns=patterns)
            )
            _validate_model_dir(cached_path)
            return str(cached_path)
        except LocalEntryNotFoundError:
            pass
        except ValueError:
            _notify(on_status, "Кэш Parakeet неполный, докачиваю...")

        _notify(on_status, f"Скачиваю модель Parakeet ({MODEL_REPO}) из Hugging Face...")
        model_dir = Path(
            snapshot_download(MODEL_REPO, local_files_only=False, allow_patterns=patterns)
        )
        _validate_model_dir(model_dir)
        return str(model_dir)

    def create_model(
        self,
        model_path: str,
        device: str,
        compute_type: str,
        cpu_threads: int = 0,
    ) -> Any:
        """Создаёт asr-pipeline с Silero VAD.

        cpu_threads: если > 0, передаётся через onnxruntime.SessionOptions.intra_op_num_threads.
        VAD загружается здесь (runtime-зависимость; первый запуск требует интернет).
        """
        quantization = "int8" if compute_type == "int8" else None

        load_kwargs: dict[str, Any] = {"path": model_path, "quantization": quantization}
        if cpu_threads and cpu_threads > 0:
            # intra_op_num_threads — атрибут SessionOptions, а НЕ provider_options.
            # provider_options ожидает EP-специфичные ключи (device_id и т.п.).
            import onnxruntime as ort
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = cpu_threads
            load_kwargs["sess_options"] = sess_options

        asr = onnx_asr.load_model(ONNX_ASR_MODEL_ALIAS, **load_kwargs)
        vad = onnx_asr.load_vad(model="silero")
        return asr.with_vad(vad=vad)

    def transcribe(
        self,
        model: Any,
        file_path: Path,
        language: str | None,
        on_segment: Callable[[Segment], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> TranscribeResult:
        """Транскрибирует файл через onnx-asr + Silero VAD.

        language=None/"auto" — ок; любое другое значение → warning (Parakeet игнорирует язык).
        Возвращает TranscribeResult с language="multi" (независимо от пользовательского hint).
        """
        if language is not None and language != "auto":
            warnings.warn(
                "Parakeet игнорирует --language; язык определяется автоматически",
                UserWarning,
                stacklevel=2,
            )

        _notify(on_status, "Транскрибирую (Parakeet + Silero VAD)...")

        raw_segments = model.recognize(str(file_path))

        segments: list[Segment] = []
        max_end = 0.0
        for raw in raw_segments:
            start = float(getattr(raw, "start", 0.0) or 0.0)
            end = float(getattr(raw, "end", start) or start)
            text = getattr(raw, "text", "") or ""
            seg = Segment(start=start, end=end, text=text)
            if on_segment is not None:
                on_segment(seg)
            segments.append(seg)
            if end > max_end:
                max_end = end
            _notify(on_status, f"Parakeet: {len(segments)} сегм. ({end:.1f}с)")

        # duration = конец последнего речевого сегмента после VAD, НЕ точная длина аудио-файла.
        # Для заголовка транскрипта и форматирования таймкодов этого достаточно: расхождение —
        # только на длину трейлинг-тишины. Декодировать аудио отдельно ради точного значения
        # (как это делает OpenVINO-бэкенд через decode_audio) не стоит: ONNX-порт сам грузит
        # файл внутри recognize(), повторное декодирование на 3-часовых файлах — заметная цена.
        return TranscribeResult(
            segments=segments,
            language="multi",
            language_probability=0.0,
            duration=max_end,
            device_used="",  # оркестратор проставит
        )


def _notify(on_status: Callable[[str], None] | None, message: str) -> None:
    if on_status is not None:
        on_status(message)


def _validate_model_dir(model_dir: Path) -> None:
    missing = [f for f in MODEL_REQUIRED_FILES if not (model_dir / f).exists()]
    if missing:
        raise ValueError(
            f"Неполная Parakeet-модель в '{model_dir}': отсутствуют {', '.join(missing)}"
        )
