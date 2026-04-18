"""Реестр бэкендов транскрипции и выбор бэкенда по устройству."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Backend


def get_backend(device: str, *, compute_type_explicit: bool = True) -> Backend:
    """Возвращает экземпляр бэкенда для указанного устройства.

    Импорты ленивые — бэкенд загружается только при запросе.
    compute_type_explicit: False если compute_type пришёл из дефолтов (влияет на fallback).
    """
    if device in ("openvino", "openvino-gpu", "openvino-cpu"):
        try:
            from .openvino import OpenVINOBackend
        except ImportError:
            raise ValueError(
                "OpenVINO бэкенд недоступен. Установите: pip install openvino-genai"
            ) from None
        return OpenVINOBackend(
            ov_device=device, compute_type_explicit=compute_type_explicit
        )

    if device in ("parakeet", "parakeet-cpu"):
        try:
            from .parakeet import ParakeetBackend
        except ImportError:
            raise ValueError(
                "Parakeet бэкенд недоступен. Установите: pip install onnx-asr[cpu,hub]"
            ) from None
        return ParakeetBackend(compute_type_explicit=compute_type_explicit)

    # cuda, cpu и всё остальное → faster-whisper
    from .faster_whisper import FasterWhisperBackend

    return FasterWhisperBackend()
