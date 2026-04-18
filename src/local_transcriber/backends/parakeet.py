"""Бэкенд транскрипции на основе NVIDIA Parakeet TDT (onnx-asr + Silero VAD)."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from local_transcriber.types import Segment, TranscribeResult

MODEL_REPO = "nvidia/parakeet-tdt-0.6b-v3"
ONNX_ASR_MODEL_ALIAS = "nemo-parakeet-tdt-0.6b-v3"

SUPPORTED_MODEL_NAMES: set[str] = {"parakeet-tdt-0.6b-v3", "parakeet"}
SUPPORTED_COMPUTE_TYPES: set[str] = {"int8", "float32"}

MODEL_REQUIRED_FILES: list[str] = [
    "config.json",
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
        raise NotImplementedError

    def create_model(
        self,
        model_path: str,
        device: str,
        compute_type: str,
        cpu_threads: int = 0,
    ) -> Any:
        raise NotImplementedError

    def transcribe(
        self,
        model: Any,
        file_path: Path,
        language: str | None,
        on_segment: Callable[[Segment], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> TranscribeResult:
        raise NotImplementedError


def _notify(on_status: Callable[[str], None] | None, message: str) -> None:
    if on_status is not None:
        on_status(message)
