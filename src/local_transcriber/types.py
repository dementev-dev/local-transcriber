"""Общие типы данных для всех бэкендов транскрипции."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


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
    device_used: str  # "cpu" / "cuda" / "openvino"


@dataclass
class TranscribeFileResult:
    result: TranscribeResult
    model: Any  # backend-specific model handle
    actual_device: str
    backend: Any = None  # backend instance (для переиспользования в батче)
    model_path: str = ""  # путь к модели (меняется при cross-backend fallback)


StatusCallback = Callable[[str], None] | None
SegmentCallback = Callable[[Segment], None] | None
