from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


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
    raise NotImplementedError
