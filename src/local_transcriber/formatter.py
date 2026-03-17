from datetime import datetime
from pathlib import Path

from .transcriber import TranscribeResult


def format_timestamp(seconds: float, use_hours: bool = False) -> str:
    raise NotImplementedError


def format_transcript(
    result: TranscribeResult,
    source_filename: str,
    model_name: str,
    device_info: str,
    language_mode: str,  # "detected" | "forced"
    transcription_date: datetime | None = None,  # None -> datetime.now()
) -> str:
    raise NotImplementedError


def write_transcript(content: str, output_path: Path) -> None:
    raise NotImplementedError
