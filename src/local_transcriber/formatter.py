from datetime import datetime
from pathlib import Path

from .transcriber import TranscribeResult


def format_timestamp(seconds: float, use_hours: bool = False) -> str:
    total_seconds = int(seconds)
    centiseconds = int(round((seconds - total_seconds) * 100))

    if use_hours:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"

    minutes = total_seconds // 60
    secs = total_seconds % 60
    return f"{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def _format_duration(seconds: float) -> str:
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_transcript(
    result: TranscribeResult,
    source_filename: str,
    model_name: str,
    device_info: str,
    language_mode: str,  # "detected" | "forced"
    transcription_date: datetime | None = None,  # None -> datetime.now()
) -> str:
    date = transcription_date or datetime.now()
    use_hours = result.duration > 3600

    lines: list[str] = []
    lines.append(f"# Транскрипт: {source_filename}")
    lines.append("")
    lines.append(f"- **Дата транскрипции**: {date.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **Модель**: {model_name}")
    lines.append(f"- **Язык**: {result.language} ({language_mode})")
    lines.append(f"- **Длительность**: {_format_duration(result.duration)}")
    lines.append(f"- **Устройство**: {device_info}")
    lines.append("")
    lines.append("---")

    if not result.segments:
        lines.append("")
        lines.append("*Речь не обнаружена.*")
    else:
        for seg in result.segments:
            start = format_timestamp(seg.start, use_hours=use_hours)
            end = format_timestamp(seg.end, use_hours=use_hours)
            lines.append("")
            lines.append(f"[{start} - {end}]{seg.text}")

    lines.append("")
    return "\n".join(lines)


def write_transcript(content: str, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
