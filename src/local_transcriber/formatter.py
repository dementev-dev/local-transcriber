from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .transcriber import Segment, TranscribeResult

_PAUSE_THRESHOLD_S = 2.0  # пауза между сегментами для разбиения на абзацы
_MAX_PARAGRAPH_S = 60.0  # максимальная длительность абзаца


@dataclass
class _Paragraph:
    start: float
    end: float
    text: str


def _group_segments(segments: list[Segment]) -> list[_Paragraph]:
    """Объединяет мелкие сегменты в абзацы по паузам и макс. длительности."""
    if not segments:
        return []

    paragraphs: list[_Paragraph] = []
    cur_start = segments[0].start
    cur_end = segments[0].end
    cur_texts: list[str] = [segments[0].text.strip()]

    for seg in segments[1:]:
        gap = seg.start - cur_end
        duration = seg.end - cur_start
        if gap >= _PAUSE_THRESHOLD_S or duration > _MAX_PARAGRAPH_S:
            paragraphs.append(_Paragraph(cur_start, cur_end, " ".join(cur_texts)))
            cur_start = seg.start
            cur_end = seg.end
            cur_texts = [seg.text.strip()]
        else:
            cur_end = seg.end
            cur_texts.append(seg.text.strip())

    paragraphs.append(_Paragraph(cur_start, cur_end, " ".join(cur_texts)))
    return paragraphs


def format_timestamp(seconds: float, use_hours: bool = False) -> str:
    total_cs = round(seconds * 100)
    centiseconds = total_cs % 100
    total_seconds = total_cs // 100

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
        for para in _group_segments(result.segments):
            start = format_timestamp(para.start, use_hours=use_hours)
            end = format_timestamp(para.end, use_hours=use_hours)
            lines.append("")
            lines.append(f"[{start} - {end}] {para.text}")

    lines.append("")
    return "\n".join(lines)


def write_transcript(content: str, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
