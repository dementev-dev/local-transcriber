"""Формирование markdown-транскрипта из результатов распознавания."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .quality import TAIL_GAP_WARN_S, find_repetition_blocks, tail_gap
from .types import Segment, SpeakerTranscript, TranscribeResult

_PAUSE_THRESHOLD_S = 2.0  # пауза между сегментами для разбиения на абзацы
_MAX_PARAGRAPH_S = 60.0  # максимальная длительность абзаца

# Источник языка в шапке транскрипта
LANGUAGE_FORCED = "задан явно"
LANGUAGE_DETECTED = "определён автоматически"
LANGUAGE_FROM_MODEL = "из профиля модели"
LANGUAGE_UNKNOWN = "не определён"

LANGUAGE_MODES = (
    LANGUAGE_FORCED,
    LANGUAGE_DETECTED,
    LANGUAGE_FROM_MODEL,
    LANGUAGE_UNKNOWN,
)


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
    """Форматирует время в ``MM:SS.cc`` или ``HH:MM:SS.cc``.

    Сотые доли (centiseconds) — максимальная точность, которую даёт Whisper.
    """
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


def format_duration(seconds: float) -> str:
    """Человекочитаемая длительность для метаданных в шапке транскрипта."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _format_speaker_timestamp(seconds: float, use_hours: bool) -> str:
    total_seconds = int(seconds)
    if use_hours:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    minutes = total_seconds // 60
    secs = total_seconds % 60
    return f"{minutes:02d}:{secs:02d}"


def format_transcript(
    result: TranscribeResult,
    source_filename: str,
    model_name: str,
    device_info: str,
    language_mode: str,  # см. LANGUAGE_MODES
    transcription_date: datetime | None = None,  # None -> datetime.now()
    speaker_transcript: SpeakerTranscript | None = None,
    diarization_warning: str | None = None,
) -> str:
    """Собирает markdown-транскрипт: шапка с метаданными + абзацы с таймкодами."""
    date = transcription_date or datetime.now()
    use_hours = result.duration > 3600

    lines: list[str] = []
    lines.append(f"# Транскрипт: {source_filename}")
    lines.append("")
    lines.append(f"- **Дата транскрипции**: {date.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **Модель**: {model_name}")
    if language_mode == LANGUAGE_UNKNOWN:
        lines.append(f"- **Язык**: {LANGUAGE_UNKNOWN}")
    else:
        lines.append(f"- **Язык**: {result.language} ({language_mode})")
    lines.append(f"- **Длительность**: {format_duration(result.duration)}")
    if tail_gap(result) > TAIL_GAP_WARN_S:
        last_end = result.segments[-1].end
        lines.append(
            f"- **Внимание**: транскрипт покрывает {format_duration(last_end)} "
            f"из {format_duration(result.duration)} — возможна потеря хвоста записи"
        )
    for block in find_repetition_blocks(result.segments):
        start = format_timestamp(block.start, use_hours=use_hours)
        end = format_timestamp(block.end, use_hours=use_hours)
        lines.append(
            f"- **Внимание**: повторы в [{start} - {end}] ({block.count}×) "
            "— возможны галлюцинации модели"
        )
    if speaker_transcript is not None:
        lines.append(f"- **Голосовых кластеров**: {speaker_transcript.cluster_count}")
        if speaker_transcript.unassigned_word_count:
            lines.append(
                "- **Внимание**: "
                f"{speaker_transcript.unassigned_word_count} слов без назначенного говорящего"
            )
        for cluster in speaker_transcript.small_clusters:
            label = (
                f"Speaker {cluster.speaker}"
                if cluster.speaker is not None
                else "кластер без номера"
            )
            lines.append(
                f"- **Внимание**: малый кластер {label}: {cluster.duration:.1f} с"
            )
    if diarization_warning is not None:
        lines.append(f"- **Внимание**: {diarization_warning}")
    lines.append(f"- **Устройство**: {device_info}")
    lines.append("")
    lines.append("---")

    if not result.segments:
        lines.append("")
        lines.append("*Речь не обнаружена.*")
    elif speaker_transcript is not None and speaker_transcript.cluster_count >= 2:
        for turn in speaker_transcript.turns:
            timestamp = _format_speaker_timestamp(turn.start, use_hours)
            speaker = turn.speaker if turn.speaker is not None else "?"
            lines.append("")
            lines.append(f"[{timestamp}] Speaker {speaker}: {turn.text}")
    else:
        for para in _group_segments(result.segments):
            start = format_timestamp(para.start, use_hours=use_hours)
            end = format_timestamp(para.end, use_hours=use_hours)
            lines.append("")
            lines.append(f"[{start} - {end}] {para.text}")

    lines.append("")
    return "\n".join(lines)


def write_transcript(content: str, output_path: Path) -> None:
    """Записывает готовый транскрипт в файл."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
