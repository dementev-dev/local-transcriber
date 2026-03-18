from datetime import datetime
from pathlib import Path

from local_transcriber.formatter import (
    _group_segments,
    format_timestamp,
    format_transcript,
    write_transcript,
)
from local_transcriber.transcriber import Segment, TranscribeResult


def test_format_timestamp_minutes():
    assert format_timestamp(0.0) == "00:00.00"
    assert format_timestamp(83.45) == "01:23.45"
    assert format_timestamp(9.1) == "00:09.10"
    assert format_timestamp(599.99) == "09:59.99"
    assert format_timestamp(0.995) == "00:01.00"  # carry-over: не даёт .100


def test_format_timestamp_hours():
    assert format_timestamp(3723.45, use_hours=True) == "01:02:03.45"
    assert format_timestamp(0.0, use_hours=True) == "00:00:00.00"
    assert format_timestamp(7261.0, use_hours=True) == "02:01:01.00"


def test_format_transcript_basic():
    result = TranscribeResult(
        segments=[
            Segment(start=0.0, end=4.82, text=" Добрый день, коллеги."),
            Segment(start=4.82, end=9.15, text=" Первый вопрос."),
        ],
        language="ru",
        language_probability=0.97,
        duration=120.0,
        device_used="cuda",
    )
    content = format_transcript(
        result,
        source_filename="meeting.mp4",
        model_name="large-v3",
        device_info="CUDA (NVIDIA GeForce RTX 3060)",
        language_mode="detected",
        transcription_date=datetime(2026, 3, 17, 14, 30, 5),
    )

    assert "# Транскрипт: meeting.mp4" in content
    assert "**Дата транскрипции**: 2026-03-17 14:30:05" in content
    assert "**Модель**: large-v3" in content
    assert "**Язык**: ru (detected)" in content
    assert "**Длительность**: 02:00" in content
    assert "**Устройство**: CUDA (NVIDIA GeForce RTX 3060)" in content
    assert "---" in content
    # Соседние сегменты без паузы объединяются в один абзац
    assert "[00:00.00 - 00:09.15] Добрый день, коллеги. Первый вопрос." in content


def test_format_transcript_segment_no_leading_space():
    """Сегменты без ведущего пробела должны форматироваться корректно."""
    result = TranscribeResult(
        segments=[Segment(start=0.0, end=2.0, text="Hello")],
        language="en",
        language_probability=0.99,
        duration=5.0,
        device_used="cpu",
    )
    content = format_transcript(
        result,
        source_filename="f.mp3",
        model_name="tiny",
        device_info="CPU",
        language_mode="detected",
        transcription_date=datetime(2026, 1, 1, 0, 0, 0),
    )
    assert "[00:00.00 - 00:02.00] Hello" in content


def test_format_transcript_empty():
    result = TranscribeResult(
        segments=[],
        language="ru",
        language_probability=0.5,
        duration=30.0,
        device_used="cpu",
    )
    content = format_transcript(
        result,
        source_filename="silence.wav",
        model_name="tiny",
        device_info="CPU",
        language_mode="detected",
        transcription_date=datetime(2026, 1, 1, 0, 0, 0),
    )

    assert "# Транскрипт: silence.wav" in content
    assert "*Речь не обнаружена.*" in content
    assert "**Модель**: tiny" in content


def test_format_transcript_long():
    result = TranscribeResult(
        segments=[
            Segment(start=0.0, end=10.5, text=" Начало."),
            Segment(start=3700.0, end=3710.25, text=" Конец."),
        ],
        language="en",
        language_probability=0.99,
        duration=3800.0,
        device_used="cuda",
    )
    content = format_transcript(
        result,
        source_filename="long.mp4",
        model_name="large-v3",
        device_info="CUDA",
        language_mode="forced",
        transcription_date=datetime(2026, 3, 17, 10, 0, 0),
    )

    assert "**Длительность**: 01:03:20" in content
    assert "**Язык**: en (forced)" in content
    # Timestamps should use hours format
    assert "[00:00:00.00 - 00:00:10.50] Начало." in content
    assert "[01:01:40.00 - 01:01:50.25] Конец." in content


def test_group_segments_merges_adjacent():
    """Соседние сегменты без паузы объединяются."""
    segments = [
        Segment(start=0.0, end=3.0, text=" Первый."),
        Segment(start=3.0, end=6.0, text=" Второй."),
        Segment(start=6.0, end=9.0, text=" Третий."),
    ]
    groups = _group_segments(segments)
    assert len(groups) == 1
    assert groups[0].start == 0.0
    assert groups[0].end == 9.0
    assert groups[0].text == "Первый. Второй. Третий."


def test_group_segments_splits_on_pause():
    """Пауза > 2с разбивает на отдельные абзацы."""
    segments = [
        Segment(start=0.0, end=3.0, text=" Первый."),
        Segment(start=3.0, end=6.0, text=" Второй."),
        Segment(start=9.0, end=12.0, text=" После паузы."),
    ]
    groups = _group_segments(segments)
    assert len(groups) == 2
    assert groups[0].text == "Первый. Второй."
    assert groups[1].text == "После паузы."


def test_group_segments_splits_on_max_duration():
    """Абзац разбивается при превышении макс. длительности."""
    segments = [
        Segment(start=0.0, end=30.0, text=" Длинный."),
        Segment(start=30.0, end=55.0, text=" Ещё."),
        Segment(start=55.0, end=80.0, text=" Перелив."),
    ]
    groups = _group_segments(segments)
    assert len(groups) == 2
    assert groups[0].text == "Длинный. Ещё."
    assert groups[1].text == "Перелив."


def test_group_segments_empty():
    assert _group_segments([]) == []


def test_write_transcript(tmp_path):
    out = tmp_path / "output.md"
    write_transcript("# Test content\n", out)
    assert out.read_text(encoding="utf-8") == "# Test content\n"
