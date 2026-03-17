from datetime import datetime
from pathlib import Path

from local_transcriber.formatter import (
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
    assert "[00:00.00 - 00:04.82] Добрый день, коллеги." in content
    assert "[00:04.82 - 00:09.15] Первый вопрос." in content


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


def test_write_transcript(tmp_path):
    out = tmp_path / "output.md"
    write_transcript("# Test content\n", out)
    assert out.read_text(encoding="utf-8") == "# Test content\n"
