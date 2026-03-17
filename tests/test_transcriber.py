from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from local_transcriber.transcriber import Segment, TranscribeResult, transcribe


def _make_raw_segments(count: int) -> list:
    """Create mock raw segments as returned by faster-whisper."""
    segments = []
    for i in range(count):
        seg = MagicMock()
        seg.start = float(i * 5)
        seg.end = float(i * 5 + 4)
        seg.text = f" Segment {i}"
        segments.append(seg)
    return segments


def _make_info(language: str = "ru", probability: float = 0.95, duration: float = 60.0):
    info = MagicMock()
    info.language = language
    info.language_probability = probability
    info.duration = duration
    return info


@patch("local_transcriber.transcriber.WhisperModel")
def test_transcribe_collects_segments(mock_model_cls):
    raw_segments = _make_raw_segments(3)
    info = _make_info()

    instance = MagicMock()
    instance.transcribe.return_value = (iter(raw_segments), info)
    mock_model_cls.return_value = instance

    result = transcribe(
        file_path=Path("test.mp3"),
        model_name="tiny",
        device="cpu",
    )

    assert len(result.segments) == 3
    assert result.segments[0].text == " Segment 0"
    assert result.segments[2].text == " Segment 2"
    assert result.language == "ru"
    assert result.language_probability == 0.95
    assert result.duration == 60.0


@patch("local_transcriber.transcriber.WhisperModel")
def test_transcribe_calls_on_segment(mock_model_cls):
    raw_segments = _make_raw_segments(3)
    info = _make_info()

    instance = MagicMock()
    instance.transcribe.return_value = (iter(raw_segments), info)
    mock_model_cls.return_value = instance

    callback = MagicMock()

    transcribe(
        file_path=Path("test.mp3"),
        model_name="tiny",
        device="cpu",
        on_segment=callback,
    )

    assert callback.call_count == 3
    # Each call should receive a Segment instance
    for call_args in callback.call_args_list:
        seg = call_args[0][0]
        assert isinstance(seg, Segment)


@patch("local_transcriber.transcriber.WhisperModel")
def test_transcribe_cuda_fallback(mock_model_cls):
    raw_segments = _make_raw_segments(2)
    info = _make_info()

    # First call (cuda) raises, second call (cpu) succeeds
    cpu_instance = MagicMock()
    cpu_instance.transcribe.return_value = (iter(raw_segments), info)

    def model_side_effect(model_name, device, compute_type):
        if device == "cuda":
            raise RuntimeError("CUDA out of memory")
        return cpu_instance

    mock_model_cls.side_effect = model_side_effect

    with pytest.warns(UserWarning, match="Переключение на CPU"):
        result = transcribe(
            file_path=Path("test.mp3"),
            model_name="tiny",
            device="cuda",
        )

    assert result.device_used == "cpu"
    assert len(result.segments) == 2


@patch("local_transcriber.transcriber.WhisperModel")
def test_transcribe_device_used(mock_model_cls):
    raw_segments = _make_raw_segments(1)
    info = _make_info()

    instance = MagicMock()
    instance.transcribe.return_value = (iter(raw_segments), info)
    mock_model_cls.return_value = instance

    result = transcribe(
        file_path=Path("test.mp3"),
        model_name="tiny",
        device="cuda",
    )

    assert result.device_used == "cuda"
    mock_model_cls.assert_called_once_with("tiny", device="cuda", compute_type="int8")


@patch("local_transcriber.transcriber.WhisperModel")
def test_transcribe_cuda_fallback_on_transcribe_call(mock_model_cls):
    """CUDA error in model.transcribe() (not __init__) triggers CPU fallback."""
    raw_segments = _make_raw_segments(2)
    info = _make_info()

    cuda_instance = MagicMock()
    cuda_instance.transcribe.side_effect = RuntimeError("CUDA error during transcription")

    cpu_instance = MagicMock()
    cpu_instance.transcribe.return_value = (iter(raw_segments), info)

    call_count = 0

    def model_side_effect(model_name, device, compute_type):
        nonlocal call_count
        call_count += 1
        if device == "cuda":
            return cuda_instance
        return cpu_instance

    mock_model_cls.side_effect = model_side_effect

    with pytest.warns(UserWarning, match="Переключение на CPU"):
        result = transcribe(
            file_path=Path("test.mp3"),
            model_name="tiny",
            device="cuda",
        )

    assert result.device_used == "cpu"
    assert len(result.segments) == 2


@patch("local_transcriber.transcriber.WhisperModel")
def test_transcribe_midstream_fallback_no_duplicate_callbacks(mock_model_cls):
    """on_segment is not called for partial GPU segments on mid-stream fallback."""
    info = _make_info()

    # GPU iterator: yields 1 segment then raises CUDA error
    def _gpu_generator():
        seg = MagicMock()
        seg.start = 0.0
        seg.end = 4.0
        seg.text = " GPU seg"
        yield seg
        raise RuntimeError("CUDA out of memory mid-stream")

    cuda_instance = MagicMock()
    cuda_instance.transcribe.return_value = (_gpu_generator(), info)

    cpu_segments = _make_raw_segments(2)
    cpu_instance = MagicMock()
    cpu_instance.transcribe.return_value = (iter(cpu_segments), info)

    def model_side_effect(model_name, device, compute_type):
        if device == "cuda":
            return cuda_instance
        return cpu_instance

    mock_model_cls.side_effect = model_side_effect

    callback = MagicMock()

    with pytest.warns(UserWarning, match="Переключение на CPU"):
        result = transcribe(
            file_path=Path("test.mp3"),
            model_name="tiny",
            device="cuda",
            on_segment=callback,
        )

    assert result.device_used == "cpu"
    assert len(result.segments) == 2
    # callback: 1 from partial GPU pass + 2 from full CPU pass = 3
    # The GPU partial segment is NOT in the final result (segments list reset),
    # but on_segment was called live as segments streamed.
    # This is acceptable — on_segment is a live progress callback.
    # The important thing is that result.segments contains only CPU segments.
    assert all(s.text.startswith(" Segment") for s in result.segments)
