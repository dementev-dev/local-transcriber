from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from local_transcriber.cli import app
from local_transcriber.transcriber import Segment, TranscribeResult

runner = CliRunner()


def _make_result(segments=None, language="ru", device_used="cpu", duration=60.0):
    return TranscribeResult(
        segments=[Segment(start=0.0, end=2.0, text="Hello")] if segments is None else segments,
        language=language,
        language_probability=0.95,
        duration=duration,
        device_used=device_used,
    )


def _patches(result=None, tmp_file=None):
    """Context managers for a standard CLI happy path."""
    if result is None:
        result = _make_result()
    return [
        patch("local_transcriber.cli.check_ffmpeg"),
        patch("local_transcriber.cli.validate_input_file", return_value=tmp_file),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.transcribe", return_value=result),
        patch("local_transcriber.cli.write_transcript"),
    ]


def test_cli_happy_path_exit_code_zero(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    result = _make_result()

    with (
        patch("local_transcriber.cli.check_ffmpeg"),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/large-v3"),
        patch("local_transcriber.cli.transcribe", return_value=result),
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(audio)])

    assert out.exit_code == 0


def test_cli_default_options_passed_to_transcribe(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    result = _make_result()
    mock_transcribe = MagicMock(return_value=result)

    with (
        patch("local_transcriber.cli.check_ffmpeg"),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/large-v3"),
        patch("local_transcriber.cli.transcribe", mock_transcribe),
        patch("local_transcriber.cli.write_transcript"),
    ):
        runner.invoke(app, [str(audio)])

    call_kwargs = mock_transcribe.call_args[1]
    assert call_kwargs["model_name"] == "/models/large-v3"
    assert call_kwargs["device"] == "cpu"
    assert call_kwargs["compute_type"] == "int8"
    assert call_kwargs["language"] is None  # "auto" → None passed to transcribe
    assert call_kwargs["on_segment"] is None  # verbose=False


def test_cli_custom_options(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    result = _make_result()
    mock_transcribe = MagicMock(return_value=result)

    with (
        patch("local_transcriber.cli.check_ffmpeg"),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cuda"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/small"),
        patch("local_transcriber.cli.transcribe", mock_transcribe),
        patch("local_transcriber.cli.write_transcript"),
        patch("local_transcriber.cli.get_gpu_name", return_value="RTX 3060"),
    ):
        runner.invoke(app, [
            str(audio),
            "--model", "small",
            "--language", "ru",
            "--device", "cuda",
            "--compute-type", "float16",
        ])

    call_kwargs = mock_transcribe.call_args[1]
    assert call_kwargs["model_name"] == "/models/small"
    assert call_kwargs["language"] == "ru"  # explicit language passed through
    assert call_kwargs["compute_type"] == "float16"


def test_cli_verbose_passes_on_segment_callback(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    result = _make_result()
    mock_transcribe = MagicMock(return_value=result)

    with (
        patch("local_transcriber.cli.check_ffmpeg"),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/large-v3"),
        patch("local_transcriber.cli.transcribe", mock_transcribe),
        patch("local_transcriber.cli.write_transcript"),
    ):
        runner.invoke(app, [str(audio), "--verbose"])

    call_kwargs = mock_transcribe.call_args[1]
    assert call_kwargs["on_segment"] is not None
    assert callable(call_kwargs["on_segment"])


def test_cli_empty_speech_warning(tmp_path):
    audio = tmp_path / "silence.wav"
    audio.write_bytes(b"fake")
    result = _make_result(segments=[])

    with (
        patch("local_transcriber.cli.check_ffmpeg"),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/large-v3"),
        patch("local_transcriber.cli.transcribe", return_value=result),
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(audio)])

    assert out.exit_code == 0
    assert "Речь не обнаружена" in out.output


def test_cli_default_output_path(tmp_path):
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"fake")
    result = _make_result()
    mock_write = MagicMock()

    with (
        patch("local_transcriber.cli.check_ffmpeg"),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/large-v3"),
        patch("local_transcriber.cli.transcribe", return_value=result),
        patch("local_transcriber.cli.write_transcript", mock_write),
    ):
        runner.invoke(app, [str(audio)])

    written_path: Path = mock_write.call_args[0][1]
    assert written_path.name == "meeting-transcript.md"


def test_cli_custom_output_path(tmp_path):
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"fake")
    out_file = tmp_path / "custom.md"
    result = _make_result()
    mock_write = MagicMock()

    with (
        patch("local_transcriber.cli.check_ffmpeg"),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/large-v3"),
        patch("local_transcriber.cli.transcribe", return_value=result),
        patch("local_transcriber.cli.write_transcript", mock_write),
    ):
        runner.invoke(app, [str(audio), "--output", str(out_file)])

    written_path: Path = mock_write.call_args[0][1]
    assert written_path == out_file


def test_cli_error_exit_code_one(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")

    with patch("local_transcriber.cli.check_ffmpeg", side_effect=SystemExit(1)):
        out = runner.invoke(app, [str(audio)])

    assert out.exit_code == 1


def test_cli_passes_status_callback_to_transcribe(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    result = _make_result()
    mock_transcribe = MagicMock(return_value=result)

    with (
        patch("local_transcriber.cli.check_ffmpeg"),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/large-v3"),
        patch("local_transcriber.cli.transcribe", mock_transcribe),
        patch("local_transcriber.cli.write_transcript"),
    ):
        runner.invoke(app, [str(audio)])

    call_kwargs = mock_transcribe.call_args[1]
    assert call_kwargs["on_status"] is not None
    assert callable(call_kwargs["on_status"])


def test_cli_resolves_model_before_transcribe(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    result = _make_result()
    mock_transcribe = MagicMock(return_value=result)

    with (
        patch("local_transcriber.cli.check_ffmpeg"),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/large-v3") as mock_ensure_model,
        patch("local_transcriber.cli.transcribe", mock_transcribe),
        patch("local_transcriber.cli.write_transcript"),
    ):
        runner.invoke(app, [str(audio), "--model", "large-v3"])

    mock_ensure_model.assert_called_once()
    call_kwargs = mock_transcribe.call_args[1]
    assert call_kwargs["model_name"] == "/models/large-v3"


def test_cli_windows_cuda_diagnostic(tmp_path):
    """CUDA error on Windows prints choco/winget install hint."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")

    with (
        patch("local_transcriber.cli.check_ffmpeg"),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cuda"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/large-v3"),
        patch("local_transcriber.cli.transcribe", side_effect=RuntimeError("CUDA error: no device")),
        patch("local_transcriber.cli.sys") as mock_sys,
    ):
        mock_sys.platform = "win32"
        out = runner.invoke(app, [str(audio), "--device", "cuda"])

    assert out.exit_code == 1
    assert "choco install cuda" in out.output
    assert "winget install" in out.output


def test_cli_linux_cuda_error_no_windows_hint(tmp_path):
    """CUDA error on Linux does NOT print Windows-specific hint."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")

    with (
        patch("local_transcriber.cli.check_ffmpeg"),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cuda"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/large-v3"),
        patch("local_transcriber.cli.transcribe", side_effect=RuntimeError("CUDA error: no device")),
        patch("local_transcriber.cli.sys") as mock_sys,
    ):
        mock_sys.platform = "linux"
        out = runner.invoke(app, [str(audio), "--device", "cuda"])

    assert out.exit_code == 1
    assert "choco install cuda" not in out.output


def test_cli_device_fallback_warning(tmp_path):
    """When auto-detected device differs from actual, show fallback warning."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    result = _make_result(device_used="cpu")

    with (
        patch("local_transcriber.cli.check_ffmpeg"),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cuda"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/large-v3"),
        patch("local_transcriber.cli.transcribe", return_value=result),
        patch("local_transcriber.cli.write_transcript"),
    ):
        # --device auto (default) -> detect_device returns "cuda" but result is "cpu"
        out = runner.invoke(app, [str(audio)])

    assert "fallback" in out.output


def test_cli_strict_device_passed_to_transcribe(tmp_path):
    """--device cuda passes strict_device=True; default auto passes False."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    result = _make_result(device_used="cuda")
    mock_transcribe = MagicMock(return_value=result)

    with (
        patch("local_transcriber.cli.check_ffmpeg"),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cuda"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/large-v3"),
        patch("local_transcriber.cli.transcribe", mock_transcribe),
        patch("local_transcriber.cli.write_transcript"),
        patch("local_transcriber.cli.get_gpu_name", return_value="RTX 3060"),
    ):
        runner.invoke(app, [str(audio), "--device", "cuda"])

    assert mock_transcribe.call_args[1]["strict_device"] is True

    mock_transcribe.reset_mock()
    result_cpu = _make_result(device_used="cpu")
    mock_transcribe.return_value = result_cpu

    with (
        patch("local_transcriber.cli.check_ffmpeg"),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/large-v3"),
        patch("local_transcriber.cli.transcribe", mock_transcribe),
        patch("local_transcriber.cli.write_transcript"),
    ):
        runner.invoke(app, [str(audio)])

    assert mock_transcribe.call_args[1]["strict_device"] is False
