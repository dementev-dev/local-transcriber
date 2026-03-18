from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from local_transcriber.cli import app
from local_transcriber.transcriber import Segment, TranscribeFileResult, TranscribeResult

runner = CliRunner()


def _make_result(segments=None, language="ru", device_used="cpu", duration=60.0):
    return TranscribeResult(
        segments=[Segment(start=0.0, end=2.0, text="Hello")] if segments is None else segments,
        language=language,
        language_probability=0.95,
        duration=duration,
        device_used=device_used,
    )


def _make_model():
    return MagicMock(name="WhisperModel")


def _make_tfr(result=None, model=None, actual_device="cpu"):
    if result is None:
        result = _make_result()
    if model is None:
        model = _make_model()
    return TranscribeFileResult(result=result, model=model, actual_device=actual_device)


def _single_patches(result=None, tmp_file=None, actual_device="cpu"):
    """Patches for a standard single-file CLI happy path."""
    if result is None:
        result = _make_result(device_used=actual_device)
    model = _make_model()
    tfr = TranscribeFileResult(result=result, model=model, actual_device=actual_device)
    return [
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=tmp_file),
        patch("local_transcriber.cli.detect_device", return_value=actual_device),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/medium"),
        patch("local_transcriber.cli.load_model", return_value=(model, actual_device)),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript"),
    ]


def test_cli_happy_path_exit_code_zero(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")

    patches = _single_patches(tmp_file=audio)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
        out = runner.invoke(app, [str(audio)])

    assert out.exit_code == 0


def test_cli_default_options_passed_to_transcribe(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    result = _make_result()
    model = _make_model()
    tfr = _make_tfr(result=result, model=model)
    mock_transcribe_file = MagicMock(return_value=tfr)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/medium"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu")),
        patch("local_transcriber.cli._transcribe_file", mock_transcribe_file),
        patch("local_transcriber.cli.write_transcript"),
    ):
        runner.invoke(app, [str(audio)])

    call_kwargs = mock_transcribe_file.call_args[1]
    assert call_kwargs["model_name"] == "/models/medium"
    assert call_kwargs["compute_type"] == "float32"
    assert call_kwargs["language"] == "ru"
    assert call_kwargs["on_segment"] is None  # verbose=False


def test_cli_custom_options(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    result = _make_result(device_used="cuda")
    model = _make_model()
    tfr = _make_tfr(result=result, model=model, actual_device="cuda")
    mock_transcribe_file = MagicMock(return_value=tfr)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cuda"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/small"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cuda")),
        patch("local_transcriber.cli._transcribe_file", mock_transcribe_file),
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

    call_kwargs = mock_transcribe_file.call_args[1]
    assert call_kwargs["model_name"] == "/models/small"
    assert call_kwargs["language"] == "ru"
    assert call_kwargs["compute_type"] == "float16"


def test_cli_verbose_passes_on_segment_callback(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    result = _make_result()
    model = _make_model()
    tfr = _make_tfr(result=result, model=model)
    mock_transcribe_file = MagicMock(return_value=tfr)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/medium"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu")),
        patch("local_transcriber.cli._transcribe_file", mock_transcribe_file),
        patch("local_transcriber.cli.write_transcript"),
    ):
        runner.invoke(app, [str(audio), "--verbose"])

    call_kwargs = mock_transcribe_file.call_args[1]
    assert call_kwargs["on_segment"] is not None
    assert callable(call_kwargs["on_segment"])


def test_cli_empty_speech_warning(tmp_path):
    audio = tmp_path / "silence.wav"
    audio.write_bytes(b"fake")
    result = _make_result(segments=[])

    patches = _single_patches(result=result, tmp_file=audio)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
        out = runner.invoke(app, [str(audio)])

    assert out.exit_code == 0
    assert "Речь не обнаружена" in out.output


def test_cli_default_output_path(tmp_path):
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"fake")
    mock_write = MagicMock()

    result = _make_result()
    model = _make_model()
    tfr = _make_tfr(result=result, model=model)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/medium"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu")),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript", mock_write),
    ):
        runner.invoke(app, [str(audio)])

    written_path: Path = mock_write.call_args[0][1]
    assert written_path.name == "meeting-transcript.md"


def test_cli_custom_output_path(tmp_path):
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"fake")
    out_file = tmp_path / "custom.md"
    mock_write = MagicMock()

    result = _make_result()
    model = _make_model()
    tfr = _make_tfr(result=result, model=model)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/medium"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu")),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript", mock_write),
    ):
        runner.invoke(app, [str(audio), "--output", str(out_file)])

    written_path: Path = mock_write.call_args[0][1]
    assert written_path == out_file


def test_cli_passes_status_callback_to_transcribe(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    result = _make_result()
    model = _make_model()
    tfr = _make_tfr(result=result, model=model)
    mock_transcribe_file = MagicMock(return_value=tfr)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/medium"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu")),
        patch("local_transcriber.cli._transcribe_file", mock_transcribe_file),
        patch("local_transcriber.cli.write_transcript"),
    ):
        runner.invoke(app, [str(audio)])

    call_kwargs = mock_transcribe_file.call_args[1]
    assert call_kwargs["on_status"] is not None
    assert callable(call_kwargs["on_status"])


def test_cli_resolves_model_before_transcribe(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    result = _make_result()
    model = _make_model()
    tfr = _make_tfr(result=result, model=model)
    mock_transcribe_file = MagicMock(return_value=tfr)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/large-v3") as mock_ensure,
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu")),
        patch("local_transcriber.cli._transcribe_file", mock_transcribe_file),
        patch("local_transcriber.cli.write_transcript"),
    ):
        runner.invoke(app, [str(audio), "--model", "large-v3"])

    mock_ensure.assert_called_once()
    call_kwargs = mock_transcribe_file.call_args[1]
    assert call_kwargs["model_name"] == "/models/large-v3"


def test_cli_windows_cuda_diagnostic(tmp_path):
    """CUDA error on Windows prints choco/winget install hint."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    model = _make_model()

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cuda"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/medium"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cuda")),
        patch("local_transcriber.cli._transcribe_file", side_effect=RuntimeError("CUDA error: no device")),
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
    model = _make_model()

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cuda"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/medium"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cuda")),
        patch("local_transcriber.cli._transcribe_file", side_effect=RuntimeError("CUDA error: no device")),
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
    model = _make_model()
    tfr = TranscribeFileResult(result=result, model=model, actual_device="cpu")

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cuda"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/medium"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cuda")),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(audio)])

    assert "fallback" in out.output


def test_cli_strict_device_passed_to_transcribe(tmp_path):
    """--device cuda passes strict_device=True; default auto passes False."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    result = _make_result(device_used="cuda")
    model = _make_model()
    tfr = TranscribeFileResult(result=result, model=model, actual_device="cuda")
    mock_transcribe_file = MagicMock(return_value=tfr)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cuda"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/medium"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cuda")),
        patch("local_transcriber.cli._transcribe_file", mock_transcribe_file),
        patch("local_transcriber.cli.write_transcript"),
        patch("local_transcriber.cli.get_gpu_name", return_value="RTX 3060"),
    ):
        runner.invoke(app, [str(audio), "--device", "cuda"])

    assert mock_transcribe_file.call_args[1]["strict_device"] is True

    mock_transcribe_file.reset_mock()
    result_cpu = _make_result(device_used="cpu")
    tfr_cpu = TranscribeFileResult(result=result_cpu, model=model, actual_device="cpu")
    mock_transcribe_file.return_value = tfr_cpu

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/medium"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu")),
        patch("local_transcriber.cli._transcribe_file", mock_transcribe_file),
        patch("local_transcriber.cli.write_transcript"),
    ):
        runner.invoke(app, [str(audio)])

    assert mock_transcribe_file.call_args[1]["strict_device"] is False


def test_cli_keyboard_interrupt(tmp_path):
    """Ctrl+C → exit code 130, 'Прервано пользователем' in output."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    model = _make_model()

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/medium"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu")),
        patch("local_transcriber.cli._transcribe_file", side_effect=KeyboardInterrupt),
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(audio)])

    assert out.exit_code == 130
    assert "Прервано пользователем" in out.output


def test_cli_user_error_no_traceback(tmp_path):
    """FileNotFoundError → clean message, no traceback."""
    audio = tmp_path / "missing.mp3"

    with patch("local_transcriber.cli.load_config", return_value={}):
        out = runner.invoke(app, [str(audio)])

    assert out.exit_code == 1
    assert "Ошибка" in out.output
    assert "Traceback" not in out.output


def test_cli_unexpected_error_verbose_traceback(tmp_path):
    """Unexpected error with --verbose → traceback shown."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    model = _make_model()

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/medium"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu")),
        patch("local_transcriber.cli._transcribe_file", side_effect=RuntimeError("unexpected boom")),
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(audio), "--verbose"])

    assert out.exit_code == 1
    assert "unexpected boom" in out.output


def test_cli_unexpected_error_no_verbose_hint(tmp_path):
    """Unexpected error without --verbose → hint to use --verbose."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    model = _make_model()

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/medium"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu")),
        patch("local_transcriber.cli._transcribe_file", side_effect=RuntimeError("unexpected boom")),
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(audio)])

    assert out.exit_code == 1
    assert "Ошибка" in out.output
    assert "--verbose" in out.output


# === Batch mode tests ===


def test_cli_batch_two_files(tmp_path):
    a = tmp_path / "a.mp3"
    b = tmp_path / "b.mp3"
    a.write_bytes(b"fake")
    b.write_bytes(b"fake")

    result = _make_result()
    model = _make_model()
    tfr = _make_tfr(result=result, model=model)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", side_effect=lambda p: p),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/medium"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu")),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(a), str(b)])

    assert out.exit_code == 0
    assert "2 обработано" in out.output


def test_cli_batch_skips_existing(tmp_path):
    a = tmp_path / "a.mp3"
    b = tmp_path / "b.mp3"
    a.write_bytes(b"fake")
    b.write_bytes(b"fake")
    # Create transcript for a
    (tmp_path / "a-transcript.md").write_text("existing")

    result = _make_result()
    model = _make_model()
    tfr = _make_tfr(result=result, model=model)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", side_effect=lambda p: p),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/medium"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu")),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(a), str(b)])

    assert out.exit_code == 0
    assert "Пропуск" in out.output
    assert "1 обработано" in out.output
    assert "1 пропущено" in out.output


def test_cli_batch_all_skipped_no_model_load(tmp_path):
    a = tmp_path / "a.mp3"
    a.write_bytes(b"fake")
    (tmp_path / "a-transcript.md").write_text("existing")
    b = tmp_path / "b.mp3"
    b.write_bytes(b"fake")
    (tmp_path / "b-transcript.md").write_text("existing")

    mock_load_model = MagicMock()

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", side_effect=lambda p: p),
        patch("local_transcriber.cli.load_model", mock_load_model),
    ):
        out = runner.invoke(app, [str(a), str(b)])

    assert out.exit_code == 0
    mock_load_model.assert_not_called()


def test_cli_batch_force_overwrites(tmp_path):
    a = tmp_path / "a.mp3"
    a.write_bytes(b"fake")
    (tmp_path / "a-transcript.md").write_text("existing")
    b = tmp_path / "b.mp3"
    b.write_bytes(b"fake")

    result = _make_result()
    model = _make_model()
    tfr = _make_tfr(result=result, model=model)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", side_effect=lambda p: p),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/medium"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu")),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(a), str(b), "--force"])

    assert out.exit_code == 0
    assert "Пропуск" not in out.output
    assert "2 обработано" in out.output


def test_cli_batch_per_file_error(tmp_path):
    a = tmp_path / "a.mp3"
    b = tmp_path / "b.mp3"
    a.write_bytes(b"fake")
    b.write_bytes(b"fake")

    result = _make_result()
    model = _make_model()
    tfr = _make_tfr(result=result, model=model)
    call_count = 0

    def transcribe_side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("oops")
        return tfr

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", side_effect=lambda p: p),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/medium"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu")),
        patch("local_transcriber.cli._transcribe_file", side_effect=transcribe_side_effect),
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(a), str(b)])

    assert out.exit_code == 1
    assert "1 обработано" in out.output
    assert "1 ошибок" in out.output


def test_cli_batch_invalid_in_prescan(tmp_path):
    a = tmp_path / "a.mp3"
    a.write_bytes(b"fake")
    b = tmp_path / "b.mp3"
    # b doesn't exist

    result = _make_result()
    model = _make_model()
    tfr = _make_tfr(result=result, model=model)

    def validate_side_effect(p):
        if not p.exists():
            raise FileNotFoundError(f"Файл не найден: {p}")
        return p

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", side_effect=validate_side_effect),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/medium"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu")),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(a), str(b)])

    assert out.exit_code == 1
    assert "1 обработано" in out.output
    assert "1 ошибок" in out.output


def test_cli_batch_output_incompatible(tmp_path):
    a = tmp_path / "a.mp3"
    b = tmp_path / "b.mp3"
    a.write_bytes(b"fake")
    b.write_bytes(b"fake")

    with patch("local_transcriber.cli.load_config", return_value={}):
        out = runner.invoke(app, [str(a), str(b), "--output", "out.md"])

    assert out.exit_code == 1
    assert "--output несовместим" in out.output


def test_cli_batch_empty_glob(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with patch("local_transcriber.cli.load_config", return_value={}):
        out = runner.invoke(app, ["*.mp3"])

    assert out.exit_code == 1
    assert "Файлы не найдены" in out.output


def test_cli_config_applied(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    model = _make_model()
    result = _make_result()
    tfr = _make_tfr(result=result, model=model)

    with (
        patch("local_transcriber.cli.load_config", return_value={"model": "tiny"}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/tiny") as mock_ensure,
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu")),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript"),
    ):
        runner.invoke(app, [str(audio)])

    mock_ensure.assert_called_once_with("tiny", on_status=mock_ensure.call_args[1]["on_status"])


def test_cli_cli_overrides_config(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    model = _make_model()
    result = _make_result()
    tfr = _make_tfr(result=result, model=model)

    with (
        patch("local_transcriber.cli.load_config", return_value={"model": "tiny"}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/small") as mock_ensure,
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu")),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript"),
    ):
        runner.invoke(app, [str(audio), "--model", "small"])

    mock_ensure.assert_called_once_with("small", on_status=mock_ensure.call_args[1]["on_status"])


def test_cli_batch_fallback_warning(tmp_path):
    """Batch mode shows fallback warning when load_model falls back to CPU."""
    a = tmp_path / "a.mp3"
    b = tmp_path / "b.mp3"
    a.write_bytes(b"fake")
    b.write_bytes(b"fake")

    result = _make_result(device_used="cpu")
    model = _make_model()
    tfr = TranscribeFileResult(result=result, model=model, actual_device="cpu")

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", side_effect=lambda p: p),
        patch("local_transcriber.cli.detect_device", return_value="cuda"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/medium"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu")),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(a), str(b)])

    assert "fallback" in out.output


def test_cli_batch_empty_speech_warning(tmp_path):
    """Batch mode warns when a file has no detected speech."""
    a = tmp_path / "a.mp3"
    b = tmp_path / "b.mp3"
    a.write_bytes(b"fake")
    b.write_bytes(b"fake")

    result_empty = _make_result(segments=[])
    result_ok = _make_result()
    model = _make_model()
    tfr_empty = _make_tfr(result=result_empty, model=model)
    tfr_ok = _make_tfr(result=result_ok, model=model)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", side_effect=lambda p: p),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/medium"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu")),
        patch("local_transcriber.cli._transcribe_file", side_effect=[tfr_empty, tfr_ok]),
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(a), str(b)])

    assert out.exit_code == 0
    assert "Речь не обнаружена" in out.output
    assert "2 обработано" in out.output


def test_cli_batch_midstream_fallback_warning(tmp_path):
    """Batch mode shows warning when _transcribe_file falls back mid-stream."""
    a = tmp_path / "a.mp3"
    b = tmp_path / "b.mp3"
    a.write_bytes(b"fake")
    b.write_bytes(b"fake")

    model_gpu = _make_model()
    model_cpu = _make_model()
    result = _make_result(device_used="cpu")
    # First file triggers mid-stream fallback
    tfr_fallback = TranscribeFileResult(result=result, model=model_cpu, actual_device="cpu")
    tfr_ok = TranscribeFileResult(result=result, model=model_cpu, actual_device="cpu")

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", side_effect=lambda p: p),
        patch("local_transcriber.cli.detect_device", return_value="cuda"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/medium"),
        patch("local_transcriber.cli.load_model", return_value=(model_gpu, "cuda")),
        patch("local_transcriber.cli._transcribe_file", side_effect=[tfr_fallback, tfr_ok]),
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(a), str(b)])

    assert "fallback" in out.output
    assert "2 обработано" in out.output


def test_cli_batch_model_loaded_once(tmp_path):
    a = tmp_path / "a.mp3"
    b = tmp_path / "b.mp3"
    a.write_bytes(b"fake")
    b.write_bytes(b"fake")

    result = _make_result()
    model = _make_model()
    tfr = _make_tfr(result=result, model=model)
    mock_load_model = MagicMock(return_value=(model, "cpu"))

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", side_effect=lambda p: p),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.ensure_model_available", return_value="/models/medium"),
        patch("local_transcriber.cli.load_model", mock_load_model),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(a), str(b)])

    assert out.exit_code == 0
    mock_load_model.assert_called_once()
