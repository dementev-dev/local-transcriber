from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from local_transcriber.cli import _format_device_info, app
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


def _make_backend():
    return MagicMock(name="Backend")


def _make_tfr(result=None, model=None, actual_device="cpu", backend=None, model_path="/models/medium"):
    if result is None:
        result = _make_result()
    if model is None:
        model = _make_model()
    if backend is None:
        backend = _make_backend()
    return TranscribeFileResult(
        result=result, model=model, actual_device=actual_device,
        backend=backend, model_path=model_path,
    )


def _single_patches(result=None, tmp_file=None, actual_device="cpu"):
    """Patches for a standard single-file CLI happy path."""
    if result is None:
        result = _make_result(device_used=actual_device)
    model = _make_model()
    backend = _make_backend()
    tfr = _make_tfr(result=result, model=model, actual_device=actual_device, backend=backend)
    return [
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=tmp_file),
        patch("local_transcriber.cli.detect_device", return_value=actual_device),
        patch("local_transcriber.cli.load_model", return_value=(model, actual_device, backend, "/models/medium")),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript"),
    ]


def test_cli_happy_path_exit_code_zero(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")

    patches = _single_patches(tmp_file=audio)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        out = runner.invoke(app, [str(audio)])

    assert out.exit_code == 0


def test_cli_default_options_passed_to_transcribe(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    result = _make_result()
    model = _make_model()
    backend = _make_backend()
    tfr = _make_tfr(result=result, model=model, backend=backend)
    mock_transcribe_file = MagicMock(return_value=tfr)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu", backend, "/models/medium")),
        patch("local_transcriber.cli._transcribe_file", mock_transcribe_file),
        patch("local_transcriber.cli.write_transcript"),
    ):
        runner.invoke(app, [str(audio)])

    call_kwargs = mock_transcribe_file.call_args[1]
    assert call_kwargs["model_name"] == "medium"
    assert call_kwargs["compute_type"] == "float32"
    assert call_kwargs["language"] == "ru"
    assert call_kwargs["on_segment"] is None  # verbose=False


def test_cli_custom_options(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    result = _make_result(device_used="cuda")
    model = _make_model()
    backend = _make_backend()
    tfr = _make_tfr(result=result, model=model, actual_device="cuda", backend=backend)
    mock_transcribe_file = MagicMock(return_value=tfr)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cuda"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cuda", backend, "/models/small")),
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
    assert call_kwargs["model_name"] == "small"
    assert call_kwargs["language"] == "ru"
    assert call_kwargs["compute_type"] == "float16"


def test_cli_verbose_passes_on_segment_callback(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    result = _make_result()
    model = _make_model()
    backend = _make_backend()
    tfr = _make_tfr(result=result, model=model, backend=backend)
    mock_transcribe_file = MagicMock(return_value=tfr)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu", backend, "/models/medium")),
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
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        out = runner.invoke(app, [str(audio)])

    assert out.exit_code == 0
    assert "Речь не обнаружена" in out.output


def test_cli_default_output_path(tmp_path):
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"fake")
    mock_write = MagicMock()

    result = _make_result()
    model = _make_model()
    backend = _make_backend()
    tfr = _make_tfr(result=result, model=model, backend=backend)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu", backend, "/models/medium")),
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
    backend = _make_backend()
    tfr = _make_tfr(result=result, model=model, backend=backend)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu", backend, "/models/medium")),
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
    backend = _make_backend()
    tfr = _make_tfr(result=result, model=model, backend=backend)
    mock_transcribe_file = MagicMock(return_value=tfr)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu", backend, "/models/medium")),
        patch("local_transcriber.cli._transcribe_file", mock_transcribe_file),
        patch("local_transcriber.cli.write_transcript"),
    ):
        runner.invoke(app, [str(audio)])

    call_kwargs = mock_transcribe_file.call_args[1]
    assert call_kwargs["on_status"] is not None
    assert callable(call_kwargs["on_status"])


def test_cli_load_model_called_with_model_name(tmp_path):
    """load_model receives model name from defaults, handles ensure internally."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    result = _make_result()
    model = _make_model()
    backend = _make_backend()
    tfr = _make_tfr(result=result, model=model, backend=backend)
    mock_load_model = MagicMock(return_value=(model, "cpu", backend, "/models/large-v3"))

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.load_model", mock_load_model),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript"),
    ):
        runner.invoke(app, [str(audio), "--model", "large-v3"])

    assert mock_load_model.call_args[0][0] == "large-v3"


def test_cli_windows_cuda_diagnostic(tmp_path):
    """CUDA error on Windows prints choco/winget install hint."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    model = _make_model()
    backend = _make_backend()

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cuda"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cuda", backend, "/models/medium")),
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
    backend = _make_backend()

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cuda"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cuda", backend, "/models/medium")),
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
    backend = _make_backend()
    tfr = _make_tfr(result=result, model=model, actual_device="cpu", backend=backend)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cuda"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cuda", backend, "/models/medium")),
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
    backend = _make_backend()
    tfr = _make_tfr(result=result, model=model, actual_device="cuda", backend=backend)
    mock_transcribe_file = MagicMock(return_value=tfr)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cuda"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cuda", backend, "/models/medium")),
        patch("local_transcriber.cli._transcribe_file", mock_transcribe_file),
        patch("local_transcriber.cli.write_transcript"),
        patch("local_transcriber.cli.get_gpu_name", return_value="RTX 3060"),
    ):
        runner.invoke(app, [str(audio), "--device", "cuda"])

    assert mock_transcribe_file.call_args[1]["strict_device"] is True

    mock_transcribe_file.reset_mock()
    result_cpu = _make_result(device_used="cpu")
    tfr_cpu = _make_tfr(result=result_cpu, model=model, backend=backend)
    mock_transcribe_file.return_value = tfr_cpu

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu", backend, "/models/medium")),
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
    backend = _make_backend()

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu", backend, "/models/medium")),
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
    backend = _make_backend()

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu", backend, "/models/medium")),
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
    backend = _make_backend()

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu", backend, "/models/medium")),
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
    backend = _make_backend()
    tfr = _make_tfr(result=result, model=model, backend=backend)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", side_effect=lambda p: p),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu", backend, "/models/medium")),
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
    (tmp_path / "a-transcript.md").write_text("existing")

    result = _make_result()
    model = _make_model()
    backend = _make_backend()
    tfr = _make_tfr(result=result, model=model, backend=backend)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", side_effect=lambda p: p),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu", backend, "/models/medium")),
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
    backend = _make_backend()
    tfr = _make_tfr(result=result, model=model, backend=backend)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", side_effect=lambda p: p),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu", backend, "/models/medium")),
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
    backend = _make_backend()
    tfr = _make_tfr(result=result, model=model, backend=backend)
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
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu", backend, "/models/medium")),
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
    backend = _make_backend()
    tfr = _make_tfr(result=result, model=model, backend=backend)

    def validate_side_effect(p):
        if not p.exists():
            raise FileNotFoundError(f"Файл не найден: {p}")
        return p

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", side_effect=validate_side_effect),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu", backend, "/models/medium")),
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
    backend = _make_backend()
    result = _make_result()
    tfr = _make_tfr(result=result, model=model, backend=backend)
    mock_load_model = MagicMock(return_value=(model, "cpu", backend, "/models/tiny"))

    with (
        patch("local_transcriber.cli.load_config", return_value={"model": "tiny"}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.load_model", mock_load_model),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript"),
    ):
        runner.invoke(app, [str(audio)])

    # load_model receives model name from config
    assert mock_load_model.call_args[0][0] == "tiny"


def test_cli_cli_overrides_config(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    model = _make_model()
    backend = _make_backend()
    result = _make_result()
    tfr = _make_tfr(result=result, model=model, backend=backend)
    mock_load_model = MagicMock(return_value=(model, "cpu", backend, "/models/small"))

    with (
        patch("local_transcriber.cli.load_config", return_value={"model": "tiny"}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.load_model", mock_load_model),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript"),
    ):
        runner.invoke(app, [str(audio), "--model", "small"])

    # CLI --model overrides config
    assert mock_load_model.call_args[0][0] == "small"


def test_cli_batch_fallback_warning(tmp_path):
    """Batch mode shows fallback warning when load_model falls back to CPU."""
    a = tmp_path / "a.mp3"
    b = tmp_path / "b.mp3"
    a.write_bytes(b"fake")
    b.write_bytes(b"fake")

    result = _make_result(device_used="cpu")
    model = _make_model()
    backend = _make_backend()
    tfr = _make_tfr(result=result, model=model, actual_device="cpu", backend=backend)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", side_effect=lambda p: p),
        patch("local_transcriber.cli.detect_device", return_value="cuda"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu", backend, "/models/medium")),
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
    backend = _make_backend()
    tfr_empty = _make_tfr(result=result_empty, model=model, backend=backend)
    tfr_ok = _make_tfr(result=result_ok, model=model, backend=backend)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", side_effect=lambda p: p),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu", backend, "/models/medium")),
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
    backend = _make_backend()
    result = _make_result(device_used="cpu")
    tfr_fallback = _make_tfr(result=result, model=model_cpu, actual_device="cpu", backend=backend)
    tfr_ok = _make_tfr(result=result, model=model_cpu, actual_device="cpu", backend=backend)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", side_effect=lambda p: p),
        patch("local_transcriber.cli.detect_device", return_value="cuda"),
        patch("local_transcriber.cli.load_model", return_value=(model_gpu, "cuda", backend, "/models/medium")),
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
    backend = _make_backend()
    tfr = _make_tfr(result=result, model=model, backend=backend)
    mock_load_model = MagicMock(return_value=(model, "cpu", backend, "/models/medium"))

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", side_effect=lambda p: p),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.load_model", mock_load_model),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(a), str(b)])

    assert out.exit_code == 0
    mock_load_model.assert_called_once()


# === _format_device_info tests ===


def test_format_device_info_openvino_gpu():
    with patch("local_transcriber.cli.get_intel_gpu_name", return_value="Intel(R) Arc(TM) 140T GPU"):
        assert _format_device_info("openvino-gpu") == "OpenVINO (Intel(R) Arc(TM) 140T GPU)"


def test_format_device_info_openvino_gpu_no_name():
    """get_intel_gpu_name вернул None → fallback на 'Intel GPU'."""
    with patch("local_transcriber.cli.get_intel_gpu_name", return_value=None):
        assert _format_device_info("openvino-gpu") == "OpenVINO (Intel GPU)"


def test_format_device_info_openvino_cpu():
    assert _format_device_info("openvino-cpu") == "OpenVINO (CPU)"


def test_format_device_info_openvino_legacy():
    """Обратная совместимость: 'openvino' → OpenVINO (CPU)."""
    assert _format_device_info("openvino") == "OpenVINO (CPU)"


def test_format_device_info_cpu():
    assert _format_device_info("cpu") == "CPU"


def test_format_device_info_cuda():
    with patch("local_transcriber.cli.get_gpu_name", return_value="RTX 4090"):
        assert _format_device_info("cuda") == "CUDA (RTX 4090)"


# === CLI with --device openvino-gpu ===


def test_cli_openvino_gpu_happy_path(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    result = _make_result(device_used="openvino-gpu")

    patches = _single_patches(result=result, tmp_file=audio, actual_device="openvino-gpu")
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        with patch("local_transcriber.cli.get_intel_gpu_name", return_value="Intel Arc 140T"):
            out = runner.invoke(app, [str(audio), "--device", "openvino-gpu"])

    assert out.exit_code == 0


def test_cli_openvino_alias_resolves_to_gpu(tmp_path):
    """--device openvino резолвится через detect_device в openvino-gpu."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    result = _make_result(device_used="openvino-gpu")
    model = _make_model()
    backend = _make_backend()
    tfr = _make_tfr(result=result, model=model, actual_device="openvino-gpu", backend=backend)
    mock_load_model = MagicMock(return_value=(model, "openvino-gpu", backend, "/models/medium"))

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="openvino-gpu"),
        patch("local_transcriber.cli.load_model", mock_load_model),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript"),
        patch("local_transcriber.cli.get_intel_gpu_name", return_value="Intel Arc 140T"),
    ):
        out = runner.invoke(app, [str(audio), "--device", "openvino"])

    assert out.exit_code == 0
    # detect_device("openvino") resolved to "openvino-gpu", load_model receives it
    assert mock_load_model.call_args[0][1] == "openvino-gpu"


# === --threads ===


def test_cli_threads_passed_to_load_model(tmp_path):
    """--threads передаётся в load_model как cpu_threads."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    result = _make_result()
    model = _make_model()
    backend = _make_backend()
    tfr = _make_tfr(result=result, model=model, backend=backend)
    mock_load_model = MagicMock(return_value=(model, "cpu", backend, "/models/medium"))

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.load_model", mock_load_model),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(audio), "--threads", "8"])

    assert out.exit_code == 0
    assert mock_load_model.call_args.kwargs["cpu_threads"] == 8


def test_cli_threads_default_zero(tmp_path):
    """Без --threads load_model получает cpu_threads=0."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    result = _make_result()
    model = _make_model()
    backend = _make_backend()
    tfr = _make_tfr(result=result, model=model, backend=backend)
    mock_load_model = MagicMock(return_value=(model, "cpu", backend, "/models/medium"))

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.load_model", mock_load_model),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(audio)])

    assert out.exit_code == 0
    assert mock_load_model.call_args.kwargs["cpu_threads"] == 0


def test_cli_threads_negative_rejected(tmp_path):
    """--threads с отрицательным значением отклоняется typer (min=0)."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    out = runner.invoke(app, [str(audio), "--threads", "-1"])
    assert out.exit_code != 0


# === Regression: Whisper headers not changed ===


def test_cli_whisper_transcript_header_has_language_forced(tmp_path):
    """Регрессия: для --device cpu --language ru шапка содержит 'ru (forced)'."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")

    captured = {}
    def capture(content, path):
        captured["content"] = content

    result = _make_result(language="ru", device_used="cpu")
    model = _make_model()
    backend = _make_backend()
    backend.effective_model_name = None  # Whisper не ставит это
    tfr = _make_tfr(result=result, model=model, actual_device="cpu", backend=backend)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu", backend, "/models/medium")),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript", side_effect=capture),
    ):
        out = runner.invoke(app, [str(audio), "--language", "ru"])

    assert out.exit_code == 0
    assert "**Язык**: ru (forced)" in captured["content"]
    assert "**Модель**: medium" in captured["content"]


def test_cli_whisper_auto_language_header_has_detected(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    captured = {}
    def capture(content, path):
        captured["content"] = content

    result = _make_result(language="en", device_used="cpu")
    model = _make_model()
    backend = _make_backend()
    backend.effective_model_name = None
    tfr = _make_tfr(result=result, model=model, actual_device="cpu", backend=backend)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu", backend, "/models/medium")),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript", side_effect=capture),
    ):
        out = runner.invoke(app, [str(audio), "--language", "auto"])

    assert out.exit_code == 0
    assert "**Язык**: en (detected)" in captured["content"]


# === Parakeet-specific CLI ===


def _make_parakeet_backend(effective_name="parakeet-tdt-0.6b-v3"):
    backend = MagicMock(name="ParakeetBackend")
    backend.effective_model_name = effective_name
    backend.actual_compute_type = "int8"
    return backend


def test_cli_parakeet_transcript_header_multi_detected(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    captured = {}
    def capture(content, path):
        captured["content"] = content

    result = _make_result(language="multi", device_used="parakeet-cpu")
    model = _make_model()
    backend = _make_parakeet_backend()
    tfr = _make_tfr(result=result, model=model, actual_device="parakeet-cpu", backend=backend)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="parakeet-cpu"),
        patch("local_transcriber.cli.load_model", return_value=(model, "parakeet-cpu", backend, "/models/parakeet")),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript", side_effect=capture),
    ):
        out = runner.invoke(app, [str(audio), "--device", "parakeet"])

    assert out.exit_code == 0
    assert "**Язык**: multi (detected)" in captured["content"]
    assert "**Модель**: parakeet-tdt-0.6b-v3" in captured["content"]


def test_cli_parakeet_effective_model_overrides_config_model(tmp_path):
    """Когда config имеет model=medium, но user'ская команда — --device parakeet --model parakeet,
    шапка транскрипта должна показать parakeet-tdt-0.6b-v3."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    captured = {}
    def capture(content, path):
        captured["content"] = content

    result = _make_result(language="multi", device_used="parakeet-cpu")
    model = _make_model()
    backend = _make_parakeet_backend()
    tfr = _make_tfr(result=result, model=model, actual_device="parakeet-cpu", backend=backend)

    with (
        patch("local_transcriber.cli.load_config", return_value={"model": "medium"}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="parakeet-cpu"),
        patch("local_transcriber.cli.load_model", return_value=(model, "parakeet-cpu", backend, "/models/parakeet")),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript", side_effect=capture),
    ):
        out = runner.invoke(app, [str(audio), "--device", "parakeet", "--model", "parakeet"])

    assert out.exit_code == 0
    assert "**Модель**: parakeet-tdt-0.6b-v3" in captured["content"]


def test_cli_parakeet_warns_on_explicit_cli_language(tmp_path):
    """Явный --language с --device parakeet → warning в stderr."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")

    result = _make_result(language="multi", device_used="parakeet-cpu")
    model = _make_model()
    backend = _make_parakeet_backend()
    tfr = _make_tfr(result=result, model=model, actual_device="parakeet-cpu", backend=backend)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="parakeet-cpu"),
        patch("local_transcriber.cli.load_model", return_value=(model, "parakeet-cpu", backend, "/models/parakeet")),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(audio), "--device", "parakeet", "--language", "en"])

    assert out.exit_code == 0
    assert "Parakeet игнорирует" in out.stderr or "Parakeet игнорирует" in out.output


def test_cli_parakeet_does_not_warn_on_config_language(tmp_path):
    """Language из конфига без явного CLI — warning НЕ срабатывает."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")

    result = _make_result(language="multi", device_used="parakeet-cpu")
    model = _make_model()
    backend = _make_parakeet_backend()
    tfr = _make_tfr(result=result, model=model, actual_device="parakeet-cpu", backend=backend)

    with (
        patch("local_transcriber.cli.load_config", return_value={"language": "ru"}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="parakeet-cpu"),
        patch("local_transcriber.cli.load_model", return_value=(model, "parakeet-cpu", backend, "/models/parakeet")),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(audio), "--device", "parakeet"])

    assert out.exit_code == 0
    assert "Parakeet игнорирует" not in out.stderr and "Parakeet игнорирует" not in out.output


def test_cli_parakeet_device_info_shows_onnx():
    assert _format_device_info("parakeet-cpu") == "Parakeet (CPU via ONNX Runtime)"


def test_cli_parakeet_config_model_mismatch_without_override_prints_friendly_error(tmp_path):
    """config: model=medium, запуск --device parakeet БЕЗ --model → ValueError из
    ensure_model_available пробрасывается; CLI должен напечатать текст ошибки
    без traceback (exit code 1)."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")

    def failing_load_model(*args, **kwargs):
        raise ValueError(
            "Parakeet поддерживает только модели: parakeet, parakeet-tdt-0.6b-v3. "
            "Получено: 'medium'. Передайте --model parakeet или уберите "
            "'model' из .transcriber.toml."
        )

    with (
        patch("local_transcriber.cli.load_config", return_value={"model": "medium"}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="parakeet-cpu"),
        patch("local_transcriber.cli.load_model", side_effect=failing_load_model),
    ):
        out = runner.invoke(app, [str(audio), "--device", "parakeet"])

    assert out.exit_code == 1
    combined = (out.stderr or "") + (out.output or "")
    assert "Parakeet поддерживает только" in combined
    assert "--model parakeet" in combined
    assert "Traceback" not in combined


def test_cli_parakeet_end_to_end_writes_file(tmp_path):
    """Проверка end-to-end пайплайна с mock backend: файл записан и содержит ожидаемые поля."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")

    segments = [Segment(start=0.0, end=2.0, text="Привет мир")]
    result = TranscribeResult(
        segments=segments, language="multi", language_probability=0.0,
        duration=2.0, device_used="parakeet-cpu",
    )
    model = _make_model()
    backend = _make_parakeet_backend()
    tfr = _make_tfr(result=result, model=model, actual_device="parakeet-cpu", backend=backend)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="parakeet-cpu"),
        patch("local_transcriber.cli.load_model", return_value=(model, "parakeet-cpu", backend, "/models/parakeet")),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
    ):
        out = runner.invoke(app, [str(audio), "--device", "parakeet"])

    assert out.exit_code == 0
    output_md = tmp_path / "test-transcript.md"
    assert output_md.exists()
    content = output_md.read_text(encoding="utf-8")
    assert "# Транскрипт: test.mp3" in content
    assert "**Модель**: parakeet-tdt-0.6b-v3" in content
    assert "**Язык**: multi (detected)" in content
    assert "Привет мир" in content
