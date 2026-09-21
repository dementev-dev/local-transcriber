import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console
from typer.testing import CliRunner

from local_transcriber.cli import _format_device_info, _format_language_mode, app
from local_transcriber.formatter import (
    LANGUAGE_DETECTED,
    LANGUAGE_FORCED,
    LANGUAGE_FROM_MODEL,
    LANGUAGE_UNKNOWN,
)
from local_transcriber.transcriber import (
    Segment,
    TranscribeFileResult,
    TranscribeResult,
)
from local_transcriber.types import (
    UNKNOWN_LANGUAGE,
    DiarizationRun,
    SpeakerInterval,
    Word,
)

runner = CliRunner()


def _make_result(segments=None, language="ru", device_used="cpu", duration=60.0):
    return TranscribeResult(
        segments=[Segment(start=0.0, end=2.0, text="Hello")]
        if segments is None
        else segments,
        language=language,
        language_probability=0.95,
        duration=duration,
        device_used=device_used,
    )


def _make_model():
    return MagicMock(name="WhisperModel")


def _make_backend():
    return MagicMock(name="Backend")


def _make_tfr(
    result=None,
    model=None,
    actual_device="cpu",
    backend=None,
    model_path="/models/medium",
):
    if result is None:
        result = _make_result()
    if model is None:
        model = _make_model()
    if backend is None:
        backend = _make_backend()
    return TranscribeFileResult(
        result=result,
        model=model,
        actual_device=actual_device,
        backend=backend,
        model_path=model_path,
    )


@pytest.mark.parametrize(
    ("requested_language", "language", "probability", "expected"),
    [
        ("ru", "ru", 1.0, LANGUAGE_FORCED),
        ("auto", "ru", 0.95, LANGUAGE_DETECTED),
        ("auto", "ru", 0.0, LANGUAGE_FROM_MODEL),
        ("auto", UNKNOWN_LANGUAGE, 0.0, LANGUAGE_UNKNOWN),
    ],
)
def test_format_language_mode(requested_language, language, probability, expected):
    result = _make_result(language=language)
    result.language_probability = probability

    assert _format_language_mode(requested_language, result) == expected


def _single_patches(result=None, tmp_file=None, actual_device="cpu"):
    """Patches for a standard single-file CLI happy path."""
    if result is None:
        result = _make_result(device_used=actual_device)
    model = _make_model()
    backend = _make_backend()
    tfr = _make_tfr(
        result=result, model=model, actual_device=actual_device, backend=backend
    )
    return [
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=tmp_file),
        patch("local_transcriber.cli.detect_device", return_value=actual_device),
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, actual_device, backend, "/models/medium"),
        ),
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


@pytest.mark.parametrize("file_count", [1, 2])
def test_cli_renders_runtime_warning_without_python_details(tmp_path, file_count):
    files = [tmp_path / f"test-{index}.mp3" for index in range(file_count)]
    for file in files:
        file.write_bytes(b"fake")

    result = _make_result()
    model = _make_model()
    backend = _make_backend()
    tfr = _make_tfr(result=result, model=model, backend=backend)
    warning_message = (
        "Тестовое [bold]предупреждение[/bold] с длинным текстом, который "
        "должен остаться одной логической строкой без служебных подробностей Python"
    )

    def transcribe_with_warning(**_kwargs):
        warnings.warn(warning_message, stacklevel=2)
        return tfr

    original_showwarning = warnings.showwarning
    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", side_effect=lambda p: p),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cpu", backend, "/models/medium"),
        ),
        patch(
            "local_transcriber.cli._transcribe_file",
            side_effect=transcribe_with_warning,
        ),
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(file) for file in files])

    warning_lines = [line for line in out.output.splitlines() if "Тестовое" in line]
    assert warning_lines == [f"Внимание: {warning_message}"]
    assert "UserWarning" not in out.output
    assert "warnings.warn" not in out.output
    assert warnings.showwarning is original_showwarning


def test_cli_default_options_passed_to_transcribe(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    result = _make_result(device_used="onnx")
    model = _make_model()
    backend = _make_backend()
    tfr = _make_tfr(result=result, model=model, actual_device="onnx", backend=backend)
    mock_transcribe_file = MagicMock(return_value=tfr)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="onnx"),
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "onnx", backend, "/models/gigaam-v3-e2e-rnnt"),
        ),
        patch("local_transcriber.cli._transcribe_file", mock_transcribe_file),
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(audio)])

    call_kwargs = mock_transcribe_file.call_args[1]
    assert call_kwargs["model_name"] == "gigaam-v3-e2e-rnnt"
    assert call_kwargs["compute_type"] == "int8"
    assert call_kwargs["language"] == "ru"
    assert call_kwargs["on_segment"] is None  # verbose=False
    assert "Модель: gigaam-v3-e2e-rnnt" in out.output
    assert "Устройство: onnx" in out.output


def test_cli_identifies_onnx_backend_in_transcript_header(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    result = _make_result(device_used="onnx")
    model = _make_model()
    backend = _make_backend()
    tfr = _make_tfr(result=result, model=model, actual_device="onnx", backend=backend)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.detect_device", return_value="onnx"),
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "onnx", backend, "/models/gigaam-v3-e2e-rnnt"),
        ),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
    ):
        out = runner.invoke(app, [str(audio)])

    content = (tmp_path / "test-transcript.md").read_text(encoding="utf-8")
    assert out.exit_code == 0
    assert "- **Устройство**: ONNX (CPU)" in content


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
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cuda", backend, "/models/small"),
        ),
        patch("local_transcriber.cli._transcribe_file", mock_transcribe_file),
        patch("local_transcriber.cli.write_transcript"),
        patch("local_transcriber.cli.get_gpu_name", return_value="RTX 3060"),
    ):
        runner.invoke(
            app,
            [
                str(audio),
                "--model",
                "small",
                "--language",
                "ru",
                "--device",
                "cuda",
                "--compute-type",
                "float16",
            ],
        )

    call_kwargs = mock_transcribe_file.call_args[1]
    assert call_kwargs["model_name"] == "small"
    assert call_kwargs["language"] == "ru"
    assert call_kwargs["compute_type"] == "float16"


def test_cli_speakers_enables_diarization_and_writes_speaker_markdown(tmp_path):
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"fake")
    result = _make_result(
        segments=[Segment(0.0, 1.3, "Первый. Второй. Неясно.")],
        duration=10.0,
    )
    result.words = [
        Word(0.0, 0.5, "Первый."),
        Word(0.5, 1.0, "Второй."),
        Word(1.1, 1.3, "Неясно."),
    ]
    model = _make_model()
    backend = _make_backend()
    backend.word_timestamps_available = True
    tfr = _make_tfr(result=result, model=model, backend=backend)
    diarizer = MagicMock()
    diarizer.process.return_value = DiarizationRun(
        intervals=[
            SpeakerInterval(0.0, 0.5, 10),
            SpeakerInterval(0.5, 1.0, 20),
        ],
        elapsed_seconds=0.2,
    )
    write = MagicMock()

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cpu", backend, "/models/medium"),
        ),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch(
            "local_transcriber.cli.load_speaker_diarizer",
            return_value=diarizer,
        ) as load_diarizer,
        patch("local_transcriber.cli.write_transcript", write),
    ):
        out = runner.invoke(
            app,
            [str(audio), "--speakers", "2", "--threads", "3"],
        )

    assert out.exit_code == 0
    load_diarizer.assert_called_once()
    assert load_diarizer.call_args.kwargs["speakers"] == 2
    assert load_diarizer.call_args.kwargs["threads"] == 3
    diarizer.process.assert_called_once()
    assert "Speaker 1: Первый." in write.call_args.args[0]
    assert "Speaker 2: Второй." in write.call_args.args[0]
    assert "Speaker ?: Неясно." in write.call_args.args[0]
    assert "1 слов без назначенного говорящего" in out.output
    assert "малый кластер Speaker 1: 0.5 с" in out.output


def test_cli_diarization_error_writes_plain_transcript_and_exits_nonzero(tmp_path):
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"fake")
    result = _make_result(
        segments=[Segment(0.0, 1.0, "Полезный текст.")],
        duration=10.0,
    )
    result.words = [Word(0.0, 1.0, "Полезный текст.")]
    model = _make_model()
    backend = _make_backend()
    backend.word_timestamps_available = True
    tfr = _make_tfr(result=result, model=model, backend=backend)
    diarizer = MagicMock()
    diarizer.process.side_effect = RuntimeError("boom")
    write = MagicMock()

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cpu", backend, "/models/medium"),
        ),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch(
            "local_transcriber.cli.load_speaker_diarizer",
            return_value=diarizer,
        ),
        patch("local_transcriber.cli.write_transcript", write),
    ):
        out = runner.invoke(app, [str(audio), "--diarize"])

    assert out.exit_code == 1
    assert write.call_count == 1
    assert "Полезный текст." in write.call_args.args[0]
    assert "Диаризация завершилась с ошибкой: boom" in write.call_args.args[0]


def test_cli_verbose_reports_diarization_counts_and_duration(tmp_path):
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"fake")
    result = _make_result(
        segments=[Segment(0.0, 1.0, "Раз два")],
        duration=10.0,
    )
    result.words = [Word(0.0, 0.5, "Раз"), Word(0.5, 1.0, "два")]
    model = _make_model()
    backend = _make_backend()
    backend.word_timestamps_available = True
    tfr = _make_tfr(result=result, model=model, backend=backend)
    diarizer = MagicMock()
    diarizer.process.return_value = DiarizationRun(
        intervals=[
            SpeakerInterval(0.0, 0.5, 1),
            SpeakerInterval(0.5, 1.0, 2),
        ],
        elapsed_seconds=0.2,
    )

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cpu", backend, "/models/medium"),
        ),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch(
            "local_transcriber.cli.load_speaker_diarizer",
            return_value=diarizer,
        ),
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(audio), "--diarize", "--verbose"])

    assert out.exit_code == 0
    assert "2 кластеров, 2 интервалов" in out.output
    assert "0.2 с" in out.output


def test_cli_empty_asr_skips_diarizer_and_reports_it(tmp_path):
    audio = tmp_path / "silence.wav"
    audio.write_bytes(b"fake")
    result = _make_result(segments=[])
    model = _make_model()
    backend = _make_backend()
    backend.word_timestamps_available = True
    tfr = _make_tfr(result=result, model=model, backend=backend)
    diarizer = MagicMock()

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cpu", backend, "/models/medium"),
        ),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch(
            "local_transcriber.cli.load_speaker_diarizer",
            return_value=diarizer,
        ),
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(audio), "--diarize"])

    assert out.exit_code == 0
    diarizer.process.assert_not_called()
    assert "диаризация не запускалась" in out.output


def test_cli_diarizer_preflight_failure_does_not_start_asr_or_write(tmp_path):
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"fake")
    model = _make_model()
    backend = _make_backend()
    backend.word_timestamps_available = True
    transcribe_file = MagicMock()
    write = MagicMock()

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cpu", backend, "/models/medium"),
        ),
        patch("local_transcriber.cli._transcribe_file", transcribe_file),
        patch(
            "local_transcriber.cli.load_speaker_diarizer",
            side_effect=RuntimeError("модель повреждена"),
        ),
        patch("local_transcriber.cli.write_transcript", write),
    ):
        out = runner.invoke(app, [str(audio), "--diarize"])

    assert out.exit_code == 1
    transcribe_file.assert_not_called()
    write.assert_not_called()


@pytest.mark.parametrize(
    ("intervals", "warning"),
    [
        ([SpeakerInterval(0.0, 1.0, 1)], "только один голосовой кластер"),
        ([], "не нашёл интервалов"),
    ],
)
def test_cli_unsuccessful_diarization_shape_writes_plain_text_and_exits_nonzero(
    tmp_path, intervals, warning
):
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"fake")
    result = _make_result(
        segments=[Segment(0.0, 1.0, "Раз два")],
        duration=10.0,
    )
    result.words = [Word(0.0, 0.5, "Раз"), Word(0.5, 1.0, "два")]
    model = _make_model()
    backend = _make_backend()
    backend.word_timestamps_available = True
    tfr = _make_tfr(result=result, model=model, backend=backend)
    diarizer = MagicMock()
    diarizer.process.return_value = DiarizationRun(intervals, elapsed_seconds=0.1)
    write = MagicMock()

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cpu", backend, "/models/medium"),
        ),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch(
            "local_transcriber.cli.load_speaker_diarizer",
            return_value=diarizer,
        ),
        patch("local_transcriber.cli.write_transcript", write),
    ):
        out = runner.invoke(app, [str(audio), "--diarize"])

    assert out.exit_code == 1
    content = write.call_args.args[0]
    assert warning in content
    assert "[00:00.00 - 00:01.00] Раз два" in content


def test_cli_rejects_nonpositive_speaker_count(tmp_path):
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"fake")

    out = runner.invoke(app, [str(audio), "--speakers", "0"])

    assert out.exit_code == 2


def test_cli_rejects_no_diarize_with_speakers_before_model_load(tmp_path):
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"fake")
    load_model = MagicMock()

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.load_model", load_model),
    ):
        out = runner.invoke(
            app,
            [str(audio), "--no-diarize", "--speakers", "2"],
        )

    assert out.exit_code == 2
    assert "--no-diarize и --speakers несовместимы" in out.output
    load_model.assert_not_called()


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
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cpu", backend, "/models/medium"),
        ),
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
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cpu", backend, "/models/medium"),
        ),
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
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cpu", backend, "/models/medium"),
        ),
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
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cpu", backend, "/models/medium"),
        ),
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
    mock_load_model = MagicMock(
        return_value=(model, "cpu", backend, "/models/large-v3")
    )

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
    """При проблеме драйвера Windows не предлагает установку Toolkit."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    model = _make_model()
    backend = _make_backend()

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cuda"),
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cuda", backend, "/models/medium"),
        ),
        patch(
            "local_transcriber.cli._transcribe_file",
            side_effect=RuntimeError("CUDA error: no device"),
        ),
        patch("local_transcriber.cli.sys") as mock_sys,
    ):
        mock_sys.platform = "win32"
        out = runner.invoke(app, [str(audio), "--device", "cuda"])

    assert out.exit_code == 1
    assert "драйвер" in out.output
    assert "CUDA error: no device" in out.output
    assert "winget install" not in out.output
    assert "uv sync --extra" not in out.output


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
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cuda", backend, "/models/medium"),
        ),
        patch(
            "local_transcriber.cli._transcribe_file",
            side_effect=RuntimeError("CUDA error: no device"),
        ),
        patch("local_transcriber.cli.sys") as mock_sys,
    ):
        mock_sys.platform = "linux"
        out = runner.invoke(app, [str(audio), "--device", "cuda"])

    assert out.exit_code == 1
    assert "choco install cuda" not in out.output


@pytest.mark.parametrize("from_config", [False, True])
@pytest.mark.parametrize("phase", ["load", "single", "batch"])
@pytest.mark.parametrize(
    ("message", "hint", "install_hint"),
    [
        ("Library cublas64_12.dll is not found or cannot be loaded", "Не найдены", True),
        ("libcublas.so.12: cannot open shared object file", "Не найдены", True),
        ("CUDA error: no kernel image is available", "GPU несовместим", False),
        ("CUDA driver version is insufficient for CUDA runtime version", "драйвер", False),
        ("CUDA out of memory", "Недостаточно памяти", False),
    ],
)
def test_explicit_cuda_errors_preserve_cause_without_fallback(
    tmp_path, monkeypatch, from_config, phase, message, hint, install_hint
):
    """CLI/TOML сохраняют CUDA при загрузке, обработке одного файла и батча."""
    monkeypatch.chdir(tmp_path)
    files = [tmp_path / "one.wav"]
    if phase == "batch":
        files.append(tmp_path / "two.wav")
    for file in files:
        file.write_bytes(b"audio")
    if from_config:
        (tmp_path / ".transcriber.toml").write_text('device = "cuda"\n')
    args = [str(file) for file in files]
    if not from_config:
        args += ["--device", "cuda"]
    backend = _make_backend()
    error = RuntimeError(message)
    if phase == "load":
        backend.create_model.side_effect = error
    else:
        backend.transcribe.side_effect = error

    with patch("local_transcriber.transcriber.get_backend", return_value=backend) as get_backend:
        out = runner.invoke(app, args)

    assert out.exit_code == 1
    assert message in out.output
    assert hint in out.output
    assert ("uv sync --extra cuda" in out.output) is install_hint
    assert "winget install" not in out.output
    assert "Переключение на CPU" not in out.output
    assert all(call.args[0] == "cuda" for call in get_backend.call_args_list)


def test_default_cli_stays_on_onnx_with_nvidia_driver(tmp_path, monkeypatch):
    """Обычный запуск с nvidia-smi использует прежнюю CPU-модель ONNX."""
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"audio")
    backend = _make_backend()
    backend.actual_ov_device = None
    backend.actual_compute_type = "int8"
    backend.transcribe.return_value = _make_result(device_used="onnx")
    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
        patch("local_transcriber.transcriber.get_backend", return_value=backend) as get_backend,
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(audio)])

    assert out.exit_code == 0, out.output
    get_backend.assert_called_once_with("onnx", compute_type_explicit=False)
    assert backend.ensure_model_available.call_args.args[:2] == ("gigaam-v3-e2e-rnnt", "int8")


@pytest.mark.parametrize("from_config", [False, True])
@pytest.mark.parametrize("phase", ["load", "single", "batch"])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("compute_type", ["float16", "int8"])
def test_unsupported_compute_type_hint_requires_cuda_context(
    tmp_path, monkeypatch, from_config, phase, device, compute_type
):
    """Одинаковая ошибка CTranslate2 получает CUDA-подсказку только на CUDA."""
    monkeypatch.chdir(tmp_path)
    files = [tmp_path / "one.wav"]
    if phase == "batch":
        files.append(tmp_path / "two.wav")
    for file in files:
        file.write_bytes(b"audio")
    args = [str(file) for file in files]
    if from_config:
        (tmp_path / ".transcriber.toml").write_text(
            f'device = "{device}"\ncompute_type = "{compute_type}"\n'
        )
    else:
        args += ["--device", device, "--compute-type", compute_type]
    message = (
        f"Requested {compute_type} compute type, but the target device or backend "
        f"do not support efficient {compute_type} computation."
    )
    backend = _make_backend()
    error = ValueError(message)
    if phase == "load":
        backend.create_model.side_effect = error
    else:
        backend.transcribe.side_effect = error

    with patch("local_transcriber.transcriber.get_backend", return_value=backend):
        out = runner.invoke(app, args)

    assert out.exit_code == 1
    assert message in " ".join(out.output.split())
    assert ("Тип вычислений несовместим" in out.output) is (device == "cuda")
    assert "uv sync --extra cuda" not in out.output
    assert "Переключение на CPU" not in out.output


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
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cuda", backend, "/models/medium"),
        ),
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
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cuda", backend, "/models/medium"),
        ),
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
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cpu", backend, "/models/medium"),
        ),
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
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cpu", backend, "/models/medium"),
        ),
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
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cpu", backend, "/models/medium"),
        ),
        patch(
            "local_transcriber.cli._transcribe_file",
            side_effect=RuntimeError("unexpected boom"),
        ),
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
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cpu", backend, "/models/medium"),
        ),
        patch(
            "local_transcriber.cli._transcribe_file",
            side_effect=RuntimeError("unexpected boom"),
        ),
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
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cpu", backend, "/models/medium"),
        ),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(a), str(b)])

    assert out.exit_code == 0
    assert "2 обработано" in out.output


@pytest.mark.parametrize(
    ("config", "cli_args"),
    [({}, ["--diarize"]), ({"diarize": True}, [])],
)
def test_cli_batch_reuses_one_diarizer_for_all_nonempty_files(
    tmp_path, config, cli_args
):
    first = tmp_path / "first.mp3"
    second = tmp_path / "second.mp3"
    first.write_bytes(b"fake")
    second.write_bytes(b"fake")
    result = _make_result(
        segments=[Segment(0.0, 1.0, "Раз два")],
        duration=10.0,
    )
    result.words = [Word(0.0, 0.5, "Раз"), Word(0.5, 1.0, "два")]
    model = _make_model()
    backend = _make_backend()
    backend.word_timestamps_available = True
    tfr = _make_tfr(result=result, model=model, backend=backend)
    diarizer = MagicMock()
    diarizer.process.return_value = DiarizationRun(
        intervals=[
            SpeakerInterval(0.0, 0.5, 1),
            SpeakerInterval(0.5, 1.0, 2),
        ],
        elapsed_seconds=0.1,
    )

    with (
        patch("local_transcriber.cli.load_config", return_value=config),
        patch(
            "local_transcriber.cli.validate_input_file",
            side_effect=lambda path: path,
        ),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cpu", backend, "/models/medium"),
        ),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch(
            "local_transcriber.cli.load_speaker_diarizer",
            return_value=diarizer,
        ) as load_diarizer,
        patch("local_transcriber.cli.write_transcript") as write,
    ):
        out = runner.invoke(app, [str(first), str(second), *cli_args])

    assert out.exit_code == 0
    load_diarizer.assert_called_once()
    assert [call.args[0] for call in diarizer.process.call_args_list] == [
        first,
        second,
    ]
    assert write.call_count == 2


def test_cli_batch_continues_after_diarization_error_and_exits_nonzero(tmp_path):
    first = tmp_path / "first.mp3"
    second = tmp_path / "second.mp3"
    first.write_bytes(b"fake")
    second.write_bytes(b"fake")
    result = _make_result(
        segments=[Segment(0.0, 1.0, "Раз два")],
        duration=10.0,
    )
    result.words = [Word(0.0, 0.5, "Раз"), Word(0.5, 1.0, "два")]
    model = _make_model()
    backend = _make_backend()
    backend.word_timestamps_available = True
    tfr = _make_tfr(result=result, model=model, backend=backend)
    diarizer = MagicMock()
    diarizer.process.side_effect = [
        RuntimeError("boom"),
        DiarizationRun(
            [
                SpeakerInterval(0.0, 0.5, 1),
                SpeakerInterval(0.5, 1.0, 2),
            ],
            elapsed_seconds=0.1,
        ),
    ]

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch(
            "local_transcriber.cli.validate_input_file",
            side_effect=lambda path: path,
        ),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cpu", backend, "/models/medium"),
        ),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch(
            "local_transcriber.cli.load_speaker_diarizer",
            return_value=diarizer,
        ),
        patch("local_transcriber.cli.write_transcript") as write,
    ):
        out = runner.invoke(app, [str(first), str(second), "--diarize"])

    assert out.exit_code == 1
    assert write.call_count == 2
    assert "Диаризация завершилась с ошибкой: boom" in write.call_args_list[0].args[0]
    assert "Speaker 1" in write.call_args_list[1].args[0]
    assert "1 с деградацией" in out.output


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
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cpu", backend, "/models/medium"),
        ),
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
    mock_load_diarizer = MagicMock()

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", side_effect=lambda p: p),
        patch("local_transcriber.cli.load_model", mock_load_model),
        patch(
            "local_transcriber.cli.load_speaker_diarizer",
            mock_load_diarizer,
        ),
    ):
        out = runner.invoke(app, [str(a), str(b), "--diarize"])

    assert out.exit_code == 0
    mock_load_model.assert_not_called()
    mock_load_diarizer.assert_not_called()


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
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cpu", backend, "/models/medium"),
        ),
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
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cpu", backend, "/models/medium"),
        ),
        patch(
            "local_transcriber.cli._transcribe_file", side_effect=transcribe_side_effect
        ),
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
        patch(
            "local_transcriber.cli.validate_input_file",
            side_effect=validate_side_effect,
        ),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cpu", backend, "/models/medium"),
        ),
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


@pytest.mark.parametrize(
    ("config_value", "cli_args", "expected_calls"),
    [
        (True, [], 1),
        (False, [], 0),
        (False, ["--diarize"], 1),
        (True, ["--no-diarize"], 0),
    ],
)
def test_cli_resolves_diarization_priority(
    tmp_path, config_value, cli_args, expected_calls
):
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"fake")
    result = _make_result(segments=[])
    model = _make_model()
    backend = _make_backend()
    backend.word_timestamps_available = True
    tfr = _make_tfr(result=result, model=model, backend=backend)
    load_diarizer = MagicMock()

    with (
        patch(
            "local_transcriber.cli.load_config",
            return_value={"diarize": config_value},
        ),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cpu", backend, "/models/medium"),
        ),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.load_speaker_diarizer", load_diarizer),
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(audio), *cli_args])

    assert out.exit_code == 0
    assert load_diarizer.call_count == expected_calls


def test_cli_config_overrides_auto_device(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    model = _make_model()
    backend = _make_backend()
    result = _make_result(device_used="openvino-cpu")
    tfr = _make_tfr(
        result=result,
        model=model,
        actual_device="openvino-cpu",
        backend=backend,
    )
    mock_detect_device = MagicMock(return_value="openvino-cpu")
    mock_load_model = MagicMock(
        return_value=(model, "openvino-cpu", backend, "/models/medium")
    )

    with (
        patch(
            "local_transcriber.cli.load_config",
            return_value={
                "device": "openvino-cpu",
                "model": "medium",
                "compute_type": "int8",
            },
        ),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", mock_detect_device),
        patch("local_transcriber.cli.load_model", mock_load_model),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(audio)])

    assert out.exit_code == 0
    assert mock_detect_device.call_args_list[0].args == ("openvino-cpu",)
    assert mock_load_model.call_args.args[:3] == ("medium", "openvino-cpu", "int8")


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
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cpu", backend, "/models/medium"),
        ),
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
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cpu", backend, "/models/medium"),
        ),
        patch(
            "local_transcriber.cli._transcribe_file", side_effect=[tfr_empty, tfr_ok]
        ),
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
    tfr_fallback = _make_tfr(
        result=result, model=model_cpu, actual_device="cpu", backend=backend
    )
    tfr_ok = _make_tfr(
        result=result, model=model_cpu, actual_device="cpu", backend=backend
    )

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", side_effect=lambda p: p),
        patch("local_transcriber.cli.detect_device", return_value="cuda"),
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model_gpu, "cuda", backend, "/models/medium"),
        ),
        patch(
            "local_transcriber.cli._transcribe_file", side_effect=[tfr_fallback, tfr_ok]
        ),
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
    with patch(
        "local_transcriber.cli.get_intel_gpu_name",
        return_value="Intel(R) Arc(TM) 140T GPU",
    ):
        assert (
            _format_device_info("openvino-gpu")
            == "OpenVINO (Intel(R) Arc(TM) 140T GPU)"
        )


def test_format_device_info_openvino_gpu_no_name():
    """get_intel_gpu_name вернул None → fallback на 'Intel GPU'."""
    with patch("local_transcriber.cli.get_intel_gpu_name", return_value=None):
        assert _format_device_info("openvino-gpu") == "OpenVINO (Intel GPU)"


def test_format_device_info_openvino_cpu():
    assert _format_device_info("openvino-cpu") == "OpenVINO (CPU)"


def test_format_device_info_openvino_legacy():
    """Обратная совместимость: 'openvino' → OpenVINO (CPU)."""
    assert _format_device_info("openvino") == "OpenVINO (CPU)"


def test_format_device_info_distinguishes_onnx_from_faster_whisper_cpu():
    assert _format_device_info("onnx") == "ONNX (CPU)"
    assert _format_device_info("cpu") == "CPU"


def test_format_device_info_cuda():
    with patch("local_transcriber.cli.get_gpu_name", return_value="RTX 4090"):
        assert _format_device_info("cuda") == "CUDA (RTX 4090)"


# === CLI with --device openvino-gpu ===


def test_cli_openvino_gpu_happy_path(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    result = _make_result(device_used="openvino-gpu")

    patches = _single_patches(
        result=result, tmp_file=audio, actual_device="openvino-gpu"
    )
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        with patch(
            "local_transcriber.cli.get_intel_gpu_name", return_value="Intel Arc 140T"
        ):
            out = runner.invoke(app, [str(audio), "--device", "openvino-gpu"])

    assert out.exit_code == 0


def test_cli_openvino_alias_resolves_to_gpu(tmp_path):
    """--device openvino резолвится через detect_device в openvino-gpu."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    result = _make_result(device_used="openvino-gpu")
    model = _make_model()
    backend = _make_backend()
    tfr = _make_tfr(
        result=result, model=model, actual_device="openvino-gpu", backend=backend
    )
    mock_load_model = MagicMock(
        return_value=(model, "openvino-gpu", backend, "/models/medium")
    )

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="openvino-gpu"),
        patch("local_transcriber.cli.load_model", mock_load_model),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript"),
        patch(
            "local_transcriber.cli.get_intel_gpu_name", return_value="Intel Arc 140T"
        ),
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


# === SendTo context menu flags ===


def test_cli_install_menu_success(tmp_path):
    cmd_path = tmp_path / "Transcribe.cmd"

    with (
        patch(
            "local_transcriber.cli.install_context_menu", return_value=cmd_path
        ) as mock_install,
        patch("local_transcriber.cli.load_config") as mock_load_config,
        patch("local_transcriber.cli.sys") as mock_sys,
    ):
        mock_sys.platform = "win32"
        out = runner.invoke(app, ["--install-menu"])

    assert out.exit_code == 0
    assert "Пункт меню установлен" in out.output
    assert cmd_path.name in out.output
    mock_install.assert_called_once_with()
    mock_load_config.assert_not_called()


def test_cli_uninstall_menu_success(tmp_path):
    cmd_path = tmp_path / "Transcribe.cmd"

    with (
        patch(
            "local_transcriber.cli.uninstall_context_menu", return_value=cmd_path
        ) as mock_uninstall,
        patch("local_transcriber.cli.load_config") as mock_load_config,
        patch("local_transcriber.cli.sys") as mock_sys,
    ):
        mock_sys.platform = "win32"
        out = runner.invoke(app, ["--uninstall-menu"])

    assert out.exit_code == 0
    assert "Пункт меню удалён" in out.output
    assert cmd_path.name in out.output
    mock_uninstall.assert_called_once_with()
    mock_load_config.assert_not_called()


def test_cli_uninstall_menu_missing_is_success():
    with (
        patch("local_transcriber.cli.uninstall_context_menu", return_value=None),
        patch("local_transcriber.cli.sys") as mock_sys,
    ):
        mock_sys.platform = "win32"
        out = runner.invoke(app, ["--uninstall-menu"])

    assert out.exit_code == 0
    assert "не был установлен" in out.output


def test_cli_menu_flags_are_mutually_exclusive():
    out = runner.invoke(app, ["--install-menu", "--uninstall-menu"])

    assert out.exit_code == 2
    assert "несовместимы" in out.output


def test_cli_menu_flag_with_file_is_rejected(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")

    out = runner.invoke(app, [str(audio), "--install-menu"])

    assert out.exit_code == 2
    assert "нельзя использовать вместе с файлами" in out.output


def test_cli_no_files_and_no_menu_flags_is_rejected():
    out = runner.invoke(app, [])

    assert out.exit_code == 2
    assert "Укажите хотя бы один файл" in out.output


def test_cli_menu_flags_available_only_on_windows():
    with patch("local_transcriber.cli.sys") as mock_sys:
        mock_sys.platform = "linux"
        out = runner.invoke(app, ["--install-menu"])

    assert out.exit_code == 1
    assert "только на Windows" in out.output


def test_cli_menu_runtime_error_has_no_verbose_hint():
    with (
        patch(
            "local_transcriber.cli.install_context_menu",
            side_effect=RuntimeError("нет APPDATA"),
        ),
        patch("local_transcriber.cli.sys") as mock_sys,
    ):
        mock_sys.platform = "win32"
        out = runner.invoke(app, ["--install-menu"])

    assert out.exit_code == 1
    assert "нет APPDATA" in out.output
    assert "--verbose" not in out.output


def test_cli_tail_gap_quality_warning_single(tmp_path):
    audio = tmp_path / "tail.mp3"
    audio.write_bytes(b"fake")
    result = _make_result(
        segments=[Segment(start=0.0, end=60.0, text="Фраза")],
        duration=600.0,
    )

    patches = _single_patches(result=result, tmp_file=audio)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patch("local_transcriber.cli.console", Console(stderr=True, width=1000)),
    ):
        out = runner.invoke(app, [str(audio)])

    assert out.exit_code == 0
    assert (
        "Внимание: транскрипт покрывает 01:00 из 10:00 — "
        "возможна потеря хвоста записи. Попробуйте другой --device."
    ) in out.output


def test_cli_repetition_quality_warning_single(tmp_path):
    audio = tmp_path / "repeat.mp3"
    audio.write_bytes(b"fake")
    result = _make_result(
        segments=[
            Segment(start=10.0, end=11.0, text="Повторяемая фраза"),
            Segment(start=11.0, end=12.0, text="повторяемая фраза"),
            Segment(start=12.0, end=13.0, text="повторяемая фраза"),
            Segment(start=13.0, end=14.0, text="повторяемая фраза"),
        ],
        duration=60.0,
    )

    patches = _single_patches(result=result, tmp_file=audio)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patch("local_transcriber.cli.console", Console(stderr=True, width=1000)),
    ):
        out = runner.invoke(app, [str(audio)])

    assert out.exit_code == 0
    assert (
        "Внимание: блоки повторов: [00:10.00 - 00:14.00] (4×) — "
        "возможны галлюцинации модели. Попробуйте другой --device."
    ) in out.output


def test_cli_quality_warning_batch_includes_file_name(tmp_path):
    a = tmp_path / "a.mp3"
    b = tmp_path / "b.mp3"
    a.write_bytes(b"fake")
    b.write_bytes(b"fake")

    result_warn = _make_result(
        segments=[Segment(start=0.0, end=60.0, text="Фраза")],
        duration=600.0,
    )
    result_ok = _make_result()
    model = _make_model()
    backend = _make_backend()
    tfr_warn = _make_tfr(result=result_warn, model=model, backend=backend)
    tfr_ok = _make_tfr(result=result_ok, model=model, backend=backend)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", side_effect=lambda p: p),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch(
            "local_transcriber.cli.load_model",
            return_value=(model, "cpu", backend, "/models/medium"),
        ),
        patch("local_transcriber.cli._transcribe_file", side_effect=[tfr_warn, tfr_ok]),
        patch("local_transcriber.cli.write_transcript"),
        patch("local_transcriber.cli.console", Console(stderr=True, width=1000)),
    ):
        out = runner.invoke(app, [str(a), str(b)])

    assert out.exit_code == 0
    assert (
        "  a.mp3: транскрипт покрывает 01:00 из 10:00 — возможна потеря хвоста записи"
    ) in out.output


def test_cli_repetition_quality_warning_truncates_after_three_blocks(tmp_path):
    audio = tmp_path / "repeat-many.mp3"
    audio.write_bytes(b"fake")

    def run(start, count, text):
        return [
            Segment(start=start + index, end=start + index + 1.0, text=text)
            for index in range(count)
        ]

    result = _make_result(
        segments=[
            *run(10.0, 6, "Первый повтор"),
            Segment(start=18.0, end=19.0, text="Разрыв один"),
            *run(20.0, 5, "Второй повтор"),
            Segment(start=28.0, end=29.0, text="Разрыв два"),
            *run(30.0, 4, "Третий повтор"),
            Segment(start=38.0, end=39.0, text="Разрыв три"),
            *run(40.0, 4, "Четвёртый повтор"),
        ],
        duration=90.0,
    )

    patches = _single_patches(result=result, tmp_file=audio)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patch("local_transcriber.cli.console", Console(stderr=True, width=1000)),
    ):
        out = runner.invoke(app, [str(audio)])

    assert out.exit_code == 0
    assert (
        "Внимание: блоки повторов: [00:10.00 - 00:16.00] (6×); "
        "[00:20.00 - 00:25.00] (5×); [00:30.00 - 00:34.00] (4×) "
        "(+ ещё 1) — возможны галлюцинации модели. Попробуйте другой --device."
    ) in out.output


def test_cli_default_result_has_no_quality_warnings(tmp_path):
    audio = tmp_path / "normal.mp3"
    audio.write_bytes(b"fake")

    patches = _single_patches(tmp_file=audio)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        out = runner.invoke(app, [str(audio)])

    assert out.exit_code == 0
    assert "потеря хвоста" not in out.output
    assert "галлюцинации" not in out.output
