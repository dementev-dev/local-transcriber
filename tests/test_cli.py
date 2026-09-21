import warnings
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pytest
from rich.console import Console
from typer.testing import CliRunner

from local_transcriber.cli import _format_language_mode, app
from local_transcriber.formatter import (
    LANGUAGE_DETECTED,
    LANGUAGE_FORCED,
    LANGUAGE_FROM_MODEL,
    LANGUAGE_UNKNOWN,
)
from local_transcriber.transcriber import Segment, TranscribeResult
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


def _make_backend(
    result=None,
    word_timestamps=True,
    actual_ov_device=None,
    runtime=None,
):
    """Fake adapter на internal seam Backend; module выполнения работает по-настоящему."""
    backend = MagicMock(name="Backend")
    backend.engine = "onnx-asr"
    backend.word_timestamps_available = word_timestamps
    backend.actual_compute_type = None
    backend.actual_ov_device = actual_ov_device
    backend.ensure_model_available.side_effect = (
        lambda model_name, compute_type, on_status=None: f"/models/{model_name}"
    )
    backend.transcribe.return_value = result if result is not None else _make_result()
    backend.runtime_info.return_value = runtime or {}
    return backend


@contextmanager
def _cli_run(result=None, tmp_file=None, backend=None, config=None, write=True):
    """Стандартный CLI-прогон: конфиг, валидация файла, fake adapter, запись.

    Отдаёт (backend, write_transcript); ``tmp_file=None`` пропускает файлы
    как есть, ``write=False`` оставляет настоящую запись транскрипта.
    """
    if backend is None:
        backend = _make_backend(result)
    validate = (
        {"return_value": tmp_file}
        if tmp_file is not None
        else {"side_effect": lambda p: p}
    )
    with ExitStack() as stack:
        stack.enter_context(
            patch("local_transcriber.cli.load_config", return_value=config or {})
        )
        stack.enter_context(patch("local_transcriber.cli.validate_input_file", **validate))
        stack.enter_context(
            patch("local_transcriber.transcriber.get_backend", return_value=backend)
        )
        write_transcript = (
            stack.enter_context(patch("local_transcriber.cli.write_transcript"))
            if write
            else None
        )
        yield backend, write_transcript


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


def test_cli_happy_path_exit_code_zero(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")

    with _cli_run(tmp_file=audio) as (backend, _write):
        out = runner.invoke(app, [str(audio)])

    assert out.exit_code == 0
    backend.ensure_model_available.assert_called_once_with(
        "gigaam-v3-e2e-rnnt", "int8", ANY
    )
    assert "Модель: gigaam-v3-e2e-rnnt  Устройство: onnx  Compute: int8" in out.output


def test_cli_verbose_prints_runtime_diagnostics(tmp_path):
    """--verbose показывает движок, потоки и версии runtime из данных module."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    backend = _make_backend(runtime={"onnxruntime": "1.28.0", "asr_providers": "CPUExecutionProvider"})

    with _cli_run(tmp_file=audio, backend=backend):
        out = runner.invoke(app, [str(audio), "--verbose", "--threads", "4"])

    assert out.exit_code == 0
    assert "Движок: onnx-asr  Потоки (запрошено): 4" in out.output
    assert "onnxruntime: 1.28.0" in out.output
    assert "asr_providers: CPUExecutionProvider" in out.output


def test_cli_without_verbose_hides_runtime_diagnostics(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    backend = _make_backend(runtime={"onnxruntime": "1.28.0"})

    with _cli_run(tmp_file=audio, backend=backend):
        out = runner.invoke(app, [str(audio)])

    assert "onnxruntime" not in out.output


@pytest.mark.parametrize("file_count", [1, 2])
def test_cli_renders_runtime_warning_without_python_details(tmp_path, file_count):
    files = [tmp_path / f"test-{index}.mp3" for index in range(file_count)]
    for file in files:
        file.write_bytes(b"fake")

    warning_message = (
        "Тестовое [bold]предупреждение[/bold] с длинным текстом, который "
        "должен остаться одной логической строкой без служебных подробностей Python"
    )

    def transcribe_with_warning(*_args, **_kwargs):
        warnings.warn(warning_message, stacklevel=2)
        return _make_result()

    backend = _make_backend()
    backend.transcribe.side_effect = transcribe_with_warning

    original_showwarning = warnings.showwarning
    with _cli_run(backend=backend):
        out = runner.invoke(app, [str(file) for file in files])

    warning_lines = [line for line in out.output.splitlines() if "Тестовое" in line]
    assert warning_lines == [f"Внимание: {warning_message}"]
    assert "UserWarning" not in out.output
    assert "warnings.warn" not in out.output
    assert warnings.showwarning is original_showwarning


def test_cli_default_options_passed_to_transcribe(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")

    with _cli_run(tmp_file=audio) as (backend, _write):
        out = runner.invoke(app, [str(audio)])

    backend.ensure_model_available.assert_called_once_with(
        "gigaam-v3-e2e-rnnt", "int8", ANY
    )
    assert backend.transcribe.call_args[0][2] == "ru"
    assert backend.transcribe.call_args[0][3] is None  # verbose=False → on_segment
    assert "Модель: gigaam-v3-e2e-rnnt" in out.output
    assert "Устройство: onnx" in out.output


def test_cli_identifies_onnx_backend_in_transcript_header(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")

    with _cli_run(tmp_file=audio, write=False):
        out = runner.invoke(app, [str(audio)])

    content = (tmp_path / "test-transcript.md").read_text(encoding="utf-8")
    assert out.exit_code == 0
    assert "- **Устройство**: ONNX (CPU)" in content


def test_cli_custom_options(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")

    with _cli_run(tmp_file=audio) as (backend, _write):
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

    backend.ensure_model_available.assert_called_once_with("small", "float16", ANY)
    assert backend.transcribe.call_args[0][2] == "ru"


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
    diarizer = MagicMock()
    diarizer.process.return_value = DiarizationRun(
        intervals=[
            SpeakerInterval(0.0, 0.5, 10),
            SpeakerInterval(0.5, 1.0, 20),
        ],
        elapsed_seconds=0.2,
    )

    with (
        _cli_run(tmp_file=audio, result=result) as (_backend, write),
        patch(
            "local_transcriber.cli.load_speaker_diarizer",
            return_value=diarizer,
        ) as load_diarizer,
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
    diarizer = MagicMock()
    diarizer.process.side_effect = RuntimeError("boom")

    with (
        _cli_run(tmp_file=audio, result=result) as (_backend, write),
        patch(
            "local_transcriber.cli.load_speaker_diarizer",
            return_value=diarizer,
        ),
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
    diarizer = MagicMock()
    diarizer.process.return_value = DiarizationRun(
        intervals=[
            SpeakerInterval(0.0, 0.5, 1),
            SpeakerInterval(0.5, 1.0, 2),
        ],
        elapsed_seconds=0.2,
    )

    with (
        _cli_run(tmp_file=audio, result=result),
        patch(
            "local_transcriber.cli.load_speaker_diarizer",
            return_value=diarizer,
        ),
    ):
        out = runner.invoke(app, [str(audio), "--diarize", "--verbose"])

    assert out.exit_code == 0
    assert "2 кластеров, 2 интервалов" in out.output
    assert "0.2 с" in out.output


def test_cli_empty_asr_skips_diarizer_and_reports_it(tmp_path):
    audio = tmp_path / "silence.wav"
    audio.write_bytes(b"fake")
    result = _make_result(segments=[])
    diarizer = MagicMock()

    with (
        _cli_run(tmp_file=audio, result=result),
        patch(
            "local_transcriber.cli.load_speaker_diarizer",
            return_value=diarizer,
        ),
    ):
        out = runner.invoke(app, [str(audio), "--diarize"])

    assert out.exit_code == 0
    diarizer.process.assert_not_called()
    assert "диаризация не запускалась" in out.output


def test_cli_diarizer_preflight_failure_does_not_start_asr_or_write(tmp_path):
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"fake")

    with (
        _cli_run(tmp_file=audio) as (backend, write),
        patch(
            "local_transcriber.cli.load_speaker_diarizer",
            side_effect=RuntimeError("модель повреждена"),
        ),
    ):
        out = runner.invoke(app, [str(audio), "--diarize"])

    assert out.exit_code == 1
    backend.transcribe.assert_not_called()
    write.assert_not_called()


def test_cli_diarize_rejects_model_without_word_timestamps_before_asr(tmp_path):
    """Проверка пословного контракта проходит в module до диаризатора и ASR."""
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"fake")
    backend = _make_backend(word_timestamps=False)

    with (
        _cli_run(tmp_file=audio, backend=backend) as (backend, write),
        patch("local_transcriber.cli.load_speaker_diarizer") as load_diarizer,
    ):
        out = runner.invoke(app, [str(audio), "--diarize"])

    assert out.exit_code == 1
    assert "пословные таймкоды" in out.output
    load_diarizer.assert_not_called()
    backend.transcribe.assert_not_called()
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
    diarizer = MagicMock()
    diarizer.process.return_value = DiarizationRun(intervals, elapsed_seconds=0.1)

    with (
        _cli_run(tmp_file=audio, result=result) as (_backend, write),
        patch(
            "local_transcriber.cli.load_speaker_diarizer",
            return_value=diarizer,
        ),
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

    with patch("local_transcriber.transcriber.get_backend") as get_backend:
        out = runner.invoke(
            app,
            [str(audio), "--no-diarize", "--speakers", "2"],
        )

    assert out.exit_code == 2
    assert "--no-diarize и --speakers несовместимы" in out.output
    get_backend.assert_not_called()


def test_cli_verbose_passes_on_segment_callback(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")

    with _cli_run(tmp_file=audio) as (backend, _write):
        runner.invoke(app, [str(audio), "--verbose"])

    on_segment = backend.transcribe.call_args[0][3]
    assert on_segment is not None
    assert callable(on_segment)


def test_cli_empty_speech_warning(tmp_path):
    audio = tmp_path / "silence.wav"
    audio.write_bytes(b"fake")
    result = _make_result(segments=[])

    with _cli_run(tmp_file=audio, result=result):
        out = runner.invoke(app, [str(audio)])

    assert out.exit_code == 0
    assert "Речь не обнаружена" in out.output


def test_cli_default_output_path(tmp_path):
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"fake")

    with _cli_run(tmp_file=audio) as (_backend, write):
        runner.invoke(app, [str(audio)])

    written_path: Path = write.call_args[0][1]
    assert written_path.name == "meeting-transcript.md"


def test_cli_custom_output_path(tmp_path):
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"fake")
    out_file = tmp_path / "custom.md"

    with _cli_run(tmp_file=audio) as (_backend, write):
        runner.invoke(app, [str(audio), "--output", str(out_file)])

    written_path: Path = write.call_args[0][1]
    assert written_path == out_file


def test_cli_passes_status_callback_to_transcribe(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")

    with _cli_run(tmp_file=audio) as (backend, _write):
        runner.invoke(app, [str(audio)])

    on_status = backend.transcribe.call_args[0][4]
    assert on_status is not None
    assert callable(on_status)


def test_cli_load_model_called_with_model_name(tmp_path):
    """ensure_model_available получает имя модели из флага CLI."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")

    with _cli_run(tmp_file=audio) as (backend, _write):
        runner.invoke(app, [str(audio), "--model", "large-v3"])

    assert backend.ensure_model_available.call_args[0][0] == "large-v3"


def test_cli_windows_cuda_diagnostic(tmp_path):
    """При проблеме драйвера Windows не предлагает установку Toolkit."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    backend = _make_backend()
    backend.transcribe.side_effect = RuntimeError("CUDA error: no device")

    with (
        _cli_run(tmp_file=audio, backend=backend),
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
    backend = _make_backend()
    backend.transcribe.side_effect = RuntimeError("CUDA error: no device")

    with (
        _cli_run(tmp_file=audio, backend=backend),
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


def test_cli_keyboard_interrupt(tmp_path):
    """Ctrl+C → exit code 130, 'Прервано пользователем' in output."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    backend = _make_backend()
    backend.transcribe.side_effect = KeyboardInterrupt

    with _cli_run(tmp_file=audio, backend=backend):
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
    backend = _make_backend()
    backend.transcribe.side_effect = RuntimeError("unexpected boom")

    with _cli_run(tmp_file=audio, backend=backend):
        out = runner.invoke(app, [str(audio), "--verbose"])

    assert out.exit_code == 1
    assert "unexpected boom" in out.output


def test_cli_unexpected_error_no_verbose_hint(tmp_path):
    """Unexpected error without --verbose → hint to use --verbose."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    backend = _make_backend()
    backend.transcribe.side_effect = RuntimeError("unexpected boom")

    with _cli_run(tmp_file=audio, backend=backend):
        out = runner.invoke(app, [str(audio)])

    assert out.exit_code == 1
    assert "Ошибка" in out.output
    assert "--verbose" in out.output


# === Batch mode tests ===


def test_cli_batch_two_files(tmp_path):
    """Два файла: одна загрузка модели, два распознавания."""
    a = tmp_path / "a.mp3"
    b = tmp_path / "b.mp3"
    a.write_bytes(b"fake")
    b.write_bytes(b"fake")

    with _cli_run() as (backend, _write):
        out = runner.invoke(app, [str(a), str(b)])

    assert out.exit_code == 0
    assert "2 обработано" in out.output
    assert backend.create_model.call_count == 1
    assert backend.transcribe.call_count == 2


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
    backend = _make_backend(result=result)
    diarizer = MagicMock()
    diarizer.process.return_value = DiarizationRun(
        intervals=[
            SpeakerInterval(0.0, 0.5, 1),
            SpeakerInterval(0.5, 1.0, 2),
        ],
        elapsed_seconds=0.1,
    )

    with (
        _cli_run(backend=backend, config=config) as (_backend, write),
        patch(
            "local_transcriber.cli.load_speaker_diarizer",
            return_value=diarizer,
        ) as load_diarizer,
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
    backend = _make_backend(result=result)
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
        _cli_run(backend=backend) as (_backend, write),
        patch(
            "local_transcriber.cli.load_speaker_diarizer",
            return_value=diarizer,
        ),
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

    with _cli_run():
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

    with (
        _cli_run() as (backend, _write),
        patch("local_transcriber.transcriber.get_backend") as get_backend,
        patch("local_transcriber.cli.load_speaker_diarizer") as load_diarizer,
    ):
        out = runner.invoke(app, [str(a), str(b), "--diarize"])

    assert out.exit_code == 0
    assert "0 обработано, 2 пропущено" in out.output
    get_backend.assert_not_called()
    backend.create_model.assert_not_called()
    load_diarizer.assert_not_called()


def test_cli_batch_force_overwrites(tmp_path):
    a = tmp_path / "a.mp3"
    a.write_bytes(b"fake")
    (tmp_path / "a-transcript.md").write_text("existing")
    b = tmp_path / "b.mp3"
    b.write_bytes(b"fake")

    with _cli_run():
        out = runner.invoke(app, [str(a), str(b), "--force"])

    assert out.exit_code == 0
    assert "Пропуск" not in out.output
    assert "2 обработано" in out.output


def test_cli_batch_per_file_error(tmp_path):
    a = tmp_path / "a.mp3"
    b = tmp_path / "b.mp3"
    a.write_bytes(b"fake")
    b.write_bytes(b"fake")
    backend = _make_backend()
    backend.transcribe.side_effect = [RuntimeError("oops"), _make_result()]

    with _cli_run(backend=backend):
        out = runner.invoke(app, [str(a), str(b)])

    assert out.exit_code == 1
    assert "1 обработано" in out.output
    assert "1 ошибок" in out.output


def test_cli_batch_invalid_in_prescan(tmp_path):
    a = tmp_path / "a.mp3"
    a.write_bytes(b"fake")
    b = tmp_path / "b.mp3"
    # b doesn't exist

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.transcriber.get_backend", return_value=_make_backend()),
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

    with _cli_run(tmp_file=audio, config={"model": "tiny"}) as (backend, _write):
        runner.invoke(app, [str(audio)])

    # ensure_model_available receives model name from config
    backend.ensure_model_available.assert_called_once_with("tiny", "int8", ANY)


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
    load_diarizer = MagicMock()

    with (
        _cli_run(tmp_file=audio, result=result, config={"diarize": config_value}),
        patch("local_transcriber.cli.load_speaker_diarizer", load_diarizer),
    ):
        out = runner.invoke(app, [str(audio), *cli_args])

    assert out.exit_code == 0
    assert load_diarizer.call_count == expected_calls


def test_cli_config_overrides_auto_device(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")

    with _cli_run(
        tmp_file=audio,
        config={"device": "openvino-cpu", "model": "medium", "compute_type": "int8"},
    ) as (backend, _write):
        out = runner.invoke(app, [str(audio)])

    assert out.exit_code == 0
    backend.ensure_model_available.assert_called_once_with("medium", "int8", ANY)
    backend.create_model.assert_called_once_with(
        "/models/medium", "openvino-cpu", "int8", cpu_threads=0
    )


def test_cli_cli_overrides_config(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")

    with _cli_run(tmp_file=audio, config={"model": "tiny"}) as (backend, _write):
        runner.invoke(app, [str(audio), "--model", "small"])

    # CLI --model overrides config
    backend.ensure_model_available.assert_called_once_with("small", "int8", ANY)


def test_cli_batch_empty_speech_warning(tmp_path):
    """Batch mode warns when a file has no detected speech."""
    a = tmp_path / "a.mp3"
    b = tmp_path / "b.mp3"
    a.write_bytes(b"fake")
    b.write_bytes(b"fake")

    result_empty = _make_result(segments=[])
    result_ok = _make_result()
    backend = _make_backend()
    backend.transcribe.side_effect = [result_empty, result_ok]

    with _cli_run(backend=backend):
        out = runner.invoke(app, [str(a), str(b)])

    assert out.exit_code == 0
    assert "Речь не обнаружена" in out.output
    assert "2 обработано" in out.output


# === CLI with --device openvino-gpu ===


def test_cli_openvino_gpu_happy_path(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    backend = _make_backend(actual_ov_device="GPU")

    with (
        _cli_run(tmp_file=audio, backend=backend),
        patch(
            "local_transcriber.transcriber.get_intel_gpu_name",
            return_value="Intel Arc 140T",
        ),
    ):
        out = runner.invoke(app, [str(audio), "--device", "openvino-gpu"])

    assert out.exit_code == 0


def test_cli_openvino_alias_resolves_to_gpu(tmp_path):
    """--device openvino резолвится в openvino-gpu, шапка берёт имя GPU из данных module."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    backend = _make_backend(actual_ov_device="GPU")

    with (
        _cli_run(tmp_file=audio, backend=backend, write=False),
        patch("local_transcriber.transcriber.detect_device", return_value="openvino-gpu"),
        patch(
            "local_transcriber.transcriber.get_intel_gpu_name",
            return_value="Intel Arc 140T",
        ),
    ):
        out = runner.invoke(app, [str(audio), "--device", "openvino"])

    content = (tmp_path / "test-transcript.md").read_text(encoding="utf-8")
    assert out.exit_code == 0
    assert backend.create_model.call_args[0][1] == "openvino-gpu"
    assert "- **Устройство**: OpenVINO (Intel Arc 140T)" in content


def test_cli_reports_openvino_cpu_when_gpu_was_requested(tmp_path):
    """OpenVINO, выбравший CPU вместо запрошенного GPU, виден пользователю."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    backend = _make_backend(actual_ov_device="CPU")

    with (
        _cli_run(tmp_file=audio, backend=backend),
        patch("local_transcriber.transcriber.detect_device", return_value="openvino-gpu"),
    ):
        out = runner.invoke(app, [str(audio), "--device", "openvino"])

    assert out.exit_code == 0
    assert "Запрошено openvino, используется openvino-cpu" in out.output


def test_cli_threads_reach_adapter(tmp_path):
    """--threads передаётся в create_model как cpu_threads."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")

    with _cli_run(tmp_file=audio) as (backend, _write):
        out = runner.invoke(app, [str(audio), "--threads", "8"])

    assert out.exit_code == 0
    assert backend.create_model.call_args.kwargs["cpu_threads"] == 8


def test_cli_threads_default_zero(tmp_path):
    """Без --threads create_model получает cpu_threads=0."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")

    with _cli_run(tmp_file=audio) as (backend, _write):
        out = runner.invoke(app, [str(audio)])

    assert out.exit_code == 0
    assert backend.create_model.call_args.kwargs["cpu_threads"] == 0


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

    with (
        _cli_run(tmp_file=audio, result=result),
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

    with (
        _cli_run(tmp_file=audio, result=result),
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
    backend = _make_backend()
    backend.transcribe.side_effect = [result_warn, result_ok]

    with (
        _cli_run(backend=backend),
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

    with (
        _cli_run(tmp_file=audio, result=result),
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

    with _cli_run(tmp_file=audio):
        out = runner.invoke(app, [str(audio)])

    assert out.exit_code == 0
    assert "потеря хвоста" not in out.output
    assert "галлюцинации" not in out.output


def test_cli_toml_compute_type_overrides_device_default(tmp_path):
    """compute_type из TOML важнее умолчания устройства и считается явным."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")

    with (
        _cli_run(tmp_file=audio, config={"device": "cuda", "compute_type": "int8"}) as (
            backend,
            _write,
        ),
        patch("local_transcriber.transcriber.get_backend", return_value=backend) as get_backend,
    ):
        out = runner.invoke(app, [str(audio)])

    assert out.exit_code == 0
    get_backend.assert_called_once_with("cuda", compute_type_explicit=True)
    backend.ensure_model_available.assert_called_once_with("medium", "int8", ANY)


def test_cli_device_default_compute_type_when_not_configured(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")

    with _cli_run(tmp_file=audio, config={"device": "cuda"}) as (backend, _write):
        out = runner.invoke(app, [str(audio)])

    assert out.exit_code == 0
    backend.ensure_model_available.assert_called_once_with("medium", "float16", ANY)
