from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from local_transcriber.transcriber import (
    Segment,
    TranscribeResult,
    _transcribe_file,
    ensure_model_available,
    load_model,
    transcribe,
)


# === Helpers ===


def _make_result(
    count: int = 2,
    language: str = "ru",
    probability: float = 0.95,
    duration: float = 60.0,
    device_used: str = "cpu",
) -> TranscribeResult:
    segments = [
        Segment(start=float(i * 5), end=float(i * 5 + 4), text=f" Segment {i}")
        for i in range(count)
    ]
    return TranscribeResult(
        segments=segments,
        language=language,
        language_probability=probability,
        duration=duration,
        device_used=device_used,
    )


def _make_backend(
    model=None,
    transcribe_result=None,
    create_model_error=None,
    transcribe_error=None,
    model_path="/mock/model",
):
    """Создаёт mock-бэкенд с настраиваемым поведением."""
    backend = MagicMock()
    backend.ensure_model_available.return_value = model_path

    if create_model_error:
        backend.create_model.side_effect = create_model_error
    else:
        backend.create_model.return_value = model or MagicMock()

    if transcribe_error:
        backend.transcribe.side_effect = transcribe_error
    elif transcribe_result:
        backend.transcribe.return_value = transcribe_result
    else:
        backend.transcribe.return_value = _make_result()

    return backend


def _create_model_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    (path / "preprocessor_config.json").write_text("{}")
    (path / "tokenizer.json").write_text("{}")
    (path / "vocabulary.json").write_text("{}")
    (path / "model.bin").write_bytes(b"ok")
    return path


# === transcribe() tests ===


@patch("local_transcriber.transcriber.get_backend")
def test_transcribe_collects_segments(mock_get_backend):
    result_data = _make_result(count=3)
    backend = _make_backend(transcribe_result=result_data)
    mock_get_backend.return_value = backend

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


@patch("local_transcriber.transcriber.get_backend")
def test_transcribe_calls_on_segment(mock_get_backend):
    result_data = _make_result(count=3)
    backend = _make_backend(transcribe_result=result_data)
    mock_get_backend.return_value = backend

    callback = MagicMock()

    transcribe(
        file_path=Path("test.mp3"),
        model_name="tiny",
        device="cpu",
        on_segment=callback,
    )

    # on_segment is passed through to backend.transcribe
    call_args = backend.transcribe.call_args
    assert call_args.kwargs.get("on_segment") is callback or call_args[0][3] is callback


@patch("local_transcriber.transcriber.get_backend")
def test_transcribe_cuda_fallback(mock_get_backend):
    """CUDA error at init -> fallback на CPU."""
    cuda_backend = _make_backend(create_model_error=RuntimeError("CUDA out of memory"))
    cpu_backend = _make_backend(
        transcribe_result=_make_result(count=2, device_used="cpu"),
        model_path="/mock/cpu/model",
    )

    def backend_for_device(device, **kwargs):
        return cuda_backend if device == "cuda" else cpu_backend

    mock_get_backend.side_effect = backend_for_device

    with pytest.warns(UserWarning, match="Переключение на CPU"):
        result = transcribe(
            file_path=Path("test.mp3"),
            model_name="tiny",
            device="cuda",
        )

    assert result.device_used == "cpu"
    assert len(result.segments) == 2


@patch("local_transcriber.transcriber.get_backend")
def test_transcribe_device_used(mock_get_backend):
    backend = _make_backend(
        transcribe_result=_make_result(count=1, device_used="cuda"),
    )
    mock_get_backend.return_value = backend

    result = transcribe(
        file_path=Path("test.mp3"),
        model_name="tiny",
        device="cuda",
    )

    assert result.device_used == "cuda"
    backend.create_model.assert_called_once()


@patch("local_transcriber.transcriber.get_backend")
def test_transcribe_cuda_fallback_on_transcribe_call(mock_get_backend):
    """CUDA error in transcribe (not init) triggers CPU fallback."""
    cuda_backend = _make_backend(
        transcribe_error=RuntimeError("CUDA error during transcription"),
    )
    cpu_backend = _make_backend(
        transcribe_result=_make_result(count=2, device_used="cpu"),
        model_path="/mock/cpu/model",
    )

    def backend_for_device(device, **kwargs):
        return cuda_backend if device == "cuda" else cpu_backend

    mock_get_backend.side_effect = backend_for_device

    with pytest.warns(UserWarning, match="Переключение на CPU"):
        result = transcribe(
            file_path=Path("test.mp3"),
            model_name="tiny",
            device="cuda",
        )

    assert result.device_used == "cpu"
    assert len(result.segments) == 2


@patch("local_transcriber.transcriber.get_backend")
def test_transcribe_reports_missing_socksio_for_proxy(mock_get_backend):
    backend = _make_backend(
        create_model_error=ImportError(
            "Using SOCKS proxy, but the 'socksio' package is not installed."
        ),
    )
    mock_get_backend.return_value = backend

    # ImportError is not caught as backend error → propagates
    with pytest.raises(ImportError, match="socksio"):
        transcribe(
            file_path=Path("test.mp3"),
            model_name="tiny",
            device="cpu",
        )


@patch("local_transcriber.transcriber.get_backend")
def test_transcribe_reports_status_transitions(mock_get_backend):
    backend = _make_backend(transcribe_result=_make_result(count=1))
    mock_get_backend.return_value = backend

    statuses: list[str] = []

    transcribe(
        file_path=Path("test.mp3"),
        model_name="tiny",
        device="cpu",
        on_status=statuses.append,
    )

    # load_model reports init status, _transcribe_file reports transcribe status
    assert any("Инициализирую модель" in s for s in statuses)
    assert any("Транскрибирую" in s for s in statuses)


@patch("local_transcriber.transcriber.get_backend")
def test_transcribe_strict_cuda_error(mock_get_backend):
    """strict_device=True + CUDA error -> raise, без fallback."""
    backend = _make_backend(create_model_error=RuntimeError("CUDA out of memory"))
    mock_get_backend.return_value = backend

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        transcribe(
            file_path=Path("test.mp3"),
            model_name="tiny",
            device="cuda",
            strict_device=True,
        )


@patch("local_transcriber.transcriber.get_backend")
def test_transcribe_non_strict_cuda_fallback(mock_get_backend):
    """strict_device=False + CUDA error -> fallback на CPU."""
    cuda_backend = _make_backend(create_model_error=RuntimeError("CUDA out of memory"))
    cpu_backend = _make_backend(
        transcribe_result=_make_result(count=2, device_used="cpu"),
        model_path="/mock/cpu/model",
    )

    def backend_for_device(device, **kwargs):
        return cuda_backend if device == "cuda" else cpu_backend

    mock_get_backend.side_effect = backend_for_device

    with pytest.warns(UserWarning, match="Переключение на CPU"):
        result = transcribe(
            file_path=Path("test.mp3"),
            model_name="tiny",
            device="cuda",
            strict_device=False,
        )

    assert result.device_used == "cpu"
    assert len(result.segments) == 2


@patch("local_transcriber.transcriber.get_backend")
def test_transcribe_strict_cuda_error_during_transcription(mock_get_backend):
    """strict_device=True + CUDA error during transcription -> raise."""
    backend = _make_backend(
        transcribe_error=RuntimeError("CUDA error during transcription"),
    )
    mock_get_backend.return_value = backend

    with pytest.raises(RuntimeError, match="CUDA error during transcription"):
        transcribe(
            file_path=Path("test.mp3"),
            model_name="tiny",
            device="cuda",
            strict_device=True,
        )


# === load_model() tests ===


@patch("local_transcriber.transcriber.get_backend")
def test_load_model_cuda_fallback(mock_get_backend):
    cuda_backend = _make_backend(create_model_error=RuntimeError("CUDA out of memory"))
    cpu_model = MagicMock()
    cpu_backend = _make_backend(model=cpu_model, model_path="/mock/cpu/model")

    def backend_for_device(device, **kwargs):
        return cuda_backend if device == "cuda" else cpu_backend

    mock_get_backend.side_effect = backend_for_device

    with pytest.warns(UserWarning, match="Переключение на CPU"):
        model, actual_device, backend, model_path = load_model("tiny", "cuda", "int8")

    assert actual_device == "cpu"
    assert model is cpu_model


@patch("local_transcriber.transcriber.get_backend")
def test_load_model_strict_raises(mock_get_backend):
    backend = _make_backend(create_model_error=RuntimeError("CUDA out of memory"))
    mock_get_backend.return_value = backend

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        load_model("tiny", "cuda", "int8", strict_device=True)


@patch("local_transcriber.transcriber.get_backend")
def test_load_model_returns_backend_and_path(mock_get_backend):
    backend = _make_backend(model_path="/mock/model/path")
    mock_get_backend.return_value = backend

    model, actual_device, returned_backend, model_path = load_model("tiny", "cpu", "int8")

    assert returned_backend is backend
    assert model_path == "/mock/model/path"
    assert actual_device == "cpu"


# === _transcribe_file() tests ===


def test__transcribe_file_basic():
    result_data = _make_result(count=2)
    backend = _make_backend(transcribe_result=result_data)

    tfr = _transcribe_file(
        model=MagicMock(),
        actual_device="cpu",
        backend=backend,
        model_path="/mock/model",
        file_path=Path("test.mp3"),
        model_name="tiny",
        compute_type="int8",
    )

    assert len(tfr.result.segments) == 2
    assert tfr.actual_device == "cpu"
    assert tfr.backend is backend
    assert tfr.model_path == "/mock/model"


# === ensure_model_available() tests (через FasterWhisperBackend) ===


@patch("local_transcriber.backends.faster_whisper.snapshot_download")
def test_ensure_model_available_uses_cache_first(mock_snapshot_download, tmp_path):
    model_dir = _create_model_dir(tmp_path / "cache-model")
    mock_snapshot_download.return_value = str(model_dir)

    result = ensure_model_available("large-v3")

    assert result == str(model_dir)
    mock_snapshot_download.assert_called_once()
    assert mock_snapshot_download.call_args.kwargs["local_files_only"] is True


@patch("local_transcriber.backends.faster_whisper._validate_model_dir")
@patch("local_transcriber.backends.faster_whisper.snapshot_download")
def test_ensure_model_available_downloads_on_cache_miss(mock_snapshot_download, mock_validate_model_dir):
    from huggingface_hub.errors import LocalEntryNotFoundError

    mock_snapshot_download.side_effect = [
        LocalEntryNotFoundError("not cached"),
        "/downloaded/model",
    ]
    statuses: list[str] = []

    result = ensure_model_available("large-v3", on_status=statuses.append)

    assert result == "/downloaded/model"
    assert mock_snapshot_download.call_args_list[0].kwargs["local_files_only"] is True
    assert mock_snapshot_download.call_args_list[1].kwargs["local_files_only"] is False
    assert "Проверяю кэш модели large-v3..." in statuses
    assert "Скачиваю модель large-v3 из Hugging Face..." in statuses


def test_ensure_model_available_accepts_local_directory(tmp_path):
    model_dir = _create_model_dir(tmp_path / "model")

    result = ensure_model_available(str(model_dir))

    assert result == str(model_dir)


def test_ensure_model_available_accepts_repo_id(tmp_path):
    model_dir = _create_model_dir(tmp_path / "repo-model")
    with patch(
        "local_transcriber.backends.faster_whisper.snapshot_download",
        return_value=str(model_dir),
    ) as mock_snapshot_download:
        result = ensure_model_available("org/model")

    assert result == str(model_dir)
    assert mock_snapshot_download.call_args.kwargs["local_files_only"] is True


def test_ensure_model_available_rejects_unsupported_alias():
    with pytest.raises(ValueError, match="Неподдерживаемая модель"):
        ensure_model_available("distil-large-v3")


@patch("local_transcriber.backends.faster_whisper.snapshot_download")
def test_ensure_model_available_redownloads_incomplete_cache(mock_snapshot_download, tmp_path):
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "config.json").write_text("{}")
    (incomplete / "preprocessor_config.json").write_text("{}")
    (incomplete / "tokenizer.json").write_text("{}")
    (incomplete / "vocabulary.json").write_text("{}")

    complete = tmp_path / "complete"
    complete.mkdir()
    (complete / "config.json").write_text("{}")
    (complete / "preprocessor_config.json").write_text("{}")
    (complete / "tokenizer.json").write_text("{}")
    (complete / "vocabulary.json").write_text("{}")
    (complete / "model.bin").write_bytes(b"ok")

    mock_snapshot_download.side_effect = [
        str(incomplete),
        str(complete),
    ]
    statuses: list[str] = []

    result = ensure_model_available("large-v3", on_status=statuses.append)

    assert result == str(complete)
    assert "Кэш модели large-v3 неполный, докачиваю..." in statuses


def test_ensure_model_available_rejects_incomplete_local_directory(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    with pytest.raises(ValueError, match="Неполная локальная модель"):
        ensure_model_available(str(model_dir))


# === Cross-backend fallback (openvino → cpu) ===


@patch("local_transcriber.transcriber.get_backend")
def test_load_model_openvino_fallback_to_cpu(mock_get_backend):
    """OpenVINO ошибка при init → fallback на CPU (FasterWhisper)."""
    ov_backend = _make_backend(
        create_model_error=RuntimeError("OpenVINO model load failed"),
    )
    cpu_model = MagicMock()
    cpu_backend = _make_backend(model=cpu_model, model_path="/mock/cpu/model")

    def backend_for_device(device, **kwargs):
        return ov_backend if device == "openvino" else cpu_backend

    mock_get_backend.side_effect = backend_for_device

    with pytest.warns(UserWarning, match="Переключение на CPU"):
        model, actual_device, backend, model_path = load_model(
            "medium", "openvino", "int8",
        )

    assert actual_device == "cpu"
    assert model is cpu_model
    assert backend is cpu_backend
    assert model_path == "/mock/cpu/model"


@patch("local_transcriber.transcriber.get_backend")
def test_transcribe_file_openvino_midstream_fallback(mock_get_backend):
    """OpenVINO ошибка при транскрипции → fallback на CPU."""
    ov_backend = _make_backend(
        transcribe_error=RuntimeError("OpenVINO inference error"),
    )
    cpu_backend = _make_backend(
        transcribe_result=_make_result(count=2, device_used="cpu"),
        model_path="/mock/cpu/model",
    )

    def backend_for_device(device, **kwargs):
        return ov_backend if device == "openvino" else cpu_backend

    mock_get_backend.side_effect = backend_for_device

    with pytest.warns(UserWarning, match="Переключение на CPU"):
        tfr = _transcribe_file(
            model=MagicMock(),
            actual_device="openvino",
            backend=ov_backend,
            model_path="/mock/ov/model",
            file_path=Path("test.mp3"),
            model_name="medium",
            compute_type="int8",
        )

    assert tfr.actual_device == "cpu"
    assert tfr.backend is cpu_backend
    assert tfr.model_path == "/mock/cpu/model"


@patch("local_transcriber.transcriber.get_backend")
def test_openvino_strict_device_no_fallback(mock_get_backend):
    """strict_device=True + OpenVINO ошибка → raise."""
    backend = _make_backend(
        create_model_error=RuntimeError("OpenVINO model load failed"),
    )
    mock_get_backend.return_value = backend

    with pytest.raises(RuntimeError, match="OpenVINO"):
        load_model("medium", "openvino", "int8", strict_device=True)
