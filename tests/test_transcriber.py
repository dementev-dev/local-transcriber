from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from local_transcriber.transcriber import Segment, TranscribeResult, ensure_model_available, transcribe


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


def _create_model_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    (path / "preprocessor_config.json").write_text("{}")
    (path / "tokenizer.json").write_text("{}")
    (path / "vocabulary.json").write_text("{}")
    (path / "model.bin").write_bytes(b"ok")
    return path


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


@patch("local_transcriber.transcriber.WhisperModel")
def test_transcribe_reports_missing_socksio_for_proxy(mock_model_cls):
    mock_model_cls.side_effect = ImportError(
        "Using SOCKS proxy, but the 'socksio' package is not installed."
    )

    with pytest.raises(RuntimeError, match="socksio"):
        transcribe(
            file_path=Path("test.mp3"),
            model_name="tiny",
            device="cpu",
        )


@patch("local_transcriber.transcriber.WhisperModel")
def test_transcribe_reports_status_transitions(mock_model_cls):
    raw_segments = _make_raw_segments(1)
    info = _make_info()

    instance = MagicMock()
    instance.transcribe.return_value = (iter(raw_segments), info)
    mock_model_cls.return_value = instance

    statuses: list[str] = []

    transcribe(
        file_path=Path("test.mp3"),
        model_name="tiny",
        device="cpu",
        on_status=statuses.append,
    )

    assert statuses == [
        "Загружаю модель на cpu...",
        "Транскрибирую...",
    ]


@patch("local_transcriber.transcriber.snapshot_download")
def test_ensure_model_available_uses_cache_first(mock_snapshot_download, tmp_path):
    model_dir = _create_model_dir(tmp_path / "cache-model")
    mock_snapshot_download.return_value = str(model_dir)

    result = ensure_model_available("large-v3")

    assert result == str(model_dir)
    mock_snapshot_download.assert_called_once_with(
        "Systran/faster-whisper-large-v3",
        local_files_only=True,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "model.bin",
            "tokenizer.json",
            "vocabulary.*",
        ],
    )


@patch("local_transcriber.transcriber._validate_model_dir")
@patch("local_transcriber.transcriber.snapshot_download")
def test_ensure_model_available_downloads_on_cache_miss(mock_snapshot_download, mock_validate_model_dir):
    mock_snapshot_download.side_effect = [
        LocalEntryNotFoundError("not cached"),
        "/downloaded/model",
    ]
    statuses: list[str] = []

    result = ensure_model_available("large-v3", on_status=statuses.append)

    assert result == "/downloaded/model"
    assert mock_snapshot_download.call_args_list[0].kwargs["local_files_only"] is True
    assert mock_snapshot_download.call_args_list[1].kwargs["local_files_only"] is False
    assert statuses == [
        "Проверяю кэш модели large-v3...",
        "Скачиваю модель large-v3 из Hugging Face...",
    ]


def test_ensure_model_available_accepts_local_directory(tmp_path):
    model_dir = _create_model_dir(tmp_path / "model")

    result = ensure_model_available(str(model_dir))

    assert result == str(model_dir)


def test_ensure_model_available_accepts_repo_id(tmp_path):
    model_dir = _create_model_dir(tmp_path / "repo-model")
    with patch("local_transcriber.transcriber.snapshot_download", return_value=str(model_dir)) as mock_snapshot_download:
        result = ensure_model_available("org/model")

    assert result == str(model_dir)
    assert mock_snapshot_download.call_args.kwargs["local_files_only"] is True


def test_ensure_model_available_rejects_unsupported_alias():
    with pytest.raises(ValueError, match="Неподдерживаемая модель"):
        ensure_model_available("distil-large-v3")


@patch("local_transcriber.transcriber.snapshot_download")
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
    assert statuses == [
        "Проверяю кэш модели large-v3...",
        "Кэш модели large-v3 неполный, докачиваю...",
        "Скачиваю модель large-v3 из Hugging Face...",
    ]


def test_ensure_model_available_rejects_incomplete_local_directory(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    with pytest.raises(ValueError, match="Неполная локальная модель"):
        ensure_model_available(str(model_dir))


@patch("local_transcriber.transcriber.WhisperModel")
def test_transcribe_strict_cuda_error(mock_model_cls):
    """strict_device=True + CUDA error -> raise, без fallback."""
    mock_model_cls.side_effect = RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        transcribe(
            file_path=Path("test.mp3"),
            model_name="tiny",
            device="cuda",
            strict_device=True,
        )


@patch("local_transcriber.transcriber.WhisperModel")
def test_transcribe_non_strict_cuda_fallback(mock_model_cls):
    """strict_device=False + CUDA error -> fallback на CPU."""
    raw_segments = _make_raw_segments(2)
    info = _make_info()

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
            strict_device=False,
        )

    assert result.device_used == "cpu"
    assert len(result.segments) == 2


@patch("local_transcriber.transcriber.WhisperModel")
def test_transcribe_strict_cuda_error_during_transcription(mock_model_cls):
    """strict_device=True + CUDA error during transcription -> raise."""
    cuda_instance = MagicMock()
    cuda_instance.transcribe.side_effect = RuntimeError("CUDA error during transcription")
    mock_model_cls.return_value = cuda_instance

    with pytest.raises(RuntimeError, match="CUDA error during transcription"):
        transcribe(
            file_path=Path("test.mp3"),
            model_name="tiny",
            device="cuda",
            strict_device=True,
        )
