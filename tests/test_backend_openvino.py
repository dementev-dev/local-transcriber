"""Тесты для OpenVINO бэкенда."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from local_transcriber.backends.openvino import (
    MODEL_REPOS,
    OpenVINOBackend,
    _validate_model_dir,
)
from local_transcriber.types import Segment


# === _resolve_repo ===


def test_resolve_repo_exact_match():
    backend = OpenVINOBackend(compute_type_explicit=True)
    assert backend._resolve_repo("medium", "int8") == ("OpenVINO/whisper-medium-int8-ov", "int8")


def test_resolve_repo_large_v3_fp16():
    backend = OpenVINOBackend(compute_type_explicit=True)
    assert backend._resolve_repo("large-v3", "fp16") == ("OpenVINO/whisper-large-v3-fp16-ov", "fp16")


def test_resolve_repo_explicit_unsupported_pair_raises():
    """Явный --compute-type с несуществующей парой → ошибка."""
    backend = OpenVINOBackend(compute_type_explicit=True)
    with pytest.raises(ValueError, match="недоступна с compute_type='fp16'"):
        backend._resolve_repo("medium", "fp16")


def test_resolve_repo_explicit_unknown_model_raises():
    backend = OpenVINOBackend(compute_type_explicit=True)
    with pytest.raises(ValueError, match="не найдена для OpenVINO"):
        backend._resolve_repo("distil-large-v3", "int8")


def test_resolve_repo_implicit_fallback():
    """Неявный compute_type: если int8 недоступен для base, fallback на fp16."""
    backend = OpenVINOBackend(compute_type_explicit=False)
    # base + int8 не существует, но base + fp16 есть
    assert backend._resolve_repo("base", "int8") == ("OpenVINO/whisper-base-fp16-ov", "fp16")


def test_resolve_repo_implicit_large_v3_prefers_fp16():
    """Неявный compute_type: large-v3 автоматически получает fp16."""
    backend = OpenVINOBackend(compute_type_explicit=False)
    # Дефолт int8, но для large-v3 override на fp16
    assert backend._resolve_repo("large-v3", "int8") == ("OpenVINO/whisper-large-v3-fp16-ov", "fp16")


def test_resolve_repo_explicit_large_v3_int8_respected():
    """Явный --compute-type int8 для large-v3 → уважается."""
    backend = OpenVINOBackend(compute_type_explicit=True)
    assert backend._resolve_repo("large-v3", "int8") == ("OpenVINO/whisper-large-v3-int8-ov", "int8")


# === ensure_model_available ===


@patch("local_transcriber.backends.openvino.snapshot_download")
def test_ensure_model_available_cache_hit(mock_download, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "openvino_encoder_model.xml").write_text("<xml/>")
    (model_dir / "openvino_decoder_model.xml").write_text("<xml/>")
    mock_download.return_value = str(model_dir)

    backend = OpenVINOBackend(compute_type_explicit=True)
    result = backend.ensure_model_available("medium", "int8")

    assert result == str(model_dir)
    mock_download.assert_called_once()
    assert mock_download.call_args.kwargs["local_files_only"] is True


@patch("local_transcriber.backends.openvino.snapshot_download")
def test_ensure_model_available_downloads(mock_download, tmp_path):
    from huggingface_hub.errors import LocalEntryNotFoundError

    model_dir = tmp_path / "downloaded"
    model_dir.mkdir()
    (model_dir / "openvino_encoder_model.xml").write_text("<xml/>")
    (model_dir / "openvino_decoder_model.xml").write_text("<xml/>")

    mock_download.side_effect = [
        LocalEntryNotFoundError("not cached"),
        str(model_dir),
    ]

    backend = OpenVINOBackend(compute_type_explicit=True)
    statuses: list[str] = []
    result = backend.ensure_model_available("medium", "int8", on_status=statuses.append)

    assert result == str(model_dir)
    assert any("Скачиваю" in s for s in statuses)


# === create_model ===


def test_create_model_cpu():
    mock_ov = MagicMock()
    mock_pipeline = MagicMock()
    mock_ov.WhisperPipeline.return_value = mock_pipeline

    backend = OpenVINOBackend(ov_device="openvino-cpu")
    with patch.dict("sys.modules", {"openvino_genai": mock_ov}):
        model = backend.create_model("/path/to/model", "openvino-cpu", "int8")

    mock_ov.WhisperPipeline.assert_called_once_with("/path/to/model", "CPU")
    assert model is mock_pipeline
    assert backend.actual_ov_device == "CPU"


def test_create_model_gpu():
    mock_ov = MagicMock()
    mock_pipeline = MagicMock()
    mock_ov.WhisperPipeline.return_value = mock_pipeline

    backend = OpenVINOBackend(ov_device="openvino-gpu")
    with patch.dict("sys.modules", {"openvino_genai": mock_ov}):
        model = backend.create_model("/path/to/model", "openvino-gpu", "fp16")

    mock_ov.WhisperPipeline.assert_called_once_with("/path/to/model", "GPU")
    assert model is mock_pipeline
    assert backend.actual_ov_device == "GPU"


def test_create_model_openvino_auto_detects_gpu():
    """ov_device='openvino' + GPU доступен → WhisperPipeline получает 'GPU'."""
    mock_ov = MagicMock()
    mock_pipeline = MagicMock()
    mock_ov.WhisperPipeline.return_value = mock_pipeline

    mock_core = MagicMock()
    mock_core.return_value.available_devices = ["CPU", "GPU"]

    backend = OpenVINOBackend(ov_device="openvino")
    with (
        patch.dict("sys.modules", {"openvino_genai": mock_ov, "openvino": MagicMock(Core=mock_core)}),
    ):
        model = backend.create_model("/path/to/model", "openvino", "int8")

    mock_ov.WhisperPipeline.assert_called_once_with("/path/to/model", "GPU")
    assert backend.actual_ov_device == "GPU"


def test_create_model_openvino_auto_falls_back_to_cpu():
    """ov_device='openvino' + нет GPU → WhisperPipeline получает 'CPU'."""
    mock_ov = MagicMock()
    mock_pipeline = MagicMock()
    mock_ov.WhisperPipeline.return_value = mock_pipeline

    mock_core = MagicMock()
    mock_core.return_value.available_devices = ["CPU"]

    backend = OpenVINOBackend(ov_device="openvino")
    with (
        patch.dict("sys.modules", {"openvino_genai": mock_ov, "openvino": MagicMock(Core=mock_core)}),
    ):
        model = backend.create_model("/path/to/model", "openvino", "int8")

    mock_ov.WhisperPipeline.assert_called_once_with("/path/to/model", "CPU")
    assert backend.actual_ov_device == "CPU"


# === transcribe ===


def test_transcribe_maps_chunks_to_segments():
    """Проверяет маппинг chunks → Segment[] и формат языка."""
    backend = OpenVINOBackend()

    mock_model = MagicMock()
    chunk1 = MagicMock()
    chunk1.start_ts = 0.0
    chunk1.end_ts = 3.5
    chunk1.text = " Привет мир"
    chunk2 = MagicMock()
    chunk2.start_ts = 3.5
    chunk2.end_ts = 7.0
    chunk2.text = " Тестовый сегмент"

    mock_result = MagicMock()
    mock_result.chunks = [chunk1, chunk2]
    mock_model.generate.return_value = mock_result

    raw_audio = np.zeros(16000 * 10, dtype=np.float32)  # 10 секунд

    with patch("faster_whisper.decode_audio", return_value=raw_audio):
        result = backend.transcribe(
            mock_model, Path("test.mp3"), language="ru",
        )

    assert len(result.segments) == 2
    assert result.segments[0].text == " Привет мир"
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 3.5
    assert result.duration == 10.0

    # Проверяем формат языка для OpenVINO GenAI
    call_kwargs = mock_model.generate.call_args
    assert call_kwargs.kwargs["language"] == "<|ru|>"
    assert call_kwargs.kwargs["return_timestamps"] is True


def test_transcribe_calls_tolist():
    """raw_speech передаётся как list, не ndarray."""
    backend = OpenVINOBackend()
    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.chunks = []
    mock_model.generate.return_value = mock_result

    raw_audio = np.zeros(160, dtype=np.float32)

    with patch("faster_whisper.decode_audio", return_value=raw_audio):
        backend.transcribe(mock_model, Path("test.mp3"), language=None)

    call_args = mock_model.generate.call_args[0][0]
    assert isinstance(call_args, list)


def test_transcribe_no_language_auto():
    """Без указания языка — не передаём language в generate."""
    backend = OpenVINOBackend()
    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.chunks = []
    mock_model.generate.return_value = mock_result

    raw_audio = np.zeros(160, dtype=np.float32)

    with patch("faster_whisper.decode_audio", return_value=raw_audio):
        result = backend.transcribe(mock_model, Path("test.mp3"), language=None)

    call_kwargs = mock_model.generate.call_args.kwargs
    assert "language" not in call_kwargs
    assert result.language == "auto"
    assert result.language_probability == 0.0


def test_transcribe_calls_on_segment():
    backend = OpenVINOBackend()
    mock_model = MagicMock()
    chunk = MagicMock()
    chunk.start_ts = 0.0
    chunk.end_ts = 2.0
    chunk.text = " Test"
    mock_result = MagicMock()
    mock_result.chunks = [chunk]
    mock_model.generate.return_value = mock_result

    raw_audio = np.zeros(16000, dtype=np.float32)
    callback = MagicMock()

    with patch("faster_whisper.decode_audio", return_value=raw_audio):
        backend.transcribe(
            mock_model, Path("test.mp3"), language="en", on_segment=callback,
        )

    callback.assert_called_once()
    seg = callback.call_args[0][0]
    assert isinstance(seg, Segment)
    assert seg.text == " Test"


# === _validate_model_dir ===


def test_validate_model_dir_ok(tmp_path):
    (tmp_path / "openvino_encoder_model.xml").write_text("<xml/>")
    (tmp_path / "openvino_decoder_model.xml").write_text("<xml/>")
    _validate_model_dir(tmp_path)  # should not raise


def test_validate_model_dir_missing(tmp_path):
    (tmp_path / "openvino_encoder_model.xml").write_text("<xml/>")
    with pytest.raises(ValueError, match="openvino_decoder_model.xml"):
        _validate_model_dir(tmp_path)
