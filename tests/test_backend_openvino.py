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
from local_transcriber.types import UNKNOWN_LANGUAGE, Segment, Word

# === _resolve_repo ===


def test_model_catalog_contains_large_v3_turbo_profiles():
    assert {
        pair: repo for pair, repo in MODEL_REPOS.items() if pair[0] == "large-v3-turbo"
    } == {
        ("large-v3-turbo", "int8"): "OpenVINO/whisper-large-v3-turbo-int8-ov",
        ("large-v3-turbo", "fp16"): "OpenVINO/whisper-large-v3-turbo-fp16-ov",
    }


def test_resolve_repo_exact_match():
    backend = OpenVINOBackend(compute_type_explicit=True)
    assert backend._resolve_repo("medium", "int8") == (
        "OpenVINO/whisper-medium-int8-ov",
        "int8",
    )


def test_resolve_repo_large_v3_fp16():
    backend = OpenVINOBackend(compute_type_explicit=True)
    assert backend._resolve_repo("large-v3", "fp16") == (
        "OpenVINO/whisper-large-v3-fp16-ov",
        "fp16",
    )


def test_resolve_repo_explicit_unsupported_pair_raises():
    """Явный --compute-type с несуществующей парой → ошибка."""
    backend = OpenVINOBackend(compute_type_explicit=True)
    with pytest.raises(ValueError, match="недоступна с compute_type='fp16'"):
        backend._resolve_repo("small", "fp16")


def test_resolve_repo_explicit_unknown_model_raises():
    backend = OpenVINOBackend(compute_type_explicit=True)
    with pytest.raises(ValueError, match="не найдена для OpenVINO"):
        backend._resolve_repo("distil-large-v3", "int8")


def test_resolve_repo_implicit_fallback():
    """Неявный compute_type: если int8 недоступен для base, fallback на fp16."""
    backend = OpenVINOBackend(compute_type_explicit=False)
    # base + int8 не существует, но base + fp16 есть
    assert backend._resolve_repo("base", "int8") == (
        "OpenVINO/whisper-base-fp16-ov",
        "fp16",
    )


@pytest.mark.parametrize(
    ("model_name", "expected_compute_type"),
    [("large-v3", "fp16"), ("large-v3-turbo", "int8")],
)
def test_resolve_repo_implicit_large_v3_profiles(model_name, expected_compute_type):
    """Неявный compute_type различает обычную и turbo-модель."""
    backend = OpenVINOBackend(compute_type_explicit=False)

    assert backend._resolve_repo(model_name, "int8") == (
        f"OpenVINO/whisper-{model_name}-{expected_compute_type}-ov",
        expected_compute_type,
    )


def test_resolve_repo_explicit_large_v3_int8_respected():
    """Явный --compute-type int8 для large-v3 → уважается."""
    backend = OpenVINOBackend(compute_type_explicit=True)
    assert backend._resolve_repo("large-v3", "int8") == (
        "OpenVINO/whisper-large-v3-int8-ov",
        "int8",
    )


@pytest.mark.parametrize("compute_type", ["int8", "fp16"])
def test_resolve_repo_large_v3_turbo_quantization(compute_type):
    backend = OpenVINOBackend(compute_type_explicit=True)

    assert backend._resolve_repo("large-v3-turbo", compute_type) == (
        f"OpenVINO/whisper-large-v3-turbo-{compute_type}-ov",
        compute_type,
    )


def test_resolve_repo_large_v3_turbo_unsupported_quantization_raises():
    backend = OpenVINOBackend(compute_type_explicit=True)

    with pytest.raises(
        ValueError,
        match="Доступные варианты: fp16, int8",
    ):
        backend._resolve_repo("large-v3-turbo", "float32")


# === ensure_model_available ===


@patch("local_transcriber.backends.openvino.snapshot_download")
def test_ensure_model_available_cache_hit(mock_download, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "openvino_encoder_model.xml").write_text("<xml/>")
    (model_dir / "openvino_decoder_model.xml").write_text("<xml/>")
    (model_dir / "generation_config.json").write_text('{"alignment_heads": [[1, 2]]}')
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
    (model_dir / "generation_config.json").write_text('{"alignment_heads": [[1, 2]]}')

    mock_download.side_effect = [
        LocalEntryNotFoundError("not cached"),
        str(model_dir),
    ]

    backend = OpenVINOBackend(compute_type_explicit=True)
    statuses: list[str] = []
    result = backend.ensure_model_available("medium", "int8", on_status=statuses.append)

    assert result == str(model_dir)
    assert any("Скачиваю" in s for s in statuses)


@patch("local_transcriber.backends.openvino.snapshot_download")
def test_large_v3_turbo_model_is_resolved_and_created(mock_download, tmp_path):
    model_dir = tmp_path / "large-v3-turbo"
    model_dir.mkdir()
    (model_dir / "openvino_encoder_model.xml").write_text("<xml/>")
    (model_dir / "openvino_decoder_model.xml").write_text("<xml/>")
    (model_dir / "generation_config.json").write_text('{"alignment_heads": [[1, 2]]}')
    mock_download.return_value = str(model_dir)
    mock_ov = MagicMock()

    backend = OpenVINOBackend(ov_device="openvino-cpu", compute_type_explicit=True)
    model_path = backend.ensure_model_available("large-v3-turbo", "int8")
    with patch.dict("sys.modules", {"openvino_genai": mock_ov}):
        backend.create_model(model_path, "openvino-cpu", "int8")

    mock_download.assert_called_once_with(
        "OpenVINO/whisper-large-v3-turbo-int8-ov",
        local_files_only=True,
    )
    mock_ov.WhisperPipeline.assert_called_once_with(
        str(model_dir), "CPU", word_timestamps=True
    )


# === create_model ===


def test_create_model_enables_word_timestamps():
    mock_ov = MagicMock()
    backend = OpenVINOBackend(ov_device="openvino-cpu")

    with patch.dict("sys.modules", {"openvino_genai": mock_ov}):
        backend.create_model("/path/to/model", "openvino-cpu", "int8")

    mock_ov.WhisperPipeline.assert_called_once_with(
        "/path/to/model",
        "CPU",
        word_timestamps=True,
    )


def test_create_model_cpu():
    mock_ov = MagicMock()
    mock_pipeline = MagicMock()
    mock_ov.WhisperPipeline.return_value = mock_pipeline

    backend = OpenVINOBackend(ov_device="openvino-cpu")
    with patch.dict("sys.modules", {"openvino_genai": mock_ov}):
        model = backend.create_model("/path/to/model", "openvino-cpu", "int8")

    mock_ov.WhisperPipeline.assert_called_once_with(
        "/path/to/model", "CPU", word_timestamps=True
    )
    assert model is mock_pipeline
    assert backend.actual_ov_device == "CPU"


def test_create_model_gpu():
    mock_ov = MagicMock()
    mock_pipeline = MagicMock()
    mock_ov.WhisperPipeline.return_value = mock_pipeline

    backend = OpenVINOBackend(ov_device="openvino-gpu")
    with patch.dict("sys.modules", {"openvino_genai": mock_ov}):
        model = backend.create_model("/path/to/model", "openvino-gpu", "fp16")

    mock_ov.WhisperPipeline.assert_called_once_with(
        "/path/to/model", "GPU", word_timestamps=True
    )
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
        patch.dict(
            "sys.modules",
            {"openvino_genai": mock_ov, "openvino": MagicMock(Core=mock_core)},
        ),
    ):
        backend.create_model("/path/to/model", "openvino", "int8")

    mock_ov.WhisperPipeline.assert_called_once_with(
        "/path/to/model", "GPU", word_timestamps=True
    )
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
        patch.dict(
            "sys.modules",
            {"openvino_genai": mock_ov, "openvino": MagicMock(Core=mock_core)},
        ),
    ):
        backend.create_model("/path/to/model", "openvino", "int8")

    mock_ov.WhisperPipeline.assert_called_once_with(
        "/path/to/model", "CPU", word_timestamps=True
    )
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
    mock_result.words = [
        MagicMock(start_ts=0.0, end_ts=3.5, word=" Привет мир"),
        MagicMock(start_ts=3.5, end_ts=7.0, word=" Тестовый сегмент"),
    ]
    mock_model.generate.return_value = mock_result

    raw_audio = np.zeros(16000 * 10, dtype=np.float32)  # 10 секунд

    with patch("faster_whisper.decode_audio", return_value=raw_audio):
        result = backend.transcribe(
            mock_model,
            Path("test.mp3"),
            language="ru",
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


def test_transcribe_maps_word_level_timestamps():
    backend = OpenVINOBackend()
    mock_model = MagicMock()
    raw_word = MagicMock()
    raw_word.start_ts = 0.2
    raw_word.end_ts = 0.8
    raw_word.word = " Привет"
    mock_result = MagicMock()
    mock_result.chunks = []
    mock_result.words = [raw_word]
    mock_model.generate.return_value = mock_result

    with patch(
        "faster_whisper.decode_audio",
        return_value=np.zeros(16_000, dtype=np.float32),
    ):
        result = backend.transcribe(mock_model, Path("test.mp3"), language="ru")

    assert result.words == [Word(start=0.2, end=0.8, text=" Привет")]
    assert mock_model.generate.call_args.kwargs["word_timestamps"] is True


def test_transcribe_keeps_zero_duration_word_timestamp():
    backend = OpenVINOBackend()
    mock_model = MagicMock()
    raw_word = MagicMock(start_ts=1.0, end_ts=1.0, word=" Слово")
    mock_result = MagicMock(chunks=[], words=[raw_word])
    mock_model.generate.return_value = mock_result

    with patch(
        "faster_whisper.decode_audio",
        return_value=np.zeros(16_000, dtype=np.float32),
    ):
        result = backend.transcribe(mock_model, Path("test.mp3"), language="ru")

    assert result.words == [Word(start=1.0, end=1.0, text=" Слово")]


def test_transcribe_rejects_nonempty_result_without_word_timestamps():
    backend = OpenVINOBackend()
    chunk = MagicMock(start_ts=0.0, end_ts=1.0, text=" Текст")
    mock_result = MagicMock(chunks=[chunk], words=None)
    mock_model = MagicMock()
    mock_model.generate.return_value = mock_result

    with (
        patch(
            "faster_whisper.decode_audio",
            return_value=np.zeros(16_000, dtype=np.float32),
        ),
        pytest.raises(RuntimeError, match="пословные таймкоды"),
    ):
        backend.transcribe(mock_model, Path("test.mp3"), language="ru")


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
    assert result.language == UNKNOWN_LANGUAGE
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
    mock_result.words = [MagicMock(start_ts=0.0, end_ts=2.0, word=" Test")]
    mock_model.generate.return_value = mock_result

    raw_audio = np.zeros(16000, dtype=np.float32)
    callback = MagicMock()

    with patch("faster_whisper.decode_audio", return_value=raw_audio):
        backend.transcribe(
            mock_model,
            Path("test.mp3"),
            language="en",
            on_segment=callback,
        )

    callback.assert_called_once()
    seg = callback.call_args[0][0]
    assert isinstance(seg, Segment)
    assert seg.text == " Test"


# === _validate_model_dir ===


def test_validate_model_dir_ok(tmp_path):
    (tmp_path / "openvino_encoder_model.xml").write_text("<xml/>")
    (tmp_path / "openvino_decoder_model.xml").write_text("<xml/>")
    (tmp_path / "generation_config.json").write_text('{"alignment_heads": [[1, 2]]}')
    _validate_model_dir(tmp_path)  # should not raise


def test_validate_model_dir_missing(tmp_path):
    (tmp_path / "openvino_encoder_model.xml").write_text("<xml/>")
    with pytest.raises(ValueError, match="openvino_decoder_model.xml"):
        _validate_model_dir(tmp_path)


def test_validate_model_dir_requires_alignment_heads_for_word_timestamps(tmp_path):
    (tmp_path / "openvino_encoder_model.xml").write_text("<xml/>")
    (tmp_path / "openvino_decoder_model.xml").write_text("<xml/>")
    (tmp_path / "generation_config.json").write_text(
        '{"alignment_heads": []}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="alignment_heads"):
        _validate_model_dir(tmp_path)
