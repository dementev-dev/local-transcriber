"""Тесты для Parakeet бэкенда."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from local_transcriber.backends.parakeet import (
    MODEL_REPO,
    ONNX_ASR_MODEL_ALIAS,
    ParakeetBackend,
    SUPPORTED_COMPUTE_TYPES,
    SUPPORTED_MODEL_NAMES,
)


def _create_parakeet_model_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    return path


# === ensure_model_available: валидация входов ===


def test_ensure_model_available_accepts_canonical_name(tmp_path):
    backend = ParakeetBackend()
    model_dir = _create_parakeet_model_dir(tmp_path / "parakeet")
    with patch(
        "local_transcriber.backends.parakeet.snapshot_download",
        return_value=str(model_dir),
    ) as mock_dl:
        result = backend.ensure_model_available("parakeet-tdt-0.6b-v3", "int8")
    mock_dl.assert_called_once()
    args, kwargs = mock_dl.call_args
    assert (args and args[0] == MODEL_REPO) or kwargs.get("repo_id") == MODEL_REPO
    assert result == str(model_dir)


def test_ensure_model_available_accepts_alias(tmp_path):
    backend = ParakeetBackend()
    model_dir = _create_parakeet_model_dir(tmp_path / "parakeet")
    with patch(
        "local_transcriber.backends.parakeet.snapshot_download",
        return_value=str(model_dir),
    ) as mock_dl:
        result = backend.ensure_model_available("parakeet", "int8")
    args, kwargs = mock_dl.call_args
    assert (args and args[0] == MODEL_REPO) or kwargs.get("repo_id") == MODEL_REPO
    assert result == str(model_dir)


def test_ensure_model_available_rejects_unknown_model():
    backend = ParakeetBackend()
    with pytest.raises(ValueError, match="parakeet-tdt-0.6b-v3"):
        backend.ensure_model_available("medium", "int8")


def test_ensure_model_available_rejects_unknown_compute_type():
    backend = ParakeetBackend()
    with pytest.raises(ValueError, match="int8 или float32"):
        backend.ensure_model_available("parakeet-tdt-0.6b-v3", "float16")


def test_ensure_model_available_sets_actual_compute_type(tmp_path):
    backend = ParakeetBackend()
    model_dir = _create_parakeet_model_dir(tmp_path / "parakeet")
    with patch(
        "local_transcriber.backends.parakeet.snapshot_download",
        return_value=str(model_dir),
    ):
        backend.ensure_model_available("parakeet-tdt-0.6b-v3", "int8")
    assert backend.actual_compute_type == "int8"


def test_ensure_model_available_validates_model_dir(tmp_path):
    """Если snapshot_download вернул путь без config.json → ValueError."""
    backend = ParakeetBackend()
    bad_dir = tmp_path / "broken"
    bad_dir.mkdir()
    with patch(
        "local_transcriber.backends.parakeet.snapshot_download",
        return_value=str(bad_dir),
    ):
        with pytest.raises(ValueError, match="Неполная"):
            backend.ensure_model_available("parakeet-tdt-0.6b-v3", "int8")


# === create_model ===


def test_create_model_maps_int8_to_quantization(tmp_path):
    backend = ParakeetBackend()
    model_dir = _create_parakeet_model_dir(tmp_path / "parakeet")
    fake_asr = MagicMock(name="asr")
    fake_asr.with_vad.return_value = MagicMock(name="asr_with_vad")

    with (
        patch("local_transcriber.backends.parakeet.onnx_asr") as mock_ox,
    ):
        mock_ox.load_model.return_value = fake_asr
        mock_ox.load_vad.return_value = MagicMock(name="vad")

        backend.create_model(str(model_dir), "parakeet-cpu", "int8")

    load_model_kwargs = mock_ox.load_model.call_args.kwargs
    assert load_model_kwargs.get("quantization") == "int8"
    assert mock_ox.load_model.call_args.args[0] == ONNX_ASR_MODEL_ALIAS
    assert load_model_kwargs.get("path") == str(model_dir)


def test_create_model_maps_float32_to_none_quantization(tmp_path):
    backend = ParakeetBackend()
    model_dir = _create_parakeet_model_dir(tmp_path / "parakeet")
    fake_asr = MagicMock()
    fake_asr.with_vad.return_value = MagicMock()

    with patch("local_transcriber.backends.parakeet.onnx_asr") as mock_ox:
        mock_ox.load_model.return_value = fake_asr
        mock_ox.load_vad.return_value = MagicMock()

        backend.create_model(str(model_dir), "parakeet-cpu", "float32")

    assert mock_ox.load_model.call_args.kwargs.get("quantization") is None


def test_create_model_wraps_with_silero_vad(tmp_path):
    backend = ParakeetBackend()
    model_dir = _create_parakeet_model_dir(tmp_path / "parakeet")
    fake_asr = MagicMock()
    wrapped = MagicMock(name="asr_with_vad")
    fake_asr.with_vad.return_value = wrapped

    with patch("local_transcriber.backends.parakeet.onnx_asr") as mock_ox:
        mock_ox.load_model.return_value = fake_asr
        fake_vad = MagicMock(name="silero_vad")
        mock_ox.load_vad.return_value = fake_vad

        returned = backend.create_model(str(model_dir), "parakeet-cpu", "int8")

    mock_ox.load_vad.assert_called_once()
    assert mock_ox.load_vad.call_args.kwargs.get("model") == "silero" \
        or mock_ox.load_vad.call_args.args[0] == "silero"
    fake_asr.with_vad.assert_called_once_with(vad=fake_vad)
    assert returned is wrapped


def test_create_model_cpu_threads_via_sess_options(tmp_path):
    """cpu_threads > 0 → SessionOptions.intra_op_num_threads, НЕ provider_options."""
    backend = ParakeetBackend()
    model_dir = _create_parakeet_model_dir(tmp_path / "parakeet")
    fake_asr = MagicMock()
    fake_asr.with_vad.return_value = MagicMock()

    with patch("local_transcriber.backends.parakeet.onnx_asr") as mock_ox:
        mock_ox.load_model.return_value = fake_asr
        mock_ox.load_vad.return_value = MagicMock()

        backend.create_model(str(model_dir), "parakeet-cpu", "int8", cpu_threads=4)

    kwargs = mock_ox.load_model.call_args.kwargs
    assert "sess_options" in kwargs
    assert kwargs["sess_options"].intra_op_num_threads == 4
    assert "provider_options" not in kwargs


def test_create_model_zero_threads_omits_sess_options(tmp_path):
    backend = ParakeetBackend()
    model_dir = _create_parakeet_model_dir(tmp_path / "parakeet")
    fake_asr = MagicMock()
    fake_asr.with_vad.return_value = MagicMock()

    with patch("local_transcriber.backends.parakeet.onnx_asr") as mock_ox:
        mock_ox.load_model.return_value = fake_asr
        mock_ox.load_vad.return_value = MagicMock()

        backend.create_model(str(model_dir), "parakeet-cpu", "int8", cpu_threads=0)

    kwargs = mock_ox.load_model.call_args.kwargs
    assert "sess_options" not in kwargs  # 0 = дефолт библиотеки, не трогаем
