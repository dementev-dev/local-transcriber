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
