from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from local_transcriber.backends.faster_whisper import FasterWhisperBackend
from local_transcriber.types import Word


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bootstrap_only_for_explicit_cuda(monkeypatch, device):
    """CPU-модель не загружает optional CUDA-библиотеки."""
    bootstrap = MagicMock()
    model_factory = MagicMock()
    monkeypatch.setattr(
        "local_transcriber.backends.faster_whisper.ensure_cublas_loadable", bootstrap
    )
    monkeypatch.setattr("faster_whisper.WhisperModel", model_factory)

    model = FasterWhisperBackend().create_model("/model", device, "float32")

    assert model is model_factory.return_value
    assert bootstrap.call_count == (1 if device == "cuda" else 0)
    model_factory.assert_called_once_with(
        "/model", device=device, compute_type="float32", cpu_threads=0
    )


def test_transcribe_returns_canonical_words(tmp_path):
    audio = tmp_path / "audio.wav"
    raw_word = SimpleNamespace(start=0.2, end=0.7, word=" Привет")
    raw_segment = SimpleNamespace(
        start=0.0,
        end=1.0,
        text=" Привет",
        words=[raw_word],
    )
    info = SimpleNamespace(duration=1.0, language="ru", language_probability=0.99)
    model = MagicMock()
    model.transcribe.return_value = (iter([raw_segment]), info)

    result = FasterWhisperBackend().transcribe(model, audio, language="ru")

    assert result.words == [Word(start=0.2, end=0.7, text=" Привет")]
    model.transcribe.assert_called_once_with(
        str(audio),
        language="ru",
        word_timestamps=True,
    )


def test_transcribe_rejects_nonempty_result_without_word_timestamps(tmp_path):
    raw_segment = SimpleNamespace(
        start=0.0,
        end=1.0,
        text=" Текст есть",
        words=None,
    )
    info = SimpleNamespace(duration=1.0, language="ru", language_probability=1.0)
    model = MagicMock()
    model.transcribe.return_value = (iter([raw_segment]), info)

    with pytest.raises(RuntimeError, match="пословные таймкоды"):
        FasterWhisperBackend().transcribe(model, tmp_path / "audio.wav", "ru")


def test_transcribe_rejects_one_nonempty_segment_without_word_timestamps(tmp_path):
    timestamped = SimpleNamespace(
        start=0.0,
        end=1.0,
        text=" Первое",
        words=[SimpleNamespace(start=0.0, end=1.0, word=" Первое")],
    )
    missing = SimpleNamespace(
        start=1.0,
        end=2.0,
        text=" Второе",
        words=None,
    )
    info = SimpleNamespace(duration=2.0, language="ru", language_probability=1.0)
    model = MagicMock()
    model.transcribe.return_value = (iter([timestamped, missing]), info)

    with pytest.raises(RuntimeError, match="пословные таймкоды"):
        FasterWhisperBackend().transcribe(model, tmp_path / "audio.wav", "ru")


def test_runtime_info_reports_versions_and_cuda_devices(monkeypatch):
    """Диагностика называет версии runtime и число CUDA-устройств, не обещая ускорения."""
    monkeypatch.setattr("ctranslate2.get_cuda_device_count", lambda: 0)
    backend = FasterWhisperBackend()
    backend.actual_compute_type = "float32"

    info = backend.runtime_info()

    assert info["engine"] == "faster-whisper"
    assert info["ctranslate2"]
    assert info["faster_whisper"]
    assert info["cuda_devices"] == "0"
    assert info["compute_type"] == "float32"
