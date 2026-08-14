import hashlib
import io
import sys
import tarfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from local_transcriber.speaker_diarizer import SpeakerDiarizer, load_speaker_diarizer
from local_transcriber.types import SpeakerInterval


def test_process_returns_sorted_domain_intervals(tmp_path):
    audio = tmp_path / "meeting.mp3"
    raw_result = MagicMock()
    raw_result.sort_by_start_time.return_value = [
        SimpleNamespace(start=0.2, end=1.1, speaker=4),
        SimpleNamespace(start=1.3, end=2.0, speaker=2),
    ]
    engine = MagicMock()
    engine.process.return_value = raw_result
    diarizer = SpeakerDiarizer(engine)
    samples = np.zeros(16_000, dtype=np.float32)

    with patch("faster_whisper.decode_audio", return_value=samples) as decode:
        run = diarizer.process(audio)

    assert run.intervals == [
        SpeakerInterval(start=0.2, end=1.1, cluster=4),
        SpeakerInterval(start=1.3, end=2.0, cluster=2),
    ]
    decode.assert_called_once_with(str(audio), sampling_rate=16_000)
    engine.process.assert_called_once_with(samples)


def test_process_reports_engine_progress(tmp_path):
    raw_result = MagicMock()
    raw_result.sort_by_start_time.return_value = []
    engine = MagicMock()

    def process(samples, callback):
        assert callback(2, 4) == 0
        return raw_result

    engine.process.side_effect = process
    statuses = []

    with patch(
        "faster_whisper.decode_audio",
        return_value=np.zeros(16_000, dtype=np.float32),
    ):
        SpeakerDiarizer(engine).process(
            tmp_path / "meeting.mp3",
            on_status=statuses.append,
        )

    assert "Определяю говорящих... 2 / 4" in statuses


def test_load_speaker_diarizer_uses_verified_cache_and_calibrated_config(
    tmp_path, monkeypatch
):
    segmentation = tmp_path / "pyannote-segmentation-3.0.onnx"
    embedding = tmp_path / "wespeaker_en_voxceleb_resnet34_LM.onnx"
    segmentation.write_bytes(b"segmentation")
    embedding.write_bytes(b"embedding")
    monkeypatch.setattr(
        "local_transcriber.speaker_diarizer._SEGMENTATION_SHA256",
        hashlib.sha256(segmentation.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        "local_transcriber.speaker_diarizer._EMBEDDING_SHA256",
        hashlib.sha256(embedding.read_bytes()).hexdigest(),
    )

    captured = {}

    def config_factory(**kwargs):
        config = SimpleNamespace(**kwargs, validate=lambda: True)
        captured["config"] = config
        return config

    engine = SimpleNamespace(sample_rate=16_000)
    sherpa = SimpleNamespace(
        OfflineSpeakerSegmentationPyannoteModelConfig=lambda **kwargs: SimpleNamespace(
            **kwargs
        ),
        OfflineSpeakerSegmentationModelConfig=lambda **kwargs: SimpleNamespace(
            **kwargs
        ),
        SpeakerEmbeddingExtractorConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        FastClusteringConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        OfflineSpeakerDiarizationConfig=config_factory,
        OfflineSpeakerDiarization=lambda config: engine,
    )

    with (
        patch("huggingface_hub.cached_assets_path", return_value=tmp_path),
        patch.dict(sys.modules, {"sherpa_onnx": sherpa}),
        patch("httpx.stream", side_effect=AssertionError("network is not expected")),
    ):
        diarizer = load_speaker_diarizer(speakers=None, threads=0)

    config = captured["config"]
    assert config.clustering.num_clusters == -1
    assert config.clustering.threshold == 0.89
    assert config.min_duration_on == 0.3
    assert config.min_duration_off == 0.5
    assert not hasattr(config.segmentation, "num_threads")
    assert not hasattr(config.embedding, "num_threads")
    assert isinstance(diarizer, SpeakerDiarizer)


def test_load_speaker_diarizer_downloads_and_verifies_missing_models(
    tmp_path, monkeypatch
):
    segmentation_bytes = b"downloaded segmentation"
    embedding_bytes = b"downloaded embedding"
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:bz2") as archive:
        member = tarfile.TarInfo("sherpa-onnx-pyannote-segmentation-3-0/model.onnx")
        member.size = len(segmentation_bytes)
        archive.addfile(member, io.BytesIO(segmentation_bytes))

    monkeypatch.setattr(
        "local_transcriber.speaker_diarizer._SEGMENTATION_SHA256",
        hashlib.sha256(segmentation_bytes).hexdigest(),
    )
    monkeypatch.setattr(
        "local_transcriber.speaker_diarizer._EMBEDDING_SHA256",
        hashlib.sha256(embedding_bytes).hexdigest(),
    )

    class FakeResponse:
        def __init__(self, content):
            self.content = content

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            yield self.content

    requested_urls = []

    def fake_stream(method, url, **kwargs):
        requested_urls.append(url)
        content = (
            archive_buffer.getvalue() if "segmentation" in url else embedding_bytes
        )
        return FakeResponse(content)

    config = SimpleNamespace(validate=lambda: True)
    engine = SimpleNamespace(sample_rate=16_000)
    sherpa = SimpleNamespace(
        OfflineSpeakerSegmentationPyannoteModelConfig=lambda **kwargs: SimpleNamespace(
            **kwargs
        ),
        OfflineSpeakerSegmentationModelConfig=lambda **kwargs: SimpleNamespace(
            **kwargs
        ),
        SpeakerEmbeddingExtractorConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        FastClusteringConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        OfflineSpeakerDiarizationConfig=lambda **kwargs: config,
        OfflineSpeakerDiarization=lambda actual_config: engine,
    )

    with (
        patch("huggingface_hub.cached_assets_path", return_value=tmp_path),
        patch.dict(sys.modules, {"sherpa_onnx": sherpa}),
        patch("httpx.stream", side_effect=fake_stream),
    ):
        load_speaker_diarizer(speakers=2, threads=4)

    assert (
        tmp_path / "pyannote-segmentation-3.0.onnx"
    ).read_bytes() == segmentation_bytes
    assert (
        tmp_path / "wespeaker_en_voxceleb_resnet34_LM.onnx"
    ).read_bytes() == embedding_bytes
    assert len(requested_urls) == 2
    assert list(tmp_path.glob("*.tmp")) == []


def test_load_speaker_diarizer_keeps_corrupt_cache_when_download_is_invalid(
    tmp_path, monkeypatch
):
    segmentation = tmp_path / "pyannote-segmentation-3.0.onnx"
    segmentation.write_bytes(b"existing corrupt model")
    monkeypatch.setattr(
        "local_transcriber.speaker_diarizer._SEGMENTATION_SHA256",
        hashlib.sha256(b"expected model").hexdigest(),
    )

    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:bz2") as archive:
        payload = b"wrong downloaded model"
        member = tarfile.TarInfo("sherpa-onnx-pyannote-segmentation-3-0/model.onnx")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            yield archive_buffer.getvalue()

    with (
        patch("huggingface_hub.cached_assets_path", return_value=tmp_path),
        patch("httpx.stream", return_value=FakeResponse()),
        pytest.raises(RuntimeError, match="Контрольная сумма"),
    ):
        load_speaker_diarizer(speakers=None)

    assert segmentation.read_bytes() == b"existing corrupt model"
    assert list(tmp_path.glob("*.tmp")) == []
