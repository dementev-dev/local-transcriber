"""Tests for onnx-asr backend."""

import pytest
from pathlib import Path

from local_transcriber.backends.onnx_asr import OnnxAsrBackend, MODEL_ALIASES
from local_transcriber.types import Segment, TranscribeResult


class FakeVadSegment:
    """Mimics onnx-asr SegmentResult."""

    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


class TestEnsureModelAvailable:
    def test_returns_model_id_for_gigaam(self):
        backend = OnnxAsrBackend()
        result = backend.ensure_model_available("gigaam-v3", "int8")
        assert result == "gigaam-v3-ctc"

    def test_returns_model_id_for_parakeet(self):
        backend = OnnxAsrBackend()
        result = backend.ensure_model_available("parakeet-v3", "fp16")
        assert result == "nemo-parakeet-tdt-0.6b-v3"

    def test_stores_compute_type(self):
        backend = OnnxAsrBackend()
        backend.ensure_model_available("gigaam-v3", "float32")
        assert backend._resolved_model_id == "gigaam-v3-ctc"
        assert backend.actual_compute_type == "float32"


class TestCreateModel:
    def test_calls_load_model_with_correct_args(self, monkeypatch):
        """Verify create_model passes correct args to onnx_asr.load_model."""
        calls = []

        def fake_load_model(model=None, path=None, quantization=None,
                           **kwargs):
            calls.append({
                "model": model, "path": path, "quantization": quantization,
            })
            return FakeAsrAdapter()

        class FakeAsrAdapter:
            def with_vad(self, vad):
                return self

        monkeypatch.setattr("onnx_asr.load_model", fake_load_model)

        backend = OnnxAsrBackend()
        backend.actual_compute_type = "int8"
        model = backend.create_model("gigaam-v3-ctc", "onnx", "int8")

        assert len(calls) == 1
        assert calls[0]["quantization"] == "int8"
        assert model is not None

    def test_loads_silero_vad(self, monkeypatch):
        """Verify Silero VAD is loaded and attached to model."""
        vad_calls = []

        def fake_load_vad(model, **kwargs):
            vad_calls.append(model)
            return "fake_vad"

        def fake_load_model(**kwargs):
            return FakeAsrAdapter()

        class FakeAsrAdapter:
            def with_vad(self, vad):
                self._vad = vad
                return self

        monkeypatch.setattr("onnx_asr.load_model", fake_load_model)
        monkeypatch.setattr("onnx_asr.load_vad", fake_load_vad)

        backend = OnnxAsrBackend()
        model = backend.create_model("gigaam-v3-ctc", "onnx", "int8")

        assert vad_calls == ["silero"]

    def test_fp16_compute_type(self, monkeypatch):
        """Verify fp16 compute_type is passed through."""
        calls = []

        def fake_load_model(model=None, quantization=None, **kwargs):
            calls.append(quantization)
            return FakeAsrAdapter()

        class FakeAsrAdapter:
            def with_vad(self, vad):
                return self

        monkeypatch.setattr("onnx_asr.load_model", fake_load_model)
        monkeypatch.setattr("onnx_asr.load_vad", lambda model, **kw: None)

        backend = OnnxAsrBackend()
        backend.create_model("parakeet-v3", "onnx", "fp16")

        assert calls == ["fp16"]


class TestTranscribe:
    def test_transcribe_collects_segments(self, monkeypatch, tmp_path):
        """Verify transcribe maps VAD segments to project Segments."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"fake audio")

        audio_samples = [0.0] * 16000  # 1 second of silence

        def fake_decode_audio(path, sampling_rate=16000):
            import numpy as np
            return np.array(audio_samples, dtype=np.float32)

        class FakeModel:
            def recognize(self, waveform, sample_rate, language=None):
                yield FakeVadSegment(0.0, 1.0, "hello")
                yield FakeVadSegment(1.0, 2.5, "world")

        monkeypatch.setattr("faster_whisper.decode_audio", fake_decode_audio)

        backend = OnnxAsrBackend()
        backend.actual_compute_type = "int8"
        result = backend.transcribe(
            FakeModel(), wav_file, language=None,
        )

        assert isinstance(result, TranscribeResult)
        assert len(result.segments) == 2
        assert result.segments[0] == Segment(start=0.0, end=1.0, text="hello")
        assert result.segments[1] == Segment(start=1.0, end=2.5, text="world")
        assert result.duration == 1.0  # 16000 samples / 16000 Hz

    def test_transcribe_calls_on_segment(self, monkeypatch, tmp_path):
        """Verify on_segment callback is invoked per segment."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"fake audio")

        def fake_decode_audio(path, sampling_rate=16000):
            import numpy as np
            return np.array([0.0] * 16000, dtype=np.float32)

        segments_captured = []

        class FakeModel:
            def recognize(self, waveform, sample_rate, language=None):
                yield FakeVadSegment(0.0, 2.0, "one")
                yield FakeVadSegment(2.0, 4.0, "two")

        monkeypatch.setattr("faster_whisper.decode_audio", fake_decode_audio)

        backend = OnnxAsrBackend()
        result = backend.transcribe(
            FakeModel(), wav_file, language=None,
            on_segment=lambda s: segments_captured.append(s),
        )

        assert len(segments_captured) == 2
        assert segments_captured[0].text == "one"
        assert segments_captured[1].text == "two"

    def test_transcribe_passes_language(self, monkeypatch, tmp_path):
        """Verify language is passed to recognize()."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"fake audio")

        def fake_decode_audio(path, sampling_rate=16000):
            import numpy as np
            return np.array([0.0] * 16000, dtype=np.float32)

        lang_received = []

        class FakeModel:
            def recognize(self, waveform, sample_rate, language=None):
                lang_received.append(language)
                yield FakeVadSegment(0.0, 1.0, "text")

        monkeypatch.setattr("faster_whisper.decode_audio", fake_decode_audio)

        backend = OnnxAsrBackend()
        backend.transcribe(FakeModel(), wav_file, language="ru")

        assert lang_received == ["ru"]

    def test_transcribe_empty_audio(self, monkeypatch, tmp_path):
        """Verify zero segments for silent audio."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"fake audio")

        def fake_decode_audio(path, sampling_rate=16000):
            import numpy as np
            return np.array([0.0] * 16000, dtype=np.float32)

        class FakeModel:
            def recognize(self, waveform, sample_rate, language=None):
                # No segments yielded
                if False:
                    yield

        monkeypatch.setattr("faster_whisper.decode_audio", fake_decode_audio)

        backend = OnnxAsrBackend()
        result = backend.transcribe(FakeModel(), wav_file, language=None)

        assert len(result.segments) == 0
        assert result.language == "unknown"
        assert result.duration == 1.0


class TestBackendRegistration:
    def test_get_backend_returns_onnx_backend(self):
        from local_transcriber.backends import get_backend
        backend = get_backend("onnx")
        assert isinstance(backend, OnnxAsrBackend)


class TestModelAliases:
    def test_gigaam_v3_resolves(self):
        backend = OnnxAsrBackend()
        result = backend._resolve_model("gigaam-v3")
        assert result == "gigaam-v3-ctc"

    def test_parakeet_v3_resolves(self):
        backend = OnnxAsrBackend()
        result = backend._resolve_model("parakeet-v3")
        assert result == "nemo-parakeet-tdt-0.6b-v3"

    def test_raw_name_passes_through(self):
        backend = OnnxAsrBackend()
        result = backend._resolve_model("nemo-canary-1b-v2")
        assert result == "nemo-canary-1b-v2"

    def test_unknown_alias_raises(self):
        backend = OnnxAsrBackend()
        with pytest.raises(ValueError, match="Неподдерживаемая модель"):
            backend._resolve_model("nonexistent-model")
