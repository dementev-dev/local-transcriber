"""Tests for onnx-asr backend."""

import pytest
from local_transcriber.backends.onnx_asr import OnnxAsrBackend, MODEL_ALIASES


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
                           cpu_preprocessing=None, **kwargs):
            calls.append({
                "model": model, "path": path, "quantization": quantization,
                "cpu_preprocessing": cpu_preprocessing,
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
        assert calls[0]["cpu_preprocessing"] is True
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
