"""Tests for onnx-asr backend."""

import pytest
from local_transcriber.backends.onnx_asr import OnnxAsrBackend, MODEL_ALIASES


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
