"""Tests for onnx-asr backend."""

import warnings
from unittest.mock import MagicMock

import pytest

from local_transcriber.backends.onnx_asr import OnnxAsrBackend
from local_transcriber.types import UNKNOWN_LANGUAGE, Segment, TranscribeResult, Word


class FakeVadSegment:
    """Mimics onnx-asr SegmentResult."""

    def __init__(self, start, end, text, tokens=None, timestamps=None):
        self.start = start
        self.end = end
        self.text = text
        self.tokens = [f" {text}"] if tokens is None else tokens
        self.timestamps = [0.0] if timestamps is None else timestamps


class TestEnsureModelAvailable:
    def test_returns_model_id_for_gigaam(self):
        backend = OnnxAsrBackend()
        result = backend.ensure_model_available("gigaam-v3", "int8")
        assert result == "gigaam-v3-ctc"

    def test_returns_model_id_for_parakeet(self):
        backend = OnnxAsrBackend()
        result = backend.ensure_model_available("parakeet-v3", "int8")
        assert result == "nemo-parakeet-tdt-0.6b-v3"

    def test_stores_compute_type(self):
        backend = OnnxAsrBackend()
        backend.ensure_model_available("gigaam-v3", "float32")
        assert backend._resolved_model_id == "gigaam-v3-ctc"
        assert backend.actual_compute_type == "float32"

    @pytest.mark.parametrize(
        "model_name",
        [
            "gigaam-multilingual-ctc",
            "gigaam-multilingual-large-ctc",
            "gigaam-v3-e2e-ctc",
            "gigaam-v3-e2e-rnnt",
        ],
    )
    def test_explicit_unavailable_compute_type_is_rejected(self, model_name):
        backend = OnnxAsrBackend(compute_type_explicit=True)

        with pytest.raises(ValueError, match="недоступна с compute_type='fp16'"):
            backend.ensure_model_available(model_name, "fp16")

    def test_implicit_unavailable_compute_type_falls_back_and_reports(
        self, monkeypatch
    ):
        quantizations = []
        statuses = []

        class FakeAsrAdapter:
            def with_vad(self, vad):
                return self

            def with_timestamps(self):
                return self

        def fake_load_model(*, model, quantization, providers):
            quantizations.append(quantization)
            return FakeAsrAdapter()

        monkeypatch.setattr("onnx_asr.load_model", fake_load_model)
        monkeypatch.setattr("onnx_asr.load_vad", lambda model, **kwargs: None)

        backend = OnnxAsrBackend(compute_type_explicit=False)
        model_id = backend.ensure_model_available(
            "gigaam-v3-e2e-ctc",
            "fp16",
            on_status=statuses.append,
        )
        backend.create_model(model_id, "onnx", "fp16")

        assert backend.actual_compute_type == "int8"
        assert quantizations == ["int8"]
        assert statuses == [
            "Модель gigaam-v3-e2e-ctc недоступна с compute_type=fp16; использую int8."
        ]


class TestCreateModel:
    def test_cpu_provider_is_explicit_for_asr_and_vad(self, monkeypatch):
        """Доступность CoreML/CUDA не меняет исполнение ASR и VAD."""
        monkeypatch.setattr(
            "onnxruntime.get_available_providers",
            lambda: ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        load_asr = MagicMock()
        load_vad = MagicMock()
        monkeypatch.setattr("onnx_asr.load_model", load_asr)
        monkeypatch.setattr("onnx_asr.load_vad", load_vad)

        OnnxAsrBackend().create_model("gigaam-v3-e2e-rnnt", "onnx", "int8")

        load_asr.assert_called_once_with(
            model="gigaam-v3-e2e-rnnt",
            quantization="int8",
            providers=["CPUExecutionProvider"],
        )
        load_vad.assert_called_once_with("silero", providers=["CPUExecutionProvider"])

    def test_thread_budget_reaches_asr_and_vad_sessions(self, monkeypatch):
        """--threads задаёт intra_op_num_threads обеим сессиям через SessionOptions."""
        load_asr = MagicMock()
        load_vad = MagicMock()
        monkeypatch.setattr("onnx_asr.load_model", load_asr)
        monkeypatch.setattr("onnx_asr.load_vad", load_vad)

        backend = OnnxAsrBackend()
        backend.create_model("gigaam-v3-e2e-rnnt", "onnx", "int8", cpu_threads=4)

        asr_options = load_asr.call_args.kwargs["sess_options"]
        vad_options = load_vad.call_args.kwargs["sess_options"]
        assert asr_options.intra_op_num_threads == 4
        assert vad_options.intra_op_num_threads == 4
        assert backend.runtime_info()["intra_op_threads"] == "4"

    def test_zero_thread_budget_keeps_library_session_defaults(self, monkeypatch):
        """0 не передаёт SessionOptions: потоки остаются на усмотрение onnxruntime."""
        load_asr = MagicMock()
        load_vad = MagicMock()
        monkeypatch.setattr("onnx_asr.load_model", load_asr)
        monkeypatch.setattr("onnx_asr.load_vad", load_vad)

        backend = OnnxAsrBackend()
        backend.create_model("gigaam-v3-e2e-rnnt", "onnx", "int8", cpu_threads=0)

        assert "sess_options" not in load_asr.call_args.kwargs
        assert "sess_options" not in load_vad.call_args.kwargs
        assert backend.runtime_info()["intra_op_threads"] == "по умолчанию"

    def test_wraps_vad_model_with_timestamps(self, monkeypatch):
        timestamped_model = object()

        class FakeVadAdapter:
            def with_timestamps(self):
                return timestamped_model

        class FakeAsrAdapter:
            def with_vad(self, vad):
                return FakeVadAdapter()

        monkeypatch.setattr("onnx_asr.load_model", lambda **kwargs: FakeAsrAdapter())
        monkeypatch.setattr("onnx_asr.load_vad", lambda model, **kwargs: object())

        model = OnnxAsrBackend().create_model("gigaam-v3-e2e-rnnt", "onnx", "int8")

        assert model is timestamped_model

    def test_calls_load_model_with_correct_args(self, monkeypatch):
        """Verify create_model passes correct args to onnx_asr.load_model."""
        calls = []

        def fake_load_model(model=None, path=None, quantization=None, **kwargs):
            calls.append(
                {
                    "model": model,
                    "path": path,
                    "quantization": quantization,
                }
            )
            return FakeAsrAdapter()

        class FakeAsrAdapter:
            def with_vad(self, vad):
                return self

            def with_timestamps(self):
                return self

        monkeypatch.setattr("onnx_asr.load_model", fake_load_model)
        monkeypatch.setattr("onnx_asr.load_vad", lambda model, **kwargs: None)

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

            def with_timestamps(self):
                return self

        monkeypatch.setattr("onnx_asr.load_model", fake_load_model)
        monkeypatch.setattr("onnx_asr.load_vad", fake_load_vad)

        backend = OnnxAsrBackend()
        backend.create_model("gigaam-v3-ctc", "onnx", "int8")

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

            def with_timestamps(self):
                return self

        monkeypatch.setattr("onnx_asr.load_model", fake_load_model)
        monkeypatch.setattr("onnx_asr.load_vad", lambda model, **kw: None)

        backend = OnnxAsrBackend()
        backend.create_model("parakeet-v3", "onnx", "fp16")

        assert calls == ["fp16"]

    def test_float32_maps_to_none(self, monkeypatch):
        """compute_type='float32' маппится в quantization=None (unquantized).

        onnx-asr использует quantization как суффикс файла; для float32 нужен None,
        строка "float32" приведёт к попытке загрузить несуществующий файл.
        """
        calls = []

        def fake_load_model(model=None, quantization="MISSING", **kwargs):
            calls.append(quantization)
            return FakeAsrAdapter()

        class FakeAsrAdapter:
            def with_vad(self, vad):
                return self

            def with_timestamps(self):
                return self

        monkeypatch.setattr("onnx_asr.load_model", fake_load_model)
        monkeypatch.setattr("onnx_asr.load_vad", lambda model, **kw: None)

        backend = OnnxAsrBackend()
        backend.create_model("gigaam-v3-ctc", "onnx", "float32")

        assert calls == [None]

    def test_fp32_maps_to_none(self, monkeypatch):
        """compute_type='fp32' тоже маппится в quantization=None."""
        calls = []

        def fake_load_model(model=None, quantization="MISSING", **kwargs):
            calls.append(quantization)
            return FakeAsrAdapter()

        class FakeAsrAdapter:
            def with_vad(self, vad):
                return self

            def with_timestamps(self):
                return self

        monkeypatch.setattr("onnx_asr.load_model", fake_load_model)
        monkeypatch.setattr("onnx_asr.load_vad", lambda model, **kw: None)

        backend = OnnxAsrBackend()
        backend.create_model("gigaam-v3-ctc", "onnx", "fp32")

        assert calls == [None]

    def test_float16_alias_maps_to_fp16(self, monkeypatch):
        """compute_type='float16' (CUDA-naming) маппится в onnx-asr 'fp16'."""
        calls = []

        def fake_load_model(model=None, quantization=None, **kwargs):
            calls.append(quantization)
            return FakeAsrAdapter()

        class FakeAsrAdapter:
            def with_vad(self, vad):
                return self

            def with_timestamps(self):
                return self

        monkeypatch.setattr("onnx_asr.load_model", fake_load_model)
        monkeypatch.setattr("onnx_asr.load_vad", lambda model, **kw: None)

        backend = OnnxAsrBackend()
        backend.create_model("gigaam-v3-ctc", "onnx", "float16")

        assert calls == ["fp16"]

    def test_unknown_compute_type_raises(self, monkeypatch):
        """Неподдерживаемый compute_type → ValueError, не silent fallback."""
        monkeypatch.setattr("onnx_asr.load_model", lambda **kw: None)
        monkeypatch.setattr("onnx_asr.load_vad", lambda model, **kw: None)

        backend = OnnxAsrBackend()
        with pytest.raises(ValueError, match="Неподдерживаемый compute_type"):
            backend.create_model("gigaam-v3-ctc", "onnx", "int8_float32")


class TestTranscribe:
    @pytest.mark.parametrize(
        ("model_name", "language", "expects_warning"),
        [
            ("gigaam-v3-e2e-rnnt", "en", True),
            ("gigaam-v3-e2e-rnnt", "ru", False),
            ("gigaam-multilingual-ctc", "en", False),
        ],
    )
    def test_warns_when_language_is_not_supported(
        self, monkeypatch, tmp_path, model_name, language, expects_warning
    ):
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"fake audio")

        monkeypatch.setattr(
            "faster_whisper.decode_audio",
            lambda path, sampling_rate=16000: [0.0] * 16000,
        )

        class FakeModel:
            def recognize(self, waveform, sample_rate, language=None):
                return iter(())

        backend = OnnxAsrBackend()
        backend.ensure_model_available(model_name, "int8")

        if expects_warning:
            with pytest.warns(
                UserWarning,
                match=(
                    r"Язык 'en'.*--device openvino-cpu --model medium.*"
                    r"--device cpu --model medium.*"
                    r"--device cuda --model medium"
                ),
            ):
                backend.transcribe(FakeModel(), wav_file, language=language)
        else:
            with warnings.catch_warnings(record=True) as caught:
                backend.transcribe(FakeModel(), wav_file, language=language)
            assert caught == []

    def test_auto_language_uses_single_supported_model_language(
        self, monkeypatch, tmp_path
    ):
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"fake audio")
        monkeypatch.setattr(
            "faster_whisper.decode_audio",
            lambda path, sampling_rate=16000: [0.0] * 16000,
        )

        class FakeModel:
            def recognize(self, waveform, sample_rate, language=None):
                return iter(())

        backend = OnnxAsrBackend()
        backend.ensure_model_available("gigaam-v3-e2e-rnnt", "int8")

        result = backend.transcribe(FakeModel(), wav_file, language=None)

        assert result.language == "ru"
        assert result.language_probability == 0.0

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
            FakeModel(),
            wav_file,
            language=None,
        )

        assert isinstance(result, TranscribeResult)
        assert len(result.segments) == 2
        assert result.segments[0] == Segment(start=0.0, end=1.0, text="hello")
        assert result.segments[1] == Segment(start=1.0, end=2.5, text="world")
        assert result.duration == 1.0  # 16000 samples / 16000 Hz

    def test_transcribe_converts_vad_token_timestamps_to_global_words(
        self, monkeypatch, tmp_path
    ):
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"fake audio")
        monkeypatch.setattr(
            "faster_whisper.decode_audio",
            lambda path, sampling_rate=16000: [0.0] * 16_000,
        )

        timestamped_segment = FakeVadSegment(
            10.0,
            12.0,
            "Привет, мир",
            tokens=[" ", "П", "р", "и", "в", "е", "т", ",", " ", "м", "и", "р"],
            timestamps=[0.0, 0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.5, 0.6, 0.6, 0.7],
        )

        class FakeModel:
            def recognize(self, waveform, sample_rate, language=None):
                yield timestamped_segment

        result = OnnxAsrBackend().transcribe(FakeModel(), wav_file, language="ru")

        assert result.words == [
            Word(start=10.0, end=10.5, text=" Привет,"),
            Word(start=10.5, end=12.0, text=" мир"),
        ]
        assert "".join(word.text for word in result.words).strip() == "Привет, мир"

    def test_transcribe_rejects_nonempty_segment_without_token_timestamps(
        self, monkeypatch, tmp_path
    ):
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"fake audio")
        monkeypatch.setattr(
            "faster_whisper.decode_audio",
            lambda path, sampling_rate=16000: [0.0] * 16_000,
        )
        segment = FakeVadSegment(0.0, 1.0, "Текст")
        segment.tokens = None
        segment.timestamps = None

        class FakeModel:
            def recognize(self, waveform, sample_rate, language=None):
                yield segment

        with pytest.raises(RuntimeError, match="пословные таймкоды"):
            OnnxAsrBackend().transcribe(FakeModel(), wav_file, language="ru")

    def test_transcribe_keeps_words_with_equal_emission_timestamps(
        self, monkeypatch, tmp_path
    ):
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"fake audio")
        monkeypatch.setattr(
            "faster_whisper.decode_audio",
            lambda path, sampling_rate=16000: [0.0] * 16_000,
        )
        segment = FakeVadSegment(
            10.0,
            12.0,
            "Да нет потом",
            tokens=[" ", "Да", " ", "нет", " ", "потом"],
            timestamps=[0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
        )

        class FakeModel:
            def recognize(self, waveform, sample_rate, language=None):
                yield segment

        result = OnnxAsrBackend().transcribe(FakeModel(), wav_file, language="ru")

        assert [word.text for word in result.words] == [" Да", " нет", " потом"]
        assert [(word.start, word.end) for word in result.words] == [
            (10.0, 10.5),
            (10.0, 10.5),
            (10.5, 12.0),
        ]
        assert "".join(word.text for word in result.words).strip() == segment.text

    def test_transcribe_keeps_word_clamped_to_segment_end(self, monkeypatch, tmp_path):
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"fake audio")
        monkeypatch.setattr(
            "faster_whisper.decode_audio",
            lambda path, sampling_rate=16000: [0.0] * 16_000,
        )
        segment = FakeVadSegment(
            10.0,
            12.0,
            "Позднее",
            tokens=[" ", "Позднее"],
            timestamps=[2.0, 2.0],
        )

        class FakeModel:
            def recognize(self, waveform, sample_rate, language=None):
                yield segment

        result = OnnxAsrBackend().transcribe(FakeModel(), wav_file, language="ru")

        assert result.words == [Word(start=12.0, end=12.0, text=" Позднее")]

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
        backend.transcribe(
            FakeModel(),
            wav_file,
            language=None,
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
        assert result.language == UNKNOWN_LANGUAGE
        assert result.duration == 1.0

    def test_transcribe_skips_zero_length_vad_segments(self, monkeypatch, tmp_path):
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"fake audio")

        def fake_decode_audio(path, sampling_rate=16000):
            import numpy as np

            return np.array([0.0] * 16000, dtype=np.float32)

        class FakeModel:
            def recognize(self, waveform, sample_rate, language=None):
                yield FakeVadSegment(0.5, 0.5, "нулевой")
                yield FakeVadSegment(0.75, 0.5, "обратный")
                yield FakeVadSegment(0.5, 1.0, "валидный")

        monkeypatch.setattr("faster_whisper.decode_audio", fake_decode_audio)

        result = OnnxAsrBackend().transcribe(FakeModel(), wav_file, language=None)

        assert result.segments == [Segment(start=0.5, end=1.0, text="валидный")]


class TestBackendRegistration:
    def test_get_backend_returns_onnx_backend(self):
        from local_transcriber.backends import get_backend

        backend = get_backend("onnx")
        assert isinstance(backend, OnnxAsrBackend)

    def test_get_backend_preserves_implicit_compute_type(self):
        from local_transcriber.backends import get_backend

        backend = get_backend("onnx", compute_type_explicit=False)
        backend.ensure_model_available("gigaam-v3-e2e-rnnt", "fp16")

        assert backend.actual_compute_type == "int8"


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

    def test_whisper_alias_error_suggests_explicit_backend(self):
        backend = OnnxAsrBackend()

        with pytest.raises(
            ValueError,
            match=r"Whisper.*--device openvino-cpu.*--device cuda",
        ):
            backend._resolve_model("medium")

    def test_whisper_error_offers_platform_independent_backend(self):
        """На macOS и ARM нет ни OpenVINO, ни CUDA — нужен путь через cpu."""
        backend = OnnxAsrBackend()

        with pytest.raises(ValueError, match=r"--device cpu --model medium"):
            backend._resolve_model("medium")

    def test_turbo_whisper_error_suggests_models_supported_by_backends(self):
        backend = OnnxAsrBackend()

        with pytest.raises(ValueError) as exc_info:
            backend._resolve_model("large-v3-turbo")

        message = str(exc_info.value)
        assert "--device openvino-cpu --model large-v3-turbo" in message
        assert "--device cuda --model medium" in message
        assert "--device cpu --model medium" in message
        assert "--device cuda --model large-v3-turbo" not in message
        assert "--device cpu --model large-v3-turbo" not in message


class TestRuntimeInfo:
    def test_reports_versions_and_configured_providers(self, monkeypatch):
        """Диагностика различает доступные providers и те, что заданы сессиям ASR и VAD."""
        monkeypatch.setattr(
            "onnxruntime.get_available_providers",
            lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        monkeypatch.setattr("onnx_asr.load_model", lambda **kwargs: MagicMock())
        monkeypatch.setattr("onnx_asr.load_vad", lambda model, **kwargs: MagicMock())
        backend = OnnxAsrBackend()
        backend.actual_compute_type = "int8"
        backend.create_model("gigaam-v3-e2e-rnnt", "onnx", "int8")

        info = backend.runtime_info()

        assert info["engine"] == "onnx-asr"
        assert info["onnxruntime"]
        assert info["onnx_asr"]
        assert info["available_providers"] == "CoreMLExecutionProvider, CPUExecutionProvider"
        assert info["asr_providers"] == "CPUExecutionProvider"
        assert info["vad_providers"] == "CPUExecutionProvider"
        assert info["quantization"] == "int8"
