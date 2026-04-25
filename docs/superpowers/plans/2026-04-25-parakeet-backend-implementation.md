# onnx-asr Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `onnx-asr` as a third pluggable backend (device `"onnx"`) supporting GigaAM v3 (Russian, 4.7% WER, 59x RTF) and Parakeet v3 (multilingual, 11% WER, 34x RTF).

**Architecture:** Follow `OpenVINOBackend` pattern in `backends/openvino.py`. Implement the structural Backend protocol (`ensure_model_available`, `create_model`, `transcribe`). Register via `get_backend("onnx")` in `__init__.py`. Audio loading reuses `faster_whisper.decode_audio()`. VAD via Silero built into onnx-asr.

**Tech Stack:** `onnx-asr>=0.11.0`, `onnxruntime` (transitive), `huggingface-hub` (transitive), existing `faster-whisper` for audio decode.

**Refs:** Spec at `docs/superpowers/specs/2026-04-25-parakeet-backend-design.md`, Backend protocol at `src/local_transcriber/backends/base.py`, Pattern to follow at `src/local_transcriber/backends/openvino.py`, Types at `src/local_transcriber/types.py`.

---

### Task 1: Add onnx-asr dependency

**Files:**
- Modify: `pyproject.toml:7-15`

- [ ] **Step 1: Add `onnx-asr[cpu,hub]` to dependencies**

Open `pyproject.toml`. In the `dependencies` list, add `"onnx-asr[cpu,hub]>=0.11.0"`:

```toml
dependencies = [
    "typer",
    "rich",
    "faster-whisper>=1.2.1",
    "socksio>=1.0.0",
    "nvidia-cublas-cu12>=12.4; sys_platform == 'linux' and platform_machine == 'x86_64'",
    "openvino-genai>=2025.0; sys_platform != 'darwin' and (platform_machine == 'x86_64' or platform_machine == 'AMD64')",
    "tomli>=2.0; python_version < '3.11'",
    "onnx-asr[cpu,hub]>=0.11.0",
]
```

- [ ] **Step 2: Install the dependency**

Run: `uv sync`
Expected: onnx-asr and onnxruntime installed without errors.

- [ ] **Step 3: Verify import works**

Run: `uv run python -c "import onnx_asr; print(onnx_asr.__version__)"`
Expected: prints version (e.g. `0.11.0`), no errors.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add onnx-asr[cpu,hub]>=0.11.0 dependency"
```

---

### Task 2: Backend skeleton — model aliases and class stub

**Files:**
- Create: `src/local_transcriber/backends/onnx_asr.py`
- Create: `tests/test_onnx_asr.py`
- Read: `src/local_transcriber/backends/openvino.py` (for pattern)
- Read: `src/local_transcriber/backends/base.py` (for protocol)
- Read: `src/local_transcriber/types.py` (for Segment, TranscribeResult)

- [ ] **Step 1: Write the failing test for model alias resolution**

Create `tests/test_onnx_asr.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_onnx_asr.py -v`
Expected: FAIL — `OnnxAsrBackend` not defined.

- [ ] **Step 3: Write minimal implementation**

Create `src/local_transcriber/backends/onnx_asr.py`:

```python
"""Бэкенд транскрипции на основе onnx-asr (GigaAM, Parakeet, FastConformer)."""

from __future__ import annotations

MODEL_ALIASES: dict[str, str] = {
    "gigaam-v3": "gigaam-v3-ctc",
    "parakeet-v3": "nemo-parakeet-tdt-0.6b-v3",
}

SUPPORTED_ALIASES = ", ".join(MODEL_ALIASES)


class OnnxAsrBackend:
    """Бэкенд транскрипции через onnx-asr (ONNX Runtime)."""

    def _resolve_model(self, model_name: str) -> str:
        """Resolve alias to onnx-asr model name. Raw names pass through."""
        if model_name in MODEL_ALIASES:
            return MODEL_ALIASES[model_name]
        if "/" in model_name or "-" in model_name:
            # Looks like a raw onnx-asr name — allow passthrough
            return model_name
        raise ValueError(
            f"Неподдерживаемая модель '{model_name}'. "
            f"Доступные алиасы: {SUPPORTED_ALIASES}. "
            f"Либо укажите полное имя модели onnx-asr."
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_onnx_asr.py::TestModelAliases -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/local_transcriber/backends/onnx_asr.py tests/test_onnx_asr.py
git commit -m "feat: add onnx-asr backend skeleton with model alias resolution"
```

---

### Task 3: Implement ensure_model_available

**Files:**
- Modify: `src/local_transcriber/backends/onnx_asr.py`
- Modify: `tests/test_onnx_asr.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_onnx_asr.py`:

```python
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
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `uv run pytest tests/test_onnx_asr.py::TestEnsureModelAvailable -v`
Expected: FAIL — `ensure_model_available` not defined.

- [ ] **Step 3: Implement ensure_model_available**

Append to `OnnxAsrBackend` class in `src/local_transcriber/backends/onnx_asr.py`:

```python
    def __init__(self):
        self.actual_compute_type: str | None = None
        self._resolved_model_id: str | None = None
        self._vad: Any = None

    def ensure_model_available(
        self,
        model_name: str,
        compute_type: str,
        on_status: Callable[[str], None] | None = None,
    ) -> str:
        """Resolves model alias and returns the onnx-asr model identifier.

        onnx-asr downloads models automatically via load_model(),
        so this just validates the alias and returns the identifier string.
        """
        self.actual_compute_type = compute_type
        self._resolved_model_id = self._resolve_model(model_name)
        return self._resolved_model_id
```

Add the import at the top of the file:

```python
from collections.abc import Callable
from typing import Any
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `uv run pytest tests/test_onnx_asr.py::TestEnsureModelAvailable -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/local_transcriber/backends/onnx_asr.py tests/test_onnx_asr.py
git commit -m "feat: implement OnnxAsrBackend.ensure_model_available"
```

---

### Task 4: Implement create_model

**Files:**
- Modify: `src/local_transcriber/backends/onnx_asr.py`
- Modify: `tests/test_onnx_asr.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_onnx_asr.py`:

```python
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
        monkeypatch.setattr("onnx_asr.load_vad", lambda **kw: None)

        backend = OnnxAsrBackend()
        backend.create_model("parakeet-v3", "onnx", "fp16")

        assert calls == ["fp16"]
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `uv run pytest tests/test_onnx_asr.py::TestCreateModel -v`
Expected: FAIL — `create_model` not defined.

- [ ] **Step 3: Implement create_model**

Append to `OnnxAsrBackend` class:

```python
    def create_model(
        self,
        model_path: str,
        device: str,
        compute_type: str,
        cpu_threads: int = 0,
    ) -> Any:
        """Creates onnx-asr model with VAD.

        model_path: onnx-asr model identifier (e.g. "gigaam-v3-ctc").
        compute_type: "int8", "fp16", or "float32" — passed as quantization.
        cpu_threads: not used by onnx-asr (onnxruntime manages threads internally).
        """
        import onnx_asr

        ct = compute_type if compute_type in ("int8", "fp16", "float32") else "int8"

        model = onnx_asr.load_model(
            model=model_path,
            quantization=ct,
            cpu_preprocessing=True,
        )
        vad = onnx_asr.load_vad("silero")
        self._vad = vad
        return model.with_vad(vad)
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `uv run pytest tests/test_onnx_asr.py::TestCreateModel -v`
Expected: 3 PASS.

Note: These tests mock `onnx_asr.load_model` and `onnx_asr.load_vad`, so no real model download happens.

- [ ] **Step 5: Commit**

```bash
git add src/local_transcriber/backends/onnx_asr.py tests/test_onnx_asr.py
git commit -m "feat: implement OnnxAsrBackend.create_model with VAD"
```

---

### Task 5: Implement transcribe

**Files:**
- Modify: `src/local_transcriber/backends/onnx_asr.py`
- Modify: `tests/test_onnx_asr.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_onnx_asr.py`:

```python
from pathlib import Path
from local_transcriber.types import Segment, TranscribeResult


class FakeVadSegment:
    """Mimics onnx-asr SegmentResult."""
    def __init__(self, start_ts, end_ts, text):
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.text = text


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
        result = backend.transcribe(FakeModel(), wav_file)

        assert len(result.segments) == 0
        assert result.language == "unknown"
        assert result.duration == 1.0
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `uv run pytest tests/test_onnx_asr.py::TestTranscribe -v`
Expected: FAIL — `transcribe` not defined.

- [ ] **Step 3: Implement transcribe**

Append to `OnnxAsrBackend` class:

```python
    def transcribe(
        self,
        model: Any,
        file_path: Path,
        language: str | None,
        on_segment: Callable[[Segment], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> TranscribeResult:
        """Transcribes audio file using onnx-asr model with VAD.

        model: result of create_model() — a SegmentResultsAsrAdapter.
        file_path: path to audio/video file (any format supported by faster-whisper decode).
        language: language code (e.g. "ru", "en") — only meaningful for multilingual models.
        """
        from faster_whisper import decode_audio

        _notify(on_status, "Загружаю аудио...")
        audio_array = decode_audio(str(file_path), sampling_rate=16000)
        duration = len(audio_array) / 16000.0

        _notify(on_status, "Транскрибирую (onnx-asr)...")
        segments: list[Segment] = []
        detected_language = language or "unknown"

        for vad_seg in model.recognize(audio_array, 16000, language=language):
            seg = Segment(
                start=max(0.0, vad_seg.start_ts),
                end=max(0.0, vad_seg.end_ts),
                text=vad_seg.text,
            )
            if on_segment is not None:
                on_segment(seg)
            segments.append(seg)
            _notify(
                on_status,
                f"Транскрибирую (onnx-asr)... [{len(segments)} сегм.]",
            )

        return TranscribeResult(
            segments=segments,
            language=detected_language,
            language_probability=1.0 if language else 0.0,
            duration=duration,
            device_used="",  # оркестратор проставит
        )
```

Add the helper function at the bottom of the file (before class):

```python
def _notify(on_status: Callable[[str], None] | None, message: str) -> None:
    if on_status is not None:
        on_status(message)
```

Update imports at the top of `onnx_asr.py` — the full import block should be:

```python
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from local_transcriber.types import Segment, TranscribeResult
```

- [ ] **Step 4: Run tests — verify all pass**

Run: `uv run pytest tests/test_onnx_asr.py -v`
Expected: ALL tests pass (4 alias + 3 ensure + 3 create + 4 transcribe = 14 PASS).

- [ ] **Step 5: Commit**

```bash
git add src/local_transcriber/backends/onnx_asr.py tests/test_onnx_asr.py
git commit -m "feat: implement OnnxAsrBackend.transcribe with VAD segments"
```

---

### Task 6: Register backend in __init__.py

**Files:**
- Modify: `src/local_transcriber/backends/__init__.py`
- Modify: `tests/test_onnx_asr.py`

- [ ] **Step 1: Write failing test for get_backend("onnx")**

Append to `tests/test_onnx_asr.py`:

```python
class TestBackendRegistration:
    def test_get_backend_returns_onnx_backend(self):
        from local_transcriber.backends import get_backend
        backend = get_backend("onnx")
        assert isinstance(backend, OnnxAsrBackend)
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `uv run pytest tests/test_onnx_asr.py::TestBackendRegistration -v`
Expected: FAIL — `ValueError` or some error from `get_backend("onnx")`.

- [ ] **Step 3: Register "onnx" device**

Open `src/local_transcriber/backends/__init__.py` and add the `"onnx"` case before the fallback. The function should look like:

```python
def get_backend(device: str, *, compute_type_explicit: bool = True) -> Backend:
    """Возвращает экземпляр бэкенда для указанного устройства.

    Импорты ленивые — бэкенд загружается только при запросе.
    compute_type_explicit: False если compute_type пришёл из дефолтов (влияет на fallback).
    """
    if device in ("openvino", "openvino-gpu", "openvino-cpu"):
        try:
            from .openvino import OpenVINOBackend
        except ImportError:
            raise ValueError(
                "OpenVINO бэкенд недоступен. Установите: pip install openvino-genai"
            ) from None
        return OpenVINOBackend(
            ov_device=device, compute_type_explicit=compute_type_explicit
        )

    if device == "onnx":
        try:
            from .onnx_asr import OnnxAsrBackend
        except ImportError:
            raise ValueError(
                "onnx-asr бэкенд недоступен. Установите: pip install onnx-asr[cpu,hub]"
            ) from None
        return OnnxAsrBackend()

    # cuda, cpu и всё остальное → faster-whisper
    from .faster_whisper import FasterWhisperBackend

    return FasterWhisperBackend()
```

- [ ] **Step 4: Run test — expect PASS**

Run: `uv run pytest tests/test_onnx_asr.py::TestBackendRegistration -v`
Expected: 1 PASS.

- [ ] **Step 5: Run ALL tests to ensure nothing broken**

Run: `uv run pytest -v`
Expected: all existing tests + new onnx-asr tests pass. No regressions in faster-whisper or OpenVINO.

- [ ] **Step 6: Commit**

```bash
git add src/local_transcriber/backends/__init__.py tests/test_onnx_asr.py
git commit -m "feat: register onnx-asr backend for --device onnx"
```

---

### Task 7: Manual smoke test with real model

**Files:**
- No code changes. Manual verification only.

- [ ] **Step 1: Verify CLI shows onnx device option**

Run: `uv run transcribe --help`
Expected: shows `--device` option, verify `onnx` is listed.

- [ ] **Step 2: Smoke test with GigaAM v3 on a short audio file**

Requires a real audio file (WAV or MP3). Use a small test file if available.

Run: `uv run transcribe /path/to/test.wav --device onnx --model gigaam-v3 --verbose`

Expected:
- Model downloads from HuggingFace (first run, ~600 MB)
- Transcription runs
- Output `.md` file created with segments and timestamps
- No errors

- [ ] **Step 3: Smoke test with Parakeet v3**

Run: `uv run transcribe /path/to/test.wav --device onnx --model parakeet-v3 --language ru --verbose`

Expected:
- Model downloads (first run)
- Transcription with Russian language
- Output `.md` created

- [ ] **Step 4: Verify unknown alias error**

Run: `uv run transcribe /path/to/test.wav --device onnx --model nonexisent`

Expected: error message with available aliases (`gigaam-v3`, `parakeet-v3`).

- [ ] **Step 5: Verify compute-type flag works**

Run: `uv run transcribe /path/to/test.wav --device onnx --model gigaam-v3 --compute-type fp16`

Expected: model loads with fp16 quantization (slightly larger download), transcription works.

---

### Task 8: Write comparison script (optional, for experiment)

**Files:**
- Create: `scripts/compare_backends.py`

- [ ] **Step 1: Create comparison script**

Create `scripts/compare_backends.py`:

```python
"""Compare onnx-asr vs OpenVINO backends on real audio files.

Usage: python scripts/compare_backends.py /path/to/audio.mp3
"""
import sys
import time
from pathlib import Path
from local_transcriber.transcriber import load_model, _transcribe_file
from local_transcriber.backends import get_backend
from local_transcriber.types import Segment


def transcribe_with_backend(file_path: Path, device: str, model_name: str,
                            compute_type: str, language: str | None) -> tuple[float, int, str]:
    """Run transcription and return (elapsed_sec, segment_count, transcript_text)."""
    start = time.monotonic()
    model_obj, actual_device, backend, model_path = load_model(
        model_name, device, compute_type,
        strict_device=True,
    )
    tfr = _transcribe_file(
        model_obj, actual_device, backend, model_path,
        file_path, model_name, compute_type,
        language=language,
    )
    elapsed = time.monotonic() - start
    text = " ".join(s.text for s in tfr.result.segments)
    return elapsed, len(tfr.result.segments), text


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/compare_backends.py <audio_file>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    models_to_test = [
        ("gigaam-v3", "onnx", "int8"),
        ("parakeet-v3", "onnx", "int8"),
        ("medium", "openvino-cpu", "int8"),
    ]

    print(f"File: {file_path.name} ({file_path.stat().st_size / 1e6:.1f} MB)")
    print()

    for model_name, device, ct in models_to_test:
        print(f"--- {model_name} on {device} (compute={ct}) ---")
        try:
            elapsed, seg_count, text = transcribe_with_backend(
                file_path, device, model_name, ct, language="ru" if "gigaam" in model_name else None,
            )
            print(f"  Time: {elapsed:.1f}s")
            print(f"  Segments: {seg_count}")
            print(f"  Text preview: {text[:200]}...")
            print()
        except Exception as e:
            print(f"  ERROR: {e}")
            print()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the comparison**

Run: `uv run python scripts/compare_backends.py /path/to/real/audio.mp3`
Expected: timing and transcript preview for each model.

- [ ] **Step 3: Commit**

```bash
git add scripts/compare_backends.py
git commit -m "test: add onnx-asr vs OpenVINO comparison script"
```

---

## Self-Review

1. **Spec coverage:** Each spec requirement maps to a task:
   - Model aliases (gigaam-v3, parakeet-v3) → Task 2
   - ensure_model_available → Task 3
   - create_model with VAD → Task 4
   - transcribe with decode_audio → Task 5
   - Registration via get_backend("onnx") → Task 6
   - Dependencies → Task 1
   - Comparison → Task 8
   - Error handling (ValueError for bad alias) → Task 2 tests

2. **Placeholder scan:** No TBD, TODO, or vague descriptions. All code is concrete.

3. **Type consistency:**
   - `_resolved_model_id` set in Task 3, used in Task 4 (consistent)
   - `actual_compute_type` set in Task 3, checked in Task 4 tests
   - `Segment` imported from types.py — matches project type
   - `TranscribeResult` fields match types.py definition
