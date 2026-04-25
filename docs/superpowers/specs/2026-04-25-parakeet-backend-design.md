# Design: onnx-asr Backend (Parakeet / GigaAM)

**Status**: Draft
**Date**: 2026-04-25
**Branch**: experiment/onnx-asr-backend

## Motivation

Current CPU backends (faster-whisper, OpenVINO on CPU) run at 0.5-1.5x realtime for large models. Target audience has Intel integrated GPU or CPU only — no discrete NVIDIA GPU.

[onnx-asr](https://github.com/istupakov/onnx-asr) is a lightweight ONNX Runtime wrapper supporting Parakeet, GigaAM, and FastConformer models. Key advantages:

- **30-90x realtime on CPU** (vs 0.5-1.5x for faster-whisper)
- **Lightweight**: `numpy` + `onnxruntime` + `huggingface-hub` (~no added weight)
- **Python >= 3.10** — compatible with project
- **Russian-optimized models** with WER 4-5% (vs 10%+ for Whisper)

## Goals

1. Add `onnx-asr` as a third pluggable backend (alongside FasterWhisper and OpenVINO)
2. Compare speed and quality against OpenVINO on real Russian audio
3. Experiment is on a separate branch — merge only if results are compelling

## Models

Two model aliases exposed via CLI `--model`:

| Alias | onnx-asr name | Language | WER (ru) | RTFx CPU |
|-------|--------------|----------|----------|----------|
| `gigaam-v3` | `gigaam-v3-ctc` | ru only | 4.72% | 59x |
| `parakeet-v3` | `nemo-parakeet-tdt-0.6b-v3` | 25 lang (auto-detect) | 10.95% | 34x |

`gigaam-v3` is the default for `--device onnx` (best Russian quality + speed).
`parakeet-v3` is the multilingual fallback.

User can also pass any valid onnx-asr model name directly (e.g. `nemo-canary-1b-v2`).

## Architecture

### New file

`src/local_transcriber/backends/onnx_asr.py` — mirrors `openvino.py` structure.

Implements the [Backend protocol](../adr/003-pluggable-backends.md) (structural typing):

```
class OnnxAsrBackend:
    def ensure_model_available(model_name, compute_type, on_status) -> str
    def create_model(model_path, device, compute_type, cpu_threads) -> Any
    def transcribe(model, file_path, language, on_segment, on_status) -> TranscribeResult
```

### Model resolution (`ensure_model_available`)

1. Resolve alias → onnx-asr model name via `MODEL_ALIASES` dict
2. Allow raw onnx-asr names (e.g. `nemo-parakeet-tdt-0.6b-v3`) to pass through
3. Return resolved model identifier string (onnx-asr handles download internally via `load_model`)

### Model creation (`create_model`)

```python
import onnx_asr
model = onnx_asr.load_model(
    model_id, quantization=compute_type, cpu_preprocessing=True,
)
vad = onnx_asr.load_vad("silero")
model = model.with_vad(vad)
```

- `compute_type`: `int8` (default, quantized, fast), `fp16`, `float32`
- `cpu_preprocessing=True`: keeps mel-spectrogram computation on CPU (faster for CPU-only inference)
- VAD (Silero): always enabled — splits audio by voice activity, handles any length
- VAD segments naturally carry `start_ts`/`end_ts` — no separate `.with_timestamps()` needed

### Transcription (`transcribe`)

1. `faster_whisper.decode_audio(file, 16000)` → numpy float32 array (reuses same audio loader as OpenVINO backend, supports all media formats)
2. `model.recognize(audio_array, 16000, language=lang)` → iterator of VAD segments with `start_ts`, `end_ts`, `text`
3. Map each segment → project's `Segment(start, end, text)` dataclass
4. Return `TranscribeResult` with segments, language, duration

Language handling:
- `gigaam-v3`: Russian only, `language` parameter ignored
- `parakeet-v3`: auto-detect (when `--language auto` / `None`) or explicit `--language ru/en/...`

### Quantization support

onnx-asr supports quantized ONNX models via `quantization` parameter:

| compute_type | Description | RAM | Quality impact |
|-------------|-------------|-----|---------------|
| `int8` | 8-bit quantized (default) | ~300 MB | Minimal |
| `fp16` | Half precision | ~600 MB | None |
| `float32` | Full precision | ~1.2 GB | None |

### Error handling

- `RuntimeError` (OOM, ONNX session failure) → warning + fallback to CPU via `CPUExecutionProvider`
- Invalid model name → `ValueError` with list of supported aliases
- Corrupt audio / unsupported format → propagated from `faster_whisper.decode_audio`

### Registration

In `backends/__init__.py`:

```python
if device == "onnx":
    from .onnx_asr import OnnxAsrBackend
    return OnnxAsrBackend()
```

Device `"onnx"` is NOT in auto-detect chain. Only explicit `--device onnx`.
Rationale: experimental backend, don't surprise existing users.

### Dependencies

Add to `pyproject.toml`:

```toml
dependencies = [
    # ... existing ...
    "onnx-asr[cpu,hub]>=0.11.0",
]
```

`onnxruntime` pulled transitively by `onnx-asr`.

### CLI integration

No CLI changes needed — existing `--device`, `--model`, `--compute-type`, `--language` flags work:

```bash
# Russian, best quality/speed
uv run transcribe meeting.mp4 --device onnx --model gigaam-v3

# Multilingual
uv run transcribe podcast.mp3 --device onnx --model parakeet-v3 --language auto

# With int8 quantization
uv run transcribe lecture.mp4 --device onnx --compute-type int8
```

## Comparison approach

Experiment compares onnx-asr against OpenVINO backend on real audio files:

1. Pick 2-3 Russian audio files of varying length (1 min, 5 min, 30 min)
2. Run both backends, measure: elapsed time, segment count, dump transcripts
3. Qualitative: can a readable summary/conspect be made from the transcript?
4. Decision criteria:
   - Noticeably faster at comparable quality → keep
   - Noticeably better quality at comparable speed → keep
   - Neither → discard

## Non-goals

- No auto-detect (device `"onnx"` must be explicit)
- No OpenVINO execution provider for onnx-asr (onnx-asr can use it, but out of scope)
- No German/French/etc language optimization — only Russian is benchmarked
- No replacing existing backends
