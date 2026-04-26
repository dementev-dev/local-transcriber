# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Local audio/video transcription CLI — no cloud, no API keys. Outputs markdown with timestamps.

**Language conventions**: code identifiers in English; docstrings, comments, UI strings, and commit messages in Russian. Style is ruff-compatible. Commits follow [Conventional Commits](https://www.conventionalcommits.org/).

## Commands

```bash
uv sync                            # install dependencies
uv run transcribe meeting.mp4      # run CLI
uv run pytest                      # run all tests
uv run pytest tests/test_cli.py    # run one test file
uv run pytest -k test_name         # run single test by name
uv run pytest -v                   # verbose output
```

Package manager is **uv** (not pip). Build backend is hatchling.

## Architecture

```
CLI (cli.py)
  → config.py      cascade: CLI arg → .transcriber.toml → device-aware default → hardcoded
  → utils.py       detect_device(), validate files, expand globs (Windows workaround)
  → transcriber.py load_model() → get_backend(device) → ensure_model_available → create_model
                    _transcribe_file() with mid-stream CUDA→CPU fallback
  → formatter.py   segments → markdown with timestamps, paragraph grouping (>2s pause or >60s)
```

### Backend system (`src/local_transcriber/backends/`)

Three backends implement the `Backend` Protocol (structural typing, no inheritance required):

| Backend | Module | Devices | Library |
|---------|--------|---------|---------|
| FasterWhisper | `faster_whisper.py` | `cpu`, `cuda` | `faster_whisper` (CTranslate2) |
| OpenVINO | `openvino.py` | `openvino`, `openvino-gpu`, `openvino-cpu` | `openvino_genai` |
| OnnxAsr | `onnx_asr.py` | `onnx` | `onnx_asr` (onnxruntime) |

`get_backend(device)` in `backends/__init__.py` maps device string to backend with lazy imports.

### Key design decisions

- **Two-level fallback**: GPU→CPU at model load time AND mid-stream during transcription (GPU visible via nvidia-smi but insufficient VRAM).
- **CUDA bootstrap** (`_cuda_bootstrap.py`): preloads `libcublas.so.12` via `ctypes.CDLL(RTLD_GLOBAL)` before importing ctranslate2, because pip's `nvidia-cublas-cu12` installs to a non-standard path and `LD_LIBRARY_PATH` can't be changed at runtime (glibc caches it).
- **Batch mode**: 3-phase pipeline (prescan → load model once → transcribe all). `TranscribeFileResult` carries updated model/backend/device state between files.
- **Device-aware defaults**: `compute_type` and `model` vary by device (float16 for CUDA, int8 for OpenVINO, float32 for CPU). Defined in `config.py` `DEVICE_DEFAULTS`.
- **OpenVINO uses pre-quantized models** — `compute_type` selects which HF repo to download, not a runtime parameter.

## Testing

All tests mock backends — no real model downloads or transcription. Key test patterns:

- CLI tests: `typer.testing.CliRunner` + mocks for `load_config`, `detect_device`, `load_model`, `_transcribe_file`, `write_transcript`
- `_single_patches()` — helper assembling standard happy-path mock set
- `_make_result()` / `_make_tfr()` — factories for test data

## Common tasks

- **New CLI option**: add `typer.Option` in `cli.py:main()` → add key to `HARDCODED_DEFAULTS` in `config.py` → write test
- **New audio/video format**: add extension to `SUPPORTED_EXTENSIONS` in `utils.py`
- **New backend**: implement `Backend` protocol → add device mapping in `backends/__init__.py` → add device-aware defaults in `config.py`
- **Change output format**: edit `format_transcript()` in `formatter.py`

## Project docs

- `docs/PRD.md` — product requirements and scope
- `docs/backlog.md` — future experiments and ideas
- `docs/gpu.md` — GPU benchmarks, platform compatibility details
- `docs/adr/` — architecture decision records (CUDA bootstrap, batch mode, pluggable backends, compute-type defaults, ONNX-ASR evaluation)

