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
  → config.py      cascade: CLI arg → .transcriber.toml (model/compute_type stay None for the module to resolve)
  → utils.py       detect_device(), validate files, expand globs (Windows workaround)
  → context_menu.py Windows SendTo: Transcribe.cmd install/uninstall (--install-menu / --uninstall-menu)
  → transcriber.py Transcriber(ExecutionRequest): resolves auto + device defaults,
                    get_backend(device) → ensure_model_available → create_model once per run,
                    transcribe(file) per file; fallback and ExecutionInfo stay inside
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

- **Execution module** (`Transcriber`): one owner per run for model handle, adapter, actual device and allowed GPU→CPU fallback (load time and mid-stream). Callers get `TranscribeResult` + `ExecutionInfo` (requested vs actual device, engine, model, compute type, threads, `runtime_info()` from the adapter for `--verbose`). Fallback is reachable only with `strict_device=False` (Python API): CLI explicit device is strict, `auto` is ONNX CPU.
- **CUDA bootstrap** (`_cuda_bootstrap.py`): preloads `libcublas.so.12` via `ctypes.CDLL(RTLD_GLOBAL)` before importing ctranslate2, because pip's `nvidia-cublas-cu12` installs to a non-standard path and `LD_LIBRARY_PATH` can't be changed at runtime (glibc caches it).
- **Batch mode**: 3-phase pipeline (prescan → create `Transcriber` → transcribe all). No state is carried between files by the CLI.
- **Device-aware defaults**: `compute_type` and `model` vary by device (float16 for CUDA, int8 for OpenVINO, float32 for CPU). Defined in `config.py` `DEVICE_DEFAULTS`, applied by `Transcriber` when the request leaves them `None`.
- **ONNX thread budget**: `--threads` sets `intra_op_num_threads` for both ASR and VAD sessions via `SessionOptions`; 0 keeps onnxruntime defaults.
- **OpenVINO uses pre-quantized models** — `compute_type` selects which HF repo to download, not a runtime parameter.

## Testing

All tests mock backends — no real model downloads or transcription. Key test patterns:

- CLI tests: `typer.testing.CliRunner` with the real `Transcriber` and a fake adapter patched at `local_transcriber.transcriber.get_backend`; `load_config`, `validate_input_file`, `write_transcript` mocked
- `_cli_run()` — context manager assembling that standard set, yields `(backend, write_transcript)`
- `_make_result()` / `_make_backend()` — factories for test data
- Module tests (`test_transcriber.py`): `Transcriber` through its interface with `_make_run_backend()`; adapter tests patch the libraries (`onnx_asr.load_model`, `faster_whisper.WhisperModel`)

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

## Agent skills

### Issue tracker

Задачи ведутся в Gitea через `tea`; GitHub используется только как зеркало, внешние PR не входят в triage. См. `docs/agents/issue-tracker.md`.

### Triage labels

Используются стандартные пять triage-меток. См. `docs/agents/triage-labels.md`.

### Domain docs

Репозиторий использует single-context layout. См. `docs/agents/domain.md`.
