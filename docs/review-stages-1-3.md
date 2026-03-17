# Review: Stages 1-3

## Executive Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 0 |
| HIGH | 0 |
| MEDIUM | 2 |
| LOW | 1 |

**Overall Risk:** MEDIUM
**Recommendation:** CONDITIONAL

**Key Metrics:**
- Files analyzed: 10/10
- Lines changed: +427 / -0
- Test coverage gaps: 3 behaviors
- High blast radius changes: 0
- Security regressions detected: 0

## What Changed

**Commit Range:** `e0fcf43..WORKTREE`
**Commits:** `ade3b23`, `510d6cc`, plus uncommitted stage 3 changes

| File | +Lines | -Lines | Risk | Notes |
|------|--------|--------|------|-------|
| pyproject.toml | 28 | 0 | LOW | Project scaffold and dependency wiring |
| src/local_transcriber/cli.py | 14 | 0 | LOW | Step 1 placeholder CLI |
| src/local_transcriber/formatter.py | 23 | 0 | LOW | Step 1 placeholder formatter |
| src/local_transcriber/transcriber.py | 99 | 0 | MEDIUM | Core transcription flow and fallback logic |
| src/local_transcriber/utils.py | 68 | 0 | MEDIUM | Environment checks and input validation |
| tests/test_transcriber.py | 121 | 0 | LOW | Mock tests for step 3 |
| tests/test_utils.py | 74 | 0 | LOW | Unit tests for step 2 |

## Findings

### MEDIUM: CUDA fallback does not trigger when `model.transcribe()` fails before returning a generator

**File:** `src/local_transcriber/transcriber.py:51`
**Test Coverage:** PARTIAL

`transcribe()` wraps exceptions from `model.transcribe(...)` into `RuntimeError`, but the CPU fallback exists only in the later generator-iteration branch. If `faster-whisper` raises a CUDA or OOM error during `model.transcribe(...)` itself, the function exits instead of retrying on CPU.

This is a direct mismatch with the step 3 requirement to fall back on CUDA/OOM failures.

**Reproduction:**
- Mock `WhisperModel(...).transcribe` to raise `RuntimeError("CUDA kernel launch failed")`
- Current result: `RuntimeError("Ошибка при транскрипции файла ...")`
- Expected result: warning + retry on CPU

**Recommendation:**
- Apply the same `_is_cuda_error()` fallback path around `model.transcribe(...)`, not only around iteration of the returned generator
- Add a test for CUDA failure raised directly by `model.transcribe(...)`

### MEDIUM: `get_gpu_name()` crashes on empty successful `nvidia-smi` output

**File:** `src/local_transcriber/utils.py:42`
**Test Coverage:** NO

When `subprocess.run(...)` returns `returncode == 0` with empty `stdout`, `splitlines()[0]` raises `IndexError`. The plan explicitly allows `get_gpu_name()` to return `None`, so this path should degrade gracefully instead of crashing.

This will surface later in CLI/device formatting and turn a non-critical metadata lookup into a hard failure.

**Reproduction:**
- Mock `subprocess.run` to return `CompletedProcess(..., returncode=0, stdout="")`
- Current result: `IndexError: list index out of range`
- Expected result: `None`

**Recommendation:**
- Check `stdout.strip()` before indexing the first line
- Add a unit test for empty `stdout`

### LOW: `on_segment` emits duplicate segments after mid-stream CUDA fallback

**File:** `src/local_transcriber/transcriber.py:63`
**Test Coverage:** NO

If the GPU generator yields one or more segments and then fails with a CUDA error, `on_segment` is called for the partial GPU output and then called again for the full CPU retry. The returned `segments` list is reset correctly, but callback side effects are not.

For the planned `--verbose` flow this means duplicated stderr output such as:

```text
first
first
second
```

while the final transcript contains only `first`, `second`.

**Recommendation:**
- Buffer callback output until the segment stream completes successfully, or
- suppress callback invocation during the first failed attempt and only emit after a successful run

## Test Coverage Analysis

**Observed coverage:** targeted unit tests exist for step 2 and step 3 happy paths, but not for several failure paths.

**Untested Changes:**

| Function | Risk | Gap |
|----------|------|-----|
| `transcribe()` | MEDIUM | No test for CUDA failure raised by `model.transcribe(...)` |
| `transcribe()` | LOW | No test for duplicate callback behavior after generator fallback |
| `get_gpu_name()` | MEDIUM | No test for empty successful stdout |

## Blast Radius Analysis

The current blast radius is low because stage 1-3 code is only consumed by placeholder CLI wiring and tests. The highest-impact function is `transcribe()`, which will become user-facing once stage 5 is implemented.

| Function | Current Callers | Risk | Priority |
|----------|-----------------|------|----------|
| `transcribe()` | tests only | MEDIUM | P1 |
| `get_gpu_name()` | not yet wired into CLI | MEDIUM | P1 |
| `validate_input_file()` | tests only | LOW | P2 |

## Historical Context

- `ade3b23` introduced the scaffold and placeholder module layout for step 1
- `510d6cc` added environment validation and output-path logic for step 2
- Stage 3 is currently in the working tree and introduces the first non-trivial runtime behavior, including fallback and callback logic

The defects above are all newly introduced in the step 3 working tree implementation, not legacy behavior.

## Recommendations

### Immediate

- [ ] Fix CUDA fallback for errors raised by `model.transcribe(...)`
- [ ] Make `get_gpu_name()` return `None` on empty stdout instead of raising
- [ ] Add regression tests for both cases

### Before Stage 5

- [ ] Decide how `on_segment` should behave across retries and make the behavior explicit
- [ ] Add one test that covers generator failure after partial output

### Residual Risk

- CLI acceptance for `uv run transcribe --help` was not reproducible in this environment because `uv run transcribe` hit a local `snap-confine` execution issue unrelated to the project code
