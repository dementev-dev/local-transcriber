# Review: Stages 4-5

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
- Files analyzed: 4
- Verified commands: `pytest`, CLI help, mocked CLI happy path
- Test coverage gaps: 1 user-facing module (`cli.py`)
- High blast radius changes: 0
- Security regressions detected: 0

## What Changed

**Commit Range:** `dfc5f46..WORKTREE`
**Commits:** `0d1a734`, plus uncommitted step 5 changes

| File | Risk | Notes |
|------|------|-------|
| `src/local_transcriber/formatter.py` | MEDIUM | Output contract and markdown formatting |
| `tests/test_formatter.py` | LOW | Unit coverage for formatter |
| `src/local_transcriber/cli.py` | MEDIUM | Main user-facing flow and file writing |
| `docs/plan.md` | LOW | Checkbox updates for step 5 |

## Findings

### MEDIUM: `format_timestamp()` can emit invalid centiseconds like `.100`

**File:** `src/local_transcriber/formatter.py:7`
**Test Coverage:** NO

The implementation rounds centiseconds independently from the integral seconds:

```python
total_seconds = int(seconds)
centiseconds = int(round((seconds - total_seconds) * 100))
```

For values such as `0.995`, this produces `00:00.100` instead of carrying into the next second.

**Reproduction:**
- `format_timestamp(0.995)` returns `00:00.100`

**Impact:**
- Breaks the PRD timestamp format contract (`SS.ss` must always have exactly two fractional digits)
- Can produce malformed transcript timestamps on real segment boundaries

**Recommendation:**
- Round the full timestamp first and then split into components, or normalize `centiseconds == 100` by incrementing seconds
- Add a regression test for `0.995`

### MEDIUM: Transcript formatting depends on segment text already containing a leading space

**File:** `src/local_transcriber/formatter.py:62`
**Test Coverage:** PARTIAL

Segment lines are written as:

```python
f"[{start} - {end}]{seg.text}"
```

This only matches the PRD format if `seg.text` already starts with a space. The current tests mask that dependency by building fixtures with leading spaces.

**Reproduction:**
- A mocked CLI run with a segment text of `"Hello"` writes:
  - `[00:00.00 - 00:01.00]Hello`
- Expected:
  - `[00:00.00 - 00:01.00] Hello`

**Impact:**
- Output format becomes model-dependent instead of being guaranteed by the formatter
- Any future normalization in `transcriber.py` will immediately break transcript formatting

**Recommendation:**
- Normalize segment text inside the formatter, e.g. `seg.text.strip()` plus an explicit single space after `]`
- Add a test case where segment text has no leading whitespace

### LOW: Step 5 has no automated tests for the public CLI contract

**File:** `src/local_transcriber/cli.py:16`
**Test Coverage:** NO

The user-facing entrypoint is now wired end to end, but there is still no `test_cli.py` coverage for:

- option parsing
- warning path for empty speech
- default output-path generation
- exit-code behavior on failure

This already matters because the mocked CLI happy path is what exposed the missing-space formatting bug above.

## Test Coverage Analysis

**Executed checks:**
- `uv run pytest` -> 22 tests passed
- `.venv/bin/transcribe --help` -> works
- mocked `CliRunner` happy path -> exit code 0 and output file written

**Coverage gaps:**

| Area | Gap | Risk |
|------|-----|------|
| `formatter.py` | No edge-case test for centisecond carry | MEDIUM |
| `formatter.py` | No test for segment text without leading whitespace | MEDIUM |
| `cli.py` | No automated tests at all | LOW |

## Blast Radius Analysis

The blast radius is still low because this is a small CLI project, but `cli.py` is now the single public entrypoint. Any formatting or wiring defect directly affects all users.

| Function | Exposure | Risk | Priority |
|----------|----------|------|----------|
| `main()` | All CLI invocations | MEDIUM | P1 |
| `format_transcript()` | All saved transcript files | MEDIUM | P1 |
| `format_timestamp()` | Every segment line | MEDIUM | P1 |

## Recommendations

### Immediate

- [ ] Fix centisecond carry handling in `format_timestamp()`
- [ ] Stop relying on leading whitespace in `seg.text`
- [ ] Add formatter regression tests for both cases

### Before Step 6

- [ ] Add `tests/test_cli.py` with a mocked happy path and one error path
- [ ] Assert output file contents through the CLI layer, not only through direct formatter calls
