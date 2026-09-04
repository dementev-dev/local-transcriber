"""Изолированный стенд эксперимента Streaming Sortformer v2 Q8_0."""

from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

SCHEMA = "local-transcriber.streaming-sortformer-experiment.v1"
FRESH_BASIS_MARKER = "issue-46-fresh-owner-reviewed-basis"
W0_RECIPE = "W0-FRESH"
CANDIDATE_RECIPE = "SORTFORMER-V2-Q8"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
LEGACY_ROOT_RE = re.compile(r"issue[-_/ ]?(?:41|42)", re.IGNORECASE)
LEGACY_ID_RE = re.compile(
    r"(?:^|[-_/])(?:41|42)(?:$|[-_/])|"
    r"^(?!W0-FRESH$)(?:[WMQEC]\d)(?:$|[-_/])",
    re.IGNORECASE,
)

TOP_LEVEL_KEYS = {
    "schema",
    "fresh_basis",
    "source",
    "machine",
    "inputs",
    "recipes",
    "schedule",
    "rss_sample_interval_ms",
}
FRESH_BASIS_KEYS = {"marker", "approval"}
APPROVAL_KEYS = {"decision", "timestamp"}
SOURCE_KEYS = {"commit"}
INPUT_KEYS = {
    "source_path",
    "source_sha256",
    "source_size_bytes",
    "fragment",
    "full_recording",
}
FULL_RECORD_KEYS = {
    "path",
    "sha256",
    "size_bytes",
    "duration_seconds",
    "sidecar_path",
    "sidecar_sha256",
    "expected_cluster_count",
}
FRAGMENT_RECORD_KEYS = FULL_RECORD_KEYS | {
    "reference_path",
    "reference_sha256",
}
MACHINE_KEYS = {
    "machine_sha256",
    "physical_cores",
    "logical_cores",
    "affinity",
    "cpuid_isa_sha256",
    "ram_bytes",
    "os_sha256",
    "compiler_sha256",
    "runtime_versions_sha256",
    "power_mode",
    "governor",
    "turbo",
}
CELL_KEYS = {
    "semantic_id",
    "pair_id",
    "stage",
    "recipe",
    "input_sha256",
    "sidecar_sha256",
    "fresh_process",
}

W0_PIN = {
    "sherpa_onnx_version": "1.13.6",
    "sherpa_onnx_core_version": "1.13.6",
    "segmentation_model_sha256": (
        "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079"
    ),
    "embedding_model_sha256": (
        "e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012"
    ),
    "window_shift_ratio": 0.1,
    "clustering": {"algorithm": "FastClustering", "mode": "auto", "threshold": 0.89},
    "segmentation_provider": "cpu",
    "embedding_provider": "cpu",
    "segmentation_debug": True,
    "embedding_debug": True,
    "min_duration_on": 0.3,
    "min_duration_off": 0.5,
    "workers": 1,
    "sessions": 1,
    "intra_op_threads": 8,
    "inter_op_threads": 1,
    "batch_size": 1,
    "process_variant": "upstream-unpatched",
}

CANDIDATE_FIXED_PIN = {
    "runtime_repo": "NVIDIA/NeMo-Speech.cpp",
    "runtime_commit": "4f9676226f667d14608487df744f375db87127f8",
    "runtime_license": "Apache-2.0",
    "model_repo": "nvidia/diar_streaming_sortformer_4spk-v2",
    "model_revision": "5240a64075176943f677d30fa2171c780229f341",
    "model_name": "diar_streaming_sortformer_4spk-v2.q8_0.gguf",
    "model_size_bytes": 147075776,
    "model_sha256": "0679cfeb1ce356d0dea9470b31274f4bfc7eb927497d82005483770666da998a",
    "model_license": "CC-BY-4.0",
    "quantization": "Q8_0",
    "preset": "cpu-diar",
    "diarization_preset": "streaming",
    "backend": "cpu",
    "gpu": -1,
    "processes": 1,
    "streams": 1,
    "internal_cpu_threads": 4,
    "submodule_command": "git submodule update --init ggml",
    "configure_command": "scripts/configure.sh cpu-diar",
    "build_command": "cmake --build --preset cpu-diar",
    "cmake_options": {
        "build_type": "Release",
        "generator": "Ninja",
        "diarization_only": True,
        "patched_ggml": False,
    },
    "geometry": {
        "chunk_frames": 340,
        "right_context_frames": 40,
        "left_context_frames": 0,
        "fifo_frames": 40,
        "spkcache_frames": 188,
        "update_period_frames": 300,
    },
    "post_processing": {
        "onset": 0.641,
        "offset": 0.561,
        "pad_onset_sec": 0.229,
        "pad_offset_sec": 0.079,
        "min_duration_sec": 0.511,
        "min_gap_sec": 0.296,
    },
    "num_speakers": 4,
    "seconds_per_frame": 0.08,
}
CANDIDATE_PROVENANCE_KEYS = {
    "submodule_commits",
    "compiler",
    "cmake_version",
    "executable_sha256",
    "library_sha256",
}


class PreflightError(ValueError):
    """Ошибка проверки с публично безопасным текстом."""


@dataclass(frozen=True)
class PreflightOutcome:
    state: str
    reason: str | None
    fragment_count: int
    full_recording_count: int
    source_commit_sha256: str


def _exact(value: Any, keys: set[str], reason: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise PreflightError(reason)
    return value


def _sha(value: Any, reason: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise PreflightError(reason)
    return value


def _commit(value: Any, reason: str) -> str:
    if not isinstance(value, str) or COMMIT_RE.fullmatch(value) is None:
        raise PreflightError(reason)
    return value


def _positive_int(value: Any, reason: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PreflightError(reason)
    return value


def _finite(value: Any, reason: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreflightError(reason)
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise PreflightError(reason)
    return result


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    """Возвращает SHA-256 канонического JSON."""
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_id(pair_id: str, stage: str, recipe: str) -> str:
    return "cell-" + canonical_sha256(
        {"pair_id": pair_id, "stage": stage, "recipe": recipe}
    )[:24]


def _cell(
    pair_id: str,
    stage: str,
    recipe: str,
    input_hashes: Sequence[str],
    sidecar_hashes: Sequence[str],
) -> dict[str, Any]:
    return {
        "semantic_id": _semantic_id(pair_id, stage, recipe),
        "pair_id": pair_id,
        "stage": stage,
        "recipe": recipe,
        "input_sha256": list(input_hashes),
        "sidecar_sha256": list(sidecar_hashes),
        "fresh_process": True,
    }


def build_fragment_schedule(
    input_hashes: Sequence[str], sidecar_hashes: Sequence[str]
) -> list[dict[str, Any]]:
    """Строит primary и ровно пять заранее чередующихся финальных пар."""
    if len(input_hashes) != 3 or len(sidecar_hashes) != 3:
        raise PreflightError("schedule-input-count")
    orders = [
        ("primary", (W0_RECIPE, CANDIDATE_RECIPE)),
        ("final-1", (CANDIDATE_RECIPE, W0_RECIPE)),
        ("final-2", (W0_RECIPE, CANDIDATE_RECIPE)),
        ("final-3", (CANDIDATE_RECIPE, W0_RECIPE)),
        ("final-4", (W0_RECIPE, CANDIDATE_RECIPE)),
        ("final-5", (CANDIDATE_RECIPE, W0_RECIPE)),
    ]
    return [
        _cell(pair_id, "primary" if pair_id == "primary" else "final", recipe, input_hashes, sidecar_hashes)
        for pair_id, recipes in orders
        for recipe in recipes
    ]


def build_full_recording_schedule(
    input_hashes: Sequence[str],
    sidecar_hashes: Sequence[str],
    *,
    final_median_speedup_percent: float,
    experiment_state: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Открывает три пары A/B для полных записей при медиане от 30%."""
    if experiment_state is not None:
        raise PreflightError("experiment-already-stopped")
    median_speedup = _finite(final_median_speedup_percent, "final-speedup")
    if median_speedup < 30.0:
        return []
    if len(input_hashes) != 3 or len(sidecar_hashes) != 3:
        raise PreflightError("full-schedule-input-count")
    result = []
    for index, (input_hash, sidecar_hash) in enumerate(
        zip(input_hashes, sidecar_hashes, strict=True), start=1
    ):
        pair_id = f"full-{index}"
        for recipe in (W0_RECIPE, CANDIDATE_RECIPE):
            result.append(
                _cell(pair_id, "full", recipe, [input_hash], [sidecar_hash])
            )
    return result


def _resolve(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise PreflightError("unsafe-relative-path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "\\" in value or "\x00" in value:
        raise PreflightError("path-escape")
    base = root.resolve()
    resolved = (base / Path(value)).resolve()
    if not resolved.is_relative_to(base):
        raise PreflightError("path-escape")
    return resolved


def _verify(root: Path, path_value: Any, expected_hash: Any, size: Any | None) -> None:
    path = _resolve(root, path_value)
    expected = _sha(expected_hash, "invalid-sha256")
    if not path.is_file():
        raise PreflightError("file-unavailable")
    if size is not None and path.stat().st_size != _positive_int(size, "invalid-size"):
        raise PreflightError("size-mismatch")
    if file_sha256(path) != expected:
        raise PreflightError("hash-mismatch")


def _validate_approval(value: Any) -> bool:
    if value is None:
        return False
    approval = _exact(value, APPROVAL_KEYS, "approval-schema")
    if approval["decision"] != "approved":
        return False
    timestamp = approval["timestamp"]
    if not isinstance(timestamp, str):
        raise PreflightError("approval-timestamp")
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise PreflightError("approval-timestamp") from exc
    if parsed.tzinfo is None:
        raise PreflightError("approval-timestamp")
    return True


def _contains_legacy(value: Any) -> bool:
    if isinstance(value, str):
        return LEGACY_ROOT_RE.search(value) is not None
    if isinstance(value, Mapping):
        return any(_contains_legacy(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_legacy(item) for item in value)
    return False


def _validate_machine(machine: Any) -> None:
    value = _exact(machine, MACHINE_KEYS, "machine-schema")
    for key in (
        "machine_sha256",
        "cpuid_isa_sha256",
        "os_sha256",
        "compiler_sha256",
        "runtime_versions_sha256",
    ):
        _sha(value[key], "machine-hash")
    if value["physical_cores"] != 8:
        raise PreflightError("machine-physical-cores")
    if _positive_int(value["logical_cores"], "machine-logical-cores") < 8:
        raise PreflightError("machine-logical-cores")
    affinity = value["affinity"]
    if (
        not isinstance(affinity, list)
        or len(affinity) != 8
        or len(set(affinity)) != 8
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in affinity)
    ):
        raise PreflightError("machine-affinity")
    _positive_int(value["ram_bytes"], "machine-ram")
    if not isinstance(value["turbo"], bool):
        raise PreflightError("machine-turbo")
    for key in ("power_mode", "governor"):
        if not isinstance(value[key], str) or not value[key]:
            raise PreflightError("machine-power")


def _validate_candidate_recipe(candidate: Any) -> None:
    expected_keys = set(CANDIDATE_FIXED_PIN) | CANDIDATE_PROVENANCE_KEYS
    _exact(candidate, expected_keys, "candidate-pin-mismatch")
    if any(candidate[key] != expected for key, expected in CANDIDATE_FIXED_PIN.items()):
        raise PreflightError("candidate-pin-mismatch")
    submodules = candidate["submodule_commits"]
    if not isinstance(submodules, Mapping) or set(submodules) != {"ggml"}:
        raise PreflightError("candidate-build-provenance")
    for commit in submodules.values():
        _commit(commit, "candidate-build-provenance")
    for key in ("compiler", "cmake_version"):
        if not isinstance(candidate[key], str) or not candidate[key].strip():
            raise PreflightError("candidate-build-provenance")
    for key in ("executable_sha256", "library_sha256"):
        _sha(candidate[key], "candidate-build-provenance")


def _validate_recipes(recipes: Any) -> None:
    validated = _exact(recipes, {W0_RECIPE, CANDIDATE_RECIPE}, "recipe-set")
    if validated[W0_RECIPE] != W0_PIN:
        raise PreflightError("w0-pin-mismatch")
    _validate_candidate_recipe(validated[CANDIDATE_RECIPE])


def _validate_schedule(
    schedule: Any, input_hashes: Sequence[str], sidecar_hashes: Sequence[str]
) -> None:
    if not isinstance(schedule, list):
        raise PreflightError("schedule-schema")
    expected = build_fragment_schedule(input_hashes, sidecar_hashes)
    if schedule != expected:
        if any(
            isinstance(cell, Mapping)
            and LEGACY_ID_RE.search(str(cell.get("semantic_id", "")))
            for cell in schedule
        ):
            raise PreflightError("legacy-identity")
        raise PreflightError("schedule-mismatch")
    for cell in schedule:
        _exact(cell, CELL_KEYS, "schedule-cell-schema")


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    root: Path,
    environment: Mapping[str, Any],
    verify_files: bool = True,
) -> PreflightOutcome:
    """Проверяет независимый manifest до загрузки runtime."""
    value = _exact(manifest, TOP_LEVEL_KEYS, "manifest-schema")
    if value["schema"] != SCHEMA:
        raise PreflightError("schema-mismatch")
    if _contains_legacy(value):
        schedule = value.get("schedule", [])
        if any(
            LEGACY_ID_RE.search(str(cell.get("semantic_id", "")))
            for cell in schedule
            if isinstance(cell, Mapping)
        ):
            raise PreflightError("legacy-identity")
        raise PreflightError("legacy-root")

    basis = _exact(value["fresh_basis"], FRESH_BASIS_KEYS, "fresh-basis-schema")
    if basis["marker"] != FRESH_BASIS_MARKER:
        raise PreflightError("fresh-basis-marker")
    approved = _validate_approval(basis["approval"])
    source = _exact(value["source"], SOURCE_KEYS, "source-schema")
    source_commit = _commit(source["commit"], "source-commit")
    if set(environment) != {"commit", "clean"}:
        raise PreflightError("environment-schema")
    if environment["commit"] != source_commit:
        raise PreflightError("source-commit-mismatch")
    if environment["clean"] is not True:
        raise PreflightError("worktree-not-clean")
    _validate_machine(value["machine"])
    _validate_recipes(value["recipes"])

    inputs = value["inputs"]
    if not isinstance(inputs, list) or len(inputs) != 3:
        raise PreflightError("input-count")
    source_hashes = [
        _sha(
            _exact(item, INPUT_KEYS, "input-schema")["source_sha256"],
            "source-hash",
        )
        for item in inputs
    ]
    if len(set(source_hashes)) != 3:
        raise PreflightError("source-hashes-not-unique")
    fragment_hashes = []
    fragment_sidecars = []
    for item in inputs:
        entry = _exact(item, INPUT_KEYS, "input-schema")
        if verify_files:
            _verify(
                root,
                entry["source_path"],
                entry["source_sha256"],
                entry["source_size_bytes"],
            )
        for role in ("fragment", "full_recording"):
            record_keys = (
                FRAGMENT_RECORD_KEYS if role == "fragment" else FULL_RECORD_KEYS
            )
            record = _exact(entry[role], record_keys, "record-schema")
            if role == "fragment" and record["duration_seconds"] != 300.0:
                raise PreflightError("fragment-duration")
            _finite(record["duration_seconds"], "record-duration", positive=True)
            count = record["expected_cluster_count"]
            if isinstance(count, bool) or not isinstance(count, int) or not 2 <= count <= 4:
                raise PreflightError("expected-cluster-count")
            media_hash = _sha(record["sha256"], "record-hash")
            sidecar_hash = _sha(record["sidecar_sha256"], "sidecar-hash")
            if role == "fragment":
                reference_path = record["reference_path"]
                reference_hash = record["reference_sha256"]
                if (reference_path is None) != (reference_hash is None):
                    raise PreflightError("reference-binding")
                if reference_hash is not None:
                    _sha(reference_hash, "reference-hash")
                fragment_hashes.append(media_hash)
                fragment_sidecars.append(sidecar_hash)
            if verify_files:
                _verify(root, record["path"], media_hash, record["size_bytes"])
                _verify(root, record["sidecar_path"], sidecar_hash, None)
                if role == "fragment" and reference_hash is not None:
                    _verify(
                        root,
                        reference_path,
                        reference_hash,
                        None,
                    )
    _validate_schedule(value["schedule"], fragment_hashes, fragment_sidecars)
    _positive_int(value["rss_sample_interval_ms"], "rss-sample-interval")
    return PreflightOutcome(
        state="valid" if approved else "needs-user",
        reason=None if approved else "owner-approval-required",
        fragment_count=3,
        full_recording_count=3,
        source_commit_sha256=hashlib.sha256(source_commit.encode("ascii")).hexdigest(),
    )


def preflight_and_load(
    manifest: Mapping[str, Any],
    *,
    root: Path,
    environment_probe: Callable[[], Mapping[str, Any]],
    w0_loader: Callable[[], Any],
    candidate_loader: Callable[[], Any],
    validate_only: bool = False,
) -> PreflightOutcome:
    """Запускает оба ленивых загрузчика после успешного preflight."""
    outcome = validate_manifest(
        manifest, root=root, environment=dict(environment_probe())
    )
    if outcome.state != "valid" or validate_only:
        return outcome
    w0_loader()
    candidate_loader()
    return outcome


def speedup_percent(baseline_wall: float, candidate_wall: float) -> float:
    """Считает сокращение wall для полной замены свежего W0."""
    baseline = _finite(baseline_wall, "baseline-wall", positive=True)
    candidate = _finite(candidate_wall, "candidate-wall", positive=True)
    return 100.0 * (baseline - candidate) / baseline


def primary_decision(
    speedup: float, *, mandatory_gates_passed: bool
) -> dict[str, Any]:
    """Применяет обязательные gates и границы 5%/10% primary."""
    value = _finite(speedup, "primary-speedup")
    if not isinstance(mandatory_gates_passed, bool):
        raise TypeError("mandatory-gates-verdict")
    if not mandatory_gates_passed:
        return {"state": "stop", "reason": "mandatory-gate-failed"}
    if value < 10.0:
        reason = "equal-speed" if abs(value) <= 5.0 else "below-continuation"
        return {"state": "stop", "reason": reason}
    return {"state": "continue", "reason": None}


def final_statistics(speedups: Sequence[float]) -> dict[str, float]:
    """Считает медиану и MAD только по пяти принятым final-парам."""
    if len(speedups) != 5:
        raise ValueError("five-final-speedups-required")
    values = [_finite(value, "final-speedup") for value in speedups]
    median = float(statistics.median(values))
    mad = float(statistics.median(abs(value - median) for value in values))
    return {
        "median_speedup_percent": median,
        "mad_percentage_points": mad,
    }


def final_decision(speedups: Sequence[float]) -> dict[str, Any]:
    """Открывает этап полных записей при медиане от 30%."""
    result = final_statistics(speedups)
    return {
        "state": (
            "full-recordings"
            if result["median_speedup_percent"] >= 30.0
            else "stop"
        ),
        "reason": (
            None
            if result["median_speedup_percent"] >= 30.0
            else "below-final-threshold"
        ),
        **result,
    }


def _cycle_speedup(results: Sequence[Mapping[str, Any]]) -> float:
    if len(results) != 2 or {result.get("recipe") for result in results} != {
        W0_RECIPE,
        CANDIDATE_RECIPE,
    }:
        raise PreflightError("cycle-recipe-set")
    totals = {}
    for result in results:
        walls = result.get("input_wall_seconds")
        if not isinstance(walls, list) or len(walls) != 3:
            raise PreflightError("cycle-input-wall-count")
        totals[result["recipe"]] = sum(
            _finite(wall, "cycle-input-wall", positive=True) for wall in walls
        )
    return speedup_percent(totals[W0_RECIPE], totals[CANDIDATE_RECIPE])


def run_fragment_state_machine(
    manifest: Mapping[str, Any],
    *,
    runner: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    store: ExperimentState,
) -> dict[str, Any]:
    """Выполняет primary и пять final-пар, затем открывает полные записи."""
    if store.current is not None:
        return dict(store.current)
    schedule = manifest.get("schedule")
    if not isinstance(schedule, list) or len(schedule) != 12:
        raise PreflightError("schedule-mismatch")
    speedups = []
    for pair_index, pair_cells in enumerate(
        zip(schedule[::2], schedule[1::2], strict=True)
    ):
        pair_results = []
        for cell in pair_cells:
            try:
                raw_result = dict(runner(cell))
            except MemoryError:
                return store.stop("oom", {"pair_index": pair_index})
            except Exception:  # noqa: BLE001 - вызов внешнего runtime
                return store.stop("runtime-error", {"pair_index": pair_index})
            if raw_result.get("mandatory_passed") is not True:
                return store.stop("mandatory-gate-failed", raw_result)
            pair_results.append({"recipe": cell["recipe"], **raw_result})
        try:
            pair_speedup = _cycle_speedup(pair_results)
        except (PreflightError, TypeError, ValueError):
            return store.stop("invalid-result-value", {"pair_index": pair_index})
        if pair_index == 0:
            primary = primary_decision(
                pair_speedup, mandatory_gates_passed=True
            )
            if primary["state"] == "stop":
                return store.stop(primary["reason"], {"speedup": pair_speedup})
        else:
            speedups.append(pair_speedup)
    decision = final_decision(speedups)
    if decision["state"] == "stop":
        return store.stop(decision["reason"], decision)
    full_hashes = [item["full_recording"]["sha256"] for item in manifest["inputs"]]
    full_sidecars = [
        item["full_recording"]["sidecar_sha256"] for item in manifest["inputs"]
    ]
    return {
        **decision,
        "full_schedule": build_full_recording_schedule(
            full_hashes,
            full_sidecars,
            final_median_speedup_percent=decision["median_speedup_percent"],
            experiment_state=store.current,
        ),
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as target:
            temporary = Path(target.name)
            json.dump(
                value,
                target,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def persist_private_result(path: Path, value: Any) -> str:
    """Атомарно сохраняет закрытый JSON и возвращает хеш его байтов."""
    require_closed_store(path)
    _atomic_json(path, value)
    return file_sha256(path)


def require_closed_store(path: Path) -> None:
    """Проверяет, что закрытый artifact лежит вне рабочей копии."""
    repository_root = Path(__file__).parents[2].resolve()
    if path.resolve().is_relative_to(repository_root):
        raise PreflightError("private-store-inside-repository")


class ExperimentState:
    """Атомарно сохраняет только первую публично безопасную stop-причину."""

    def __init__(self, path: Path):
        self.path = path
        self._state = self._read_existing()

    def _read_existing(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            _exact(value, {"state", "first_reason", "result_sha256"}, "state-schema")
            if value["state"] != "stop" or sanitize_reason(value["first_reason"]) != value["first_reason"]:
                raise PreflightError("state-schema")
            _sha(value["result_sha256"], "state-schema")
        except (OSError, json.JSONDecodeError) as exc:
            raise PreflightError("state-unavailable") from exc
        return dict(value)

    def stop(self, reason: str, private_result: Mapping[str, Any]) -> dict[str, Any]:
        if self._state is None:
            self._state = self._read_existing()
        if self._state is not None:
            return self._state
        try:
            result_sha256 = canonical_sha256(private_result)
        except (TypeError, ValueError):
            reason = "invalid-result-value"
            result_sha256 = canonical_sha256({"invalid_result": True})
        self._state = {
            "state": "stop",
            "first_reason": sanitize_reason(reason),
            "result_sha256": result_sha256,
        }
        _atomic_json(self.path, self._state)
        return self._state

    @property
    def current(self) -> Mapping[str, Any] | None:
        """Возвращает копию сохраненного stop."""
        return None if self._state is None else dict(self._state)


def execute_until_stop(
    cells: Sequence[Mapping[str, Any]],
    runner: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    *,
    store: ExperimentState,
) -> dict[str, Any]:
    """Останавливает выполнение при первом обязательном провале."""
    completed = 0
    for cell in cells:
        if cell.get("recipe") not in {W0_RECIPE, CANDIDATE_RECIPE}:
            return store.stop("recipe-not-allowed", {"mandatory_passed": False})
        try:
            result = dict(runner(cell))
        except MemoryError:
            return store.stop("oom", {"mandatory_passed": False})
        except Exception:  # noqa: BLE001 - вызов внешнего runtime
            return store.stop("runtime-error", {"mandatory_passed": False})
        completed += 1
        if result.get("mandatory_passed") is not True:
            return store.stop("mandatory-gate-failed", result)
    return {"state": "complete", "completed_cells": completed}


def sanitize_reason(reason: Any) -> str:
    """Проверяет стабильный публичный код причины без путей."""
    if (
        not isinstance(reason, str)
        or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", reason) is None
    ):
        return "internal-error"
    return reason


def _normalized_text(value: Any) -> bytes:
    if not isinstance(value, str):
        raise TypeError("word-text")
    return "".join(value.split()).encode("utf-8")


def _validated_word_sequence(
    words: Sequence[Mapping[str, Any]], *, with_speaker: bool
) -> tuple[list[tuple[float, float]], list[bytes]]:
    positions = []
    texts = []
    previous_start = -math.inf
    expected_keys = {"start", "end", "text"}
    if with_speaker:
        expected_keys.add("speaker")
    for word in words:
        if not isinstance(word, Mapping) or set(word) != expected_keys:
            raise PreflightError("word-schema")
        start = float(word["start"])
        end = float(word["end"])
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
            or start < previous_start
        ):
            raise ValueError("word-order")
        previous_start = start
        positions.append((start, end))
        texts.append(_normalized_text(word["text"]))
    return positions, texts


def canonical_speaker_labels(
    intervals: Sequence[Mapping[str, Any]], output_words: Sequence[Mapping[str, Any]]
) -> list[int | None]:
    """Нумерует анонимные кластеры по первому назначенному слову."""
    interval_speakers = {interval["speaker"] for interval in intervals}
    order: dict[Any, int] = {}
    labels = []
    for word in output_words:
        speaker = word.get("speaker")
        if speaker is None or speaker not in interval_speakers:
            labels.append(None)
        else:
            labels.append(order.setdefault(speaker, len(order) + 1))
    return labels


def quality_gate(
    *,
    expected_cluster_count: int,
    observed_cluster_count: int,
    sidecar_words: Sequence[Mapping[str, Any]],
    output_words: Sequence[Mapping[str, Any]],
    intervals: Sequence[Mapping[str, Any]],
    audio_duration_seconds: float,
    permitted_padding_seconds: float,
) -> dict[str, Any]:
    """Проверяет число кластеров, слова, интервалы и канонические метки."""
    if observed_cluster_count != expected_cluster_count:
        return {"passed": False, "reason": "cluster-count-mismatch", "canonical_labels": []}
    if len(output_words) < len(sidecar_words):
        reason = "missing-words"
    elif len(output_words) > len(sidecar_words):
        reason = "duplicate-words"
    else:
        try:
            sidecar_positions, sidecar_texts = _validated_word_sequence(
                sidecar_words, with_speaker=False
            )
            output_positions, output_texts = _validated_word_sequence(
                output_words, with_speaker=True
            )
        except PreflightError as exc:
            reason = sanitize_reason(str(exc))
        except (KeyError, TypeError, ValueError):
            reason = "word-order-mismatch"
        else:
            if output_positions != sidecar_positions:
                reason = "word-order-mismatch"
            elif output_texts != sidecar_texts:
                reason = "text-mismatch"
            else:
                reason = None
    if reason is not None:
        return {"passed": False, "reason": reason, "canonical_labels": []}

    duration = _finite(audio_duration_seconds, "audio-duration", positive=True)
    padding = _finite(permitted_padding_seconds, "interval-padding")
    if padding < 0:
        raise ValueError("interval-padding")
    previous_start = -math.inf
    try:
        for interval in intervals:
            if set(interval) != {"start", "end", "speaker"}:
                raise ValueError
            speaker = interval["speaker"]
            if isinstance(speaker, bool) or not isinstance(speaker, (int, str)):
                raise TypeError
            start = float(interval["start"])
            end = float(interval["end"])
            if (
                not math.isfinite(start)
                or not math.isfinite(end)
                or end <= start
                or start < -padding
                or end > duration + padding
                or start < previous_start
            ):
                raise ValueError
            previous_start = start
    except (KeyError, TypeError, ValueError):
        return {"passed": False, "reason": "invalid-intervals", "canonical_labels": []}
    if len({interval["speaker"] for interval in intervals}) != observed_cluster_count:
        return {
            "passed": False,
            "reason": "cluster-count-mismatch",
            "canonical_labels": [],
        }
    labels = canonical_speaker_labels(intervals, output_words)
    if any(label is None for word, label in zip(output_words, labels, strict=True) if word.get("speaker") is not None):
        return {"passed": False, "reason": "noncanonical-label", "canonical_labels": []}
    return {"passed": True, "reason": None, "canonical_labels": labels}


DIAGNOSTIC_KEYS = {
    "purity",
    "unmatched_words",
    "channel_stability",
    "short_turns",
    "overlap",
    "residual_clusters",
}
HEARING_CATEGORIES = (
    "short-responses-and-interruptions",
    "speaker-changes",
    "overlap",
    "voice-cluster-stability",
    "unmatched-words-and-residual-clusters",
)
PROCEDURAL_BLINDING_LIMITATION = "procedural-blinding"


def diagnostic_listening_locations(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    location_sha256: Sequence[str],
) -> dict[str, Any]:
    """Собирает по изменениям диагностики места для прослушивания."""
    _exact(baseline, DIAGNOSTIC_KEYS, "diagnostic-schema")
    _exact(candidate, DIAGNOSTIC_KEYS, "diagnostic-schema")
    locations = sorted({_sha(value, "diagnostic-location") for value in location_sha256})
    changed = any(candidate[key] != baseline[key] for key in DIAGNOSTIC_KEYS)
    return {
        "mandatory_passed": True,
        "location_sha256": locations if changed else [],
    }


def build_blinded_review_material(
    *,
    diagnostic_locations: Mapping[str, Sequence[str]],
    matching_locations: Mapping[str, Sequence[str]],
    private_seed_sha256: str,
    matching_sample_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Готовит материал для процедурного ослепления A/B.

    Держатель ``private_seed_sha256`` может восстановить порядок вариантов,
    поэтому порядок раскрывают только после записи вердикта.
    """
    _sha(private_seed_sha256, "blinding-seed")
    if (
        set(diagnostic_locations) != set(HEARING_CATEGORIES)
        or set(matching_locations) != set(HEARING_CATEGORIES)
    ):
        raise PreflightError("blinding-category-set")
    if (
        isinstance(matching_sample_size, bool)
        or not isinstance(matching_sample_size, int)
        or matching_sample_size <= 0
    ):
        raise PreflightError("blinding-sample-size")
    categories = {}
    pair_order = {}
    for category in HEARING_CATEGORIES:
        diagnostic = sorted(
            {_sha(value, "blinding-location") for value in diagnostic_locations[category]}
        )
        matching = sorted(
            {
                _sha(value, "blinding-location")
                for value in matching_locations[category]
                if value not in diagnostic
            },
            key=lambda value: hashlib.sha256(
                f"{private_seed_sha256}:{category}:{value}".encode("ascii")
            ).digest(),
        )
        if len(matching) < matching_sample_size:
            raise PreflightError("blinding-matching-sample")
        selected = [
            *(('diagnostic', value) for value in diagnostic),
            *(('matching-sample', value) for value in matching[:matching_sample_size]),
        ]
        items = []
        for selection, location_sha256 in selected:
            pair_token = hashlib.sha256(
                f"pair:{category}:{location_sha256}".encode("ascii")
            ).hexdigest()
            order_digest = hashlib.sha256(
                (
                    f"order:{private_seed_sha256}:{category}:"
                    f"{location_sha256}"
                ).encode("ascii")
            ).digest()
            candidate_first = order_digest[0] % 2 == 0
            pair_order[pair_token] = (
                (CANDIDATE_RECIPE, W0_RECIPE)
                if candidate_first
                else (W0_RECIPE, CANDIDATE_RECIPE)
            )
            items.append(
                {
                    "location_sha256": location_sha256,
                    "selection": selection,
                    "pair_token": pair_token,
                    "variants": ["A", "B"],
                }
            )
        categories[category] = items
    reveal = {"pair_order": pair_order}
    blinded = {
        "categories": categories,
        "reveal_sha256": canonical_sha256(reveal),
    }
    privacy_scan(blinded)
    return blinded, reveal


def hearing_gate(decisions: Mapping[str, str]) -> dict[str, Any]:
    """Требует решения владельца по всем пяти слуховым категориям."""
    if set(decisions) != set(HEARING_CATEGORIES) or any(
        value not in {"pass", "fail"} for value in decisions.values()
    ):
        return {"state": "needs-user", "reason": "hearing-decisions-required"}
    if any(value == "fail" for value in decisions.values()):
        return {"state": "stop", "reason": "hearing-gate-failed"}
    return {"state": "go", "reason": None}


RSS_PLATEAU_ABS_BYTES = 64 * 1024**2
RSS_PLATEAU_RELATIVE = 0.05
RSS_PLATEAU_POLICY = (
    "fragment-only:last-delta<=max(64MiB,5%-of-max-post-input-rss);"
    "linear=all-deltas>same-tolerance"
)


def memory_gate(
    *,
    stage: str,
    oom: bool,
    swap_before_bytes: int,
    swap_after_bytes: int,
    peak_rss_bytes: int,
    rss_after_inputs: Sequence[int],
    machine_ram_bytes: int,
) -> dict[str, Any]:
    """Применяет локальную детерминированную политику плато RSS."""
    base = {"plateau_policy": RSS_PLATEAU_POLICY}
    if oom:
        return {"state": "stop", "reason": "oom", **base}
    if stage not in {"fragment", "full"}:
        raise ValueError("memory-stage")
    integer_values = (
        swap_before_bytes,
        swap_after_bytes,
        peak_rss_bytes,
        machine_ram_bytes,
        *rss_after_inputs,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in integer_values
    ):
        raise ValueError("memory-value")
    if stage == "fragment":
        if len(rss_after_inputs) != 3:
            return {"state": "stop", "reason": "rss-no-plateau", **base}
        tolerance = max(
            RSS_PLATEAU_ABS_BYTES,
            int(max(rss_after_inputs) * RSS_PLATEAU_RELATIVE),
        )
        deltas = [right - left for left, right in pairwise(rss_after_inputs)]
        plateau = abs(deltas[-1]) <= tolerance
        linear_growth = all(delta > tolerance for delta in deltas)
        if linear_growth:
            return {"state": "stop", "reason": "rss-linear-growth", **base}
        if not plateau:
            return {"state": "stop", "reason": "rss-no-plateau", **base}
    elif len(rss_after_inputs) != 1:
        raise ValueError("full-rss-sample-count")
    gib = 1024**3
    if 15 * gib <= machine_ram_bytes <= 17 * gib and peak_rss_bytes > 2 * gib:
        return {"state": "needs-user", "reason": "peak-rss-review", **base}
    return {"state": "pass", "reason": None, **base}


def _candidate_recipe_valid(candidate: Any) -> bool:
    try:
        _validate_candidate_recipe(candidate)
    except PreflightError:
        return False
    return True


def sortformer_gate(
    *,
    effective_recipe: Mapping[str, Any],
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    frame_count: int,
    frame_probs_start: int,
    segments: Sequence[Mapping[str, Any]],
    channel_interval_counts: Mapping[int, int],
    channel_durations_seconds: Mapping[int, float],
    inference_wall_seconds: float,
    post_processing_wall_seconds: float,
) -> dict[str, Any]:
    """Проверяет вывод и фактическую конфигурацию runtime Sortformer."""
    if not _candidate_recipe_valid(effective_recipe):
        return {"passed": False, "reason": "effective-config-mismatch"}
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count <= 0
        or isinstance(frame_probs_start, bool)
        or not isinstance(frame_probs_start, int)
        or frame_probs_start < 0
    ):
        return {"passed": False, "reason": "frame-metadata"}
    for row in probabilities:
        if not isinstance(row, Sequence) or len(row) != 4:
            return {"passed": False, "reason": "probability-shape"}
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in row
        ):
            return {"passed": False, "reason": "probability-value"}
    if frame_probs_start > frame_count or frame_count - frame_probs_start != len(probabilities):
        return {"passed": False, "reason": "frame-metadata"}
    if any(isinstance(label, bool) or not isinstance(label, int) or not 1 <= label <= 4 for label in labels):
        return {"passed": False, "reason": "speaker-label"}
    if set(channel_interval_counts) != {1, 2, 3, 4} or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in channel_interval_counts.values()
    ):
        return {"passed": False, "reason": "channel-counts"}
    if set(channel_durations_seconds) != {1, 2, 3, 4} or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
        for value in channel_durations_seconds.values()
    ):
        return {"passed": False, "reason": "channel-durations"}
    actual_counts = {speaker: 0 for speaker in range(1, 5)}
    actual_durations = {speaker: 0.0 for speaker in range(1, 5)}
    try:
        for segment in segments:
            if not isinstance(segment, Mapping) or set(segment) != {
                "start",
                "end",
                "speaker",
            }:
                raise ValueError
            speaker = segment["speaker"]
            start = float(segment["start"])
            end = float(segment["end"])
            if (
                isinstance(speaker, bool)
                or not isinstance(speaker, int)
                or speaker not in actual_counts
                or not math.isfinite(start)
                or not math.isfinite(end)
                or end <= start
            ):
                raise ValueError
            actual_counts[speaker] += 1
            actual_durations[speaker] += end - start
    except (KeyError, TypeError, ValueError):
        return {"passed": False, "reason": "invalid-segments"}
    if dict(channel_interval_counts) != actual_counts:
        return {"passed": False, "reason": "channel-counts"}
    if any(
        not math.isclose(
            float(channel_durations_seconds[speaker]),
            actual_durations[speaker],
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        for speaker in actual_durations
    ):
        return {"passed": False, "reason": "channel-durations"}
    for wall in (inference_wall_seconds, post_processing_wall_seconds):
        if (
            isinstance(wall, bool)
            or not isinstance(wall, (int, float))
            or not math.isfinite(float(wall))
            or float(wall) < 0
        ):
            return {"passed": False, "reason": "invalid-stage-wall"}
    return {"passed": True, "reason": None}


class MeasurementAdapter(Protocol):
    """Измеряет переданную операцию на границе с системой."""

    def measure(
        self, label: str, operation: Callable[[], Any]
    ) -> tuple[Any, Mapping[str, float | int]]: ...


class ProcessMeasurementAdapter:
    """Измеряет wall time, CPU и пиковый RSS всего процесса."""

    def __init__(self, interval_seconds: float):
        self.interval_seconds = interval_seconds

    def measure(
        self, label: str, operation: Callable[[], Any]
    ) -> tuple[Any, Mapping[str, float | int]]:
        del label
        import psutil

        from .experimental_diarization import measure_call

        return measure_call(
            operation,
            psutil.Process(os.getpid()),
            self.interval_seconds,
        )


def measure_w0(
    diarizer: Any, samples: Any, measurement: MeasurementAdapter
) -> dict[str, Any]:
    """Ставит таймер ровно вокруг одного ``diarizer.process(samples)``."""
    result, metrics = measurement.measure(
        "w0-process", lambda: diarizer.process(samples)
    )
    return {"result": result, "metrics": dict(metrics)}


def measure_candidate(
    runtime: Any,
    samples: Any,
    post_process: Callable[[Any], Any],
    measurement: MeasurementAdapter,
) -> dict[str, Any]:
    """Включает операции stream и post-processing в wall time кандидата."""
    prepared = runtime.prepare_samples(samples)

    def infer() -> Any:
        stream = runtime.stream_open()
        try:
            runtime.stream_push(stream, prepared)
            return stream, runtime.stream_finish(stream)
        except Exception:
            runtime.stream_close(stream)
            raise

    (stream, raw), inference_metrics = measurement.measure(
        "candidate-inference", infer
    )

    def finish_post_processing() -> Any:
        try:
            return post_process(raw)
        finally:
            runtime.stream_close(stream)

    result, post_metrics = measurement.measure(
        "candidate-post-processing", finish_post_processing
    )
    inference = dict(inference_metrics)
    post = dict(post_metrics)
    wall = float(inference["wall_seconds"]) + float(post["wall_seconds"])
    metrics: dict[str, float | int] = {"wall_seconds": wall}
    for key in ("user_cpu_seconds", "system_cpu_seconds", "total_cpu_seconds"):
        if key in inference and key in post:
            metrics[key] = float(inference[key]) + float(post[key])
    if "peak_rss_bytes" in inference and "peak_rss_bytes" in post:
        metrics["peak_rss_bytes"] = max(
            int(inference["peak_rss_bytes"]), int(post["peak_rss_bytes"])
        )
    if "total_cpu_seconds" in metrics:
        metrics["average_cpu_cores"] = (
            float(metrics["total_cpu_seconds"]) / wall if wall else 0.0
        )
    return {
        "result": result,
        "metrics": metrics,
        "inference_wall_seconds": float(inference["wall_seconds"]),
        "post_processing_wall_seconds": float(post["wall_seconds"]),
    }


GUARD_KEYS = {
    "machine_sha256",
    "affinity",
    "logical_cores",
    "physical_cores",
    "cpuid_isa_sha256",
    "ram_bytes",
    "os_sha256",
    "compiler_sha256",
    "runtime_versions_sha256",
    "recipe",
    "effective_threads",
    "power_mode",
    "governor",
    "turbo",
    "throttling",
    "throttle_count",
    "swap_used_bytes",
    "swap_sin_bytes",
    "swap_sout_bytes",
}
GUARD_STATIC_KEYS = {
    "machine_sha256",
    "affinity",
    "logical_cores",
    "physical_cores",
    "cpuid_isa_sha256",
    "ram_bytes",
    "os_sha256",
    "compiler_sha256",
    "runtime_versions_sha256",
}


def _validate_guard(value: Mapping[str, Any]) -> None:
    _exact(value, GUARD_KEYS, "runtime-guard-schema")
    for key in (
        "machine_sha256",
        "cpuid_isa_sha256",
        "os_sha256",
        "compiler_sha256",
        "runtime_versions_sha256",
    ):
        _sha(value[key], "runtime-guard-hash")
    if value["physical_cores"] != 8 or _positive_int(
        value["logical_cores"], "runtime-logical-cores"
    ) < 8:
        raise PreflightError("runtime-affinity")
    affinity = value["affinity"]
    if (
        not isinstance(affinity, list)
        or len(affinity) != 8
        or len(set(affinity)) != 8
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in affinity
        )
    ):
        raise PreflightError("runtime-affinity")
    if value["recipe"] not in {W0_RECIPE, CANDIDATE_RECIPE}:
        raise PreflightError("runtime-recipe")
    if (
        isinstance(value["effective_threads"], bool)
        or not isinstance(value["effective_threads"], int)
        or value["effective_threads"] <= 0
    ):
        raise PreflightError("runtime-effective-threads")
    if not isinstance(value["turbo"], bool) or not isinstance(value["throttling"], bool):
        raise PreflightError("runtime-guard-type")
    if any(
        not isinstance(value[key], str) or not value[key]
        for key in ("power_mode", "governor")
    ):
        raise PreflightError("runtime-guard-type")
    for key in (
        "ram_bytes",
        "throttle_count",
        "swap_used_bytes",
        "swap_sin_bytes",
        "swap_sout_bytes",
    ):
        if isinstance(value[key], bool) or not isinstance(value[key], int) or value[key] < 0:
            raise PreflightError("runtime-guard-type")


def _manifest_guard_reasons(
    snapshot: Mapping[str, Any],
    expected_machine: Mapping[str, Any],
    *,
    require_power: bool = True,
) -> list[str]:
    _validate_guard(snapshot)
    _validate_machine(expected_machine)
    expected_keys = MACHINE_KEYS & GUARD_STATIC_KEYS
    if require_power:
        expected_keys |= {"power_mode", "governor", "turbo"}
    if any(snapshot[key] != expected_machine[key] for key in expected_keys):
        return ["manifest-machine-mismatch"]
    expected_threads = 8 if snapshot["recipe"] == W0_RECIPE else 4
    if snapshot["effective_threads"] != expected_threads:
        return ["effective-threads-mismatch"]
    return []


def _guard_transition_reasons(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> list[str]:
    _validate_guard(before)
    _validate_guard(after)
    reasons = []
    if any(before[key] != after[key] for key in GUARD_STATIC_KEYS):
        reasons.append("machine-condition-changed")
    if before["recipe"] != after["recipe"] or before["effective_threads"] != after["effective_threads"]:
        reasons.append("effective-threads-mismatch")
    if before["power_mode"] != after["power_mode"]:
        reasons.append("power-mode-changed")
    if before["governor"] != after["governor"] or before["turbo"] != after["turbo"]:
        reasons.append("runtime-power-setting-changed")
    if after["throttling"] is True or after["throttle_count"] > before["throttle_count"]:
        reasons.append("new-throttling")
    return reasons


def _swap_active(before: Mapping[str, Any], after: Mapping[str, Any] | None = None) -> bool:
    if after is None:
        return False
    return bool(
        after["swap_sin_bytes"] != before["swap_sin_bytes"]
        or after["swap_sout_bytes"] != before["swap_sout_bytes"]
    )


def _safe_guard_read(
    guard_reader: Callable[[], Mapping[str, Any]],
) -> dict[str, Any] | None:
    try:
        snapshot = dict(guard_reader())
        _validate_guard(snapshot)
    except Exception:  # noqa: BLE001 - вызов внешней системы
        return None
    return snapshot


def validate_pair_guards(
    w0_before: Mapping[str, Any],
    w0_after: Mapping[str, Any],
    candidate_before: Mapping[str, Any],
    candidate_after: Mapping[str, Any],
    *,
    expected_machine: Mapping[str, Any],
) -> list[str]:
    """Сверяет машину, affinity и закрепленное число потоков для пары."""
    snapshots = (w0_before, w0_after, candidate_before, candidate_after)
    for snapshot in snapshots:
        _validate_guard(snapshot)
    reasons = []
    for snapshot in snapshots:
        for reason in _manifest_guard_reasons(snapshot, expected_machine):
            if reason not in reasons:
                reasons.append(reason)
    affinity = w0_before["affinity"]
    if any(snapshot["affinity"] != affinity for snapshot in snapshots[1:]):
        reasons.append("pair-affinity-mismatch")
    identity_keys = GUARD_STATIC_KEYS - {"affinity"}
    if any(
        snapshot[key] != w0_before[key]
        for snapshot in snapshots[1:]
        for key in identity_keys
    ):
        reasons.append("pair-machine-mismatch")
    power_keys = {"power_mode", "governor", "turbo"}
    if any(
        snapshot[key] != w0_before[key]
        for snapshot in snapshots[1:]
        for key in power_keys
    ):
        reasons.append("pair-power-mismatch")
    if (
        w0_before["recipe"] != W0_RECIPE
        or w0_after["recipe"] != W0_RECIPE
        or w0_before["effective_threads"] != 8
        or w0_after["effective_threads"] != 8
        or candidate_before["recipe"] != CANDIDATE_RECIPE
        or candidate_after["recipe"] != CANDIDATE_RECIPE
        or candidate_before["effective_threads"] != 4
        or candidate_after["effective_threads"] != 4
    ):
        reasons.append("effective-threads-mismatch")
    return reasons


def run_guarded_cell(
    request: Mapping[str, Any],
    *,
    runner: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    guard_reader: Callable[[], Mapping[str, Any]],
    stabilize: Callable[[], None],
    expected_machine: Mapping[str, Any],
) -> dict[str, Any]:
    """Отбрасывает неверный wall time и разрешает один повтор после стабилизации."""
    attempts = []
    first_reason = None
    for attempt_number in (1, 2):
        before = _safe_guard_read(guard_reader)
        if before is None:
            return {
                "state": "stop",
                "reason": "runtime-guard-unavailable",
                "attempts": attempts,
                "result_sha256": canonical_sha256(
                    {"reason": "runtime-guard-unavailable", "attempt": attempt_number}
                ),
            }
        if request.get("recipe") != before["recipe"]:
            return {
                "state": "stop",
                "reason": "runtime-recipe-mismatch",
                "attempts": attempts,
                "result_sha256": canonical_sha256(
                    {"reason": "runtime-recipe-mismatch", "attempt": attempt_number}
                ),
            }
        manifest_reasons = _manifest_guard_reasons(before, expected_machine)
        if manifest_reasons:
            reason = manifest_reasons[0]
            return {
                "state": "stop",
                "reason": reason,
                "attempts": attempts,
                "result_sha256": canonical_sha256(
                    {"reason": reason, "attempt": attempt_number}
                ),
            }
        if _swap_active(before):
            return {
                "state": "stop",
                "reason": "active-swap",
                "attempts": attempts,
                "result_sha256": canonical_sha256(
                    {"reason": "active-swap", "attempt": attempt_number}
                ),
            }
        try:
            result = dict(runner(request))
        except MemoryError:
            return {
                "state": "stop",
                "reason": "oom",
                "attempts": attempts,
                "result_sha256": canonical_sha256(
                    {"reason": "oom", "attempt": attempt_number}
                ),
            }
        except Exception:  # noqa: BLE001 - вызов внешнего runtime
            return {
                "state": "stop",
                "reason": "runtime-error",
                "attempts": attempts,
                "result_sha256": canonical_sha256(
                    {"reason": "runtime-error", "attempt": attempt_number}
                ),
            }
        try:
            result_sha256 = canonical_sha256(result)
        except (TypeError, ValueError):
            return {
                "state": "stop",
                "reason": "invalid-result-value",
                "attempts": attempts,
                "result_sha256": canonical_sha256({"invalid_result": True}),
            }
        after = _safe_guard_read(guard_reader)
        if after is None:
            attempts.append(
                {
                    "attempt": attempt_number,
                    "before": before,
                    "valid": False,
                    "reasons": ["runtime-guard-unavailable"],
                    "invalid_result_sha256": result_sha256,
                }
            )
            return {
                "state": "stop",
                "reason": "runtime-guard-unavailable",
                "attempts": attempts,
                "result_sha256": result_sha256,
            }
        manifest_reasons = _manifest_guard_reasons(
            after, expected_machine, require_power=False
        )
        if manifest_reasons:
            reason = manifest_reasons[0]
            attempts.append(
                {
                    "attempt": attempt_number,
                    "before": before,
                    "after": after,
                    "valid": False,
                    "reasons": manifest_reasons,
                    "invalid_result_sha256": result_sha256,
                }
            )
            return {
                "state": "stop",
                "reason": reason,
                "attempts": attempts,
                "result_sha256": result_sha256,
            }
        if _swap_active(before, after):
            attempts.append(
                {
                    "attempt": attempt_number,
                    "before": before,
                    "after": after,
                    "valid": False,
                    "reasons": ["active-swap"],
                    "invalid_result_sha256": result_sha256,
                }
            )
            return {
                "state": "stop",
                "reason": "active-swap",
                "attempts": attempts,
                "result_sha256": result_sha256,
            }
        reasons = _guard_transition_reasons(before, after)
        if not reasons:
            attempts.append(
                {
                    "attempt": attempt_number,
                    "before": before,
                    "after": after,
                    "valid": True,
                }
            )
            return {
                "state": "accepted",
                "result": result,
                "result_sha256": result_sha256,
                "attempts": attempts,
            }
        first_reason = first_reason or reasons[0]
        attempts.append(
            {
                "attempt": attempt_number,
                "before": before,
                "after": after,
                "valid": False,
                "reasons": reasons,
                "invalid_result_sha256": result_sha256,
            }
        )
        retryable = set(reasons) <= {"power-mode-changed", "new-throttling"}
        if attempt_number == 1 and retryable:
            try:
                stabilize()
            except Exception:  # noqa: BLE001 - вызов внешней системы
                return {
                    "state": "stop",
                    "reason": "stabilization-error",
                    "attempts": attempts,
                    "result_sha256": canonical_sha256(
                        {"reason": "stabilization-error", "attempts": attempts}
                    ),
                }
        else:
            break
    return {
        "state": "stop",
        "reason": first_reason,
        "attempts": attempts,
        "result_sha256": canonical_sha256(
            {"reason": first_reason, "attempts": attempts}
        ),
    }


class PrivacyError(ValueError):
    """Публичные данные содержат закрытое поле или значение."""


FORBIDDEN_PUBLIC_KEYS = {
    "path",
    "paths",
    "recording_id",
    "source_id",
    "text",
    "interval",
    "intervals",
    "raw",
    "raw_rows",
    "log",
    "logs",
    "participant",
    "participant_name",
    "manifest",
    "audio",
}
FORBIDDEN_PUBLIC_SUFFIXES = (
    "_path",
    "_paths",
    "_text",
    "_interval",
    "_intervals",
    "_raw",
    "_rows",
    "_log",
    "_logs",
)
ABSOLUTE_PATH_RE = re.compile(
    r"(?:^|[\s\"'=(])/(?!/)|(?:^|[\s\"'=(])[A-Za-z]:[\\/]"
)


def privacy_scan(value: Any) -> None:
    """Отклоняет закрытые поля и строки с абсолютными путями."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in FORBIDDEN_PUBLIC_KEYS or normalized.endswith(
                FORBIDDEN_PUBLIC_SUFFIXES
            ):
                raise PrivacyError("forbidden-public-field")
            privacy_scan(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            privacy_scan(item)
    elif isinstance(value, str) and ABSOLUTE_PATH_RE.search(value):
        raise PrivacyError("forbidden-public-content")


def safe_public_plan(
    manifest: Mapping[str, Any], outcome: PreflightOutcome
) -> dict[str, Any]:
    """Готовит обезличенный план validate-only без путей, ID и содержимого."""
    candidate = manifest["recipes"][CANDIDATE_RECIPE]
    w0_public = copy.deepcopy(W0_PIN)
    result = {
        "state": outcome.state,
        "reason": outcome.reason,
        "schema": SCHEMA,
        "schema_status": "pass",
        "fresh_basis_status": (
            "pass" if outcome.state == "valid" else "needs-user"
        ),
        "pin_status": "pass",
        "schedule_status": "pass",
        "gate_status": "pending",
        "recipes": [W0_RECIPE, CANDIDATE_RECIPE],
        "recipe_pins": {
            W0_RECIPE: w0_public,
            CANDIDATE_RECIPE: copy.deepcopy(candidate),
        },
        "fragment_count": outcome.fragment_count,
        "fragment_duration_seconds": 300.0,
        "full_recording_count": outcome.full_recording_count,
        "scheduled_cell_count": len(manifest["schedule"]),
        "semantic_cell_sha256": [
            hashlib.sha256(cell["semantic_id"].encode("ascii")).hexdigest()
            for cell in manifest["schedule"]
        ],
        "input_sha256": [
            item["fragment"]["sha256"] for item in manifest["inputs"]
        ],
        "sidecar_sha256": [
            item["fragment"]["sidecar_sha256"] for item in manifest["inputs"]
        ],
        "source_commit_sha256": outcome.source_commit_sha256,
    }
    privacy_scan(result)
    return result


def safe_public_result(
    *,
    source_commit: str,
    result_sha256: str,
    aggregate_timings: Mapping[str, float],
    deltas: Mapping[str, float],
    statistics_values: Mapping[str, float],
    gate_verdicts: Mapping[str, str],
    limitations: Sequence[str],
    first_stop_reason: str | None,
) -> dict[str, Any]:
    """Оставляет в публичном результате разрешенные агрегаты."""
    _commit(source_commit, "source-commit")
    _sha(result_sha256, "result-sha256")
    numeric_groups = {
        "aggregate_timings": aggregate_timings,
        "deltas": deltas,
        "statistics": statistics_values,
    }
    for group in numeric_groups.values():
        if not isinstance(group, Mapping):
            raise PreflightError("public-numeric-group")
        for key, value in group.items():
            if not isinstance(key, str) or re.fullmatch(r"[a-z][a-z0-9_]*", key) is None:
                raise PreflightError("public-numeric-key")
            _finite(value, "public-numeric-value")
    if any(
        not isinstance(value, str)
        or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value) is None
        for value in limitations
    ):
        raise PreflightError("public-limitation")
    public_limitations = list(
        dict.fromkeys([*limitations, PROCEDURAL_BLINDING_LIMITATION])
    )
    if first_stop_reason is not None:
        first_stop_reason = sanitize_reason(first_stop_reason)
    verdicts = dict(gate_verdicts)
    if any(
        not isinstance(key, str)
        or re.fullmatch(r"[a-z][a-z0-9-]*", key) is None
        for key in verdicts
    ):
        raise PreflightError("gate-verdict")
    allowed_verdicts = {"pass", "fail", "needs-user", "pending"}
    if any(value not in allowed_verdicts for value in verdicts.values()):
        raise PreflightError("gate-verdict")
    required_gates = {"quality", "sortformer", "memory", "measurement", "hearing"}
    median = statistics_values.get("median_speedup_percent")
    if median is not None:
        median = _finite(median, "final-speedup")
    if first_stop_reason is not None or "fail" in verdicts.values():
        state = "stop"
        if first_stop_reason is None:
            first_stop_reason = "mandatory-gate-failed"
    elif median is not None and median < 30.0:
        state = "stop"
        first_stop_reason = "below-final-threshold"
    elif (
        required_gates.issubset(verdicts)
        and all(verdicts[gate] == "pass" for gate in required_gates)
        and median is not None
        and median >= 30.0
    ):
        state = "go"
    else:
        state = "needs-user"
    result = {
        "state": state,
        "recipes": [W0_RECIPE, CANDIDATE_RECIPE],
        "source_commit": source_commit,
        "result_sha256": result_sha256,
        "aggregate_timings": dict(aggregate_timings),
        "deltas": dict(deltas),
        "statistics": dict(statistics_values),
        "gate_verdicts": verdicts,
        "limitations": public_limitations,
        "first_stop_reason": first_stop_reason,
    }
    privacy_scan(result)
    return result


def serialize_public(value: Mapping[str, Any]) -> str:
    """Сканирует и сериализует публичный JSON детерминированно."""
    privacy_scan(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def git_environment_probe(repository_root: Path) -> dict[str, Any]:
    """Возвращает commit и признак чистого дерева без имен измененных файлов."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if commit.returncode != 0 or status.returncode != 0:
        raise PreflightError("source-environment-unavailable")
    return {"commit": commit.stdout.strip(), "clean": not status.stdout}


def run_cli(
    args: Any,
    *,
    environment_probe: Callable[[], Mapping[str, Any]] | None = None,
    w0_loader: Callable[[], Any] = lambda: None,
    candidate_loader: Callable[[], Any] = lambda: None,
) -> int:
    """Печатает обезличенный план validate-only или загружает runtime."""
    require_closed_store(args.manifest)
    raw = json.loads(args.manifest.read_text(encoding="utf-8"))
    probe = environment_probe or (
        lambda: git_environment_probe(Path(__file__).parents[2])
    )
    outcome = preflight_and_load(
        raw,
        root=args.manifest.parent,
        environment_probe=probe,
        w0_loader=w0_loader,
        candidate_loader=candidate_loader,
        validate_only=args.validate_only,
    )
    print(serialize_public(safe_public_plan(raw, outcome)), end="")
    return 0


WORKER_SCHEMA = "streaming-sortformer-cell.v1"
WORKER_REQUEST_KEYS = {
    "worker_schema",
    "semantic_id",
    "recipe",
    "input_sha256",
    "sidecar_sha256",
    "response_path",
}
WORKER_RESULT_KEYS = {
    "worker_schema",
    "semantic_id",
    "status",
    "result",
    "result_sha256",
}


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as target:
            temporary = Path(target.name)
            target.write(value)
            target.flush()
            os.fsync(target.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_child_process(
    request: Mapping[str, Any],
    *,
    work_dir: Path,
    process_runner: Callable[[Sequence[str]], Any] | None = None,
) -> dict[str, Any]:
    """Передает одну semantic cell отдельному процессу."""
    require_closed_store(work_dir)
    if process_runner is None:
        raise PreflightError("worker-runner-required")
    _exact(
        request,
        WORKER_REQUEST_KEYS - {"worker_schema", "response_path"},
        "worker-request-schema",
    )
    semantic_id = request["semantic_id"]
    if not isinstance(semantic_id, str) or re.fullmatch(r"[a-z0-9-]+", semantic_id) is None:
        raise PreflightError("worker-semantic-id")
    if request["recipe"] not in {W0_RECIPE, CANDIDATE_RECIPE}:
        raise PreflightError("worker-recipe-not-allowed")
    for key in ("input_sha256", "sidecar_sha256"):
        values = request[key]
        if not isinstance(values, list) or not values:
            raise PreflightError("worker-input-hashes")
        for value in values:
            _sha(value, "worker-input-hashes")
    request_dir = work_dir / "requests"
    response_path = request_dir / f"{semantic_id}.result.json"
    request_path = request_dir / f"{semantic_id}.json"
    response_path.unlink(missing_ok=True)
    payload = {
        "worker_schema": WORKER_SCHEMA,
        **dict(request),
        "response_path": str(response_path),
    }
    _atomic_json(request_path, payload)
    script = Path(__file__).parents[2] / "scripts" / "benchmarks" / "streaming_sortformer_experiment.py"
    command = [sys.executable, str(script), "--worker", str(request_path)]
    completed = process_runner(command)
    _atomic_text(work_dir / "logs" / f"{semantic_id}.stdout.log", completed.stdout)
    _atomic_text(work_dir / "logs" / f"{semantic_id}.stderr.log", completed.stderr)
    if completed.returncode != 0 or not response_path.is_file():
        raise RuntimeError("child-process-failed")
    result = json.loads(response_path.read_text(encoding="utf-8"))
    _exact(result, WORKER_RESULT_KEYS, "worker-result-schema")
    if (
        result["worker_schema"] != WORKER_SCHEMA
        or result["semantic_id"] != semantic_id
        or result["status"] != "complete"
        or result["result_sha256"] != canonical_sha256(result["result"])
    ):
        raise RuntimeError("child-result-invalid")
    return dict(result)


class CAbiError(RuntimeError):
    """Ошибка несоответствия закрепленному C ABI standalone runtime."""


class SortformerModelConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("model_path", ctypes.c_char_p),
        ("gpu", ctypes.c_int32),
        ("preset", ctypes.c_char_p),
        ("chunk_frames", ctypes.c_int32),
        ("right_context_frames", ctypes.c_int32),
        ("left_context_frames", ctypes.c_int32),
        ("fifo_frames", ctypes.c_int32),
        ("spkcache_frames", ctypes.c_int32),
        ("update_period_frames", ctypes.c_int32),
    ]


class SortformerSegmentationConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("onset", ctypes.c_float),
        ("offset", ctypes.c_float),
        ("pad_onset_sec", ctypes.c_double),
        ("pad_offset_sec", ctypes.c_double),
        ("min_gap_sec", ctypes.c_double),
        ("min_duration_sec", ctypes.c_double),
    ]


class SortformerSegment(ctypes.Structure):
    _fields_ = [
        ("start_time", ctypes.c_double),
        ("end_time", ctypes.c_double),
        ("speaker", ctypes.c_int32),
    ]


@dataclass(frozen=True)
class PreparedFloat32Samples:
    """Хранит владельца заранее подготовленного mono float32 буфера."""

    pointer: Any
    count: int
    owner: Any


ABI_LAYOUT = {
    "model_config_size": 56,
    "model_size": 0,
    "model_path": 8,
    "model_gpu": 16,
    "model_preset": 24,
    "model_chunk_frames": 32,
    "model_right_context_frames": 36,
    "model_left_context_frames": 40,
    "model_fifo_frames": 44,
    "model_spkcache_frames": 48,
    "model_update_period_frames": 52,
    "segmentation_config_size": 48,
    "segmentation_size": 0,
    "segmentation_onset": 8,
    "segmentation_offset": 12,
    "segmentation_pad_onset_sec": 16,
    "segmentation_pad_offset_sec": 24,
    "segmentation_min_gap_sec": 32,
    "segmentation_min_duration_sec": 40,
    "segment_size": 24,
    "segment_start_time": 0,
    "segment_end_time": 8,
    "segment_speaker": 16,
}


class SortformerCAbi:
    """Ленивый адаптер к standalone C ABI диаризации NeMo-Speech.cpp."""

    def __init__(
        self, library: Any, *, expected_layout: Mapping[str, int] | None = None
    ):
        self.library = library
        expected = ABI_LAYOUT if expected_layout is None else expected_layout
        actual = {
            "model_config_size": ctypes.sizeof(SortformerModelConfig),
            "model_size": SortformerModelConfig.size.offset,
            "model_path": SortformerModelConfig.model_path.offset,
            "model_gpu": SortformerModelConfig.gpu.offset,
            "model_preset": SortformerModelConfig.preset.offset,
            "model_chunk_frames": SortformerModelConfig.chunk_frames.offset,
            "model_right_context_frames": SortformerModelConfig.right_context_frames.offset,
            "model_left_context_frames": SortformerModelConfig.left_context_frames.offset,
            "model_fifo_frames": SortformerModelConfig.fifo_frames.offset,
            "model_spkcache_frames": SortformerModelConfig.spkcache_frames.offset,
            "model_update_period_frames": SortformerModelConfig.update_period_frames.offset,
            "segmentation_config_size": ctypes.sizeof(SortformerSegmentationConfig),
            "segmentation_size": SortformerSegmentationConfig.size.offset,
            "segmentation_onset": SortformerSegmentationConfig.onset.offset,
            "segmentation_offset": SortformerSegmentationConfig.offset.offset,
            "segmentation_pad_onset_sec": SortformerSegmentationConfig.pad_onset_sec.offset,
            "segmentation_pad_offset_sec": SortformerSegmentationConfig.pad_offset_sec.offset,
            "segmentation_min_gap_sec": SortformerSegmentationConfig.min_gap_sec.offset,
            "segmentation_min_duration_sec": SortformerSegmentationConfig.min_duration_sec.offset,
            "segment_size": ctypes.sizeof(SortformerSegment),
            "segment_start_time": SortformerSegment.start_time.offset,
            "segment_end_time": SortformerSegment.end_time.offset,
            "segment_speaker": SortformerSegment.speaker.offset,
        }
        if any(actual.get(key) != value for key, value in expected.items()):
            raise CAbiError("c-abi-layout-mismatch")
        self._configure_signatures()
        self._model: ctypes.c_void_p | None = None

    def _configure_signatures(self) -> None:
        signatures = {
            "nemo_speech_diar_create": (
                [
                    ctypes.POINTER(SortformerModelConfig),
                    ctypes.POINTER(ctypes.c_void_p),
                ],
                ctypes.c_int32,
            ),
            "nemo_speech_diar_destroy": ([ctypes.c_void_p], None),
            "nemo_speech_diar_num_speakers": ([ctypes.c_void_p], ctypes.c_int32),
            "nemo_speech_diar_seconds_per_frame": (
                [ctypes.c_void_p],
                ctypes.c_double,
            ),
            "nemo_speech_diar_stream_open": (
                [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)],
                ctypes.c_int32,
            ),
            "nemo_speech_diar_stream_push_f32": (
                [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_size_t,
                    ctypes.c_int32,
                ],
                ctypes.c_int32,
            ),
            "nemo_speech_diar_stream_finish": ([ctypes.c_void_p], ctypes.c_int32),
            "nemo_speech_diar_stream_close": ([ctypes.c_void_p], None),
            "nemo_speech_diar_frame_count": ([ctypes.c_void_p], ctypes.c_int64),
            "nemo_speech_diar_frame_probs_start": (
                [ctypes.c_void_p],
                ctypes.c_int64,
            ),
            "nemo_speech_diar_frame_probs": (
                [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_size_t,
                ],
                ctypes.c_int32,
            ),
            "nemo_speech_diar_segments": (
                [
                    ctypes.c_void_p,
                    ctypes.POINTER(SortformerSegmentationConfig),
                    ctypes.POINTER(SortformerSegment),
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_size_t),
                ],
                ctypes.c_int32,
            ),
        }
        for name, (argtypes, restype) in signatures.items():
            try:
                function = getattr(self.library, name)
            except AttributeError as exc:
                raise CAbiError("c-abi-symbol-missing") from exc
            try:
                function.argtypes = argtypes
                function.restype = restype
            except AttributeError as exc:
                if isinstance(self.library, ctypes.CDLL):
                    raise CAbiError("c-abi-signature-unavailable") from exc

    @classmethod
    def load(
        cls,
        library_path: Path,
        *,
        loader: Callable[[str], Any] | None = None,
    ) -> SortformerCAbi:
        """Подключает библиотеку через ``ctypes.CDLL`` на границе runtime."""
        factory = ctypes.CDLL if loader is None else loader
        library = factory(str(library_path))
        return cls(library)

    @staticmethod
    def frozen_model_config(model_path: str) -> SortformerModelConfig:
        geometry = CANDIDATE_FIXED_PIN["geometry"]
        return SortformerModelConfig(
            size=ctypes.sizeof(SortformerModelConfig),
            model_path=model_path.encode("utf-8"),
            gpu=-1,
            preset=CANDIDATE_FIXED_PIN["diarization_preset"].encode("ascii"),
            **geometry,
        )

    @staticmethod
    def frozen_segmentation_config() -> SortformerSegmentationConfig:
        post = CANDIDATE_FIXED_PIN["post_processing"]
        return SortformerSegmentationConfig(
            size=ctypes.sizeof(SortformerSegmentationConfig),
            onset=post["onset"],
            offset=post["offset"],
            pad_onset_sec=post["pad_onset_sec"],
            pad_offset_sec=post["pad_offset_sec"],
            min_gap_sec=post["min_gap_sec"],
            min_duration_sec=post["min_duration_sec"],
        )

    def create_model(self, model_path: str) -> ctypes.c_void_p:
        if self._model is not None:
            raise CAbiError("c-abi-model-already-created")
        config = self.frozen_model_config(model_path)
        model = ctypes.c_void_p()
        status = int(self.library.nemo_speech_diar_create(ctypes.pointer(config), ctypes.byref(model)))
        if status != 0 or not model.value:
            raise CAbiError("c-abi-effective-config-mismatch")
        self._model = model
        return model

    def destroy_model(self, model: ctypes.c_void_p) -> None:
        self.library.nemo_speech_diar_destroy(model)
        if self._model is not None and self._model.value == model.value:
            self._model = None

    def stream_open(self, model: ctypes.c_void_p | None = None) -> ctypes.c_void_p:
        model = self._model if model is None else model
        if model is None:
            raise CAbiError("c-abi-model-not-created")
        stream = ctypes.c_void_p()
        status = int(self.library.nemo_speech_diar_stream_open(model, ctypes.byref(stream)))
        if status != 0 or not stream.value:
            raise CAbiError("c-abi-stream-open-failed")
        return stream

    @staticmethod
    def prepare_samples(samples: Any) -> PreparedFloat32Samples:
        """Проверяет zero-copy буфер и закрепляет его указатель вне таймера."""
        try:
            dtype = str(samples.dtype)
            ndim = int(samples.ndim)
            count = int(samples.size)
            contiguous = bool(samples.flags.c_contiguous)
            address = int(samples.ctypes.data)
        except (AttributeError, TypeError, ValueError) as exc:
            raise CAbiError("c-abi-samples-not-float32-buffer") from exc
        if dtype != "float32" or ndim != 1 or not contiguous or count <= 0 or address <= 0:
            raise CAbiError("c-abi-samples-not-float32-buffer")
        pointer = ctypes.cast(address, ctypes.POINTER(ctypes.c_float))
        return PreparedFloat32Samples(pointer=pointer, count=count, owner=samples)

    def stream_push(
        self,
        stream: ctypes.c_void_p,
        samples: PreparedFloat32Samples,
        *,
        sample_rate: int = 16_000,
    ) -> None:
        if not isinstance(samples, PreparedFloat32Samples) or sample_rate != 16_000:
            raise CAbiError("c-abi-samples-not-float32-buffer")
        status = int(
            self.library.nemo_speech_diar_stream_push_f32(
                stream, samples.pointer, samples.count, sample_rate
            )
        )
        if status != 0:
            raise CAbiError("c-abi-stream-push-failed")

    def stream_finish(self, stream: ctypes.c_void_p) -> ctypes.c_void_p:
        if int(self.library.nemo_speech_diar_stream_finish(stream)) != 0:
            raise CAbiError("c-abi-stream-finish-failed")
        return stream

    def stream_close(self, stream: ctypes.c_void_p) -> None:
        self.library.nemo_speech_diar_stream_close(stream)

    def collect_result(
        self, stream: ctypes.c_void_p, model: ctypes.c_void_p | None = None
    ) -> dict[str, Any]:
        model = self._model if model is None else model
        if model is None:
            raise CAbiError("c-abi-model-not-created")
        frame_count = int(self.library.nemo_speech_diar_frame_count(stream))
        frame_start = int(self.library.nemo_speech_diar_frame_probs_start(stream))
        speakers = int(self.library.nemo_speech_diar_num_speakers(model))
        retained = frame_count - frame_start
        if frame_count < 0 or frame_start < 0 or retained < 0 or speakers != 4:
            raise CAbiError("c-abi-frame-metadata-invalid")
        probability_count = retained * speakers
        probability_buffer = (ctypes.c_float * probability_count)()
        if int(
            self.library.nemo_speech_diar_frame_probs(
                stream, probability_buffer, probability_count
            )
        ) != 0:
            raise CAbiError("c-abi-frame-probs-failed")
        probabilities = [
            [
                float(probability_buffer[frame * speakers + speaker])
                for speaker in range(speakers)
            ]
            for frame in range(retained)
        ]
        segmentation = self.frozen_segmentation_config()
        segment_count = ctypes.c_size_t()
        if int(
            self.library.nemo_speech_diar_segments(
                stream,
                ctypes.pointer(segmentation),
                None,
                0,
                ctypes.byref(segment_count),
            )
        ) != 0:
            raise CAbiError("c-abi-segment-count-failed")
        segment_capacity = segment_count.value
        segment_buffer = (SortformerSegment * segment_capacity)()
        if int(
            self.library.nemo_speech_diar_segments(
                stream,
                ctypes.pointer(segmentation),
                segment_buffer,
                segment_capacity,
                ctypes.byref(segment_count),
            )
        ) != 0:
            raise CAbiError("c-abi-segments-failed")
        if segment_count.value > segment_capacity:
            raise CAbiError("c-abi-segment-count-changed")
        segments = [
            {
                "start": float(item.start_time),
                "end": float(item.end_time),
                "speaker": int(item.speaker),
            }
            for item in segment_buffer[: segment_count.value]
        ]
        return {
            "frame_count": frame_count,
            "frame_probs_start": frame_start,
            "probabilities": probabilities,
            "segments": segments,
        }

    def synthetic_check(self) -> dict[str, Any]:
        """Синтетически проверяет ABI и конфигурацию без модели и аудио."""
        model = self.create_model("synthetic-model.gguf")
        speakers = int(self.library.nemo_speech_diar_num_speakers(model))
        if speakers != 4:
            self.destroy_model(model)
            raise CAbiError("c-abi-num-speakers-mismatch")
        seconds = float(self.library.nemo_speech_diar_seconds_per_frame(model))
        if not math.isclose(seconds, 0.08, rel_tol=0.0, abs_tol=1e-6):
            self.destroy_model(model)
            raise CAbiError("c-abi-seconds-per-frame-mismatch")
        config = self.frozen_model_config("synthetic-model.gguf")
        self.destroy_model(model)
        return {
            "model_config_size": ctypes.sizeof(SortformerModelConfig),
            "segmentation_config_size": ctypes.sizeof(SortformerSegmentationConfig),
            "segment_size": ctypes.sizeof(SortformerSegment),
            "gpu": int(config.gpu),
            "num_speakers": speakers,
            "seconds_per_frame": seconds,
            "chunk_frames": int(config.chunk_frames),
            "right_context_frames": int(config.right_context_frames),
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Проверяет изолированный стенд Streaming Sortformer v2 Q8_0."
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker is None and (args.manifest is None or not args.validate_only):
        parser.error("укажите --manifest вместе с --validate-only")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.worker is not None:
        raise SystemExit("worker-runtime-requires-injected-handler")
    try:
        code = run_cli(args)
    except (OSError, json.JSONDecodeError, PreflightError, PrivacyError) as exc:
        reason = sanitize_reason(str(exc))
        print(serialize_public({"state": "invalid", "reason": reason}), end="")
        raise SystemExit(2) from None
    raise SystemExit(code)
