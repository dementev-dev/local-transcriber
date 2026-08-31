"""Стенд поэтапного CPU-эксперимента диаризации."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

SCHEMA_VERSION = 3
EXPECTED_DEPENDENCIES = {
    "sherpa-onnx": "1.13.6",
    "sherpa-onnx-core": "1.13.6",
}
RUNTIME_PACKAGES = (
    "sherpa-onnx",
    "sherpa-onnx-core",
    "onnxruntime",
    "numpy",
    "onnx",
    "kaldi-native-fbank",
    "psutil",
)
EXPERIMENT_BUILD_SHA256_ENV = "LOCAL_TRANSCRIBER_EXPERIMENT_BUILD_SHA256"
SHA256_LENGTH = 64
WINDOW_SHIFTS = (0.1, 0.15, 0.2, 0.25, 0.5)
CELL_STAGES = {"intel", "ryzen"}
CELL_PHASES = {"baseline", "mechanism", "window", "model", "execution", "combination"}
COUNTERS = {"threshold", "eigengap", "nme", "known"}
INFERENCE_MODES = {
    "sequential",
    "shared-session",
    "separate-session",
    "titanet-batch",
}

TOP_LEVEL_KEYS = {
    "schema_version",
    "experiment",
    "machine",
    "artifacts",
    "recordings",
    "calibration",
    "qdq",
    "cells",
    "handoff",
    "rss_sample_interval_ms",
}
EXPERIMENT_KEYS = {
    "id",
    "stage",
    "source_commit",
    "handoff_commit",
    "sherpa_patch_sha256",
    "python_version",
    "dependencies",
}
MACHINE_KEYS = {
    "id",
    "sku",
    "physical_cores",
    "logical_cores",
    "ram_bytes",
    "os",
    "cpu_flags",
    "power",
}
POWER_KEYS = {
    "supply",
    "profile",
    "governor",
    "turbo",
    "power_limits",
    "temperature_celsius",
    "throttling",
    "swap_total_bytes",
    "swap_used_bytes",
}
ARTIFACT_KEYS = {
    "id",
    "kind",
    "path",
    "sha256",
    "size_bytes",
    "source_url",
    "revision",
    "license",
}
RECORDING_KEYS = {
    "id",
    "role",
    "path",
    "sha256",
    "start",
    "duration",
    "expected_speakers",
    "reference",
    "asr_words",
}
SIDE_FILE_KEYS = {"path", "sha256", "format"}
CALIBRATION_KEYS = {
    "id",
    "path",
    "sha256",
    "source_id",
    "source_sha256",
    "source_start",
    "source_duration",
    "derived",
}
QDQ_KEYS = {
    "enabled",
    "source_ids",
    "preprocessing",
    "reader",
    "quant_pre_process",
    "quantize_static",
    "excluded_nodes",
    "tool_versions",
    "graph_artifact_id",
    "skip_reason",
}
CELL_KEYS = {
    "name",
    "stage",
    "phase",
    "recording_ids",
    "window_shift_ratio",
    "segmentation_artifact_id",
    "embedding_artifact_id",
    "counter",
    "clustering",
    "inference",
    "repetition",
    "schedule_position",
    "pair_id",
    "pair_role",
    "enabled",
    "skip_reason",
    "build_sha256",
}
CLUSTERING_KEYS = {"mode", "threshold", "num_clusters"}
INFERENCE_KEYS = {
    "mode",
    "outer_workers",
    "session_count",
    "intra_op_threads",
    "inter_op_threads",
    "batch_size",
}
HANDOFF_KEYS = {
    "outcome",
    "source_commit",
    "candidate_recipes",
    "public_report_sha256",
    "capsule_id",
    "capsule_sha256",
}
CANDIDATE_EVIDENCE_KEYS = {
    "name",
    "cell_id",
    "result_file",
    "result_sha256",
    "mandatory_passed",
    "memory_passed",
    "diagnostic_approved",
}
RUNTIME_GUARD_KEYS = {
    "supply",
    "profile",
    "governor",
    "turbo",
    "power_limits",
    "temperature_celsius",
    "throttling",
    "throttle_count",
    "swap_used_bytes",
    "swap_sin_bytes",
    "swap_sout_bytes",
}


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{label}: неверный набор полей; нет={missing}, лишние={extra}"
        )


def _require_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or any(ch.isspace() for ch in value):
        raise ValueError(f"{label}: нужен непустой идентификатор без пробелов")
    return value


def _require_sha256(value: Any, label: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != SHA256_LENGTH
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{label}: нужен SHA-256 в нижнем регистре")
    return value


def _require_commit(value: Any, label: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) not in {40, 64}
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{label}: нужен полный Git object ID в нижнем регистре")
    return value


def _require_version(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label}: нужна точная непустая версия")
    return value


def _require_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}: нужен непустой относительный путь")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"{label}: путь должен быть относительным и безопасным")
    return value


def _require_positive(value: Any, label: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label}: нужно число")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (not allow_zero and result == 0):
        raise ValueError(f"{label}: нужно положительное конечное число")
    return result


def _require_integer(value: Any, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label}: нужно целое число")
    if value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{label}: нужно положительное целое число")
    return value


def _unique_ids(items: Sequence[Mapping[str, Any]], label: str) -> set[str]:
    identifiers = [_require_id(item.get("id"), f"{label}.id") for item in items]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{label}: идентификаторы должны быть уникальны")
    return set(identifiers)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(item: Mapping[str, Any], label: str) -> None:
    path = Path(item["path"])
    if not path.is_file():
        raise ValueError(f"{label}: файл недоступен")
    if path.stat().st_size != item.get("size_bytes", path.stat().st_size):
        raise ValueError(f"{label}: размер файла не совпал")
    if file_sha256(path) != item["sha256"]:
        raise ValueError(f"{label}: SHA-256 не совпал")


def resolve_manifest_paths(
    manifest: Mapping[str, Any], manifest_dir: Path
) -> dict[str, Any]:
    """Разрешает приватные пути, не добавляя их в семантическую идентичность."""
    result = copy.deepcopy(manifest)
    base = manifest_dir.resolve()

    def resolve(value: str) -> str:
        path = Path(value)
        if path.is_absolute():
            return str(path.resolve())
        resolved = (base / path).resolve()
        if not resolved.is_relative_to(base):
            raise ValueError("Относительный путь выходит за корень manifest")
        return str(resolved)

    for artifact in result["artifacts"]:
        artifact["path"] = resolve(artifact["path"])
    for recording in result["recordings"]:
        recording["path"] = resolve(recording["path"])
        for key in ("reference", "asr_words"):
            if recording[key] is not None:
                recording[key]["path"] = resolve(recording[key]["path"])
    for source in result["calibration"]:
        source["path"] = resolve(source["path"])
    return result


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    verify_files: bool = True,
    require_relative_paths: bool = False,
) -> list[str]:
    """Проверяет manifest v3 и возвращает безопасные предупреждения."""
    _exact_keys(manifest, TOP_LEVEL_KEYS, "manifest")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Поддерживается manifest schema_version=3")

    experiment = manifest["experiment"]
    _exact_keys(experiment, EXPERIMENT_KEYS, "experiment")
    _require_id(experiment["id"], "experiment.id")
    if experiment["stage"] not in CELL_STAGES:
        raise ValueError("experiment.stage: допустимы intel и ryzen")
    _require_commit(experiment["source_commit"], "experiment.source_commit")
    _require_commit(
        experiment["handoff_commit"], "experiment.handoff_commit", optional=True
    )
    _require_sha256(
        experiment["sherpa_patch_sha256"],
        "experiment.sherpa_patch_sha256",
        optional=True,
    )
    _require_version(experiment["python_version"], "experiment.python_version")
    if experiment["stage"] == "intel" and experiment["handoff_commit"] is not None:
        raise ValueError("Intel-этап не принимает handoff_commit")
    if experiment["stage"] == "ryzen" and experiment["handoff_commit"] is None:
        raise ValueError("Ryzen-этап требует handoff_commit")
    dependencies = experiment["dependencies"]
    _exact_keys(dependencies, set(RUNTIME_PACKAGES), "experiment.dependencies")
    for name, version in dependencies.items():
        _require_version(version, f"experiment.dependencies.{name}")
    if any(dependencies[name] != version for name, version in EXPECTED_DEPENDENCIES.items()):
        raise ValueError("Версии sherpa-onnx и sherpa-onnx-core должны быть 1.13.6")

    machine = manifest["machine"]
    _exact_keys(machine, MACHINE_KEYS, "machine")
    _require_id(machine["id"], "machine.id")
    if not isinstance(machine["sku"], str) or not machine["sku"].strip():
        raise ValueError("machine.sku: нужна непустая строка")
    physical = _require_integer(machine["physical_cores"], "physical_cores")
    logical = _require_integer(machine["logical_cores"], "logical_cores")
    if logical < physical:
        raise ValueError("logical_cores не может быть меньше physical_cores")
    _require_positive(machine["ram_bytes"], "machine.ram_bytes")
    if not isinstance(machine["cpu_flags"], list) or not machine["cpu_flags"]:
        raise ValueError("machine.cpu_flags: нужен непустой список")
    if machine["cpu_flags"] != sorted(set(machine["cpu_flags"])):
        raise ValueError("machine.cpu_flags: список должен быть сортирован без дублей")
    _exact_keys(machine["power"], POWER_KEYS, "machine.power")
    _require_positive(
        machine["power"]["temperature_celsius"],
        "machine.power.temperature_celsius",
        allow_zero=True,
    )
    _require_positive(
        machine["power"]["swap_total_bytes"],
        "machine.power.swap_total_bytes",
        allow_zero=True,
    )
    _require_positive(
        machine["power"]["swap_used_bytes"],
        "machine.power.swap_used_bytes",
        allow_zero=True,
    )
    if not isinstance(machine["power"]["throttling"], bool):
        raise TypeError("machine.power.throttling: нужен bool")
    if machine["power"]["supply"] != "ac":
        raise ValueError("machine.power.supply: эксперимент выполняется от сети")
    if machine["power"]["profile"] != "balanced":
        raise ValueError("machine.power.profile: нужен стабильный balanced")
    if machine["power"]["throttling"]:
        raise ValueError("machine.power.throttling: исходное состояние нестабильно")
    if not isinstance(machine["power"]["turbo"], bool):
        raise TypeError("machine.power.turbo: нужен bool")

    artifact_ids = _unique_ids(manifest["artifacts"], "artifacts")
    for artifact in manifest["artifacts"]:
        _exact_keys(artifact, ARTIFACT_KEYS, f"artifact:{artifact['id']}")
        _require_sha256(artifact["sha256"], f"artifact:{artifact['id']}.sha256")
        _require_positive(artifact["size_bytes"], f"artifact:{artifact['id']}.size")
        if require_relative_paths:
            _require_relative_path(artifact["path"], f"artifact:{artifact['id']}.path")
        if verify_files:
            _verify_file(artifact, f"artifact:{artifact['id']}")

    recording_ids = _unique_ids(manifest["recordings"], "recordings")
    recording_sets_by_role: dict[str, set[str]] = {"control": set(), "full": set()}
    control_hashes: set[str] = set()
    for recording in manifest["recordings"]:
        _exact_keys(recording, RECORDING_KEYS, f"recording:{recording['id']}")
        if recording["role"] not in {"control", "full"}:
            raise ValueError(f"recording:{recording['id']}: неизвестная роль")
        sha256 = _require_sha256(
            recording["sha256"], f"recording:{recording['id']}.sha256"
        )
        if recording["role"] == "control":
            control_hashes.add(sha256)
        recording_sets_by_role[recording["role"]].add(recording["id"])
        _require_positive(
            recording["duration"], f"recording:{recording['id']}.duration"
        )
        _require_positive(
            recording["start"], f"recording:{recording['id']}.start", allow_zero=True
        )
        _require_integer(
            recording["expected_speakers"],
            f"recording:{recording['id']}.expected_speakers",
        )
        for key in ("reference", "asr_words"):
            side = recording[key]
            if side is None:
                if key == "asr_words":
                    raise ValueError(f"recording:{recording['id']}: нужен asr_words")
                continue
            _exact_keys(side, SIDE_FILE_KEYS, f"recording:{recording['id']}.{key}")
            expected_format = (
                "hypescribe_markdown"
                if key == "reference"
                else "local_transcriber_words_v1"
            )
            if side["format"] != expected_format:
                raise ValueError(
                    f"recording:{recording['id']}.{key}: неверный формат"
                )
            if require_relative_paths:
                _require_relative_path(
                    side["path"], f"recording:{recording['id']}.{key}.path"
                )
            _require_sha256(side["sha256"], f"recording:{recording['id']}.{key}")
            if verify_files:
                _verify_file(side, f"recording:{recording['id']}.{key}")
        if require_relative_paths:
            _require_relative_path(
                recording["path"], f"recording:{recording['id']}.path"
            )
        if verify_files:
            _verify_file(recording, f"recording:{recording['id']}")

    calibration_ids = _unique_ids(manifest["calibration"], "calibration")
    original_calibration_hashes = {
        source["sha256"]
        for source in manifest["calibration"]
        if source.get("derived") is False
    }
    source_hashes_by_id: dict[str, str] = {}
    missing_calibration: list[str] = []
    for source in manifest["calibration"]:
        _exact_keys(source, CALIBRATION_KEYS, f"calibration:{source['id']}")
        _require_sha256(source["sha256"], f"calibration:{source['id']}.sha256")
        if require_relative_paths:
            _require_relative_path(source["path"], f"calibration:{source['id']}.path")
        source_sha = _require_sha256(
            source["source_sha256"], f"calibration:{source['id']}.source_sha256"
        )
        if not isinstance(source["derived"], bool):
            raise TypeError(f"calibration:{source['id']}.derived: нужен bool")
        if source_sha in control_hashes:
            raise ValueError(
                f"calibration:{source['id']}: контрольная запись запрещена"
            )
        if not source["derived"] and source["sha256"] != source_sha:
            raise ValueError(
                f"calibration:{source['id']}: исходник должен совпадать с source_sha256"
            )
        if source["derived"] and source_sha not in original_calibration_hashes:
            raise ValueError(
                f"calibration:{source['id']}: исходный материал не включен в manifest"
            )
        known_source_hash = source_hashes_by_id.setdefault(source["source_id"], source_sha)
        if known_source_hash != source_sha:
            raise ValueError(
                f"calibration:{source['id']}: source_id ссылается на разные исходники"
            )
        _require_positive(
            source["source_start"],
            f"calibration:{source['id']}.source_start",
            allow_zero=True,
        )
        _require_positive(
            source["source_duration"], f"calibration:{source['id']}.source_duration"
        )
        if verify_files:
            path = Path(source["path"])
            if not path.is_file() or file_sha256(path) != source["sha256"]:
                missing_calibration.append(source["id"])

    qdq = manifest["qdq"]
    _exact_keys(qdq, QDQ_KEYS, "qdq")
    if not isinstance(qdq["enabled"], bool):
        raise TypeError("qdq.enabled: нужен bool")
    if not isinstance(qdq["source_ids"], list) or len(qdq["source_ids"]) != len(
        set(qdq["source_ids"])
    ):
        raise ValueError("qdq.source_ids: нужен список без дублей")
    if not set(qdq["source_ids"]) <= calibration_ids:
        raise ValueError("qdq.source_ids: неизвестный калибровочный источник")
    if qdq["enabled"]:
        if not qdq["source_ids"]:
            missing_calibration.append("provenance")
        if qdq["graph_artifact_id"] is None:
            raise ValueError("qdq.graph_artifact_id: для enabled QDQ нужен граф")
        if qdq["skip_reason"] is not None:
            raise ValueError("qdq.skip_reason: для enabled QDQ должен быть null")
    elif not isinstance(qdq["skip_reason"], str) or not qdq["skip_reason"].strip():
        raise ValueError("qdq.skip_reason: для disabled QDQ нужна причина")
    if (
        qdq["graph_artifact_id"] is not None
        and qdq["graph_artifact_id"] not in artifact_ids
    ):
        raise ValueError("qdq.graph_artifact_id: неизвестный артефакт")

    cell_ids: set[str] = set()
    schedule_positions: set[int] = set()
    for index, cell in enumerate(manifest["cells"]):
        _exact_keys(cell, CELL_KEYS, f"cell:{index}")
        _require_id(cell["name"], f"cell:{index}.name")
        if cell["stage"] != experiment["stage"]:
            raise ValueError(f"cell:{cell['name']}: нарушена граница Intel/Ryzen")
        if cell["phase"] not in CELL_PHASES:
            raise ValueError(f"cell:{cell['name']}: неизвестная фаза")
        if (
            not cell["recording_ids"]
            or len(cell["recording_ids"]) != len(set(cell["recording_ids"]))
            or not set(cell["recording_ids"]) <= recording_ids
        ):
            raise ValueError(f"cell:{cell['name']}: неизвестная запись")
        selected_recordings = set(cell["recording_ids"])
        complete_sets = [items for items in recording_sets_by_role.values() if items]
        if selected_recordings not in complete_sets:
            raise ValueError(
                f"cell:{cell['name']}: нужен полный набор control или full"
            )
        ratio = _require_positive(
            cell["window_shift_ratio"], f"cell:{cell['name']}.window_shift_ratio"
        )
        if ratio not in WINDOW_SHIFTS:
            raise ValueError(f"cell:{cell['name']}: шаг окна не входит в сетку")
        for field in ("segmentation_artifact_id", "embedding_artifact_id"):
            if cell[field] not in artifact_ids:
                raise ValueError(f"cell:{cell['name']}.{field}: неизвестный артефакт")
        if cell["counter"] not in COUNTERS:
            raise ValueError(f"cell:{cell['name']}: неизвестный счетчик")
        _exact_keys(
            cell["clustering"], CLUSTERING_KEYS, f"cell:{cell['name']}.clustering"
        )
        clustering = cell["clustering"]
        expected_clustering_mode = (
            "threshold" if cell["counter"] == "threshold" else "counter"
        )
        if clustering["mode"] != expected_clustering_mode:
            raise ValueError(f"cell:{cell['name']}: counter и clustering.mode не совпали")
        _require_positive(clustering["threshold"], f"cell:{cell['name']}.threshold")
        if clustering["num_clusters"] is not None:
            _require_integer(
                clustering["num_clusters"], f"cell:{cell['name']}.num_clusters"
            )
        _exact_keys(cell["inference"], INFERENCE_KEYS, f"cell:{cell['name']}.inference")
        inference = cell["inference"]
        if inference["mode"] not in INFERENCE_MODES:
            raise ValueError(f"cell:{cell['name']}: неизвестный inference mode")
        outer = _require_integer(inference["outer_workers"], "outer_workers")
        sessions = _require_integer(inference["session_count"], "session_count")
        intra = _require_integer(inference["intra_op_threads"], "intra_op_threads")
        inter = _require_integer(inference["inter_op_threads"], "inter_op_threads")
        batch = _require_integer(inference["batch_size"], "batch_size")
        if outer > physical or max(outer, sessions * intra) > physical:
            raise ValueError(f"cell:{cell['name']}: превышен бюджет физических ядер")
        if inter != 1:
            raise ValueError(f"cell:{cell['name']}: inter_op_threads должен быть 1")
        if inference["mode"] != "titanet-batch" and batch != 1:
            raise ValueError(f"cell:{cell['name']}: batch применим только к TitaNet")
        if inference["mode"] == "shared-session" and sessions != 1:
            raise ValueError(
                f"cell:{cell['name']}: shared-session требует одну session"
            )
        if inference["mode"] == "shared-session" and outer == 1:
            raise ValueError(
                f"cell:{cell['name']}: shared-session требует несколько worker"
            )
        if inference["mode"] == "separate-session" and (
            outer != physical or sessions != physical or intra != 1
        ):
            raise ValueError(
                f"cell:{cell['name']}: separate-session требует P worker/session и intra-op=1"
            )
        if inference["mode"] == "titanet-batch" and (
            outer != 1
            or sessions != 1
            or intra != physical
            or batch not in {1, 4}
            or cell["counter"] != "nme"
        ):
            raise ValueError(
                f"cell:{cell['name']}: нарушен контракт TitaNet batch"
            )
        repetition = _require_integer(cell["repetition"], "repetition")
        position = _require_integer(cell["schedule_position"], "schedule_position")
        if position in schedule_positions:
            raise ValueError("schedule_position должен быть уникален")
        schedule_positions.add(position)
        if cell["pair_role"] not in {None, "baseline", "candidate"}:
            raise ValueError(f"cell:{cell['name']}: неверная роль A/B")
        if (cell["pair_id"] is None) != (cell["pair_role"] is None):
            raise ValueError(
                f"cell:{cell['name']}: pair_id и pair_role задаются вместе"
            )
        if not isinstance(cell["enabled"], bool):
            raise TypeError(f"cell:{cell['name']}: enabled должен быть bool")
        _require_sha256(cell["build_sha256"], f"cell:{cell['name']}.build_sha256")
        identity = make_cell_id(manifest, cell)
        if identity in cell_ids:
            raise ValueError(f"cell:{cell['name']}: дублирующая семантическая ячейка")
        cell_ids.add(identity)
        del repetition

    _validate_ab_schedule(manifest["cells"])
    combination_cells = {
        cell["name"]: cell
        for cell in manifest["cells"]
        if cell["phase"] == "combination" and cell["enabled"]
    }
    model_cells = [cell for cell in manifest["cells"] if cell["phase"] == "model"]
    for cell in manifest["cells"]:
        if cell["inference"]["mode"] != "titanet-batch":
            continue
        if not any(
            model["embedding_artifact_id"] == cell["embedding_artifact_id"]
            and model["counter"] == "nme"
            and model["schedule_position"] < cell["schedule_position"]
            and model["enabled"]
            for model in model_cells
        ):
            raise ValueError(
                f"cell:{cell['name']}: TitaNet batch требует предшествующую model-ячейку NME"
            )
    _validate_handoff(manifest["handoff"], experiment, combination_cells, manifest)
    _require_positive(manifest["rss_sample_interval_ms"], "rss_sample_interval_ms")

    warnings = []
    if missing_calibration:
        warnings.append(
            "QDQ пропущен: калибровочный материал или происхождение не подтверждены"
        )
    return warnings


def _validate_ab_schedule(cells: Sequence[Mapping[str, Any]]) -> None:
    pairs: dict[str, list[Mapping[str, Any]]] = {}
    for cell in cells:
        if cell["pair_id"] is not None:
            pairs.setdefault(cell["pair_id"], []).append(cell)
    first_roles = []
    for pair_id, pair_cells in pairs.items():
        ordered = sorted(pair_cells, key=lambda item: item["schedule_position"])
        roles = [item["pair_role"] for item in ordered]
        if len(ordered) != 2 or set(roles) != {"baseline", "candidate"}:
            raise ValueError(f"A/B {pair_id}: нужна ровно одна пара baseline/candidate")
        positions = [item["schedule_position"] for item in ordered]
        if positions[1] != positions[0] + 1:
            raise ValueError(f"A/B {pair_id}: позиции пары должны быть соседними")
        first_roles.append((ordered[0]["schedule_position"], roles[0]))
    first_roles.sort()
    if any(left[1] == right[1] for left, right in pairwise(first_roles)):
        raise ValueError("A/B: начальная роль должна меняться между парами")


def _validate_handoff(
    handoff: Mapping[str, Any],
    experiment: Mapping[str, Any],
    combination_cells: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    _exact_keys(handoff, HANDOFF_KEYS, "handoff")
    if handoff["outcome"] not in {"pending", "handoff", "stop-before-ryzen"}:
        raise ValueError("handoff.outcome: неизвестный итог")
    candidates = handoff["candidate_recipes"]
    if not isinstance(candidates, list) or len(candidates) > 3:
        raise ValueError("handoff: допускается не больше трех кандидатов")
    if handoff["outcome"] == "stop-before-ryzen" and candidates:
        raise ValueError("stop-before-ryzen требует ноль кандидатов")
    if handoff["outcome"] == "handoff" and not candidates:
        raise ValueError("handoff требует от одного до трех кандидатов")
    candidate_names = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise TypeError(f"handoff.candidate_recipes:{index}: нужен объект evidence")
        _exact_keys(
            candidate,
            CANDIDATE_EVIDENCE_KEYS,
            f"handoff.candidate_recipes:{index}",
        )
        name = _require_id(candidate["name"], f"handoff.candidate_recipes:{index}.name")
        candidate_names.append(name)
        cell = combination_cells.get(name)
        if cell is None:
            raise ValueError(
                "handoff: кандидат не описан включенной combination-ячейкой"
            )
        expected_cell_id = make_cell_id(manifest, cell)
        if candidate["cell_id"] != expected_cell_id:
            raise ValueError(f"handoff:{name}: cell_id не совпадает с manifest")
        _require_sha256(candidate["result_sha256"], f"handoff:{name}.result_sha256")
        _require_relative_path(candidate["result_file"], f"handoff:{name}.result_file")
        for gate in ("mandatory_passed", "memory_passed", "diagnostic_approved"):
            if candidate[gate] is not True:
                raise ValueError(f"handoff:{name}: не пройден gate {gate}")
    if len(candidate_names) != len(set(candidate_names)):
        raise ValueError("handoff: рецепты кандидатов должны быть уникальны")
    _require_commit(handoff["source_commit"], "handoff.source_commit", optional=True)
    _require_sha256(
        handoff["public_report_sha256"], "handoff.public_report_sha256", optional=True
    )
    _require_sha256(handoff["capsule_sha256"], "handoff.capsule_sha256", optional=True)
    if experiment["stage"] == "ryzen" and handoff["outcome"] == "pending":
        raise ValueError("Ryzen-этап не принимает pending handoff")
    finalized = handoff["outcome"] != "pending"
    required = (
        handoff["source_commit"],
        handoff["public_report_sha256"],
        handoff["capsule_id"],
    )
    if finalized and any(value is None for value in required):
        raise ValueError("Итог handoff требует commit, отчет и приватную капсулу")
    final_links = (*required, handoff["capsule_sha256"])
    if not finalized and any(value is not None for value in final_links):
        raise ValueError("Pending handoff не должен содержать итоговые ссылки")
    if finalized and handoff["source_commit"] != experiment["source_commit"]:
        raise ValueError("Commit handoff не совпадает с исходным commit эксперимента")
    if handoff["capsule_id"] is not None:
        _require_id(handoff["capsule_id"], "handoff.capsule_id")


def semantic_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Возвращает идентичность manifest без локальных путей и приватного текста."""
    result = copy.deepcopy(manifest)
    result.pop("handoff", None)
    for artifact in result["artifacts"]:
        artifact.pop("path", None)
    for recording in result["recordings"]:
        recording.pop("path", None)
        for key in ("reference", "asr_words"):
            if recording[key] is not None:
                recording[key].pop("path", None)
    for source in result["calibration"]:
        source.pop("path", None)
    return result


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()


def _digest_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def make_experiment_id(manifest: Mapping[str, Any]) -> str:
    return _digest_json(semantic_manifest(manifest))


def make_cell_id(manifest: Mapping[str, Any], cell: Mapping[str, Any]) -> str:
    artifact_hashes = {
        item["id"]: item["sha256"]
        for item in manifest["artifacts"]
        if item["id"]
        in {cell["segmentation_artifact_id"], cell["embedding_artifact_id"]}
    }
    recording_inputs = {
        item["id"]: {
            "media_sha256": item["sha256"],
            "start": item["start"],
            "duration": item["duration"],
            "expected_speakers": item["expected_speakers"],
            "reference": (
                None
                if item["reference"] is None
                else {
                    "sha256": item["reference"]["sha256"],
                    "format": item["reference"]["format"],
                }
            ),
            "asr_words": {
                "sha256": item["asr_words"]["sha256"],
                "format": item["asr_words"]["format"],
            },
        }
        for item in manifest["recordings"]
        if item["id"] in cell["recording_ids"]
    }
    qdq_identity = None
    if cell["embedding_artifact_id"] == manifest["qdq"]["graph_artifact_id"]:
        selected_sources = set(manifest["qdq"]["source_ids"])
        qdq_identity = {
            "recipe": manifest["qdq"],
            "calibration": [
                {key: value for key, value in source.items() if key != "path"}
                for source in manifest["calibration"]
                if source["id"] in selected_sources
            ],
        }
    identity = {
        "experiment": manifest["experiment"],
        "machine": manifest["machine"],
        "artifact_hashes": artifact_hashes,
        "recording_inputs": recording_inputs,
        "qdq": qdq_identity,
        "cell": cell,
    }
    return _digest_json(identity)


def _linux_cpu_identity() -> tuple[str | None, list[str]]:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return platform.processor() or None, []
    sku = None
    flags: list[str] = []
    for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        if key.strip() == "model name" and sku is None:
            sku = value.strip()
        elif key.strip() in {"flags", "Features"} and not flags:
            flags = sorted(set(value.split()))
        if sku is not None and flags:
            break
    return sku or platform.processor() or None, flags


def environment_snapshot() -> dict[str, Any]:
    import psutil

    packages = {
        name: importlib.metadata.version(name)
        for name in RUNTIME_PACKAGES
    }
    sku, cpu_flags = _linux_cpu_identity()
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "packages": packages,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cores": os.cpu_count(),
        "physical_cores": psutil.cpu_count(logical=False),
        "cpu_sku": sku,
        "cpu_flags": cpu_flags,
        "ram_bytes": memory.total,
        "swap_total_bytes": swap.total,
        "swap_used_bytes": swap.used,
        "build_sha256": os.environ.get(EXPERIMENT_BUILD_SHA256_ENV),
    }


def validate_environment(
    manifest: Mapping[str, Any], environment: Mapping[str, Any]
) -> None:
    experiment = manifest["experiment"]
    for name, expected in experiment["dependencies"].items():
        if environment["packages"].get(name) != expected:
            raise ValueError(f"Ожидался {name}=={expected}")
    if environment.get("python") != experiment["python_version"]:
        raise ValueError(
            f"Ожидался Python {experiment['python_version']}"
        )
    runtime_build = _require_sha256(
        environment.get("build_sha256"),
        f"переменная {EXPERIMENT_BUILD_SHA256_ENV}",
    )
    expected_builds = {
        cell["build_sha256"] for cell in manifest["cells"] if cell["enabled"]
    }
    if expected_builds != {runtime_build}:
        raise ValueError("Хеш runtime-сборки не совпал с enabled-ячейками")
    machine = manifest["machine"]
    if environment.get("platform") != machine["os"]:
        raise ValueError("ОС не совпала с manifest")
    if environment["logical_cores"] != machine["logical_cores"]:
        raise ValueError("Число логических ядер не совпало с manifest")
    if environment.get("physical_cores") != machine["physical_cores"]:
        raise ValueError("Число физических ядер не совпало с manifest")
    if environment.get("cpu_sku") != machine["sku"]:
        raise ValueError("CPU SKU не совпал с manifest")
    if environment.get("ram_bytes") != machine["ram_bytes"]:
        raise ValueError("Объем RAM не совпал с manifest")
    if environment.get("swap_total_bytes") != machine["power"]["swap_total_bytes"]:
        raise ValueError("Объем swap не совпал с manifest")
    if environment.get("cpu_flags") != machine["cpu_flags"]:
        raise ValueError("CPUID/ISA не совпали с manifest")


def _read_optional_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _runtime_supply() -> str | None:
    for device in sorted(Path("/sys/class/power_supply").glob("*")):
        kind = _read_optional_text(device / "type")
        online = _read_optional_text(device / "online")
        if kind in {"Mains", "USB", "USB_PD"} and online == "1":
            return "ac"
    return None


def _runtime_profile() -> str | None:
    profile = _read_optional_text(Path("/sys/firmware/acpi/platform_profile"))
    if profile is not None:
        return profile
    try:
        completed = subprocess.run(
            ["powerprofilesctl", "get"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = completed.stdout.strip()
    return value or None if completed.returncode == 0 else None


def _runtime_governor() -> str | None:
    values = {
        value
        for path in Path("/sys/devices/system/cpu").glob(
            "cpu[0-9]*/cpufreq/scaling_governor"
        )
        if (value := _read_optional_text(path))
    }
    return next(iter(values)) if len(values) == 1 else None


def _runtime_turbo() -> bool | None:
    no_turbo = _read_optional_text(
        Path("/sys/devices/system/cpu/intel_pstate/no_turbo")
    )
    if no_turbo in {"0", "1"}:
        return no_turbo == "0"
    boost = _read_optional_text(Path("/sys/devices/system/cpu/cpufreq/boost"))
    if boost in {"0", "1"}:
        return boost == "1"
    return None


def _runtime_power_limits() -> dict[str, str]:
    limits = {}
    for path in sorted(
        Path("/sys/class/powercap").glob(
            "intel-rapl*/constraint_*_power_limit_uw"
        )
    ):
        value = _read_optional_text(path)
        if value is not None:
            limits[path.name] = value
    return limits


def _runtime_temperature() -> float | None:
    import psutil

    try:
        values = [
            float(item.current)
            for entries in psutil.sensors_temperatures().values()
            for item in entries
            if item.current is not None and math.isfinite(float(item.current))
        ]
    except (AttributeError, OSError):
        return None
    return max(values, default=None)


def _runtime_throttle_count() -> int | None:
    values = []
    for path in Path("/sys/devices/system/cpu").glob(
        "cpu[0-9]*/thermal_throttle/*_throttle_count"
    ):
        value = _read_optional_text(path)
        if value is not None and value.isdigit():
            values.append(int(value))
    return sum(values) if values else None


def runtime_guard_snapshot() -> dict[str, Any]:
    """Снимает изменяемые условия непосредственно вокруг одной ячейки."""
    override = os.environ.get("LOCAL_TRANSCRIBER_RUNTIME_GUARD_JSON")
    if override is not None:
        result = json.loads(override)
        _exact_keys(result, RUNTIME_GUARD_KEYS, "runtime guard")
        return result

    import psutil

    swap = psutil.swap_memory()
    return {
        "supply": _runtime_supply(),
        "profile": _runtime_profile(),
        "governor": _runtime_governor(),
        "turbo": _runtime_turbo(),
        "power_limits": _runtime_power_limits(),
        "temperature_celsius": _runtime_temperature(),
        "throttling": False,
        "throttle_count": _runtime_throttle_count(),
        "swap_used_bytes": int(swap.used),
        "swap_sin_bytes": int(swap.sin),
        "swap_sout_bytes": int(swap.sout),
    }


def validate_runtime_guard_transition(
    manifest: Mapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> list[str]:
    """Возвращает стабильные коды причин, по которым прогон нельзя учитывать."""
    _exact_keys(before, RUNTIME_GUARD_KEYS, "runtime guard before")
    _exact_keys(after, RUNTIME_GUARD_KEYS, "runtime guard after")
    expected = manifest["machine"]["power"]
    reasons = []
    for key in ("supply", "profile", "governor", "turbo"):
        if before[key] != expected[key] or after[key] != expected[key]:
            reasons.append(f"{key}-mismatch")
        elif before[key] != after[key]:
            reasons.append(f"{key}-changed")
    if before["power_limits"] != after["power_limits"]:
        reasons.append("power-limits-changed")
    if before["throttling"] or after["throttling"]:
        reasons.append("throttling")
    if before["throttle_count"] is None or after["throttle_count"] is None:
        reasons.append("throttle-count-unavailable")
    elif before["throttle_count"] != after["throttle_count"]:
        reasons.append("throttle-count-changed")
    for key in ("swap_sin_bytes", "swap_sout_bytes"):
        if before[key] != after[key]:
            reasons.append(f"{key}-changed")
    if after["swap_used_bytes"] > before["swap_used_bytes"]:
        reasons.append("swap-used-increased")
    return reasons


def safe_plan(
    manifest: Mapping[str, Any], warnings: Sequence[str], environment: Mapping[str, Any]
) -> dict[str, Any]:
    """Возвращает план без локальных путей, текста и интервалов записей."""
    cells = sorted(manifest["cells"], key=lambda item: item["schedule_position"])
    qdq_available = not warnings and manifest["qdq"]["enabled"]
    qdq_artifact_id = manifest["qdq"]["graph_artifact_id"]
    return {
        "status": "valid",
        "schema_version": SCHEMA_VERSION,
        "experiment_id": make_experiment_id(manifest),
        "stage": manifest["experiment"]["stage"],
        "machine_id": manifest["machine"]["id"],
        "recording_ids": [item["id"] for item in manifest["recordings"]],
        "artifact_hashes": {
            item["id"]: item["sha256"] for item in manifest["artifacts"]
        },
        "cells": [
            {
                "name": item["name"],
                "cell_id": make_cell_id(manifest, item),
                "phase": item["phase"],
                "position": item["schedule_position"],
                "enabled": item["enabled"]
                and (qdq_available or item["embedding_artifact_id"] != qdq_artifact_id),
            }
            for item in cells
        ],
        "qdq_available": qdq_available,
        "warnings": list(warnings),
        "dependencies": environment["packages"],
        "python_version": environment["python"],
        "build_sha256": environment["build_sha256"],
    }


def _stable_top_indices(row: np.ndarray, count: int) -> np.ndarray:
    indices = np.arange(row.shape[0])
    order = np.lexsort((indices, -row))
    return order[:count]


def _affinity(embeddings: np.ndarray, neighbors: int) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1)
    if values.ndim != 2 or not np.isfinite(values).all() or (norms == 0).any():
        raise ValueError("Матрица эмбеддингов должна быть конечной и ненулевой")
    normalized = values / norms[:, None]
    similarity = normalized @ normalized.T
    binary = np.zeros_like(similarity)
    for row_index, row in enumerate(similarity):
        binary[row_index, _stable_top_indices(row, neighbors)] = 1.0
    graph = (binary + binary.T) / 2.0
    np.fill_diagonal(graph, 0.0)
    return graph


def _eigen_result(embeddings: np.ndarray, neighbors: int) -> dict[str, Any]:
    graph = _affinity(embeddings, neighbors)
    degree = np.diag(np.sum(np.abs(graph), axis=1))
    eigenvalues = np.linalg.eigvalsh(degree - graph)
    max_clusters = min(8, len(eigenvalues) - 1)
    gaps = np.diff(eigenvalues)[:max_clusters]
    count = 1 + int(np.argmax(gaps))
    connected = int(np.count_nonzero(eigenvalues < 1e-8)) == 1
    return {
        "num_clusters": count,
        "max_gap": float(gaps[count - 1]),
        "lambda_max": float(eigenvalues[-1]),
        "connected": connected,
    }


def estimate_eigengap(embeddings: np.ndarray, pval: float = 0.012) -> dict[str, Any]:
    """Считает контрольный eigengap с фиксированным прореживанием."""
    row_count = int(np.asarray(embeddings).shape[0])
    if row_count < 2:
        raise ValueError("Для eigengap нужны минимум две строки")
    neighbors = max(6, round(row_count * pval))
    neighbors = min(row_count, neighbors)
    return {"p": neighbors, **_eigen_result(embeddings, neighbors)}


def _nme_grid(row_count: int) -> list[int]:
    upper = row_count // 4
    if row_count < 6 or upper < 1:
        return []
    if upper <= 20:
        return list(range(1, upper + 1))
    return sorted({int(value) for value in np.linspace(1, upper, 20)})


def estimate_nme(embeddings: np.ndarray) -> dict[str, Any]:
    """Оценивает N и p по закрепленному контракту NME Sparse-Search-20."""
    values = np.asarray(embeddings)
    row_count = int(values.shape[0])
    grid = _nme_grid(row_count)
    if not grid:
        return {"status": "skipped", "reason": "fewer-than-six-rows"}
    candidates = []
    epsilon = 1e-10
    for neighbors in grid:
        result = _eigen_result(values, neighbors)
        normalized_gap = result["max_gap"] / (result["lambda_max"] + epsilon)
        score = (neighbors / row_count) / (normalized_gap + epsilon)
        candidates.append({"p": neighbors, "score": score, **result})
    selected_index = min(
        range(len(candidates)),
        key=lambda index: (candidates[index]["score"], candidates[index]["p"]),
    )
    while (
        selected_index < len(candidates) and not candidates[selected_index]["connected"]
    ):
        selected_index += 1
    if selected_index == len(candidates):
        raise ValueError("NME: в сетке нет связного графа")
    selected = candidates[selected_index]
    return {
        "status": "complete",
        "p": selected["p"],
        "num_clusters": selected["num_clusters"],
        "score": selected["score"],
    }


def select_counter(results: Sequence[Mapping[str, Any]]) -> dict[str, str | None]:
    """Выбирает счетчик отдельно для каждой embedding-модели."""
    selected: dict[str, str | None] = {}
    model_ids = sorted({str(item["embedding_id"]) for item in results})
    for model_id in model_ids:
        passing = []
        for counter in ("eigengap", "nme"):
            rows = [
                item
                for item in results
                if item["embedding_id"] == model_id and item["counter"] == counter
            ]
            if rows and [item["num_clusters"] for item in rows] == [3, 2, 2]:
                passing.append(
                    (counter, sum(float(item["wall_seconds"]) for item in rows))
                )
        if not passing:
            selected[model_id] = None
        elif len(passing) == 1:
            selected[model_id] = passing[0][0]
        else:
            times = dict(passing)
            faster = min(times, key=times.get)
            slower = max(times.values())
            difference = 0.0 if slower == 0 else (slower - min(times.values())) / slower
            selected[model_id] = "eigengap" if difference <= 0.1 else faster
    return selected


def select_working_shift(results: Sequence[Mapping[str, Any]]) -> float:
    """Выбирает безопасный шаг с порогом продолжения 10%."""
    by_shift = {float(item["window_shift_ratio"]): item for item in results}
    baseline = by_shift.get(0.1)
    if baseline is None or not baseline.get("quality_passed"):
        raise ValueError("Нет принятого W0 baseline")
    eligible = [
        item
        for item in results
        if item.get("quality_passed")
        and item.get("work_counters_explained")
        and float(item["wall_seconds"]) <= float(baseline["wall_seconds"]) * 0.9
    ]
    if not eligible:
        return 0.1
    fastest = min(float(item["wall_seconds"]) for item in eligible)
    equal = [
        item
        for item in eligible
        if abs(float(item["wall_seconds"]) - fastest) / fastest <= 0.05
    ]
    return min(float(item["window_shift_ratio"]) for item in equal)


def assemble_candidate_recipes(
    working_shift: float,
    branches: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Собирает C0-C2 без декартова перебора и дублей."""
    recipes: list[dict[str, Any]] = []
    if working_shift != 0.1:
        recipes.append(
            {
                "id": "C0",
                "window_shift_ratio": working_shift,
                "embedding": "wespeaker-fp32",
                "inference": "sequential",
            }
        )
    passing = [
        item
        for item in branches
        if item.get("quality_passed") is True
        and item.get("memory_passed") is True
        and item.get("diagnostic_approved") is True
        and item.get("compatible", True)
    ]
    continuation = [item for item in passing if item.get("speedup", 0.0) >= 0.1]
    if continuation:
        strongest = max(continuation, key=lambda item: float(item["speedup"]))
        recipes.append(
            {"id": "C1", "window_shift_ratio": working_shift, **strongest["recipe"]}
        )
    isa = [item for item in passing if item.get("isa_stack")]
    if isa:
        recipe = {"id": "C2", "window_shift_ratio": working_shift}
        for item in sorted(isa, key=lambda value: value.get("stack_order", 0)):
            recipe.update(item["recipe"])
        recipes.append(recipe)
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for recipe in recipes:
        semantic = {key: value for key, value in recipe.items() if key != "id"}
        digest = _digest_json(semantic)
        if digest not in seen:
            seen.add(digest)
            unique.append(recipe)
    return unique[:3]


def finalize_handoff(
    manifest: Mapping[str, Any],
    output: Mapping[str, Any],
    candidate_names: Sequence[str],
    *,
    capsule_root: Path,
    gate_evidence: Mapping[str, Mapping[str, Any]],
    public_report_sha256: str,
    capsule_id: str,
    capsule_sha256: str | None = None,
) -> dict[str, Any]:
    """Финализирует handoff только из завершенных и явно принятых результатов."""
    if output.get("experiment_id") != make_experiment_id(manifest):
        raise ValueError("Output относится к другому experiment")
    if len(candidate_names) != len(set(candidate_names)) or len(candidate_names) > 3:
        raise ValueError("Handoff принимает не больше трех уникальных кандидатов")
    cells_by_name = {
        cell["name"]: cell
        for cell in manifest["cells"]
        if cell["phase"] == "combination" and cell["enabled"]
    }
    results_by_id = {
        item.get("cell_id"): item
        for item in output.get("cells", [])
        if item.get("status") == "complete"
    }
    candidates = []
    result_files = []
    for name in candidate_names:
        cell = cells_by_name.get(name)
        if cell is None:
            raise ValueError(f"Кандидат {name} не является enabled combination")
        cell_id = make_cell_id(manifest, cell)
        result_row = results_by_id.get(cell_id)
        if result_row is None or not isinstance(result_row.get("result"), Mapping):
            raise ValueError(f"Для кандидата {name} нет завершенного результата")
        gates = gate_evidence.get(name)
        if not isinstance(gates, Mapping):
            raise TypeError(f"Для кандидата {name} нет решения по gates")
        _exact_keys(
            gates,
            {"memory_passed", "diagnostic_approved"},
            f"gates:{name}",
        )
        mandatory_passed = (
            result_row["result"].get("aggregate", {}).get("quality_passed") is True
        )
        result_file = f"results/{cell_id}.json"
        result_bytes = _canonical_json_bytes(result_row["result"])
        evidence = {
            "name": name,
            "cell_id": cell_id,
            "result_file": result_file,
            "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
            "mandatory_passed": mandatory_passed,
            "memory_passed": gates["memory_passed"] is True,
            "diagnostic_approved": gates["diagnostic_approved"] is True,
        }
        if not all(
            evidence[key]
            for key in (
                "mandatory_passed",
                "memory_passed",
                "diagnostic_approved",
            )
        ):
            raise ValueError(f"Кандидат {name} не прошел обязательные gates")
        candidates.append(evidence)
        result_files.append((result_file, result_bytes))

    _require_private_location(capsule_root, "capsule-root")
    for relative_path, content in result_files:
        _save_text(capsule_root / relative_path, content.decode("ascii"))
    result = copy.deepcopy(manifest)
    result["handoff"] = {
        "outcome": "handoff" if candidates else "stop-before-ryzen",
        "source_commit": manifest["experiment"]["source_commit"],
        "candidate_recipes": candidates,
        "public_report_sha256": public_report_sha256,
        "capsule_id": capsule_id,
        "capsule_sha256": capsule_sha256,
    }
    validate_manifest(result, verify_files=False)
    return result


def residual_cluster_metrics(
    intervals: Sequence[Mapping[str, Any]],
    words: Sequence[Mapping[str, Any]],
    expected_speakers: int,
) -> dict[str, float]:
    """Разделяет долю времени речи и долю слов в остаточных кластерах."""
    durations: dict[int, float] = {}
    for interval in intervals:
        speaker = int(interval["speaker"])
        durations[speaker] = durations.get(speaker, 0.0) + max(
            0.0, float(interval["end"]) - float(interval["start"])
        )
    ordered = sorted(durations, key=lambda speaker: (-durations[speaker], speaker))
    residual = set(ordered[expected_speakers:])
    total_duration = sum(durations.values())
    assigned_words = [word for word in words if word.get("speaker") is not None]
    residual_words = [
        word for word in assigned_words if int(word["speaker"]) in residual
    ]
    return {
        "residual_speech_time_share": (
            sum(durations[speaker] for speaker in residual) / total_duration
            if total_duration
            else 0.0
        ),
        "residual_assigned_word_share": (
            len(residual_words) / len(assigned_words) if assigned_words else 0.0
        ),
    }


def pending_cells(
    manifest: Mapping[str, Any], output: Mapping[str, Any], warnings: Sequence[str]
) -> list[dict[str, Any]]:
    complete = {
        item["cell_id"]
        for item in output.get("cells", [])
        if item.get("status") == "complete"
    }
    qdq_available = not warnings and manifest["qdq"]["enabled"]
    qdq_artifact_id = manifest["qdq"]["graph_artifact_id"]
    result = []
    for cell in sorted(manifest["cells"], key=lambda item: item["schedule_position"]):
        cell_id = make_cell_id(manifest, cell)
        enabled = cell["enabled"] and (
            qdq_available or cell["embedding_artifact_id"] != qdq_artifact_id
        )
        if enabled and cell_id not in complete:
            result.append({**copy.deepcopy(cell), "cell_id": cell_id})
    return result


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as target:
        temporary = Path(target.name)
        json.dump(value, target, ensure_ascii=False, indent=2)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def _save_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as target:
        temporary = Path(target.name)
        target.write(value)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def _require_private_location(path: Path, label: str) -> None:
    repository_root = Path(__file__).parents[2].resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(repository_root):
        return
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", str(resolved)],
        cwd=repository_root,
        check=False,
    )
    if ignored.returncode != 0:
        raise ValueError(f"{label}: приватный путь не игнорируется Git")


def _load_output(
    path: Path, manifest: Mapping[str, Any], environment: Mapping[str, Any]
) -> dict[str, Any]:
    experiment_id = make_experiment_id(manifest)
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "environment": environment,
            "cells": [],
        }
    output = json.loads(path.read_text(encoding="utf-8"))
    if (
        output.get("schema_version") != SCHEMA_VERSION
        or output.get("experiment_id") != experiment_id
    ):
        raise ValueError("Существующий output относится к другому experiment")
    if _stable_environment_identity(output.get("environment", {})) != (
        _stable_environment_identity(environment)
    ):
        raise ValueError("Существующий output снят в другом окружении")
    return output


def _stable_environment_identity(environment: Mapping[str, Any]) -> dict[str, Any]:
    """Исключает только значения, которые закономерно меняются между resume."""
    result = copy.deepcopy(dict(environment))
    result.pop("swap_used_bytes", None)
    return result


def _replace_cell(output: dict[str, Any], result: Mapping[str, Any]) -> None:
    output["cells"] = [
        item for item in output["cells"] if item.get("cell_id") != result["cell_id"]
    ] + [dict(result)]


def run_schedule(
    manifest: Mapping[str, Any],
    output_path: Path,
    work_dir: Path,
    runner: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    *,
    environment: Mapping[str, Any] | None = None,
    guard_reader: Callable[[], Mapping[str, Any]] = runtime_guard_snapshot,
) -> dict[str, Any]:
    """Выполняет оставшиеся ячейки и сохраняет каждый сырой повтор."""
    warnings = validate_manifest(manifest)
    current_environment = dict(environment or environment_snapshot())
    validate_environment(manifest, current_environment)
    output = _load_output(output_path, manifest, current_environment)
    _save_json(output_path, output)
    for cell in pending_cells(manifest, output, warnings):
        request = {
            "worker_schema_version": SCHEMA_VERSION,
            "experiment_id": output["experiment_id"],
            "cell": cell,
            "manifest": manifest,
            "work_dir": str(work_dir),
        }
        previous = next(
            (
                item
                for item in output["cells"]
                if item.get("cell_id") == cell["cell_id"]
            ),
            {},
        )
        attempts = list(previous.get("guard", {}).get("attempts", []))
        if len(attempts) >= 2 and all(not item.get("valid") for item in attempts):
            raise RuntimeError(f"Ячейка {cell['name']} исчерпала retry runtime guard")
        result = None
        for attempt_number in range(len(attempts) + 1, 3):
            before = dict(guard_reader())
            payload = None
            error = None
            try:
                payload = dict(runner(request))
            except Exception as exc:  # noqa: BLE001 — граница изолированного worker
                error = exc
            after = dict(guard_reader())
            reasons = validate_runtime_guard_transition(manifest, before, after)
            attempt = {
                "attempt": attempt_number,
                "before": before,
                "after": after,
                "valid": not reasons,
                "reasons": reasons,
            }
            attempts.append(attempt)
            if error is not None:
                result = {
                    "cell_id": cell["cell_id"],
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "guard": {"attempts": attempts},
                }
                break
            if reasons:
                result = {
                    "cell_id": cell["cell_id"],
                    "status": "retrying" if attempt_number == 1 else "failed",
                    "error_type": "RuntimeGuardError",
                    "guard": {"attempts": attempts},
                }
                _replace_cell(output, result)
                _save_json(output_path, output)
                if attempt_number == 1:
                    continue
                break
            result = {
                "cell_id": cell["cell_id"],
                "status": "complete",
                "result": payload,
                "guard": {"attempts": attempts},
            }
            break
        if result is None:
            raise AssertionError("Runtime guard не сформировал результат")
        _replace_cell(output, result)
        _save_json(output_path, output)
        if result["status"] != "complete":
            raise RuntimeError(f"Ячейка {cell['name']} завершилась с ошибкой")
    return output


def subprocess_runner(request: Mapping[str, Any]) -> Mapping[str, Any]:
    """Запускает ячейку в новом процессе без публикации приватного stderr."""
    work_dir = Path(request["work_dir"])
    request_dir = work_dir / "requests"
    request_dir.mkdir(parents=True, exist_ok=True)
    request_path = request_dir / f"{request['cell']['cell_id']}.json"
    response_path = request_dir / f"{request['cell']['cell_id']}.result.json"
    payload = dict(request)
    payload["response_path"] = str(response_path)
    _save_json(request_path, payload)
    worker_python = os.environ.get(
        "LOCAL_TRANSCRIBER_EXPERIMENT_PYTHON", sys.executable
    )
    completed = subprocess.run(
        [
            worker_python,
            str(
                Path(__file__).parents[2]
                / "scripts"
                / "benchmarks"
                / "diarization_calibration.py"
            ),
            "--worker",
            str(request_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    log_dir = work_dir / "logs"
    _save_text(log_dir / f"{request['cell']['cell_id']}.stdout.log", completed.stdout)
    _save_text(log_dir / f"{request['cell']['cell_id']}.stderr.log", completed.stderr)
    if completed.returncode != 0 or not response_path.is_file():
        raise RuntimeError("Дочерний процесс завершился с ошибкой")
    return json.loads(response_path.read_text(encoding="utf-8"))


def worker_main(request_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    from local_transcriber.experimental_diarization import run_cell

    result = run_cell(request)
    _save_json(Path(request["response_path"]), result)


def verify_capsule(
    archive_path: Path,
    expected_sha256: str,
    *,
    expected_manifest_path: Path,
    expected_source_commit: str | None = None,
    expected_report_sha256: str | None = None,
    expected_capsule_id: str | None = None,
) -> dict[str, Any]:
    """Сверяет внешний и внутренние хеши приватной ZIP-капсулы."""
    if file_sha256(archive_path) != expected_sha256:
        raise ValueError("SHA-256 ZIP-капсулы не совпал")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Капсула содержит повторяющиеся пути")
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError("Капсула содержит небезопасный путь")
        if "capsule.json" not in names:
            raise ValueError("В капсуле нет capsule.json")
        capsule = json.loads(archive.read("capsule.json"))
        _exact_keys(
            capsule,
            {"capsule_id", "source_commit", "public_report_sha256", "files"},
            "capsule.json",
        )
        _require_id(capsule["capsule_id"], "capsule.json.capsule_id")
        _require_commit(capsule["source_commit"], "capsule.json.source_commit")
        _require_sha256(
            capsule["public_report_sha256"],
            "capsule.json.public_report_sha256",
        )
        if expected_capsule_id and capsule["capsule_id"] != expected_capsule_id:
            raise ValueError("Идентификатор капсулы не совпал")
        if (
            expected_source_commit
            and capsule.get("source_commit") != expected_source_commit
        ):
            raise ValueError("Исходный commit капсулы не совпал")
        if (
            expected_report_sha256
            and capsule.get("public_report_sha256") != expected_report_sha256
        ):
            raise ValueError("Хеш публичного отчета в капсуле не совпал")
        declared = capsule.get("files")
        if not isinstance(declared, dict):
            raise TypeError("capsule.json: нужен словарь files")
        if "manifest.json" not in declared or "manifest.json" not in names:
            raise ValueError("Капсула должна содержать корневой manifest.json")
        actual_files = {name for name in names if not name.endswith("/")}
        if actual_files != {"capsule.json", *declared}:
            raise ValueError("capsule.json должен перечислять каждый файл архива")
        for name, expected in declared.items():
            if name == "capsule.json" or name not in names:
                raise ValueError("capsule.json ссылается на отсутствующий файл")
            actual = hashlib.sha256(archive.read(name)).hexdigest()
            if actual != expected:
                raise ValueError("Внутренний SHA-256 капсулы не совпал")
        manifest_bytes = expected_manifest_path.read_bytes()
        archived_manifest = archive.read("manifest.json")
        if archived_manifest != manifest_bytes:
            raise ValueError("manifest аргумента не совпал с manifest.json капсулы")
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if declared["manifest.json"] != manifest_sha256:
            raise ValueError("Хеш manifest.json не совпал с индексом капсулы")
        archived_manifest_value = json.loads(archived_manifest)
        handoff = archived_manifest_value.get("handoff", {})
        for candidate in handoff.get("candidate_recipes", []):
            result_file = candidate.get("result_file")
            result_sha256 = candidate.get("result_sha256")
            if declared.get(result_file) != result_sha256:
                raise ValueError(
                    "Результат кандидата не совпал с индексом файлов капсулы"
                )
    return {
        "capsule_id": capsule.get("capsule_id"),
        "source_commit": capsule.get("source_commit"),
        "public_report_sha256": capsule.get("public_report_sha256"),
        "file_count": len(declared),
        "manifest_sha256": manifest_sha256,
    }


def run_cli(args: Any, raw_manifest: Mapping[str, Any]) -> None:
    _require_private_location(args.manifest, "manifest")
    if args.output is not None:
        _require_private_location(args.output, "output")
    if args.work_dir is not None:
        _require_private_location(args.work_dir, "work-dir")
    validate_manifest(
        raw_manifest,
        verify_files=False,
        require_relative_paths=True,
    )
    manifest = resolve_manifest_paths(raw_manifest, args.manifest.parent)
    warnings = validate_manifest(manifest)
    environment = environment_snapshot()
    validate_environment(manifest, environment)
    capsule_path = getattr(args, "capsule", None)
    capsule_sha256 = getattr(args, "capsule_sha256", None)
    finalized_handoff = manifest["handoff"]["outcome"] != "pending"
    if finalized_handoff and (capsule_path is None or capsule_sha256 is None):
        raise ValueError("Итоговый handoff требует --capsule и --capsule-sha256")
    capsule_result = None
    if capsule_path is not None and capsule_sha256 is not None:
        capsule_result = verify_capsule(
            capsule_path,
            capsule_sha256,
            expected_manifest_path=args.manifest,
            expected_source_commit=manifest["handoff"]["source_commit"],
            expected_report_sha256=manifest["handoff"]["public_report_sha256"],
            expected_capsule_id=manifest["handoff"]["capsule_id"],
        )
    if args.validate_only:
        plan = safe_plan(manifest, warnings, environment)
        if capsule_result is not None:
            plan["capsule"] = capsule_result
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if args.prepare_asr is not None:
        raise ValueError("manifest v3 не поддерживает --prepare-asr")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    run_schedule(
        manifest, args.output, args.work_dir, subprocess_runner, environment=environment
    )


def synthetic_block_embeddings(
    cluster_sizes: Iterable[int],
    *,
    dimensions: int | None = None,
    seed: int = 0,
    noise: float = 0.0,
) -> np.ndarray:
    """Строит детерминированную фикстуру для дешевого smoke-test счетчиков."""
    sizes = tuple(cluster_sizes)
    random = np.random.default_rng(seed)
    dimension = dimensions or len(sizes)
    centers = random.normal(size=(len(sizes), dimension))
    centers /= np.linalg.norm(centers, axis=1)[:, None]
    rows = []
    for cluster, size in enumerate(sizes):
        rows.extend(
            centers[cluster] + random.normal(0.0, noise, dimension) for _ in range(size)
        )
    return np.asarray(rows)
