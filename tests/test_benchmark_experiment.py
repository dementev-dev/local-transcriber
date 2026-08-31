import copy
import hashlib
import json
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_transcriber import benchmark_experiment as experiment


def _write(path: Path, content: bytes) -> dict[str, object]:
    path.write_bytes(content)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def _artifact(tmp_path: Path, artifact_id: str, content: bytes) -> dict[str, object]:
    return {
        "id": artifact_id,
        "kind": "onnx-model",
        **_write(tmp_path / f"{artifact_id}.onnx", content),
        "source_url": f"https://example.invalid/{artifact_id}",
        "revision": "v1",
        "license": "Apache-2.0",
    }


def _cell(name: str, position: int, role: str) -> dict[str, object]:
    return {
        "name": name,
        "stage": "intel",
        "phase": "window",
        "recording_ids": ["data-test", "t2-bdma", "yantar"],
        "window_shift_ratio": 0.1 if role == "baseline" else 0.2,
        "segmentation_artifact_id": "pyannote-fp32",
        "embedding_artifact_id": "wespeaker-fp32",
        "counter": "threshold",
        "clustering": {"mode": "threshold", "threshold": 0.89, "num_clusters": None},
        "inference": {
            "mode": "sequential",
            "outer_workers": 1,
            "session_count": 1,
            "intra_op_threads": 4,
            "inter_op_threads": 1,
            "batch_size": 1,
        },
        "repetition": 1,
        "schedule_position": position,
        "pair_id": "window-ab",
        "pair_role": role,
        "enabled": True,
        "skip_reason": None,
        "build_sha256": "9" * 64,
    }


def _manifest(tmp_path: Path) -> dict[str, object]:
    artifacts = [
        _artifact(tmp_path, "pyannote-fp32", b"segmentation"),
        _artifact(tmp_path, "wespeaker-fp32", b"wespeaker"),
        _artifact(tmp_path, "wespeaker-qdq", b"qdq"),
    ]
    recordings = []
    for index, (recording_id, speakers) in enumerate(
        (("data-test", 3), ("t2-bdma", 2), ("yantar", 2))
    ):
        media = _write(tmp_path / f"{recording_id}.wav", recording_id.encode())
        words = _write(
            tmp_path / f"{recording_id}.words.json",
            json.dumps([{"start": 0.0, "end": 0.5, "text": "слово"}]).encode(),
        )
        reference = None
        if index:
            side = _write(tmp_path / f"{recording_id}.md", b"reference")
            reference = {
                "path": side["path"],
                "sha256": side["sha256"],
                "format": "hypescribe_markdown",
            }
        recordings.append(
            {
                "id": recording_id,
                "role": "control",
                "path": media["path"],
                "sha256": media["sha256"],
                "start": 0.0,
                "duration": 300.0,
                "expected_speakers": speakers,
                "reference": reference,
                "asr_words": {
                    "path": words["path"],
                    "sha256": words["sha256"],
                    "format": "local_transcriber_words_v1",
                },
            }
        )
    calibration = _write(tmp_path / "calibration.wav", b"separate calibration")
    return {
        "schema_version": 3,
        "experiment": {
            "id": "cpu-diarization-intel",
            "stage": "intel",
            "source_commit": "1" * 64,
            "handoff_commit": None,
            "sherpa_patch_sha256": "2" * 64,
            "python_version": "3.13.0",
            "dependencies": {
                "sherpa-onnx": "1.13.6",
                "sherpa-onnx-core": "1.13.6",
                "onnxruntime": "1.28.0",
                "numpy": "2.4.3",
                "onnx": "1.20.0",
                "kaldi-native-fbank": "1.22.3",
                "psutil": "7.0.0",
            },
        },
        "machine": {
            "id": "intel-i7-6820hq",
            "sku": "Intel Core i7-6820HQ",
            "physical_cores": 4,
            "logical_cores": os.cpu_count(),
            "ram_bytes": 32 * 1024**3,
            "os": "Linux test",
            "cpu_flags": ["avx", "avx2"],
            "power": {
                "supply": "ac",
                "profile": "balanced",
                "governor": "powersave",
                "turbo": True,
                "power_limits": "platform-default",
                "temperature_celsius": 55.0,
                "throttling": False,
                "swap_total_bytes": 8 * 1024**3,
                "swap_used_bytes": 0,
            },
        },
        "artifacts": artifacts,
        "recordings": recordings,
        "calibration": [
            {
                "id": "calibration-1",
                "path": calibration["path"],
                "sha256": calibration["sha256"],
                "source_id": "calibration-source-1",
                "source_sha256": calibration["sha256"],
                "source_start": 10.0,
                "source_duration": 30.0,
                "derived": False,
            }
        ],
        "qdq": {
            "enabled": True,
            "source_ids": ["calibration-1"],
            "preprocessing": {"sample_rate": 16000},
            "reader": {"name": "speaker-features-v1"},
            "quant_pre_process": {"skip_optimization": False},
            "quantize_static": {
                "format": "QDQ",
                "activation": "QInt8",
                "weight": "QInt8",
            },
            "excluded_nodes": [],
            "tool_versions": {"onnxruntime": "1.28.0"},
            "graph_artifact_id": "wespeaker-qdq",
            "skip_reason": None,
        },
        "cells": [_cell("W0", 1, "baseline"), _cell("W2", 2, "candidate")],
        "handoff": {
            "outcome": "pending",
            "source_commit": None,
            "candidate_recipes": [],
            "public_report_sha256": None,
            "capsule_id": None,
            "capsule_sha256": None,
        },
        "rss_sample_interval_ms": 10,
    }


def _environment() -> dict[str, object]:
    return {
        "packages": {
            "sherpa-onnx": "1.13.6",
            "sherpa-onnx-core": "1.13.6",
            "onnxruntime": "1.28.0",
            "numpy": "2.4.3",
            "onnx": "1.20.0",
            "kaldi-native-fbank": "1.22.3",
            "psutil": "7.0.0",
        },
        "python": "3.13.0",
        "platform": "Linux test",
        "logical_cores": os.cpu_count(),
        "physical_cores": 4,
        "cpu_sku": "Intel Core i7-6820HQ",
        "cpu_flags": ["avx", "avx2"],
        "ram_bytes": 32 * 1024**3,
        "swap_total_bytes": 8 * 1024**3,
        "swap_used_bytes": 0,
        "build_sha256": "9" * 64,
    }


def _finalized_handoff_manifest(tmp_path: Path) -> dict[str, object]:
    manifest = _manifest(tmp_path)
    baseline = manifest["cells"][0]
    baseline.update(
        name="W0",
        phase="combination",
        pair_id="combination-c2",
        pair_role="baseline",
        schedule_position=10,
    )
    candidate = copy.deepcopy(baseline)
    candidate.update(
        name="C2",
        pair_role="candidate",
        schedule_position=11,
    )
    candidate["inference"].update(
        mode="shared-session",
        outer_workers=2,
        session_count=1,
        intra_op_threads=4,
    )
    manifest["cells"] = [baseline, candidate]
    cell_id = experiment.make_cell_id(manifest, candidate)
    manifest["handoff"] = {
        "outcome": "handoff",
        "source_commit": manifest["experiment"]["source_commit"],
        "candidate_recipes": [
            {
                "name": "C2",
                "cell_id": cell_id,
                "result_file": "results/w0.json",
                "result_sha256": "3" * 64,
                "mandatory_passed": True,
                "memory_passed": True,
                "diagnostic_approved": True,
            }
        ],
        "public_report_sha256": "4" * 64,
        "capsule_id": "opaque-41",
        "capsule_sha256": "5" * 64,
    }
    return manifest


def _ryzen_capsule_fixture(tmp_path: Path):
    intel = _finalized_handoff_manifest(tmp_path)
    payloads = {}

    def make_relative(item, archive_path):
        content = Path(item["path"]).read_bytes()
        item["path"] = archive_path
        payloads[archive_path] = content
        local_path = tmp_path / archive_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(content)

    for artifact in intel["artifacts"]:
        make_relative(artifact, f"artifacts/{artifact['id']}.bin")
    for recording in intel["recordings"]:
        make_relative(recording, f"recordings/{recording['id']}.wav")
        make_relative(
            recording["asr_words"], f"sidecars/{recording['id']}.words.json"
        )
        if recording["reference"] is not None:
            make_relative(
                recording["reference"], f"references/{recording['id']}.md"
            )
    for source in intel["calibration"]:
        make_relative(source, f"calibration/{source['id']}.wav")

    result_bytes = b'{"aggregate":{"quality_passed":true}}'
    result_path = "results/c2.json"
    payloads[result_path] = result_bytes
    candidate = intel["handoff"]["candidate_recipes"][0]
    candidate["result_file"] = result_path
    candidate["result_sha256"] = hashlib.sha256(result_bytes).hexdigest()
    intel_bytes = json.dumps(intel, separators=(",", ":")).encode()
    files = {
        "manifest.json": hashlib.sha256(intel_bytes).hexdigest(),
        **{name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()},
    }
    capsule = {
        "capsule_id": intel["handoff"]["capsule_id"],
        "source_commit": intel["handoff"]["source_commit"],
        "public_report_sha256": intel["handoff"]["public_report_sha256"],
        "files": files,
    }
    archive_path = tmp_path / "capsule.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("capsule.json", json.dumps(capsule))
        archive.writestr("manifest.json", intel_bytes)
        for name, content in payloads.items():
            archive.writestr(name, content)

    ryzen = copy.deepcopy(intel)
    ryzen["experiment"].update(
        id="cpu-diarization-ryzen",
        stage="ryzen",
        handoff_commit="6" * 40,
    )
    ryzen["machine"].update(
        id="ryzen-8845h",
        sku="AMD Ryzen 7 8845H",
        physical_cores=8,
        logical_cores=16,
    )
    for cell in ryzen["cells"]:
        cell["stage"] = "ryzen"
        cell["inference"]["intra_op_threads"] = 8
    return archive_path, ryzen, files["manifest.json"]


def _guard(**changes) -> dict[str, object]:
    result = {
        "supply": "ac",
        "profile": "balanced",
        "governor": "powersave",
        "turbo": True,
        "power_limits": {"package": "platform-default"},
        "temperature_celsius": 55.0,
        "throttling": False,
        "throttle_count": 10,
        "swap_used_bytes": 0,
        "swap_sin_bytes": 100,
        "swap_sout_bytes": 200,
    }
    result.update(changes)
    return result


def test_manifest_identity_is_path_independent_and_covers_schedule(tmp_path):
    manifest = _manifest(tmp_path)
    experiment.validate_manifest(manifest)
    first_id = experiment.make_cell_id(manifest, manifest["cells"][0])
    moved = copy.deepcopy(manifest)
    for artifact in moved["artifacts"]:
        artifact["path"] = "/private/moved/" + Path(artifact["path"]).name
    for recording in moved["recordings"]:
        recording["path"] = "/private/moved/recording"
        recording["asr_words"]["path"] = "/private/moved/words"
        if recording["reference"]:
            recording["reference"]["path"] = "/private/moved/reference"
    moved["calibration"][0]["path"] = "/private/moved/calibration"

    assert experiment.make_cell_id(moved, moved["cells"][0]) == first_id

    moved["cells"][0]["schedule_position"] = 3
    assert experiment.make_cell_id(moved, moved["cells"][0]) != first_id
    assert experiment.make_experiment_id(moved) != experiment.make_experiment_id(
        manifest
    )

    finalized = copy.deepcopy(manifest)
    finalized["handoff"]["outcome"] = "stop-before-ryzen"
    assert experiment.make_experiment_id(finalized) == experiment.make_experiment_id(
        manifest
    )


def test_cell_identity_covers_sidecars_and_qdq_recipe(tmp_path):
    manifest = _manifest(tmp_path)
    cell = manifest["cells"][0]
    cell["embedding_artifact_id"] = "wespeaker-qdq"
    first_id = experiment.make_cell_id(manifest, cell)

    changed_sidecar = copy.deepcopy(manifest)
    changed_sidecar["recordings"][0]["asr_words"]["sha256"] = "7" * 64
    assert experiment.make_cell_id(changed_sidecar, changed_sidecar["cells"][0]) != first_id

    changed_recipe = copy.deepcopy(manifest)
    changed_recipe["qdq"]["quantize_static"]["weight"] = "QInt4"
    assert experiment.make_cell_id(changed_recipe, changed_recipe["cells"][0]) != first_id


def test_identity_and_runtime_validation_cover_exact_versions_and_build(tmp_path):
    manifest = _manifest(tmp_path)
    first_id = experiment.make_cell_id(manifest, manifest["cells"][0])

    changed_python = copy.deepcopy(manifest)
    changed_python["experiment"]["python_version"] = "3.13.1"
    assert (
        experiment.make_cell_id(changed_python, changed_python["cells"][0])
        != first_id
    )

    changed_numpy = copy.deepcopy(manifest)
    changed_numpy["experiment"]["dependencies"]["numpy"] = "2.4.4"
    assert (
        experiment.make_cell_id(changed_numpy, changed_numpy["cells"][0])
        != first_id
    )
    with pytest.raises(ValueError, match="numpy==2.4.4"):
        experiment.validate_environment(changed_numpy, _environment())

    changed_build = copy.deepcopy(_environment())
    changed_build["build_sha256"] = "8" * 64
    with pytest.raises(ValueError, match="runtime-сборки"):
        experiment.validate_environment(manifest, changed_build)


def test_portable_handoff_check_skips_only_machine_identity(tmp_path):
    manifest = _manifest(tmp_path)
    foreign_environment = copy.deepcopy(_environment())
    foreign_environment.update(
        platform="Linux Ryzen",
        logical_cores=24,
        physical_cores=12,
        cpu_sku="AMD Ryzen 9 5900X 12-Core Processor",
        cpu_flags=["avx", "avx2", "avx512f"],
        ram_bytes=64 * 1024**3,
        swap_total_bytes=0,
    )

    experiment.validate_environment(
        manifest,
        foreign_environment,
        check_machine=False,
    )

    foreign_environment["packages"]["numpy"] = "2.4.4"
    with pytest.raises(ValueError, match="numpy==2.4.3"):
        experiment.validate_environment(
            manifest,
            foreign_environment,
            check_machine=False,
        )


def test_ryzen_manifest_preserves_intel_capsule_inputs_and_evidence(tmp_path):
    intel = _finalized_handoff_manifest(tmp_path)
    experiment.validate_manifest(intel, verify_files=False)
    ryzen = copy.deepcopy(intel)
    ryzen["experiment"].update(
        id="cpu-diarization-ryzen",
        stage="ryzen",
        handoff_commit="6" * 40,
    )
    ryzen["machine"].update(
        id="ryzen-8845h",
        sku="AMD Ryzen 7 8845H",
        physical_cores=8,
        logical_cores=16,
    )
    for cell in ryzen["cells"]:
        cell["stage"] = "ryzen"
        cell["inference"]["intra_op_threads"] = 8
    ryzen["cells"][0]["name"] = "R-W0-primary"
    ryzen["cells"][1]["name"] = "R-C2-primary"

    experiment.validate_manifest(ryzen, verify_files=False)
    experiment._validate_ryzen_capsule_manifest(ryzen, intel)

    changed_input = copy.deepcopy(ryzen)
    changed_input["artifacts"][0]["sha256"] = "7" * 64
    with pytest.raises(ValueError, match="закрепленные входы"):
        experiment._validate_ryzen_capsule_manifest(changed_input, intel)

    changed_handoff = copy.deepcopy(ryzen)
    changed_handoff["handoff"]["public_report_sha256"] = "8" * 64
    with pytest.raises(ValueError, match="handoff"):
        experiment._validate_ryzen_capsule_manifest(changed_handoff, intel)

    changed_recipe = copy.deepcopy(ryzen)
    changed_recipe["cells"][1]["window_shift_ratio"] = 0.2
    with pytest.raises(ValueError, match="рецептом handoff"):
        experiment._validate_ryzen_capsule_manifest(changed_recipe, intel)

    changed_candidate = copy.deepcopy(intel)
    changed_candidate["handoff"]["candidate_recipes"][0]["cell_id"] = "8" * 64
    changed_candidate_ryzen = copy.deepcopy(ryzen)
    changed_candidate_ryzen["handoff"]["candidate_recipes"][0]["cell_id"] = (
        "8" * 64
    )
    with pytest.raises(ValueError, match="Intel combination"):
        experiment._validate_ryzen_capsule_manifest(
            changed_candidate_ryzen, changed_candidate
        )

    changed_baseline = copy.deepcopy(ryzen)
    changed_baseline["cells"][0]["window_shift_ratio"] = 0.5
    with pytest.raises(ValueError, match="baseline"):
        experiment._validate_ryzen_capsule_manifest(changed_baseline, intel)

    changed_phase = copy.deepcopy(ryzen)
    changed_phase["cells"][1]["phase"] = "mechanism"
    changed_phase["cells"][1]["window_shift_ratio"] = 0.5
    with pytest.raises(ValueError, match="механизмом handoff"):
        experiment._validate_ryzen_capsule_manifest(changed_phase, intel)

    changed_patch = copy.deepcopy(ryzen)
    changed_patch["experiment"]["sherpa_patch_sha256"] = "7" * 64
    with pytest.raises(ValueError, match="runtime"):
        experiment._validate_ryzen_capsule_manifest(changed_patch, intel)

    changed_dependencies = copy.deepcopy(ryzen)
    changed_dependencies["experiment"]["dependencies"]["onnxruntime"] = "1.99.0"
    with pytest.raises(ValueError, match="runtime"):
        experiment._validate_ryzen_capsule_manifest(changed_dependencies, intel)

    changed_rss_interval = copy.deepcopy(ryzen)
    changed_rss_interval["rss_sample_interval_ms"] = 5000
    with pytest.raises(ValueError, match="RSS"):
        experiment._validate_ryzen_capsule_manifest(changed_rss_interval, intel)

    literal_workers_intel = copy.deepcopy(intel)
    literal_workers_intel["cells"][1]["inference"]["outer_workers"] = 4
    literal_workers_intel["handoff"]["candidate_recipes"][0]["cell_id"] = (
        experiment.make_cell_id(
            literal_workers_intel, literal_workers_intel["cells"][1]
        )
    )
    literal_workers_ryzen = copy.deepcopy(ryzen)
    literal_workers_ryzen["handoff"] = copy.deepcopy(
        literal_workers_intel["handoff"]
    )
    literal_workers_ryzen["cells"][1]["inference"].update(
        outer_workers=4,
        intra_op_threads=4,
    )
    experiment._validate_ryzen_capsule_manifest(
        literal_workers_ryzen, literal_workers_intel
    )
    scaled_workers_ryzen = copy.deepcopy(literal_workers_ryzen)
    scaled_workers_ryzen["cells"][1]["inference"].update(
        outer_workers=8,
        intra_op_threads=8,
    )
    with pytest.raises(ValueError, match="рецептом handoff"):
        experiment._validate_ryzen_capsule_manifest(
            scaled_workers_ryzen, literal_workers_intel
        )


@pytest.mark.parametrize("cell_id", [None, "not-a-sha256"])
def test_ryzen_manifest_rejects_invalid_handoff_cell_id(tmp_path, cell_id):
    manifest = _finalized_handoff_manifest(tmp_path)
    manifest["experiment"].update(stage="ryzen", handoff_commit="6" * 40)
    for cell in manifest["cells"]:
        cell["stage"] = "ryzen"
    manifest["handoff"]["candidate_recipes"][0]["cell_id"] = cell_id

    with pytest.raises(ValueError, match="cell_id"):
        experiment.validate_manifest(manifest, verify_files=False)


def test_manifest_accepts_unavailable_temperature_sensor(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["machine"]["power"]["temperature_celsius"] = None

    experiment.validate_manifest(manifest, verify_files=False)

    manifest["machine"]["power"]["temperature_celsius"] = -1.0
    with pytest.raises(ValueError, match="temperature_celsius"):
        experiment.validate_manifest(manifest, verify_files=False)


def test_manifest_rejects_stage_boundary_thread_budget_and_non_alternating_ab(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["cells"][0]["stage"] = "ryzen"
    with pytest.raises(ValueError, match="граница"):
        experiment.validate_manifest(manifest, verify_files=False)

    manifest = _manifest(tmp_path)
    manifest["cells"][0]["inference"]["outer_workers"] = 5
    with pytest.raises(ValueError, match="бюджет"):
        experiment.validate_manifest(manifest, verify_files=False)

    manifest = _manifest(tmp_path)
    manifest["cells"][1]["pair_role"] = "baseline"
    with pytest.raises(ValueError, match="baseline/candidate"):
        experiment.validate_manifest(manifest, verify_files=False)

    manifest = _manifest(tmp_path)
    second_pair = []
    for source, name, position in (
        (manifest["cells"][0], "W0-repeat", 3),
        (manifest["cells"][1], "W2-repeat", 4),
    ):
        cell = copy.deepcopy(source)
        cell.update(name=name, schedule_position=position, pair_id="window-ab-2")
        second_pair.append(cell)
    manifest["cells"].extend(second_pair)
    with pytest.raises(ValueError, match="начальная роль"):
        experiment.validate_manifest(manifest, verify_files=False)


def test_ab_pair_requires_exactly_two_adjacent_schedule_positions(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["cells"][1]["schedule_position"] = 3
    with pytest.raises(ValueError, match="соседними"):
        experiment.validate_manifest(manifest, verify_files=False)

    manifest = _manifest(tmp_path)
    for source, name, position in (
        (manifest["cells"][0], "W0-repeat", 3),
        (manifest["cells"][1], "W2-repeat", 4),
    ):
        cell = copy.deepcopy(source)
        cell.update(name=name, schedule_position=position)
        manifest["cells"].append(cell)
    with pytest.raises(ValueError, match="ровно одна пара"):
        experiment.validate_manifest(manifest, verify_files=False)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda manifest: manifest["cells"][0]["inference"].update(outer_workers=1.5), "целое"),
        (lambda manifest: manifest["cells"][0].update(window_shift_ratio=0.3), "сетку"),
        (lambda manifest: manifest["cells"][0]["clustering"].update(threshold=None), "число"),
        (
            lambda manifest: manifest["cells"][0].update(
                recording_ids=["data-test", "data-test"]
            ),
            "запись",
        ),
    ],
)
def test_manifest_rejects_fractional_or_ambiguous_cell_fields(
    tmp_path, mutate, message
):
    manifest = _manifest(tmp_path)
    mutate(manifest)

    with pytest.raises((TypeError, ValueError), match=message):
        experiment.validate_manifest(manifest, verify_files=False)


def test_manifest_rejects_enabled_qdq_without_graph(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["qdq"]["graph_artifact_id"] = None

    with pytest.raises(ValueError, match="нужен граф"):
        experiment.validate_manifest(manifest, verify_files=False)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"supply": "battery"}, "от сети"),
        ({"profile": "performance"}, "balanced"),
        ({"throttling": True}, "нестабильно"),
    ],
)
def test_manifest_rejects_unstable_power_contract(tmp_path, changes, message):
    manifest = _manifest(tmp_path)
    manifest["machine"]["power"].update(changes)

    with pytest.raises(ValueError, match=message):
        experiment.validate_manifest(manifest, verify_files=False)


def test_resolve_manifest_paths_rejects_symlink_escape(tmp_path):
    manifest = _manifest(tmp_path)
    capsule = tmp_path / "capsule"
    capsule.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (capsule / "escape").symlink_to(outside, target_is_directory=True)
    manifest["artifacts"][0]["path"] = "escape/model.onnx"

    with pytest.raises(ValueError, match="выходит за корень"):
        experiment.resolve_manifest_paths(manifest, capsule)


def test_raw_manifest_requires_capsule_relative_paths(tmp_path):
    manifest = _manifest(tmp_path)

    with pytest.raises(ValueError, match="относительным"):
        experiment.validate_manifest(
            manifest,
            verify_files=False,
            require_relative_paths=True,
        )


def test_private_locations_inside_repository_must_be_git_ignored():
    repository_root = Path(experiment.__file__).parents[2]

    with pytest.raises(ValueError, match="не игнорируется Git"):
        experiment._require_private_location(
            repository_root / "docs" / "private-manifest.json",
            "manifest",
        )

    experiment._require_private_location(
        repository_root / "tmp" / "private-manifest.json",
        "manifest",
    )


def test_qdq_rejects_control_origin_and_skips_missing_material_safely(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["calibration"][0]["source_sha256"] = manifest["recordings"][0]["sha256"]
    with pytest.raises(ValueError, match="контрольная запись"):
        experiment.validate_manifest(manifest)

    manifest = _manifest(tmp_path)
    Path(manifest["calibration"][0]["path"]).unlink()
    warnings = experiment.validate_manifest(manifest)
    plan = experiment.safe_plan(manifest, warnings, _environment())

    assert plan["qdq_available"] is False
    serialized = json.dumps(plan, ensure_ascii=False)
    assert str(tmp_path) not in serialized
    assert "separate calibration" not in serialized


@pytest.mark.parametrize(
    "cluster_sizes, dimensions, seed, noise, expected",
    [((3, 17), 4, 35, 0.1, 2), ((2, 3, 15), 3, 1, 0.3, 3)],
)
def test_nme_and_eigengap_count_block_fixtures(
    cluster_sizes, dimensions, seed, noise, expected
):
    embeddings = experiment.synthetic_block_embeddings(
        cluster_sizes, dimensions=dimensions, seed=seed, noise=noise
    )

    assert experiment.estimate_eigengap(embeddings)["num_clusters"] == expected
    assert experiment.estimate_nme(embeddings)["num_clusters"] == expected


def test_counter_selection_handles_mixed_model_outcomes():
    results = []
    for model_id, nme_counts, eigengap_counts, nme_time, eigengap_time in (
        ("campplus", [3, 2, 2], [3, 2, 2], 12.0, 11.5),
        ("titanet", [3, 2, 2], [3, 3, 2], 8.0, 3.0),
        ("broken", [3, 3, 2], [3, 2, 3], 1.0, 1.0),
    ):
        for counter, counts, wall in (
            ("nme", nme_counts, nme_time),
            ("eigengap", eigengap_counts, eigengap_time),
        ):
            for num_clusters in counts:
                results.append(
                    {
                        "embedding_id": model_id,
                        "counter": counter,
                        "num_clusters": num_clusters,
                        "wall_seconds": wall / 3,
                    }
                )

    assert experiment.select_counter(results) == {
        "broken": None,
        "campplus": "eigengap",
        "titanet": "nme",
    }


def test_window_selection_and_recipe_assembly_cover_step_only_and_deduplication():
    results = [
        {
            "window_shift_ratio": 0.1,
            "wall_seconds": 100.0,
            "quality_passed": True,
            "work_counters_explained": True,
        },
        {
            "window_shift_ratio": 0.15,
            "wall_seconds": 88.0,
            "quality_passed": True,
            "work_counters_explained": True,
        },
        {
            "window_shift_ratio": 0.2,
            "wall_seconds": 85.0,
            "quality_passed": True,
            "work_counters_explained": True,
        },
    ]
    assert experiment.select_working_shift(results) == 0.15
    assert experiment.assemble_candidate_recipes(0.15, []) == [
        {
            "id": "C0",
            "window_shift_ratio": 0.15,
            "embedding": "wespeaker-fp32",
            "inference": "sequential",
        }
    ]

    branches = [
        {
            "quality_passed": True,
            "memory_passed": True,
            "diagnostic_approved": True,
            "speedup": 0.2,
            "recipe": {"embedding": "wespeaker-qdq", "inference": "shared-session"},
            "isa_stack": True,
            "stack_order": 1,
        },
        {
            "quality_passed": True,
            "memory_passed": True,
            "diagnostic_approved": True,
            "speedup": 0.15,
            "recipe": {"embedding": "wespeaker-qdq", "inference": "shared-session"},
            "isa_stack": False,
        },
    ]
    recipes = experiment.assemble_candidate_recipes(0.15, branches)
    assert len(recipes) <= 3
    assert len(
        {
            json.dumps({k: v for k, v in item.items() if k != "id"}, sort_keys=True)
            for item in recipes
        }
    ) == len(recipes)

    unapproved = copy.deepcopy(branches)
    unapproved[0]["diagnostic_approved"] = False
    unapproved[0]["speedup"] = 0.9
    assert all(
        item.get("embedding") != "wespeaker-qdq"
        for item in experiment.assemble_candidate_recipes(0.15, unapproved[:1])
    )


def test_schedule_preserves_raw_repeats_and_resumes_only_success(tmp_path):
    manifest = _manifest(tmp_path)
    output_path = tmp_path / "results.json"
    work_dir = tmp_path / "work"
    calls = []

    def runner(request):
        calls.append(request["cell"]["name"])
        return {"raw": request["cell"]["repetition"]}

    first = experiment.run_schedule(
        manifest,
        output_path,
        work_dir,
        runner,
        environment=_environment(),
        guard_reader=_guard,
    )
    resumed_environment = _environment()
    resumed_environment["swap_used_bytes"] = 4096
    second = experiment.run_schedule(
        manifest,
        output_path,
        work_dir,
        runner,
        environment=resumed_environment,
        guard_reader=_guard,
    )

    assert calls == ["W0", "W2"]
    assert len(first["cells"]) == 2
    assert len(second["cells"]) == 2
    assert all(item["result"] == {"raw": 1} for item in second["cells"])
    assert all(item["guard"]["attempts"][0]["valid"] for item in second["cells"])


def test_runtime_profile_falls_back_to_powerprofilesctl(monkeypatch):
    monkeypatch.setattr(experiment, "_read_optional_text", lambda _path: None)
    monkeypatch.setattr(
        experiment.subprocess,
        "run",
        lambda *args, **kwargs: experiment.subprocess.CompletedProcess(
            args[0], 0, stdout="balanced\n", stderr=""
        ),
    )

    assert experiment._runtime_profile() == "balanced"


def test_runtime_guard_retries_once_and_persists_transition_evidence(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["cells"][1]["enabled"] = False
    snapshots = iter(
        [
            _guard(),
            _guard(swap_sin_bytes=101),
            _guard(temperature_celsius=57.0),
            _guard(temperature_celsius=58.0),
        ]
    )
    calls = []

    output = experiment.run_schedule(
        manifest,
        tmp_path / "results.json",
        tmp_path / "work",
        lambda request: calls.append(request["cell"]["name"]) or {"raw": 1},
        environment=_environment(),
        guard_reader=lambda: next(snapshots),
    )

    assert calls == ["W0", "W0"]
    attempts = output["cells"][0]["guard"]["attempts"]
    assert [item["valid"] for item in attempts] == [False, True]
    assert attempts[0]["reasons"] == ["swap_sin_bytes-changed"]
    assert attempts[0]["before"]["temperature_celsius"] == 55.0
    assert attempts[1]["after"]["temperature_celsius"] == 58.0


def test_runtime_guard_rejects_second_invalid_transition(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["cells"][1]["enabled"] = False
    snapshots = iter(
        [
            _guard(),
            _guard(throttle_count=11),
            _guard(),
            _guard(swap_sout_bytes=201),
        ]
    )
    output_path = tmp_path / "results.json"

    with pytest.raises(RuntimeError, match="W0"):
        experiment.run_schedule(
            manifest,
            output_path,
            tmp_path / "work",
            lambda _request: {"raw": 1},
            environment=_environment(),
            guard_reader=lambda: next(snapshots),
        )

    persisted = json.loads(output_path.read_text(encoding="utf-8"))["cells"][0]
    assert persisted["status"] == "failed"
    assert persisted["error_type"] == "RuntimeGuardError"
    assert len(persisted["guard"]["attempts"]) == 2


@pytest.mark.parametrize(
    ("after", "reason"),
    [
        (_guard(governor="performance"), "governor-mismatch"),
        (_guard(turbo=False), "turbo-mismatch"),
        (_guard(throttling=True), "throttling"),
        (
            _guard(power_limits={"package": "changed"}),
            "power-limits-changed",
        ),
        (_guard(swap_sout_bytes=201), "swap_sout_bytes-changed"),
    ],
)
def test_runtime_guard_checks_stable_machine_conditions(tmp_path, after, reason):
    reasons = experiment.validate_runtime_guard_transition(
        _manifest(tmp_path), _guard(), after
    )

    assert reason in reasons


def test_failed_cell_is_persisted_without_private_exception_text(tmp_path):
    manifest = _manifest(tmp_path)
    output_path = tmp_path / "results.json"

    def runner(_request):
        raise RuntimeError(f"private path: {tmp_path / 'secret.wav'}")

    with pytest.raises(RuntimeError, match="W0"):
        experiment.run_schedule(
            manifest,
            output_path,
            tmp_path / "work",
            runner,
            environment=_environment(),
            guard_reader=_guard,
        )

    serialized = output_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in serialized
    assert "secret.wav" not in serialized
    assert "RuntimeError" in serialized


def test_residual_cluster_metrics_keep_time_and_word_units_separate():
    intervals = [
        {"speaker": 0, "start": 0.0, "end": 8.0},
        {"speaker": 1, "start": 8.0, "end": 10.0},
        {"speaker": 2, "start": 10.0, "end": 11.0},
    ]
    words = [
        {"speaker": 0},
        {"speaker": 1},
        {"speaker": 2},
        {"speaker": 2},
        {"speaker": None},
    ]

    metrics = experiment.residual_cluster_metrics(intervals, words, 2)

    assert metrics["residual_speech_time_share"] == pytest.approx(1 / 11)
    assert metrics["residual_assigned_word_share"] == pytest.approx(1 / 2)


def test_capsule_verification_checks_hashes_commits_and_safe_paths(tmp_path):
    payload = b"private"
    payload_hash = hashlib.sha256(payload).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_value = {
        "schema_version": 3,
        "artifacts": [
            {
                "id": "input",
                "path": "data/payload.bin",
                "sha256": payload_hash,
            }
        ],
        "recordings": [],
        "calibration": [],
    }
    manifest_bytes = json.dumps(manifest_value).encode()
    manifest_path.write_bytes(manifest_bytes)
    capsule = {
        "capsule_id": "opaque-41",
        "source_commit": "3" * 64,
        "public_report_sha256": "4" * 64,
        "files": {
            "manifest.json": hashlib.sha256(manifest_bytes).hexdigest(),
            "data/payload.bin": payload_hash,
        },
    }
    archive_path = tmp_path / "capsule.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("capsule.json", json.dumps(capsule))
        archive.writestr("manifest.json", manifest_bytes)
        archive.writestr("data/payload.bin", payload)
    archive_hash = experiment.file_sha256(archive_path)

    result = experiment.verify_capsule(
        archive_path,
        archive_hash,
        expected_manifest_path=manifest_path,
        expected_source_commit="3" * 64,
        expected_report_sha256="4" * 64,
        expected_capsule_id="opaque-41",
    )

    assert result == {
        "capsule_id": "opaque-41",
        "source_commit": "3" * 64,
        "public_report_sha256": "4" * 64,
        "file_count": 2,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }

    with pytest.raises(ValueError, match="ровно один"):
        experiment.verify_capsule(
            archive_path,
            archive_hash,
            expected_manifest_path=None,
        )
    with pytest.raises(ValueError, match="ровно один"):
        experiment.verify_capsule(
            archive_path,
            archive_hash,
            expected_manifest_path=manifest_path,
            expected_ryzen_manifest=manifest_value,
        )

    incomplete_capsule = copy.deepcopy(capsule)
    incomplete_capsule["files"].pop("data/payload.bin")
    incomplete_path = tmp_path / "incomplete.zip"
    with zipfile.ZipFile(incomplete_path, "w") as archive:
        archive.writestr("capsule.json", json.dumps(incomplete_capsule))
        archive.writestr("manifest.json", manifest_bytes)
    with pytest.raises(ValueError, match="Входной файл manifest"):
        experiment.verify_capsule(
            incomplete_path,
            experiment.file_sha256(incomplete_path),
            expected_manifest_path=manifest_path,
        )

    unrelated_manifest = tmp_path / "unrelated-manifest.json"
    unrelated_manifest.write_bytes(b'{"schema_version":3,"other":true}\n')
    with pytest.raises(ValueError, match="не совпал с manifest.json капсулы"):
        experiment.verify_capsule(
            archive_path,
            archive_hash,
            expected_manifest_path=unrelated_manifest,
        )

    unsafe_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe_path, "w") as archive:
        archive.writestr("capsule.json", json.dumps(capsule))
        archive.writestr("manifest.json", manifest_bytes)
        archive.writestr("../payload.bin", payload)
    with pytest.raises(ValueError, match="небезопасный путь"):
        experiment.verify_capsule(
            unsafe_path,
            experiment.file_sha256(unsafe_path),
            expected_manifest_path=manifest_path,
        )


def test_run_cli_accepts_verified_ryzen_capsule_manifest(
    tmp_path, capsys, monkeypatch
):
    archive_path, ryzen, intel_manifest_sha256 = _ryzen_capsule_fixture(tmp_path)
    manifest_path = tmp_path / "ryzen.json"
    manifest_path.write_text(json.dumps(ryzen), encoding="utf-8")
    args = SimpleNamespace(
        manifest=manifest_path,
        output=None,
        work_dir=tmp_path / "work",
        capsule=archive_path,
        capsule_sha256=experiment.file_sha256(archive_path),
        validate_only=True,
        prepare_asr=None,
    )
    monkeypatch.setattr(experiment, "environment_snapshot", _environment)

    experiment.run_cli(args, ryzen)

    plan = json.loads(capsys.readouterr().out)
    assert plan["stage"] == "ryzen"
    assert plan["capsule"]["manifest_sha256"] == intel_manifest_sha256
    assert plan["capsule"]["file_count"] == 14


def test_verify_capsule_rejects_unlinked_ryzen_manifest(tmp_path):
    archive_path, ryzen, _manifest_sha256 = _ryzen_capsule_fixture(tmp_path)
    ryzen["artifacts"][0]["sha256"] = "7" * 64

    with pytest.raises(ValueError, match="закрепленные входы"):
        experiment.verify_capsule(
            archive_path,
            experiment.file_sha256(archive_path),
            expected_manifest_path=None,
            expected_ryzen_manifest=ryzen,
        )


def test_capsule_binds_candidate_result_evidence_to_file_index(tmp_path):
    input_bytes = b"input"
    input_hash = hashlib.sha256(input_bytes).hexdigest()
    result_bytes = b'{"aggregate":{"quality_passed":true}}'
    result_hash = hashlib.sha256(result_bytes).hexdigest()
    manifest_value = {
        "artifacts": [
            {"id": "input", "path": "data/input.bin", "sha256": input_hash}
        ],
        "recordings": [],
        "calibration": [],
        "handoff": {
            "candidate_recipes": [
                {
                    "result_file": "results/candidate.json",
                    "result_sha256": result_hash,
                }
            ]
        }
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_bytes = json.dumps(manifest_value, separators=(",", ":")).encode()
    manifest_path.write_bytes(manifest_bytes)
    capsule = {
        "capsule_id": "opaque-41",
        "source_commit": "3" * 64,
        "public_report_sha256": "4" * 64,
        "files": {
            "manifest.json": hashlib.sha256(manifest_bytes).hexdigest(),
            "data/input.bin": input_hash,
            "results/candidate.json": result_hash,
        },
    }
    archive_path = tmp_path / "capsule.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("capsule.json", json.dumps(capsule))
        archive.writestr("manifest.json", manifest_bytes)
        archive.writestr("data/input.bin", input_bytes)
        archive.writestr("results/candidate.json", result_bytes)

    experiment.verify_capsule(
        archive_path,
        experiment.file_sha256(archive_path),
        expected_manifest_path=manifest_path,
    )

    broken_manifest_value = copy.deepcopy(manifest_value)
    broken_manifest_value["handoff"]["candidate_recipes"][0]["result_sha256"] = (
        "5" * 64
    )
    broken_manifest_bytes = json.dumps(
        broken_manifest_value, separators=(",", ":")
    ).encode()
    broken_manifest_path = tmp_path / "broken-manifest.json"
    broken_manifest_path.write_bytes(broken_manifest_bytes)
    capsule["files"]["manifest.json"] = hashlib.sha256(
        broken_manifest_bytes
    ).hexdigest()
    broken_path = tmp_path / "broken-capsule.zip"
    with zipfile.ZipFile(broken_path, "w") as archive:
        archive.writestr("capsule.json", json.dumps(capsule))
        archive.writestr("manifest.json", broken_manifest_bytes)
        archive.writestr("data/input.bin", input_bytes)
        archive.writestr("results/candidate.json", result_bytes)
    with pytest.raises(ValueError, match="Результат кандидата"):
        experiment.verify_capsule(
            broken_path,
            experiment.file_sha256(broken_path),
            expected_manifest_path=broken_manifest_path,
        )


def test_handoff_enforces_candidate_limit_and_stop_contract(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["handoff"]["outcome"] = "stop-before-ryzen"
    manifest["handoff"]["candidate_recipes"] = ["C0"]
    with pytest.raises(ValueError, match="ноль кандидатов"):
        experiment.validate_manifest(manifest, verify_files=False)

    manifest = _manifest(tmp_path)
    manifest["handoff"]["outcome"] = "handoff"
    manifest["handoff"]["candidate_recipes"] = ["C0", "C1", "C2", "C3"]
    with pytest.raises(ValueError, match="трех"):
        experiment.validate_manifest(manifest, verify_files=False)


def test_finalize_handoff_binds_candidate_result_and_all_gates(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["cells"][0]["phase"] = "combination"
    cell_id = experiment.make_cell_id(manifest, manifest["cells"][0])
    payload = {"aggregate": {"quality_passed": True}, "raw": [1, 2, 3]}
    output = {
        "experiment_id": experiment.make_experiment_id(manifest),
        "cells": [{"cell_id": cell_id, "status": "complete", "result": payload}],
    }

    finalized = experiment.finalize_handoff(
        manifest,
        output,
        ["W0"],
        capsule_root=tmp_path / "capsule",
        gate_evidence={
            "W0": {"memory_passed": True, "diagnostic_approved": True}
        },
        public_report_sha256="4" * 64,
        capsule_id="opaque-41",
    )

    candidate = finalized["handoff"]["candidate_recipes"][0]
    assert candidate == {
        "name": "W0",
        "cell_id": cell_id,
        "result_file": f"results/{cell_id}.json",
        "result_sha256": experiment._digest_json(payload),
        "mandatory_passed": True,
        "memory_passed": True,
        "diagnostic_approved": True,
    }
    result_file = tmp_path / "capsule" / candidate["result_file"]
    assert experiment.file_sha256(result_file) == candidate["result_sha256"]
    assert json.loads(result_file.read_text(encoding="utf-8")) == payload
    experiment.validate_manifest(finalized, verify_files=False)

    finalized["handoff"]["capsule_sha256"] = "5" * 64
    experiment.validate_manifest(finalized, verify_files=False)


def test_finalize_handoff_rejects_missing_diagnostic_approval(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["cells"][0]["phase"] = "combination"
    cell_id = experiment.make_cell_id(manifest, manifest["cells"][0])
    output = {
        "experiment_id": experiment.make_experiment_id(manifest),
        "cells": [
            {
                "cell_id": cell_id,
                "status": "complete",
                "result": {"aggregate": {"quality_passed": True}},
            }
        ],
    }

    with pytest.raises(ValueError, match="обязательные gates"):
        experiment.finalize_handoff(
            manifest,
            output,
            ["W0"],
            capsule_root=tmp_path / "capsule",
            gate_evidence={
                "W0": {"memory_passed": True, "diagnostic_approved": False}
            },
            public_report_sha256="4" * 64,
            capsule_id="opaque-41",
        )
