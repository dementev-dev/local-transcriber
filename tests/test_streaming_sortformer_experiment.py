import copy
import ctypes
import hashlib
import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from local_transcriber import streaming_sortformer_experiment as experiment


def test_experiment_module_is_excluded_from_production_wheel():
    project = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    excluded = project["tool"]["hatch"]["build"]["targets"]["wheel"]["exclude"]
    assert "src/local_transcriber/streaming_sortformer_experiment.py" in excluded


def _write(root: Path, name: str, payload: bytes) -> dict[str, object]:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _record(
    root: Path, prefix: str, *, duration: float, reference: bool
) -> dict[str, object]:
    media = _write(root, f"media/{prefix}.wav", f"audio-{prefix}".encode())
    sidecar = _write(root, f"sidecars/{prefix}.json", f"words-{prefix}".encode())
    result = {
        **media,
        "duration_seconds": duration,
        "sidecar_path": sidecar["path"],
        "sidecar_sha256": sidecar["sha256"],
        "expected_cluster_count": 2,
    }
    if reference:
        reference_record = _write(
            root, f"references/{prefix}.json", f"ref-{prefix}".encode()
        )
        result.update(
            reference_path=reference_record["path"],
            reference_sha256=reference_record["sha256"],
        )
    return result


def _manifest(root: Path) -> dict[str, object]:
    inputs = []
    for index in range(3):
        source = _write(root, f"sources/{index}.wav", f"source-{index}".encode())
        inputs.append(
            {
                "source_path": source["path"],
                "source_sha256": source["sha256"],
                "source_size_bytes": source["size_bytes"],
                "fragment": _record(
                    root, f"fragment-{index}", duration=300.0, reference=True
                ),
                "full_recording": _record(
                    root, f"full-{index}", duration=900.0, reference=False
                ),
            }
        )
    return {
        "schema": "local-transcriber.streaming-sortformer-experiment.v1",
        "fresh_basis": {
            "marker": "issue-46-fresh-owner-reviewed-basis",
            "approval": {
                "decision": "approved",
                "timestamp": "2026-09-01T09:00:00+00:00",
            },
        },
        "source": {"commit": "a" * 40},
        "machine": _machine(),
        "inputs": inputs,
        "recipes": _recipes(),
        "schedule": experiment.build_fragment_schedule(
            [item["fragment"]["sha256"] for item in inputs],
            [item["fragment"]["sidecar_sha256"] for item in inputs],
        ),
        "rss_sample_interval_ms": 10,
    }


def _machine() -> dict[str, object]:
    return {
        "machine_sha256": "1" * 64,
        "physical_cores": 8,
        "logical_cores": 16,
        "affinity": list(range(8)),
        "cpuid_isa_sha256": "2" * 64,
        "ram_bytes": 16 * 1024**3,
        "os_sha256": "3" * 64,
        "compiler_sha256": "4" * 64,
        "runtime_versions_sha256": "5" * 64,
        "power_mode": "balanced",
        "governor": "powersave",
        "turbo": True,
    }


def _recipes() -> dict[str, object]:
    return {
        "W0-FRESH": {
            "sherpa_onnx_version": "1.13.6",
            "sherpa_onnx_core_version": "1.13.6",
            "segmentation_model_sha256": "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079",
            "embedding_model_sha256": "e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012",
            "window_shift_ratio": 0.1,
            "clustering": {
                "algorithm": "FastClustering",
                "mode": "auto",
                "threshold": 0.89,
            },
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
        },
        "SORTFORMER-V2-Q8": {
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
            "submodule_commits": {"ggml": "6" * 40},
            "compiler": "pinned-compiler",
            "cmake_version": "pinned-cmake",
            "executable_sha256": "7" * 64,
            "library_sha256": "8" * 64,
        },
    }


def _environment() -> dict[str, object]:
    return {"commit": "a" * 40, "clean": True}


def test_preflight_validates_fresh_exact_schema_before_loading(tmp_path):
    manifest = _manifest(tmp_path)
    calls = []

    outcome = experiment.preflight_and_load(
        manifest,
        root=tmp_path,
        environment_probe=_environment,
        w0_loader=lambda: calls.append("w0"),
        candidate_loader=lambda: calls.append("candidate"),
        validate_only=True,
    )

    assert outcome.state == "valid"
    assert outcome.fragment_count == 3
    assert outcome.full_recording_count == 3
    assert calls == []


def test_approved_fresh_basis_loads_models_after_preflight(tmp_path):
    manifest = _manifest(tmp_path)
    calls = []

    outcome = experiment.preflight_and_load(
        manifest,
        root=tmp_path,
        environment_probe=_environment,
        w0_loader=lambda: calls.append("w0"),
        candidate_loader=lambda: calls.append("candidate"),
    )

    assert outcome.state == "valid"
    assert calls == ["w0", "candidate"]


def test_fragment_without_reference_loads_models_after_preflight(tmp_path):
    manifest = _manifest(tmp_path)
    fragment = manifest["inputs"][0]["fragment"]
    (tmp_path / fragment["reference_path"]).unlink()
    fragment["reference_path"] = None
    fragment["reference_sha256"] = None
    calls = []

    outcome = experiment.preflight_and_load(
        manifest,
        root=tmp_path,
        environment_probe=_environment,
        w0_loader=lambda: calls.append("w0"),
        candidate_loader=lambda: calls.append("candidate"),
    )

    assert outcome.state == "valid"
    assert calls == ["w0", "candidate"]


@pytest.mark.parametrize("field", ["reference_path", "reference_sha256"])
def test_fragment_rejects_half_present_reference_before_loaders(tmp_path, field):
    manifest = _manifest(tmp_path)
    manifest["inputs"][0]["fragment"][field] = None
    calls = []

    with pytest.raises(experiment.PreflightError, match="reference-binding"):
        experiment.preflight_and_load(
            manifest,
            root=tmp_path,
            environment_probe=_environment,
            w0_loader=lambda: calls.append("w0"),
            candidate_loader=lambda: calls.append("candidate"),
        )

    assert calls == []


def test_missing_owner_approval_is_needs_user_and_never_loads(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["fresh_basis"]["approval"] = None
    calls = []

    outcome = experiment.preflight_and_load(
        manifest,
        root=tmp_path,
        environment_probe=_environment,
        w0_loader=lambda: calls.append("w0"),
        candidate_loader=lambda: calls.append("candidate"),
    )

    assert outcome.state == "needs-user"
    assert outcome.reason == "owner-approval-required"
    assert calls == []


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(schema="3"),
        lambda value: value.update(legacy_manifest="issue-41"),
        lambda value: value["inputs"].__setitem__(
            1,
            {**value["inputs"][1], "source_sha256": value["inputs"][0]["source_sha256"]},
        ),
        lambda value: value["inputs"][0]["fragment"].update(duration_seconds=299.9),
        lambda value: value["inputs"][0]["full_recording"].update(
            expected_cluster_count=5
        ),
        lambda value: value["source"].update(commit="b" * 40),
    ],
)
def test_preflight_rejects_schema_basis_counts_and_source_before_load(
    tmp_path, mutation
):
    manifest = _manifest(tmp_path)
    mutation(manifest)
    calls = []

    with pytest.raises(experiment.PreflightError):
        experiment.preflight_and_load(
            manifest,
            root=tmp_path,
            environment_probe=_environment,
            w0_loader=lambda: calls.append("w0"),
            candidate_loader=lambda: calls.append("candidate"),
        )

    assert calls == []


def test_preflight_rejects_hash_path_and_symlink_escape_without_leaking_path(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["inputs"][0]["fragment"]["sha256"] = "f" * 64
    calls = []
    with pytest.raises(experiment.PreflightError) as raised:
        experiment.preflight_and_load(
            manifest,
            root=tmp_path,
            environment_probe=_environment,
            w0_loader=lambda: calls.append("w0"),
            candidate_loader=lambda: calls.append("candidate"),
        )
    assert str(tmp_path) not in str(raised.value)
    assert calls == []

    outside = tmp_path.parent / f"outside-{datetime.now(UTC).timestamp()}"
    outside.mkdir()
    (outside / "escape.wav").write_bytes(b"escape")
    (tmp_path / "link").symlink_to(outside, target_is_directory=True)
    manifest = _manifest(tmp_path)
    manifest["inputs"][0]["fragment"]["path"] = "link/escape.wav"
    with pytest.raises(experiment.PreflightError, match="path-escape"):
        experiment.validate_manifest(manifest, root=tmp_path, environment=_environment())


def test_preflight_rejects_nul_path_without_leaking_or_loading(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["inputs"][0]["source_path"] = "private\x00source.wav"
    calls = []

    with pytest.raises(experiment.PreflightError) as raised:
        experiment.preflight_and_load(
            manifest,
            root=tmp_path,
            environment_probe=_environment,
            w0_loader=lambda: calls.append("w0"),
            candidate_loader=lambda: calls.append("candidate"),
        )

    assert str(raised.value) == "path-escape"
    assert "private" not in str(raised.value)
    assert calls == []


def test_preflight_rejects_legacy_cell_names_roots_and_recipe_pin_mismatch(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["schedule"][0]["semantic_id"] = "41-W0"
    with pytest.raises(experiment.PreflightError, match="legacy-identity"):
        experiment.validate_manifest(manifest, root=tmp_path, environment=_environment())

    manifest = _manifest(tmp_path)
    manifest["schedule"][0]["semantic_id"] = "W0-FRESH"
    with pytest.raises(experiment.PreflightError, match="schedule-mismatch"):
        experiment.validate_manifest(manifest, root=tmp_path, environment=_environment())

    manifest = _manifest(tmp_path)
    manifest["inputs"][0]["fragment"]["path"] = "issue-42-store/audio.wav"
    with pytest.raises(experiment.PreflightError, match="legacy-root"):
        experiment.validate_manifest(manifest, root=tmp_path, environment=_environment())

    manifest = _manifest(tmp_path)
    manifest["recipes"]["W0-FRESH"]["intra_op_threads"] = 4
    with pytest.raises(experiment.PreflightError, match="w0-pin-mismatch"):
        experiment.validate_manifest(manifest, root=tmp_path, environment=_environment())

    manifest = _manifest(tmp_path)
    manifest["recipes"]["SORTFORMER-V2-Q8"]["geometry"]["chunk_frames"] = 320
    with pytest.raises(experiment.PreflightError, match="candidate-pin-mismatch"):
        experiment.validate_manifest(manifest, root=tmp_path, environment=_environment())

    manifest = _manifest(tmp_path)
    manifest["recipes"]["SORTFORMER-V2-Q8"]["submodule_commits"]["extra"] = "9" * 40
    with pytest.raises(experiment.PreflightError, match="candidate-build-provenance"):
        experiment.validate_manifest(manifest, root=tmp_path, environment=_environment())


def test_schedule_has_exact_primary_five_final_and_three_gated_full_pairs(tmp_path):
    manifest = _manifest(tmp_path)
    schedule = manifest["schedule"]
    assert [(cell["pair_id"], cell["recipe"]) for cell in schedule] == [
        ("primary", "W0-FRESH"),
        ("primary", "SORTFORMER-V2-Q8"),
        ("final-1", "SORTFORMER-V2-Q8"),
        ("final-1", "W0-FRESH"),
        ("final-2", "W0-FRESH"),
        ("final-2", "SORTFORMER-V2-Q8"),
        ("final-3", "SORTFORMER-V2-Q8"),
        ("final-3", "W0-FRESH"),
        ("final-4", "W0-FRESH"),
        ("final-4", "SORTFORMER-V2-Q8"),
        ("final-5", "SORTFORMER-V2-Q8"),
        ("final-5", "W0-FRESH"),
    ]
    assert len({cell["semantic_id"] for cell in schedule}) == 12
    assert all(cell["fresh_process"] for cell in schedule)
    for left, right in zip(schedule[::2], schedule[1::2], strict=True):
        assert left["input_sha256"] == right["input_sha256"]
        assert left["sidecar_sha256"] == right["sidecar_sha256"]

    full_hashes = [item["full_recording"]["sha256"] for item in manifest["inputs"]]
    sidecars = [
        item["full_recording"]["sidecar_sha256"] for item in manifest["inputs"]
    ]
    assert experiment.build_full_recording_schedule(
        full_hashes,
        sidecars,
        final_median_speedup_percent=29.999,
        experiment_state=None,
    ) == []
    full = experiment.build_full_recording_schedule(
        full_hashes,
        sidecars,
        final_median_speedup_percent=30.0,
        experiment_state=None,
    )
    assert len(full) == 6
    assert [cell["pair_id"] for cell in full] == [
        "full-1", "full-1", "full-2", "full-2", "full-3", "full-3"
    ]


def test_fragment_state_machine_applies_primary_stop_before_final_cells(tmp_path):
    manifest = _manifest(tmp_path)
    store = experiment.ExperimentState(tmp_path / "primary-stop.json")
    calls = []

    def runner(cell):
        calls.append(cell["pair_id"])
        walls = [10.0, 10.0, 10.0]
        if cell["recipe"] == "SORTFORMER-V2-Q8":
            walls = [9.7, 9.7, 9.7]
        return {"mandatory_passed": True, "input_wall_seconds": walls}

    result = experiment.run_fragment_state_machine(
        manifest, runner=runner, store=store
    )

    assert result["state"] == "stop"
    assert result["first_reason"] == "equal-speed"
    assert calls == ["primary", "primary"]
    with pytest.raises(experiment.PreflightError, match="already-stopped"):
        experiment.build_full_recording_schedule(
            [item["full_recording"]["sha256"] for item in manifest["inputs"]],
            [
                item["full_recording"]["sidecar_sha256"]
                for item in manifest["inputs"]
            ],
            final_median_speedup_percent=40.0,
            experiment_state=store.current,
        )


def test_fragment_state_machine_aggregates_three_walls_and_opens_full_stage(tmp_path):
    manifest = _manifest(tmp_path)
    store = experiment.ExperimentState(tmp_path / "full-open.json")
    calls = []

    def runner(cell):
        calls.append((cell["pair_id"], cell["recipe"]))
        wall = 10.0 if cell["recipe"] == "W0-FRESH" else 6.0
        return {
            "mandatory_passed": True,
            "input_wall_seconds": [wall, wall, wall],
        }

    result = experiment.run_fragment_state_machine(
        manifest, runner=runner, store=store
    )

    assert len(calls) == 12
    assert result["state"] == "full-recordings"
    assert result["median_speedup_percent"] == pytest.approx(40.0)
    assert len(result["full_schedule"]) == 6


@pytest.mark.parametrize(
    ("speedup", "state", "reason"),
    [
        (-5.0, "stop", "equal-speed"),
        (5.0, "stop", "equal-speed"),
        (5.0001, "stop", "below-continuation"),
        (9.9999, "stop", "below-continuation"),
        (10.0, "continue", None),
    ],
)
def test_primary_decision_preserves_five_and_ten_percent_boundaries(
    speedup, state, reason
):
    assert experiment.primary_decision(speedup, mandatory_gates_passed=True) == {
        "state": state,
        "reason": reason,
    }


def test_final_statistics_use_exactly_five_values_and_thirty_percent_boundary():
    values = [28.0, 29.0, 30.0, 31.0, 80.0]
    assert experiment.final_statistics(values) == {
        "median_speedup_percent": 30.0,
        "mad_percentage_points": 1.0,
    }
    assert experiment.final_decision([28.0, 29.0, 29.999, 31.0, 80.0]) == {
        "state": "stop",
        "reason": "below-final-threshold",
        "median_speedup_percent": 29.999,
        "mad_percentage_points": pytest.approx(1.001),
    }
    assert experiment.final_decision(values)["state"] == "full-recordings"
    with pytest.raises(ValueError, match="five-final-speedups"):
        experiment.final_statistics([30.0] * 4)


def test_first_stop_is_atomic_and_later_cells_are_not_scheduled(tmp_path):
    store = experiment.ExperimentState(tmp_path / "public-state.json")
    calls = []
    cells = [
        {"semantic_id": "cell-a", "recipe": "W0-FRESH"},
        {"semantic_id": "cell-b", "recipe": "SORTFORMER-V2-Q8"},
        {"semantic_id": "cell-c", "recipe": "W0-FRESH"},
    ]

    state = experiment.execute_until_stop(
        cells,
        lambda cell: calls.append(cell["semantic_id"])
        or ({"mandatory_passed": cell["semantic_id"] != "cell-b"}),
        store=store,
    )

    assert calls == ["cell-a", "cell-b"]
    assert state["state"] == "stop"
    assert state["first_reason"] == "mandatory-gate-failed"
    assert experiment.SHA256_RE.fullmatch(state["result_sha256"])
    persisted = json.loads((tmp_path / "public-state.json").read_text())
    assert persisted == state
    assert store.stop("later-reason", {"ignored": True}) == state
    reloaded = experiment.ExperimentState(tmp_path / "public-state.json")
    assert reloaded.stop("process-restarted", {"ignored": "again"}) == state


def _word(start, end, text, speaker):
    return {"start": start, "end": end, "text": text, "speaker": speaker}


def _sidecar_word(start, end, text):
    return {"start": start, "end": end, "text": text}


def test_quality_gate_checks_text_order_intervals_counts_and_canonical_labels():
    sidecar = [
        _sidecar_word(0.0, 0.5, "один"),
        _sidecar_word(0.5, 1.0, "два"),
    ]
    output = [_word(0.0, 0.5, "один", 9), _word(0.5, 1.0, "два", 4)]
    intervals = [
        {"start": 0.0, "end": 0.5, "speaker": 9},
        {"start": 0.5, "end": 1.0, "speaker": 4},
    ]

    verdict = experiment.quality_gate(
        expected_cluster_count=2,
        observed_cluster_count=2,
        sidecar_words=sidecar,
        output_words=output,
        intervals=intervals,
        audio_duration_seconds=1.0,
        permitted_padding_seconds=0.0,
    )

    assert verdict == {"passed": True, "reason": None, "canonical_labels": [1, 2]}
    reordered = [output[1], output[0]]
    assert experiment.quality_gate(
        expected_cluster_count=2,
        observed_cluster_count=2,
        sidecar_words=sidecar,
        output_words=reordered,
        intervals=intervals,
        audio_duration_seconds=1.0,
        permitted_padding_seconds=0.0,
    )["reason"] == "word-order-mismatch"
    invalid = copy.deepcopy(intervals)
    invalid[1]["start"] = float("nan")
    assert experiment.quality_gate(
        expected_cluster_count=2,
        observed_cluster_count=2,
        sidecar_words=sidecar,
        output_words=output,
        intervals=invalid,
        audio_duration_seconds=1.0,
        permitted_padding_seconds=0.0,
    )["reason"] == "invalid-intervals"


def test_quality_gate_normalizes_text_and_canonicalizes_tied_intervals():
    sidecar = [
        _sidecar_word(0.0, 0.5, " один "),
        _sidecar_word(0.5, 1.0, " два "),
    ]
    output = [_word(0.0, 0.5, "один", "z"), _word(0.5, 1.0, "два", "a")]
    intervals = [
        {"start": 0.0, "end": 0.5, "speaker": "z"},
        {"start": 0.0, "end": 1.0, "speaker": "a"},
    ]

    first = experiment.quality_gate(
        expected_cluster_count=2,
        observed_cluster_count=2,
        sidecar_words=sidecar,
        output_words=output,
        intervals=intervals,
        audio_duration_seconds=1.0,
        permitted_padding_seconds=0.0,
    )
    second = experiment.quality_gate(
        expected_cluster_count=2,
        observed_cluster_count=2,
        sidecar_words=sidecar,
        output_words=output,
        intervals=list(reversed(intervals)),
        audio_duration_seconds=1.0,
        permitted_padding_seconds=0.0,
    )

    assert first == second == {
        "passed": True,
        "reason": None,
        "canonical_labels": [1, 2],
    }
    duplicate_cluster = [
        {"start": 0.0, "end": 0.5, "speaker": "z"},
        {"start": 0.5, "end": 1.0, "speaker": "z"},
    ]
    assert experiment.quality_gate(
        expected_cluster_count=2,
        observed_cluster_count=2,
        sidecar_words=sidecar,
        output_words=output,
        intervals=duplicate_cluster,
        audio_duration_seconds=1.0,
        permitted_padding_seconds=0.0,
    )["reason"] == "cluster-count-mismatch"


def test_diagnostics_only_add_safe_listening_locations_and_hearing_needs_user():
    diagnostics = experiment.diagnostic_listening_locations(
        baseline={
            "purity": 0.8,
            "unmatched_words": 2,
            "channel_stability": 0.9,
            "short_turns": 4,
            "overlap": 0.1,
            "residual_clusters": 0,
        },
        candidate={
            "purity": 0.7,
            "unmatched_words": 3,
            "channel_stability": 0.8,
            "short_turns": 5,
            "overlap": 0.2,
            "residual_clusters": 1,
        },
        location_sha256=["9" * 64],
    )
    assert diagnostics == {"mandatory_passed": True, "location_sha256": ["9" * 64]}
    assert experiment.hearing_gate({}) == {
        "state": "needs-user",
        "reason": "hearing-decisions-required",
    }
    categories = {category: "pass" for category in experiment.HEARING_CATEGORIES}
    assert experiment.hearing_gate(categories) == {"state": "go", "reason": None}
    categories[experiment.HEARING_CATEGORIES[2]] = "fail"
    assert experiment.hearing_gate(categories) == {
        "state": "stop",
        "reason": "hearing-gate-failed",
    }


def test_blinded_review_material_is_seeded_categorized_and_procedural():
    diagnostic = {
        category: [hashlib.sha256(f"diagnostic-{category}".encode()).hexdigest()]
        for category in experiment.HEARING_CATEGORIES
    }
    matching = {
        category: [
            hashlib.sha256(f"match-a-{category}".encode()).hexdigest(),
            hashlib.sha256(f"match-b-{category}".encode()).hexdigest(),
        ]
        for category in experiment.HEARING_CATEGORIES
    }

    blinded, reveal = experiment.build_blinded_review_material(
        diagnostic_locations=diagnostic,
        matching_locations=matching,
        private_seed_sha256="a" * 64,
        matching_sample_size=1,
    )
    repeated, repeated_reveal = experiment.build_blinded_review_material(
        diagnostic_locations=diagnostic,
        matching_locations=matching,
        private_seed_sha256="a" * 64,
        matching_sample_size=1,
    )

    assert blinded == repeated
    assert reveal == repeated_reveal
    assert set(blinded["categories"]) == set(experiment.HEARING_CATEGORIES)
    assert all(len(items) == 2 for items in blinded["categories"].values())
    assert set(reveal["pair_order"].values()) <= {
        ("W0-FRESH", "SORTFORMER-V2-Q8"),
        ("SORTFORMER-V2-Q8", "W0-FRESH"),
    }
    serialized = experiment.serialize_public(blinded)
    assert "W0-FRESH" not in serialized
    assert "SORTFORMER-V2-Q8" not in serialized
    assert blinded["reveal_sha256"] == experiment.canonical_sha256(reveal)
    assert "процедурного ослепления" in (
        experiment.build_blinded_review_material.__doc__ or ""
    )
    assert "может восстановить порядок" in (
        experiment.build_blinded_review_material.__doc__ or ""
    )


def test_memory_gate_uses_frozen_plateau_and_linear_growth_policy_boundaries():
    mib = 1024**2
    passing = experiment.memory_gate(
        stage="fragment",
        oom=False,
        swap_before_bytes=0,
        swap_after_bytes=0,
        peak_rss_bytes=300 * mib,
        rss_after_inputs=[100 * mib, 170 * mib, 234 * mib],
        machine_ram_bytes=16 * 1024**3,
    )
    assert passing == {
        "state": "pass",
        "reason": None,
        "plateau_policy": "fragment-only:last-delta<=max(64MiB,5%-of-max-post-input-rss);linear=all-deltas>same-tolerance",
    }
    assert experiment.memory_gate(
        stage="fragment",
        oom=False,
        swap_before_bytes=0,
        swap_after_bytes=0,
        peak_rss_bytes=300 * mib,
        rss_after_inputs=[100 * mib, 170 * mib, 235 * mib],
        machine_ram_bytes=16 * 1024**3,
    )["reason"] == "rss-linear-growth"
    assert experiment.memory_gate(
        stage="fragment",
        oom=False,
        swap_before_bytes=0,
        swap_after_bytes=1,
        peak_rss_bytes=300 * mib,
        rss_after_inputs=[100 * mib, 120 * mib, 121 * mib],
        machine_ram_bytes=16 * 1024**3,
    )["reason"] == "active-swap"
    assert experiment.memory_gate(
        stage="fragment",
        oom=True,
        swap_before_bytes=0,
        swap_after_bytes=0,
        peak_rss_bytes=0,
        rss_after_inputs=[0, 0, 0],
        machine_ram_bytes=16 * 1024**3,
    )["reason"] == "oom"


def test_memory_gate_marks_two_gibibyte_review_as_needs_user():
    result = experiment.memory_gate(
        stage="fragment",
        oom=False,
        swap_before_bytes=0,
        swap_after_bytes=0,
        peak_rss_bytes=2 * 1024**3 + 1,
        rss_after_inputs=[2 * 1024**3, 2 * 1024**3, 2 * 1024**3],
        machine_ram_bytes=16 * 1024**3,
    )
    assert result["state"] == "needs-user"
    assert result["reason"] == "peak-rss-review"

    reserved_ram = experiment.memory_gate(
        stage="fragment",
        oom=False,
        swap_before_bytes=0,
        swap_after_bytes=0,
        peak_rss_bytes=3 * 1024**3,
        rss_after_inputs=[1024**3, 1024**3, 1024**3],
        machine_ram_bytes=16_601_477_120,
    )
    assert reserved_ram["state"] == "needs-user"


def test_memory_gate_skips_unsupported_plateau_in_single_full_recording_cell():
    result = experiment.memory_gate(
        stage="full",
        oom=False,
        swap_before_bytes=0,
        swap_after_bytes=0,
        peak_rss_bytes=1024**3,
        rss_after_inputs=[900 * 1024**2],
        machine_ram_bytes=16 * 1024**3,
    )
    assert result == {
        "state": "pass",
        "reason": None,
        "plateau_policy": "fragment-only:last-delta<=max(64MiB,5%-of-max-post-input-rss);linear=all-deltas>same-tolerance",
    }

    gib = 1024**3
    divergent = experiment.memory_gate(
        stage="fragment",
        oom=False,
        swap_before_bytes=0,
        swap_after_bytes=0,
        peak_rss_bytes=4 * gib,
        rss_after_inputs=[gib, int(1.1 * gib), int(1.18 * gib)],
        machine_ram_bytes=32 * gib,
    )
    assert divergent["reason"] == "rss-linear-growth"


def _sortformer_gate(**changes):
    recipe = _recipes()["SORTFORMER-V2-Q8"]
    values = {
        "effective_recipe": recipe,
        "probabilities": [[0.0, 0.25, 0.5, 1.0], [0.1, 0.2, 0.3, 0.4]],
        "labels": [1, 4],
        "frame_count": 2,
        "frame_probs_start": 0,
        "segments": [
            {"start": 0.0, "end": 0.08, "speaker": 1},
            {"start": 0.08, "end": 0.16, "speaker": 4},
        ],
        "channel_interval_counts": {1: 1, 2: 0, 3: 0, 4: 1},
        "channel_durations_seconds": {1: 0.08, 2: 0.0, 3: 0.0, 4: 0.08},
        "inference_wall_seconds": 1.0,
        "post_processing_wall_seconds": 0.1,
    }
    values.update(changes)
    return experiment.sortformer_gate(**values)


def test_sortformer_gate_checks_effective_config_shape_values_labels_and_metrics():
    assert _sortformer_gate() == {"passed": True, "reason": None}
    assert _sortformer_gate(probabilities=[[0.1, 0.2, 0.3]])["reason"] == (
        "probability-shape"
    )
    assert _sortformer_gate(probabilities=[[0.1, 0.2, float("nan"), 0.4]])[
        "reason"
    ] == "probability-value"
    assert _sortformer_gate(labels=[0, 4])["reason"] == "speaker-label"
    changed = _recipes()["SORTFORMER-V2-Q8"]
    changed["seconds_per_frame"] = 0.1
    assert _sortformer_gate(effective_recipe=changed)["reason"] == (
        "effective-config-mismatch"
    )
    assert _sortformer_gate(inference_wall_seconds=-1)["reason"] == (
        "invalid-stage-wall"
    )
    assert _sortformer_gate(
        frame_count=5,
        frame_probs_start=3,
    ) == {"passed": True, "reason": None}
    assert _sortformer_gate(channel_interval_counts={1: 2, 2: 0, 3: 0, 4: 0})[
        "reason"
    ] == "channel-counts"


class _Measurement:
    def __init__(self, trace):
        self.trace = trace
        self.walls = {
            "w0-process": 10.0,
            "candidate-inference": 5.5,
            "candidate-post-processing": 0.5,
        }

    def measure(self, label, operation):
        self.trace.append(f"timer-start:{label}")
        value = operation()
        self.trace.append(f"timer-stop:{label}")
        return value, {
            "wall_seconds": self.walls[label],
            "user_cpu_seconds": 1.0,
            "system_cpu_seconds": 0.1,
            "peak_rss_bytes": 100,
        }


def test_measurement_boundary_times_only_exact_w0_and_candidate_operations():
    trace = []
    samples = np.asarray([0.0, 0.25], dtype=np.float32)

    class W0:
        def process(self, actual):
            assert actual is samples
            trace.append("w0-process")
            return "w0-result"

    class Runtime:
        def prepare_samples(self, actual):
            assert actual is samples
            trace.append("prepare-samples")
            return "prepared-buffer"

        def stream_open(self):
            trace.append("stream-open")
            return "stream"

        def stream_push(self, stream, actual):
            assert stream == "stream" and actual == "prepared-buffer"
            trace.append("stream-push")

        def stream_finish(self, stream):
            assert stream == "stream"
            trace.append("stream-finish")
            return "raw-segments"

        def stream_close(self, stream):
            assert stream == "stream"
            trace.append("stream-close")

    measurement = _Measurement(trace)
    w0 = experiment.measure_w0(W0(), samples, measurement)
    candidate = experiment.measure_candidate(
        Runtime(),
        samples,
        lambda raw: trace.append("post-processing") or [raw],
        measurement,
    )

    assert w0["result"] == "w0-result"
    assert w0["metrics"]["wall_seconds"] == 10.0
    assert candidate["result"] == ["raw-segments"]
    assert candidate["metrics"]["wall_seconds"] == 6.0
    assert candidate["inference_wall_seconds"] == 5.5
    assert candidate["post_processing_wall_seconds"] == 0.5
    assert trace == [
        "timer-start:w0-process",
        "w0-process",
        "timer-stop:w0-process",
        "prepare-samples",
        "timer-start:candidate-inference",
        "stream-open",
        "stream-push",
        "stream-finish",
        "timer-stop:candidate-inference",
        "timer-start:candidate-post-processing",
        "post-processing",
        "stream-close",
        "timer-stop:candidate-post-processing",
    ]


def _guard(recipe="W0-FRESH", **changes):
    result = {
        "machine_sha256": "1" * 64,
        "affinity": list(range(8)),
        "logical_cores": 16,
        "physical_cores": 8,
        "cpuid_isa_sha256": "2" * 64,
        "ram_bytes": 16 * 1024**3,
        "os_sha256": "3" * 64,
        "compiler_sha256": "4" * 64,
        "runtime_versions_sha256": "5" * 64,
        "recipe": recipe,
        "effective_threads": 8 if recipe == "W0-FRESH" else 4,
        "power_mode": "balanced",
        "governor": "powersave",
        "turbo": True,
        "throttling": False,
        "throttle_count": 10,
        "swap_used_bytes": 0,
        "swap_sin_bytes": 0,
        "swap_sout_bytes": 0,
    }
    result.update(changes)
    return result


def test_pair_guards_require_same_machine_affinity_and_frozen_thread_counts():
    assert experiment.validate_pair_guards(
        _guard(),
        _guard(),
        _guard("SORTFORMER-V2-Q8"),
        _guard("SORTFORMER-V2-Q8"),
        expected_machine=_machine(),
    ) == []
    affinity_reasons = experiment.validate_pair_guards(
        _guard(),
        _guard(),
        _guard("SORTFORMER-V2-Q8", affinity=list(range(1, 9))),
        _guard("SORTFORMER-V2-Q8", affinity=list(range(1, 9))),
        expected_machine=_machine(),
    )
    assert set(affinity_reasons) == {
        "manifest-machine-mismatch",
        "pair-affinity-mismatch",
    }
    assert "effective-threads-mismatch" in experiment.validate_pair_guards(
        _guard(effective_threads=4),
        _guard(effective_threads=4),
        _guard("SORTFORMER-V2-Q8"),
        _guard("SORTFORMER-V2-Q8"),
        expected_machine=_machine(),
    )
    assert "pair-power-mismatch" in experiment.validate_pair_guards(
        _guard(),
        _guard(),
        _guard("SORTFORMER-V2-Q8", governor="performance"),
        _guard("SORTFORMER-V2-Q8", governor="performance"),
        expected_machine=_machine(),
    )
    wrong = _guard(affinity=list(range(8, 16)))
    assert "manifest-machine-mismatch" in experiment.validate_pair_guards(
        wrong,
        wrong,
        _guard("SORTFORMER-V2-Q8", affinity=list(range(8, 16))),
        _guard("SORTFORMER-V2-Q8", affinity=list(range(8, 16))),
        expected_machine=_machine(),
    )


def test_guarded_cell_excludes_invalid_wall_and_retries_once_after_stabilization():
    snapshots = iter(
        [
            _guard(),
            _guard(power_mode="performance"),
            _guard(),
            _guard(),
        ]
    )
    walls = iter([100.0, 50.0])
    stabilizations = []

    result = experiment.run_guarded_cell(
        {"semantic_id": "safe-cell", "recipe": "W0-FRESH"},
        runner=lambda _request: {"wall_seconds": next(walls)},
        guard_reader=lambda: next(snapshots),
        stabilize=lambda: stabilizations.append("stabilized"),
        expected_machine=_machine(),
    )

    assert result["state"] == "accepted"
    assert result["result"] == {"wall_seconds": 50.0}
    assert stabilizations == ["stabilized"]
    assert result["attempts"][0]["valid"] is False
    assert "result" not in result["attempts"][0]
    assert result["attempts"][1]["valid"] is True


def test_guarded_cell_accepts_stable_manifest_declared_performance_mode():
    machine = _machine()
    machine["power_mode"] = "performance"
    result = experiment.run_guarded_cell(
        {"semantic_id": "safe-cell", "recipe": "W0-FRESH"},
        runner=lambda _request: {"wall_seconds": 1.0},
        guard_reader=lambda: _guard(power_mode="performance"),
        stabilize=lambda: pytest.fail("stable mode must not retry"),
        expected_machine=machine,
    )
    assert result["state"] == "accepted"
    assert len(result["attempts"]) == 1


def test_second_guard_invalidation_stops_with_first_reason_and_active_swap_never_retries():
    throttled = iter(
        [
            _guard(),
            _guard(throttling=True, throttle_count=11),
            _guard(),
            _guard(throttling=True, throttle_count=11),
        ]
    )
    second = experiment.run_guarded_cell(
        {"semantic_id": "safe-cell", "recipe": "W0-FRESH"},
        runner=lambda _request: {"wall_seconds": 99.0},
        guard_reader=lambda: next(throttled),
        stabilize=lambda: None,
        expected_machine=_machine(),
    )
    assert second["state"] == "stop"
    assert second["reason"] == "new-throttling"
    assert len(second["attempts"]) == 2

    calls = []
    swapped = experiment.run_guarded_cell(
        {"semantic_id": "safe-cell", "recipe": "W0-FRESH"},
        runner=lambda _request: calls.append("run") or {"wall_seconds": 1.0},
        guard_reader=lambda: _guard(swap_used_bytes=1),
        stabilize=lambda: calls.append("stabilize"),
        expected_machine=_machine(),
    )
    assert swapped["state"] == "stop"
    assert swapped["reason"] == "active-swap"
    assert calls == []


def test_oom_stops_without_retry_and_private_result_is_written_atomically(tmp_path):
    calls = []
    result = experiment.run_guarded_cell(
        {"semantic_id": "safe-cell", "recipe": "W0-FRESH"},
        runner=lambda _request: (_ for _ in ()).throw(MemoryError("private")),
        guard_reader=lambda: _guard(),
        stabilize=lambda: calls.append("stabilize"),
        expected_machine=_machine(),
    )
    assert result["state"] == "stop"
    assert result["reason"] == "oom"
    assert calls == []
    assert "private" not in json.dumps(result)

    private_path = tmp_path / "closed" / "raw-result.json"
    digest = experiment.persist_private_result(
        private_path, {"raw_rows": [1, 2, 3], "secret": "allowed-here"}
    )
    assert digest == experiment.file_sha256(private_path)
    assert sorted(path.name for path in private_path.parent.iterdir()) == [
        "raw-result.json"
    ]
    failed_path = tmp_path / "failed" / "raw-result.json"
    with pytest.raises(ValueError):
        experiment.persist_private_result(
            failed_path, {"probability": float("nan"), "words": ["private"]}
        )
    assert list(failed_path.parent.iterdir()) == []


def test_guarded_cell_does_not_retry_hard_mismatch_and_sanitizes_exceptions():
    snapshots = iter([_guard(), _guard(governor="performance")])
    stabilizations = []
    hard = experiment.run_guarded_cell(
        {"semantic_id": "safe-cell", "recipe": "W0-FRESH"},
        runner=lambda _request: {"wall_seconds": 1.0},
        guard_reader=lambda: next(snapshots),
        stabilize=lambda: stabilizations.append("unexpected"),
        expected_machine=_machine(),
    )
    assert hard["state"] == "stop"
    assert hard["reason"] == "runtime-power-setting-changed"
    assert len(hard["attempts"]) == 1
    assert stabilizations == []

    failed = experiment.run_guarded_cell(
        {"semantic_id": "safe-cell", "recipe": "W0-FRESH"},
        runner=lambda _request: (_ for _ in ()).throw(RuntimeError("/private/path")),
        guard_reader=lambda: _guard(),
        stabilize=lambda: None,
        expected_machine=_machine(),
    )
    assert failed["state"] == "stop"
    assert failed["reason"] == "runtime-error"
    assert "/private/path" not in json.dumps(failed)

    unreadable = experiment.run_guarded_cell(
        {"semantic_id": "safe-cell", "recipe": "W0-FRESH"},
        runner=lambda _request: {"wall_seconds": 1.0},
        guard_reader=lambda: (_ for _ in ()).throw(OSError("/private/guard")),
        stabilize=lambda: None,
        expected_machine=_machine(),
    )
    assert unreadable["state"] == "stop"
    assert unreadable["reason"] == "runtime-guard-unavailable"
    assert "/private/guard" not in json.dumps(unreadable)

    calls = []
    wrong_recipe = experiment.run_guarded_cell(
        {"semantic_id": "safe-cell", "recipe": "SORTFORMER-V2-Q8"},
        runner=lambda _request: calls.append("run") or {"wall_seconds": 1.0},
        guard_reader=lambda: _guard("W0-FRESH"),
        stabilize=lambda: None,
        expected_machine=_machine(),
    )
    assert wrong_recipe["reason"] == "runtime-recipe-mismatch"
    assert calls == []


def test_guarded_cell_hashes_result_rejected_by_after_run_swap():
    snapshots = iter([_guard(), _guard(swap_sin_bytes=1)])
    raw_result = {"wall_seconds": 1.0, "private": "not-published"}
    result = experiment.run_guarded_cell(
        {"semantic_id": "safe-cell", "recipe": "W0-FRESH"},
        runner=lambda _request: raw_result,
        guard_reader=lambda: next(snapshots),
        stabilize=lambda: None,
        expected_machine=_machine(),
    )
    assert result["state"] == "stop"
    assert result["reason"] == "active-swap"
    assert result["attempts"][0]["invalid_result_sha256"] == (
        experiment.canonical_sha256(raw_result)
    )
    assert "result" not in result["attempts"][0]


def test_non_finite_private_result_becomes_sanitized_first_stop(tmp_path):
    guarded = experiment.run_guarded_cell(
        {"semantic_id": "safe-cell", "recipe": "W0-FRESH"},
        runner=lambda _request: {"wall_seconds": float("nan")},
        guard_reader=lambda: _guard(),
        stabilize=lambda: None,
        expected_machine=_machine(),
    )
    assert guarded["state"] == "stop"
    assert guarded["reason"] == "invalid-result-value"
    assert "nan" not in json.dumps(guarded)

    store = experiment.ExperimentState(tmp_path / "public-state.json")
    state = store.stop("mandatory-gate-failed", {"probability": float("nan")})
    assert state["first_reason"] == "invalid-result-value"
    assert experiment.SHA256_RE.fullmatch(state["result_sha256"])


def test_public_plan_and_result_are_redacted_and_privacy_scanned(tmp_path):
    manifest = _manifest(tmp_path)
    outcome = experiment.validate_manifest(
        manifest, root=tmp_path, environment=_environment()
    )
    plan = experiment.safe_public_plan(manifest, outcome)
    serialized = experiment.serialize_public(plan)

    assert plan["state"] == "valid"
    assert plan["recipes"] == ["W0-FRESH", "SORTFORMER-V2-Q8"]
    assert plan["fragment_count"] == 3
    assert plan["scheduled_cell_count"] == 12
    assert str(tmp_path) not in serialized
    assert "fragment-0" not in serialized
    assert "words-fragment" not in serialized
    experiment.privacy_scan(plan)

    report = experiment.safe_public_result(
        source_commit="a" * 40,
        result_sha256="b" * 64,
        aggregate_timings={"w0_wall_seconds": 100.0, "candidate_wall_seconds": 60.0},
        deltas={"speedup_percent": 40.0},
        statistics_values={
            "median_speedup_percent": 40.0,
            "mad_percentage_points": 1.0,
        },
        gate_verdicts={"quality": "pass", "hearing": "needs-user"},
        limitations=["four-speaker-output-limit", "non-english-domain-risk"],
        first_stop_reason=None,
    )
    experiment.privacy_scan(report)
    assert report["state"] == "needs-user"
    assert report["limitations"][-1] == "procedural-blinding"
    with pytest.raises(experiment.PrivacyError):
        experiment.privacy_scan({"intervals": [{"start": 1.0}]})
    with pytest.raises(experiment.PrivacyError):
        experiment.privacy_scan({"note": str(tmp_path / "secret.wav")})
    with pytest.raises(experiment.PrivacyError):
        experiment.privacy_scan({"source_path": "relative-but-private.wav"})
    with pytest.raises(experiment.PrivacyError):
        experiment.privacy_scan({"note": "/opt/private/model.bin"})
    with pytest.raises(experiment.PrivacyError):
        experiment.privacy_scan({"note": "failure at /opt/private/model.bin"})
    with pytest.raises(experiment.PreflightError, match="public-numeric-value"):
        experiment.safe_public_result(
            source_commit="a" * 40,
            result_sha256="b" * 64,
            aggregate_timings={"note": "private transcript"},
            deltas={},
            statistics_values={},
            gate_verdicts={},
            limitations=[],
            first_stop_reason=None,
        )


def test_public_result_requires_all_gates_and_final_speed_threshold_for_go():
    common = {
        "source_commit": "a" * 40,
        "result_sha256": "b" * 64,
        "aggregate_timings": {"candidate_wall_seconds": 60.0},
        "deltas": {"speedup_percent": 40.0},
        "limitations": ["four-speaker-output-limit"],
        "first_stop_reason": None,
    }
    automatic = {
        "quality": "pass",
        "sortformer": "pass",
        "memory": "pass",
        "measurement": "pass",
        "hearing": "pass",
    }
    below = experiment.safe_public_result(
        **common,
        statistics_values={"median_speedup_percent": 29.999},
        gate_verdicts=automatic,
    )
    assert below["state"] == "stop"
    assert below["first_stop_reason"] == "below-final-threshold"

    incomplete = experiment.safe_public_result(
        **common,
        statistics_values={"median_speedup_percent": 40.0},
        gate_verdicts={**automatic, "quality": "needs-user"},
    )
    assert incomplete["state"] == "needs-user"
    go = experiment.safe_public_result(
        **common,
        statistics_values={"median_speedup_percent": 30.0},
        gate_verdicts=automatic,
    )
    assert go["state"] == "go"
    assert go["limitations"] == [
        "four-speaker-output-limit",
        "procedural-blinding",
    ]

    deduplicated = experiment.safe_public_result(
        **{**common, "limitations": ["procedural-blinding"]},
        statistics_values={"median_speedup_percent": 30.0},
        gate_verdicts=automatic,
    )
    assert deduplicated["limitations"] == ["procedural-blinding"]


def test_validate_only_cli_is_useful_redacted_and_never_calls_loader(
    tmp_path, capsys
):
    manifest = _manifest(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    calls = []
    exit_code = experiment.run_cli(
        SimpleNamespace(manifest=manifest_path, validate_only=True),
        environment_probe=_environment,
        w0_loader=lambda: calls.append("w0"),
        candidate_loader=lambda: calls.append("candidate"),
    )
    output = capsys.readouterr().out
    assert exit_code == 0
    assert calls == []
    assert '"schema_status": "pass"' in output
    assert '"pin_status": "pass"' in output
    assert '"schedule_status": "pass"' in output
    assert str(tmp_path) not in output

    manifest["fresh_basis"]["approval"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert experiment.run_cli(
        SimpleNamespace(manifest=manifest_path, validate_only=True),
        environment_probe=_environment,
        w0_loader=lambda: calls.append("w0"),
        candidate_loader=lambda: calls.append("candidate"),
    ) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "needs-user"


def test_invalid_cli_output_has_one_trailing_newline(tmp_path, capsys):
    manifest_path = tmp_path / "invalid.json"
    manifest_path.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        experiment.main(["--manifest", str(manifest_path), "--validate-only"])
    assert raised.value.code == 2
    output = capsys.readouterr().out
    assert output.endswith("\n")
    assert not output.endswith("\n\n")


def test_child_process_request_result_seam_uses_atomic_private_files(tmp_path):
    request = {
        "semantic_id": "cell-safe",
        "recipe": "W0-FRESH",
        "input_sha256": ["1" * 64, "2" * 64, "3" * 64],
        "sidecar_sha256": ["4" * 64, "5" * 64, "6" * 64],
    }
    commands = []

    def fake_process(command):
        commands.append(command)
        request_path = Path(command[-1])
        payload = json.loads(request_path.read_text())
        response = {
            "worker_schema": experiment.WORKER_SCHEMA,
            "semantic_id": payload["semantic_id"],
            "status": "complete",
            "result": {"wall_seconds": 1.0},
            "result_sha256": experiment.canonical_sha256({"wall_seconds": 1.0}),
        }
        Path(payload["response_path"]).write_text(json.dumps(response))
        return SimpleNamespace(returncode=0, stdout="private log", stderr="")

    result = experiment.run_child_process(
        request, work_dir=tmp_path, process_runner=fake_process
    )

    assert result["result"] == {"wall_seconds": 1.0}
    assert len(commands) == 1
    assert (tmp_path / "logs" / "cell-safe.stdout.log").read_text() == "private log"

    stale_runner = lambda _command: SimpleNamespace(
        returncode=0, stdout="", stderr=""
    )
    with pytest.raises(RuntimeError, match="child-process-failed"):
        experiment.run_child_process(
            request, work_dir=tmp_path, process_runner=stale_runner
        )


def test_child_process_requires_an_explicit_worker_runner(tmp_path):
    request = {
        "semantic_id": "cell-safe",
        "recipe": "W0-FRESH",
        "input_sha256": ["1" * 64],
        "sidecar_sha256": ["2" * 64],
    }
    with pytest.raises(experiment.PreflightError, match="worker-runner-required"):
        experiment.run_child_process(request, work_dir=tmp_path)


class _FakeFunction:
    def __init__(self, function):
        self.function = function
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.function(*args)


class _FakeCAbi:
    def __init__(
        self,
        *,
        speakers=4,
        seconds_per_frame=0.08,
        create_status=0,
        grown_segment_count=False,
    ):
        self.speakers = speakers
        self.seconds = seconds_per_frame
        self.create_status = create_status
        self.grown_segment_count = grown_segment_count
        self.seen = None
        self.pushed = None
        self.closed = False
        for name in dir(self):
            if name.startswith("nemo_speech_"):
                setattr(self, name, _FakeFunction(getattr(self, name)))

    def nemo_speech_diar_create(self, config, output):
        self.seen = config.contents
        output._obj.value = 101
        return self.create_status

    def nemo_speech_diar_destroy(self, _model):
        return None

    def nemo_speech_diar_num_speakers(self, _model):
        return self.speakers

    def nemo_speech_diar_seconds_per_frame(self, _model):
        return self.seconds

    def nemo_speech_diar_stream_open(self, _model, output):
        self.closed = False
        output._obj.value = 202
        return 0

    def nemo_speech_diar_stream_push_f32(self, _stream, samples, count, sample_rate):
        self.pushed = ([samples[index] for index in range(count)], sample_rate)
        return 0

    def nemo_speech_diar_stream_finish(self, _stream):
        return 0

    def nemo_speech_diar_stream_close(self, _stream):
        self.closed = True

    def nemo_speech_diar_frame_count(self, _stream):
        assert not self.closed
        return 2

    def nemo_speech_diar_frame_probs_start(self, _stream):
        assert not self.closed
        return 0

    def nemo_speech_diar_frame_probs(self, _stream, output, capacity):
        assert not self.closed
        assert capacity == 8
        for index, value in enumerate((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)):
            output[index] = value
        return 0

    def nemo_speech_diar_segments(self, _stream, config, output, capacity, count):
        assert not self.closed
        assert config.contents.onset == pytest.approx(0.641)
        if not output:
            count._obj.value = 2
            return 0
        assert capacity == 2
        if self.grown_segment_count:
            count._obj.value = 3
            return 0
        output[0].start_time = 0.0
        output[0].end_time = 0.8
        output[0].speaker = 1
        output[1].start_time = 0.8
        output[1].end_time = 1.6
        output[1].speaker = 4
        return 0


def test_fake_c_abi_validates_layout_cpu_selection_and_effective_config():
    library = _FakeCAbi()
    adapter = experiment.SortformerCAbi(library)

    result = adapter.synthetic_check()
    model = adapter.create_model("synthetic-model.gguf")
    with pytest.raises(experiment.CAbiError, match="model-already-created"):
        adapter.create_model("second-model.gguf")
    stream = adapter.stream_open(model)
    samples = np.asarray([0.0, 0.25], dtype=np.float32)
    prepared = adapter.prepare_samples(samples)
    assert prepared.owner is samples
    assert ctypes.addressof(prepared.pointer.contents) == samples.ctypes.data
    adapter.stream_push(stream, prepared, sample_rate=16_000)
    adapter.stream_finish(stream)
    output = adapter.collect_result(stream, model)
    adapter.stream_close(stream)
    adapter.destroy_model(model)

    assert result == {
        "model_config_size": 56,
        "segmentation_config_size": 48,
        "segment_size": 24,
        "gpu": -1,
        "num_speakers": 4,
        "seconds_per_frame": pytest.approx(0.08),
        "chunk_frames": 340,
        "right_context_frames": 40,
    }
    assert library.seen.gpu == -1
    assert library.seen.preset == b"streaming"
    assert library.seen.update_period_frames == 300
    assert library.pushed[0] == pytest.approx([0.0, 0.25])
    assert library.pushed[1] == 16_000
    assert output["frame_count"] == 2
    assert output["frame_probs_start"] == 0
    assert output["probabilities"][1] == pytest.approx([0.5, 0.6, 0.7, 0.8])
    assert output["segments"] == [
        {"start": 0.0, "end": 0.8, "speaker": 1},
        {"start": 0.8, "end": 1.6, "speaker": 4},
    ]


def test_fake_c_abi_rejects_layout_and_runtime_config_mismatches():
    with pytest.raises(experiment.CAbiError, match="layout"):
        experiment.SortformerCAbi(_FakeCAbi(), expected_layout={"size": 999})
    with pytest.raises(experiment.CAbiError, match="num-speakers"):
        experiment.SortformerCAbi(_FakeCAbi(speakers=3)).synthetic_check()
    with pytest.raises(experiment.CAbiError, match="seconds-per-frame"):
        experiment.SortformerCAbi(_FakeCAbi(seconds_per_frame=0.1)).synthetic_check()
    with pytest.raises(experiment.CAbiError, match="effective-config"):
        experiment.SortformerCAbi(_FakeCAbi(create_status=1)).synthetic_check()


def test_fake_c_abi_load_boundary_configures_and_returns_adapter():
    calls = []
    library = _FakeCAbi()
    adapter = experiment.SortformerCAbi.load(
        Path("closed/libnemo_speech.so"),
        loader=lambda path: calls.append(path) or library,
    )
    assert calls == ["closed/libnemo_speech.so"]
    assert library.nemo_speech_diar_seconds_per_frame.restype is ctypes.c_double
    assert library.nemo_speech_diar_frame_count.restype is ctypes.c_int64
    assert adapter.synthetic_check()["num_speakers"] == 4


def test_fake_c_abi_rejects_segment_count_growth_beyond_owned_buffer():
    adapter = experiment.SortformerCAbi(_FakeCAbi(grown_segment_count=True))
    model = adapter.create_model("synthetic-model.gguf")
    stream = adapter.stream_open(model)
    with pytest.raises(experiment.CAbiError, match="segment-count-changed"):
        adapter.collect_result(stream, model)


def test_candidate_measurement_collects_c_abi_result_before_stream_close():
    trace = []
    library = _FakeCAbi()
    adapter = experiment.SortformerCAbi(library)
    model = adapter.create_model("synthetic-model.gguf")

    measured = experiment.measure_candidate(
        adapter,
        np.asarray([0.0, 0.25], dtype=np.float32),
        lambda stream: adapter.collect_result(stream, model),
        _Measurement(trace),
    )

    assert measured["result"]["frame_count"] == 2
    assert library.closed is True
