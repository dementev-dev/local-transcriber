import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "benchmarks" / "diarization_calibration.py"
)
SPEC = importlib.util.spec_from_file_location("diarization_calibration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
calibration = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = calibration
SPEC.loader.exec_module(calibration)


def _write(path, content):
    path.write_bytes(content)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def _manifest(tmp_path):
    fp32 = _write(tmp_path / "fp32.onnx", b"fp32")
    int8 = _write(tmp_path / "int8.onnx", b"int8")
    wespeaker = _write(tmp_path / "wespeaker.onnx", b"wespeaker")
    titanet = _write(tmp_path / "titanet.onnx", b"titanet")
    segmentations = [
        {
            "id": "pyannote-fp32",
            "precision": "fp32",
            "source_url": "https://example/fp32",
            "license": "MIT",
            **fp32,
        },
        {
            "id": "pyannote-int8",
            "precision": "int8",
            "source_url": "https://example/int8",
            "license": "MIT",
            **int8,
        },
    ]
    embeddings = [
        {
            "id": "wespeaker",
            "source_url": "https://example/wespeaker",
            "license": "CC-BY-4.0",
            **wespeaker,
        },
        {
            "id": "titanet-small",
            "source_url": "https://example/titanet",
            "license": "Apache-2.0",
            **titanet,
        },
    ]
    recordings = []
    for index, (recording_id, speakers) in enumerate(
        (("data-test", 3), ("t2-bdma", 2), ("yantar", 2))
    ):
        media = _write(tmp_path / f"{recording_id}.mp4", recording_id.encode())
        words_content = json.dumps(
            [{"start": 0.0, "end": 0.5, "text": " Слово"}]
        ).encode()
        words = _write(tmp_path / f"{recording_id}.words.json", words_content)
        reference = None
        if index:
            reference_content = b"**[00:00] Speaker 1:**\n"
            ref = _write(tmp_path / f"{recording_id}.md", reference_content)
            reference = {
                "path": ref["path"],
                "sha256": ref["sha256"],
                "format": "hypescribe_markdown",
            }
        recordings.append(
            {
                "id": recording_id,
                "path": media["path"],
                "sha256": media["sha256"],
                "start": 0.0,
                "duration": 10.0,
                "expected_speakers": speakers,
                "reference": reference,
                "asr_words": {
                    "path": words["path"],
                    "sha256": words["sha256"],
                    "format": "local_transcriber_words_v1",
                },
            }
        )
    return {
        "schema_version": 2,
        "segmentation_models": segmentations,
        "embedding_models": embeddings,
        "combinations": [
            {
                "id": "fp32-wespeaker",
                "segmentation_id": "pyannote-fp32",
                "embedding_id": "wespeaker",
                "role": "baseline",
                "thresholds": [0.8, 0.9],
            },
            {
                "id": "fp32-titanet",
                "segmentation_id": "pyannote-fp32",
                "embedding_id": "titanet-small",
                "role": "candidate",
                "thresholds": [0.8],
            },
            {
                "id": "int8-wespeaker",
                "segmentation_id": "pyannote-int8",
                "embedding_id": "wespeaker",
                "role": "candidate",
                "thresholds": [0.8],
            },
            {
                "id": "int8-titanet",
                "segmentation_id": "pyannote-int8",
                "embedding_id": "titanet-small",
                "role": "candidate",
                "thresholds": [0.8],
            },
        ],
        "recordings": recordings,
        "threads": 4,
        "rss_sample_interval_ms": 5,
    }


def _environment():
    return {
        "sherpa_onnx_version": "1.13.6",
        "onnxruntime_version": "1",
        "numpy_version": "2",
        "psutil_version": "7",
    }


def test_validate_only_requires_only_manifest():
    args = calibration.parse_args(["--manifest", "manifest.json", "--validate-only"])

    assert args.manifest == Path("manifest.json")
    assert args.output is None
    assert args.work_dir is None


@pytest.mark.parametrize(
    "extra",
    [
        ["--output", "unused"],
        ["--work-dir", "unused"],
        ["--prepare-asr", "data-test"],
        ["--asr-model", "model"],
        ["--asr-device", "cpu"],
        ["--asr-compute-type", "float32"],
        ["--asr-language", "ru"],
    ],
)
def test_validate_only_rejects_unused_run_options(extra):
    with pytest.raises(SystemExit):
        calibration.parse_args(
            ["--manifest", "manifest.json", "--validate-only", *extra]
        )


def test_relative_manifest_paths_resolve_from_manifest_directory(tmp_path):
    manifest = _manifest(tmp_path)
    for model in manifest["segmentation_models"] + manifest["embedding_models"]:
        model["path"] = Path(model["path"]).name
    for recording in manifest["recordings"]:
        recording["path"] = Path(recording["path"]).name
        recording["asr_words"]["path"] = Path(recording["asr_words"]["path"]).name
        if recording["reference"] is not None:
            recording["reference"]["path"] = Path(recording["reference"]["path"]).name

    resolved = calibration.resolve_manifest_paths(manifest, tmp_path)

    assert calibration.validate_manifest(resolved)["schema_version"] == 2
    assert Path(resolved["recordings"][0]["path"]).is_absolute()


def test_validate_only_prints_safe_plan(tmp_path, monkeypatch, capsys):
    manifest_path = tmp_path / "manifest.json"
    manifest = _manifest(tmp_path)
    for model in manifest["segmentation_models"] + manifest["embedding_models"]:
        model["path"] = Path(model["path"]).name
    for recording in manifest["recordings"]:
        recording["path"] = Path(recording["path"]).name
        recording["asr_words"]["path"] = Path(recording["asr_words"]["path"]).name
        if recording["reference"] is not None:
            recording["reference"]["path"] = Path(recording["reference"]["path"]).name
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(calibration, "validate_original_corpus", lambda manifest: None)
    monkeypatch.setattr(calibration, "environment_snapshot", _environment)

    calibration.main(["--manifest", str(manifest_path), "--validate-only"])

    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "valid"
    assert summary["automatic_cells"] == 15
    assert summary["known_cells_if_all_selected"] == 12
    assert str(tmp_path) not in json.dumps(summary)


@pytest.mark.parametrize(
    ("key", "value"),
    [("sherpa_onnx_version", "1.14.0"), ("psutil_version", "unavailable")],
)
def test_validate_environment_rejects_incompatible_dependencies(key, value):
    environment = _environment()
    environment[key] = value

    with pytest.raises(RuntimeError):
        calibration.validate_environment(environment)


def _complete_cell(spec, clusters, purity=None, residual=0.0):
    return {
        **spec,
        "status": "complete",
        "diagnostics": {
            "clusters": clusters,
            "mapped_speaker_purity": purity,
            "residual_share_after_expected": residual,
        },
    }


def test_manifest_validates_required_matrix_and_files(tmp_path):
    assert calibration.validate_manifest(_manifest(tmp_path))["schema_version"] == 2


@pytest.mark.parametrize(
    "mutation", ["missing_combination", "duplicate_id", "unsorted_threshold"]
)
def test_manifest_rejects_invalid_matrix_and_ids(tmp_path, mutation):
    manifest = _manifest(tmp_path)
    if mutation == "missing_combination":
        manifest["combinations"].pop()
    elif mutation == "duplicate_id":
        manifest["recordings"][1]["id"] = manifest["recordings"][0]["id"]
    else:
        manifest["combinations"][0]["thresholds"] = [0.9, 0.8]
    with pytest.raises(ValueError):
        calibration.validate_manifest(manifest, verify_files=False)


def test_manifest_rejects_hash_mismatch_before_work(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["recordings"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        calibration.validate_manifest(manifest)


@pytest.mark.parametrize("model_kind", ["segmentation", "embedding"])
def test_manifest_rejects_multiple_exact_model_ids_per_matrix_axis(
    tmp_path, model_kind
):
    manifest = _manifest(tmp_path)
    if model_kind == "segmentation":
        duplicate = dict(manifest["segmentation_models"][0])
        duplicate["id"] = "pyannote-fp32-other"
        manifest["segmentation_models"].append(duplicate)
        manifest["combinations"][1]["segmentation_id"] = duplicate["id"]
    else:
        duplicate = dict(manifest["embedding_models"][0])
        duplicate["id"] = "wespeaker-other"
        manifest["embedding_models"].append(duplicate)
        manifest["combinations"][2]["embedding_id"] = duplicate["id"]

    with pytest.raises(ValueError, match="ровно один"):
        calibration.validate_manifest(manifest, verify_files=False)


def test_complete_run_rejects_substitute_corpus(tmp_path):
    manifest = _manifest(tmp_path)
    with pytest.raises(ValueError, match="Корпус"):
        calibration.validate_original_corpus(manifest)
    for recording in manifest["recordings"]:
        recording.update(calibration.ORIGINAL_CORPUS[recording["id"]])
    calibration.validate_original_corpus(manifest)


def test_experiment_identity_ignores_paths_but_rejects_semantic_resume(tmp_path):
    manifest = _manifest(tmp_path)
    first = calibration.make_experiment_id(manifest, _environment())
    moved = json.loads(json.dumps(manifest))
    moved["recordings"][0]["path"] = "/different/private/path.mp4"
    assert calibration.make_experiment_id(moved, _environment()) == first
    output = tmp_path / "result.json"
    calibration.save_output(output, {"schema_version": 2, "experiment_id": "f" * 64})
    with pytest.raises(ValueError, match="другому семантическому"):
        calibration._load_or_create_output(output, manifest, _environment(), first)


def test_resume_rejects_different_measurement_host_without_path_identity(tmp_path):
    manifest = _manifest(tmp_path)
    first_environment = {
        **_environment(),
        "started_at_utc": "2026-08-29T00:00:00+00:00",
        "platform": "host-a",
        "cpu": "cpu-a",
        "logical_cpus": 16,
        "python_version": "3.13",
    }
    experiment_id = calibration.make_experiment_id(manifest, first_environment)
    output_path = tmp_path / "result.json"
    calibration.save_output(
        output_path,
        calibration._new_output(manifest, first_environment, experiment_id),
    )
    moved = json.loads(json.dumps(manifest))
    moved["recordings"][0]["path"] = "/moved/private/recording.mp4"
    assert calibration.make_experiment_id(moved, first_environment) == experiment_id
    second_environment = {
        **first_environment,
        "started_at_utc": "2026-08-30T00:00:00+00:00",
        "platform": "host-b",
        "cpu": "cpu-b",
        "logical_cpus": 2,
    }

    with pytest.raises(ValueError, match="другой platform/cpu/logical_cpus"):
        calibration._load_or_create_output(
            output_path, moved, second_environment, experiment_id
        )


def test_cell_expansion_is_rectangular_deterministic_and_path_independent(tmp_path):
    manifest = _manifest(tmp_path)
    experiment_id = calibration.make_experiment_id(manifest, _environment())
    cells = calibration.expand_automatic_cells(manifest, experiment_id)
    assert len(cells) == 15
    assert [
        (cell["combination_id"], cell["recording_id"], cell["threshold"])
        for cell in cells[:4]
    ] == [
        ("fp32-wespeaker", "data-test", 0.8),
        ("fp32-wespeaker", "t2-bdma", 0.8),
        ("fp32-wespeaker", "yantar", 0.8),
        ("fp32-wespeaker", "data-test", 0.9),
    ]
    assert all(cell["num_clusters"] == -1 for cell in cells)
    moved = json.loads(json.dumps(manifest))
    for model in moved["segmentation_models"] + moved["embedding_models"]:
        model["path"] = f"/moved/models/{model['id']}.onnx"
    for recording in moved["recordings"]:
        recording["path"] = f"/moved/media/{recording['id']}.mp4"
        recording["asr_words"]["path"] = f"/moved/words/{recording['id']}.json"
        if recording["reference"]:
            recording["reference"]["path"] = f"/moved/refs/{recording['id']}.md"
    moved_experiment_id = calibration.make_experiment_id(moved, _environment())
    assert moved_experiment_id == experiment_id
    assert calibration.expand_automatic_cells(moved, moved_experiment_id) == cells


def test_threshold_selection_requires_complete_rectangle_and_uses_shared_gate(tmp_path):
    manifest = _manifest(tmp_path)
    combination = manifest["combinations"][0]
    specs = calibration.expand_automatic_cells(manifest, "e" * 64)[:6]
    assert calibration.select_threshold(combination, manifest["recordings"], []) is None
    counts = [3, 2, 3, 3, 2, 2]
    purities = [None, 0.7, 0.9, None, 0.8, 0.8]
    cells = [
        _complete_cell(spec, count, purity)
        for spec, count, purity in zip(specs, counts, purities, strict=True)
    ]
    selection = calibration.select_threshold(combination, manifest["recordings"], cells)
    assert selection["selected_threshold"] == 0.9
    assert selection["meets_count_gate"] is True
    known = calibration.derive_known_cells(manifest, "e" * 64, selection)
    assert [cell["num_clusters"] for cell in known] == [3, 2, 2]
    assert {cell["threshold"] for cell in known} == {0.9}


def test_threshold_selection_records_failed_gate(tmp_path):
    manifest = _manifest(tmp_path)
    combination = manifest["combinations"][1]
    specs = [
        cell
        for cell in calibration.expand_automatic_cells(manifest, "e" * 64)
        if cell["combination_id"] == combination["id"]
    ]
    cells = [
        _complete_cell(spec, count, purity)
        for spec, count, purity in zip(specs, [4, 3, 3], [None, 0.8, 0.8], strict=True)
    ]
    assert (
        calibration.select_threshold(combination, manifest["recordings"], cells)[
            "meets_count_gate"
        ]
        is False
    )


def test_stage_timing_parser_accepts_native_lines():
    log = (
        "[I] OfflineSpeakerDiarization: segmentation 1.000 s\n"
        "[I] OfflineSpeakerDiarization: embedding 2.000 s\n"
        "[I] OfflineSpeakerDiarization: clustering 0.100 s\n"
        "[I] OfflineSpeakerDiarization: total 3.100 s, audio 10.000 s, RTF 0.310"
    )
    stages = calibration.parse_stage_timings(log, "1.13.6")
    assert stages["engine_total_seconds"] == 3.1
    assert stages["engine_audio_seconds"] == 10.0
    assert stages["engine_rtf"] == 0.31


@pytest.mark.parametrize(
    "log",
    [
        "OfflineSpeakerDiarization: segmentation 1.0 s",
        "OfflineSpeakerDiarization: segmentation 1.0 s\nOfflineSpeakerDiarization: segmentation 1.0 s\nOfflineSpeakerDiarization: embedding 1.0 s\nOfflineSpeakerDiarization: clustering 1.0 s\nOfflineSpeakerDiarization: total 3.0 s, audio 4 s, RTF .75",
        "OfflineSpeakerDiarization: segmentation -1.0 s\nOfflineSpeakerDiarization: embedding 1.0 s\nOfflineSpeakerDiarization: clustering 1.0 s\nOfflineSpeakerDiarization: total 1.0 s, audio 4 s, RTF .25",
    ],
)
def test_stage_timing_parser_rejects_missing_duplicate_and_negative(log):
    with pytest.raises(ValueError):
        calibration.parse_stage_timings(log, "1.13.6")


def test_stage_timing_parser_rejects_extra_malformed_record():
    log = (
        "OfflineSpeakerDiarization: segmentation ??? s\n"
        "OfflineSpeakerDiarization: segmentation 1.000 s\n"
        "OfflineSpeakerDiarization: embedding 2.000 s\n"
        "OfflineSpeakerDiarization: clustering 0.100 s\n"
        "OfflineSpeakerDiarization: total 3.100 s, audio 10.000 s, RTF 0.310"
    )
    with pytest.raises(ValueError, match="повреждённую"):
        calibration.parse_stage_timings(log, "1.13.6")


def test_measurement_uses_injected_cpu_clock_and_rss_once():
    process = SimpleNamespace()
    calls = []
    cpu_times = iter(
        [SimpleNamespace(user=1.0, system=2.0), SimpleNamespace(user=2.0, system=2.5)]
    )

    def read_cpu_times():
        calls.append("cpu")
        return next(cpu_times)

    process.cpu_times = read_cpu_times

    class Sampler:
        def __init__(self, actual_process, interval):
            assert actual_process is process
            assert interval == 0.01

        def start(self):
            calls.append("start")

        def stop(self):
            calls.append("stop")
            return 1234

    clock = iter([10.0, 12.0]).__next__
    result, metrics = calibration.measure_call(
        lambda: calls.append("process") or "ok",
        process,
        clock=clock,
        sampler_factory=Sampler,
        interval_seconds=0.01,
        logical_cpus=4,
    )
    assert result == "ok"
    assert calls == ["start", "cpu", "process", "cpu", "stop"]
    assert metrics["cpu_total_seconds"] == 1.5
    assert metrics["average_cpu_percent_machine"] == 18.75
    assert metrics["peak_rss_bytes"] == 1234


def test_measurement_preserves_operation_error_when_sampler_stop_also_fails():
    process = SimpleNamespace(
        cpu_times=iter(
            [
                SimpleNamespace(user=1.0, system=2.0),
                SimpleNamespace(user=1.1, system=2.1),
            ]
        ).__next__
    )

    class Sampler:
        def __init__(self, process, interval):
            pass

        def start(self):
            pass

        def stop(self):
            raise RuntimeError("sampler failed")

    def operation():
        raise ValueError("operation failed")

    with pytest.raises(ValueError, match="operation failed") as error:
        calibration.measure_call(
            operation,
            process,
            clock=iter([1.0, 2.0]).__next__,
            sampler_factory=Sampler,
            interval_seconds=0.01,
            logical_cpus=4,
        )
    assert any("sampler failed" in note for note in error.value.__notes__)


def test_decode_cache_identity_covers_hash_bounds_and_decoder():
    recording = {"sha256": "a" * 64, "start": 1.0, "duration": 2.0}
    original = calibration.decode_cache_key(recording)
    assert calibration.decode_cache_key({**recording, "start": 2.0}) != original
    assert calibration.decode_cache_key({**recording, "sha256": "b" * 64}) != original
    assert calibration.decode_cache_key(recording, "another-decoder") != original


def test_decode_clip_declares_wav_format_for_temporary_output(tmp_path, monkeypatch):
    recording = {
        "id": "synthetic",
        "path": str(tmp_path / "source.mp4"),
        "sha256": "a" * 64,
        "start": 1.0,
        "duration": 2.0,
    }
    commands = []

    def run(command, *, check):
        assert check is True
        commands.append(command)
        Path(command[-1]).touch()

    monkeypatch.setattr(calibration.subprocess, "run", run)
    monkeypatch.setattr(calibration, "read_wav", lambda path: None)

    output = calibration.decode_clip(recording, tmp_path / "work")

    assert output.is_file()
    assert not output.with_suffix(".wav.tmp").exists()
    assert commands[0][-3:-1] == ["-f", "wav"]


def test_hypescribe_parser_mapping_and_empty_diagnostics(tmp_path):
    reference = tmp_path / "reference.md"
    reference.write_text(
        "**[00:10] Speaker 1:**\ntext\n**[00:15] Speaker 2:**\n", encoding="utf-8"
    )
    recording = {
        "id": "clip",
        "path": str(tmp_path / "clip.mp4"),
        "sha256": "a" * 64,
        "start": 12.0,
        "duration": 6.0,
        "expected_speakers": 2,
        "reference": {"path": str(reference)},
    }
    turns = calibration.read_reference_turns(recording)
    assert turns == [
        {"speaker": "1", "start": 0.0, "end": 3.0},
        {"speaker": "2", "start": 3.0, "end": 6.0},
    ]
    mapping = calibration.best_mapping(
        [
            {"speaker": 7, "start": 0.0, "end": 3.0},
            {"speaker": 8, "start": 3.0, "end": 6.0},
        ],
        turns,
    )
    assert mapping["mapped_speaker_purity"] == 1.0
    assert calibration.best_mapping([], turns) is None
    summary = calibration.summarize_segments([], recording)
    assert summary["clusters"] == 0 and summary["residual_share_after_expected"] is None


def test_asr_sidecar_validation_text_invariance_and_unknown_words(tmp_path):
    sidecar = tmp_path / "words.json"
    sidecar.write_text(
        json.dumps(
            [
                {"start": 0.0, "end": 0.5, "text": "Привет"},
                {"start": 0.5, "end": 0.6, "text": "!"},
            ]
        ),
        encoding="utf-8",
    )
    words = calibration.load_asr_words(sidecar, 2.0)
    recording = {"duration": 2.0, "expected_speakers": 1}
    diagnostics, invariant = calibration.evaluate_segments(
        [{"speaker": 0, "start": 0.0, "end": 0.5}], recording, words, []
    )
    assert diagnostics["unassigned_word_count"] == 1
    assert diagnostics["mapped_speaker_purity"] is None
    assert invariant["equal"] is True and invariant["first_mismatch_index"] is None
    sidecar.write_text(
        json.dumps(
            [
                {"start": 1.0, "end": 1.1, "text": "a"},
                {"start": 0.5, "end": 0.6, "text": "b"},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="границы"):
        calibration.load_asr_words(sidecar, 2.0)


def test_asr_sidecar_rejects_empty_baseline(tmp_path):
    sidecar = tmp_path / "words.json"
    sidecar.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="хотя бы одно слово"):
        calibration.load_asr_words(sidecar, 300.0)


def test_text_invariance_records_first_mismatch_without_text():
    transcript = SimpleNamespace(turns=[SimpleNamespace(text="другой")])
    words = [SimpleNamespace(text="текст")]
    evidence = calibration.text_invariance(words, transcript)
    assert evidence["equal"] is False
    assert evidence["first_mismatch_index"] == 0
    assert "текст" not in json.dumps(evidence, ensure_ascii=False)


def test_text_invariance_ignores_unicode_whitespace_at_turn_boundaries():
    words = [
        SimpleNamespace(text="Вопрос?"),
        SimpleNamespace(text=" — "),
        SimpleNamespace(text="ответ"),
    ]
    transcript = SimpleNamespace(
        turns=[SimpleNamespace(text="Вопрос?"), SimpleNamespace(text="— ответ")]
    )

    evidence = calibration.text_invariance(words, transcript)

    assert calibration.canonical_non_whitespace_sequence(words) == "Вопрос?—ответ"
    assert evidence["equal"] is True
    assert evidence["first_mismatch_index"] is None


@pytest.mark.parametrize("output", ["Вопрос?ответ", "Вопрос!—ответ"])
def test_text_invariance_rejects_non_whitespace_deletion_or_substitution(output):
    words = [SimpleNamespace(text="Вопрос?"), SimpleNamespace(text="— ответ")]
    transcript = SimpleNamespace(turns=[SimpleNamespace(text=output)])

    evidence = calibration.text_invariance(words, transcript)

    assert evidence["equal"] is False
    assert evidence["input_sha256"] != evidence["output_sha256"]
    assert evidence["first_mismatch_index"] is not None


def test_atomic_persistence_leaves_no_temporary_file(tmp_path):
    output = tmp_path / "nested" / "result.json"
    calibration.save_output(output, {"private": "value"})
    assert json.loads(output.read_text()) == {"private": "value"}
    assert list(output.parent.glob("*.tmp")) == []


def test_atomic_persistence_preserves_target_and_cleans_temp_on_dump_failure(tmp_path):
    output = tmp_path / "result.json"
    calibration.save_output(output, {"stable": True})

    with pytest.raises(TypeError):
        calibration.save_output(output, {"unserializable": object()})

    assert json.loads(output.read_text(encoding="utf-8")) == {"stable": True}
    assert list(output.parent.glob("*.tmp")) == []


def test_worker_failure_is_persisted_and_retried(tmp_path, monkeypatch):
    output = {"cells": []}
    output_path = tmp_path / "result.json"
    spec = {
        "cell_id": "cell",
        "combination_id": "c",
        "recording_id": "r",
        "mode": "automatic",
        "threshold": 0.8,
        "num_clusters": -1,
    }
    attempts = []

    def execute(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("boom")
        return {**spec, "status": "complete"}

    monkeypatch.setattr(calibration, "_execute_cell", execute)
    calibration._run_specs([spec], {}, output, {}, output_path, tmp_path)
    assert output["cells"][0]["status"] == "failed"
    assert output["cells"][0]["metrics"] is None
    calibration._run_specs([spec], {}, output, {}, output_path, tmp_path)
    assert output["cells"][0]["status"] == "complete"
    assert len(attempts) == 2


def test_execute_cell_rejects_text_mismatch_without_persisting_private_text(
    tmp_path, monkeypatch
):
    manifest = _manifest(tmp_path)
    output = {"environment": {"logical_cpus": 4, "sherpa_onnx_version": "1.13.6"}}
    spec = calibration.expand_automatic_cells(manifest, "e" * 64)[0]
    recording = manifest["recordings"][0]
    private_text = " СОВЕРШЕННО-СЕКРЕТНЫЙ-ТЕКСТ"
    Path(recording["asr_words"]["path"]).write_text(
        json.dumps([{"start": 0.0, "end": 0.5, "text": private_text}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        calibration,
        "run_worker",
        lambda *args, **kwargs: (
            {"metrics": {"wall_seconds": 1.0}, "segments": []},
            (
                "OfflineSpeakerDiarization: segmentation 0.300 s\n"
                "OfflineSpeakerDiarization: embedding 0.600 s\n"
                "OfflineSpeakerDiarization: clustering 0.100 s\n"
                "OfflineSpeakerDiarization: total 1.000 s, "
                "audio 10.000 s, RTF 0.100"
            ),
        ),
    )
    monkeypatch.setattr(calibration, "read_reference_turns", lambda *args: [])

    def evaluate(segments, recording, words, reference_turns):
        source = calibration.canonical_non_whitespace_sequence(words)
        assert private_text.strip() in source
        output = source[:-1]
        return {}, {
            "input_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "equal": False,
            "first_mismatch_index": len(output),
        }

    monkeypatch.setattr(calibration, "evaluate_segments", evaluate)

    with pytest.raises(RuntimeError, match="first_mismatch_index=") as error:
        calibration._execute_cell(
            spec,
            manifest,
            output,
            {recording["id"]: tmp_path / "decoded.wav"},
            tmp_path,
        )
    assert private_text.strip() not in str(error.value)


def test_execute_cell_rejects_engine_audio_duration_mismatch(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    output = {"environment": {"logical_cpus": 4, "sherpa_onnx_version": "1.13.6"}}
    spec = calibration.expand_automatic_cells(manifest, "e" * 64)[0]
    recording = manifest["recordings"][0]
    monkeypatch.setattr(
        calibration,
        "run_worker",
        lambda *args, **kwargs: (
            {"metrics": {"wall_seconds": 1.0}, "segments": []},
            (
                "OfflineSpeakerDiarization: segmentation 0.300 s\n"
                "OfflineSpeakerDiarization: embedding 0.600 s\n"
                "OfflineSpeakerDiarization: clustering 0.100 s\n"
                "OfflineSpeakerDiarization: total 1.000 s, "
                "audio 9.000 s, RTF 0.111"
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="Длительность audio"):
        calibration._execute_cell(
            spec,
            manifest,
            output,
            {recording["id"]: tmp_path / "decoded.wav"},
            tmp_path,
        )


def test_main_fails_closed_when_a_cell_failed(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    for recording in manifest["recordings"]:
        recording.update(calibration.ORIGINAL_CORPUS[recording["id"]])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_path = tmp_path / "result.json"
    work_dir = tmp_path / "work"
    monkeypatch.setattr(calibration, "validate_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        calibration,
        "environment_snapshot",
        lambda: {
            **_environment(),
            "started_at_utc": "2026-08-29T00:00:00+00:00",
            "platform": "test",
            "cpu": "test",
            "logical_cpus": 4,
            "python_version": "3.13",
        },
    )
    monkeypatch.setattr(
        calibration,
        "decode_clip",
        lambda recording, target: tmp_path / f"{recording['id']}.wav",
    )

    def run_specs(specs, manifest, output, decoded, output_path, work_dir):
        for spec in specs:
            calibration._replace_cell(
                output, calibration._failed_cell(spec, RuntimeError("boom"))
            )
        calibration.save_output(output_path, output)

    monkeypatch.setattr(calibration, "_run_specs", run_specs)

    with pytest.raises(RuntimeError, match="незавершённых ячеек 15"):
        calibration.main(
            [
                "--manifest",
                str(manifest_path),
                "--output",
                str(output_path),
                "--work-dir",
                str(work_dir),
            ]
        )


def test_main_fails_closed_for_complete_rectangle_without_mapped_purity(
    tmp_path, monkeypatch
):
    manifest = _manifest(tmp_path)
    for recording in manifest["recordings"]:
        recording.update(calibration.ORIGINAL_CORPUS[recording["id"]])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_path = tmp_path / "result.json"
    work_dir = tmp_path / "work"
    monkeypatch.setattr(calibration, "validate_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        calibration,
        "environment_snapshot",
        lambda: {
            **_environment(),
            "started_at_utc": "2026-08-29T00:00:00+00:00",
            "platform": "test",
            "cpu": "test",
            "logical_cpus": 4,
            "python_version": "3.13",
        },
    )
    monkeypatch.setattr(
        calibration,
        "decode_clip",
        lambda recording, target: tmp_path / f"{recording['id']}.wav",
    )
    modes = []

    def run_specs(specs, manifest, output, decoded, output_path, work_dir):
        modes.extend({spec["mode"] for spec in specs})
        expected = {
            recording["id"]: recording["expected_speakers"]
            for recording in manifest["recordings"]
        }
        for spec in specs:
            calibration._replace_cell(
                output,
                _complete_cell(spec, expected[spec["recording_id"]], purity=None),
            )
        calibration.save_output(output_path, output)

    monkeypatch.setattr(calibration, "_run_specs", run_specs)

    with pytest.raises(RuntimeError, match="свип непригоден"):
        calibration.main(
            [
                "--manifest",
                str(manifest_path),
                "--output",
                str(output_path),
                "--work-dir",
                str(work_dir),
            ]
        )

    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert modes == ["automatic"]
    assert {cell["mode"] for cell in output["cells"]} == {"automatic"}
    assert output["threshold_selection"] == []


def test_run_worker_fake_protocol_reports_failure_and_success(tmp_path):
    request = {"cell_id": "abc"}
    commands = []

    def failed(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1, stdout="", stderr="worker failed\n")

    with pytest.raises(RuntimeError, match="worker failed"):
        calibration.run_worker(request, tmp_path, runner=failed)

    def success(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout='{"metrics":{},"segments":[]}',
            stderr="native log",
        )

    payload, stderr = calibration.run_worker(request, tmp_path, runner=success)
    assert payload["segments"] == [] and stderr == "native log"
    for command in commands:
        assert command[:3] == [
            sys.executable,
            str(SCRIPT.resolve()),
            "--worker",
        ]
        request_path = Path(command[3])
        assert request_path.is_file()
        assert json.loads(request_path.read_text(encoding="utf-8")) == request
