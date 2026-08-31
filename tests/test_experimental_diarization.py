import io
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from local_transcriber import experimental_diarization as runner


def _run_shared_cell(monkeypatch, tmp_path, *, counters=None, drop_output_text=False):
    events = []
    active_interval = {"name": None}
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    work_counters = (
        {
            "embedding_jobs": 2,
            "embedding_run_calls": 2,
            "embedding_rows_dropped": 0,
        }
        if counters is None
        else counters
    )

    class Diarization:
        def sort_by_start_time(self):
            return [
                SimpleNamespace(speaker=0, start=0.0, end=0.5),
                SimpleNamespace(speaker=1, start=0.5, end=1.0),
            ]

    class Engine:
        sample_rate = runner.SAMPLE_RATE

        def __init__(self, config):
            self.config = config

        def prepare(self, samples, workers):
            events.append((active_interval["name"], f"prepare:{workers}"))
            return SimpleNamespace(
                is_terminal=False,
                embeddings=embeddings.copy(),
                row_mapping=((0, 0, [(0, 8_000)]), (1, 0, [(8_000, 16_000)])),
                counters=dict(work_counters),
            )

        def finalize(self, prepared, labels):
            events.append((active_interval["name"], "finalize"))
            return Diarization()

    class Clustering:
        def __init__(self, config):
            self.config = config

        def __call__(self, values):
            events.append((active_interval["name"], "cluster"))
            return np.asarray([0, 1], dtype=np.int32)

    fake_sherpa = SimpleNamespace(
        OfflineSpeakerDiarization=Engine,
        FastClusteringConfig=lambda **values: SimpleNamespace(**values),
        FastClustering=Clustering,
    )
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_sherpa)
    monkeypatch.setattr(
        runner,
        "make_config",
        lambda cell, artifacts, provider_config_path: SimpleNamespace(
            validate=lambda: True
        ),
    )
    monkeypatch.setattr(
        runner,
        "_decode_clip",
        lambda recording: np.zeros(runner.SAMPLE_RATE, dtype=np.float32),
    )

    def estimate(values):
        events.append((active_interval["name"], "counter"))
        return {"status": "complete", "num_clusters": 2, "p": 3}

    monkeypatch.setattr(runner, "estimate_nme", estimate)
    measurement_names = iter(("published", "validation"))

    def measure(operation, process, interval_seconds):
        name = next(measurement_names)
        active_interval["name"] = name
        try:
            result = operation()
        finally:
            active_interval["name"] = None
        suffix = 0 if name == "published" else 1
        return result, {
            "wall_seconds": 10.0 + suffix,
            "user_cpu_seconds": 7.0 + suffix,
            "system_cpu_seconds": 1.0,
            "total_cpu_seconds": 8.0 + suffix,
            "average_cpu_cores": 0.8,
            "peak_rss_bytes": 100 + suffix,
        }

    monkeypatch.setattr(runner, "measure_call", measure)

    @contextmanager
    def native_log():
        yield io.BytesIO()

    monkeypatch.setattr(runner, "capture_native_stderr", native_log)
    monkeypatch.setattr(
        runner,
        "parse_prepare_timings",
        lambda log: {"segmentation_seconds": 0.2, "embedding_seconds": 0.1},
    )
    if drop_output_text:
        original_builder = runner.build_speaker_transcript

        def build_without_last_word(words, intervals, duration):
            transcript = original_builder(words, intervals, duration)
            transcript.turns = transcript.turns[:-1]
            return transcript

        monkeypatch.setattr(
            runner, "build_speaker_transcript", build_without_last_word
        )

    words_path = tmp_path / "words.json"
    words_path.write_text(
        '[{"start": 0.1, "end": 0.4, "text": "Привет"}, '
        '{"start": 0.6, "end": 0.9, "text": "мир"}]',
        encoding="utf-8",
    )
    cell = {
        "name": "E1",
        "cell_id": "shared-cell",
        "counter": "nme",
        "window_shift_ratio": 0.1,
        "segmentation_artifact_id": "segmentation",
        "embedding_artifact_id": "embedding",
        "recording_ids": ["recording"],
        "clustering": {"threshold": 0.89, "num_clusters": None},
        "inference": {
            "mode": "shared-session",
            "outer_workers": 2,
            "session_count": 1,
            "intra_op_threads": 2,
            "inter_op_threads": 1,
            "batch_size": 1,
        },
    }
    request = {
        "manifest": {
            "rss_sample_interval_ms": 10,
            "artifacts": [],
            "recordings": [
                {
                    "id": "recording",
                    "path": str(tmp_path / "recording.wav"),
                    "start": 0.0,
                    "duration": 1.0,
                    "expected_speakers": 2,
                    "asr_words": {"path": str(words_path)},
                    "reference": None,
                }
            ],
        },
        "cell": cell,
        "work_dir": str(tmp_path),
    }
    return runner.run_cell(request), events


def test_run_cell_shared_session_separates_measurement_from_validation(
    monkeypatch, tmp_path
):
    result, events = _run_shared_cell(monkeypatch, tmp_path)

    recording = result["recordings"][0]
    assert result["aggregate"]["quality_passed"] is True
    assert recording["text_equal"] is True
    assert recording["prepared"]["inference"] == {
        "mode": "shared-session",
        "rows_equal_to_sequential": True,
        "finite_embeddings": True,
        "labels_equal_to_sequential": True,
        "final_equal_to_sequential": True,
        "max_abs_drift": 0.0,
        "min_cosine": 1.0,
    }
    assert recording["metrics"]["wall_seconds"] == 10.0
    assert recording["metrics"]["peak_rss_bytes"] == 100
    assert recording["metrics"]["validation_wall_seconds"] == 11.0
    assert recording["metrics"]["validation_peak_rss_bytes"] == 101
    assert events == [
        ("published", "prepare:2"),
        ("published", "counter"),
        ("published", "cluster"),
        ("published", "finalize"),
        ("validation", "counter"),
        ("validation", "cluster"),
        ("validation", "prepare:1"),
        ("validation", "counter"),
        ("validation", "cluster"),
        ("validation", "finalize"),
    ]


def test_run_cell_rejects_changed_output_text(monkeypatch, tmp_path):
    result, _ = _run_shared_cell(
        monkeypatch, tmp_path, drop_output_text=True
    )

    assert result["recordings"][0]["text_equal"] is False
    assert result["aggregate"]["quality_passed"] is False


def test_run_cell_rejects_missing_prepared_work_counters(monkeypatch, tmp_path):
    result, _ = _run_shared_cell(monkeypatch, tmp_path, counters={})

    assert result["recordings"][0]["prepared"]["validation_passed"] is False
    assert result["aggregate"]["quality_passed"] is False


def test_segmentation_window_count_includes_last_padded_window():
    assert runner.segmentation_window_count(160_000, 16_000) == 1
    assert runner.segmentation_window_count(160_001, 16_000) == 2
    assert runner.segmentation_window_count(176_000, 16_000) == 2


def test_prepare_timing_parser_accepts_exactly_two_stages():
    log = (
        "OfflineSpeakerDiarization: segmentation 1.250 s\n"
        "OfflineSpeakerDiarization: embedding 2.500 s"
    )

    assert runner.parse_prepare_timings(log) == {
        "segmentation_seconds": 1.25,
        "embedding_seconds": 2.5,
    }

    with pytest.raises(ValueError, match="обе стадии"):
        runner.parse_prepare_timings("OfflineSpeakerDiarization: embedding 2.5 s")


def test_known_counter_clusters_independent_embedding_copies():
    original = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    prepared = SimpleNamespace(embeddings=original, is_terminal=False)

    class FakeClustering:
        calls = 0

        def __init__(self, config):
            self.config = config

        def __call__(self, features):
            FakeClustering.calls += 1
            features[:] = -100
            return list(range(self.config.num_clusters)) + [0]

    fake_sherpa = SimpleNamespace(
        FastClusteringConfig=lambda **values: SimpleNamespace(**values),
        FastClustering=FakeClustering,
    )
    cell = {
        "counter": "known",
        "clustering": {"threshold": 0.89},
    }

    labels, counter, _ = runner._cluster_prepared(
        fake_sherpa, prepared, cell, expected_speakers=2
    )

    assert labels.tolist() == [0, 1, 0]
    assert counter["num_clusters"] == 2
    assert FakeClustering.calls == 1
    validated = runner._validate_cluster_repeatability(
        fake_sherpa,
        prepared,
        cell,
        expected_speakers=2,
        labels=labels,
        counter_result=counter,
    )
    assert validated["repeatable_labels"] is True
    assert FakeClustering.calls == 2
    assert np.array_equal(prepared.embeddings, original)


def _special_cell(mode, *, batch_size=1, sessions=1, outer=1, intra=4):
    return {
        "counter": "nme" if mode == "titanet-batch" else "known",
        "clustering": {"threshold": 0.89},
        "embedding_artifact_id": "embedding",
        "inference": {
            "mode": mode,
            "outer_workers": outer,
            "session_count": sessions,
            "intra_op_threads": intra,
            "inter_op_threads": 1,
            "batch_size": batch_size,
        },
    }


def test_special_inference_enforces_exact_thread_budgets():
    runner._validate_special_inference(
        _special_cell("separate-session", sessions=4, outer=4, intra=1), 4
    )
    runner._validate_special_inference(
        _special_cell("titanet-batch", batch_size=4), 4
    )

    with pytest.raises(ValueError, match="P worker"):
        runner._validate_special_inference(
            _special_cell("separate-session", sessions=2, outer=4, intra=1), 4
        )
    with pytest.raises(ValueError, match="бюджет физических ядер"):
        runner._validate_special_inference(
            _special_cell("separate-session", sessions=4, outer=4, intra=2), 4
        )
    with pytest.raises(ValueError, match="batch 1 или 4"):
        runner._validate_special_inference(
            _special_cell("titanet-batch", batch_size=2), 4
        )


def test_separate_sessions_restore_row_order_without_loss():
    class Stream:
        def __init__(self):
            self.samples = []

        def accept_waveform(self, sample_rate, samples):
            assert sample_rate == runner.SAMPLE_RATE
            self.samples.extend(samples)

        def input_finished(self):
            pass

    class Extractor:
        def create_stream(self):
            return Stream()

        def is_ready(self, stream):
            return bool(stream.samples)

        def compute(self, stream):
            return [stream.samples[0], len(stream.samples)]

    samples = np.arange(30, dtype=np.float32)
    mapping = [
        (0, 0, [(9, 12)]),
        (0, 1, [(1, 3)]),
        (1, 0, [(20, 24)]),
        (1, 1, [(5, 6)]),
        (2, 0, [(14, 19)]),
    ]

    embeddings = runner._compute_separate_embeddings(
        [Extractor(), Extractor()], mapping, samples, outer_workers=2
    )

    assert embeddings.tolist() == [
        [9.0, 3.0],
        [1.0, 2.0],
        [20.0, 4.0],
        [5.0, 1.0],
        [14.0, 5.0],
    ]
    assert np.isfinite(embeddings).all()


def test_separate_runner_measures_validation_and_embedding_rss_separately(
    monkeypatch,
):
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    prepared = SimpleNamespace(
        is_terminal=False,
        embeddings=embeddings.copy(),
        row_mapping=((0, 0, [(0, 5)]), (1, 0, [(5, 10)])),
        counters={
            "embedding_jobs": 2,
            "embedding_run_calls": 2,
            "embedding_rows_dropped": 0,
        },
    )
    events = []
    active_interval = {"name": None}

    class Clustering:
        def __init__(self, config):
            self.config = config

        def __call__(self, values):
            events.append((active_interval["name"], "cluster"))
            return np.asarray([0, 1], dtype=np.int32)

    fake_sherpa = SimpleNamespace(
        FastClusteringConfig=lambda **values: SimpleNamespace(**values),
        FastClustering=Clustering,
    )
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_sherpa)

    def estimate(values):
        events.append((active_interval["name"], "counter"))
        return {"status": "complete", "num_clusters": 2, "p": 3}

    monkeypatch.setattr(runner, "estimate_nme", estimate)
    monkeypatch.setattr(
        runner,
        "_compute_separate_embeddings",
        lambda extractors, mapping, samples, outer_workers: embeddings.copy(),
    )
    measurement_names = iter(("published", "embedding", "validation"))

    def measure(operation, process, interval_seconds):
        name = next(measurement_names)
        previous = active_interval["name"]
        active_interval["name"] = name
        try:
            result = operation()
        finally:
            active_interval["name"] = previous
        values = {
            "published": (10.0, 100),
            "embedding": (3.0, 50),
            "validation": (4.0, 200),
        }
        wall, peak = values[name]
        return result, {
            "wall_seconds": wall,
            "user_cpu_seconds": wall / 2,
            "system_cpu_seconds": 1.0,
            "total_cpu_seconds": wall / 2 + 1.0,
            "average_cpu_cores": 0.6,
            "peak_rss_bytes": peak,
        }

    monkeypatch.setattr(runner, "measure_call", measure)

    @contextmanager
    def native_log():
        yield io.BytesIO()

    monkeypatch.setattr(runner, "capture_native_stderr", native_log)
    monkeypatch.setattr(
        runner,
        "parse_prepare_timings",
        lambda log: {"segmentation_seconds": 0.2, "embedding_seconds": 0.1},
    )

    class Diarization:
        def sort_by_start_time(self):
            return [
                SimpleNamespace(speaker=0, start=0.0, end=0.5),
                SimpleNamespace(speaker=1, start=0.5, end=1.0),
            ]

    class Engine:
        def prepare(self, samples, workers):
            events.append((active_interval["name"], "prepare"))
            return prepared

        def finalize(self, prepared_value, labels):
            events.append((active_interval["name"], "finalize"))
            return Diarization()

    cell = _special_cell("separate-session", sessions=2, outer=2, intra=1)
    cell["counter"] = "nme"
    _, metrics, _, data = runner._run_separate_session_recording(
        Engine(),
        [object(), object()],
        np.zeros(10, dtype=np.float32),
        cell,
        {"expected_speakers": 2},
        0.01,
    )

    assert metrics["peak_rss_bytes"] == 100
    assert metrics["embedding_peak_rss_bytes"] == 50
    assert metrics["validation_peak_rss_bytes"] == 200
    assert metrics["validation_wall_seconds"] == 4.0
    assert metrics["instrumented_pipeline_wall_seconds"] == 10.0
    assert "measured_validation_wall_seconds" not in metrics
    assert data["validation_passed"] is True
    assert events == [
        ("published", "prepare"),
        ("published", "counter"),
        ("published", "cluster"),
        ("published", "finalize"),
        ("validation", "counter"),
        ("validation", "cluster"),
        ("validation", "counter"),
        ("validation", "cluster"),
        ("validation", "finalize"),
    ]


def test_titanet_batches_keep_variable_length_row_order():
    features = [
        np.full((3, 80), 10.0, dtype=np.float32),
        np.full((5, 80), 20.0, dtype=np.float32),
        np.full((2, 80), 30.0, dtype=np.float32),
        np.full((4, 80), 40.0, dtype=np.float32),
        np.full((6, 80), 50.0, dtype=np.float32),
    ]

    class Session:
        def __init__(self):
            self.length_batches = []

        def run(self, output_names, inputs):
            assert output_names == ["embs"]
            values = inputs["audio_signal"]
            lengths = inputs["length"]
            self.length_batches.append(lengths.tolist())
            assert values.shape == (len(lengths), 80, max(lengths))
            return [
                np.asarray(
                    [[values[index, 0, 0], length] for index, length in enumerate(lengths)],
                    dtype=np.float32,
                )
            ]

    session = Session()
    batch4 = runner._infer_titanet_batches(session, features, batch_size=4)
    batch1 = runner._infer_titanet_batches(Session(), features, batch_size=1)

    assert session.length_batches == [[3, 5, 2, 4], [6]]
    assert batch4.tolist() == batch1.tolist() == [
        [10.0, 3.0],
        [20.0, 5.0],
        [30.0, 2.0],
        [40.0, 4.0],
        [50.0, 6.0],
    ]


def test_titanet_runner_compares_batch_1_and_4_before_finalize(monkeypatch):
    embeddings = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32
    )
    prepared = SimpleNamespace(
        is_terminal=False,
        embeddings=embeddings.copy(),
        row_mapping=((0, 0, [(0, 4)]), (0, 1, [(4, 8)]), (1, 0, [(8, 12)])),
        counters={"embedding_jobs": 3, "embedding_run_calls": 3, "embedding_rows_dropped": 0},
    )

    events = []
    active_interval = {"name": None}

    class Clustering:
        def __init__(self, config):
            self.config = config

        def __call__(self, values):
            events.append((active_interval["name"], "cluster"))
            return np.asarray([0, 0, 1], dtype=np.int32)

    fake_sherpa = SimpleNamespace(
        FastClusteringConfig=lambda **values: SimpleNamespace(**values),
        FastClustering=Clustering,
    )
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_sherpa)
    monkeypatch.setattr(
        runner,
        "estimate_nme",
        lambda values: (
            events.append((active_interval["name"], "counter"))
            or {"status": "complete", "num_clusters": 2, "p": 2}
        ),
    )
    monkeypatch.setattr(
        runner,
        "_prepare_titanet_features",
        lambda mapping, samples: [
            np.ones((3, 80), dtype=np.float32),
            np.ones((4, 80), dtype=np.float32),
            np.ones((5, 80), dtype=np.float32),
        ],
    )
    seen_batches = []

    def infer(session, features, batch_size):
        seen_batches.append((active_interval["name"], batch_size))
        return embeddings.copy()

    monkeypatch.setattr(runner, "_infer_titanet_batches", infer)
    measurement_names = iter(
        ("published", "published-embedding", "validation", "validation-embedding")
    )

    def measure(operation, process, interval_seconds):
        name = next(measurement_names)
        previous = active_interval["name"]
        active_interval["name"] = name
        try:
            result = operation()
        finally:
            active_interval["name"] = previous
        return result, {
            "wall_seconds": 1.0,
            "user_cpu_seconds": 0.7,
            "system_cpu_seconds": 0.1,
            "total_cpu_seconds": 0.8,
            "average_cpu_cores": 0.8,
            "peak_rss_bytes": 100,
        }

    monkeypatch.setattr(runner, "measure_call", measure)

    @contextmanager
    def native_log():
        yield io.BytesIO()

    monkeypatch.setattr(runner, "capture_native_stderr", native_log)
    monkeypatch.setattr(
        runner,
        "parse_prepare_timings",
        lambda log: {"segmentation_seconds": 0.2, "embedding_seconds": 0.4},
    )

    class Diarization:
        def __init__(self, labels):
            self.labels = labels

        def sort_by_start_time(self):
            return [
                SimpleNamespace(speaker=int(label), start=index, end=index + 1)
                for index, label in enumerate(self.labels)
            ]

    class Engine:
        def prepare(self, samples, workers):
            assert workers == 1
            events.append((active_interval["name"], "prepare"))
            return prepared

        def finalize(self, prepared_value, labels):
            assert prepared_value is prepared
            events.append((active_interval["name"], "finalize"))
            return Diarization(labels)

    cell = _special_cell("titanet-batch", batch_size=4)
    result, metrics, _, data = runner._run_titanet_batch_recording(
        Engine(),
        object(),
        np.arange(20, dtype=np.float32),
        cell,
        {"expected_speakers": 2},
        0.01,
    )

    assert seen_batches == [("published-embedding", 4), ("validation-embedding", 1)]
    assert data["validation_passed"] is True
    assert data["inference"]["n_equal"] is True
    assert data["inference"]["labels_equal"] is True
    assert data["inference"]["final_equal"] is True
    assert data["counters"]["embedding_run_calls"] == 1
    assert metrics["measurement_kind"] == "composed-estimate"
    assert metrics["wall_seconds_is_composed"] is True
    assert metrics["instrumented_pipeline_wall_seconds"] == 1.0
    assert metrics["embedding_peak_rss_bytes"] == 100
    assert metrics["validation_wall_seconds"] == 1.0
    assert metrics["validation_peak_rss_bytes"] == 100
    assert events == [
        ("published", "prepare"),
        ("published", "counter"),
        ("published", "cluster"),
        ("published", "finalize"),
        ("validation", "counter"),
        ("validation", "cluster"),
        ("validation", "counter"),
        ("validation", "cluster"),
        ("validation", "finalize"),
    ]
    assert [item.speaker for item in result.sort_by_start_time()] == [0, 0, 1]
