"""Исполнитель ячеек исследовательского CPU-стенда."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark_experiment import (
    estimate_eigengap,
    estimate_nme,
    residual_cluster_metrics,
)
from .diarization import _assign_cluster, build_speaker_transcript
from .types import SpeakerInterval, Word

SAMPLE_RATE = 16_000
PYANNOTE_WINDOW_SAMPLES = 160_000
NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
STAGE_RE = re.compile(
    rf"OfflineSpeakerDiarization:\s+(segmentation|embedding|clustering)\s+"
    rf"({NUMBER_PATTERN})\s+s"
)
TOTAL_RE = re.compile(
    rf"OfflineSpeakerDiarization:\s+total\s+({NUMBER_PATTERN})\s+s,\s+"
    rf"audio\s+({NUMBER_PATTERN})\s+s,\s+RTF\s+({NUMBER_PATTERN})"
)
TURN_RE = re.compile(r"^\*\*\[(\d{2}):(\d{2})(?::(\d{2}))?\] Speaker (\d+):\*\*")


class PeakRssSampler:
    """Сэмплирует RSS только вокруг измеряемого вызова."""

    def __init__(self, process: Any, interval_seconds: float):
        self._process = process
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_bytes = 0

    def _sample(self) -> None:
        self.peak_bytes = max(self.peak_bytes, int(self._process.memory_info().rss))

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self._sample()

    def start(self) -> None:
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()
        return self.peak_bytes


def segmentation_window_count(sample_count: int, shift_samples: int) -> int:
    """Повторяет схему полного и последнего дополненного окна sherpa."""
    if sample_count <= 0 or shift_samples <= 0:
        raise ValueError("Число сэмплов и шаг должны быть положительными")
    if sample_count <= PYANNOTE_WINDOW_SAMPLES:
        return 1
    full = (sample_count - PYANNOTE_WINDOW_SAMPLES) // shift_samples + 1
    covered = (full - 1) * shift_samples + PYANNOTE_WINDOW_SAMPLES
    return full + int(covered < sample_count)


def make_config(
    cell: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    provider_config_path: Path | None = None,
) -> Any:
    """Создаёт конфигурацию 1.13.6 для одной исследовательской ячейки."""
    import sherpa_onnx

    inference = cell["inference"]
    pyannote_options = {
        "model": str(artifacts[cell["segmentation_artifact_id"]]["path"]),
    }
    version = tuple(int(part) for part in sherpa_onnx.__version__.split("."))
    if version >= (1, 13, 6):
        pyannote_options["window_shift_ratio"] = float(cell["window_shift_ratio"])
    elif not math.isclose(float(cell["window_shift_ratio"]), 0.1):
        raise RuntimeError("sherpa-onnx 1.13.5 поддерживает только шаг 0.1")
    provider = "cpu" if provider_config_path is None else f"cpu:{provider_config_path}"
    segmentation = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
        pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
            **pyannote_options,
        ),
        num_threads=int(inference["intra_op_threads"]),
        provider=provider,
        debug=True,
    )
    embedding = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(artifacts[cell["embedding_artifact_id"]]["path"]),
        num_threads=int(inference["intra_op_threads"]),
        provider=provider,
        debug=True,
    )
    clustering = cell["clustering"]
    num_clusters = clustering["num_clusters"]
    return sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=segmentation,
        embedding=embedding,
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=-1 if num_clusters is None else int(num_clusters),
            threshold=float(clustering["threshold"]),
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )


def parse_stage_timings(log: str) -> dict[str, float]:
    """Разбирает штатные агрегаты debug-профиля 1.13.6."""
    stages: dict[str, list[float]] = {}
    for name, raw_value in STAGE_RE.findall(log):
        stages.setdefault(name, []).append(float(raw_value))
    total_matches = TOTAL_RE.findall(log)
    if set(stages) != {"segmentation", "embedding", "clustering"}:
        raise ValueError("Debug-журнал не содержит все стадии")
    if any(len(values) != 1 for values in stages.values()) or len(total_matches) != 1:
        raise ValueError("Debug-журнал содержит дубли стадий")
    total, audio, rtf = map(float, total_matches[0])
    values = {
        "segmentation_seconds": stages["segmentation"][0],
        "embedding_seconds": stages["embedding"][0],
        "clustering_seconds": stages["clustering"][0],
        "engine_total_seconds": total,
        "engine_audio_seconds": audio,
        "engine_rtf": rtf,
    }
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("Debug-журнал содержит недопустимое время")
    return values


def parse_prepare_timings(log: str) -> dict[str, float]:
    """Разбирает две стадии экспериментального вызова prepare."""
    stages: dict[str, list[float]] = {}
    for name, raw_value in STAGE_RE.findall(log):
        stages.setdefault(name, []).append(float(raw_value))
    if set(stages) != {"segmentation", "embedding"}:
        raise ValueError("Debug-журнал prepare не содержит обе стадии")
    if any(len(values) != 1 for values in stages.values()):
        raise ValueError("Debug-журнал prepare содержит дубли стадий")
    result = {
        "segmentation_seconds": stages["segmentation"][0],
        "embedding_seconds": stages["embedding"][0],
    }
    if any(not math.isfinite(value) or value < 0 for value in result.values()):
        raise ValueError("Debug-журнал prepare содержит недопустимое время")
    return result


@contextmanager
def capture_native_stderr() -> Any:
    """Перехватывает C/C++ stderr в приватный временный файл."""
    saved_fd = os.dup(2)
    with tempfile.TemporaryFile(mode="w+b") as stream:
        try:
            os.dup2(stream.fileno(), 2)
            yield stream
        finally:
            os.dup2(saved_fd, 2)
            os.close(saved_fd)


def measure_call(
    operation: Callable[[], Any], process: Any, interval_seconds: float
) -> tuple[Any, dict[str, float | int]]:
    """Измеряет wall/CPU/RSS одного process-вызова."""
    before = process.cpu_times()
    sampler = PeakRssSampler(process, interval_seconds)
    sampler.start()
    started = time.perf_counter()
    try:
        result = operation()
    finally:
        wall = time.perf_counter() - started
        peak = sampler.stop()
    after = process.cpu_times()
    user = float(after.user - before.user)
    system = float(after.system - before.system)
    total = user + system
    return result, {
        "wall_seconds": wall,
        "user_cpu_seconds": user,
        "system_cpu_seconds": system,
        "total_cpu_seconds": total,
        "average_cpu_cores": total / wall if wall else 0.0,
        "peak_rss_bytes": peak,
    }


def _load_words(path: Path, duration: float) -> list[Word]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("ASR sidecar должен быть непустым JSON-массивом")
    words = []
    previous = -math.inf
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != {"start", "end", "text"}:
            raise ValueError(f"ASR sidecar[{index}]: неверная структура")
        start = float(item["start"])
        end = float(item["end"])
        if not 0 <= start <= end <= duration or start < previous:
            raise ValueError(f"ASR sidecar[{index}]: неверные временные границы")
        if not isinstance(item["text"], str):
            raise TypeError(f"ASR sidecar[{index}].text: нужна строка")
        words.append(Word(start=start, end=end, text=item["text"]))
        previous = start
    return words


def _text_hash(words: Sequence[Word]) -> str:
    normalized = "".join("".join(word.text.split()) for word in words)
    return hashlib.sha256(normalized.encode()).hexdigest()


def _transcript_text_hash(transcript: Any) -> str:
    normalized = "".join("".join(turn.text.split()) for turn in transcript.turns)
    return hashlib.sha256(normalized.encode()).hexdigest()


def _timestamp_seconds(match: re.Match[str]) -> float:
    first, second, third = match.group(1), match.group(2), match.group(3)
    if third is None:
        return int(first) * 60 + int(second)
    return int(first) * 3600 + int(second) * 60 + int(third)


def _read_reference_turns(recording: Mapping[str, Any]) -> list[dict[str, Any]]:
    reference = recording["reference"]
    if reference is None:
        return []
    starts = []
    for line in Path(reference["path"]).read_text(encoding="utf-8").splitlines():
        match = TURN_RE.match(line)
        if match:
            starts.append((_timestamp_seconds(match), match.group(4)))
    clip_start = float(recording["start"])
    clip_end = clip_start + float(recording["duration"])
    turns = []
    for index, (start, speaker) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else clip_end
        overlap_start = max(start, clip_start)
        overlap_end = min(end, clip_end)
        if overlap_end > overlap_start:
            turns.append(
                {
                    "speaker": speaker,
                    "start": overlap_start - clip_start,
                    "end": overlap_end - clip_start,
                }
            )
    return turns


def _mapped_speaker_purity(
    segments: Sequence[Mapping[str, Any]],
    reference_turns: Sequence[Mapping[str, Any]],
) -> float | None:
    import itertools
    from collections import defaultdict

    if not segments or not reference_turns:
        return None
    predicted = sorted({str(item["speaker"]) for item in segments})
    reference = sorted({str(item["speaker"]) for item in reference_turns})
    overlap: dict[tuple[str, str], float] = defaultdict(float)
    total = 0.0
    for segment in segments:
        for turn in reference_turns:
            value = max(
                0.0,
                min(float(segment["end"]), float(turn["end"]))
                - max(float(segment["start"]), float(turn["start"])),
            )
            overlap[(str(segment["speaker"]), str(turn["speaker"]))] += value
            total += value
    if len(predicted) >= len(reference):
        candidates = (
            zip(choice, reference, strict=True)
            for choice in itertools.permutations(predicted, len(reference))
        )
    else:
        candidates = (
            zip(predicted, choice, strict=True)
            for choice in itertools.permutations(reference, len(predicted))
        )
    best = max(sum(overlap[pair] for pair in candidate) for candidate in candidates)
    return best / total if total else None


def _decode_clip(recording: Mapping[str, Any]) -> np.ndarray:
    from faster_whisper import decode_audio

    samples = decode_audio(str(recording["path"]), sampling_rate=SAMPLE_RATE)
    if isinstance(samples, tuple):
        raise TypeError("Декодер вернул раздельные стереоканалы")
    start = round(float(recording["start"]) * SAMPLE_RATE)
    length = round(float(recording["duration"]) * SAMPLE_RATE)
    clip = np.asarray(samples[start : start + length], dtype=np.float32)
    if len(clip) != length:
        raise ValueError("Запись короче объявленного интервала")
    return clip


def _make_progress_callback() -> tuple[Callable[[int, int], int], dict[str, Any]]:
    state: dict[str, Any] = {"total": 0, "seen": set()}

    def callback(processed: int, total: int) -> int:
        state["total"] = max(state["total"], int(total))
        state["seen"].add(int(processed))
        return 0

    return callback, state


def _write_provider_config(cell: Mapping[str, Any], work_dir: Path) -> Path:
    inference = cell["inference"]
    directory = work_dir / "provider-configs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{cell['cell_id']}.config"
    path.write_text(
        "\n".join(
            (
                f"IntraOpNumThreads={int(inference['intra_op_threads'])}",
                f"InterOpNumThreads={int(inference['inter_op_threads'])}",
                "ExecutionMode=0",
                "",
            )
        ),
        encoding="utf-8",
    )
    return path


def _cluster_prepared(
    sherpa_onnx: Any,
    prepared: Any,
    cell: Mapping[str, Any],
    expected_speakers: int,
    embeddings_override: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any], float]:
    source = prepared.embeddings if embeddings_override is None else embeddings_override
    embeddings = np.ascontiguousarray(np.asarray(source, dtype=np.float32))
    if prepared.is_terminal:
        return np.empty(0, dtype=np.int32), {"status": "terminal"}, 0.0
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError("Prepare вернул недопустимую матрицу эмбеддингов")

    counter_started = time.perf_counter()
    counter = cell["counter"]
    if counter == "eigengap":
        counter_result = estimate_eigengap(embeddings.copy())
        num_clusters = int(counter_result["num_clusters"])
    elif counter == "nme":
        counter_result = estimate_nme(embeddings.copy())
        if counter_result.get("status") != "complete":
            raise ValueError("NME не смог выбрать число кластеров")
        num_clusters = int(counter_result["num_clusters"])
    elif counter == "known":
        num_clusters = expected_speakers
        counter_result = {"status": "known", "num_clusters": num_clusters}
    elif counter == "threshold":
        num_clusters = -1
        counter_result = {
            "status": "threshold",
            "threshold": float(cell["clustering"]["threshold"]),
        }
    else:
        raise ValueError("Неизвестный счетчик говорящих")
    counter_seconds = time.perf_counter() - counter_started

    clustering = sherpa_onnx.FastClustering(
        sherpa_onnx.FastClusteringConfig(
            num_clusters=num_clusters,
            threshold=float(cell["clustering"]["threshold"]),
        )
    )
    cluster_started = time.perf_counter()
    labels = np.asarray(clustering(embeddings.copy()), dtype=np.int32)
    cluster_seconds = time.perf_counter() - cluster_started
    counter_result = {
        **counter_result,
        "seconds": counter_seconds,
        "row_count": int(embeddings.shape[0]),
    }
    return labels, counter_result, cluster_seconds


def _validate_cluster_repeatability(
    sherpa_onnx: Any,
    prepared: Any,
    cell: Mapping[str, Any],
    expected_speakers: int,
    labels: np.ndarray,
    counter_result: Mapping[str, Any],
    embeddings_override: np.ndarray | None = None,
) -> dict[str, Any]:
    """Повторяет счетчик и кластеризацию вне публикуемого измерения."""
    repeated_labels, repeated_counter, _ = _cluster_prepared(
        sherpa_onnx,
        prepared,
        cell,
        expected_speakers,
        embeddings_override,
    )
    repeated_keys = {"status", "p", "num_clusters", "threshold"}
    repeatable_parameters = all(
        counter_result.get(key) == repeated_counter.get(key)
        for key in repeated_keys
        if key in counter_result or key in repeated_counter
    )
    if not repeatable_parameters:
        raise RuntimeError("Счетчик дал неповторяемые p или N на одной матрице")
    repeatable_labels = np.array_equal(labels, repeated_labels)
    if not repeatable_labels:
        raise RuntimeError("FastClustering дал неповторяемые метки на одной матрице")
    return {
        **counter_result,
        "repeatable_parameters": True,
        "repeatable_labels": True,
    }


def _validation_metrics(measured: Mapping[str, Any]) -> dict[str, Any]:
    """Отделяет стоимость и RSS обязательных проверок от результата ячейки."""
    return {f"validation_{name}": value for name, value in measured.items()}


def _validate_special_inference(
    cell: Mapping[str, Any], physical_cores: int
) -> None:
    """Повторно проверяет бюджеты специальных путей внутри worker-процесса."""
    inference = cell["inference"]
    mode = inference["mode"]
    outer = int(inference["outer_workers"])
    sessions = int(inference["session_count"])
    intra = int(inference["intra_op_threads"])
    inter = int(inference["inter_op_threads"])
    batch = int(inference["batch_size"])
    if min(outer, sessions, intra, inter, batch, physical_cores) <= 0:
        raise ValueError("Бюджеты специального runner должны быть положительными")
    if inter != 1:
        raise ValueError("Специальный runner требует inter_op_threads=1")
    if max(outer, sessions * intra) > physical_cores:
        raise ValueError("Специальный runner превышает бюджет физических ядер")
    if mode == "separate-session":
        if (
            outer != physical_cores
            or sessions != physical_cores
            or intra != 1
            or batch != 1
        ):
            raise ValueError(
                "separate-session требует P worker, P session, intra-op=1 и batch=1"
            )
    elif mode == "titanet-batch":
        if (
            outer != 1
            or sessions != 1
            or intra != physical_cores
            or batch not in {1, 4}
        ):
            raise ValueError(
                "titanet-batch требует одну session, один worker, intra-op=P "
                "и batch 1 или 4"
            )
        if cell["counter"] != "nme":
            raise ValueError("titanet-batch проверяется только со счетчиком NME")
    else:
        raise ValueError("Ожидался специальный inference mode")


def _validate_embedding_rows(embeddings: np.ndarray, expected_rows: int) -> None:
    if embeddings.ndim != 2 or embeddings.shape[0] != expected_rows:
        raise RuntimeError("Экспериментальный inference потерял строки эмбеддингов")
    if not np.isfinite(embeddings).all():
        raise RuntimeError("Экспериментальный inference вернул нечисловые значения")


def _prepared_rows_preserved(prepared: Any) -> bool:
    counters = dict(prepared.counters)
    row_count = len(prepared.row_mapping)
    required = {
        "embedding_jobs",
        "embedding_run_calls",
        "embedding_rows_dropped",
    }
    if not required <= counters.keys():
        return False
    try:
        embedding_jobs = int(counters["embedding_jobs"])
        embedding_run_calls = int(counters["embedding_run_calls"])
        embedding_rows_dropped = int(counters["embedding_rows_dropped"])
        return (
            embedding_jobs == row_count
            and embedding_run_calls >= 0
            and embedding_rows_dropped == 0
        )
    except (TypeError, ValueError):
        return False


def _optional_work_counter(counters: Mapping[str, Any], name: str) -> int | None:
    """Не подставляет правдоподобное значение вместо отсутствующего счетчика."""
    try:
        return int(counters[name])
    except (KeyError, TypeError, ValueError):
        return None


def _compute_separate_embeddings(
    extractors: Sequence[Any],
    row_mapping: Sequence[Any],
    samples: np.ndarray,
    outer_workers: int,
) -> np.ndarray:
    """Вычисляет строки независимыми session и восстанавливает исходный порядок."""
    if len(extractors) != outer_workers:
        raise ValueError("Число extractor должно совпадать с outer_workers")
    partitions = [
        list(range(worker, len(row_mapping), outer_workers))
        for worker in range(outer_workers)
    ]

    def run_partition(extractor: Any, indices: Sequence[int]) -> list[tuple[int, Any]]:
        rows = []
        for index in indices:
            stream = extractor.create_stream()
            intervals = row_mapping[index][2]
            if not intervals:
                raise ValueError("Строка эмбеддинга не содержит аудиоинтервалов")
            for start, end in intervals:
                if not 0 <= start < end:
                    raise ValueError("Prepare вернул неверный аудиоинтервал")
                stream.accept_waveform(
                    SAMPLE_RATE,
                    samples[int(start) : min(int(end), len(samples))],
                )
            stream.input_finished()
            if not extractor.is_ready(stream):
                raise RuntimeError("Embedding stream не готов к вычислению")
            rows.append((index, extractor.compute(stream)))
        return rows

    with ThreadPoolExecutor(max_workers=outer_workers) as pool:
        futures = [
            pool.submit(run_partition, extractor, indices)
            for extractor, indices in zip(extractors, partitions, strict=True)
        ]
    indexed = [item for future in futures for item in future.result()]
    indexed.sort(key=lambda item: item[0])
    if [index for index, _ in indexed] != list(range(len(row_mapping))):
        raise RuntimeError("Отдельные session потеряли или продублировали строки")
    embeddings = np.asarray([row for _, row in indexed], dtype=np.float32)
    _validate_embedding_rows(embeddings, len(row_mapping))
    return embeddings


def _titanet_features(waveform: np.ndarray) -> np.ndarray:
    """Извлекает признаки по контракту официального TitaNet small."""
    import kaldi_native_fbank as knf

    options = knf.FbankOptions()
    options.frame_opts.samp_freq = SAMPLE_RATE
    options.frame_opts.dither = 0.0
    options.frame_opts.snip_edges = True
    options.frame_opts.frame_shift_ms = 10.0
    options.frame_opts.frame_length_ms = 25.0
    options.frame_opts.remove_dc_offset = False
    options.frame_opts.preemph_coeff = 0.97
    options.frame_opts.window_type = "hann"
    options.frame_opts.round_to_power_of_two = True
    options.mel_opts.num_bins = 80
    options.mel_opts.low_freq = 0.0
    options.mel_opts.high_freq = -400.0
    options.mel_opts.is_librosa = True
    extractor = knf.OnlineFbank(options)
    extractor.accept_waveform(SAMPLE_RATE, waveform.tolist())
    extractor.input_finished()
    features = np.asarray(
        [extractor.get_frame(index) for index in range(extractor.num_frames_ready)],
        dtype=np.float32,
    )
    if features.ndim != 2 or not len(features) or not np.isfinite(features).all():
        raise RuntimeError("Не удалось получить конечные признаки TitaNet")
    mean = features.mean(axis=0, keepdims=True)
    variance = np.square(features - mean).mean(axis=0, keepdims=True)
    return (features - mean) / (np.sqrt(variance) + np.float32(1e-5))


def _prepare_titanet_features(
    row_mapping: Sequence[Any], samples: np.ndarray
) -> list[np.ndarray]:
    features = []
    for row in row_mapping:
        intervals = row[2]
        if not intervals:
            raise ValueError("Строка TitaNet не содержит аудиоинтервалов")
        chunks = [
            samples[int(start) : min(int(end), len(samples))]
            for start, end in intervals
            if 0 <= start < end
        ]
        if len(chunks) != len(intervals) or any(not len(chunk) for chunk in chunks):
            raise ValueError("Prepare вернул неверный аудиоинтервал TitaNet")
        features.append(_titanet_features(np.concatenate(chunks)))
    if len({len(item) for item in features}) < 2:
        raise RuntimeError("TitaNet batch требует входы переменной длины")
    return features


def _infer_titanet_batches(
    session: Any, features: Sequence[np.ndarray], batch_size: int
) -> np.ndarray:
    """Выполняет TitaNet batch без изменения порядка строк."""
    rows = []
    for offset in range(0, len(features), batch_size):
        batch = features[offset : offset + batch_size]
        lengths = np.asarray([len(item) for item in batch], dtype=np.int64)
        values = np.zeros((len(batch), 80, int(lengths.max())), dtype=np.float32)
        for index, item in enumerate(batch):
            if item.ndim != 2 or item.shape[1] != 80:
                raise ValueError("Признаки TitaNet должны иметь форму [T, 80]")
            values[index, :, : len(item)] = item.T
        output = session.run(
            ["embs"], {"audio_signal": values, "length": lengths}
        )[0]
        rows.extend(output)
    embeddings = np.asarray(rows, dtype=np.float32)
    _validate_embedding_rows(embeddings, len(features))
    return embeddings


def _segments_signature(diarization: Any) -> tuple[tuple[int, float, float], ...]:
    return tuple(
        (int(item.speaker), float(item.start), float(item.end))
        for item in diarization.sort_by_start_time()
    )


def _composed_metrics(
    measured: Mapping[str, Any], composed_wall_seconds: float
) -> dict[str, Any]:
    """Помечает wall-оценку составного экспериментального pipeline."""
    if not math.isfinite(composed_wall_seconds) or composed_wall_seconds < 0:
        raise ValueError("Составной wall-time должен быть конечным и неотрицательным")
    return {
        **measured,
        "wall_seconds": composed_wall_seconds,
        "measurement_kind": "composed-estimate",
        "wall_seconds_is_composed": True,
        "instrumented_pipeline_wall_seconds": measured["wall_seconds"],
    }


def _run_prepared_recording(
    engine: Any,
    samples: np.ndarray,
    cell: Mapping[str, Any],
    recording: Mapping[str, Any],
    interval_seconds: float,
) -> tuple[Any, dict[str, Any], dict[str, Any], Mapping[str, Any]]:
    import psutil
    import sherpa_onnx

    if cell["inference"]["mode"] not in {"sequential", "shared-session"}:
        raise RuntimeError(
            "Этот inference mode требует отдельного экспериментального runner"
        )
    operation_state: dict[str, Any] = {}

    def operation() -> Any:
        prepared = engine.prepare(
            samples,
            int(cell["inference"]["outer_workers"]),
        )
        labels, counter_result, cluster_seconds = _cluster_prepared(
            sherpa_onnx,
            prepared,
            cell,
            int(recording["expected_speakers"]),
        )
        finalized_started = time.perf_counter()
        result = engine.finalize(prepared, labels)
        operation_state.update(
            {
                "prepared": prepared,
                "labels": labels,
                "counter": counter_result,
                "clustering_seconds": cluster_seconds,
                "finalize_seconds": time.perf_counter() - finalized_started,
            }
        )
        return result

    with capture_native_stderr() as native_log:
        diarization, metrics = measure_call(
            operation,
            psutil.Process(os.getpid()),
            interval_seconds,
        )
        native_log.flush()
        native_log.seek(0)
        prepare_metrics = parse_prepare_timings(
            native_log.read().decode(errors="replace")
        )
    prepared = operation_state["prepared"]
    validation_state: dict[str, Any] = {}

    def validate() -> None:
        validation_state["counter"] = _validate_cluster_repeatability(
            sherpa_onnx,
            prepared,
            cell,
            int(recording["expected_speakers"]),
            operation_state["labels"],
            operation_state["counter"],
        )
        validation_passed = _prepared_rows_preserved(prepared)
        if cell["inference"]["mode"] == "shared-session":
            baseline = engine.prepare(samples, 1)
            baseline_labels, _, _ = _cluster_prepared(
                sherpa_onnx,
                baseline,
                cell,
                int(recording["expected_speakers"]),
            )
            baseline_diarization = engine.finalize(baseline, baseline_labels)
            rows_equal = tuple(prepared.row_mapping) == tuple(baseline.row_mapping)
            candidate_embeddings = np.asarray(prepared.embeddings, dtype=np.float32)
            baseline_embeddings = np.asarray(baseline.embeddings, dtype=np.float32)
            finite_embeddings = bool(
                np.isfinite(candidate_embeddings).all()
                and np.isfinite(baseline_embeddings).all()
            )
            labels_equal = np.array_equal(operation_state["labels"], baseline_labels)
            final_equal = _segments_signature(diarization) == _segments_signature(
                baseline_diarization
            )
            validation_passed = bool(
                validation_passed
                and rows_equal
                and finite_embeddings
                and labels_equal
                and final_equal
            )
            validation_state["inference"] = {
                "mode": "shared-session",
                "rows_equal_to_sequential": rows_equal,
                "finite_embeddings": finite_embeddings,
                "labels_equal_to_sequential": labels_equal,
                "final_equal_to_sequential": final_equal,
                **_embedding_drift(candidate_embeddings, baseline_embeddings),
            }
        validation_state["passed"] = validation_passed

    with capture_native_stderr():
        _, validation_measurement = measure_call(
            validate,
            psutil.Process(os.getpid()),
            interval_seconds,
        )
    metrics = {**metrics, **_validation_metrics(validation_measurement)}
    stage_metrics = {
        **prepare_metrics,
        "clustering_seconds": operation_state["clustering_seconds"],
        "finalize_seconds": operation_state["finalize_seconds"],
        "engine_total_seconds": metrics["wall_seconds"],
        "engine_audio_seconds": len(samples) / SAMPLE_RATE,
        "engine_rtf": metrics["wall_seconds"] / (len(samples) / SAMPLE_RATE),
    }
    prepared_data = {
        "counters": dict(prepared.counters),
        "counter": validation_state["counter"],
        "row_mapping_count": len(prepared.row_mapping),
        "row_mapping": tuple(prepared.row_mapping),
        "row_labels": operation_state["labels"],
        "embeddings": np.asarray(prepared.embeddings, dtype=np.float32),
        "validation_passed": validation_state["passed"],
    }
    if "inference" in validation_state:
        prepared_data["inference"] = validation_state["inference"]
    return diarization, metrics, stage_metrics, prepared_data


def _embedding_drift(
    candidate: np.ndarray, baseline: np.ndarray
) -> dict[str, float | None]:
    if candidate.shape != baseline.shape:
        raise RuntimeError("Контрольная и экспериментальная матрицы имеют разную форму")
    denominator = np.linalg.norm(candidate, axis=1) * np.linalg.norm(baseline, axis=1)
    valid = denominator > 0
    cosine = np.sum(candidate[valid] * baseline[valid], axis=1) / denominator[valid]
    return {
        "max_abs_drift": float(np.max(np.abs(candidate - baseline))),
        "min_cosine": float(np.min(cosine)) if len(cosine) else None,
    }


def _run_separate_session_recording(
    engine: Any,
    extractors: Sequence[Any],
    samples: np.ndarray,
    cell: Mapping[str, Any],
    recording: Mapping[str, Any],
    interval_seconds: float,
) -> tuple[Any, dict[str, Any], dict[str, Any], Mapping[str, Any]]:
    import psutil
    import sherpa_onnx

    inference = cell["inference"]
    session_count = int(inference["session_count"])
    operation_state: dict[str, Any] = {}

    def operation() -> Any:
        prepare_started = time.perf_counter()
        prepared = engine.prepare(samples, 1)
        prepare_wall = time.perf_counter() - prepare_started
        if prepared.is_terminal:
            raise RuntimeError("separate-session не получил строки эмбеддингов")
        mapping = tuple(prepared.row_mapping)

        def compute_embeddings() -> np.ndarray:
            return _compute_separate_embeddings(
                extractors,
                mapping,
                samples,
                int(inference["outer_workers"]),
            )

        replacement, replacement_metrics = measure_call(
            compute_embeddings,
            psutil.Process(os.getpid()),
            interval_seconds,
        )
        baseline = np.asarray(prepared.embeddings, dtype=np.float32)
        _validate_embedding_rows(baseline, len(mapping))
        labels, counter, clustering_seconds = _cluster_prepared(
            sherpa_onnx,
            prepared,
            cell,
            int(recording["expected_speakers"]),
            replacement,
        )
        finalize_started = time.perf_counter()
        diarization = engine.finalize(prepared, labels)
        finalize_seconds = time.perf_counter() - finalize_started
        operation_state.update(
            {
                "prepared": prepared,
                "baseline_embeddings": baseline,
                "embeddings": replacement,
                "labels": labels,
                "counter": counter,
                "prepare_wall_seconds": prepare_wall,
                "replacement_metrics": replacement_metrics,
                "clustering_seconds": clustering_seconds,
                "finalize_seconds": finalize_seconds,
            }
        )
        return diarization

    with capture_native_stderr() as native_log:
        diarization, measured = measure_call(
            operation,
            psutil.Process(os.getpid()),
            interval_seconds,
        )
        native_log.flush()
        native_log.seek(0)
        prepare_metrics = parse_prepare_timings(
            native_log.read().decode(errors="replace")
        )
    validation_state: dict[str, Any] = {}

    def validate() -> None:
        prepared = operation_state["prepared"]
        validation_state["counter"] = _validate_cluster_repeatability(
            sherpa_onnx,
            prepared,
            cell,
            int(recording["expected_speakers"]),
            operation_state["labels"],
            operation_state["counter"],
            operation_state["embeddings"],
        )
        baseline_labels, _, _ = _cluster_prepared(
            sherpa_onnx,
            prepared,
            cell,
            int(recording["expected_speakers"]),
        )
        baseline_diarization = engine.finalize(prepared, baseline_labels)
        labels_equal = np.array_equal(operation_state["labels"], baseline_labels)
        final_equal = _segments_signature(diarization) == _segments_signature(
            baseline_diarization
        )
        validation_state.update(
            {
                "labels_equal": labels_equal,
                "final_equal": final_equal,
                "drift": _embedding_drift(
                    operation_state["embeddings"],
                    operation_state["baseline_embeddings"],
                ),
            }
        )

    with capture_native_stderr():
        _, validation_measurement = measure_call(
            validate,
            psutil.Process(os.getpid()),
            interval_seconds,
        )
    replacement_metrics = operation_state["replacement_metrics"]
    composed_wall = (
        operation_state["prepare_wall_seconds"]
        - prepare_metrics["embedding_seconds"]
        + replacement_metrics["wall_seconds"]
        + operation_state["counter"]["seconds"]
        + operation_state["clustering_seconds"]
        + operation_state["finalize_seconds"]
    )
    metrics = _composed_metrics(measured, composed_wall)
    metrics.update(
        {
            "embedding_wall_seconds": replacement_metrics["wall_seconds"],
            "embedding_peak_rss_bytes": replacement_metrics["peak_rss_bytes"],
            **_validation_metrics(validation_measurement),
        }
    )
    stage_metrics = {
        "segmentation_seconds": prepare_metrics["segmentation_seconds"],
        "embedding_seconds": replacement_metrics["wall_seconds"],
        "clustering_seconds": operation_state["clustering_seconds"],
        "finalize_seconds": operation_state["finalize_seconds"],
        "baseline_prepare_embedding_seconds": prepare_metrics["embedding_seconds"],
        "engine_total_seconds": composed_wall,
        "engine_audio_seconds": len(samples) / SAMPLE_RATE,
        "engine_rtf": composed_wall / (len(samples) / SAMPLE_RATE),
    }
    prepared = operation_state["prepared"]
    counters = dict(prepared.counters)
    counters["embedding_run_calls"] = len(prepared.row_mapping)
    rows_preserved = _prepared_rows_preserved(prepared)
    validation_passed = bool(
        rows_preserved
        and validation_state["labels_equal"]
        and validation_state["final_equal"]
    )
    prepared_data = {
        "counters": counters,
        "counter": validation_state["counter"],
        "row_mapping_count": len(prepared.row_mapping),
        "row_mapping": tuple(prepared.row_mapping),
        "row_labels": operation_state["labels"],
        "embeddings": operation_state["embeddings"],
        "validation_passed": validation_passed,
        "inference": {
            "mode": "separate-session",
            "session_count": session_count,
            "outer_workers": int(inference["outer_workers"]),
            "intra_op_threads": int(inference["intra_op_threads"]),
            "inter_op_threads": int(inference["inter_op_threads"]),
            "rows_preserved": rows_preserved,
            "finite_embeddings": True,
            "labels_equal_to_baseline": validation_state["labels_equal"],
            "final_equal_to_baseline": validation_state["final_equal"],
            **validation_state["drift"],
        },
    }
    return diarization, metrics, stage_metrics, prepared_data


def _make_titanet_session(model_path: Path, inference: Mapping[str, Any]) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = int(inference["intra_op_threads"])
    options.inter_op_num_threads = int(inference["inter_op_threads"])
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def _make_separate_extractors(
    sherpa_onnx: Any,
    cell: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    provider_config_path: Path,
) -> list[Any]:
    inference = cell["inference"]
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(artifacts[cell["embedding_artifact_id"]]["path"]),
        num_threads=int(inference["intra_op_threads"]),
        provider=f"cpu:{provider_config_path}",
        debug=False,
    )
    return [
        sherpa_onnx.SpeakerEmbeddingExtractor(config)
        for _ in range(int(inference["session_count"]))
    ]


def _run_titanet_batch_recording(
    engine: Any,
    session: Any,
    samples: np.ndarray,
    cell: Mapping[str, Any],
    recording: Mapping[str, Any],
    interval_seconds: float,
) -> tuple[Any, dict[str, Any], dict[str, Any], Mapping[str, Any]]:
    import psutil
    import sherpa_onnx

    inference = cell["inference"]
    selected_batch = int(inference["batch_size"])
    comparison_batch = 4 if selected_batch == 1 else 1
    operation_state: dict[str, Any] = {}

    def operation() -> Any:
        prepare_started = time.perf_counter()
        prepared = engine.prepare(samples, 1)
        prepare_wall = time.perf_counter() - prepare_started
        if prepared.is_terminal:
            raise RuntimeError("titanet-batch не получил строки эмбеддингов")
        mapping = tuple(prepared.row_mapping)
        feature_started = time.perf_counter()
        features = _prepare_titanet_features(mapping, samples)
        feature_seconds = time.perf_counter() - feature_started
        embeddings, inference_metrics = measure_call(
            lambda: _infer_titanet_batches(
                session, features, selected_batch
            ),
            psutil.Process(os.getpid()),
            interval_seconds,
        )
        labels, counter, clustering_seconds = _cluster_prepared(
            sherpa_onnx,
            prepared,
            cell,
            int(recording["expected_speakers"]),
            embeddings,
        )
        finalize_started = time.perf_counter()
        diarization = engine.finalize(prepared, labels)
        finalize_seconds = time.perf_counter() - finalize_started
        baseline = np.asarray(prepared.embeddings, dtype=np.float32)
        _validate_embedding_rows(baseline, len(mapping))
        operation_state.update(
            {
                "prepared": prepared,
                "prepare_wall_seconds": prepare_wall,
                "feature_seconds": feature_seconds,
                "features": features,
                "baseline_embeddings": baseline,
                "embeddings": embeddings,
                "inference_metrics": inference_metrics,
                "labels": labels,
                "counter": counter,
                "clustering_seconds": clustering_seconds,
                "finalize_seconds": finalize_seconds,
            }
        )
        return diarization

    with capture_native_stderr() as native_log:
        diarization, measured = measure_call(
            operation,
            psutil.Process(os.getpid()),
            interval_seconds,
        )
        native_log.flush()
        native_log.seek(0)
        prepare_metrics = parse_prepare_timings(
            native_log.read().decode(errors="replace")
        )
    validation_state: dict[str, Any] = {}

    def validate() -> None:
        prepared = operation_state["prepared"]
        selected_counter = _validate_cluster_repeatability(
            sherpa_onnx,
            prepared,
            cell,
            int(recording["expected_speakers"]),
            operation_state["labels"],
            operation_state["counter"],
            operation_state["embeddings"],
        )
        comparison_embeddings, comparison_metrics = measure_call(
            lambda: _infer_titanet_batches(
                session,
                operation_state["features"],
                comparison_batch,
            ),
            psutil.Process(os.getpid()),
            interval_seconds,
        )
        comparison_labels, comparison_counter, _ = _cluster_prepared(
            sherpa_onnx,
            prepared,
            cell,
            int(recording["expected_speakers"]),
            comparison_embeddings,
        )
        comparison_diarization = engine.finalize(prepared, comparison_labels)
        batches = {
            selected_batch: {
                "embeddings": operation_state["embeddings"],
                "labels": operation_state["labels"],
                "counter": selected_counter,
                "diarization": diarization,
            },
            comparison_batch: {
                "embeddings": comparison_embeddings,
                "labels": comparison_labels,
                "counter": comparison_counter,
                "diarization": comparison_diarization,
            },
        }
        labels_equal = np.array_equal(batches[1]["labels"], batches[4]["labels"])
        final_equal = _segments_signature(
            batches[1]["diarization"]
        ) == _segments_signature(batches[4]["diarization"])
        n_equal = (
            batches[1]["counter"]["num_clusters"]
            == batches[4]["counter"]["num_clusters"]
        )
        validation_state.update(
            {
                "selected_counter": selected_counter,
                "comparison_metrics": comparison_metrics,
                "batches": batches,
                "labels_equal": labels_equal,
                "final_equal": final_equal,
                "n_equal": n_equal,
                "drift": _embedding_drift(
                    batches[4]["embeddings"], batches[1]["embeddings"]
                ),
                "baseline_drift": _embedding_drift(
                    batches[1]["embeddings"],
                    operation_state["baseline_embeddings"],
                ),
            }
        )

    with capture_native_stderr():
        _, validation_measurement = measure_call(
            validate,
            psutil.Process(os.getpid()),
            interval_seconds,
        )
    selected_metrics = operation_state["inference_metrics"]
    composed_wall = (
        operation_state["prepare_wall_seconds"]
        - prepare_metrics["embedding_seconds"]
        + operation_state["feature_seconds"]
        + selected_metrics["wall_seconds"]
        + operation_state["counter"]["seconds"]
        + operation_state["clustering_seconds"]
        + operation_state["finalize_seconds"]
    )
    metrics = _composed_metrics(measured, composed_wall)
    metrics.update(
        {
            "embedding_wall_seconds": selected_metrics["wall_seconds"],
            "embedding_peak_rss_bytes": selected_metrics["peak_rss_bytes"],
            **_validation_metrics(validation_measurement),
        }
    )
    stage_metrics = {
        "segmentation_seconds": prepare_metrics["segmentation_seconds"],
        "embedding_seconds": (
            operation_state["feature_seconds"] + selected_metrics["wall_seconds"]
        ),
        "clustering_seconds": operation_state["clustering_seconds"],
        "finalize_seconds": operation_state["finalize_seconds"],
        "baseline_prepare_embedding_seconds": prepare_metrics["embedding_seconds"],
        "engine_total_seconds": composed_wall,
        "engine_audio_seconds": len(samples) / SAMPLE_RATE,
        "engine_rtf": composed_wall / (len(samples) / SAMPLE_RATE),
    }
    prepared = operation_state["prepared"]
    counters = dict(prepared.counters)
    counters["embedding_run_calls"] = math.ceil(
        len(prepared.row_mapping) / selected_batch
    )
    rows_preserved = _prepared_rows_preserved(prepared)
    validation_passed = bool(
        rows_preserved
        and validation_state["n_equal"]
        and validation_state["labels_equal"]
        and validation_state["final_equal"]
    )
    prepared_data = {
        "counters": counters,
        "counter": validation_state["selected_counter"],
        "row_mapping_count": len(prepared.row_mapping),
        "row_mapping": tuple(prepared.row_mapping),
        "row_labels": operation_state["labels"],
        "embeddings": operation_state["embeddings"],
        "validation_passed": validation_passed,
        "inference": {
            "mode": "titanet-batch",
            "batch_size": selected_batch,
            "session_count": int(inference["session_count"]),
            "intra_op_threads": int(inference["intra_op_threads"]),
            "inter_op_threads": int(inference["inter_op_threads"]),
            "variable_frame_lengths": sorted(
                {len(item) for item in operation_state["features"]}
            ),
            "rows_preserved": rows_preserved,
            "finite_embeddings": True,
            "batch_1_n": validation_state["batches"][1]["counter"]["num_clusters"],
            "batch_4_n": validation_state["batches"][4]["counter"]["num_clusters"],
            "n_equal": validation_state["n_equal"],
            "labels_equal": validation_state["labels_equal"],
            "final_equal": validation_state["final_equal"],
            **validation_state["drift"],
            "baseline_batch_1_max_abs_drift": validation_state["baseline_drift"][
                "max_abs_drift"
            ],
            "baseline_batch_1_min_cosine": validation_state["baseline_drift"][
                "min_cosine"
            ],
        },
    }
    return diarization, metrics, stage_metrics, prepared_data


def _run_public_cell(request: Mapping[str, Any]) -> dict[str, Any]:
    import psutil
    import sherpa_onnx

    manifest = request["manifest"]
    cell = request["cell"]
    patched = hasattr(sherpa_onnx.OfflineSpeakerDiarization, "prepare")
    needs_patch = (
        cell["counter"] != "threshold" or cell["inference"]["mode"] != "sequential"
    )
    if needs_patch and not patched:
        raise RuntimeError("Ячейке нужен экспериментальный Prepare/Finalize")
    mode = cell["inference"]["mode"]
    if mode in {"separate-session", "titanet-batch"}:
        _validate_special_inference(
            cell, int(manifest["machine"]["physical_cores"])
        )

    artifacts = {item["id"]: item for item in manifest["artifacts"]}
    recordings = {item["id"]: item for item in manifest["recordings"]}
    provider_config_path = (
        _write_provider_config(cell, Path(request["work_dir"])) if patched else None
    )
    config = make_config(cell, artifacts, provider_config_path)
    if not config.validate():
        raise RuntimeError("Конфигурация диаризатора недействительна")
    engine = sherpa_onnx.OfflineSpeakerDiarization(config)
    if engine.sample_rate != SAMPLE_RATE:
        raise RuntimeError("Частота дискретизации модели не совпала")
    special_runtime = None
    if mode == "separate-session":
        if provider_config_path is None:
            raise RuntimeError("separate-session требует provider config")
        special_runtime = _make_separate_extractors(
            sherpa_onnx,
            cell,
            artifacts,
            provider_config_path,
        )
    elif mode == "titanet-batch":
        special_runtime = _make_titanet_session(
            Path(artifacts[cell["embedding_artifact_id"]]["path"]),
            cell["inference"],
        )

    results = []
    for recording_id in cell["recording_ids"]:
        recording = recordings[recording_id]
        samples = _decode_clip(recording)
        interval_seconds = float(manifest["rss_sample_interval_ms"]) / 1000.0
        prepared_data = None
        if mode == "separate-session":
            diarization, metrics, stage_metrics, prepared_data = (
                _run_separate_session_recording(
                    engine,
                    special_runtime,
                    samples,
                    cell,
                    recording,
                    interval_seconds,
                )
            )
            progress_total = _optional_work_counter(
                prepared_data["counters"], "embedding_jobs"
            )
            progress_values_seen = _optional_work_counter(
                prepared_data["counters"], "embedding_run_calls"
            )
        elif mode == "titanet-batch":
            diarization, metrics, stage_metrics, prepared_data = (
                _run_titanet_batch_recording(
                    engine,
                    special_runtime,
                    samples,
                    cell,
                    recording,
                    interval_seconds,
                )
            )
            progress_total = _optional_work_counter(
                prepared_data["counters"], "embedding_jobs"
            )
            progress_values_seen = _optional_work_counter(
                prepared_data["counters"], "embedding_run_calls"
            )
        elif needs_patch:
            diarization, metrics, stage_metrics, prepared_data = (
                _run_prepared_recording(
                    engine,
                    samples,
                    cell,
                    recording,
                    interval_seconds,
                )
            )
            progress_total = _optional_work_counter(
                prepared_data["counters"], "embedding_jobs"
            )
            progress_values_seen = _optional_work_counter(
                prepared_data["counters"], "embedding_run_calls"
            )
        else:
            progress, progress_state = _make_progress_callback()
            with capture_native_stderr() as native_log:
                diarization, metrics = measure_call(
                    lambda samples=samples, progress=progress: engine.process(
                        samples, progress
                    ),
                    psutil.Process(os.getpid()),
                    interval_seconds,
                )
                native_log.flush()
                native_log.seek(0)
                stage_metrics = parse_stage_timings(
                    native_log.read().decode(errors="replace")
                )
            progress_total = int(progress_state["total"])
            progress_values_seen = len(progress_state["seen"])
        segments = [
            {
                "speaker": int(item.speaker),
                "start": float(item.start),
                "end": float(item.end),
            }
            for item in diarization.sort_by_start_time()
        ]
        intervals = [
            SpeakerInterval(
                start=item["start"], end=item["end"], cluster=item["speaker"]
            )
            for item in segments
        ]
        words = _load_words(
            Path(recording["asr_words"]["path"]), float(recording["duration"])
        )
        transcript = build_speaker_transcript(
            words, intervals, float(recording["duration"])
        )
        reference_turns = _read_reference_turns(recording)
        word_rows = []
        for word in words:
            assigned = _assign_cluster(word, intervals)
            word_rows.append(
                {"start": word.start, "end": word.end, "speaker": assigned}
            )
        shift = int(float(cell["window_shift_ratio"]) * PYANNOTE_WINDOW_SAMPLES)
        recording_result = {
            "recording_id": recording_id,
            "expected_speakers": recording["expected_speakers"],
            "cluster_count": transcript.cluster_count,
            "unassigned_word_count": transcript.unassigned_word_count,
            "small_cluster_count": len(transcript.small_clusters),
            "small_cluster_seconds": sum(
                item.duration for item in transcript.small_clusters
            ),
            "mapped_speaker_purity": _mapped_speaker_purity(segments, reference_turns),
            "text_input_sha256": _text_hash(words),
            "text_output_sha256": _transcript_text_hash(transcript),
            "text_equal": _text_hash(words) == _transcript_text_hash(transcript),
            "window_shift_samples": shift,
            "segmentation_windows": segmentation_window_count(len(samples), shift),
            "embedding_jobs": progress_total,
            "progress_values_seen": progress_values_seen,
            "metrics": {**metrics, **stage_metrics},
            "segments": segments,
            "word_assignments": word_rows,
            **residual_cluster_metrics(
                segments, word_rows, int(recording["expected_speakers"])
            ),
        }
        if prepared_data is not None:
            matrix_dir = Path(request["work_dir"]) / "prepared"
            matrix_dir.mkdir(parents=True, exist_ok=True)
            matrix_path = matrix_dir / f"{cell['cell_id']}-{recording_id}.npz"
            np.savez_compressed(
                matrix_path,
                embeddings=prepared_data["embeddings"],
                row_labels=prepared_data["row_labels"],
                row_mapping=np.asarray(
                    [item[:2] for item in prepared_data["row_mapping"]],
                    dtype=np.int32,
                ),
                sample_intervals_json=json.dumps(
                    [item[2] for item in prepared_data["row_mapping"]]
                ),
            )
            recording_result["prepared"] = {
                "sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
                "rows": int(prepared_data["embeddings"].shape[0]),
                "columns": int(prepared_data["embeddings"].shape[1]),
                "row_mapping_count": prepared_data["row_mapping_count"],
                "counters": prepared_data["counters"],
                "counter": prepared_data["counter"],
                "validation_passed": prepared_data.get("validation_passed", True),
            }
            if "inference" in prepared_data:
                recording_result["prepared"]["inference"] = prepared_data["inference"]
        results.append(recording_result)

    return {
        "cell_name": cell["name"],
        "recordings": results,
        "aggregate": {
            "wall_seconds": sum(item["metrics"]["wall_seconds"] for item in results),
            "peak_rss_bytes": max(
                item["metrics"]["peak_rss_bytes"] for item in results
            ),
            "quality_passed": all(
                item["cluster_count"] == item["expected_speakers"]
                and item["text_equal"]
                and (
                    "prepared" not in item
                    or (
                        item["prepared"]["counters"].get("embedding_rows_dropped")
                        == 0
                        and item["prepared"]["validation_passed"]
                        and item["prepared"]["counter"].get(
                            "repeatable_parameters", False
                        )
                        and item["prepared"]["counter"].get(
                            "repeatable_labels",
                            item["prepared"]["counter"].get("status") == "terminal",
                        )
                    )
                )
                for item in results
            ),
        },
    }


def run_cell(request: Mapping[str, Any]) -> dict[str, Any]:
    """Выбирает публичный или патченный путь одной ячейки."""
    return _run_public_cell(request)
