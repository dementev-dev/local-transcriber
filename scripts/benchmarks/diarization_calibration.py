"""Воспроизводимый CPU-benchmark офлайн-диаризации sherpa-onnx."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time
import wave
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

SCHEMA_VERSION = 2
EXPECTED_SHERPA_ONNX_VERSION = "1.13.6"
SAMPLE_RATE = 16_000
SAMPLE_WIDTH_BYTES = 2
CHANNELS = 1
DECODER_ID = "ffmpeg-mono-16000-pcm_s16le-v1"
AUDIO_DURATION_TOLERANCE_SECONDS = 0.05
TOP_LEVEL_KEYS = {
    "schema_version",
    "segmentation_models",
    "embedding_models",
    "combinations",
    "recordings",
    "threads",
    "rss_sample_interval_ms",
}
SEGMENTATION_KEYS = {
    "id",
    "path",
    "precision",
    "sha256",
    "size_bytes",
    "source_url",
    "license",
}
EMBEDDING_KEYS = {"id", "path", "sha256", "size_bytes", "source_url", "license"}
COMBINATION_KEYS = {"id", "segmentation_id", "embedding_id", "role", "thresholds"}
RECORDING_KEYS = {
    "id",
    "path",
    "sha256",
    "start",
    "duration",
    "expected_speakers",
    "reference",
    "asr_words",
}
REFERENCE_KEYS = {"path", "sha256", "format"}
ASR_WORDS_KEYS = {"path", "sha256", "format"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TURN_RE = re.compile(r"^\*\*\[(\d{2}):(\d{2})(?::(\d{2}))?\] Speaker (\d+):\*\*")
NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
STAGE_RE = re.compile(
    rf"OfflineSpeakerDiarization:\s+(segmentation|embedding|clustering)\s+({NUMBER_PATTERN})\s+s"
)
TOTAL_RE = re.compile(
    rf"OfflineSpeakerDiarization:\s+total\s+({NUMBER_PATTERN})\s+s,\s+"
    rf"audio\s+({NUMBER_PATTERN})\s+s,\s+RTF\s+({NUMBER_PATTERN})"
)
ORIGINAL_CORPUS = {
    "data-test": {
        "sha256": "1057616b42e8add00e0eb975b02bdef0ec9f6cdfec6dbf488e0c60423c9b7b87",
        "start": 420.0,
        "duration": 300.0,
        "expected_speakers": 3,
    },
    "t2-bdma": {
        "sha256": "51866d247fe3eda134cdd884f707b1f1db8855b492e8bd14d2b56b62476255ed",
        "start": 0.0,
        "duration": 300.0,
        "expected_speakers": 2,
    },
    "yantar": {
        "sha256": "4422f04e2771091a0648e5422d14a31dd7ca2c8eef7ed9f4a5c63f43d8ca6400",
        "start": 0.0,
        "duration": 300.0,
        "expected_speakers": 2,
    },
}


class PeakRssSampler:
    """Сэмплирует RSS процесса только внутри измеряемого интервала."""

    def __init__(self, process: Any, interval_seconds: float):
        self._process = process
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_bytes: int | None = None

    def _sample(self) -> None:
        value = int(self._process.memory_info().rss)
        self.peak_bytes = (
            value if self.peak_bytes is None else max(self.peak_bytes, value)
        )

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
        if self.peak_bytes is None:
            raise RuntimeError("Не удалось измерить пиковый RSS")
        return self.peak_bytes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Проверяет и запускает воспроизводимый CPU-benchmark диаризации. "
            "Приватные пути и результаты остаются вне Git."
        ),
        epilog=(
            "Сначала выполните --validate-only. Полный запуск атомарно сохраняет "
            "каждую ячейку и продолжает совместимый output после прерывания."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="manifest v2/v3; относительные пути считаются от его каталога",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="приватный JSON результата; существующий совместимый файл продолжится",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="каталог декодированных WAV и запросов дочерним процессам",
    )
    parser.add_argument(
        "--capsule",
        type=Path,
        help="исходный ZIP приватной капсулы для проверки handoff",
    )
    parser.add_argument(
        "--capsule-sha256",
        help="ожидаемый внешний SHA-256 ZIP-капсулы",
    )
    # Оставлено для воспроизводимости старых команд; значение входит в experiment_id.
    parser.add_argument(
        "--threads",
        type=int,
        help="число потоков; если задано, должно совпасть с manifest.threads",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="проверить файлы, корпус и матрицу без декодирования и запуска моделей",
    )
    parser.add_argument(
        "--prepare-asr",
        metavar="RECORDING_ID",
        help="создать ASR-sidecar одной записи и вывести его SHA-256",
    )
    parser.add_argument("--asr-model", help="production ASR-модель для --prepare-asr")
    parser.add_argument("--asr-device", help="устройство ASR для --prepare-asr")
    parser.add_argument("--asr-compute-type", help="точность ASR для --prepare-asr")
    parser.add_argument("--asr-language", help="язык ASR для --prepare-asr")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker is None:
        validate_only_conflicts = (
            "output",
            "work_dir",
            "prepare_asr",
            "asr_model",
            "asr_device",
            "asr_compute_type",
            "asr_language",
        )
        if args.validate_only and any(
            getattr(args, name) is not None for name in validate_only_conflicts
        ):
            parser.error(
                "--validate-only нельзя сочетать с параметрами запуска или подготовки ASR"
            )
        missing = ["manifest"] if args.manifest is None else []
        if not args.validate_only and args.work_dir is None:
            missing.append("work_dir")
        if not args.validate_only and args.prepare_asr is None and args.output is None:
            missing.append("output")
        if missing:
            parser.error("обязательные параметры: " + ", ".join(missing))
        if (args.capsule is None) != (args.capsule_sha256 is None):
            parser.error("--capsule и --capsule-sha256 задаются вместе")
        if args.prepare_asr is not None:
            asr_missing = [
                name
                for name in (
                    "asr_model",
                    "asr_device",
                    "asr_compute_type",
                    "asr_language",
                )
                if getattr(args, name) is None
            ]
            if asr_missing:
                parser.error("для --prepare-asr нужны: " + ", ".join(asr_missing))
    return args


def resolve_manifest_paths(
    manifest: dict[str, Any], manifest_dir: Path
) -> dict[str, Any]:
    """Разрешает приватные относительные пути от каталога manifest."""
    result = copy.deepcopy(manifest)
    base = manifest_dir.resolve()

    def resolve(value: str) -> str:
        path = Path(value)
        return str(path if path.is_absolute() else (base / path).resolve())

    for model in result["segmentation_models"]:
        model["path"] = resolve(model["path"])
    for model in result["embedding_models"]:
        model["path"] = resolve(model["path"])
    for recording in result["recordings"]:
        recording["path"] = resolve(recording["path"])
        if recording["reference"] is not None:
            recording["reference"]["path"] = resolve(recording["reference"]["path"])
        recording["asr_words"]["path"] = resolve(recording["asr_words"]["path"])
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_exact_keys(item: dict[str, Any], expected: set[str], label: str) -> None:
    if set(item) != expected:
        missing = sorted(expected - set(item))
        extra = sorted(set(item) - expected)
        raise ValueError(
            f"{label}: неверные поля; отсутствуют={missing}, лишние={extra}"
        )


def _require_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label}: id должен быть непустой стабильной строкой")
    if "/" in value or "\\" in value:
        raise ValueError(f"{label}: id не должен зависеть от пути")
    return value


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label}: ожидается SHA-256 в нижнем регистре")
    return value


def _require_finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label}: ожидается число")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{label}: недопустимое число")
    return result


def _unique_ids(items: list[dict[str, Any]], label: str) -> None:
    ids = [
        _require_id(item.get("id"), f"{label}[{index}]")
        for index, item in enumerate(items)
    ]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label}: id должны быть уникальными")


def _embedding_family(model_id: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]", "", model_id.lower())
    if "wespeaker" in normalized:
        return "wespeaker"
    if "titanet" in normalized:
        return "titanet"
    return None


def _recording_family(recording_id: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]", "", recording_id.lower())
    if "datatest" in normalized:
        return "data-test"
    if "t2bdma" in normalized:
        return "t2-bdma"
    if "yantar" in normalized:
        return "yantar"
    return None


def validate_manifest(
    manifest: dict[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    """Строго проверяет manifest v2 до запуска дорогих операций."""
    if not isinstance(manifest, dict):
        raise TypeError("Manifest должен быть JSON-объектом")
    _require_exact_keys(manifest, TOP_LEVEL_KEYS, "manifest")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Поддерживается только schema_version=2")
    for key in (
        "segmentation_models",
        "embedding_models",
        "combinations",
        "recordings",
    ):
        if (
            not isinstance(manifest[key], list)
            or not manifest[key]
            or not all(isinstance(item, dict) for item in manifest[key])
        ):
            raise ValueError(f"manifest.{key}: ожидается непустой массив объектов")
        _unique_ids(manifest[key], f"manifest.{key}")
    if (
        isinstance(manifest["threads"], bool)
        or not isinstance(manifest["threads"], int)
        or manifest["threads"] <= 0
    ):
        raise ValueError("manifest.threads должен быть положительным целым")
    interval = manifest["rss_sample_interval_ms"]
    if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
        raise ValueError(
            "manifest.rss_sample_interval_ms должен быть положительным целым"
        )

    for index, model in enumerate(manifest["segmentation_models"]):
        label = f"segmentation_models[{index}]"
        _require_exact_keys(model, SEGMENTATION_KEYS, label)
        if model["precision"] not in {"fp32", "int8"}:
            raise ValueError(f"{label}.precision: ожидается fp32 или int8")
        _validate_artifact(model, label, verify_files)
    for index, model in enumerate(manifest["embedding_models"]):
        label = f"embedding_models[{index}]"
        _require_exact_keys(model, EMBEDDING_KEYS, label)
        if _embedding_family(model["id"]) is None:
            raise ValueError(f"{label}.id: ожидается WeSpeaker или TitaNet")
        _validate_artifact(model, label, verify_files)

    segmentation = {item["id"]: item for item in manifest["segmentation_models"]}
    embedding = {item["id"]: item for item in manifest["embedding_models"]}
    segmentation_ids_by_precision: defaultdict[str, set[str]] = defaultdict(set)
    for model in manifest["segmentation_models"]:
        segmentation_ids_by_precision[model["precision"]].add(model["id"])
    embedding_ids_by_family: defaultdict[str | None, set[str]] = defaultdict(set)
    for model in manifest["embedding_models"]:
        embedding_ids_by_family[_embedding_family(model["id"])].add(model["id"])
    if set(segmentation_ids_by_precision) != {"fp32", "int8"} or any(
        len(model_ids) != 1 for model_ids in segmentation_ids_by_precision.values()
    ):
        raise ValueError("Нужен ровно один segmentation model id для FP32 и INT8")
    if set(embedding_ids_by_family) != {"wespeaker", "titanet"} or any(
        len(model_ids) != 1 for model_ids in embedding_ids_by_family.values()
    ):
        raise ValueError("Нужен ровно один embedding model id для WeSpeaker и TitaNet")
    matrix: set[tuple[str, str | None]] = set()
    baselines: list[tuple[str, str | None]] = []
    for index, combination in enumerate(manifest["combinations"]):
        label = f"combinations[{index}]"
        _require_exact_keys(combination, COMBINATION_KEYS, label)
        if (
            combination["segmentation_id"] not in segmentation
            or combination["embedding_id"] not in embedding
        ):
            raise ValueError(f"{label}: неизвестная модель")
        if combination["role"] not in {"baseline", "candidate"}:
            raise ValueError(f"{label}.role: ожидается baseline или candidate")
        thresholds = combination["thresholds"]
        if not isinstance(thresholds, list) or not thresholds:
            raise ValueError(f"{label}.thresholds: ожидается непустой массив")
        values = [_require_finite(value, f"{label}.thresholds") for value in thresholds]
        if values != sorted(values) or len(values) != len(set(values)):
            raise ValueError(
                f"{label}.thresholds должны быть отсортированы без повторов"
            )
        pair = (
            segmentation[combination["segmentation_id"]]["precision"],
            _embedding_family(combination["embedding_id"]),
        )
        if pair in matrix:
            raise ValueError(f"{label}: сочетание моделей объявлено повторно")
        matrix.add(pair)
        if combination["role"] == "baseline":
            baselines.append(pair)
    required_matrix = {
        (precision, family)
        for precision in ("fp32", "int8")
        for family in ("wespeaker", "titanet")
    }
    if matrix != required_matrix or len(manifest["combinations"]) != 4:
        raise ValueError(
            "Manifest должен содержать ровно четыре сочетания FP32/INT8 × WeSpeaker/TitaNet"
        )
    if baselines != [("fp32", "wespeaker")]:
        raise ValueError("Должен быть ровно один baseline FP32+WeSpeaker")

    recording_families: set[str | None] = set()
    for index, recording in enumerate(manifest["recordings"]):
        label = f"recordings[{index}]"
        _require_exact_keys(recording, RECORDING_KEYS, label)
        if not isinstance(recording["path"], str) or not recording["path"]:
            raise ValueError(f"{label}.path: ожидается непустая строка")
        _require_hash(recording["sha256"], f"{label}.sha256")
        start = _require_finite(recording["start"], f"{label}.start")
        if start < 0:
            raise ValueError(f"{label}.start не может быть отрицательным")
        _require_finite(recording["duration"], f"{label}.duration", positive=True)
        expected = recording["expected_speakers"]
        if isinstance(expected, bool) or not isinstance(expected, int) or expected <= 0:
            raise ValueError(
                f"{label}.expected_speakers должен быть положительным целым"
            )
        reference = recording["reference"]
        if reference is not None:
            if not isinstance(reference, dict):
                raise ValueError(f"{label}.reference: ожидается объект или null")
            _require_exact_keys(reference, REFERENCE_KEYS, f"{label}.reference")
            if not isinstance(reference["path"], str) or not reference["path"]:
                raise ValueError(f"{label}.reference.path: ожидается непустая строка")
            _require_hash(reference["sha256"], f"{label}.reference.sha256")
            if reference["format"] != "hypescribe_markdown":
                raise ValueError(f"{label}.reference.format не поддерживается")
        words = recording["asr_words"]
        if not isinstance(words, dict):
            raise TypeError(f"{label}.asr_words: ожидается объект")
        _require_exact_keys(words, ASR_WORDS_KEYS, f"{label}.asr_words")
        if not isinstance(words["path"], str) or not words["path"]:
            raise ValueError(f"{label}.asr_words.path: ожидается непустая строка")
        _require_hash(words["sha256"], f"{label}.asr_words.sha256")
        if words["format"] != "local_transcriber_words_v1":
            raise ValueError(f"{label}.asr_words.format не поддерживается")
        recording_families.add(_recording_family(recording["id"]))
        if verify_files:
            _verify_file(Path(recording["path"]), recording["sha256"], None, label)
            if reference is not None:
                _verify_file(
                    Path(reference["path"]),
                    reference["sha256"],
                    None,
                    f"{label}.reference",
                )
            _verify_file(
                Path(words["path"]), words["sha256"], None, f"{label}.asr_words"
            )
            load_asr_words(Path(words["path"]), float(recording["duration"]))
    if len(manifest["recordings"]) != 3 or recording_families != {
        "data-test",
        "t2-bdma",
        "yantar",
    }:
        raise ValueError("Нужны ровно три исходные записи: Data Test, T2 BDMA и Yantar")
    return manifest


def validate_original_corpus(manifest: dict[str, Any]) -> None:
    """Не допускает подмену корпуса issue #25 другой локальной выборкой."""
    for recording in manifest["recordings"]:
        family = _recording_family(recording["id"])
        expected = ORIGINAL_CORPUS.get(family)
        if expected is None or any(
            recording[key] != value for key, value in expected.items()
        ):
            raise ValueError(
                "Корпус не совпадает с исходными Data Test, T2 BDMA и Yantar"
            )
        reference_expected = family in {"t2-bdma", "yantar"}
        if (recording["reference"] is not None) != reference_expected:
            raise ValueError("Опорная разметка корпуса не совпадает с исходной")


def _validate_artifact(item: dict[str, Any], label: str, verify_files: bool) -> None:
    _require_hash(item["sha256"], f"{label}.sha256")
    size = item["size_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"{label}.size_bytes должен быть положительным целым")
    for key in ("path", "source_url", "license"):
        if not isinstance(item[key], str) or not item[key]:
            raise ValueError(f"{label}.{key}: ожидается непустая строка")
    if verify_files:
        _verify_file(Path(item["path"]), item["sha256"], size, label)


def _verify_file(
    path: Path, expected_hash: str, expected_size: int | None, label: str
) -> None:
    if not path.is_file():
        raise ValueError(f"{label}: файл не найден: {path}")
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ValueError(f"{label}: размер файла не совпал")
    if file_sha256(path) != expected_hash:
        raise ValueError(f"{label}: SHA-256 не совпал")


def environment_snapshot() -> dict[str, Any]:
    """Возвращает версии, которые влияют на семантику эксперимента."""

    def version(distribution: str) -> str:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return "unavailable"

    return {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count() or 1,
        "python_version": platform.python_version(),
        "sherpa_onnx_version": version("sherpa-onnx"),
        "onnxruntime_version": version("onnxruntime"),
        "numpy_version": version("numpy"),
        "psutil_version": version("psutil"),
    }


def validate_environment(environment: dict[str, Any]) -> None:
    """Отсекает окружение, которое заведомо испортит дорогой прогон."""
    missing = [
        key
        for key in (
            "sherpa_onnx_version",
            "onnxruntime_version",
            "numpy_version",
            "psutil_version",
        )
        if environment.get(key) == "unavailable"
    ]
    if missing:
        raise RuntimeError(f"Не установлены зависимости benchmark: {missing}")
    if environment.get("sherpa_onnx_version") != EXPECTED_SHERPA_ONNX_VERSION:
        raise RuntimeError(
            "Stage timing поддержан только для sherpa-onnx "
            f"{EXPECTED_SHERPA_ONNX_VERSION}"
        )


def semantic_manifest(
    manifest: dict[str, Any], environment: dict[str, Any]
) -> dict[str, Any]:
    """Исключает пути и описательные метаданные из идентичности опыта."""
    return {
        "schema_version": manifest["schema_version"],
        "segmentation_models": [
            {key: item[key] for key in ("id", "precision", "sha256", "size_bytes")}
            for item in manifest["segmentation_models"]
        ],
        "embedding_models": [
            {key: item[key] for key in ("id", "sha256", "size_bytes")}
            for item in manifest["embedding_models"]
        ],
        "combinations": manifest["combinations"],
        "recordings": [
            {
                "id": item["id"],
                "sha256": item["sha256"],
                "start": item["start"],
                "duration": item["duration"],
                "expected_speakers": item["expected_speakers"],
                "reference_sha256": item["reference"]["sha256"]
                if item["reference"]
                else None,
                "asr_words_sha256": item["asr_words"]["sha256"],
            }
            for item in manifest["recordings"]
        ],
        "threads": manifest["threads"],
        "rss_sample_interval_ms": manifest["rss_sample_interval_ms"],
        "decode": {
            "decoder_id": DECODER_ID,
            "channels": CHANNELS,
            "sample_rate": SAMPLE_RATE,
            "sample_width_bytes": SAMPLE_WIDTH_BYTES,
            "codec": "pcm_s16le",
        },
        "dependencies": {
            key: environment[key]
            for key in ("sherpa_onnx_version", "onnxruntime_version", "numpy_version")
        },
    }


def make_experiment_id(manifest: dict[str, Any], environment: dict[str, Any]) -> str:
    return _canonical_sha256(semantic_manifest(manifest, environment))


def make_cell_id(
    experiment_id: str,
    combination_id: str,
    recording_id: str,
    mode: str,
    threshold: float,
    num_clusters: int,
) -> str:
    return _canonical_sha256(
        {
            "experiment_id": experiment_id,
            "combination_id": combination_id,
            "recording_id": recording_id,
            "mode": mode,
            "threshold": threshold,
            "num_clusters": num_clusters,
        }
    )


def _cell_spec(
    experiment_id: str,
    combination_id: str,
    recording_id: str,
    mode: str,
    threshold: float,
    num_clusters: int,
) -> dict[str, Any]:
    return {
        "cell_id": make_cell_id(
            experiment_id, combination_id, recording_id, mode, threshold, num_clusters
        ),
        "combination_id": combination_id,
        "recording_id": recording_id,
        "mode": mode,
        "threshold": threshold,
        "num_clusters": num_clusters,
    }


def expand_automatic_cells(
    manifest: dict[str, Any], experiment_id: str
) -> list[dict[str, Any]]:
    """Разворачивает прямоугольный свип в стабильном порядке manifest."""
    return [
        _cell_spec(
            experiment_id,
            combination["id"],
            recording["id"],
            "automatic",
            float(threshold),
            -1,
        )
        for combination in manifest["combinations"]
        for threshold in combination["thresholds"]
        for recording in manifest["recordings"]
    ]


def derive_known_cells(
    manifest: dict[str, Any], experiment_id: str, selection: dict[str, Any]
) -> list[dict[str, Any]]:
    """Создаёт known-count ячейки с единым выбранным порогом."""
    return [
        _cell_spec(
            experiment_id,
            selection["combination_id"],
            recording["id"],
            "known",
            selection["selected_threshold"],
            recording["expected_speakers"],
        )
        for recording in manifest["recordings"]
    ]


def select_threshold(
    combination: dict[str, Any],
    recordings: list[dict[str, Any]],
    cells: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Выбирает один порог только после полного прямоугольного свипа."""
    expected = {item["id"]: item["expected_speakers"] for item in recordings}
    cell_by_key = {
        (cell["recording_id"], cell["threshold"]): cell
        for cell in cells
        if cell["combination_id"] == combination["id"]
        and cell["mode"] == "automatic"
        and cell["status"] == "complete"
    }
    reference_ids = {item["id"] for item in recordings if item["reference"] is not None}
    rectangles = []
    for threshold in combination["thresholds"]:
        rectangle = [
            cell_by_key.get((item["id"], float(threshold))) for item in recordings
        ]
        if any(cell is None for cell in rectangle):
            return None
        rectangles.append((float(threshold), rectangle))

    candidates = []
    for threshold, rectangle in rectangles:
        purities = []
        differences = []
        residuals = []
        for cell in rectangle:
            assert cell is not None
            differences.append(
                abs(cell["diagnostics"]["clusters"] - expected[cell["recording_id"]])
            )
            residuals.append(
                cell["diagnostics"]["residual_share_after_expected"] or 0.0
            )
            if cell["recording_id"] in reference_ids:
                purity = cell["diagnostics"]["mapped_speaker_purity"]
                if not isinstance(purity, (int, float)) or not math.isfinite(purity):
                    raise RuntimeError(
                        "Полный автоматический свип непригоден для выбора порога: "
                        f"нет mapped speaker purity для {cell['recording_id']} "
                        f"при threshold={threshold}"
                    )
                purities.append(float(purity))
        score: list[float | int] = [max(differences), sum(differences)]
        if purities:
            score.extend([-min(purities), -(sum(purities) / len(purities))])
        score.extend([sum(residuals), threshold])
        candidates.append((tuple(score), threshold, rectangle, differences))
    score, threshold, rectangle, differences = min(candidates, key=lambda item: item[0])
    return {
        "combination_id": combination["id"],
        "selected_threshold": threshold,
        "meets_count_gate": all(value == 0 for value in differences),
        "score": list(score),
        "recording_cells": [cell["cell_id"] for cell in rectangle if cell is not None],
    }


def parse_stage_timings(stderr: str, sherpa_version: str) -> dict[str, float]:
    """Строго разбирает недокументированный debug-контракт sherpa 1.13.6."""
    if sherpa_version != EXPECTED_SHERPA_ONNX_VERSION:
        raise ValueError(
            f"Неизвестный формат stage timing sherpa-onnx {sherpa_version}"
        )
    found: defaultdict[str, list[float]] = defaultdict(list)
    totals: list[tuple[float, float, float]] = []
    marker = "OfflineSpeakerDiarization:"
    for line in stderr.splitlines():
        if marker not in line:
            continue
        message = line[line.index(marker) :].strip()
        stage_match = STAGE_RE.fullmatch(message)
        if stage_match is not None:
            found[stage_match.group(1)].append(float(stage_match.group(2)))
            continue
        total_match = TOTAL_RE.fullmatch(message)
        if total_match is not None:
            totals.append(tuple(float(value) for value in total_match.groups()))
            continue
        raise ValueError("Stage timing содержит повреждённую debug-строку")
    required = ("segmentation", "embedding", "clustering")
    if any(len(found[name]) != 1 for name in required) or len(totals) != 1:
        raise ValueError(
            "Debug-лог должен содержать по одной записи каждого этапа и total"
        )
    total, audio_duration, logged_rtf = totals[0]
    values = [found[name][0] for name in required] + [total]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Stage timing содержит отрицательное или нечисловое значение")
    if (
        not math.isfinite(audio_duration)
        or audio_duration <= 0
        or not math.isfinite(logged_rtf)
        or logged_rtf < 0
    ):
        raise ValueError("Stage timing содержит недопустимые audio/RTF значения")
    if abs(sum(values[:3]) - values[3]) > max(0.05, values[3] * 0.05):
        raise ValueError("Сумма этапов существенно расходится с engine total")
    calculated_rtf = total / audio_duration
    if abs(calculated_rtf - logged_rtf) > max(0.005, logged_rtf * 0.05):
        raise ValueError("Stage timing содержит несогласованный RTF")
    return {
        "segmentation_seconds": values[0],
        "embedding_seconds": values[1],
        "clustering_seconds": values[2],
        "engine_total_seconds": values[3],
        "engine_audio_seconds": audio_duration,
        "engine_rtf": logged_rtf,
    }


def decode_cache_key(recording: dict[str, Any], decoder_id: str = DECODER_ID) -> str:
    return _canonical_sha256(
        {
            "source_sha256": recording["sha256"],
            "start": recording["start"],
            "duration": recording["duration"],
            "channels": CHANNELS,
            "sample_rate": SAMPLE_RATE,
            "sample_width_bytes": SAMPLE_WIDTH_BYTES,
            "codec": "pcm_s16le",
            "decoder_id": decoder_id,
        }
    )


def decode_clip(recording: dict[str, Any], work_dir: Path) -> Path:
    """Декодирует фрагмент один раз в контентно-адресуемый WAV-кэш."""
    cache_dir = work_dir / "decoded"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / f"{decode_cache_key(recording)}.wav"
    if output.exists():
        read_wav(output)
        return output
    temporary = output.with_suffix(".wav.tmp")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(recording["start"]),
        "-t",
        str(recording["duration"]),
        "-i",
        str(recording["path"]),
        "-vn",
        "-ac",
        str(CHANNELS),
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        read_wav(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def read_wav(path: Path) -> Any:
    import numpy as np

    with wave.open(str(path), "rb") as source:
        if (
            source.getnchannels() != CHANNELS
            or source.getsampwidth() != SAMPLE_WIDTH_BYTES
        ):
            raise ValueError(f"Ожидался mono PCM16 WAV: {path}")
        if source.getframerate() != SAMPLE_RATE:
            raise ValueError(f"Ожидалась частота 16 кГц: {path}")
        samples = np.frombuffer(source.readframes(source.getnframes()), np.int16)
    return samples.astype(np.float32) / 32768.0


def timestamp_seconds(match: re.Match[str]) -> float:
    first, second, third = match.group(1), match.group(2), match.group(3)
    return (
        int(first) * 60 + int(second)
        if third is None
        else int(first) * 3600 + int(second) * 60 + int(third)
    )


def read_reference_turns(recording: dict[str, Any]) -> list[dict[str, Any]]:
    reference = Path(recording["reference"]["path"]) if recording["reference"] else None
    start, duration = float(recording["start"]), float(recording["duration"])
    if reference is None:
        return []
    starts = []
    for line in reference.read_text(encoding="utf-8").splitlines():
        match = TURN_RE.match(line)
        if match:
            starts.append((timestamp_seconds(match), match.group(4)))
    clip_end = start + duration
    turns = []
    for index, (turn_start, speaker) in enumerate(starts):
        turn_end = starts[index + 1][0] if index + 1 < len(starts) else clip_end
        overlap_start, overlap_end = max(turn_start, start), min(turn_end, clip_end)
        if overlap_end > overlap_start:
            turns.append(
                {
                    "speaker": speaker,
                    "start": overlap_start - start,
                    "end": overlap_end - start,
                }
            )
    return turns


def interval_overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
    return max(0.0, min(left["end"], right["end"]) - max(left["start"], right["start"]))


def best_mapping(
    segments: list[dict[str, Any]], reference_turns: list[dict[str, Any]]
) -> dict[str, Any] | None:
    if not reference_turns or not segments:
        return None
    predicted = sorted({str(segment["speaker"]) for segment in segments})
    reference = sorted({str(turn["speaker"]) for turn in reference_turns})
    overlap: defaultdict[tuple[str, str], float] = defaultdict(float)
    total = 0.0
    for segment in segments:
        for turn in reference_turns:
            value = interval_overlap(segment, turn)
            if value:
                overlap[(str(segment["speaker"]), str(turn["speaker"]))] += value
                total += value
    best_score, best_pairs = -1.0, []
    if len(predicted) >= len(reference):
        candidates = (
            list(zip(candidate, reference, strict=True))
            for candidate in itertools.permutations(predicted, len(reference))
        )
    else:
        candidates = (
            list(zip(predicted, candidate, strict=True))
            for candidate in itertools.permutations(reference, len(predicted))
        )
    for pairs in candidates:
        score = sum(overlap[pair] for pair in pairs)
        if score > best_score:
            best_score, best_pairs = score, pairs
    return {
        "mapped_speaker_purity": best_score / total if total else None,
        "mapped_overlap_seconds": best_score,
        "total_overlap_seconds": total,
        "mapping": {
            predicted_id: reference_id for predicted_id, reference_id in best_pairs
        },
    }


def summarize_segments(
    segments: list[dict[str, Any]], recording: dict[str, Any]
) -> dict[str, Any]:
    duration = float(recording["duration"])
    expected = int(recording["expected_speakers"])
    durations: defaultdict[str, float] = defaultdict(float)
    for segment in segments:
        durations[str(segment["speaker"])] += max(
            0.0, float(segment["end"]) - float(segment["start"])
        )
    ordered = sorted(durations.items(), key=lambda item: (-item[1], item[0]))
    total = sum(durations.values())
    residual = sum(value for _, value in ordered[expected:])
    substantial_threshold = max(5.0, duration * 0.02)
    return {
        "clusters": len(ordered),
        "substantial_clusters": sum(
            value >= substantial_threshold for _, value in ordered
        ),
        "substantial_threshold_seconds": substantial_threshold,
        "cluster_durations_seconds": dict(ordered),
        "speaker_time_seconds": total,
        "residual_seconds_after_expected": residual,
        "residual_share_after_expected": residual / total if total else None,
    }


def load_asr_words(path: Path, duration: float) -> list[Any]:
    """Проверяет приватный sidecar и возвращает доменные слова."""
    from local_transcriber.types import Word

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError("ASR sidecar должен быть JSON-массивом")
    if not raw:
        raise ValueError("ASR sidecar должен содержать хотя бы одно слово")
    words, previous_start = [], -math.inf
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != {"start", "end", "text"}:
            raise ValueError(f"ASR sidecar[{index}]: неверная структура")
        start = _require_finite(item["start"], f"ASR sidecar[{index}].start")
        end = _require_finite(item["end"], f"ASR sidecar[{index}].end")
        if start < 0 or end < start or end > duration or start < previous_start:
            raise ValueError(f"ASR sidecar[{index}]: нарушены временные границы")
        if not isinstance(item["text"], str):
            raise TypeError(f"ASR sidecar[{index}].text: ожидается строка")
        words.append(Word(start=start, end=end, text=item["text"]))
        previous_start = start
    return words


def canonical_non_whitespace_sequence(words: list[Any]) -> str:
    """Сохраняет все Unicode-символы текста, кроме любых пробельных."""
    from local_transcriber.diarization import _append_word_text, _normalize_turn_text

    text = ""
    for word in words:
        text = _append_word_text(text, word.text)
    normalized = _normalize_turn_text(text)
    return "".join(character for character in normalized if not character.isspace())


def text_invariance(words: list[Any], transcript: Any) -> dict[str, Any]:
    from local_transcriber.diarization import _append_word_text, _normalize_turn_text

    source = canonical_non_whitespace_sequence(words)
    output = ""
    for turn in transcript.turns:
        output = _append_word_text(output, turn.text)
    output = _normalize_turn_text(output)
    output = "".join(character for character in output if not character.isspace())
    mismatch = next(
        (
            index
            for index, pair in enumerate(
                itertools.zip_longest(source, output, fillvalue=None)
            )
            if pair[0] != pair[1]
        ),
        None,
    )
    return {
        "input_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "equal": source == output,
        "first_mismatch_index": mismatch,
    }


def evaluate_segments(
    segments: list[dict[str, Any]],
    recording: dict[str, Any],
    words: list[Any],
    reference_turns: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from local_transcriber.diarization import build_speaker_transcript
    from local_transcriber.types import SpeakerInterval

    intervals = [
        SpeakerInterval(
            start=float(item["start"]),
            end=float(item["end"]),
            cluster=int(item["speaker"]),
        )
        for item in segments
    ]
    transcript = build_speaker_transcript(
        words, intervals, float(recording["duration"])
    )
    diagnostics = summarize_segments(segments, recording)
    mapping = best_mapping(segments, reference_turns)
    diagnostics.update(
        {
            "unassigned_word_count": transcript.unassigned_word_count,
            "mapped_speaker_purity": mapping["mapped_speaker_purity"]
            if mapping
            else None,
            "mapped_overlap_seconds": mapping["mapped_overlap_seconds"]
            if mapping
            else None,
            "total_overlap_seconds": mapping["total_overlap_seconds"]
            if mapping
            else None,
            "mapping": mapping["mapping"] if mapping else None,
        }
    )
    return diagnostics, text_invariance(words, transcript)


def measure_call(
    operation: Callable[[], Any],
    process: Any,
    *,
    clock: Callable[[], float] = time.perf_counter,
    sampler_factory: Callable[[Any, float], Any] = PeakRssSampler,
    interval_seconds: float,
    logical_cpus: int,
) -> tuple[Any, dict[str, Any]]:
    """Измеряет ровно один вызов через внедряемые CPU/RSS адаптеры."""
    sampler = sampler_factory(process, interval_seconds)
    sampler.start()
    before = process.cpu_times()
    started = clock()
    operation_error: BaseException | None = None
    try:
        result = operation()
    except BaseException as exc:
        operation_error = exc
        raise
    finally:
        stopped = clock()
        after = process.cpu_times()
        try:
            peak_rss = sampler.stop()
        except BaseException as exc:
            if operation_error is None:
                raise
            operation_error.add_note(
                f"Дополнительная ошибка остановки RSS sampler: {type(exc).__name__}: {exc}"
            )
    wall = stopped - started
    if wall <= 0:
        raise RuntimeError("Измеренный wall time должен быть положительным")
    cpu_user, cpu_system = (
        float(after.user - before.user),
        float(after.system - before.system),
    )
    if (
        not math.isfinite(cpu_user)
        or not math.isfinite(cpu_system)
        or cpu_user < 0
        or cpu_system < 0
        or logical_cpus <= 0
        or isinstance(peak_rss, bool)
        or not isinstance(peak_rss, int)
        or peak_rss <= 0
    ):
        raise RuntimeError("CPU/RSS-измерение вернуло недопустимое значение")
    cpu_total = cpu_user + cpu_system
    cores = cpu_total / wall
    return result, {
        "wall_seconds": wall,
        "cpu_user_seconds": cpu_user,
        "cpu_system_seconds": cpu_system,
        "cpu_total_seconds": cpu_total,
        "average_cpu_cores": cores,
        "average_cpu_percent_machine": 100.0 * cores / logical_cpus,
        "peak_rss_bytes": peak_rss,
    }


def make_config(
    segmentation_model: Path,
    embedding_model: Path,
    threshold: float,
    num_clusters: int,
    threads: int,
) -> Any:
    """Создаёт строго CPU-конфигурацию и включает native stage timing."""
    import sherpa_onnx

    segmentation = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
        pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
            model=str(segmentation_model)
        ),
        num_threads=threads,
        provider="cpu",
        debug=True,
    )
    embedding = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(embedding_model), num_threads=threads, provider="cpu", debug=True
    )
    return sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=segmentation,
        embedding=embedding,
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_clusters, threshold=threshold
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )


def _worker_main(request_path: Path) -> None:
    """Исполняет ровно одну ячейку в свежем процессе."""
    import psutil
    import sherpa_onnx

    request = json.loads(request_path.read_text(encoding="utf-8"))
    samples = read_wav(Path(request["wav_path"]))
    config = make_config(
        Path(request["segmentation_path"]),
        Path(request["embedding_path"]),
        request["threshold"],
        request["num_clusters"],
        request["threads"],
    )
    if not config.validate():
        raise RuntimeError("Конфигурация диаризатора недействительна")
    diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
    result, metrics = measure_call(
        lambda: diarizer.process(samples),
        psutil.Process(os.getpid()),
        interval_seconds=request["rss_sample_interval_ms"] / 1000.0,
        logical_cpus=request["logical_cpus"],
    )
    segments = [
        {
            "speaker": int(item.speaker),
            "start": float(item.start),
            "end": float(item.end),
        }
        for item in result.sort_by_start_time()
    ]
    print(
        json.dumps({"metrics": metrics, "segments": segments}, separators=(",", ":")),
        flush=True,
    )


def run_worker(
    request: dict[str, Any],
    work_dir: Path,
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> tuple[dict[str, Any], str]:
    """Запускает изолированного worker; seam допускает fake protocol в тестах."""
    request_dir = work_dir / "requests"
    request_dir.mkdir(parents=True, exist_ok=True)
    request_path = request_dir / f"{request['cell_id']}.json"
    save_output(request_path, request)
    completed = runner(
        [sys.executable, str(Path(__file__).resolve()), "--worker", str(request_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = (
            completed.stderr.strip().splitlines()[-1]
            if completed.stderr.strip()
            else "worker завершился с ошибкой"
        )
        raise RuntimeError(message)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Worker вернул некорректный JSON") from exc
    if set(payload) != {"metrics", "segments"}:
        raise RuntimeError("Worker нарушил протокол результата")
    return payload, completed.stderr


def save_output(path: Path, output: Any) -> None:
    """Атомарно сохраняет JSON рядом с целевым файлом."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(output, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _artifacts(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for kind, models in (
        ("segmentation", manifest["segmentation_models"]),
        ("embedding", manifest["embedding_models"]),
    ):
        for item in models:
            result.append(
                {
                    "kind": kind,
                    "id": item["id"],
                    "filename": Path(item["path"]).name,
                    "sha256": item["sha256"],
                    "size_bytes": item["size_bytes"],
                    "source_url": item["source_url"],
                    "license": item["license"],
                }
            )
    return result


def _asr_baselines(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    baselines = []
    for recording in manifest["recordings"]:
        words = load_asr_words(
            Path(recording["asr_words"]["path"]), recording["duration"]
        )
        baselines.append(
            {
                "recording_id": recording["id"],
                "words_sha256": recording["asr_words"]["sha256"],
                "word_count": len(words),
                "canonical_text_sha256": hashlib.sha256(
                    canonical_non_whitespace_sequence(words).encode()
                ).hexdigest(),
            }
        )
    return baselines


def _new_output(
    manifest: dict[str, Any], environment: dict[str, Any], experiment_id: str
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "manifest": manifest,
        "environment": environment,
        "artifacts": _artifacts(manifest),
        "asr_baselines": _asr_baselines(manifest),
        "cells": [],
        "threshold_selection": [],
    }


def _load_or_create_output(
    path: Path,
    manifest: dict[str, Any],
    environment: dict[str, Any],
    experiment_id: str,
) -> dict[str, Any]:
    if not path.exists():
        return _new_output(manifest, environment, experiment_id)
    output = json.loads(path.read_text(encoding="utf-8"))
    if (
        output.get("schema_version") != SCHEMA_VERSION
        or output.get("experiment_id") != experiment_id
    ):
        raise ValueError(
            "Существующий output относится к другому семантическому эксперименту"
        )
    host_keys = ("platform", "cpu", "logical_cpus")
    previous_environment = output.get("environment")
    if not isinstance(previous_environment, dict) or any(
        previous_environment.get(key) != environment.get(key) for key in host_keys
    ):
        raise ValueError(
            "Существующий output создан на другой platform/cpu/logical_cpus"
        )
    output["manifest"] = manifest
    output["artifacts"] = _artifacts(manifest)
    output["asr_baselines"] = _asr_baselines(manifest)
    return output


def _replace_cell(output: dict[str, Any], cell: dict[str, Any]) -> None:
    output["cells"] = [
        item for item in output["cells"] if item.get("cell_id") != cell["cell_id"]
    ]
    output["cells"].append(cell)


def _failed_cell(spec: dict[str, Any], exc: BaseException) -> dict[str, Any]:
    return {
        **spec,
        "status": "failed",
        "metrics": None,
        "diagnostics": None,
        "text_invariance": None,
        "segments": None,
        "error": {"type": type(exc).__name__, "message": str(exc)},
    }


def _execute_cell(
    spec: dict[str, Any],
    manifest: dict[str, Any],
    output: dict[str, Any],
    decoded: dict[str, Path],
    work_dir: Path,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    combinations = {item["id"]: item for item in manifest["combinations"]}
    segmentations = {item["id"]: item for item in manifest["segmentation_models"]}
    embeddings = {item["id"]: item for item in manifest["embedding_models"]}
    recordings = {item["id"]: item for item in manifest["recordings"]}
    combination, recording = (
        combinations[spec["combination_id"]],
        recordings[spec["recording_id"]],
    )
    request = {
        **spec,
        "wav_path": str(decoded[recording["id"]]),
        "segmentation_path": segmentations[combination["segmentation_id"]]["path"],
        "embedding_path": embeddings[combination["embedding_id"]]["path"],
        "threads": manifest["threads"],
        "rss_sample_interval_ms": manifest["rss_sample_interval_ms"],
        "logical_cpus": output["environment"]["logical_cpus"],
    }
    payload, stderr = run_worker(request, work_dir, runner=runner)
    stages = parse_stage_timings(stderr, output["environment"]["sherpa_onnx_version"])
    if not math.isclose(
        stages["engine_audio_seconds"],
        float(recording["duration"]),
        rel_tol=0.0,
        abs_tol=AUDIO_DURATION_TOLERANCE_SECONDS,
    ):
        raise RuntimeError(
            "Длительность audio из sherpa debug-лога не совпала с manifest clip: "
            f"engine_audio_seconds={stages['engine_audio_seconds']}, "
            f"manifest_duration={recording['duration']}"
        )
    metrics = payload["metrics"]
    metrics["rtf"] = metrics["wall_seconds"] / recording["duration"]
    metrics["stages"] = stages
    words = load_asr_words(Path(recording["asr_words"]["path"]), recording["duration"])
    diagnostics, invariance = evaluate_segments(
        payload["segments"], recording, words, read_reference_turns(recording)
    )
    if not invariance["equal"]:
        raise RuntimeError(
            "ASR-текст изменился после сведения: "
            f"input_sha256={invariance['input_sha256']}, "
            f"output_sha256={invariance['output_sha256']}, "
            f"first_mismatch_index={invariance['first_mismatch_index']}"
        )
    return {
        **spec,
        "status": "complete",
        "metrics": metrics,
        "diagnostics": diagnostics,
        "text_invariance": invariance,
        "segments": payload["segments"],
        "error": None,
    }


def _run_specs(
    specs: list[dict[str, Any]],
    manifest: dict[str, Any],
    output: dict[str, Any],
    decoded: dict[str, Path],
    output_path: Path,
    work_dir: Path,
) -> None:
    complete = {
        cell["cell_id"] for cell in output["cells"] if cell.get("status") == "complete"
    }
    for spec in specs:
        if spec["cell_id"] in complete:
            continue
        try:
            cell = _execute_cell(spec, manifest, output, decoded, work_dir)
        except Exception as exc:  # noqa: BLE001 - ошибка ячейки должна сохраниться для retry
            cell = _failed_cell(spec, exc)
        _replace_cell(output, cell)
        save_output(output_path, output)


def generate_asr_sidecar(
    wav_path: Path,
    output_path: Path,
    *,
    model: str,
    device: str,
    compute_type: str,
    language: str,
    threads: int,
) -> str:
    """Один раз запускает production ASR и атомарно сохраняет только слова."""
    from local_transcriber.transcriber import transcribe

    result = transcribe(
        wav_path,
        model_name=model,
        device=device,
        compute_type=compute_type,
        language=language,
        strict_device=True,
        cpu_threads=threads,
    )
    save_output(
        output_path,
        [
            {"start": word.start, "end": word.end, "text": word.text}
            for word in result.words
        ],
    )
    return file_sha256(output_path)


def _prepare_asr(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    validate_manifest(manifest, verify_files=False)
    validate_original_corpus(manifest)
    recordings = {item["id"]: item for item in manifest["recordings"]}
    if args.prepare_asr not in recordings:
        raise ValueError("Неизвестный recording id для --prepare-asr")
    recording = recordings[args.prepare_asr]
    _verify_file(Path(recording["path"]), recording["sha256"], None, "recording")
    digest = generate_asr_sidecar(
        decode_clip(recording, args.work_dir),
        Path(recording["asr_words"]["path"]),
        model=args.asr_model,
        device=args.asr_device,
        compute_type=args.asr_compute_type,
        language=args.asr_language,
        threads=args.threads or manifest["threads"],
    )
    print(digest)


def validation_summary(
    manifest: dict[str, Any], environment: dict[str, Any]
) -> dict[str, Any]:
    """Возвращает безопасный план без приватных путей и текста."""
    experiment_id = make_experiment_id(manifest, environment)
    return {
        "status": "valid",
        "experiment_id": experiment_id,
        "recordings": [item["id"] for item in manifest["recordings"]],
        "combinations": [item["id"] for item in manifest["combinations"]],
        "automatic_cells": len(expand_automatic_cells(manifest, experiment_id)),
        "known_cells_if_all_selected": len(manifest["recordings"])
        * len(manifest["combinations"]),
        "threads": manifest["threads"],
        "dependencies": {
            key: environment[key]
            for key in (
                "sherpa_onnx_version",
                "onnxruntime_version",
                "numpy_version",
                "psutil_version",
            )
        },
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.worker is not None:
        request = json.loads(args.worker.read_text(encoding="utf-8"))
        if request.get("worker_schema_version") == 3:
            from local_transcriber.benchmark_experiment import worker_main

            worker_main(args.worker)
            return
        _worker_main(args.worker)
        return
    raw_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if raw_manifest.get("schema_version") == 3:
        from local_transcriber.benchmark_experiment import run_cli

        run_cli(args, raw_manifest)
        return
    validate_manifest(raw_manifest, verify_files=False)
    manifest = resolve_manifest_paths(raw_manifest, args.manifest.parent)
    if args.threads is not None:
        if args.threads <= 0:
            raise ValueError("--threads должен быть положительным")
        if args.threads != manifest.get("threads"):
            raise ValueError("--threads должен совпадать с manifest.threads")
    if args.prepare_asr is not None:
        _prepare_asr(args, manifest)
        return
    validate_manifest(manifest)
    validate_original_corpus(manifest)
    environment = environment_snapshot()
    validate_environment(environment)
    if args.validate_only:
        print(
            json.dumps(
                validation_summary(manifest, environment),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    args.work_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = make_experiment_id(manifest, environment)
    output = _load_or_create_output(args.output, manifest, environment, experiment_id)
    decoded = {
        recording["id"]: decode_clip(recording, args.work_dir)
        for recording in manifest["recordings"]
    }
    save_output(args.output, output)
    automatic_specs = expand_automatic_cells(manifest, experiment_id)
    _run_specs(
        automatic_specs,
        manifest,
        output,
        decoded,
        args.output,
        args.work_dir,
    )
    expected_specs = list(automatic_specs)
    for combination in manifest["combinations"]:
        selection = select_threshold(
            combination, manifest["recordings"], output["cells"]
        )
        if selection is None:
            continue
        output["threshold_selection"] = [
            item
            for item in output["threshold_selection"]
            if item["combination_id"] != combination["id"]
        ] + [selection]
        save_output(args.output, output)
        known_specs = derive_known_cells(manifest, experiment_id, selection)
        expected_specs.extend(known_specs)
        _run_specs(
            known_specs,
            manifest,
            output,
            decoded,
            args.output,
            args.work_dir,
        )
    status_by_id = {cell.get("cell_id"): cell.get("status") for cell in output["cells"]}
    incomplete = [
        spec["cell_id"]
        for spec in expected_specs
        if status_by_id.get(spec["cell_id"]) != "complete"
    ]
    if incomplete:
        raise RuntimeError(
            f"Benchmark не завершён: незавершённых ячеек {len(incomplete)}"
        )


if __name__ == "__main__":
    main()
