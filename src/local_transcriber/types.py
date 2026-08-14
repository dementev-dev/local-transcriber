"""Общие типы данных для всех бэкендов транскрипции."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Единый признак «язык неизвестен» для всех бэкендов
UNKNOWN_LANGUAGE = "unknown"


class WordTimestampsUnavailableError(RuntimeError):
    """ASR распознал текст, но нарушил обязательный пословный контракт."""


@dataclass
class Segment:
    start: float  # seconds
    end: float  # seconds
    text: str


@dataclass(frozen=True)
class Word:
    """Слово с временной привязкой на шкале исходной записи."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class SpeakerInterval:
    """Интервал разметки говорящих с анонимным голосовым кластером."""

    start: float
    end: float
    cluster: int


@dataclass(frozen=True)
class SpeakerTurn:
    """Реплика говорящего; ``speaker=None`` означает неизвестного говорящего."""

    start: float
    end: float
    text: str
    speaker: int | None


@dataclass(frozen=True)
class SmallSpeakerCluster:
    """Малый голосовой кластер, о котором нужно предупредить пользователя."""

    speaker: int | None
    duration: float


@dataclass
class SpeakerTranscript:
    """Результат сведения слов с разметкой говорящих."""

    turns: list[SpeakerTurn]
    cluster_count: int
    unassigned_word_count: int
    small_clusters: list[SmallSpeakerCluster]


@dataclass
class DiarizationRun:
    """Разметка одного файла и длительность прохода диаризации."""

    intervals: list[SpeakerInterval]
    elapsed_seconds: float


@dataclass
class TranscribeResult:
    segments: list[Segment]
    language: str  # код языка или UNKNOWN_LANGUAGE, если он неизвестен
    language_probability: float
    duration: float  # seconds
    device_used: str  # "cpu" / "cuda" / "onnx" / "openvino-gpu" / "openvino-cpu"
    words: list[Word] = field(default_factory=list)


@dataclass
class TranscribeFileResult:
    result: TranscribeResult
    model: Any  # backend-specific model handle
    actual_device: str
    backend: Any = None  # backend instance (для переиспользования в батче)
    model_path: str = ""  # путь к модели (меняется при cross-backend fallback)


StatusCallback = Callable[[str], None] | None
SegmentCallback = Callable[[Segment], None] | None
