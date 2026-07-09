"""Эвристики качества транскрипта."""

from dataclasses import dataclass

from .types import Segment, TranscribeResult

TAIL_GAP_WARN_S = 120.0
REPETITION_MIN_RUN = 4
REPETITION_MIN_RUN_SHORT = 10
REPETITION_MIN_LEN = 6

_PUNCTUATION_TO_REMOVE = ".,!?…:;—–-\"'«»()[]<>"
_REMOVE_PUNCTUATION = str.maketrans("", "", _PUNCTUATION_TO_REMOVE)


@dataclass
class RepetitionBlock:
    start: float
    end: float
    count: int
    text: str


def tail_gap(result: TranscribeResult) -> float:
    """Возвращает непокрытый хвост записи в секундах."""
    if not result.segments:
        return 0.0
    return max(0.0, result.duration - result.segments[-1].end)


def _normalize(text: str) -> str:
    """Нормализует текст сегмента для поиска межсегментных повторов."""
    text = text.casefold()
    text = text.translate(_REMOVE_PUNCTUATION)
    return " ".join(text.split())


def find_repetition_blocks(segments: list[Segment]) -> list[RepetitionBlock]:
    """Находит серии подряд идущих одинаковых сегментов."""
    blocks: list[RepetitionBlock] = []
    run_start = 0
    run_norm = ""

    def append_run(run_end: int) -> None:
        count = run_end - run_start
        if not run_norm:
            return
        min_run = (
            REPETITION_MIN_RUN
            if len(run_norm) >= REPETITION_MIN_LEN
            else REPETITION_MIN_RUN_SHORT
        )
        if count >= min_run:
            blocks.append(
                RepetitionBlock(
                    start=segments[run_start].start,
                    end=segments[run_end - 1].end,
                    count=count,
                    text=segments[run_start].text,
                )
            )

    for index, segment in enumerate(segments):
        norm = _normalize(segment.text)
        if index == 0:
            run_start = 0
            run_norm = norm
            continue
        if norm == run_norm:
            continue
        append_run(index)
        run_start = index
        run_norm = norm

    if segments:
        append_run(len(segments))

    return blocks
