import pytest

from local_transcriber.quality import (
    REPETITION_MIN_LEN,
    TAIL_GAP_WARN_S,
    _normalize,
    find_repetition_blocks,
    tail_gap,
)
from local_transcriber.types import Segment, TranscribeResult


def _result(segments, duration):
    return TranscribeResult(
        segments=segments,
        language="ru",
        language_probability=0.95,
        duration=duration,
        device_used="cpu",
    )


def _segments(texts, start=0.0):
    return [
        Segment(start=start + index, end=start + index + 1.0, text=text)
        for index, text in enumerate(texts)
    ]


def test_tail_gap_returns_positive_gap():
    result = _result([Segment(0.0, 10.0, "Текст")], duration=42.0)

    assert tail_gap(result) == 32.0


def test_tail_gap_empty_segments_returns_zero():
    assert tail_gap(_result([], duration=42.0)) == 0.0


def test_tail_gap_negative_gap_returns_zero():
    result = _result([Segment(0.0, 43.0, "Текст")], duration=42.0)

    assert tail_gap(result) == 0.0


@pytest.mark.parametrize("gap", [119.99, 120.0])
def test_tail_gap_boundary_does_not_warn(gap):
    result = _result([Segment(0.0, 10.0, "Текст")], duration=10.0 + gap)

    assert tail_gap(result) <= TAIL_GAP_WARN_S


def test_tail_gap_boundary_warns_above_threshold():
    result = _result([Segment(0.0, 10.0, "Текст")], duration=130.01)

    assert tail_gap(result) > TAIL_GAP_WARN_S


def test_normalize_removes_case_punctuation_and_collapses_spaces():
    assert _normalize('  «ПРИВЕТ…» — (мир) [тест] <да>  ') == "привет мир тест да"
    assert _normalize("раз–два-три: да; нет!") == "раздватри да нет"


def test_normalize_punctuation_only_returns_empty():
    assert _normalize('.,!?…:;—–-"\'«»()[]<>   ') == ""


def test_find_repetition_blocks_four_long_segments_with_normalization():
    segments = _segments(["Повтор!", "повтор", "«ПОВТОР»", "повтор…"])

    blocks = find_repetition_blocks(segments)

    assert len(blocks) == 1
    assert blocks[0].start == 0.0
    assert blocks[0].end == 4.0
    assert blocks[0].count == 4
    assert blocks[0].text == "Повтор!"


def test_find_repetition_blocks_three_long_segments_is_empty():
    assert find_repetition_blocks(_segments(["Повтор", "повтор", "повтор"])) == []


def test_find_repetition_blocks_length_boundary():
    assert find_repetition_blocks(_segments(["пять5"] * 4)) == []

    blocks = find_repetition_blocks(_segments(["шесть6"] * 4))

    assert len("шесть6") == REPETITION_MIN_LEN
    assert len(blocks) == 1


def test_find_repetition_blocks_short_text_boundary():
    assert find_repetition_blocks(_segments(["Ага."] * 9)) == []

    blocks = find_repetition_blocks(_segments(["Ага."] * 10))

    assert len(blocks) == 1
    assert blocks[0].count == 10


def test_find_repetition_blocks_empty_normalized_text_is_ignored():
    assert find_repetition_blocks(_segments([":", "", "  "] * 10)) == []


def test_find_repetition_blocks_two_separate_runs():
    segments = [
        *_segments(["Первый повтор"] * 4),
        Segment(10.0, 11.0, "Разрыв"),
        *_segments(["Второй повтор"] * 4, start=20.0),
    ]

    blocks = find_repetition_blocks(segments)

    assert len(blocks) == 2
    assert blocks[0].text == "Первый повтор"
    assert blocks[1].text == "Второй повтор"


def test_find_repetition_blocks_empty_list_is_empty():
    assert find_repetition_blocks([]) == []


def test_find_repetition_blocks_single_segment_is_empty():
    assert find_repetition_blocks([Segment(0.0, 1.0, "Повтор")]) == []
