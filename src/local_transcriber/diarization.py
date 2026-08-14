"""Сведение слов с временной привязкой и разметки говорящих."""

from collections import defaultdict
from math import isclose
from unicodedata import category

from .types import (
    SmallSpeakerCluster,
    SpeakerInterval,
    SpeakerTranscript,
    SpeakerTurn,
    Word,
)

_PAUSE_THRESHOLD_S = 2.0
_MAX_TURN_S = 60.0


def build_speaker_transcript(
    words: list[Word],
    intervals: list[SpeakerInterval],
    recording_duration: float,
) -> SpeakerTranscript:
    """Назначает словам говорящих и собирает линейные реплики."""
    cluster_numbers: dict[int, int] = {}
    assigned: list[tuple[Word, int | None]] = []
    unassigned = 0

    for word in words:
        speaker_cluster = _assign_cluster(word, intervals)
        if speaker_cluster is None:
            speaker = None
            unassigned += 1
        else:
            speaker = cluster_numbers.setdefault(
                speaker_cluster,
                len(cluster_numbers) + 1,
            )
        assigned.append((word, speaker))

    return SpeakerTranscript(
        turns=_group_words(assigned),
        cluster_count=len({interval.cluster for interval in intervals}),
        unassigned_word_count=unassigned,
        small_clusters=_find_small_clusters(
            intervals,
            cluster_numbers,
            recording_duration,
        ),
    )


def _assign_cluster(word: Word, intervals: list[SpeakerInterval]) -> int | None:
    overlaps: defaultdict[int, float] = defaultdict(float)
    for interval in intervals:
        overlap = min(word.end, interval.end) - max(word.start, interval.start)
        if overlap > 0:
            overlaps[interval.cluster] += overlap

    if not overlaps:
        return None
    largest = max(overlaps.values())
    winners = [
        cluster
        for cluster, overlap in overlaps.items()
        if isclose(overlap, largest, rel_tol=1e-9, abs_tol=1e-9)
    ]
    return winners[0] if len(winners) == 1 else None


def _group_words(assigned: list[tuple[Word, int | None]]) -> list[SpeakerTurn]:
    if not assigned:
        return []

    turns: list[SpeakerTurn] = []
    first_word, current_speaker = assigned[0]
    start = first_word.start
    end = first_word.end
    text = first_word.text

    for word, speaker in assigned[1:]:
        should_split = (
            speaker != current_speaker
            or word.start - end >= _PAUSE_THRESHOLD_S
            or word.end - start > _MAX_TURN_S
        )
        if should_split:
            turns.append(
                SpeakerTurn(start, end, _normalize_turn_text(text), current_speaker)
            )
            start = word.start
            text = word.text
            current_speaker = speaker
        else:
            text = _append_word_text(text, word.text)
        end = word.end

    turns.append(SpeakerTurn(start, end, _normalize_turn_text(text), current_speaker))
    return turns


def _append_word_text(current: str, word_text: str) -> str:
    if not current or not word_text or word_text[:1].isspace():
        return current + word_text
    if category(word_text[0])[:1] in {"P", "S"}:
        return current + word_text
    return f"{current} {word_text}"


def _normalize_turn_text(text: str) -> str:
    return " ".join(text.split())


def _find_small_clusters(
    intervals: list[SpeakerInterval],
    cluster_numbers: dict[int, int],
    recording_duration: float,
) -> list[SmallSpeakerCluster]:
    durations: defaultdict[int, float] = defaultdict(float)
    for interval in intervals:
        durations[interval.cluster] += max(0.0, interval.end - interval.start)

    threshold = max(5.0, recording_duration * 0.02)
    return [
        SmallSpeakerCluster(
            speaker=cluster_numbers.get(cluster),
            duration=duration,
        )
        for cluster, duration in durations.items()
        if duration < threshold
    ]
