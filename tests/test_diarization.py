from local_transcriber.diarization import build_speaker_transcript
from local_transcriber.types import (
    SmallSpeakerCluster,
    SpeakerInterval,
    SpeakerTurn,
    Word,
)


def test_build_speaker_transcript_assigns_and_groups_words():
    words = [
        Word(start=0.0, end=0.8, text="Добрый"),
        Word(start=0.8, end=1.4, text="день."),
        Word(start=1.5, end=2.1, text="Привет!"),
    ]
    intervals = [
        SpeakerInterval(start=0.0, end=1.4, cluster=7),
        SpeakerInterval(start=1.4, end=2.3, cluster=3),
    ]

    transcript = build_speaker_transcript(words, intervals, recording_duration=30.0)

    assert [
        (turn.speaker, turn.start, turn.end, turn.text) for turn in transcript.turns
    ] == [
        (1, 0.0, 1.4, "Добрый день."),
        (2, 1.5, 2.1, "Привет!"),
    ]
    assert transcript.cluster_count == 2
    assert transcript.unassigned_word_count == 0


def test_build_speaker_transcript_reports_small_cluster_without_filtering_it():
    words = [
        Word(start=0.0, end=1.0, text="Редкая реплика."),
        Word(start=5.0, end=6.0, text="Основная реплика."),
    ]
    intervals = [
        SpeakerInterval(start=0.0, end=4.9, cluster=4),
        SpeakerInterval(start=5.0, end=10.0, cluster=9),
    ]

    transcript = build_speaker_transcript(words, intervals, recording_duration=100.0)

    assert [turn.speaker for turn in transcript.turns] == [1, 2]
    assert transcript.small_clusters == [SmallSpeakerCluster(speaker=1, duration=4.9)]


def test_build_speaker_transcript_keeps_equal_overlap_unassigned():
    words = [Word(start=0.0, end=1.0, text="Спорное слово")]
    intervals = [
        SpeakerInterval(start=0.0, end=0.1, cluster=8),
        SpeakerInterval(start=0.3, end=0.5, cluster=8),
        SpeakerInterval(start=0.0, end=0.3, cluster=2),
    ]

    transcript = build_speaker_transcript(words, intervals, recording_duration=10.0)

    assert transcript.turns[0].speaker is None
    assert transcript.unassigned_word_count == 1


def test_build_speaker_transcript_keeps_word_without_overlap_unknown():
    transcript = build_speaker_transcript(
        [Word(start=5.0, end=6.0, text="Вне разметки")],
        [SpeakerInterval(start=0.0, end=1.0, cluster=1)],
        recording_duration=10.0,
    )

    assert transcript.turns == [SpeakerTurn(5.0, 6.0, "Вне разметки", None)]
    assert transcript.unassigned_word_count == 1


def test_build_speaker_transcript_splits_at_two_second_pause():
    transcript = build_speaker_transcript(
        [
            Word(0.0, 1.0, "До паузы."),
            Word(3.0, 4.0, "После паузы."),
        ],
        [SpeakerInterval(0.0, 4.0, 1)],
        recording_duration=10.0,
    )

    assert [turn.text for turn in transcript.turns] == [
        "До паузы.",
        "После паузы.",
    ]


def test_build_speaker_transcript_does_not_exceed_sixty_seconds():
    transcript = build_speaker_transcript(
        [
            Word(0.0, 30.0, "Начало."),
            Word(30.0, 60.0, "Продолжение."),
            Word(60.0, 61.0, "Новая реплика."),
        ],
        [SpeakerInterval(0.0, 61.0, 1)],
        recording_duration=70.0,
    )

    assert [turn.text for turn in transcript.turns] == [
        "Начало. Продолжение.",
        "Новая реплика.",
    ]


def test_build_speaker_transcript_preserves_punctuation_without_leading_space():
    transcript = build_speaker_transcript(
        [
            Word(0.0, 0.4, "Тарадата"),
            Word(0.4, 0.5, "+"),
            Word(0.5, 0.7, "Click"),
            Word(0.7, 0.8, "—"),
            Word(0.8, 1.0, "это"),
        ],
        [SpeakerInterval(0.0, 1.0, 1)],
        recording_duration=10.0,
    )

    assert transcript.turns[0].text == "Тарадата+ Click— это"
