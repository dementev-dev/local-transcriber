"""Свип порога кластеризации и явного числа говорящих.

    uv run --with sherpa-onnx python .scratch/diarization/bench_sweep.py <файл> [потоки]

Каждая конфигурация — полный прогон сегментации и эмбеддингов (около 2,5 минут
на 26-минутную запись), поэтому свип имеет смысл вести на коротком фрагменте, а
полные записи оставить для проверки финального кандидата.
"""

from __future__ import annotations

import sys
import time

from common import DEFAULT_THREADS, SAMPLE_RATE, load_audio, make_diarizer

# подпись, num_clusters, threshold
CONFIGS = [
    ("авто, порог 0.5", -1, 0.5),
    ("авто, порог 0.7", -1, 0.7),
    ("авто, порог 0.9", -1, 0.9),
    ("явно k=5", 5, 0.5),
]

# говорящий с речью короче порога считается остаточным кластером, не участником
MIN_SPEAKER_S = 30.0


def main(audio_path: str, threads: int) -> None:
    samples = load_audio(audio_path)
    duration = len(samples) / SAMPLE_RATE
    print(f"файл: {audio_path}")
    print(f"длительность: {duration / 60:.1f} мин, потоков: {threads}\n")

    for label, num_clusters, threshold in CONFIGS:
        diarizer = make_diarizer(
            threshold=threshold, num_clusters=num_clusters, threads=threads
        )
        t0 = time.perf_counter()
        segments = diarizer.process(samples).sort_by_start_time()
        elapsed = time.perf_counter() - t0

        totals: dict[int, float] = {}
        for seg in segments:
            totals[seg.speaker] = totals.get(seg.speaker, 0.0) + (seg.end - seg.start)
        real = [spk for spk, t in totals.items() if t >= MIN_SPEAKER_S]
        top = sorted(totals.values(), reverse=True)[:8]

        print(f"--- {label}")
        print(
            f"    {elapsed:.0f} с ({duration / elapsed:.1f}x RTF), "
            f"говорящих: {len(totals)}, из них >= {MIN_SPEAKER_S:.0f} с речи: {len(real)}, "
            f"интервалов: {len(segments)}"
        )
        print("    топ по времени (мин): " + ", ".join(f"{t / 60:.1f}" for t in top))
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_THREADS)
