"""Один прогон диаризации: скорость, память, распределение по говорящим.

    uv run --with sherpa-onnx python .scratch/diarization/bench_diar.py <файл> [потоки]

Сохраняет разметку в ``segments-<порог>.tsv`` рядом со скриптом — она нужна
тикету про проверку границ на слух и скрипту bench_conflict.py.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from common import (
    DEFAULT_THREADS,
    DISCOVERY_THRESHOLD,
    HERE,
    SAMPLE_RATE,
    load_audio,
    make_diarizer,
    peak_rss_mb,
)


def main(audio_path: str, threads: int, threshold: float) -> None:
    print(f"файл: {audio_path}")
    print(f"потоков: {threads}, порог кластеризации: {threshold}")

    t0 = time.perf_counter()
    samples = load_audio(audio_path)
    t_decode = time.perf_counter() - t0
    duration = len(samples) / SAMPLE_RATE
    print(f"длительность: {duration / 60:.1f} мин ({duration:.0f} с)")
    print(f"декодирование: {t_decode:.1f} с ({duration / t_decode:.0f}x RTF)")

    t0 = time.perf_counter()
    diarizer = make_diarizer(threshold=threshold, threads=threads)
    print(f"инициализация моделей: {time.perf_counter() - t0:.1f} с")

    progress = {"shown": 0.0}
    t_start = time.perf_counter()

    def on_progress(processed: int, total: int, _arg=None) -> int:
        pct = processed / total * 100
        if pct - progress["shown"] >= 20:
            progress["shown"] = pct
            print(f"  ... {pct:.0f}%  ({time.perf_counter() - t_start:.0f} с)", flush=True)
        return 0

    segments = diarizer.process(samples, callback=on_progress).sort_by_start_time()
    t_diar = time.perf_counter() - t_start

    speakers = sorted({s.speaker for s in segments})
    speech = sum(s.end - s.start for s in segments)
    rss = peak_rss_mb()

    print()
    print(f"ДИАРИЗАЦИЯ: {t_diar:.1f} с  ->  {duration / t_diar:.1f}x RTF")
    print(f"пиковая память процесса: {rss:.0f} МБ" if rss else "память: снять не удалось")
    print(f"спикеров: {len(speakers)}, интервалов: {len(segments)}")
    print(f"речи: {speech / 60:.1f} мин ({speech / duration * 100:.0f}% файла)")
    print()
    print("распределение по говорящим:")
    for spk in speakers:
        own = [s for s in segments if s.speaker == spk]
        total = sum(s.end - s.start for s in own)
        median = np.median([s.end - s.start for s in own])
        print(f"  spk{spk:<3} {total / 60:6.1f} мин  {len(own):4d} интерв.  медиана {median:.1f} с")

    out = HERE / f"segments-{threshold}.tsv"
    out.write_text(
        "\n".join(f"{s.start:.3f}\t{s.end:.3f}\t{s.speaker}" for s in segments),
        encoding="utf-8",
    )
    print(f"\nразметка сохранена: {out.name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(
        sys.argv[1],
        int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_THREADS,
        float(sys.argv[3]) if len(sys.argv) > 3 else DISCOVERY_THRESHOLD,
    )
