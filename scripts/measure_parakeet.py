"""Измерение производительности Parakeet на длинном файле.

Usage:
    uv run scripts/measure_parakeet.py <audio-file> [--compute-type int8|float32]

Логирует:
  - peak RSS (через psutil, опрашивает раз в секунду в отдельном потоке).
  - timestamp каждого on_status callback (первый update, интервалы между).
  - wall-clock и RTFx.
Результат печатается в stdout + записывается в /tmp/parakeet-measure.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import psutil

from local_transcriber.transcriber import transcribe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--model", default="parakeet-tdt-0.6b-v3")
    parser.add_argument("--output", type=Path, default=Path("/tmp/parakeet-measure.json"))
    args = parser.parse_args()

    # RSS sampler в отдельном потоке
    stop = threading.Event()
    peak_rss = [0]
    proc = psutil.Process()

    def sample_rss() -> None:
        while not stop.is_set():
            rss = proc.memory_info().rss
            if rss > peak_rss[0]:
                peak_rss[0] = rss
            time.sleep(1.0)

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()

    # on_status log
    status_events: list[dict] = []
    start = time.monotonic()

    def on_status(msg: str) -> None:
        status_events.append({"t": time.monotonic() - start, "msg": msg})

    try:
        result = transcribe(
            file_path=args.audio,
            model_name=args.model,
            device="parakeet-cpu",
            compute_type=args.compute_type,
            on_status=on_status,
        )
        wall = time.monotonic() - start
    finally:
        stop.set()
        sampler.join(timeout=2.0)

    # Анализ интервалов
    times = [e["t"] for e in status_events]
    first_update = times[0] if times else None
    max_gap = max((b - a for a, b in zip(times, times[1:])), default=0.0)

    rtfx = (result.duration / wall) if wall > 0 else 0.0
    peak_rss_gb = peak_rss[0] / (1024 ** 3)

    report = {
        "audio": str(args.audio),
        "duration_sec": result.duration,
        "wall_sec": wall,
        "rtfx": rtfx,
        "peak_rss_gb": peak_rss_gb,
        "first_status_sec": first_update,
        "max_status_gap_sec": max_gap,
        "status_events_count": len(status_events),
        "segments": len(result.segments),
        "compute_type": args.compute_type,
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # Gate thresholds
    ok_rss = peak_rss_gb <= 4.0
    ok_first = first_update is not None and first_update <= 10.0
    ok_gap = max_gap <= 30.0

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nGate: rss={'✓' if ok_rss else '✗'}  "
          f"first_update={'✓' if ok_first else '✗'}  "
          f"max_gap={'✓' if ok_gap else '✗'}")

    return 0 if (ok_rss and ok_first and ok_gap) else 1


if __name__ == "__main__":
    sys.exit(main())
