"""Quality bench: Parakeet int8 vs OpenVINO medium int8 CPU на 15-мин отрывках.

Usage:
    uv run scripts/quality_bench.py

Делает:
  - Нарезает 2 отрывка из реальных записей установочных встреч через PyAV.
  - Прогоняет через Parakeet (int8, CPU EP) и OpenVINO medium int8 (CPU).
  - Собирает метрики: wall-clock, RTFx, peak RSS, first status, max status gap.
  - Сохраняет транскрипты в /tmp/parakeet-bench/bench-<label>-<backend>.md.
  - Печатает сводную таблицу + пишет /tmp/parakeet-bench/summary.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import av
import psutil

from local_transcriber.formatter import format_transcript
from local_transcriber.transcriber import transcribe

EXCERPTS = [
    {
        "label": "artur-intro",
        "source": "/mnt/c/ddmitry/Videos/OBS/2026-02-10 Установочная встреча Артур @a90903.mp4",
        "start_sec": 0,
        "duration_sec": 15 * 60,
    },
    {
        "label": "mar15-mid",
        "source": "/mnt/c/ddmitry/Videos/OBS/2026-03-15 17-20-59.mp4",
        "start_sec": 10 * 60,
        "duration_sec": 15 * 60,
    },
]

BACKENDS = [
    {
        "label": "parakeet-int8",
        "device": "parakeet-cpu",
        "model": "parakeet-tdt-0.6b-v3",
        "compute_type": "int8",
        "language": None,
    },
    {
        "label": "openvino-medium-int8",
        "device": "openvino-cpu",
        "model": "medium",
        "compute_type": "int8",
        "language": "ru",
    },
]

BENCH_DIR = Path("/tmp/parakeet-bench")


def cut_excerpt(source: Path, start_sec: int, duration_sec: int, out: Path) -> None:
    """Вырезает отрывок в 16kHz mono wav через PyAV."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"[cut] skip (exists): {out.name}", flush=True)
        return

    print(f"[cut] {source.name} [{start_sec}s +{duration_sec}s] -> {out.name}", flush=True)

    in_container = av.open(str(source))
    in_stream = next(s for s in in_container.streams if s.type == "audio")

    out_container = av.open(str(out), mode="w", format="wav")
    out_stream = out_container.add_stream("pcm_s16le", rate=16000, layout="mono")

    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)

    in_container.seek(int(start_sec / in_stream.time_base), stream=in_stream, any_frame=False)

    end_sec = start_sec + duration_sec
    started = False
    for frame in in_container.decode(in_stream):
        t = float(frame.pts * in_stream.time_base)
        if t < start_sec:
            continue
        if t >= end_sec:
            break
        if not started:
            started = True
        resampled = resampler.resample(frame)
        for rf in resampled:
            for p in out_stream.encode(rf):
                out_container.mux(p)

    for p in out_stream.encode(None):
        out_container.mux(p)

    out_container.close()
    in_container.close()


def measure_run(audio: Path, backend: dict) -> dict:
    """Transcribe с измерением wall/rss/status intervals."""
    stop = threading.Event()
    peak_rss = [0]
    proc = psutil.Process()

    def sample() -> None:
        while not stop.is_set():
            r = proc.memory_info().rss
            if r > peak_rss[0]:
                peak_rss[0] = r
            time.sleep(1.0)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()

    status_events: list[float] = []
    start = time.monotonic()

    def on_status(msg: str) -> None:
        status_events.append(time.monotonic() - start)

    try:
        result = transcribe(
            file_path=audio,
            model_name=backend["model"],
            device=backend["device"],
            compute_type=backend["compute_type"],
            language=backend["language"],
            on_status=on_status,
        )
        wall = time.monotonic() - start
    finally:
        stop.set()
        sampler.join(timeout=2.0)

    times = status_events
    gaps = [b - a for a, b in zip(times, times[1:])]

    return {
        "result": result,
        "wall_sec": round(wall, 2),
        "peak_rss_gb": round(peak_rss[0] / (1024**3), 3),
        "first_status_sec": round(times[0], 2) if times else None,
        "max_status_gap_sec": round(max(gaps), 2) if gaps else 0.0,
        "status_events_count": len(times),
        "duration_sec": round(result.duration, 2),
        "segments": len(result.segments),
        "rtfx": round(result.duration / wall, 2) if wall > 0 else 0.0,
    }


def main() -> int:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    # Sanity: ensure source files exist
    for e in EXCERPTS:
        if not Path(e["source"]).exists():
            print(f"ERROR: source not found: {e['source']}", file=sys.stderr)
            return 2

    summary: list[dict] = []

    for excerpt in EXCERPTS:
        src = Path(excerpt["source"])
        audio = BENCH_DIR / f"bench-{excerpt['label']}.wav"
        cut_excerpt(src, excerpt["start_sec"], excerpt["duration_sec"], audio)

        for backend in BACKENDS:
            print(f"\n=== {excerpt['label']} :: {backend['label']} ===", flush=True)
            m = measure_run(audio, backend)
            result = m.pop("result")

            tr = BENCH_DIR / f"bench-{excerpt['label']}-{backend['label']}.md"
            lang_mode = "detected" if backend["device"].startswith("parakeet") else "forced"
            content = format_transcript(
                result=result,
                source_filename=f"{src.name} [{excerpt['start_sec']}s +{excerpt['duration_sec']}s]",
                model_name=backend["model"],
                device_info=f"{backend['device']} ({backend['compute_type']})",
                language_mode=lang_mode,
            )
            tr.write_text(content, encoding="utf-8")

            row = {
                "excerpt": excerpt["label"],
                "backend": backend["label"],
                **m,
                "transcript": str(tr),
            }
            summary.append(row)
            print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)

    (BENCH_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n| Excerpt | Backend | Wall | Dur | RTFx | Peak RSS | First | MaxGap | Seg |")
    print("|---------|---------|------|-----|------|----------|-------|--------|-----|")
    for r in summary:
        first = f"{r['first_status_sec']:.1f}s" if r["first_status_sec"] is not None else "—"
        print(
            f"| {r['excerpt']} | {r['backend']} | {r['wall_sec']:.1f}s | "
            f"{r['duration_sec']:.0f}s | {r['rtfx']:.2f} | "
            f"{r['peak_rss_gb']:.2f}GB | {first} | "
            f"{r['max_status_gap_sec']:.1f}s | {r['segments']} |"
        )

    print(f"\nArtifacts: {BENCH_DIR}")
    print(f"Summary JSON: {BENCH_DIR / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
