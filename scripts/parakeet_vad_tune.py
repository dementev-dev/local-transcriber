"""One-off: re-run Parakeet int8 на тех же отрывках с разными VAD-порогами.

Usage:
    uv run scripts/parakeet_vad_tune.py

Использует уже нарезанные WAV-файлы из /tmp/parakeet-bench/ (предполагается,
что quality_bench.py уже пробегал).

Цель: проверить гипотезу, что реплики менти пропадали из-за дефолтного
Silero threshold=0.5 на тихом микрофоне.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import onnx_asr
from huggingface_hub import snapshot_download

from local_transcriber.backends.parakeet import (
    MODEL_REPO,
    ONNX_ASR_MODEL_ALIAS,
    _INT8_PATTERNS,
)

BENCH_DIR = Path("/tmp/parakeet-bench")

EXCERPTS = [
    ("artur-intro", BENCH_DIR / "bench-artur-intro.wav"),
    ("mar15-mid", BENCH_DIR / "bench-mar15-mid.wav"),
]

VAD_VARIANTS = [
    {"label": "threshold-0.5-default", "kwargs": {}},
    {"label": "threshold-0.3", "kwargs": {"threshold": 0.3}},
    {"label": "threshold-0.2", "kwargs": {"threshold": 0.2}},
]


def main() -> int:
    # Проверяем наличие нарезанных WAV
    for _label, wav in EXCERPTS:
        if not wav.exists():
            print(f"ERROR: не найден {wav}. Сначала прогоните quality_bench.py.", file=sys.stderr)
            return 2

    print("Загрузка модели...", flush=True)
    model_path = snapshot_download(MODEL_REPO, local_files_only=True, allow_patterns=_INT8_PATTERNS)
    asr = onnx_asr.load_model(ONNX_ASR_MODEL_ALIAS, path=model_path, quantization="int8")
    vad = onnx_asr.load_vad(model="silero")

    summary: list[dict] = []

    for label, wav in EXCERPTS:
        for variant in VAD_VARIANTS:
            tag = f"{label}-parakeet-int8-{variant['label']}"
            out = BENCH_DIR / f"bench-{tag}.txt"
            print(f"\n=== {tag} ===", flush=True)

            pipeline = asr.with_vad(vad=vad, **variant["kwargs"])
            t0 = time.monotonic()
            segments = list(pipeline.recognize(str(wav)))
            wall = time.monotonic() - t0

            # Пишем транскрипт
            lines = []
            for s in segments:
                start = float(getattr(s, "start", 0.0) or 0.0)
                end = float(getattr(s, "end", start) or start)
                text = getattr(s, "text", "") or ""
                lines.append(f"[{start:7.2f}-{end:7.2f}]  {text.strip()}")
            out.write_text("\n".join(lines), encoding="utf-8")

            duration = float(segments[-1].end) if segments else 0.0
            row = {
                "excerpt": label,
                "variant": variant["label"],
                "wall_sec": round(wall, 2),
                "segments": len(segments),
                "duration_sec": round(duration, 2),
                "rtfx": round(duration / wall, 2) if wall > 0 else 0.0,
                "transcript": str(out),
            }
            summary.append(row)
            print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)

    (BENCH_DIR / "vad_tune_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n| Excerpt | Variant | Wall | Seg | RTFx |")
    print("|---------|---------|------|-----|------|")
    for r in summary:
        print(
            f"| {r['excerpt']} | {r['variant']} | {r['wall_sec']:.1f}s | "
            f"{r['segments']} | {r['rtfx']:.2f} |"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
