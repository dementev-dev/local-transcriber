"""Чистота ASR-сегментов: как часто внутри одного сегмента меняется говорящий.

    uv run --with sherpa-onnx python .scratch/diarization/bench_conflict.py <файл>

Прогоняет ASR и диаризацию по одному файлу и считает, какая доля ASR-сегментов
содержит чужую речь. Это мера того, насколько огрубляет привязка спикера к
целому сегменту по мажоритарному перекрытию.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from common import (
    DEFAULT_THREADS,
    DISCOVERY_THRESHOLD,
    HERE,
    load_audio,
    make_diarizer,
    use_project_sources,
)

use_project_sources()

ASR_MODEL = "gigaam-v3-e2e-rnnt"
COMPUTE_TYPE = "int8"

# чужая речь короче порога — поддакивание, дольше — потерянная реплика
INTERJECTION_S = 1.0

PURITY_LEVELS = (0.95, 0.90, 0.80, 0.70)


def run_asr(audio_path: str):
    from local_transcriber.backends.onnx_asr import OnnxAsrBackend

    backend = OnnxAsrBackend(compute_type_explicit=False)
    path = backend.ensure_model_available(ASR_MODEL, COMPUTE_TYPE)
    model = backend.create_model(path, "onnx", COMPUTE_TYPE)
    t0 = time.perf_counter()
    result = backend.transcribe(model, Path(audio_path), "ru")
    print(f"ASR: {time.perf_counter() - t0:.0f} с, {len(result.segments)} сегм.")
    return result


def run_diar(samples, threshold: float, threads: int):
    diarizer = make_diarizer(threshold=threshold, threads=threads)
    t0 = time.perf_counter()
    segments = diarizer.process(samples).sort_by_start_time()
    print(f"диаризация: {time.perf_counter() - t0:.0f} с, {len(segments)} интервалов")
    return [(s.start, s.end, s.speaker) for s in segments]


def main(audio_path: str, threshold: float, threads: int) -> None:
    samples = load_audio(audio_path)
    asr = run_asr(audio_path)
    diar = run_diar(samples, threshold, threads)
    print(f"речи по диаризации: {sum(e - s for s, e, _ in diar) / 60:.1f} мин\n")

    rows = []
    for seg in asr.segments:
        per_speaker: dict[int, float] = defaultdict(float)
        for start, end, speaker in diar:
            overlap = min(seg.end, end) - max(seg.start, start)
            if overlap > 0:
                per_speaker[speaker] += overlap
        total = sum(per_speaker.values())
        if total <= 0:
            rows.append((seg, None, 0.0, 0.0, {}))
            continue
        major = max(per_speaker, key=lambda k: per_speaker[k])
        rows.append(
            (seg, major, per_speaker[major] / total, total - per_speaker[major], dict(per_speaker))
        )

    n = len(rows)
    unattributed = [r for r in rows if r[1] is None]
    attributed = [r for r in rows if r[1] is not None]
    lost = [r for r in attributed if r[3] >= INTERJECTION_S]
    interjection = [r for r in attributed if 0 < r[3] < INTERJECTION_S]
    clean = [r for r in attributed if r[3] == 0]

    def minutes(rs) -> float:
        return sum(r[0].end - r[0].start for r in rs) / 60

    print("=" * 64)
    print(f"ASR-сегментов: {n} ({minutes(rows):.1f} мин)\n")
    for label, group in (
        ("чистых (один говорящий)", clean),
        (f"с поддакиванием (<{INTERJECTION_S:.0f} с чужой)", interjection),
        (f"с чужой репликой (>={INTERJECTION_S:.0f} с)", lost),
        ("без говорящего вообще", unattributed),
    ):
        print(f"  {label:<34} {len(group):4d}  {len(group) / n * 100:5.1f}%  {minutes(group):5.1f} мин")

    print()
    for level in PURITY_LEVELS:
        bad = [r for r in attributed if r[2] < level]
        print(
            f"  чистота мажоритарного < {level:.2f}: {len(bad):4d} сегм. "
            f"({len(bad) / n * 100:.1f}%), {minutes(bad):.1f} мин"
        )

    print("\n" + "=" * 64)
    print("ХУДШИЕ 12 СЕГМЕНТОВ (больше всего чужой речи внутри):")
    for seg, major, purity, others, per_speaker in sorted(attributed, key=lambda r: -r[3])[:12]:
        share = ", ".join(
            f"spk{k}={v:.1f}с" for k, v in sorted(per_speaker.items(), key=lambda x: -x[1])
        )
        print(
            f"\n  [{seg.start:7.1f}-{seg.end:7.1f}] ({seg.end - seg.start:4.1f} с) "
            f"мажор spk{major}, чистота {purity:.2f}, чужой {others:.1f} с"
        )
        print(f"    {share}")
        print(f"    «{seg.text.strip()[:150]}»")

    out = HERE / f"conflict-{Path(audio_path).stem[:40]}.json"
    out.write_text(
        json.dumps(
            {
                "file": Path(audio_path).name,
                "threshold": threshold,
                "asr_segments": n,
                "clean": len(clean),
                "interjection": len(interjection),
                "lost_utterance": len(lost),
                "unattributed": len(unattributed),
                "minutes_lost_utterance": round(minutes(lost), 2),
                "minutes_total": round(minutes(rows), 2),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nсводка сохранена: {out.name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(
        sys.argv[1],
        float(sys.argv[2]) if len(sys.argv) > 2 else DISCOVERY_THRESHOLD,
        int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_THREADS,
    )
