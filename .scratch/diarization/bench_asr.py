"""Замер ASR тем же путём, что использует CLI — для соотношения с диаризацией.

    uv run python .scratch/diarization/bench_asr.py <файл> [модель]

sherpa-onnx здесь не нужен: скрипт зовёт бэкенд проекта напрямую.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from common import peak_rss_mb, use_project_sources

use_project_sources()

from local_transcriber.backends.onnx_asr import OnnxAsrBackend  # noqa: E402

DEFAULT_MODEL = "gigaam-v3-e2e-rnnt"
COMPUTE_TYPE = "int8"


def main(audio_path: str, model_name: str) -> None:
    backend = OnnxAsrBackend(compute_type_explicit=False)

    t0 = time.perf_counter()
    model_path = backend.ensure_model_available(model_name, COMPUTE_TYPE)
    model = backend.create_model(model_path, "onnx", COMPUTE_TYPE)
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = backend.transcribe(model, Path(audio_path), "ru")
    t_asr = time.perf_counter() - t0
    rss = peak_rss_mb()

    print(f"файл: {audio_path}")
    print(f"модель: {model_name} ({COMPUTE_TYPE})")
    print(f"длительность: {result.duration / 60:.1f} мин")
    print(f"загрузка модели: {t_load:.1f} с")
    print(f"ASR: {t_asr:.1f} с  ->  {result.duration / t_asr:.1f}x RTF")
    print(f"пиковая память процесса: {rss:.0f} МБ" if rss else "память: снять не удалось")
    print(f"сегментов: {len(result.segments)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL)
