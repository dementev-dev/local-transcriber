"""Общая обвязка для замеров диаризации.

Скрипты в этом каталоге — исследовательские, не часть пакета. Они опираются на
``sherpa-onnx``, которого нет в зависимостях проекта, поэтому запускаются через
``uv run --with sherpa-onnx``.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
MODELS = HERE / "models"

SEGMENTATION = MODELS / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
EMBEDDING = MODELS / "wespeaker_en_voxceleb_resnet34_LM.onnx"

SAMPLE_RATE = 16_000

# Конфигурация, выбранная калибровкой 2026-08-14 на трёх записях.
# На 0.5 из примеров sherpa-onnx получалось 29 говорящих вместо трёх.
DISCOVERY_THRESHOLD = 0.89
DEFAULT_THREADS = 8


def use_project_sources() -> None:
    """Делает пакет проекта импортируемым без установки."""
    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def require_models() -> None:
    """Останавливает запуск с внятным сообщением, если модели не скачаны."""
    missing = [p for p in (SEGMENTATION, EMBEDDING) if not p.exists()]
    if missing:
        names = "\n  ".join(str(p) for p in missing)
        raise SystemExit(
            f"Не найдены модели диаризации:\n  {names}\n\n"
            "Скачайте их по инструкции из README.md в этом каталоге."
        )


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD),
        ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def peak_rss_mb() -> float | None:
    """Пиковая рабочая память процесса в МБ; None, если снять не удалось.

    На Linux ``ru_maxrss`` измеряется в КиБ, на macOS — в байтах. В Windows
    используются системные счётчики процесса.

    Два подвоха, на которых замер в разведке 2026-08-12 вернул ноль в Windows:
    экспорт на современных Windows живёт в kernel32 как
    ``K32GetProcessMemoryInfo``, а без явных ``restype``/``argtypes``
    псевдодескриптор процесса уезжает в вызов как 32-битное число и функция
    молча не срабатывает.
    """
    if sys.platform != "win32":
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
        return peak / divisor

    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    handle = kernel32.GetCurrentProcess()

    pmc = _ProcessMemoryCounters()
    pmc.cb = ctypes.sizeof(_ProcessMemoryCounters)

    for dll, name in (
        (kernel32, "K32GetProcessMemoryInfo"),
        (ctypes.windll.psapi, "GetProcessMemoryInfo"),
    ):
        func = getattr(dll, name, None)
        if func is None:
            continue
        func.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCounters),
            wt.DWORD,
        ]
        func.restype = wt.BOOL
        if func(handle, ctypes.byref(pmc), pmc.cb):
            return pmc.PeakWorkingSetSize / 1024 / 1024
    return None


def load_audio(audio_path: str | Path):
    """Декодирует файл в моно 16 кГц — тот же путь, что использует ONNX-бэкенд."""
    from faster_whisper import decode_audio

    return decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)


def make_diarizer(
    threshold: float = DISCOVERY_THRESHOLD,
    num_clusters: int = -1,
    threads: int = DEFAULT_THREADS,
) -> Any:
    """Собирает OfflineSpeakerDiarization с параметрами разведки."""
    import sherpa_onnx as so

    require_models()
    config = so.OfflineSpeakerDiarizationConfig(
        segmentation=so.OfflineSpeakerSegmentationModelConfig(
            pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(SEGMENTATION)
            ),
            num_threads=threads,
            provider="cpu",
        ),
        embedding=so.SpeakerEmbeddingExtractorConfig(
            model=str(EMBEDDING), num_threads=threads, provider="cpu"
        ),
        clustering=so.FastClusteringConfig(
            num_clusters=num_clusters, threshold=threshold
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    diarizer = so.OfflineSpeakerDiarization(config)
    assert diarizer.sample_rate == SAMPLE_RATE, diarizer.sample_rate
    return diarizer
