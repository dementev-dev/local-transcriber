"""Утилиты для валидации входных файлов, определения устройства и работы с путями."""

import glob
import platform
import shutil
import subprocess
import warnings
from pathlib import Path

SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma", ".aac",
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts",
}


def detect_device(requested: str = "auto") -> str:
    """Определяет устройство для вычислений.

    При ``requested="auto"`` проверяет: CUDA → OpenVINO GPU → OpenVINO CPU → CPU.
    ``"openvino"`` — алиас для авто-детекта внутри OpenVINO (GPU если доступен, иначе CPU).
    Возвращает всегда конкретное значение (не абстрактный ``"openvino"``).
    """
    if requested == "auto":
        if shutil.which("nvidia-smi") is not None:
            return "cuda"
        if _is_openvino_gpu_available():
            return "openvino-gpu"
        if _is_openvino_available():
            return "openvino-cpu"
        return "cpu"
    if requested == "openvino":
        if _is_openvino_gpu_available():
            return "openvino-gpu"
        return "openvino-cpu"
    return requested


def _is_openvino_available() -> bool:
    """Проверяет доступность OpenVINO: x86/AMD64 архитектура + пакет установлен."""
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        return False
    try:
        import openvino_genai  # noqa: F401

        return True
    except ImportError:
        return False


def _is_openvino_gpu_available() -> bool:
    """Проверяет доступность Intel GPU через OpenVINO.

    Сохраняет x86/AMD64 gate и глотает все ошибки импорта/runtime.
    """
    if not _is_openvino_available():
        return False
    try:
        from openvino import Core

        return "GPU" in Core().available_devices
    except Exception:
        return False


def get_intel_gpu_name() -> str | None:
    """Возвращает название Intel GPU через OpenVINO (для метаданных транскрипта)."""
    try:
        from openvino import Core

        name = Core().get_property("GPU", "FULL_DEVICE_NAME")
        return name if name else None
    except Exception:
        return None


def get_gpu_name() -> str | None:
    """Возвращает название GPU через ``nvidia-smi`` (для метаданных транскрипта)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().splitlines()
            if lines:
                name = lines[0].strip()
                return name if name else None
    except FileNotFoundError:
        pass
    return None


def validate_input_file(path: Path) -> Path:
    """Проверяет существование, непустоту и расширение файла.

    Неизвестное расширение — предупреждение, а не ошибка: файл может оказаться
    корректным контейнером, просто с нестандартным суффиксом.
    Возвращает resolved-путь.
    """
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    if not path.is_file():
        raise ValueError(f"Путь не является файлом: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Файл пустой: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        warnings.warn(
            f"Расширение '{path.suffix}' не входит в список поддерживаемых. "
            "Попытка продолжить.",
            stacklevel=2,
        )
    return path.resolve()


def build_output_path(input_path: Path, output: Path | None = None) -> Path:
    """Строит путь выходного файла: ``<имя>-transcript.md`` рядом с исходным."""
    if output is not None:
        return output
    return input_path.with_stem(input_path.stem + "-transcript").with_suffix(".md")


def expand_globs(paths: list[Path]) -> list[Path]:
    """Раскрывает glob-паттерны в списке путей с дедупликацией.

    Typer на Windows не раскрывает ``*.mp4`` самостоятельно,
    поэтому глобы обрабатываются вручную. Дедупликация — по resolved-пути.
    """
    seen: set[Path] = set()
    result: list[Path] = []
    for p in paths:
        s = str(p)
        if any(c in s for c in ("*", "?", "[")):
            candidates = [Path(m) for m in sorted(glob.glob(s))]
        else:
            candidates = [p]
        for c in candidates:
            resolved = c.resolve()
            if resolved not in seen:
                seen.add(resolved)
                result.append(c)
    return result


def has_existing_transcript(input_path: Path) -> bool:
    """Проверяет наличие транскрипта для skip-логики батч-режима."""
    return build_output_path(input_path).exists()
