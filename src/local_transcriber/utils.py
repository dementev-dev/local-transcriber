import shutil
import subprocess
import sys
import warnings
from pathlib import Path

SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma", ".aac",
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts",
}


def check_ffmpeg() -> None:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=False)
    except FileNotFoundError:
        sys.exit(
            "ffmpeg не найден в PATH. Установите ffmpeg:\n"
            "  Linux:   apt install ffmpeg\n"
            "  Windows: winget install ffmpeg\n"
            "  macOS:   brew install ffmpeg"
        )


def detect_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if shutil.which("nvidia-smi") is not None:
        return "cuda"
    return "cpu"


def get_gpu_name() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            name = result.stdout.strip().splitlines()[0].strip()
            return name if name else None
    except FileNotFoundError:
        pass
    return None


def validate_input_file(path: Path) -> Path:
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
    if output is not None:
        return output
    return input_path.with_stem(input_path.stem + "-transcript").with_suffix(".md")
