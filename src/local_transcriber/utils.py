import glob
import shutil
import subprocess
import warnings
from pathlib import Path

SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma", ".aac",
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts",
}


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
            lines = result.stdout.strip().splitlines()
            if lines:
                name = lines[0].strip()
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


def expand_globs(paths: list[Path]) -> list[Path]:
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
    return build_output_path(input_path).exists()
