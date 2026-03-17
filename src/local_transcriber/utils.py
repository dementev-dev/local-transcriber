from pathlib import Path


def check_ffmpeg() -> None:
    raise NotImplementedError


def detect_device(requested: str = "auto") -> str:
    raise NotImplementedError


def get_gpu_name() -> str | None:
    raise NotImplementedError


def validate_input_file(path: Path) -> Path:
    raise NotImplementedError


def build_output_path(input_path: Path, output: Path | None = None) -> Path:
    raise NotImplementedError
