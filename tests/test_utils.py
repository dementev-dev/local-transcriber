import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from local_transcriber.utils import (
    build_output_path,
    detect_device,
    get_gpu_name,
    validate_input_file,
)


def test_validate_input_file_not_found(tmp_path):
    missing = tmp_path / "no_such_file.mp3"
    with pytest.raises(FileNotFoundError):
        validate_input_file(missing)


def test_validate_input_file_empty(tmp_path):
    empty = tmp_path / "empty.mp3"
    empty.touch()
    with pytest.raises(ValueError, match="пустой"):
        validate_input_file(empty)


def test_validate_input_file_unknown_ext(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hello")
    with pytest.warns(UserWarning, match="не входит в список"):
        result = validate_input_file(f)
    assert result == f.resolve()


def test_validate_input_file_ok(tmp_path):
    f = tmp_path / "audio.mp3"
    f.write_bytes(b"\x00" * 16)
    result = validate_input_file(f)
    assert result == f.resolve()


def test_build_output_path_default():
    inp = Path("/some/dir/meeting-2026-03-17.mp4")
    result = build_output_path(inp)
    assert result == Path("/some/dir/meeting-2026-03-17-transcript.md")


def test_build_output_path_custom():
    inp = Path("/some/dir/audio.mp3")
    custom = Path("/out/result.md")
    result = build_output_path(inp, output=custom)
    assert result == custom


def test_detect_device_explicit():
    assert detect_device("cpu") == "cpu"
    assert detect_device("cuda") == "cuda"


def test_get_gpu_name_no_nvidia_smi():
    with patch("shutil.which", return_value=None):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = get_gpu_name()
    assert result is None


def test_get_gpu_name_success():
    mock_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="NVIDIA GeForce RTX 3060\n", stderr=""
    )
    with patch("subprocess.run", return_value=mock_result):
        result = get_gpu_name()
    assert result == "NVIDIA GeForce RTX 3060"
