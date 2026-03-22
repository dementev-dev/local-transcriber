import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from local_transcriber.utils import (
    _is_openvino_gpu_available,
    build_output_path,
    detect_device,
    expand_globs,
    get_gpu_name,
    get_intel_gpu_name,
    has_existing_transcript,
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


def test_detect_device_explicit_passthrough():
    """Явные device strings проходят без изменений."""
    assert detect_device("cpu") == "cpu"
    assert detect_device("cuda") == "cuda"
    assert detect_device("openvino-gpu") == "openvino-gpu"
    assert detect_device("openvino-cpu") == "openvino-cpu"


def test_detect_device_auto_openvino_gpu():
    """auto + нет nvidia-smi + есть OpenVINO GPU → openvino-gpu."""
    with (
        patch("local_transcriber.utils.shutil.which", return_value=None),
        patch("local_transcriber.utils._is_openvino_gpu_available", return_value=True),
    ):
        assert detect_device("auto") == "openvino-gpu"


def test_detect_device_auto_openvino_cpu():
    """auto + нет nvidia-smi + есть OpenVINO, нет GPU → openvino-cpu."""
    with (
        patch("local_transcriber.utils.shutil.which", return_value=None),
        patch("local_transcriber.utils._is_openvino_gpu_available", return_value=False),
        patch("local_transcriber.utils._is_openvino_available", return_value=True),
    ):
        assert detect_device("auto") == "openvino-cpu"


def test_detect_device_cuda_over_openvino():
    """nvidia-smi доступен и openvino тоже → cuda побеждает."""
    with (
        patch("local_transcriber.utils.shutil.which", return_value="/usr/bin/nvidia-smi"),
        patch("local_transcriber.utils._is_openvino_gpu_available", return_value=True),
    ):
        assert detect_device("auto") == "cuda"


def test_detect_device_auto_cpu_fallback():
    """Ни nvidia-smi, ни openvino → cpu."""
    with (
        patch("local_transcriber.utils.shutil.which", return_value=None),
        patch("local_transcriber.utils._is_openvino_gpu_available", return_value=False),
        patch("local_transcriber.utils._is_openvino_available", return_value=False),
    ):
        assert detect_device("auto") == "cpu"


def test_detect_device_openvino_resolves_to_gpu():
    """--device openvino + GPU доступен → openvino-gpu."""
    with patch("local_transcriber.utils._is_openvino_gpu_available", return_value=True):
        assert detect_device("openvino") == "openvino-gpu"


def test_detect_device_openvino_resolves_to_cpu():
    """--device openvino + GPU недоступен → openvino-cpu."""
    with patch("local_transcriber.utils._is_openvino_gpu_available", return_value=False):
        assert detect_device("openvino") == "openvino-cpu"


# === _is_openvino_gpu_available ===


def test_is_openvino_gpu_available_true():
    mock_core = MagicMock()
    mock_core.return_value.available_devices = ["CPU", "GPU"]
    with (
        patch("local_transcriber.utils._is_openvino_available", return_value=True),
        patch("local_transcriber.utils.Core", mock_core, create=True),
    ):
        # Need to patch the import inside the function
        with patch.dict("sys.modules", {"openvino": MagicMock(Core=mock_core)}):
            assert _is_openvino_gpu_available() is True


def test_is_openvino_gpu_available_no_gpu():
    mock_core = MagicMock()
    mock_core.return_value.available_devices = ["CPU"]
    with (
        patch("local_transcriber.utils._is_openvino_available", return_value=True),
        patch.dict("sys.modules", {"openvino": MagicMock(Core=mock_core)}),
    ):
        assert _is_openvino_gpu_available() is False


def test_is_openvino_gpu_available_no_openvino():
    """OpenVINO не установлен → False."""
    with patch("local_transcriber.utils._is_openvino_available", return_value=False):
        assert _is_openvino_gpu_available() is False


def test_is_openvino_gpu_available_runtime_error():
    """Runtime error от OpenVINO → False (fail-safe)."""
    with (
        patch("local_transcriber.utils._is_openvino_available", return_value=True),
        patch.dict("sys.modules", {"openvino": MagicMock(Core=MagicMock(side_effect=RuntimeError("broken")))}),
    ):
        assert _is_openvino_gpu_available() is False


# === get_intel_gpu_name ===


def test_get_intel_gpu_name_success():
    mock_core = MagicMock()
    mock_core.return_value.get_property.return_value = "Intel(R) Arc(TM) 140T GPU (16GB) (iGPU)"
    with patch.dict("sys.modules", {"openvino": MagicMock(Core=mock_core)}):
        assert get_intel_gpu_name() == "Intel(R) Arc(TM) 140T GPU (16GB) (iGPU)"


def test_get_intel_gpu_name_error():
    """Ошибка → None (fail-safe)."""
    with patch.dict("sys.modules", {"openvino": MagicMock(Core=MagicMock(side_effect=Exception("no gpu")))}):
        assert get_intel_gpu_name() is None


def test_get_gpu_name_no_nvidia_smi():
    with patch("shutil.which", return_value=None):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = get_gpu_name()
    assert result is None


def test_get_gpu_name_empty_stdout():
    """nvidia-smi returns 0 but stdout is empty — should return None, not crash."""
    mock_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="", stderr=""
    )
    with patch("subprocess.run", return_value=mock_result):
        result = get_gpu_name()
    assert result is None


def test_get_gpu_name_success():
    mock_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="NVIDIA GeForce RTX 3060\n", stderr=""
    )
    with patch("subprocess.run", return_value=mock_result):
        result = get_gpu_name()
    assert result == "NVIDIA GeForce RTX 3060"


def test_expand_globs_no_patterns(tmp_path):
    a = tmp_path / "a.mp3"
    b = tmp_path / "b.mp3"
    result = expand_globs([a, b])
    assert result == [a, b]


def test_expand_globs_with_star(tmp_path):
    (tmp_path / "x.mp3").write_bytes(b"fake")
    (tmp_path / "y.mp3").write_bytes(b"fake")
    (tmp_path / "z.txt").write_bytes(b"fake")
    result = expand_globs([Path(str(tmp_path / "*.mp3"))])
    assert len(result) == 2
    assert all(p.suffix == ".mp3" for p in result)


def test_expand_globs_no_match(tmp_path):
    result = expand_globs([Path(str(tmp_path / "*.wav"))])
    assert result == []


def test_expand_globs_deduplicates(tmp_path):
    f = tmp_path / "a.mp3"
    f.write_bytes(b"fake")
    result = expand_globs([f, Path(str(tmp_path / "*.mp3"))])
    assert len(result) == 1


def test_has_existing_transcript_true(tmp_path):
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"fake")
    transcript = tmp_path / "meeting-transcript.md"
    transcript.write_text("content")
    assert has_existing_transcript(audio) is True


def test_has_existing_transcript_false(tmp_path):
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"fake")
    assert has_existing_transcript(audio) is False
