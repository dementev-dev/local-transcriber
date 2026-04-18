from pathlib import Path
from unittest.mock import patch

import pytest

from local_transcriber.config import (
    apply_device_defaults,
    find_config_file,
    load_config,
    resolve_defaults,
)


def test_find_config_file_cwd(tmp_path, monkeypatch):
    config = tmp_path / ".transcriber.toml"
    config.write_text('model = "small"\n')
    monkeypatch.chdir(tmp_path)
    assert find_config_file() == config


def test_find_config_file_user_home(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no .transcriber.toml in CWD
    global_config = tmp_path / ".config" / "transcriber" / "config.toml"
    global_config.parent.mkdir(parents=True)
    global_config.write_text('language = "ru"\n')
    with patch("local_transcriber.config.Path.home", return_value=tmp_path):
        result = find_config_file()
    assert result == global_config


def test_find_config_file_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("local_transcriber.config.Path.home", return_value=tmp_path):
        assert find_config_file() is None


def test_load_config_valid(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('model = "small"\nlanguage = "ru"\n')
    result = load_config(config)
    assert result == {"model": "small", "language": "ru"}


def test_load_config_malformed(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("this is not valid toml [[[")
    with pytest.raises(ValueError, match="Ошибка чтения конфига"):
        load_config(config)


def test_load_config_unknown_keys_warned(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('modle = "small"\nmodel = "tiny"\n')
    with pytest.warns(UserWarning, match="modle"):
        result = load_config(config)
    assert result == {"model": "tiny"}


def test_load_config_non_string_value(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("device = 123\n")
    with pytest.raises(ValueError, match="должно быть строкой"):
        load_config(config)


def test_load_config_invalid_device(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('device = "tpu"\n')
    with pytest.raises(ValueError, match="Недопустимое значение device"):
        load_config(config)


def test_resolve_defaults_cli_wins():
    config = {"model": "tiny", "language": "en"}
    cli = {"model": "small", "language": None, "device": None, "compute_type": None}
    result = resolve_defaults(cli, config)
    assert result["model"] == "small"
    assert result["language"] == "en"


def test_resolve_defaults_config_wins():
    config = {"model": "tiny"}
    cli = {"model": None, "language": None, "device": None, "compute_type": None}
    result = resolve_defaults(cli, config)
    assert result["model"] == "tiny"


def test_resolve_defaults_hardcoded_fallback():
    result = resolve_defaults(
        {"model": None, "language": None, "device": None, "compute_type": None}, {}
    )
    assert result == {
        "model": "medium",
        "language": "ru",
        "device": "auto",
        "compute_type": "float32",
    }


def test_apply_device_defaults_cuda():
    defaults = {"model": "medium", "language": "ru", "device": "auto", "compute_type": "float32"}
    cli = {"model": None, "language": None, "device": None, "compute_type": None}
    result = apply_device_defaults(defaults, "cuda", cli, {})
    assert result["model"] == "medium"
    assert result["compute_type"] == "float16"


def test_apply_device_defaults_cpu():
    defaults = {"model": "medium", "language": "ru", "device": "auto", "compute_type": "float32"}
    cli = {"model": None, "language": None, "device": None, "compute_type": None}
    result = apply_device_defaults(defaults, "cpu", cli, {})
    assert result["model"] == "medium"
    assert result["compute_type"] == "float32"


def test_apply_device_defaults_cli_overrides():
    defaults = {"model": "large-v3", "language": "ru", "device": "auto", "compute_type": "int8"}
    cli = {"model": "large-v3", "language": None, "device": None, "compute_type": "int8"}
    result = apply_device_defaults(defaults, "cuda", cli, {})
    assert result["model"] == "large-v3"
    assert result["compute_type"] == "int8"


def test_apply_device_defaults_config_overrides():
    defaults = {"model": "small", "language": "ru", "device": "auto", "compute_type": "int8"}
    cli = {"model": None, "language": None, "device": None, "compute_type": None}
    config = {"model": "small", "compute_type": "int8"}
    result = apply_device_defaults(defaults, "cuda", cli, config)
    assert result["model"] == "small"
    assert result["compute_type"] == "int8"


def test_load_config_openvino_device(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('device = "openvino"\n')
    result = load_config(config)
    assert result == {"device": "openvino"}


def test_load_config_openvino_gpu_device(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('device = "openvino-gpu"\n')
    result = load_config(config)
    assert result == {"device": "openvino-gpu"}


def test_load_config_openvino_cpu_device(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('device = "openvino-cpu"\n')
    result = load_config(config)
    assert result == {"device": "openvino-cpu"}


def test_apply_device_defaults_openvino():
    defaults = {"model": "medium", "language": "ru", "device": "auto", "compute_type": "float32"}
    cli = {"model": None, "language": None, "device": None, "compute_type": None}
    result = apply_device_defaults(defaults, "openvino", cli, {})
    assert result["model"] == "medium"
    assert result["compute_type"] == "int8"


def test_apply_device_defaults_openvino_gpu():
    defaults = {"model": "medium", "language": "ru", "device": "auto", "compute_type": "float32"}
    cli = {"model": None, "language": None, "device": None, "compute_type": None}
    result = apply_device_defaults(defaults, "openvino-gpu", cli, {})
    assert result["model"] == "medium"
    assert result["compute_type"] == "int8"


def test_apply_device_defaults_openvino_cpu():
    defaults = {"model": "medium", "language": "ru", "device": "auto", "compute_type": "float32"}
    cli = {"model": None, "language": None, "device": None, "compute_type": None}
    result = apply_device_defaults(defaults, "openvino-cpu", cli, {})
    assert result["model"] == "medium"
    assert result["compute_type"] == "int8"


def test_load_config_accepts_parakeet_device(tmp_path):
    cfg = tmp_path / ".transcriber.toml"
    cfg.write_text('device = "parakeet"\n')
    assert load_config(cfg) == {"device": "parakeet"}


def test_load_config_accepts_parakeet_cpu_device(tmp_path):
    cfg = tmp_path / ".transcriber.toml"
    cfg.write_text('device = "parakeet-cpu"\n')
    assert load_config(cfg) == {"device": "parakeet-cpu"}
