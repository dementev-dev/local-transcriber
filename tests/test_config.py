from unittest.mock import patch

import pytest

from local_transcriber.config import (
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


@pytest.mark.parametrize("value", [True, False])
def test_load_config_accepts_diarize_boolean(tmp_path, value):
    config = tmp_path / "config.toml"
    config.write_text(f"diarize = {str(value).lower()}\n")

    assert load_config(config) == {"diarize": value}


@pytest.mark.parametrize("toml_value", ['"true"', "1"])
def test_load_config_rejects_non_boolean_diarize(tmp_path, toml_value):
    config = tmp_path / "config.toml"
    config.write_text(f"diarize = {toml_value}\n")

    with pytest.raises(ValueError, match="должно быть логическим"):
        load_config(config)


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
    config = {"model": "tiny", "language": "en", "device": "openvino-cpu"}
    cli = {
        "model": "small",
        "language": None,
        "device": "onnx",
        "compute_type": None,
    }
    result = resolve_defaults(cli, config)
    assert result["model"] == "small"
    assert result["language"] == "en"
    assert result["device"] == "onnx"


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
        "diarize": False,
    }


@pytest.mark.parametrize(
    ("cli_value", "config_value", "expected"),
    [
        (None, True, True),
        (None, False, False),
        (True, False, True),
        (False, True, False),
    ],
)
def test_resolve_defaults_diarize_priority(cli_value, config_value, expected):
    cli = {"diarize": cli_value}
    config = {"diarize": config_value}

    assert resolve_defaults(cli, config)["diarize"] is expected


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
