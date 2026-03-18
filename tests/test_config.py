from pathlib import Path
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
        "model": "large-v3",
        "language": "auto",
        "device": "auto",
        "compute_type": "int8",
    }
