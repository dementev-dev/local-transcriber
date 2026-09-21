"""Загрузка конфигурации из ``.transcriber.toml`` и каскад приоритетов."""

import tomllib
import warnings
from pathlib import Path
from typing import TypedDict, TypeVar, cast

type ConfigValue = str | bool
_T = TypeVar("_T")


class ConfigValues(TypedDict, total=False):
    """Проверенные значения из TOML-конфига."""

    model: str
    language: str
    device: str
    compute_type: str
    diarize: bool


class CliValues(TypedDict, total=False):
    """Явные значения CLI до разрешения каскада."""

    model: str | None
    language: str | None
    device: str | None
    compute_type: str | None
    diarize: bool | None


class ResolvedConfig(TypedDict):
    """Полная конфигурация после разрешения каскада."""

    model: str
    language: str
    device: str
    compute_type: str
    diarize: bool


HARDCODED_DEFAULTS: ResolvedConfig = {
    "model": "medium",
    "language": "ru",
    "device": "auto",
    "compute_type": "float32",
    "diarize": False,
}

DEVICE_DEFAULTS: dict[str, dict[str, str]] = {
    "cuda": {"model": "medium", "compute_type": "float16"},
    "cpu": {"model": "medium", "compute_type": "float32"},
    "openvino": {"model": "medium", "compute_type": "int8"},
    "openvino-gpu": {"model": "medium", "compute_type": "int8"},
    "openvino-cpu": {"model": "medium", "compute_type": "int8"},
    "onnx": {"model": "gigaam-v3-e2e-rnnt", "compute_type": "int8"},
}

# Одно место правды для допустимых ключей конфига
_VALID_KEYS = set(HARDCODED_DEFAULTS)
_VALID_DEVICES = {
    "auto",
    "cpu",
    "cuda",
    "openvino",
    "openvino-gpu",
    "openvino-cpu",
    "onnx",
}


def find_config_file() -> Path | None:
    """Ищет конфиг: сначала ``.transcriber.toml`` в cwd, затем ``~/.config/transcriber/config.toml``."""
    cwd_config = Path.cwd() / ".transcriber.toml"
    if cwd_config.is_file():
        return cwd_config

    global_config = Path.home() / ".config" / "transcriber" / "config.toml"
    if global_config.is_file():
        return global_config

    return None


def load_config(path: Path | None = None) -> ConfigValues:
    """Загружает и валидирует TOML-конфиг.

    Неизвестные ключи вызывают предупреждение (а не ошибку) для forward
    compatibility: новые версии могут добавить ключи, которых ещё нет в текущей.
    """
    if path is None:
        path = find_config_file()
    if path is None:
        return {}

    try:
        raw = path.read_bytes()
        data = tomllib.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Ошибка чтения конфига {path}: {exc}") from exc

    unknown = set(data) - _VALID_KEYS
    if unknown:
        warnings.warn(
            f"Неизвестные ключи в {path}: {', '.join(sorted(unknown))}",
            stacklevel=2,
        )

    result: dict[str, ConfigValue] = {}
    for key in _VALID_KEYS:
        if key not in data:
            continue
        value = data[key]
        if key == "diarize":
            if not isinstance(value, bool):
                raise ValueError(
                    f"Значение 'diarize' в {path} должно быть логическим, "
                    f"получено {type(value).__name__}"
                )
            result[key] = value
            continue
        if not isinstance(value, str):
            raise ValueError(
                f"Значение '{key}' в {path} должно быть строкой, получено {type(value).__name__}"
            )
        if key == "device" and value not in _VALID_DEVICES:
            raise ValueError(
                f"Недопустимое значение device = '{value}' в {path}. "
                f"Ожидается: {', '.join(sorted(_VALID_DEVICES))}"
            )
        if key == "language" and not value:
            raise ValueError(f"Значение 'language' в {path} не может быть пустым")
        result[key] = value

    return cast(ConfigValues, result)


def resolve_defaults(cli_values: CliValues, config: ConfigValues) -> ResolvedConfig:
    """Каскад приоритетов: CLI > конфиг-файл > hardcoded-дефолты."""
    return {
        "model": _resolve_value(
            cli_values.get("model"), config.get("model"), HARDCODED_DEFAULTS["model"]
        ),
        "language": _resolve_value(
            cli_values.get("language"),
            config.get("language"),
            HARDCODED_DEFAULTS["language"],
        ),
        "device": _resolve_value(
            cli_values.get("device"),
            config.get("device"),
            HARDCODED_DEFAULTS["device"],
        ),
        "compute_type": _resolve_value(
            cli_values.get("compute_type"),
            config.get("compute_type"),
            HARDCODED_DEFAULTS["compute_type"],
        ),
        "diarize": _resolve_value(
            cli_values.get("diarize"),
            config.get("diarize"),
            HARDCODED_DEFAULTS["diarize"],
        ),
    }


def _resolve_value(cli_value: _T | None, config_value: _T | None, default: _T) -> _T:
    """Выбирает первое явно заданное значение каскада."""
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default
