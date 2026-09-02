"""Загрузка конфигурации из ``.transcriber.toml`` и каскад приоритетов."""

import tomllib
import warnings
from pathlib import Path

type ConfigValue = str | bool

HARDCODED_DEFAULTS: dict[str, ConfigValue] = {
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


def load_config(path: Path | None = None) -> dict[str, ConfigValue]:
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

    return result


def resolve_defaults(
    cli_values: dict[str, ConfigValue | None], config: dict[str, ConfigValue]
) -> dict[str, ConfigValue]:
    """Каскад приоритетов: CLI > конфиг-файл > hardcoded-дефолты."""
    result: dict[str, ConfigValue] = {}
    for key in HARDCODED_DEFAULTS:
        cli_val = cli_values.get(key)
        if cli_val is not None:
            result[key] = cli_val
        elif key in config:
            result[key] = config[key]
        else:
            result[key] = HARDCODED_DEFAULTS[key]
    return result


def apply_device_defaults(
    defaults: dict[str, ConfigValue],
    resolved_device: str,
    cli_values: dict[str, ConfigValue | None],
    config: dict[str, ConfigValue],
) -> dict[str, ConfigValue]:
    """Применяет device-aware дефолты для model и compute_type,
    если они не были явно заданы через CLI или конфиг."""
    device_defs = DEVICE_DEFAULTS.get(resolved_device, {})
    if not device_defs:
        return defaults

    result = dict(defaults)
    for key in ("model", "compute_type"):
        if cli_values.get(key) is None and key not in config:
            if key in device_defs:
                result[key] = device_defs[key]
    return result
