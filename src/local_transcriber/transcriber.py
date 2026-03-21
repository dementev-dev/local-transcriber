"""Оркестрация транскрипции: выбор бэкенда, загрузка модели, fallback."""

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from local_transcriber.backends import get_backend

# Re-export из types.py для обратной совместимости
from local_transcriber.types import (  # noqa: F401
    Segment,
    TranscribeFileResult,
    TranscribeResult,
)


def load_model(
    model_name: str,
    device: str,
    compute_type: str,
    on_status: Callable[[str], None] | None = None,
    strict_device: bool = False,
    compute_type_explicit: bool = False,
) -> tuple[Any, str, Any, str]:
    """Загружает модель: ensure + create с fallback.

    Возвращает (model, actual_device, backend, model_path).
    compute_type_explicit: True если пользователь явно указал --compute-type.
    """
    backend = get_backend(device, compute_type_explicit=compute_type_explicit)
    actual_device = device

    model_path = backend.ensure_model_available(model_name, compute_type, on_status)

    try:
        _notify_status(on_status, f"Инициализирую модель на {device}...")
        model = backend.create_model(model_path, device, compute_type)
    except (RuntimeError, ValueError) as exc:
        if device != "cpu" and _is_backend_error(exc, device):
            if strict_device:
                raise
            warnings.warn(
                f"Не удалось загрузить модель на {device}: {exc}. "
                "Переключение на CPU.",
                stacklevel=2,
            )
            actual_device = "cpu"
            backend = get_backend("cpu")
            model_path = backend.ensure_model_available(model_name, compute_type, on_status)
            _notify_status(on_status, "Инициализирую модель на cpu...")
            model = backend.create_model(model_path, "cpu", compute_type)
        else:
            raise

    return model, actual_device, backend, model_path


def _transcribe_file(
    model: Any,
    actual_device: str,
    backend: Any,
    model_path: str,
    file_path: Path,
    model_name: str,
    compute_type: str,
    language: str | None = None,
    on_segment: Callable[[Segment], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    strict_device: bool = False,
) -> TranscribeFileResult:
    """Транскрибирует один файл. При mid-stream fallback перезагружает модель."""
    lang_arg = language if language and language != "auto" else None

    try:
        _notify_status(on_status, "Транскрибирую...")
        result = backend.transcribe(model, file_path, lang_arg, on_segment, on_status)
        result.device_used = actual_device
    except (RuntimeError, ValueError) as exc:
        if actual_device != "cpu" and _is_backend_error(exc, actual_device):
            if strict_device:
                raise
            warnings.warn(
                f"Ошибка при транскрипции на {actual_device}: {exc}. "
                "Переключение на CPU и повтор.",
                stacklevel=2,
            )
            actual_device = "cpu"
            backend = get_backend("cpu")
            model_path = backend.ensure_model_available(model_name, compute_type, on_status)
            _notify_status(on_status, "Инициализирую модель на cpu...")
            model = backend.create_model(model_path, "cpu", compute_type)
            _notify_status(on_status, "Транскрибирую...")
            result = backend.transcribe(model, file_path, lang_arg, on_segment, on_status)
            result.device_used = actual_device
        else:
            raise

    return TranscribeFileResult(
        result=result,
        model=model,
        actual_device=actual_device,
        backend=backend,
        model_path=model_path,
    )


def transcribe(
    file_path: Path,
    model_name: str = "large-v3",
    device: str = "auto",
    compute_type: str = "int8",
    language: str | None = None,
    on_segment: Callable[[Segment], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    strict_device: bool = False,
) -> TranscribeResult:
    """High-level API: загрузка модели + транскрипция за один вызов."""
    model, actual_device, backend, model_path = load_model(
        model_name, device, compute_type, on_status, strict_device,
    )
    tfr = _transcribe_file(
        model, actual_device, backend, model_path,
        file_path, model_name, compute_type,
        language, on_segment, on_status, strict_device,
    )
    return tfr.result


def ensure_model_available(
    model_name: str,
    device: str = "cpu",
    compute_type: str = "float32",
    on_status: Callable[[str], None] | None = None,
) -> str:
    """Публичный helper: гарантирует наличие модели для указанного бэкенда."""
    backend = get_backend(device)
    return backend.ensure_model_available(model_name, compute_type, on_status)


def _is_cuda_error(exc: BaseException) -> bool:
    """Проверка CUDA ошибок — используется в cli.py для Windows-диагностики."""
    msg = str(exc).lower()
    return any(k in msg for k in ("cuda", "cublas", "cudnn", "out of memory"))


def _is_backend_error(exc: BaseException, device: str) -> bool:
    """Определяет, связана ли ошибка с конкретным бэкендом (а не с пользовательскими данными)."""
    if device in ("cuda", "cpu"):
        return _is_cuda_error(exc)
    if device == "openvino":
        return _is_openvino_error(exc)
    return False


def _is_openvino_error(exc: BaseException) -> bool:
    """Проверка ошибок OpenVINO runtime."""
    msg = str(exc).lower()
    return any(k in msg for k in ("openvino", "ov_", "inference_engine"))


def _notify_status(on_status: Callable[[str], None] | None, message: str) -> None:
    if on_status is not None:
        on_status(message)
