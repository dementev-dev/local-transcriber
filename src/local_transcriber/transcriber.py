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
    WordTimestampsUnavailableError,
)


def load_model(
    model_name: str,
    device: str,
    compute_type: str,
    on_status: Callable[[str], None] | None = None,
    strict_device: bool = False,
    compute_type_explicit: bool = False,
    cpu_threads: int = 0,
) -> tuple[Any, str, Any, str]:
    """Загружает модель: ensure + create с fallback.

    Возвращает (model, actual_device, backend, model_path).
    compute_type_explicit: True если пользователь явно указал --compute-type.
    cpu_threads: число потоков для CPU inference (0 = дефолт библиотеки).
    """
    backend = get_backend(device, compute_type_explicit=compute_type_explicit)
    actual_device = device

    model_path = backend.ensure_model_available(model_name, compute_type, on_status)

    try:
        _notify_status(on_status, f"Инициализирую модель на {device}...")
        model = backend.create_model(
            model_path, device, compute_type, cpu_threads=cpu_threads
        )
        # Резолвим actual_device по реальному OpenVINO device
        ov_dev = getattr(backend, "actual_ov_device", None)
        if ov_dev == "GPU" and actual_device != "openvino-gpu":
            actual_device = "openvino-gpu"
        elif (
            ov_dev == "CPU"
            and actual_device.startswith("openvino")
            and actual_device != "openvino-cpu"
        ):
            actual_device = "openvino-cpu"
    except (RuntimeError, ValueError) as exc:
        if device != "cpu" and _is_backend_error(exc, device):
            if strict_device:
                raise
            warnings.warn(
                f"Не удалось загрузить модель на {device}: {exc}. Переключение на CPU.",
                stacklevel=2,
            )
            actual_device = "cpu"
            backend = get_backend("cpu")
            model_path = backend.ensure_model_available(
                model_name, compute_type, on_status
            )
            _notify_status(on_status, "Инициализирую модель на cpu...")
            model = backend.create_model(
                model_path, "cpu", compute_type, cpu_threads=cpu_threads
            )
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
    cpu_threads: int = 0,
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
            model_path = backend.ensure_model_available(
                model_name, compute_type, on_status
            )
            _notify_status(on_status, "Инициализирую модель на cpu...")
            model = backend.create_model(
                model_path, "cpu", compute_type, cpu_threads=cpu_threads
            )
            _notify_status(on_status, "Транскрибирую...")
            result = backend.transcribe(
                model, file_path, lang_arg, on_segment, on_status
            )
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
    cpu_threads: int = 0,
) -> TranscribeResult:
    """High-level API: загрузка модели + транскрипция за один вызов."""
    model, actual_device, backend, model_path = load_model(
        model_name,
        device,
        compute_type,
        on_status,
        strict_device,
        compute_type_explicit=True,  # Python API — caller explicitly chose compute_type
        cpu_threads=cpu_threads,
    )
    tfr = _transcribe_file(
        model,
        actual_device,
        backend,
        model_path,
        file_path,
        model_name,
        compute_type,
        language,
        on_segment,
        on_status,
        strict_device,
        cpu_threads=cpu_threads,
    )
    return tfr.result


def ensure_model_available(
    model_name: str,
    device: str = "cpu",
    compute_type: str | None = None,
    on_status: Callable[[str], None] | None = None,
) -> str:
    """Публичный helper: гарантирует наличие модели для указанного бэкенда."""
    from local_transcriber.config import DEVICE_DEFAULTS, HARDCODED_DEFAULTS

    if compute_type is None:
        device_defs = DEVICE_DEFAULTS.get(device, {})
        compute_type = device_defs.get(
            "compute_type", HARDCODED_DEFAULTS["compute_type"]
        )
        explicit = False
    else:
        explicit = True
    backend = get_backend(device, compute_type_explicit=explicit)
    return backend.ensure_model_available(model_name, compute_type, on_status)


def _is_cuda_error(exc: BaseException) -> bool:
    """Проверяет, относится ли ошибка к CUDA-бэкенду."""
    msg = str(exc).lower()
    return any(k in msg for k in ("cuda", "cublas", "cudnn", "out of memory"))


def cuda_error_hint(exc: BaseException, *, device: str | None = None) -> str | None:
    """Подсказывает действие только для распознанной причины CUDA-ошибки."""
    msg = str(exc).lower()
    if device == "cuda" and "requested " in msg and "compute type" in msg and (
        "do not support efficient" in msg
    ):
        return (
            "Тип вычислений несовместим с выбранным GPU/CUDA runtime. "
            "Установка extra не добавит аппаратную поддержку. Выберите "
            "поддерживаемый --compute-type или --device onnx / --device cpu."
        )
    if any(k in msg for k in (
        "no kernel image", "invalid device function", "unsupported gpu",
        "cublas_status_arch_mismatch", "cuda_error_no_binary_for_gpu",
    )):
        return (
            "GPU несовместим с выбранным CUDA runtime. Установка extra не исправит "
            "аппаратную несовместимость. Используйте --device onnx или --device cpu."
        )
    if any(k in msg for k in (
        "driver version is insufficient", "no cuda-capable device",
        "cuda error: no device", "cuda_error_no_device", "libcuda.so", "nvcuda.dll",
    )):
        return (
            "CUDA не видит совместимого GPU/драйвера. Проверьте драйвер NVIDIA "
            "и совместимость устройства с runtime. Extra не устанавливает драйвер."
        )
    if any(k in msg for k in ("cublas", "cudnn", "cudart")) and any(
        k in msg for k in (
            "not found", "cannot be loaded", "could not load", "could not find",
            "cannot open shared object", "no such file", "winerror 126",
        )
    ):
        return (
            "Не найдены CUDA-библиотеки или их зависимости. Подключите extra cuda "
            "в окружении приложения:\n"
            "  Из клона: uv tool install --force '.[cuda]'\n"
            "  Для разработки: uv sync --extra cuda\n"
            "При установке из Git используйте инструкцию CUDA в README. "
            "На Windows также нужен Visual C++ Runtime x64."
        )
    if "out of memory" in msg and any(k in msg for k in ("cuda", "cublas", "cudnn")):
        return (
            "Недостаточно памяти для выбранного CUDA-пути. Выберите меньшую модель "
            "или явно используйте --device onnx / --device cpu."
        )
    return None


def _is_backend_error(exc: BaseException, device: str) -> bool:
    """Определяет, связана ли ошибка с конкретным бэкендом (а не с пользовательскими данными)."""
    if isinstance(exc, WordTimestampsUnavailableError):
        return False
    if device in ("cuda", "cpu"):
        return _is_cuda_error(exc)
    if device.startswith("openvino"):
        return _is_openvino_error(exc)
    return False


def _is_openvino_error(exc: BaseException) -> bool:
    """Проверка ошибок OpenVINO runtime.

    OpenVINO runtime кидает RuntimeError с разнообразными сообщениями
    (openvino, ov_, inference, plugins, src/...). Пользовательские ошибки
    (файл не найден, неверный формат) приходят как FileNotFoundError/ValueError
    и не попадают сюда. Поэтому для RuntimeError считаем это backend failure.
    """
    return isinstance(exc, RuntimeError)


def _notify_status(on_status: Callable[[str], None] | None, message: str) -> None:
    if on_status is not None:
        on_status(message)
