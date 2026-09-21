"""Оркестрация транскрипции: выбор бэкенда, загрузка модели, fallback."""

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_transcriber.backends import get_backend
from local_transcriber.config import DEVICE_DEFAULTS, HARDCODED_DEFAULTS

# Re-export из types.py для обратной совместимости
from local_transcriber.types import (  # noqa: F401
    Segment,
    TranscribeResult,
    WordTimestampsUnavailableError,
)
from local_transcriber.utils import detect_device, get_gpu_name, get_intel_gpu_name

# Движок распознавания для каждого значения device; остальное, как и в
# get_backend, уходит в faster-whisper. Способ исполнения (CPU/GPU) остаётся
# в самом значении device и трактуется adapter'ом.
_ENGINES: dict[str, str] = {
    "cpu": "faster-whisper",
    "cuda": "faster-whisper",
    "openvino-cpu": "openvino",
    "openvino-gpu": "openvino",
    "onnx": "onnx-asr",
}


@dataclass(frozen=True)
class ExecutionRequest:
    """Запрошенные настройки запуска до разрешения умолчаний.

    ``None`` в model/compute_type означает device-aware умолчание;
    ``strict_device=None`` — strict для любого явного device, кроме ``auto``.
    """

    device: str = "auto"
    model: str | None = None
    compute_type: str | None = None
    language: str | None = None
    cpu_threads: int = 0
    strict_device: bool | None = None
    require_word_timestamps: bool = False


@dataclass(frozen=True)
class ExecutionInfo:
    """Сведения о выполнении: что запрошено, что выбрано и чем выполняется."""

    requested_device: str
    resolved_device: str  # выбор до загрузки; отличается от device после fallback
    device: str
    engine: str
    model: str
    compute_type: str
    cpu_threads: int
    word_timestamps_available: bool
    description: str  # строка исполнения для шапки транскрипта
    runtime: dict[str, str] = field(default_factory=dict)


class Transcriber:
    """Module выполнения распознавания на один запуск.

    Владеет загруженной моделью, adapter'ом, фактическими настройками
    и разрешёнными переходами исполнения. Caller передаёт файлы по одному
    и получает результат; состояние между файлами не переносит.
    """

    def __init__(self, request: ExecutionRequest) -> None:
        self._request = request
        self._strict = (
            request.strict_device
            if request.strict_device is not None
            else request.device != "auto"
        )
        self._compute_type_explicit = request.compute_type is not None
        resolved = detect_device(request.device)
        defaults = DEVICE_DEFAULTS.get(resolved, {})
        self._model_name = request.model or defaults.get(
            "model", HARDCODED_DEFAULTS["model"]
        )
        self._compute_type = request.compute_type or defaults.get(
            "compute_type", HARDCODED_DEFAULTS["compute_type"]
        )
        self._resolved_device = resolved
        self._device = resolved
        self._backend: Any = None
        self._model: Any = None
        self._info: ExecutionInfo | None = None

    @property
    def execution(self) -> ExecutionInfo:
        """Текущие сведения о выполнении; до prepare() — без загрузки."""
        if self._info is None:
            raise RuntimeError("Модель ещё не подготовлена: вызовите prepare()")
        return self._info

    def prepare(self, on_status: Callable[[str], None] | None = None) -> ExecutionInfo:
        """Загружает модель и проверяет возможности до первого файла. Идемпотентно."""
        if self._info is not None:
            return self._info
        try:
            self._load(self._device, on_status)
        except (RuntimeError, ValueError) as exc:
            if not self._may_fall_back(exc):
                raise
            warnings.warn(
                f"Не удалось загрузить модель на {self._device}: {exc}. "
                "Переключение на CPU.",
                stacklevel=2,
            )
            self._load("cpu", on_status)
        if (
            self._request.require_word_timestamps
            and not self._info.word_timestamps_available
        ):
            raise ValueError(
                "Выбранный движок или модель не поддерживает пословные таймкоды"
            )
        return self._info

    def transcribe(
        self,
        file_path: Path,
        on_segment: Callable[[Segment], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> TranscribeResult:
        """Транскрибирует один файл; при разрешённом fallback перезагружает модель."""
        self.prepare(on_status)
        lang_arg = self._request.language
        if lang_arg == "auto":
            lang_arg = None
        try:
            result = self._recognize(file_path, lang_arg, on_segment, on_status)
        except (RuntimeError, ValueError) as exc:
            if not self._may_fall_back(exc):
                raise
            warnings.warn(
                f"Ошибка при транскрипции на {self._device}: {exc}. "
                "Переключение на CPU и повтор.",
                stacklevel=2,
            )
            self._load("cpu", on_status)
            result = self._recognize(file_path, lang_arg, on_segment, on_status)
        return result

    def _recognize(
        self,
        file_path: Path,
        language: str | None,
        on_segment: Callable[[Segment], None] | None,
        on_status: Callable[[str], None] | None,
    ) -> TranscribeResult:
        _notify_status(on_status, "Транскрибирую...")
        result = self._backend.transcribe(
            self._model, file_path, language, on_segment, on_status
        )
        result.device_used = self._device
        return result

    def _may_fall_back(self, exc: BaseException) -> bool:
        """Разрешённый переход: не strict, не CPU и ошибка самого движка."""
        if self._strict or self._device == "cpu":
            return False
        return _is_backend_error(exc, self._device)

    def _load(self, device: str, on_status: Callable[[str], None] | None) -> None:
        backend = get_backend(device, compute_type_explicit=self._compute_type_explicit)
        model_path = backend.ensure_model_available(
            self._model_name, self._compute_type, on_status
        )
        _notify_status(on_status, f"Инициализирую модель на {device}...")
        model = backend.create_model(
            model_path,
            device,
            self._compute_type,
            cpu_threads=self._request.cpu_threads,
        )
        self._backend = backend
        self._model = model
        self._device = _refine_openvino_device(device, backend)
        self._info = self._describe()

    def _describe(self) -> ExecutionInfo:
        backend = self._backend
        compute_type = (
            getattr(backend, "actual_compute_type", None) or self._compute_type
        )
        return ExecutionInfo(
            requested_device=self._request.device,
            resolved_device=self._resolved_device,
            device=self._device,
            engine=_ENGINES.get(self._device, "faster-whisper"),
            model=self._model_name,
            compute_type=compute_type,
            cpu_threads=self._request.cpu_threads,
            word_timestamps_available=bool(backend.word_timestamps_available),
            description=_describe_device(self._device),
            runtime=dict(backend.runtime_info()),
        )


def _refine_openvino_device(device: str, backend: Any) -> str:
    """Заменяет запрошенный openvino-* на устройство, которое реально выбрал OpenVINO."""
    ov_dev = getattr(backend, "actual_ov_device", None)
    if ov_dev == "GPU" and device != "openvino-gpu":
        return "openvino-gpu"
    if ov_dev == "CPU" and device.startswith("openvino") and device != "openvino-cpu":
        return "openvino-cpu"
    return device


def _describe_device(device: str) -> str:
    """Строка исполнения для шапки транскрипта; GPU называется только по данным драйвера."""
    if device == "cuda":
        return f"CUDA ({get_gpu_name() or 'Unknown GPU'})"
    if device == "openvino-gpu":
        return f"OpenVINO ({get_intel_gpu_name() or 'Intel GPU'})"
    if device == "openvino-cpu":
        return "OpenVINO (CPU)"
    if device == "onnx":
        return "ONNX (CPU)"
    return "CPU"


def transcribe(
    file_path: Path,
    model_name: str | None = None,
    device: str = "auto",
    compute_type: str | None = None,
    language: str | None = None,
    on_segment: Callable[[Segment], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    strict_device: bool = False,
    cpu_threads: int = 0,
) -> TranscribeResult:
    """Публичный путь одного файла через тот же module выполнения, что и CLI.

    ``None`` в model_name/compute_type берёт умолчания выбранного исполнения.
    """
    run = Transcriber(
        ExecutionRequest(
            device=device,
            model=model_name,
            compute_type=compute_type,
            language=language,
            cpu_threads=cpu_threads,
            strict_device=strict_device,
        )
    )
    return run.transcribe(file_path, on_segment, on_status)


def ensure_model_available(
    model_name: str,
    device: str = "cpu",
    compute_type: str | None = None,
    on_status: Callable[[str], None] | None = None,
) -> str:
    """Публичный helper: гарантирует наличие модели для указанного бэкенда."""
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
