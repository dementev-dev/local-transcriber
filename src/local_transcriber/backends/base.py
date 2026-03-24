"""Протокол бэкенда транскрипции."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from local_transcriber.types import Segment, TranscribeResult


class Backend(Protocol):
    """Минимальный интерфейс бэкенда транскрипции.

    Бэкенды реализуют этот протокол (structural typing) —
    наследование не требуется.
    """

    def ensure_model_available(
        self,
        model_name: str,
        compute_type: str,
        on_status: Callable[[str], None] | None = None,
    ) -> str:
        """Гарантирует наличие модели, возвращает путь к файлам."""
        ...

    def create_model(
        self,
        model_path: str,
        device: str,
        compute_type: str,
        cpu_threads: int = 0,
    ) -> Any:
        """Создаёт модель. Возвращает backend-специфичный объект.

        cpu_threads: число потоков для CPU inference (0 = дефолт библиотеки).
        """
        ...

    def transcribe(
        self,
        model: Any,
        file_path: Path,
        language: str | None,
        on_segment: Callable[[Segment], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> TranscribeResult:
        """Транскрибирует файл, возвращает результат."""
        ...
