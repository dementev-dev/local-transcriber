"""
Preload CUDA-библиотек из pip-пакетов до импорта ctranslate2.

Проблема: ctranslate2 на Linux делает dlopen("libcublas.so.12"),
но не знает, что библиотека лежит внутри pip-пакета nvidia-cublas-cu12.
На Windows ctranslate2 решает это сам через os.add_dll_directory.

Решение: загружаем libcublas.so.12 по полному пути через ctypes.CDLL
с флагом RTLD_GLOBAL до первого import ctranslate2. Динамический линкер
кеширует загруженные библиотеки по soname — когда ctranslate2 потом
вызовет dlopen("libcublas.so.12"), линкер вернёт уже загруженный handle.

Почему нельзя просто os.environ["LD_LIBRARY_PATH"] = ...:
На Linux/glibc динамический линкер (ld.so) кеширует пути поиска
при первом вызове и НЕ перечитывает LD_LIBRARY_PATH из environ
в рамках уже запущенного процесса.
"""

import ctypes
import glob
import os
import sys


def ensure_cublas_loadable() -> None:
    """Загружает libcublas из nvidia-cublas-cu12 в адресное пространство процесса.

    Вызывать ДО первого import ctranslate2.
    Безопасно вызывать многократно и на платформах без nvidia-cublas-cu12.
    """
    if sys.platform != "linux":
        return

    try:
        import nvidia.cublas  # type: ignore[import-untyped]
    except ImportError:
        # nvidia-cublas-cu12 не установлен (Windows, macOS, или CPU-only setup)
        return

    # nvidia.cublas может быть namespace package (__file__ == None),
    # используем __path__ для определения директории пакета
    cublas_paths = getattr(nvidia.cublas, "__path__", None)
    if not cublas_paths:
        return
    cublas_lib_dir = os.path.join(cublas_paths[0], "lib")
    if not os.path.isdir(cublas_lib_dir):
        return

    # Ищем libcublas.so.12* (например libcublas.so.12, libcublas.so.12.4.2.1)
    # Загружаем с RTLD_GLOBAL чтобы символы были видны ctranslate2
    for so_path in sorted(glob.glob(os.path.join(cublas_lib_dir, "libcublas.so.12*"))):
        try:
            ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        break  # достаточно загрузить одну versioned .so


def is_cublas_available() -> bool:
    """Проверяет, что libcublas.so.12 реально резолвится через dlopen.

    Используется в тестах для проверки, что bootstrap сработал.
    На платформах без CUDA возвращает False.
    """
    if sys.platform != "linux":
        return False
    try:
        ctypes.CDLL("libcublas.so.12")
        return True
    except OSError:
        return False
