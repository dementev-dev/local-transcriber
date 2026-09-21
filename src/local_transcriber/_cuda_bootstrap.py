"""
Подготовка CUDA-библиотек из extra перед созданием CUDA-модели.

Проблема: ctranslate2 на Linux делает dlopen("libcublas.so.12"),
но не знает, что библиотека лежит внутри pip-пакета nvidia-cublas-cu12.
На Windows ctranslate2 не добавляет nvidia/cublas/bin в поиск DLL.
Регистрируем каталог и загружаем cuBLAS Lt, затем cuBLAS по полному пути.
Храним DLL и регистрацию каталога до завершения процесса.

Решение: загружаем libcublas.so.12 по полному пути через ctypes.CDLL
с флагом RTLD_GLOBAL до создания CUDA-модели. Динамический линкер
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

# Закрытие регистрации удаляет каталог из поиска DLL, поэтому храним handles.
_dll_directories: dict[str, object] = {}
_dll_libraries: dict[str, object] = {}


def ensure_cublas_loadable() -> None:
    """Загружает cuBLAS из nvidia-cublas-cu12 в адресное пространство процесса.

    Вызывать перед созданием CUDA-модели; CPU-путь не требует bootstrap.
    Безопасно вызывать многократно и на платформах без nvidia-cublas-cu12.
    """
    if sys.platform not in ("linux", "win32"):
        return

    try:
        import nvidia.cublas  # type: ignore[import-untyped]
    except ImportError:
        # Optional-пакет не установлен; системная cuBLAS всё ещё может работать.
        return

    # nvidia.cublas может быть namespace package (__file__ == None),
    # используем __path__ для определения директории пакета
    cublas_paths = getattr(nvidia.cublas, "__path__", None)
    if not cublas_paths:
        return
    if sys.platform == "win32":
        for package_path in cublas_paths:
            bin_dir = os.path.join(package_path, "bin")
            if not os.path.isdir(bin_dir):
                continue
            if bin_dir not in _dll_directories:
                _dll_directories[bin_dir] = os.add_dll_directory(bin_dir)
            # cuBLAS зависит от Lt; явный preload также обслуживает LoadLibrary
            # внутри CTranslate2. Порядок здесь существенен.
            for filename in ("cublasLt64_12.dll", "cublas64_12.dll"):
                dll_path = os.path.join(bin_dir, filename)
                if dll_path not in _dll_libraries:
                    _dll_libraries[dll_path] = ctypes.WinDLL(dll_path)
            return
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
