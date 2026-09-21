import ctypes
import glob
import os
import sys
import types

import pytest

from local_transcriber._cuda_bootstrap import ensure_cublas_loadable, is_cublas_available


def test_ensure_cublas_no_nvidia_package(monkeypatch):
    """Без nvidia-cublas-cu12 -- ничего не падает."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "nvidia.cublas", None)

    ensure_cublas_loadable()  # не должно бросать исключений


def test_ensure_cublas_loads_library(monkeypatch, tmp_path):
    """С nvidia.cublas -- вызывает ctypes.CDLL с полным путём и RTLD_GLOBAL."""
    monkeypatch.setattr(sys, "platform", "linux")

    # Создаём фейковый nvidia.cublas с lib/libcublas.so.12
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    fake_so = lib_dir / "libcublas.so.12"
    fake_so.touch()

    # Мокаем родительский пакет nvidia (иначе import nvidia.cublas упадёт)
    fake_nvidia = types.ModuleType("nvidia")
    fake_nvidia.__path__ = [str(tmp_path)]

    fake_cublas = types.ModuleType("nvidia.cublas")
    fake_cublas.__path__ = [str(tmp_path)]
    fake_nvidia.cublas = fake_cublas

    monkeypatch.setitem(sys.modules, "nvidia", fake_nvidia)
    monkeypatch.setitem(sys.modules, "nvidia.cublas", fake_cublas)

    calls = []
    monkeypatch.setattr(ctypes, "CDLL", lambda path, mode=0: calls.append((path, mode)))

    ensure_cublas_loadable()

    assert len(calls) == 1
    assert calls[0][0] == str(fake_so)
    assert calls[0][1] == ctypes.RTLD_GLOBAL


def test_ensure_cublas_skips_macos(monkeypatch):
    """На macOS bootstrap не загружает CUDA."""
    monkeypatch.setattr(sys, "platform", "darwin")
    ensure_cublas_loadable()  # не должно бросать исключений


def test_windows_optional_package_absent(monkeypatch):
    """Windows работает без extra и не меняет поиск DLL."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "nvidia.cublas", None)
    ensure_cublas_loadable()


@pytest.mark.parametrize("broken", [False, True])
def test_windows_preloads_dlls_and_keeps_handles(monkeypatch, tmp_path, broken):
    """DLL загружаются по порядку один раз, ошибки не скрываются."""
    from local_transcriber import _cuda_bootstrap as bootstrap

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    nvidia = types.ModuleType("nvidia")
    nvidia.__path__ = []
    cublas = types.ModuleType("nvidia.cublas")
    cublas.__path__ = [str(tmp_path)]
    nvidia.cublas = cublas
    monkeypatch.setitem(sys.modules, "nvidia", nvidia)
    monkeypatch.setitem(sys.modules, "nvidia.cublas", cublas)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(bootstrap, "_dll_directories", {})
    monkeypatch.setattr(bootstrap, "_dll_libraries", {})
    calls = []
    directory_handle = object()
    library_handle = object()
    failure = OSError("cublas64_12.dll: dependency not found")

    def add_directory(path):
        calls.append(("directory", path))
        return directory_handle

    def load_library(path):
        calls.append(("load", path))
        if broken:
            raise failure
        return library_handle

    monkeypatch.setattr(os, "add_dll_directory", add_directory, raising=False)
    monkeypatch.setattr(ctypes, "WinDLL", load_library, raising=False)
    original_path = os.environ.get("PATH")

    if broken:
        with pytest.raises(OSError) as caught:
            ensure_cublas_loadable()
        assert caught.value is failure
        assert bootstrap._dll_libraries == {}
    else:
        ensure_cublas_loadable()
        ensure_cublas_loadable()
        assert calls == [
            ("directory", str(bin_dir)),
            ("load", str(bin_dir / "cublasLt64_12.dll")),
            ("load", str(bin_dir / "cublas64_12.dll")),
        ]
        assert list(bootstrap._dll_libraries.values()) == [library_handle] * 2
    assert bootstrap._dll_directories[str(bin_dir)] is directory_handle
    assert os.environ.get("PATH") == original_path


def _nvidia_cublas_installed() -> bool:
    """Проверяет, что pip-пакет nvidia-cublas-cu12 установлен."""
    try:
        import nvidia.cublas  # type: ignore[import-untyped]

        cublas_paths = getattr(nvidia.cublas, "__path__", None)
        if not cublas_paths:
            return False
        lib_dir = os.path.join(cublas_paths[0], "lib")
        return any(glob.glob(os.path.join(lib_dir, "libcublas.so.12*")))
    except ImportError:
        return False


def _system_cublas_available() -> bool:
    """Проверяет, что libcublas.so.12 доступна через системный линкер (без bootstrap)."""
    try:
        ctypes.CDLL("libcublas.so.12")
        return True
    except OSError:
        return False


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="CUDA bootstrap только для Linux",
)
@pytest.mark.skipif(
    not _nvidia_cublas_installed(),
    reason="nvidia-cublas-cu12 не установлен",
)
def test_bootstrap_makes_cublas_resolvable():
    """Bootstrap из pip-пакета делает libcublas.so.12 резолвимой.

    Тест проходит ТОЛЬКО если:
    1. nvidia-cublas-cu12 установлен (иначе skip)
    2. libcublas НЕ доступна через системный линкер до bootstrap
       (иначе skip -- тест не может доказать, что сработал именно bootstrap)
    3. После ensure_cublas_loadable() -- libcublas доступна
    """
    if _system_cublas_available():
        pytest.skip(
            "libcublas.so.12 уже доступна через системный линкер -- "
            "невозможно проверить, что сработал именно bootstrap"
        )

    ensure_cublas_loadable()
    assert is_cublas_available(), (
        "nvidia-cublas-cu12 установлен, но после bootstrap "
        "libcublas.so.12 всё ещё не резолвится через dlopen"
    )
