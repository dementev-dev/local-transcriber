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


def test_ensure_cublas_skips_non_linux(monkeypatch):
    """На не-Linux платформах -- no-op."""
    monkeypatch.setattr(sys, "platform", "win32")
    ensure_cublas_loadable()  # не должно бросать исключений


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
