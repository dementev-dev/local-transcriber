import sys
from pathlib import Path

import pytest

from local_transcriber import context_menu


def _prepare_exe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    scripts_dir = tmp_path / "venv" / "Scripts"
    scripts_dir.mkdir(parents=True)
    python_exe = scripts_dir / "python.exe"
    transcribe_exe = scripts_dir / "transcribe.exe"
    python_exe.write_bytes(b"")
    transcribe_exe.write_bytes(b"")
    monkeypatch.setattr(context_menu.sys, "executable", str(python_exe))
    return transcribe_exe


def test_install_menu_creates_expected_cmd(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    monkeypatch.setattr(context_menu, "CMD_ENCODING", "utf-8")
    transcribe_exe = _prepare_exe(tmp_path, monkeypatch)

    cmd_path = context_menu.install_menu()

    assert cmd_path.name == "Transcribe.cmd"
    assert cmd_path.exists()
    assert cmd_path.read_bytes() == (
        f'@echo off\r\n"{transcribe_exe}" %*\r\npause\r\n'.encode("utf-8")
    )
    assert b"chcp" not in cmd_path.read_bytes().lower()


@pytest.mark.skipif(sys.platform != "win32", reason="Кодировка oem доступна только на Windows")
def test_install_menu_writes_real_oem_encoding_on_windows(tmp_path, monkeypatch):
    appdata = tmp_path / "AppData" / "Roaming"
    scripts_dir = tmp_path / "проект" / "Scripts"
    scripts_dir.mkdir(parents=True)
    python_exe = scripts_dir / "python.exe"
    transcribe_exe = scripts_dir / "transcribe.exe"
    python_exe.write_bytes(b"")
    transcribe_exe.write_bytes(b"")
    monkeypatch.setenv("APPDATA", str(appdata))
    monkeypatch.setattr(context_menu.sys, "executable", str(python_exe))

    cmd_path = context_menu.install_menu()

    assert context_menu.CMD_ENCODING == "oem"
    assert cmd_path.read_bytes() == (
        f'@echo off\r\n"{transcribe_exe}" %*\r\npause\r\n'.encode("oem")
    )


def test_install_menu_overwrites_existing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    monkeypatch.setattr(context_menu, "CMD_ENCODING", "utf-8")
    _prepare_exe(tmp_path, monkeypatch)

    cmd_path = context_menu.install_menu()
    cmd_path.write_text("old", encoding="utf-8")

    second_path = context_menu.install_menu()

    assert second_path == cmd_path
    assert "old" not in cmd_path.read_text(encoding="utf-8")
    assert "%*" in cmd_path.read_text(encoding="utf-8")


def test_install_menu_creates_missing_sendto_dir(tmp_path, monkeypatch):
    appdata = tmp_path / "AppData" / "Roaming"
    monkeypatch.setenv("APPDATA", str(appdata))
    monkeypatch.setattr(context_menu, "CMD_ENCODING", "utf-8")
    _prepare_exe(tmp_path, monkeypatch)

    cmd_path = context_menu.install_menu()

    assert cmd_path.parent == appdata / "Microsoft" / "Windows" / "SendTo"
    assert cmd_path.parent.is_dir()


def test_install_menu_oem_encoding_error_is_runtime_error(tmp_path, monkeypatch):
    appdata = tmp_path / "AppData" / "Roaming"
    cmd_path = appdata / "Microsoft" / "Windows" / "SendTo" / "Transcribe.cmd"
    monkeypatch.setenv("APPDATA", str(appdata))
    monkeypatch.setattr(context_menu, "CMD_ENCODING", "ascii")
    monkeypatch.setattr(
        context_menu,
        "get_transcribe_exe",
        lambda: tmp_path / "测试" / "transcribe.exe",
    )

    with pytest.raises(RuntimeError, match="OEM"):
        context_menu.install_menu()

    assert not cmd_path.exists()


def test_uninstall_menu_removes_file_and_missing_is_not_error(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    monkeypatch.setattr(context_menu, "CMD_ENCODING", "utf-8")
    _prepare_exe(tmp_path, monkeypatch)
    cmd_path = context_menu.install_menu()

    removed_path = context_menu.uninstall_menu()
    missing_path = context_menu.uninstall_menu()

    assert removed_path == cmd_path
    assert not cmd_path.exists()
    assert missing_path is None


def test_get_sendto_dir_requires_appdata(monkeypatch):
    monkeypatch.delenv("APPDATA", raising=False)

    with pytest.raises(RuntimeError, match="APPDATA"):
        context_menu.get_sendto_dir()


def test_get_transcribe_exe_requires_existing_exe(tmp_path, monkeypatch):
    scripts_dir = tmp_path / "venv" / "Scripts"
    scripts_dir.mkdir(parents=True)
    python_exe = scripts_dir / "python.exe"
    python_exe.write_bytes(b"")
    monkeypatch.setattr(context_menu.sys, "executable", str(python_exe))

    with pytest.raises(RuntimeError, match="uv sync"):
        context_menu.get_transcribe_exe()