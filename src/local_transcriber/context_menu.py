"""Установка пункта Transcribe в меню SendTo проводника Windows."""

import os
import sys
from pathlib import Path

CMD_NAME = "Transcribe.cmd"
CMD_ENCODING = "oem"


def get_sendto_dir() -> Path:
    """Возвращает путь к пользовательской папке SendTo."""
    appdata = os.environ.get("APPDATA")
    if appdata is None:
        raise RuntimeError("Переменная окружения APPDATA не задана.")
    return Path(appdata) / "Microsoft" / "Windows" / "SendTo"


def get_transcribe_exe() -> Path:
    """Возвращает путь к transcribe.exe рядом с текущим интерпретатором."""
    transcribe_exe = Path(sys.executable).parent / "transcribe.exe"
    if not transcribe_exe.exists():
        raise RuntimeError(
            f"Не найден transcribe.exe рядом с Python: {transcribe_exe}. "
            "Выполните uv sync и повторите установку пункта меню."
        )
    return transcribe_exe


def install_menu() -> Path:
    """Создаёт или обновляет Transcribe.cmd в папке SendTo."""
    sendto_dir = get_sendto_dir()
    transcribe_exe = get_transcribe_exe()
    cmd_path = sendto_dir / CMD_NAME
    content = f'@echo off\r\n"{transcribe_exe}" %*\r\npause\r\n'

    try:
        encoded_content = content.encode(CMD_ENCODING)
    except UnicodeEncodeError as exc:
        raise RuntimeError(
            "Путь к transcribe.exe содержит символы, которые нельзя записать "
            "в OEM-кодировке cmd.exe. Установите проект в путь без таких символов "
            "и повторите --install-menu."
        ) from exc

    sendto_dir.mkdir(parents=True, exist_ok=True)
    cmd_path.write_bytes(encoded_content)
    return cmd_path


def uninstall_menu() -> Path | None:
    """Удаляет Transcribe.cmd из папки SendTo, если он существует."""
    cmd_path = get_sendto_dir() / CMD_NAME
    if not cmd_path.exists():
        return None
    cmd_path.unlink()
    return cmd_path
