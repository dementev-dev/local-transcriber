# plan.md — План реализации local-transcriber

> Источник требований: docs/PRD.md
> Каждый шаг — атомарный коммит. После выполнения шага — отметить `[x]`.
> Агент: реализуй шаги последовательно. Читай ТОЛЬКО свой текущий шаг + секции PRD, на которые он ссылается. Не реализуй функциональность из других шагов. Не рефактори код предыдущих шагов без явной необходимости.

---

## Шаг 1: Scaffold проекта (без ML-зависимостей)

- [x] Создать структуру:
  ```
  local-transcriber/
  ├── pyproject.toml
  ├── .gitignore
  ├── src/
  │   └── local_transcriber/
  │       ├── __init__.py
  │       ├── cli.py
  │       ├── transcriber.py
  │       ├── formatter.py
  │       └── utils.py
  └── tests/
      ├── __init__.py
      ├── test_formatter.py   # заглушка
      ├── test_transcriber.py # заглушка
      └── test_utils.py       # заглушка
  ```
- [x] `pyproject.toml`:
  - `name = "local-transcriber"`, `python = ">=3.10"`
  - `[project.scripts]`: `transcribe = "local_transcriber.cli:app"`
  - dependencies: `typer`, `rich` (НЕ faster-whisper — он добавляется в шаге 3)
  - dev-dependencies: `pytest`
- [x] `.gitignore`: `__pycache__/`, `.venv/`, `*.egg-info/`, `.mypy_cache/`, `dist/`, `*.pyc`
- [x] В каждом модуле — заглушки с сигнатурами и `raise NotImplementedError`:

  **utils.py**:
  ```python
  from pathlib import Path

  def check_ffmpeg() -> None: ...
  def detect_device(requested: str = "auto") -> str: ...  # возвращает только device
  def get_gpu_name() -> str | None: ...  # nvidia-smi → "NVIDIA GeForce RTX 3060" или None
  def validate_input_file(path: Path) -> Path: ...
  def build_output_path(input_path: Path, output: Path | None = None) -> Path: ...
  ```

  **transcriber.py**:
  ```python
  from collections.abc import Callable
  from dataclasses import dataclass
  from pathlib import Path

  @dataclass
  class Segment:
      start: float   # секунды
      end: float     # секунды
      text: str

  @dataclass
  class TranscribeResult:
      segments: list[Segment]
      language: str
      language_probability: float
      duration: float       # секунды
      device_used: str      # фактическое устройство ("cpu" / "cuda") — может отличаться от запрошенного после fallback

  def transcribe(
      file_path: Path,
      model_name: str = "large-v3",
      device: str = "auto",
      compute_type: str = "int8",
      language: str | None = None,
      on_segment: Callable[[Segment], None] | None = None,  # callback для --verbose (вызывается на каждый сегмент)
  ) -> TranscribeResult: ...
  ```

  **formatter.py**:
  ```python
  from datetime import datetime
  from pathlib import Path
  from .transcriber import TranscribeResult

  def format_timestamp(seconds: float, use_hours: bool = False) -> str: ...

  def format_transcript(
      result: TranscribeResult,
      source_filename: str,
      model_name: str,
      device_info: str,
      language_mode: str,       # "detected" | "forced"
      transcription_date: datetime | None = None,  # None → datetime.now()
  ) -> str: ...

  def write_transcript(content: str, output_path: Path) -> None: ...
  ```

  **cli.py**:
  ```python
  from pathlib import Path
  import typer
  app = typer.Typer()

  @app.command()
  def main(file: Path) -> None:
      typer.echo("TODO: not implemented")

  if __name__ == "__main__":
      app()
  ```

- [x] `uv sync` → `uv run transcribe --help` работает

**Критерий готовности**: `uv run transcribe --help` показывает аргументы. `uv run pytest` проходит (тесты пустые, но pytest находит test_formatter.py, test_transcriber.py и test_utils.py). Все модули импортируются без ошибок.

---

## Шаг 2: utils.py — проверки окружения

> PRD-ссылки: 3.1 (flow), 3.2 (опции --device), 3.4 (форматы), 4.2 (ffmpeg)

- [x] `check_ffmpeg()`:
  - `subprocess.run(["ffmpeg", "-version"], capture_output=True)`
  - При `FileNotFoundError` → `SystemExit` с сообщением и инструкцией: `apt install ffmpeg` / `winget install ffmpeg` / `brew install ffmpeg`
- [x] `detect_device(requested: str = "auto") -> str`:
  - Если `requested != "auto"` → вернуть `requested`
  - Иначе: проверить CUDA через `shutil.which("nvidia-smi")` как быстрый хинт
  - Если nvidia-smi найден → `"cuda"`
  - Иначе → `"cpu"`
  - **Не импортировать** ctranslate2 или torch здесь — faster-whisper ещё не в зависимостях
  - Точная проверка CUDA будет при загрузке модели (шаг 3), здесь — best effort
  - `--compute-type` остаётся независимым параметром CLI, не связан с detect_device
- [x] `get_gpu_name() -> str | None`:
  - `subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True)`
  - Вернуть первую строку stdout (strip) или `None` если nvidia-smi недоступен / ошибка
  - Используется для формирования `device_info` в шапке markdown: `"CUDA (NVIDIA GeForce RTX 3060)"` или `"CPU"`
- [x] `validate_input_file(path: Path) -> Path`:
  - Проверить: существует, является файлом (не директорией), размер > 0
  - Расширение из допустимых (PRD 3.4) → если нет, **warning** (не ошибка), продолжить
  - Вернуть `path.resolve()`
- [x] `build_output_path(input_path: Path, output: Path | None = None) -> Path`:
  - Если `output` задан → вернуть его
  - Иначе → `input_path.with_stem(input_path.stem + "-transcript").with_suffix(".md")`

- [x] Тесты в `tests/test_utils.py`:
  - `test_validate_input_file_not_found` — несуществующий файл → ошибка
  - `test_validate_input_file_empty` — пустой файл → ошибка
  - `test_validate_input_file_unknown_ext` — `.txt` → warning, но не ошибка
  - `test_validate_input_file_ok` — валидный файл → возвращает resolved path
  - `test_build_output_path_default` — без `--output` → `*-transcript.md`
  - `test_build_output_path_custom` — с `--output` → возвращает его
  - `test_detect_device_explicit` — `requested="cpu"` → `"cpu"`
  - `test_get_gpu_name_no_nvidia_smi` — nvidia-smi недоступен → `None`
  - `test_get_gpu_name_success` — mock nvidia-smi → возвращает строку с именем GPU

**Критерий готовности**: `uv run pytest tests/test_utils.py -v` — все тесты зелёные. `uv run python -c "from local_transcriber.utils import check_ffmpeg, detect_device; check_ffmpeg(); print(detect_device())"` — работает (CPU fallback).

---

## Шаг 3: transcriber.py — обёртка над faster-whisper

> PRD-ссылки: 5.1 (faster-whisper), 4.1.1 (GPU совместимость), 6 (риски OOM)

- [x] `uv add faster-whisper` — добавить в зависимости
- [x] Реализовать `transcribe()`:
  - Создать `WhisperModel(model_name, device=device, compute_type=compute_type)`
  - При ошибке загрузки на CUDA (OOM, CUDA error) → **поймать**, вывести warning, **повторить с device="cpu"**
  - Запомнить фактический device → записать в `TranscribeResult.device_used`
  - `model.transcribe(str(file_path), language=language if language != "auto" else None)`
  - faster-whisper возвращает `(segment_generator, info)` — итерировать generator, для каждого сегмента вызвать `on_segment(segment)` если callback передан, затем собрать в `list[Segment]`
  - Заполнить `TranscribeResult` из info (language, duration и т.д.)
- [x] Обработка ошибок:
  - `RuntimeError` с "CUDA" / "out of memory" → fallback на CPU + warning
  - Ошибка ffmpeg (невалидный медиафайл) → пробросить с понятным текстом
- [x] Тесты в `tests/test_transcriber.py` (mock WhisperModel, без реальной модели):
  - `test_transcribe_collects_segments` — mock возвращает 3 сегмента → результат содержит 3 Segment
  - `test_transcribe_calls_on_segment` — callback вызывается для каждого сегмента
  - `test_transcribe_cuda_fallback` — mock бросает RuntimeError("CUDA") при device="cuda" → fallback, `device_used == "cpu"`
  - `test_transcribe_device_used` — без fallback → `device_used` совпадает с запрошенным

**Критерий готовности**: `uv run pytest tests/test_transcriber.py -v` — зелёное. Дополнительно на машине агента — `transcribe()` работает с `device="cpu"`, `model="tiny"`. Полноценная проверка с large-v3 и GPU — на локальной машине.


---

## Шаг 4: formatter.py + тесты

> PRD-ссылки: 3.3 (формат выходного файла)

- [x] `format_timestamp(seconds: float, use_hours: bool = False) -> str`:
  - `False` → `"01:23.45"` (MM:SS.ss)
  - `True` → `"01:23:45.67"` (HH:MM:SS.ss)
  - Сотые — всегда 2 знака после точки
- [x] `format_transcript(...)`:
  - Шапка по шаблону PRD 3.3 (заголовок, метаданные, разделитель)
  - `language_mode`: `"detected"` если CLI получил `--language auto`, `"forced"` если язык задан явно
  - В шапке: `**Язык**: {language} ({language_mode})` → например `ru (detected)` или `en (forced)`
  - `transcription_date`: если `None` → `datetime.now()`. Формат в шапке: `YYYY-MM-DD HH:MM:SS`
  - Автоматически `use_hours=True` если `result.duration > 3600`
  - Сегменты: `[MM:SS.ss - MM:SS.ss] текст\n\n`
  - Если `len(result.segments) == 0` → после разделителя: `\n*Речь не обнаружена.*\n` (файл создаётся с полной шапкой метаданных; warning в stderr выводит CLI в шаге 5)
- [x] `write_transcript(content: str, output_path: Path)`:
  - `open(output_path, "w", encoding="utf-8")`
- [x] Тесты в `tests/test_formatter.py`:
  - `test_format_timestamp_minutes` — обычный таймкод
  - `test_format_timestamp_hours` — формат с часами
  - `test_format_transcript_basic` — проверить шапку + пару сегментов
  - `test_format_transcript_empty` — 0 сегментов → содержит "Речь не обнаружена"
  - `test_format_transcript_long` — duration > 3600 → таймкоды с часами

**Критерий готовности**: `uv run pytest tests/test_formatter.py -v` — все тесты зелёные.

---

## Шаг 5: cli.py — связка всех модулей, happy path

> PRD-ссылки: 3.1 (flow), 3.2 (CLI-интерфейс)

- [ ] Typer command с аргументами и опциями по PRD 3.2:
  - `file: Path` — позиционный аргумент
  - `--model` / `-m` → default `"large-v3"`
  - `--language` / `-l` → default `"auto"`
  - `--output` / `-o` → optional Path
  - `--device` / `-d` → default `"auto"`
  - `--compute-type` → default `"int8"`
  - `--verbose` / `-v` → flag, default False
- [ ] Happy path flow:
  1. `check_ffmpeg()`
  2. `validate_input_file(file)`
  3. `detect_device(device)` → получить device; `--compute-type` используется как есть (независим от device)
  4. rich Console → stderr: информация о запуске (модель, устройство, файл)
  5. rich Spinner/Status во время транскрипции
  6. `transcribe(...)` — передать `on_segment=<callback>` если `--verbose`
  7. Если 0 сегментов → `console.print("⚠ Речь не обнаружена в файле ...", style="yellow")`
  8. `format_transcript(...)` — `device_info`: если `result.device_used == "cuda"` → `"CUDA ({get_gpu_name() or 'Unknown GPU'})"`, иначе `"CPU"`
  9. `write_transcript(...)`
  10. `console.print("✓ Транскрипт сохранён: <путь>", style="green")`
  11. Статистика: кол-во сегментов, время работы (замерить через `time.monotonic()`)
- [ ] Exit codes: 0 — успех (включая пустую речь), 1 — ошибка

**Критерий готовности**: `uv run transcribe test.mp3` — создаёт корректный .md файл (проверить на локальной машине с реальным файлом).


---

## Шаг 6: Error handling и UX polish

> PRD-ссылки: 6 (риски)

- [ ] Graceful Ctrl+C: перехват `KeyboardInterrupt` в cli.py → `console.print("Прервано пользователем", style="yellow")` + `raise SystemExit(130)`
- [ ] Красивые ошибки: обернуть main в try/except, для пользовательских ошибок (файл не найден, ffmpeg, OOM) — вывод через rich без traceback; для неожиданных — traceback только с `--verbose`
- [ ] `--verbose` режим: реализуется через `on_segment` callback в `transcribe()` (уже заложен в шаге 3) — печатать каждый сегмент в stderr по мере поступления
- [ ] Проверка: неподдерживаемое расширение → warning, но попытка продолжить

**Критерий готовности**: ручной прогон edge cases — несуществующий файл, .txt файл, Ctrl+C во время работы.


---

## Шаг 7: README.md

> PRD-ссылки: 4.2 (требования), 4.1.1 (GPU таблица), 4.3 (кроссплатформенность)

- [ ] Описание: что делает, зачем
- [ ] Требования: Python ≥ 3.10, ffmpeg, (опционально) NVIDIA GPU + CUDA
- [ ] Установка:
  ```bash
  git clone <repo>
  cd local-transcriber
  uv sync
  ```
- [ ] Использование: 3-4 примера команд (простой, с языком, с моделью, CPU)
- [ ] Установка ffmpeg: Linux (`apt`), Windows (`winget`/`scoop`), WSL2, macOS (`brew`)
- [ ] GPU и CUDA: краткое пояснение, ссылка на NVIDIA docs, что CTranslate2 ставит нужное
- [ ] Таблица моделей: имя, размер на диске, VRAM (int8), относительная скорость, качество
- [ ] Пример выходного файла (сокращённый)

**Критерий готовности**: коллега может по README установить и запустить на Windows/WSL2 без вопросов.


---

## Инструкция для агента

Команда на каждый шаг:
```
Прочитай docs/PRD.md (только секции, указанные в текущем шаге) и plan.md (только текущий шаг).
Реализуй шаг N.
После реализации — отметь все чекбоксы шага как [x] в plan.md.
Не трогай код и чекбоксы из других шагов.
```