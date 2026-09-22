# Contributing

## Быстрый старт для разработчика

```bash
git clone https://github.com/dementev-dev/local-transcriber
cd local-transcriber
uv sync
```

## Запуск

```bash
uv run transcribe meeting.mp4        # CLI
uv run pytest                         # тесты
uv run pytest -v                      # подробный вывод тестов
```

## Структура проекта

```
src/local_transcriber/
├── cli.py           # Точка входа CLI (typer)
├── config.py        # Загрузка .transcriber.toml, device-aware дефолты
├── formatter.py     # Форматирование результата в markdown
├── transcriber.py   # Обёртка над faster-whisper (загрузка модели, транскрипция)
└── utils.py         # Утилиты: валидация файлов, детект устройства, глобы
tests/
├── test_cli.py      # Тесты CLI (typer runner + моки)
└── ...
```

## Соглашения

- Тесты: `uv run pytest` должен проходить перед PR
- Стиль: стандартный Python (ruff-совместимый)
- Коммиты: [Conventional Commits](https://www.conventionalcommits.org/)
- Язык кода: английский (имена переменных/функций); docstrings, комментарии и UI-строки — русский

## Архитектура

```
CLI (cli.py)
  → config.py: загрузка .transcriber.toml, каскад дефолтов
  → utils.py: валидация файлов, определение устройства
  → transcriber.py: Transcriber — разрешение устройства, загрузка модели один раз на запуск, транскрипция, fallback
  → formatter.py: сегменты → markdown с таймкодами
  → запись результата
```

## Ключевые архитектурные решения

- **Module выполнения** (`Transcriber` в `transcriber.py`) — один владелец модели, adapter'а и фактического исполнения на запуск. Fallback GPU→CPU (при загрузке и mid-stream) живёт внутри и доступен только при `strict_device=False` из Python API: явный device в CLI strict, `auto` — ONNX CPU.
- **Device-aware дефолты** — `model` и `compute_type` без явного значения разрешаются по устройству при загрузке (`float16`/`float32`). `float16` не работает на CPU, `float32` расточителен на GPU.
- **cuBLAS bootstrap** (`_cuda_bootstrap.py`) — preload через ctypes до импорта ctranslate2. pip-пакет `nvidia-cublas-cu12` ставит `.so` в нестандартное место, а `LD_LIBRARY_PATH` нельзя изменить в рантайме.
- **Батч-режим** — 3 фазы (prescan → создание `Transcriber` → transcribe). Модель загружается один раз (~2-5 сек), невалидные файлы отсеиваются до загрузки; CLI не переносит состояние между файлами.
- **Ручной glob в utils** — typer на Windows не раскрывает `*.mp4`, поэтому глобы обрабатываются явно.

## Тестирование

- Все CLI-тесты через `typer.testing.CliRunner` с настоящим `Transcriber` и fake adapter, подменённым в `local_transcriber.transcriber.get_backend` (библиотеки не вызываются)
- Моки: `load_config`, `validate_input_file`, `write_transcript`
- Паттерн: `_cli_run()` — контекстный менеджер стандартного набора, отдаёт `(backend, write_transcript)`
- `_make_result()` / `_make_backend()` — фабрики тестовых данных

## Частые задачи

- **Новая CLI-опция**: добавить `typer.Option` в `main()` → добавить ключ в `HARDCODED_DEFAULTS` в `config.py` → написать тест
- **Поддержка нового формата**: добавить расширение в `SUPPORTED_EXTENSIONS` в `utils.py`
- **Изменение формата вывода**: редактировать `format_transcript()` в `formatter.py`

## Как сделать PR

1. Форкните репозиторий
2. Создайте ветку: `git checkout -b feat/my-feature`
3. Убедитесь, что тесты проходят: `uv run pytest`
4. Откройте Pull Request с описанием изменений
