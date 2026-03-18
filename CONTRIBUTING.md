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
  → transcriber.py: загрузка модели (с CUDA fallback), транскрипция
  → formatter.py: сегменты → markdown с таймкодами
  → запись результата
```

## Ключевые архитектурные решения

- **CUDA fallback** — двухуровневый: при загрузке модели и при транскрипции (mid-stream). GPU может быть видна через nvidia-smi, но не иметь достаточно VRAM.
- **Device-aware дефолты** — `compute_type` зависит от устройства (`float16`/`float32`). `float16` не работает на CPU, `float32` расточителен на GPU.
- **cuBLAS bootstrap** (`_cuda_bootstrap.py`) — preload через ctypes до импорта ctranslate2. pip-пакет `nvidia-cublas-cu12` ставит `.so` в нестандартное место, а `LD_LIBRARY_PATH` нельзя изменить в рантайме.
- **Батч-режим** — 3 фазы (prescan → load model → transcribe). Модель загружается один раз (~2-5 сек), невалидные файлы отсеиваются до загрузки.
- **Ручной glob в utils** — typer на Windows не раскрывает `*.mp4`, поэтому глобы обрабатываются явно.

## Тестирование

- Все CLI-тесты через `typer.testing.CliRunner` + моки (faster-whisper не вызывается)
- Моки: `load_config`, `validate_input_file`, `detect_device`, `ensure_model_available`, `load_model`, `_transcribe_file`, `write_transcript`
- Паттерн: `_single_patches()` — хелпер для стандартного happy-path набора моков
- `_make_result()` / `_make_tfr()` — фабрики тестовых данных

## Частые задачи

- **Новая CLI-опция**: добавить `typer.Option` в `main()` → добавить ключ в `HARDCODED_DEFAULTS` в `config.py` → написать тест
- **Поддержка нового формата**: добавить расширение в `SUPPORTED_EXTENSIONS` в `utils.py`
- **Изменение формата вывода**: редактировать `format_transcript()` в `formatter.py`

## Как сделать PR

1. Форкните репозиторий
2. Создайте ветку: `git checkout -b feat/my-feature`
3. Убедитесь, что тесты проходят: `uv run pytest`
4. Откройте Pull Request с описанием изменений
