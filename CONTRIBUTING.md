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
- Язык кода: английский (имена, комментарии); UI-строки для пользователя — русский

## Как сделать PR

1. Форкните репозиторий
2. Создайте ветку: `git checkout -b feat/my-feature`
3. Убедитесь, что тесты проходят: `uv run pytest`
4. Откройте Pull Request с описанием изменений
