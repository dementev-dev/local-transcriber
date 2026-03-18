# local-transcriber

Локальная транскрипция аудио и видео в markdown — без облака, без API-ключей.

```bash
transcribe meeting.mp4
# → meeting-transcript.md
```

- **Полностью локально** — данные не покидают машину
- **Авто-GPU** — автоматически использует NVIDIA CUDA, если доступен
- **Батч-режим** — обработка нескольких файлов за один вызов
- **Markdown с таймкодами** — удобен для суммаризации ИИ
- **Аудио и видео** — mp3, wav, mp4, mkv и [другие форматы](#поддерживаемые-форматы)

## Установка

**1. Установить [uv](https://docs.astral.sh/uv/getting-started/installation/)** (если ещё нет):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # Linux / macOS
```
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows
```

**2. Установить transcriber:**

```bash
uv tool install git+https://github.com/dementev-dev/local-transcriber
```

**3. Готово:**

```bash
transcribe meeting.mp4
```
Модели скачиваются автоматически при первом запуске (~1.5 GB для medium), нужен доступ в интернет.

<details>
<summary><code>transcribe: command not found</code></summary>

Выполните `uv tool update-shell` — это добавит нужный путь в PATH автоматически.

</details>

**Обновление:**

```bash
uv tool install --force git+https://github.com/dementev-dev/local-transcriber
```

## Использование

```bash
# Простой запуск (medium, русский, автодетект устройства)
transcribe meeting.mp4

# Указать язык
transcribe lecture.mp3 --language en

# Максимальное качество на GPU
transcribe podcast.wav --model large-v3 --compute-type float16

# Сохранить в конкретный файл
transcribe interview.m4a --output result.md
```

### Батч-режим

Обработка нескольких файлов за один вызов — модель загружается один раз:

```bash
# Все mp4 в директории
transcribe ./recordings/*.mp4

# Несколько файлов
transcribe meeting1.mp3 meeting2.mp3

# Перезаписать существующие транскрипты
transcribe *.mp4 --force
```

- Файлы с существующим транскриптом (`*-transcript.md`) автоматически пропускаются
- `--force` / `-f` — перезаписать существующие транскрипты
- При ошибке в одном файле остальные продолжают обрабатываться
- `--output` несовместим с несколькими файлами

### Опции CLI

| Опция | Сокращение | По умолчанию | Описание |
|-------|-----------|-------------|----------|
| `--model` | `-m` | `medium` | Модель Whisper |
| `--language` | `-l` | `ru` | Язык (ru, en, auto и др.) |
| `--output` | `-o` | `<файл>-transcript.md` | Путь к выходному файлу |
| `--device` | `-d` | `auto` | Устройство (auto, cpu, cuda) |
| `--compute-type` | — | float16 (GPU) / float32 (CPU) | Тип вычислений |
| `--force` | `-f` | — | Перезаписать существующие транскрипты |
| `--verbose` | `-v` | — | Подробный вывод |

## Платформы

|  | Linux / WSL2 | macOS | Windows |
|---|---|---|---|
| CPU | ✅ | ✅ | ✅ |
| GPU (NVIDIA) | ✅ авто | — | ✅ (нужен CUDA toolkit) |

<details>
<summary>Linux / WSL2</summary>

- GPU работает из коробки — cuBLAS ставится автоматически как зависимость
- Нужен только драйвер NVIDIA (проверка: `nvidia-smi`)
- На ARM (aarch64) cuBLAS через pip недоступен — нужен системный CUDA toolkit

</details>

<details>
<summary>macOS</summary>

- Работает на CPU (Intel и Apple Silicon)
- GPU (CUDA) недоступен — NVIDIA не поддерживает macOS

</details>

<details>
<summary>Windows</summary>

- CPU работает из коробки
- Для GPU нужен CUDA toolkit:
  ```
  winget install -e --id Nvidia.CUDA
  ```
  > `winget install` требует запуска от имени администратора (elevated terminal).
  > `uv tool install` работает без админа (ставит в пользовательскую директорию).
- После установки CUDA перезапустите терминал

</details>

## Конфигурация

Дефолтные параметры можно задать в `.transcriber.toml`:

```toml
model = "large-v3"
language = "en"
```

Порядок поиска:
1. `.transcriber.toml` в текущей директории
2. `~/.config/transcriber/config.toml`

Приоритет: **CLI-аргумент > конфиг > device-aware дефолт > встроенный дефолт**.

Дефолты зависят от устройства:

| Параметр | GPU (CUDA) | CPU |
|----------|-----------|-----|
| model | medium | medium |
| compute_type | float16 | float32 |
| language | ru | ru |

## Модели и GPU

Рекомендации:
- **По умолчанию:** `medium` — хороший баланс скорости и качества
- **Макс. качество:** `large-v3` + `--compute-type float16` (GPU)
- **Быстрый тест:** `tiny` — для проверки пайплайна

<details>
<summary>Таблица моделей</summary>

| Модель | Размер на диске | VRAM (int8) | Скорость (GPU) | Качество |
|--------|----------------|-------------|----------------|----------|
| `tiny` | ~75 MB | ~1 GB | ★★★★★ | ★ |
| `base` | ~140 MB | ~1 GB | ★★★★ | ★★ |
| `small` | ~460 MB | ~1.5 GB | ★★★ | ★★★ |
| `medium` | ~1.5 GB | ~2.5 GB | ★★ | ★★★★ |
| `large-v3` | ~3 GB | ~2.5 GB | ★ | ★★★★★ |

</details>

<details>
<summary>Типы квантизации (--compute-type)</summary>

| Тип | Устройство | VRAM/RAM | Качество | Когда использовать |
|-----|-----------|----------|----------|--------------------|
| `float16` | GPU | ~4.5-5 GB | Отлично | **По умолчанию для GPU** |
| `int8_float16` | GPU | ~4.7 GB | Отлично | GPU от 6 GB, альтернатива float16 |
| `int8` | GPU/CPU | Низкое | Хорошо, но бывают галлюцинации | GPU от 4 GB, CPU |
| `float32` | CPU | Среднее | Отлично | **По умолчанию для CPU** |

**Важно:** `int8` на длинных записях может давать галлюцинации (повтор фраз, потеря контента).
`float16` и `float32` значительно стабильнее на записях >20 минут.

</details>

Подробнее: бенчмарки, совместимость GPU, результаты тестирования — [docs/gpu.md](docs/gpu.md).

<details>
<summary>Формат вывода</summary>

```markdown
# Транскрипт: meeting.mp4

- **Дата транскрипции**: 2026-03-17 14:30:05
- **Модель**: large-v3
- **Язык**: ru (detected)
- **Длительность**: 01:23:45
- **Устройство**: CUDA (NVIDIA GeForce RTX 3060)

---

[00:00:00.00 - 00:00:15.40] Добрый день, коллеги. Сегодня мы обсудим результаты
квартала. Первый вопрос — по метрикам продукта.

[00:00:18.10 - 00:00:25.73] Теперь перейдём к финансовым показателям.
```

Близкие по времени сегменты автоматически объединяются в абзацы (пауза > 2 сек или длительность > 60 сек разделяет абзацы).
Таймкоды: `MM:SS.ss`, для записей длиннее 1 часа — `HH:MM:SS.ss`.

</details>

## Поддерживаемые форматы

- **Аудио**: mp3, wav, flac, ogg, m4a, wma, aac
- **Видео**: mp4, mkv, avi, mov, webm, ts

## Для разработчиков

```bash
git clone https://github.com/dementev-dev/local-transcriber
cd local-transcriber
uv sync
uv run transcribe meeting.mp4   # запуск CLI
uv run pytest                    # тесты
```

Подробнее — [CONTRIBUTING.md](CONTRIBUTING.md).
