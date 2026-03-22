# local-transcriber

Локальная транскрипция аудио и видео в markdown — без облака, без API-ключей.

```bash
transcribe meeting.mp4
# → meeting-transcript.md
```

- **Полностью локально** — данные не покидают машину
- **Авто-ускорение** — NVIDIA CUDA, Intel GPU (OpenVINO), OpenVINO CPU или CPU fallback
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

**3. Ускорение (ставится автоматически):**

- **Intel GPU** (Arc, встроенная графика): работает через OpenVINO — ускорение в ~2x vs CPU
- **OpenVINO** (Intel/AMD x86 CPU): ставится автоматически на Linux и Windows — ускорение в 2-4x
- **NVIDIA CUDA** (GPU): если есть GPU — транскрипция в 5-10× быстрее
  - **Windows**: `winget install -e --id Nvidia.CUDA --version 12.9` (от администратора), перезапустить терминал
  - **Linux / WSL2**: работает из коробки (нужен только драйвер: `nvidia-smi`)

**4. Готово:**

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

**Удаление:**

```bash
uv tool uninstall local-transcriber
```

**Очистка моделей:**

Модели кешируются в `~/.cache/huggingface/hub/` и могут занимать несколько гигабайт.
На Windows без Developer Mode файлы копируются без симлинков — место удваивается.

```bash
# Linux / macOS — посмотреть размер кеша
du -sh ~/.cache/huggingface/hub/models--*

# Удалить все скачанные модели
rm -rf ~/.cache/huggingface/hub/models--Systran--faster-whisper-*
rm -rf ~/.cache/huggingface/hub/models--OpenVINO--whisper-*
```

```powershell
# Windows
dir "$env:USERPROFILE\.cache\huggingface\hub\models--*"

# Удалить все скачанные модели
Remove-Item -Recurse "$env:USERPROFILE\.cache\huggingface\hub\models--Systran--faster-whisper-*"
Remove-Item -Recurse "$env:USERPROFILE\.cache\huggingface\hub\models--OpenVINO--whisper-*"
```

При следующем запуске нужная модель скачается заново.

<details>
<summary>Windows: ошибка WinError 1314 при первом запуске</summary>

HuggingFace Hub использует симлинки для экономии места. На Windows без Developer Mode первая загрузка модели может упасть с ошибкой `WinError 1314`. Повторный запуск команды обычно помогает — HF Hub переключается на копирование файлов.

Чтобы избежать проблемы и сэкономить место, включите Developer Mode:
[Инструкция Microsoft](https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development)

</details>

## Использование

```bash
# Простой запуск (medium, русский, автодетект устройства)
transcribe meeting.mp4

# Указать язык
transcribe lecture.mp3 --language en

# Максимальное качество на NVIDIA GPU
transcribe podcast.wav --model large-v3 --compute-type float16

# Максимальное качество на Intel GPU
transcribe podcast.wav --model large-v3 --device openvino-gpu

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
| `--device` | `-d` | `auto` | Устройство (auto, cpu, cuda, openvino, openvino-gpu, openvino-cpu) |
| `--compute-type` | — | float16 (CUDA) / int8 (OpenVINO GPU/CPU) / float32 (CPU) | Тип вычислений |
| `--force` | `-f` | — | Перезаписать существующие транскрипты |
| `--verbose` | `-v` | — | Подробный вывод |

## Платформы

|  | Linux / WSL2 | macOS | Windows |
|---|---|---|---|
| CPU | ✅ | ✅ | ✅ |
| OpenVINO (x86 CPU) | ✅ авто | — | ✅ авто |
| OpenVINO (Intel GPU) | ✅ авто | — | ✅ авто |
| GPU (NVIDIA) | ✅ авто | — | ✅ (нужен CUDA 12) |

<details>
<summary>Linux / WSL2</summary>

- **Intel GPU** (Arc, встроенная графика) работает из коробки через OpenVINO
- **NVIDIA GPU** работает из коробки — cuBLAS ставится автоматически как зависимость
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
- **Intel GPU** (Arc, встроенная графика) работает из коробки через OpenVINO
- Для **NVIDIA GPU** нужен **CUDA 12** (ctranslate2 4.7 не совместим с CUDA 11 и 13):
  ```
  winget install -e --id Nvidia.CUDA --version 12.9
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

| Параметр | CUDA | OpenVINO (GPU) | OpenVINO (CPU) | CPU |
|----------|------|----------------|----------------|-----|
| model | medium | medium | medium | medium |
| compute_type | float16 | int8 | int8 | float32 |
| language | ru | ru | ru | ru |

## Модели и GPU

Рекомендации:
- **По умолчанию:** `medium` — хороший баланс скорости и качества
- **Макс. качество (NVIDIA):** `large-v3` + `--compute-type float16`
- **Макс. качество (Intel GPU):** `large-v3` + `--device openvino-gpu`
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

| Тип | Бэкенд | VRAM/RAM | Качество | Когда использовать |
|-----|--------|----------|----------|--------------------|
| `float16` | CUDA | ~4.5-5 GB | Отлично | **По умолчанию для CUDA** |
| `int8_float16` | CUDA | ~4.7 GB | Отлично | GPU от 6 GB, альтернатива float16 |
| `int8` | CUDA / OpenVINO | Низкое | Хорошо, но бывают галлюцинации | **По умолчанию для OpenVINO** |
| `fp16` | OpenVINO | Низкое | Отлично | OpenVINO large-v3 (выбирается автоматически) |
| `float32` | CPU | Среднее | Отлично | **По умолчанию для CPU** |

**Важно:** `int8` на длинных записях может давать галлюцинации (повтор фраз, потеря контента).
`float16`/`fp16` и `float32` значительно стабильнее на записях >20 минут.

> Для OpenVINO `--compute-type` выбирает предквантизированную модель (int8 или fp16),
> а не runtime-параметр. Для `large-v3` по умолчанию выбирается `fp16`.

</details>

Подробнее: бенчмарки, OpenVINO, совместимость GPU, результаты тестирования — [docs/gpu.md](docs/gpu.md).

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
