# local-transcriber

Локальная транскрипция аудио и видео в markdown — без облака, без API-ключей.

```bash
transcribe meeting.mp4
# → meeting-transcript.md
```

- **Полностью локально** — данные не покидают машину
- **ONNX на CPU по умолчанию** — CUDA и OpenVINO включаются явно
- **Батч-режим** — обработка нескольких файлов за один вызов
- **Разделение говорящих** — локальная диаризация по флагу `--diarize`
- **Из проводника Windows** — пункт Transcribe в меню «Отправить» ([установка](#контекстное-меню-проводника-windows))
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

После установки откройте новую консоль, чтобы перечитались переменные окружения.

**2. Установить transcriber.** Напрямую из git, без скачивания исходников:

```bash
uv tool install --python 3.13 git+https://github.com/dementev-dev/local-transcriber
```

Или из клона репозитория (`<URL>` — адрес репозитория, из которого вы ставите):

```bash
git clone <URL>
cd local-transcriber
uv tool install --python 3.13 .
```

**3. Пункт `Transcribe` в меню «Отправить»** (Windows, по желанию):

```bash
transcribe --install-menu
```

Подробнее — в разделе [контекстное меню проводника](#контекстное-меню-проводника-windows).

**4. Выбор исполнения:**

- По умолчанию используется ONNX с GigaAM RNN-T на CPU, независимо от наличия
  NVIDIA. Эта модель понимает только русскую речь.
- Для других языков выбирайте Whisper явно:
  `--device openvino-cpu --model medium` на x86 или `--device cpu --model medium`.
- OpenVINO для Intel GPU или x86 CPU доступен через `--device openvino`,
  `--device openvino-gpu` или `--device openvino-cpu`.
- Для NVIDIA установите дополнительные библиотеки и выберите CUDA явно:
  [подключение CUDA](#подключение-cuda).

**5. Готово:**

```bash
transcribe meeting.mp4
```
Модели скачиваются автоматически при первом запуске; размер зависит от выбранного
профиля, нужен доступ в интернет.

<details>
<summary><code>transcribe: command not found</code></summary>

Выполните `uv tool update-shell` — это добавит нужный путь в PATH автоматически.

</details>

**Обновление:**

```bash
uv tool install --python 3.13 --force git+https://github.com/dementev-dev/local-transcriber
```

Или в папке со склонированным репозиторием:

```bash
git pull
uv tool install --python 3.13 --force .
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

## Подключение CUDA

Обычная установка не требует NVIDIA-пакетов. Extra `cuda` добавляет cuBLAS
в окружение приложения на Linux/WSL x86_64 и Windows x64. Полный системный
CUDA Toolkit и ручная правка системного PATH для этого пути не нужны.
Совместимые GPU и драйвер NVIDIA необходимы отдельно: extra не устанавливает
драйвер и не исправляет несовместимость старой видеокарты.

Для установки из Git или подключения CUDA к уже установленной программе:

```bash
uv tool install --python 3.13 --force "local-transcriber[cuda] @ git+https://github.com/dementev-dev/local-transcriber"
```

Из клона репозитория:

```bash
uv tool install --python 3.13 --force ".[cuda]"
```

Для локальной разработки:

```bash
uv sync --extra cuda
uv run --extra cuda transcribe meeting.mp4 --device cuda
```

После установки extra выберите CUDA при запуске:

```bash
transcribe meeting.mp4 --device cuda
```

Либо задайте в `.transcriber.toml`:

```toml
device = "cuda"
```

Установка extra сама по себе не меняет `auto`: он всегда использует ONNX CPU.
Явная CUDA при ошибке завершает обработку с исходной причиной; для распознанных
причин выводится подсказка. Молчаливого перехода на CPU нет. В батче ошибка
учитывается для каждого файла.

На Windows нужен также [Visual C++ Runtime x64](https://aka.ms/vs/17/release/vc_redist.x64.exe).
Состав DLL проверен для закреплённых версий CTranslate2 и cuBLAS;
реальный прогон нового пути на Windows без Toolkit пока не выполнен.
Подробности и ограничения проверки — [ADR-001](docs/adr/001-cuda-bootstrap.md).

При обновлении из старой установки:

- Для CPU повторите обычную команду `uv tool install` с `--force` без `[cuda]`.
  В окружении разработки выполните `uv sync` — прежняя обязательная cuBLAS будет удалена.
- Для CUDA повторите соответствующую команду выше. Старый автоматический выбор
  по `nvidia-smi` больше не действует; сохраните `device = "cuda"` в конфиге
  либо передавайте `--device cuda`.

## Использование

```bash
# Простой запуск (ONNX GigaAM RNN-T на CPU, язык ru)
transcribe meeting.mp4

# Указать язык
transcribe lecture.mp3 --device cpu --model medium --language en

# Максимальное качество на NVIDIA GPU
transcribe podcast.wav --device cuda --model large-v3 --compute-type float16

# Максимальное качество на Intel GPU
transcribe podcast.wav --model large-v3 --device openvino-gpu

# Максимальная скорость на CPU (русский)
transcribe meeting.mp4 --device onnx --model gigaam-v3

# CPU с пунктуацией и нормализацией русского текста
transcribe podcast.wav --device onnx --model gigaam-v3-e2e-ctc

# Смешанная русско-английская речь
transcribe meeting.wav --device onnx --model gigaam-multilingual-ctc

# Повышенная точность смешанной речи (медленнее, ~590 MB)
transcribe meeting.wav --device onnx --model gigaam-multilingual-large-ctc

# Сохранить в конкретный файл
transcribe interview.m4a --output result.md

# Разделить встречу на реплики говорящих
transcribe meeting.mp4 --diarize

# Если число участников известно заранее
transcribe interview.m4a --speakers 2
```

### Разделение говорящих

`--diarize` добавляет к транскрипту реплики `Speaker 1`, `Speaker 2` и так
далее. `--speakers N` задаёт ожидаемое число участников и автоматически включает
диаризацию; без него число кластеров определяется автоматически.

При первом таком запуске дополнительно скачиваются две ONNX-модели Sherpa-ONNX:
сегментация (~6 МБ) и голосовые эмбеддинги (~27 МБ). Они сохраняются в кеше
Hugging Face и используются повторно. Диаризация выполняется после распознавания
речи и добавляет отдельный проход по записи. На измеренном слабом Intel Core
i7-6820HQ последовательные ASR и диаризация увеличивали полное время примерно в
2,4 раза, но оставались быстрее реального времени; фактическая скорость зависит
от процессора и режима питания ([замеры](docs/benchmarks/2026-08-14-diarization-intel-i7.md)).

Если найдено меньше двух говорящих или диаризация конкретного файла завершилась
ошибкой, текст не теряется: сохраняется обычный транскрипт, в Markdown
записывается причина, а команда завершается с кодом `1`. Если выбранный ASR-путь
не поддерживает пословные таймкоды или диаризатор не удалось инициализировать,
запуск останавливается до первого ASR и не создаёт частичных транскриптов. Малый
кластер только отмечается предупреждением и не удаляется. Слова без однозначного
говорящего попадают в реплику `Speaker ?`.

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
- При ошибке диаризации сохраняется обычный транскрипт, остальные файлы
  продолжают обрабатываться; итоговый код батча — `1`
- `--output` несовместим с несколькими файлами

### Контекстное меню проводника (Windows)

Установить пункт `Transcribe` в меню «Отправить»:

```bash
transcribe --install-menu
```

(при запуске из клона репозитория — `uv run transcribe --install-menu`)

Использование: выделите один или несколько аудио/видеофайлов в проводнике, откройте контекстное меню правой кнопкой. В Windows 11 выберите «Показать дополнительные параметры» или нажмите Shift+F10, затем «Отправить» → «Transcribe». Несколько выделенных файлов передаются в один процесс и обрабатываются одним батчем.

Удалить пункт меню:

```bash
transcribe --uninstall-menu
```

Если что-то пошло не так, пункт можно удалить вручную: Win+R → `shell:sendto` → удалить `Transcribe.cmd`.

Известные ограничения:

- После переноса или пересоздания проекта/venv выполните `--install-menu` заново: внутри `Transcribe.cmd` хранится абсолютный путь к `transcribe.exe`.
- Очень большой мультивыбор с суммарной длиной путей ≳8000 символов упирается в лимит командной строки cmd.exe. Обрабатывайте такие файлы частями.

### Опции CLI

| Опция | Сокращение | По умолчанию | Описание |
|-------|-----------|-------------|----------|
| `--model` | `-m` | medium (CUDA) / gigaam-v3-e2e-rnnt (ONNX) | Модель распознавания |
| `--language` | `-l` | `ru` | Язык (ru, en, auto и др.); автоматический профиль понимает только русский |
| `--output` | `-o` | `<файл>-transcript.md` | Путь к выходному файлу |
| `--device` | `-d` | `auto` | Устройство (auto, cpu, cuda, openvino, openvino-gpu, openvino-cpu, onnx) |
| `--compute-type` | — | float16 (CUDA) / int8 (ONNX/OpenVINO) / float32 (CPU) | Тип вычислений |
| `--threads` | `-t` | 0 (авто) | Потоки CPU (рекомендуется = число физ. ядер); для ONNX задаёт потоки и ASR, и VAD |
| `--diarize` / `--no-diarize` | — | `false` | Включить или отключить разделение на реплики говорящих |
| `--speakers` | — | авто | Ожидаемое число говорящих; включает диаризацию и несовместим с `--no-diarize` |
| `--force` | `-f` | — | Перезаписать существующие транскрипты |
| `--verbose` | `-v` | — | Подробный вывод: сегменты по ходу, движок, потоки, версии runtime |

### Диагностика исполнения

Перед распознаванием программа печатает модель, устройство и тип вычислений,
которые выбраны фактически: после `auto` это `onnx`, после `openvino` —
`openvino-gpu` или `openvino-cpu`. Если после загрузки устройство оказалось
не тем, что выбрано до неё (например, OpenVINO взял CPU вместо GPU),
выводится строка `Запрошено ..., используется ...`.

С `--verbose` добавляются движок, бюджет потоков и версии runtime: для ONNX —
`onnxruntime`, `onnx-asr`, доступные providers и те, что заданы сессиям ASR
и VAD; для CUDA — `ctranslate2` и число видимых CUDA-устройств; для OpenVINO —
версия и доступные устройства. Этих строк достаточно, чтобы разобрать проблему
на чужой машине без доступа к записи: попросите прислать вывод с `--verbose`.

Наличие provider в списке доступных не означает, что распознавание идёт
на GPU или NPU: сессии получают только явно заданный список, сейчас это CPU.

## Платформы

|  | Linux / WSL2 | macOS | Windows |
|---|---|---|---|
| CPU через ONNX | ✅ авто | ✅ авто | ✅ авто |
| OpenVINO (x86 CPU) | ✅ явно | — | ✅ явно |
| OpenVINO (Intel GPU) | ✅ явно | — | ✅ явно |
| GPU (NVIDIA) | extra + явно (x86_64) | — | extra + явно (x64; прогон без Toolkit ожидается) |

Данные по macOS основаны на доступности пакетов onnxruntime: прогонов на этой
платформе не было.

<details>
<summary>Linux / WSL2</summary>

- **Intel GPU** (Arc, встроенная графика) работает из коробки через OpenVINO
- **NVIDIA GPU**: [extra `cuda` и явный выбор](#подключение-cuda), отдельно совместимый драйвер
- `nvidia-smi` проверяет наличие драйвера, но не доказывает совместимость GPU с runtime
- CUDA extra на ARM (aarch64) этим проектом не поддерживается

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
- **NVIDIA GPU**: [extra `cuda` и явный выбор](#подключение-cuda), отдельно совместимый драйвер
- Требуется Visual C++ Runtime x64; системный CUDA Toolkit не нужен для extra
- Реальная транскрипция с extra без Toolkit пока не проверена на Windows

</details>

## Конфигурация

Дефолтные параметры можно задать в `.transcriber.toml`:

```toml
device = "openvino-cpu"
model = "large-v3-turbo"
compute_type = "int8"
language = "ru"
diarize = true
```

Порядок поиска:
1. `.transcriber.toml` в текущей директории
2. `~/.config/transcriber/config.toml`

Приоритет: **CLI-аргумент > конфиг > device-aware дефолт > встроенный дефолт**.
`--diarize` и `--no-diarize` позволяют переопределить `diarize` из конфига для
отдельного запуска.

При `device = "auto"` всегда выбирается ONNX CPU, даже если доступны
`nvidia-smi` и extra `cuda`. Другие варианты выбираются явно из CLI или конфига.
Каталоги моделей различаются между бэкендами, поэтому при закреплении `model`
в конфиге рекомендуется явно закрепить и совместимый `device`. То же с языком:
автоматический ONNX-профиль рассчитан на русскую речь, а для остальных языков
нужен Whisper — например, `device = "openvino-cpu"` и `model = "medium"`.

Дефолты зависят от устройства:

| Параметр | CUDA | OpenVINO (GPU) | OpenVINO (CPU) | ONNX | CPU |
|----------|------|----------------|----------------|------|-----|
| model | medium | medium | medium | gigaam-v3-e2e-rnnt | medium |
| compute_type | float16 | int8 | int8 | int8 | float32 |
| language | ru | ru | ru | ru | ru |

## Модели и GPU

Рекомендации:
- **По умолчанию:** ONNX `gigaam-v3-e2e-rnnt` — читаемый русский текст с
  пунктуацией почти без потери скорости относительно сырого `gigaam-v3`
- **Макс. качество (NVIDIA):** `--device cuda --model large-v3 --compute-type float16`
- **Макс. качество (Intel GPU):** `large-v3` + `--device openvino-gpu`
- **Макс. скорость CPU (русский):** `--device onnx --model gigaam-v3` (17-29× RTF, без пунктуации; рекомендуется LLM-нормализация терминов после)
- **Быстрый OpenVINO с низким WER:** `--device openvino-cpu --model large-v3-turbo --compute-type int8` (7,2× RTFx на контрольном Intel CPU; пунктуация может быть слабой)
- **OpenVINO для чтения и конспекта:** `--device openvino-cpu --model medium` (около 6× RTFx; независимая оценка показала лучшую сохранность содержания, чем turbo)
- **Быстрый тест:** `tiny` — для проверки пайплайна

<details>
<summary>Таблица моделей</summary>

#### Whisper через faster-whisper (`--device cuda`, `--device cpu`)

| Модель | Размер на диске | VRAM (int8) | Скорость (GPU) | Качество |
|--------|----------------|-------------|----------------|----------|
| `tiny` | ~75 MB | ~1 GB | ★★★★★ | ★ |
| `base` | ~140 MB | ~1 GB | ★★★★ | ★★ |
| `small` | ~460 MB | ~1.5 GB | ★★★ | ★★★ |
| `medium` | ~1.5 GB | ~2.5 GB | ★★ | ★★★★ |
| `large-v3` | ~3 GB | ~2.5 GB | ★ | ★★★★★ |

#### Whisper через OpenVINO (`--device openvino-cpu`, `--device openvino-gpu`)

Здесь те же модели Whisper, но предквантизированные, поэтому на диске они
занимают меньше места: `medium` int8 — 748 MB, `large-v3-turbo` int8 — 790 MB,
`large-v3-turbo` fp16 — 1552 MB. Модель `large-v3-turbo` доступна только здесь:
faster-whisper её не поддерживает, и запуск с `--device cuda` завершится
ошибкой. Полный список репозиториев —
[docs/gpu.md](docs/gpu.md#доступные-openvino-модели).

#### ONNX-модели (`--device onnx`)

Другие архитектуры, не Whisper. Работают через onnxruntime на CPU:

| Модель | Размер (int8) | RTFx CPU | Языки | Пунктуация |
|--------|--------------|----------|-------|-----------|
| `gigaam-v3` | ~300 MB | 17-29× | ru | ❌ |
| `gigaam-multilingual-ctc` | ~300 MB | 10,0×* | ru, en, kk, ky, uz | ❌ |
| `gigaam-multilingual-large-ctc` | ~590 MB | 4,8×* | ru, en, kk, ky, uz | ❌ |
| `gigaam-v3-e2e-ctc` | ~300 MB | 11,9×* | ru | ✅ |
| `gigaam-v3-e2e-rnnt` | ~300 MB | 11,5×* | ru | ✅ |
| `parakeet-v3` | ~600 MB | 7,6×* | 25 языков | ✅ |

\* Наблюдение на AMD Ryzen 7 8845H, Windows, `int8`, три записи общей
длительностью 43:37. Замер описывает только эту тестовую машину; скорость на
других CPU требует отдельного прогона.
Методика и качественное сравнение:
[benchmark GigaAM и Whisper](docs/benchmarks/2026-08-11-gigaam-model-comparison.md).

> **Рекомендация**: ONNX по умолчанию использует `gigaam-v3-e2e-rnnt` для
> готового читаемого русского текста. Для последующей машинной обработки можно
> явно выбрать более точный по словам `gigaam-v3` без пунктуации. Для смешанной
> речи с приоритетом качества используйте `gigaam-multilingual-large-ctc`: она примерно вдвое
> медленнее small-варианта, но приблизилась к monolingual GigaAM по WER.
> `parakeet-v3` в 1,58 раза быстрее Large и ставит пунктуацию, но на тех же
> трёх записях хуже по WER и вставляет ложные английские фразы в русскую речь;
> это подтверждает проблемы из [ADR-005](docs/adr/005-parakeet-evaluation.md).

Обе GigaAM Multilingual сами распознают русский, английский, казахский,
кыргызский и узбекский внутри одной записи. `onnx-asr` не передаёт этим моделям
подсказку языка, поэтому `--language` не управляет выбором языка.

Если модель не понимает запрошенный язык, CLI предупреждает об этом до начала
распознавания и подсказывает совместимый профиль, но работу не прерывает.

Для моделей из таблицы опубликованы `int8` и `float32`. Если неявный
device-aware дефолт недоступен для выбранной модели, CLI сообщит о подстановке
доступного варианта. Явное значение из `--compute-type` или
`.transcriber.toml` вместо подстановки завершится ошибкой.

</details>

<details>
<summary>Типы квантизации (--compute-type)</summary>

| Тип | Бэкенд | VRAM/RAM | Качество | Когда использовать |
|-----|--------|----------|----------|--------------------|
| `float16` | CUDA | ~4.5-5 GB | Отлично | **По умолчанию для CUDA** |
| `int8_float16` | CUDA | ~4.7 GB | Отлично | GPU от 6 GB, альтернатива float16 |
| `int8_float32` | CPU | Среднее | Отлично | **Рекомендуется для CPU** — 1.5x быстрее float32 при том же качестве |
| `int8` | CUDA / OpenVINO / ONNX | Низкое | Хорошо, но бывают галлюцинации | **По умолчанию для OpenVINO и ONNX** |
| `fp16` | OpenVINO | Низкое | Отлично | OpenVINO large-v3 (выбирается автоматически) |
| `float32` | CPU / ONNX | Среднее | Отлично | **По умолчанию для CPU** |

**Важно:** `int8` на длинных записях может давать галлюцинации (повтор фраз, потеря контента).
`float16`/`fp16` и `float32` значительно стабильнее на записях >20 минут.

> Для OpenVINO `--compute-type` выбирает предквантизированную модель (int8 или fp16),
> а не параметр времени выполнения. Для `large-v3` по умолчанию выбирается
> `fp16`; для `large-v3-turbo` доступны явные варианты `int8` и `fp16`, а
> неявный профиль OpenVINO использует `int8`.

</details>

Подробнее: бенчмарки, OpenVINO, совместимость GPU, результаты тестирования —
[docs/gpu.md](docs/gpu.md). Сравнение `large-v3-turbo` с CPU-профилями:
[OpenVINO 2026.3 и large-v3-turbo](docs/benchmarks/2026-08-12-openvino-large-v3-turbo-comparison.md).

<details>
<summary>Формат вывода</summary>

```markdown
# Транскрипт: meeting.mp4

- **Дата транскрипции**: 2026-03-17 14:30:05
- **Модель**: gigaam-v3-e2e-rnnt
- **Язык**: ru (задан явно)
- **Длительность**: 01:23:45
- **Устройство**: ONNX (CPU)

---

[00:00:00.00 - 00:00:15.40] Добрый день, коллеги. Сегодня мы обсудим результаты
квартала. Первый вопрос — по метрикам продукта.

[00:00:18.10 - 00:00:25.73] Теперь перейдём к финансовым показателям.
```

Близкие по времени сегменты автоматически объединяются в абзацы (пауза > 2 сек или длительность > 60 сек разделяет абзацы).
Таймкоды: `MM:SS.ss`, для записей длиннее 1 часа — `HH:MM:SS.ss`.

В скобках после языка указан его источник:

- `задан явно` — язык взят из `--language` или конфига;
- `определён автоматически` — распознан моделью при `--language auto`;
- `из профиля модели` — у модели всего один язык, как у GigaAM.

Если язык определить не удалось, строка выглядит так: `- **Язык**: не определён`.

С `--diarize` при успешном обнаружении нескольких говорящих основная часть
выглядит так:

```markdown
[00:00] Speaker 1: Добрый день, коллеги.

[00:04] Speaker 2: Начнём с результатов квартала.
```

Таймкод реплики показывает начало: `MM:SS`, а после часа — `HH:MM:SS`.

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
