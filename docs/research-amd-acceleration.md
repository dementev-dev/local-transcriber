# Исследование: ускорение на AMD Radeon / NPU

**Дата**: 2026-03-23
**Железо**: AMD Ryzen 7 8845H, Radeon 780M iGPU (RDNA3, 12 CU), XDNA 1 NPU (16 TOPS)

## Мотивация

Бенчмарк ([benchmark-2026-03-23.md](benchmark-2026-03-23.md)) показал: лучшее качество
даёт CTranslate2 float32/int8_float32, но это 600–880с на 15-минутную запись.
OpenVINO fp16 быстрее (240с), но теряет качество на русской речи.

Нужен путь: качество float32 + скорость GPU.

## Вариант 1: DirectML (onnxruntime-directml)

### Вердикт: ❌ НЕ рекомендуется

**Блокирующая проблема**: оператор `DecoderMaskedMultiHeadAttention` не реализован
в DirectML ([Olive#1213](https://github.com/microsoft/Olive/issues/1213), открыт с июня 2024).
Whisper-декодер fallback-ится на CPU — GPU ускоряет только encoder, выигрыш минимален.

**DirectML в maintenance mode** — Microsoft переключился на Windows ML (WinML).
Команда DML расформирована, новых операторов не будет. Фикс маловероятен.

**WinML** (замена DirectML) — GA с сентября 2025, но Python-пакет `onnxruntime-winml`
ещё не доступен. Требует Windows 11 25H2+. Потенциально интересен в будущем.

## Вариант 2: whisper.cpp + Vulkan ✅

### Вердикт: рекомендуется как основной путь

**Почему:**
- Проверен на AMD Radeon 780M — баг softmax ([#2596](https://github.com/ggml-org/whisper.cpp/issues/2596))
  исправлен в декабре 2024
- **3–4x ускорение** vs CPU на Radeon 680M (RDNA2), 780M (RDNA3) должен быть не хуже
- Весь pipeline внутри: аудио → mel → encoder → decoder → beam search
- GGUF-модели компактнее ONNX
- Активная разработка, Vulkan-бэкенд зрелый

**Бенчмарки (чужие, похожее железо):**

| GPU | Модель | Результат |
|-----|--------|-----------|
| Radeon 680M (RDNA2) | medium | ~3–4x быстрее CPU |
| Ryzen 5 4500U (Vega) | medium | 1ч аудио за ~25 мин |
| Ryzen AI Max+ 395 (Strix Halo) | large-v3-turbo | 1ч20м аудио за ~3 мин |

**Память:** Radeon 780M использует системную RAM. whisper-medium в GGUF ≈ 1.5 ГБ —
помещается с запасом.

### Интеграция в проект

Новый бэкенд `backends/whisper_cpp.py`, реализующий Backend Protocol:

```
uv add pywhispercpp    # Python-биндинги к whisper.cpp
```

Либо Vulkan-сборка whisper.cpp как subprocess (более надёжно,
но менее удобно для пользователя).

**Зависимость в pyproject.toml:**
```toml
# Опциональная, т.к. требует Vulkan-совместимый GPU
"pywhispercpp>=0.4; sys_platform == 'win32'"
```

Либо через dependency group:
```toml
[dependency-groups]
vulkan = ["pywhispercpp>=0.4"]
```

Тогда установка: `uv sync --group vulkan`

**Открытые вопросы:**
- pywhispercpp собран с Vulkan или нужна своя сборка?
- Качество GGUF-квантизации vs CTranslate2 float32 — нужен бенчмарк
- Формат моделей другой (GGUF vs CTranslate2/OpenVINO) — ещё одно скачивание

## Вариант 3: AMD NPU (XDNA)

### Вердикт: ⏳ рано для XDNA 1, перспективно для XDNA 2

**Что есть:**
- Ryzen AI Software 1.7.0 (февраль 2026) поддерживает 8845H (Hawk Point)
- [AMD LIRA](https://github.com/amd/LIRA) — CLI для ASR на NPU, поддерживает whisper base/small/medium
- [AMD-форк whisper.cpp](https://github.com/amd/whisper.cpp) с NPU-бэкендом
- Pre-квантизированные модели на HuggingFace (`amd/NPU-Whisper-Base-Small`)

**Ограничения на XDNA 1 (8845H):**
- 16 TOPS — в 3 раза слабее XDNA 2 (50 TOPS)
- **Только encoder** на NPU, decoder на CPU — ускорение частичное
- whisper-large-v3 не помещается
- Установка через отдельный installer AMD, не pip/uv
- Форк whisper.cpp не в mainline — риск отставания

**Когда станет интересно:**
- На железе с XDNA 2 (Ryzen AI 300+, 50 TOPS)
- Когда AMD выпустит pip-пакет или upstreamит в whisper.cpp
- Для сценариев, где CPU/GPU заняты другой работой

## Сводка

| Подход | Применимость | Ускорение | Сложность | Стабильность |
|--------|-------------|-----------|-----------|--------------|
| **whisper.cpp + Vulkan** | ✅ высокая | 3–4x vs CPU | низкая | хорошая |
| DirectML (onnxruntime) | ❌ заблокирован | — | — | — |
| AMD NPU (XDNA 1) | ⚠️ ограничен | частичное | высокая | средняя |
| WinML (будущее) | ⏳ не готов | неизвестно | средняя | — |

## Детали по whisper.cpp + Vulkan

### Модели (GGML .bin формат, НЕ GGUF)

Скачивать с https://huggingface.co/ggerganov/whisper.cpp/tree/main

| Модель | Файл | Размер |
|--------|------|--------|
| medium f16 | `ggml-medium.bin` | 1.53 ГБ |
| medium q8_0 | `ggml-medium-q8_0.bin` | 823 МБ |
| medium q5_0 | `ggml-medium-q5_0.bin` | 539 МБ |
| large-v3 f16 | `ggml-large-v3.bin` | 3.1 ГБ |
| large-v3 q5_0 | `ggml-large-v3-q5_0.bin` | 1.08 ГБ |
| large-v3-turbo f16 | `ggml-large-v3-turbo.bin` | 1.62 ГБ |
| large-v3-turbo q8_0 | `ggml-large-v3-turbo-q8_0.bin` | 874 МБ |

### Python-биндинги

**pywhispercpp** (v1.4.1) — pip wheel CPU-only. Для Vulkan нужна сборка из исходников:
```bash
GGML_VULKAN=1 pip install git+https://github.com/absadiki/pywhispercpp
```
Требует: VulkanSDK, cmake, C++ компилятор.

### Subprocess-подход (альтернатива)

whisper-cli принимает WAV 16kHz mono, выдаёт JSON с таймкодами:
```bash
whisper-cli -m ggml-medium.bin -f audio.wav -l ru -oj -of output
```

Для видео/mp3 нужна предварительная конвертация через ffmpeg (PyAV уже есть в зависимостях).

### Сборка whisper.cpp с Vulkan

```bash
# Linux/WSL
sudo apt install cmake build-essential libvulkan-dev
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
# -> build/bin/whisper-cli
```

## Прототип: whisper.cpp в WSL2 (2026-03-23)

### Сборка

whisper.cpp собран с `-DGGML_VULKAN=ON`, но **Vulkan GPU не обнаружен**:

```
ggml_vulkan: No devices found.
whisper_backend_init_gpu: no GPU found
```

**Причина**: WSL2 не предоставляет Vulkan-доступ к AMD GPU.
- `/dev/dxg` есть (DirectX 12 paravirtualization), но **dozen** (Vulkan-over-D3D12) драйвер
  отсутствует в текущей mesa (25.2.8)
- RADV (нативный Vulkan) не работает — нет `/dev/dri/renderD128`
- Единственный Vulkan-девайс — `llvmpipe` (CPU-эмуляция), whisper.cpp его игнорирует

Whisper.cpp отработал **на CPU** (4 потока, AVX512).

### Скорость (CPU fallback)

| Параметр | Значение |
|----------|----------|
| Модель | ggml-medium.bin (f16, 1533 МБ) |
| Аудио | 14:41 (881с), русский, 3 участника |
| Общее время | **886с** |
| encode | 368с (32 прохода, 11.5с/проход) |
| batchd (decode) | 424с |
| Потоки | 4 / 16 |
| Beam search | 5 beams + best of 5 |

Сравнение с бенчмарком (тот же аудиофайл):

| Конфигурация | Время, с |
|---|---|
| OpenVINO fp16 (medium) | ~240 |
| CT2 int8_float32 (medium) | ~599 |
| CT2 float32 (medium) | ~881 |
| **whisper.cpp CPU (medium f16)** | **~886** |

На CPU whisper.cpp ≈ CTranslate2 float32. Без GPU ускорения нет.

### Качество (сравнение с бенчмарком)

#### Техтермины

| Термин | whisper.cpp (CPU) | CT2 f32 | CT2 int8f32 |
|--------|------------------|---------|-------------|
| DWH | ✅ DWH | ⚠️ ДВХ | ✅ DWH |
| Teradata | ✅ TeraData | ⚠️ Тирадату | ✅ Teradata |
| NiFi | ⚠️ Найфай/Найфаю | ⚠️ Найфай | ✅ NiFi |
| FineBI | ✅ fineBI/FNBI | ✅ FNBI | ✅ finebi/FNBI |
| Alation | ✅ Allation | ✅ Allation | ✅ Allation |
| ETL | ✅ ETL | ✅ ETL | ✅ ETL |
| DBC Tables V | ✅ DBC tables-V | ✅ | ✅ |
| OpenLineage | ✅ OpenLinage.io | ✅ OpenLinage.io | ✅ OpenLinage.io |

#### Русская разговорная речь

| Фраза | whisper.cpp (CPU) | CT2 f32 | CT2 int8f32 |
|-------|------------------|---------|-------------|
| "на другую мониторочку" | ✅ "на другом мониторчике" | ✅ "мониторчику" | ⚠️ "мою точку" |
| "верхнеуровневое" | ✅ | ✅ | ✅ |
| "датарентген" | ✅ "даторентген" | ✅ "даторентген" | ✅ "даторентген" |
| "девовские схемы" | ✅ | ✅ | ✅ |
| "с накиданными правами" | ✅ | ✅ | ✅ |
| "Вьюхи" | ⚠️ "Yuhi" | ✅ "Вьюхи" | ⚠️ "Yuhi" |

#### Структура и пунктуация

| Аспект | whisper.cpp (CPU) | CT2 f32 | CT2 int8f32 |
|--------|------------------|---------|-------------|
| Пунктуация | хорошая | хорошая | хорошая |
| Длина сегментов | 1–10 сек | ~1 мин | ~1 мин |

**Вывод по качеству**: whisper.cpp medium f16 на уровне CTranslate2 float32/int8_float32.
Качество сопоставимо, различия минимальны. Сегменты короче (1–10 сек vs ~1 мин) —
это даже лучше для навигации по транскрипту.

### Блокер: Vulkan в WSL2 для AMD

Для получения GPU-ускорения нужно одно из:

1. **Нативный Linux** (dual boot / bare metal) — RADV драйвер работает с Radeon 780M
2. **Windows-сборка** — нативный Vulkan через AMD Adrenalin драйвер
3. **WSL2 + dozen** — экспериментальный Vulkan-over-D3D12 драйвер mesa.
   Статус: [в разработке](https://gitlab.freedesktop.org/mesa/mesa/-/merge_requests?search=dozen),
   не включён в стандартную mesa Ubuntu.
   Потенциально можно собрать mesa из исходников с `-Dvulkan-drivers=microsoft-experimental`

**Рекомендация**: попробовать Windows-сборку whisper.cpp — это самый простой путь
к Vulkan + Radeon 780M. Либо собрать mesa с dozen для WSL2.

## Прототип: whisper.cpp + Vulkan на Windows (2026-03-23)

### Сборка

whisper.cpp собран на Windows с MinGW (MSYS2) + Vulkan:
```
cmake -B build-vulkan -G "MinGW Makefiles" -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
```

Vulkan GPU обнаружен:
```
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon(TM) 780M (AMD proprietary driver) | uma: 1 | fp16: 1 |
  warp size: 64 | shared memory: 32768 | int dot: 1 | matrix cores: KHR_coopmat
```

### Проблема: зацикливание (repetition loop)

Первый запуск (дефолтные параметры): **520с**, но с ~07:00 модель зациклилась
на фразе "Ну, опять же, они могут завести…" — повторяется ~170 раз до конца записи.
Потеряна половина транскрипта.

**Причина**: whisper.cpp передаёт контекст предыдущего сегмента в декодер (`-mc -1`),
что провоцирует repetition loop на длинных записях с паузами/перебивками.

**Решение**: `-mc 0` (не передавать контекст между сегментами).

### Скорость (Vulkan, -mc 0)

| Параметр | Значение |
|----------|----------|
| Модель | ggml-medium.bin (f16, 1533 МБ) на Vulkan GPU |
| Аудио | 14:41 (881с), русский, 3 участника |
| Общее время | **432с** |
| encode | 123.5с (35 проходов, 3.5с/проход) |
| batchd (decode) | 252.7с |
| Потоки | 4 / 16 |

Сравнение с бенчмарком (тот же аудиофайл):

| Конфигурация | Время, с | Ускорение vs CPU f32 |
|---|---|---|
| OpenVINO fp16 (medium) | ~240 | 3.7x |
| **whisper.cpp Vulkan + Radeon 780M** | **~432** | **2.0x** |
| CT2 int8_float32 (medium) | ~599 | 1.5x |
| CT2 float32 / whisper.cpp CPU | ~881-886 | 1.0x |

### Качество (Vulkan, -mc 0)

#### Техтермины

| Термин | whisper.cpp Vulkan | CT2 int8f32 |
|--------|-------------------|-------------|
| DWH | ✅ DWH | ✅ DWH |
| Teradata | ✅ TeraData | ✅ Teradata |
| NiFi | ⚠️ IFI | ✅ NiFi |
| FineBI | ✅ FindBI/FineBI/FNBI | ✅ finebi/FNBI |
| Alation | ✅ Allation | ✅ Allation |
| ETL | ✅ ETL | ✅ ETL |
| DBC Tables V | ⚠️ dbc tables v (lowercase) | ✅ DBC Tables V |
| OpenLineage | ✅ OpenLinage.io | ✅ OpenLinage.io |
| Вьюхи | ✅ Вьюхи | ⚠️ Yuhi |

#### Русская разговорная речь и пунктуация

Качество сопоставимо с CT2 int8_float32. Пунктуация хорошая в обоих случаях.
Сегменты у whisper.cpp короче (2-10 сек vs ~1 мин) — удобнее для навигации.

### Вывод

whisper.cpp + Vulkan на Radeon 780M даёт **432с** — всего **28% быстрее** CT2 int8_float32
(599с). Разница недостаточна, чтобы оправдать сложность интеграции:

- Отдельная сборка whisper.cpp (cmake + MinGW + VulkanSDK)
- Другой формат моделей (GGML вместо CTranslate2)
- Workaround `-mc 0` от зацикливания
- Конвертация в WAV (whisper.cpp не принимает mp4 напрямую)
- Ещё одна зависимость для пользователя

**CT2 int8_float32 остаётся оптимальным выбором**: уже встроен в проект, zero-config,
хорошее качество, приемлемая скорость.

## Рекомендуемый план (обновлённый)

1. ~~**Прототип в WSL** — собрать whisper.cpp с Vulkan, прогнать тестовый файл~~ ✅ Сделано
2. ~~**Качество подтверждено** — whisper.cpp medium f16 ≈ CT2 float32/int8_float32~~ ✅
3. ~~**GPU-ускорение в WSL2** — Vulkan не видит AMD GPU~~ ✅ Подтверждено
4. ~~**Windows-сборка с Vulkan** — 432с, 2x vs CPU, но всего 28% vs CT2 int8_float32~~ ✅
5. **Решение**: whisper.cpp + Vulkan не оправдывает себя на Radeon 780M iGPU.
   CT2 int8_float32 — оптимальный дефолт для CPU без CUDA.
6. NPU — отложить до XDNA 2 или до появления pip-пакета от AMD
7. Пересмотреть при появлении дискретного AMD GPU или WinML с Python-поддержкой
