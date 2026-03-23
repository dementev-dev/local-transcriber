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

## Рекомендуемый план

1. **Прототип в WSL** — собрать whisper.cpp с Vulkan, прогнать тестовый файл
2. **Бенчмарк** на том же тестовом файле — сравнить скорость и качество GGML vs CTranslate2
3. **Решение** о включении в основной проект по результатам бенчмарка
4. Если ОК — оформить как бэкенд `backends/whisper_cpp.py`
5. NPU — отложить до XDNA 2 или до появления pip-пакета от AMD
