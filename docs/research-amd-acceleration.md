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

## Рекомендуемый план

1. **Прототип** whisper.cpp + Vulkan бэкенда (pywhispercpp или subprocess)
2. **Бенчмарк** на том же тестовом файле — сравнить скорость и качество GGUF vs CTranslate2
3. **Решение** о включении в основной проект по результатам бенчмарка
4. NPU — отложить до XDNA 2 или до появления pip-пакета от AMD
