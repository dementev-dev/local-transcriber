# Ускорение транскрипции

## Режимы `--device`

- `auto` (по умолчанию) — CUDA → OpenVINO → CPU (первый доступный)
- `cuda` — строго NVIDIA GPU, ошибка если недоступен
- `openvino` — OpenVINO на CPU (ускорение 2-4x на x86)
- `cpu` — строго CPU (faster-whisper/CTranslate2)

## Какой бэкенд на каком оборудовании

| Оборудование | Рекомендуемый `--device` | Бэкенд | Ожидаемая скорость |
|---|---|---|---|
| NVIDIA GPU (6+ GB VRAM) | `auto` / `cuda` | faster-whisper (CTranslate2) | 7-19x реалтайм |
| Intel/AMD x86 CPU | `auto` / `openvino` | OpenVINO GenAI | 3-6x реалтайм* |
| Любой CPU (fallback) | `cpu` | faster-whisper (CTranslate2) | ~1.5x реалтайм |
| Apple Silicon (macOS) | `cpu` | faster-whisper (CTranslate2) | ~2x реалтайм |

\* Ожидаемая оценка на основе бенчмарков OpenVINO. Реальная скорость зависит от CPU и модели.

## OpenVINO

OpenVINO ускоряет inference на x86 процессорах (Intel и AMD) через оптимизированные инструкции
(AVX2, AVX-512, VNNI, AMX). Ставится автоматически на Linux и Windows (x86_64/AMD64).

- **Модели**: предконвертированные из [HuggingFace](https://huggingface.co/OpenVINO) (int8/fp16)
- **Дефолт**: `medium` + `int8` (для `large-v3` автоматически выбирается `fp16`)
- **Аудиодекодирование**: через PyAV (бандлит FFmpeg), системный ffmpeg не нужен

### Доступные OpenVINO модели

| Модель | int8 | fp16 |
|--------|------|------|
| tiny | OpenVINO/whisper-tiny-int8-ov | — |
| base | — | OpenVINO/whisper-base-fp16-ov |
| small | OpenVINO/whisper-small-int8-ov | — |
| medium | OpenVINO/whisper-medium-int8-ov | — |
| large-v3 | OpenVINO/whisper-large-v3-int8-ov | OpenVINO/whisper-large-v3-fp16-ov |

### Качество OpenVINO int8

OpenVINO использует NNCF (калиброванная post-training квантизация), отличается от runtime-квантизации
CTranslate2. Качество может быть другим — **тестирование на реальных сэмплах рекомендуется**.

## Настройка по платформам

### Linux / WSL2 (x86_64)

Библиотека cuBLAS ставится автоматически (зависимость `nvidia-cublas-cu12` подтягивается при установке).
Дополнительных шагов не требуется.

> На ARM (aarch64) cuBLAS через pip недоступен — нужен системный CUDA toolkit.

### Windows

Нужен системный **CUDA 12** (ctranslate2 4.7 не совместим с CUDA 11 и 13):

```bash
winget install -e --id Nvidia.CUDA --version 12.9   # требует запуска от имени администратора
```

После установки перезапустите терминал.

## Совместимость GPU

| GPU | VRAM | medium float16 | large-v3 float16 | Рекомендация |
|-----|------|---------------|-----------------|--------------|
| RTX 3060 | 6 GB | ✅ | ✅ | medium float16 (дефолт) |
| RTX 4050 | 6 GB | ✅ | ✅ | medium float16 |
| Quadro M3000M | 4 GB | ✅ | ⚠️ tight | medium float16 или int8 |

## Ожидаемая скорость

Замеры на RTX 3060 Laptop (6 GB) и Intel CPU (WSL2):

| Конфигурация | 16 мин файл | 42 мин файл | Отн. скорость |
|-------------|-------------|-------------|---------------|
| GPU + medium float16 | ~35с | ~133с | ~19x реалтайм |
| GPU + large-v3 float16 | ~90с | ~350с | ~7x реалтайм |
| CPU + medium float32 | 613с (10 мин) | ~26 мин* | ~1.5x реалтайм |
| CPU + large-v3 int8 | 839с (14 мин) | ~37 мин* | ~1:1 реалтайм |

*Оценка на основе пропорции.

## Результаты тестирования качества

Тесты проведены на реальных записях рабочих созвонов (русский язык, технические термины:
SQL, PostgreSQL, Greenplum, Airflow, ClickHouse, Docker, CDR, GTP, MAP).

| Конфигурация | Качество (длинная запись, 42 мин) | Проблемы |
|---|---|---|
| large-v3 int8 GPU | Плохо | Галлюцинации (фразы ×25), потеря контента |
| large-v3 float16 GPU | Отлично | — |
| medium float16 GPU | Хорошо | Редкие мелкие ляпы в терминах |
| medium float32 CPU | Хорошо | Сопоставимо с large-v3 int8, без галлюцинаций |
| large-v3 int8 CPU | Хорошо | Без галлюцинаций (на коротких файлах) |

### Ключевые выводы

1. **Указание языка (`--language ru`) критично** — auto-detect может ошибиться и выдать мусор
2. **float16/float32 стабильнее int8** — особенно на записях >20 минут
3. **medium + float16 на GPU — лучший баланс** скорости и качества для повседневного использования
4. **large-v3 + float16 на GPU** — для максимального качества важных записей

## Troubleshooting

### CUDA не обнаружен

Убедитесь, что `nvidia-smi` возвращает информацию о GPU. Драйвер NVIDIA предоставляет `libcuda.so.1`,
без которого CUDA не работает — его нельзя поставить через pip.

### Windows: ошибка при загрузке модели на GPU

GPU на Windows требует **CUDA 12** (ctranslate2 4.7 не совместим с CUDA 11 и 13). Установите:
```bash
winget install -e --id Nvidia.CUDA --version 12.9   # требует запуска от имени администратора
```
После установки перезапустите терминал.

### ARM (aarch64)

cuBLAS через pip недоступен на ARM — нужен системный CUDA toolkit.
