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

\* По результатам тестирования на Intel и AMD CPU. Реальная скорость зависит от CPU и модели.

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

### Результаты тестирования OpenVINO

Реальные записи рабочих созвонов (русский, техтермины: SQL, PostgreSQL, LDAP, DLP и др.).

**Скорость:**

| Конфигурация | Intel i7 (WSL2) | AMD Ryzen 7 8845H | CPU float32 |
|---|---|---|---|
| OpenVINO medium int8, 16 мин | 171-205с | **185с** | 658-734с |
| OpenVINO medium int8, 42 мин | 413с (6.9 мин) | — | — |
| OpenVINO large-v3 fp16, 16 мин | — | **416с** | — |

**Ускорение vs CPU:** **3-4x** на обоих процессорах (Intel и AMD).

**Качество (сравнение на одном файле, 16 мин, AMD Ryzen 7 8845H):**

| | OpenVINO medium int8 | OpenVINO large-v3 fp16 | CPU medium float32 |
|---|---|---|---|
| Время | 185с | 416с | 734с |
| Техтермины | корректно | корректно | корректно |
| Галлюцинации | нет | нет | нет |
| Характерные ошибки | "Атака новенького" (вместо "а так, новенького") | — | те же ошибки, что medium int8 |
| Пунктуация | базовая | заметно лучше | базовая |

`large-v3` на OpenVINO качественнее `medium`, но в 2.2 раза медленнее. При этом `large-v3` на OpenVINO (416с)
быстрее, чем `medium` на чистом CPU (734с) — можно получить и лучшее качество, и выше скорость.

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
| **OpenVINO + medium int8** | **171-205с** | **413с (6.9 мин)** | **~4-6x реалтайм** |
| CPU + medium float32 | 658с (11 мин) | ~26 мин | ~1.5x реалтайм |
| CPU + large-v3 int8 | 839с (14 мин) | ~37 мин | ~1:1 реалтайм |

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
