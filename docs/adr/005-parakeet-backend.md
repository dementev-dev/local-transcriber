# ADR-005: Parakeet backend (ONNX Runtime + Silero VAD)

**Статус**: Принято
**Дата**: 2026-04-18

## Контекст

На встроенных GPU AMD/Intel ноутбуков Whisper (через OpenVINO) работает медленно или даёт
невысокое качество на русском. Гипотеза — NVIDIA Parakeet TDT 0.6B v3 (multilingual, 25 языков
включая RU), запущенный через ONNX Runtime CPU EP, даст лучший profile скорости/качества для
этой аудитории. Parakeet изначально CUDA-ориентирован, но community-порт (`onnx-asr`) снимает
это ограничение.

## Решение

### onnx-asr как runtime

Python-пакет `onnx-asr` (istupakov) — лёгкий (numpy + onnxruntime + hf_hub), cross-platform,
поддерживает Parakeet v2/v3 и Silero VAD через `asr.with_vad()`. Альтернативы: NVIDIA NeMo
(~1-2GB deps, PyTorch), OpenVINO порт Parakeet (AMD iGPU не поддерживается). Выбор продиктован
минимализмом зависимостей и кроссплатформенностью.

### Silero VAD — runtime-зависимость, не prefetch

`onnx_asr.load_vad(model="silero")` вызывается в `create_model`. При первом запуске это
качает VAD (~15MB) через свой HF-mechanism. `ensure_model_available` не trogaet VAD —
избегаем double-download-path без точного знания repo/layout. Документировано, что первый
запуск требует сеть.

### Device как селектор (следует ADR-003)

`--device parakeet` и `--device parakeet-cpu` выбирают ParakeetBackend. `parakeet` =
`parakeet-cpu` в MVP (только CPU EP). Extension points для будущих фаз: `parakeet-directml`,
`parakeet-openvino-ep`, `parakeet-cuda`.

### Язык игнорируется, language="multi"

Parakeet v3 в onnx-asr игнорирует `language=` параметр (доступен только для
Whisper/Canary). Backend возвращает `result.language = "multi"`. CLI принудительно
устанавливает `language_mode = "detected"` для Parakeet. Warning на явный CLI `--language`
(не на config) — пользователь предупреждён, что параметр не действует.

### Effective_model_name через backend-атрибут

Parakeet принимает ровно одну модель (`parakeet-tdt-0.6b-v3`) — но `apply_device_defaults`
может подставить из конфига другую. Бэкенд выставляет `self.effective_model_name =
"parakeet-tdt-0.6b-v3"`, CLI читает через `getattr(backend, "effective_model_name", ...)`.
Сигнатура `TranscribeFileResult` не меняется — инвариант остальных бэкендов сохранён.
Альтернатива (добавить поле в `TranscribeFileResult`) отклонена как лишний контрактный шум.

### Config-conflict на `model` — явная ошибка

Если глобальный `.transcriber.toml` содержит `model = "medium"`, а пользователь запускает
`--device parakeet` без `--model`, backend кидает `ValueError` с текстом «передайте --model
parakeet или уберите из конфига». Скрытый override в `apply_device_defaults` отклонён — это
ломает инвариант «CLI > config > defaults».

### No cross-backend fallback

В отличие от OpenVINO/CUDA (fallback на CPU faster-whisper), ошибка Parakeet пробрасывается
наружу. Обоснование: Parakeet — другое семейство моделей (multi vs ru, другое качество),
silent подмена на Whisper ломает интерпретацию результата.

### Quantization

Поддерживаются только `int8` (дефолт) и `float32`. `float16` / `int8_float32` отклоняются с
понятным текстом — onnx-asr Parakeet репозиторий не предоставляет этих вариантов.

## Последствия

- `onnx-asr[cpu,hub]>=0.7` добавлена в `pyproject.toml` (без platform markers — pure Python).
- Первый запуск `--device parakeet` требует интернет (Silero VAD ~15MB + модель ~670MB int8).
- Повторный запуск — оффлайн (проверяется long-audio gate шагом `HF_HUB_OFFLINE=1`).
- Новый device value (`parakeet`/`parakeet-cpu`) в `_VALID_DEVICES`, `DEVICE_DEFAULTS`.
- `transcriber.py::_is_backend_error` знает про Parakeet; fallback пропускается для
  `device.startswith("parakeet")`.
- `TranscribeResult.duration` для Parakeet — время конца последнего речевого сегмента
  (после VAD), не точная длина аудио-файла. Расхождение — только на длину трейлинг-тишины;
  для шапки транскрипта и форматирования таймкодов этого достаточно.

## Known tunables (не включены в MVP, открыты для будущих итераций)

- **`asr.with_vad(...)` параметры Silero**: `threshold`, `min_speech_duration_ms`,
  `max_speech_duration_s`, `min_silence_duration_ms`, `speech_pad_ms`. MVP использует
  дефолты. На специфическом аудио (много пауз, шум, музыка) дефолты могут давать
  субоптимальное разбиение — в этом случае их стоит параметризовать через CLI/config.
- **ONNX Runtime `SessionOptions`**: кроме `intra_op_num_threads` (уже используется через
  `--threads`), есть `inter_op_num_threads`, `execution_mode`, `graph_optimization_level`.
  Для однопоточного inference на слабых CPU может помочь подстройка.
- **Quantization вариант `fp16`**: onnx-asr Parakeet репозиторий сейчас не предоставляет
  fp16 варианта. Если появится — добавить в `SUPPORTED_COMPUTE_TYPES`.

## Отклонённые альтернативы

| Альтернатива | Почему отклонена |
|---|---|
| NVIDIA NeMo toolkit | ~1-2GB зависимостей, тянет PyTorch — избыточно для MVP |
| OpenVINO порт Parakeet (FluidInference) | AMD iGPU не поддерживается; Intel — только экспериментально; у проекта нет компенсирующего выигрыша |
| DirectML EP в MVP | Windows-only; задерживает валидацию гипотезы на Linux |
| ORT-OpenVINO EP в MVP | Двойной слой; сначала нужно подтвердить baseline CPU |
| `onnxruntime-gpu` (CUDA) | Целевая аудитория — без NVIDIA, не приоритет MVP |
| Prefetch Silero VAD через snapshot_download | Двойной download-path без точного знания repo/layout — создаёт расхождение |
| Скрытый override `model` в apply_device_defaults | Ломает инвариант CLI > config > defaults; явная ошибка чище |
| Cross-backend fallback Parakeet → Whisper CPU | Другая модель, другое качество; silent подмена вводит в заблуждение |
| Расширение `TranscribeFileResult.effective_*` | Контрактный шум для всех бэкендов; backend-атрибут + getattr проще |
| Ручной chunking длинных файлов в MVP | VAD должен справляться; проверяем gate, добавляем только при провале |
| `--device auto` включает Parakeet | Экспериментальный; могут возникнуть регрессии качества на RU |

## См. также

- Спек: `docs/superpowers/specs/2026-04-18-parakeet-backend-design.md`
- План реализации: `docs/superpowers/plans/2026-04-18-parakeet-backend.md`
- Long-audio gate результаты: `docs/gpu.md`, секция «Parakeet TDT v3»
