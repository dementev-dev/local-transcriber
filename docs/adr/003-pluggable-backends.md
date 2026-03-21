# ADR-003: Pluggable backends и OpenVINO

**Статус**: Принято
**Дата**: 2026-03-21

## Контекст

На CPU (faster-whisper/CTranslate2) транскрипция работает медленно (~1.5x реалтайм для medium).
CUDA доступна на малом проценте машин (ноутбуки с NVIDIA GPU), на офисных ПК её нет.

OpenVINO ускоряет inference на x86 CPU (Intel и AMD) в 2-4 раза. Для его поддержки
нужен второй движок транскрипции, а архитектура должна позволять добавлять новые
бэкенды (CoreML для Mac, AMD XDNA NPU) без переписывания существующего кода.

## Решение

### Backend Protocol (structural typing)

Минимальный интерфейс в `backends/base.py`:

```python
class Backend(Protocol):
    def ensure_model_available(self, model_name, compute_type, on_status) -> str: ...
    def create_model(self, model_path, device, compute_type) -> Any: ...
    def transcribe(self, model, file_path, language, on_segment, on_status) -> TranscribeResult: ...
```

Protocol вместо ABC — бэкенды не наследуются, достаточно реализовать методы.
Соответствует стилю проекта (наследование нигде не используется).

### Ленивые импорты

Бэкенды импортируются только при выборе — `get_backend(device)` делает import внутри.
Импорт faster-whisper запускает CUDA bootstrap (~1ms), импорт openvino-genai загружает ~50MB
shared libraries. Ни то, ни другое не должно происходить, если бэкенд не выбран.

### Device как селектор бэкенда

Вместо отдельного `--backend` флага устройство само определяет бэкенд:
- `cuda`, `cpu` → FasterWhisperBackend
- `openvino` → OpenVINOBackend
- `auto` → CUDA (nvidia-smi) → OpenVINO (import check + x86) → CPU

### load_model() — единственный владелец pipeline

`load_model()` выполняет ensure_model_available + create_model в одном вызове.
CLI не вызывает ensure_model_available отдельно — это убирает двойной resolution
и гарантирует, что модель скачивается для правильного бэкенда.

### Cross-backend fallback

Fallback живёт в `transcriber.py` (оркестратор), не в бэкендах:
- CUDA ошибка → CPU (FasterWhisper)
- OpenVINO ошибка → CPU (FasterWhisper)
- `strict_device=True` (явный `--device`) → ошибка без fallback

При fallback в батч-режиме обновляются model, backend, model_path и actual_device
через TranscribeFileResult — следующий файл использует правильный бэкенд.

### Аудио для OpenVINO

OpenVINO GenAI WhisperPipeline принимает raw PCM float массив, не путь к файлу.
Используем `faster_whisper.decode_audio()` (PyAV) → `.tolist()` → `pipe.generate()`.
Системный ffmpeg не требуется — PyAV бандлит FFmpeg внутри wheel.

### compute_type для OpenVINO

OpenVINO модели предквантизированы (int8/fp16), compute_type определяет какую модель
скачать. Контракт:
- Явный `--compute-type` или значение из конфига — уважается всегда
- Из дефолтов: для large-v3 автоматически выбирается fp16 (стабильнее по качеству)
- Несуществующая пара (model + compute_type) при явном выборе → ошибка

### Обе зависимости по умолчанию

faster-whisper (~37MB) и openvino-genai (~69MB) ставятся вместе — суммарно ~106MB,
приемлемо. Модели скачиваются только для активного бэкенда. CUDA (nvidia-cublas-cu12,
~554MB) остаётся conditional (Linux x86_64). OpenVINO — conditional (x86_64/AMD64, не macOS).

## Последствия

- Обратная совместимость: `transcribe()` сохранён; `load_model()` изменил сигнатуру (возвращает 4-tuple вместо 2-tuple, добавлен `compute_type_explicit`)
- Новый бэкенд добавляется одним файлом в `backends/` + регистрацией в `__init__.py`
- Модели скачиваются по запросу — CUDA пользователь не качает OpenVINO модели, и наоборот
- ARM и macOS: OpenVINO не ставится (platform markers), работает CPU через faster-whisper

## Отклонённые альтернативы

| Альтернатива | Почему отклонена |
|---|---|
| OpenVINO как optional extra (`pip install .[openvino]`) | Теряется zero-config UX; пользователь должен знать про extras |
| whisper.cpp (pywhispercpp) | Другой движок, больший объём интеграции; OpenVINO GenAI проще |
| Единый бэкенд с OpenVINO для всего | CTranslate2 лучше оптимизирован для CUDA; OpenVINO — для CPU |
| ABC вместо Protocol | Наследование не используется в проекте; Protocol проще |
| librosa для загрузки аудио в OpenVINO | Лишняя зависимость; для видеоконтейнеров ненадёжна без системного ffmpeg |
| `--backend` как отдельный флаг | Усложняет CLI; device уже однозначно определяет бэкенд |
