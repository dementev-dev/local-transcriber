# ADR-003: Pluggable backends и OpenVINO

**Статус**: Принято
**Дата**: 2026-03-21
**Обновлено**: 2026-09-21

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
CUDA bootstrap вызывается только при создании CUDA-модели (см. ADR-001).
Импорт openvino-genai загружает ~50MB shared libraries и не должен происходить,
если бэкенд не выбран.

### Device как селектор бэкенда

Вместо отдельного `--backend` флага устройство само определяет бэкенд:
- `cuda`, `cpu` → FasterWhisperBackend
- `openvino` → OpenVINOBackend
- `onnx` → OnnxAsrBackend
- `auto` → ONNX на CPU независимо от `nvidia-smi` и установки extra

### Transcriber — единственный владелец загрузки и исполнения

Обновлено 2026-09-21. `Transcriber(ExecutionRequest)` в `transcriber.py`
разрешает `auto` и device-aware умолчания, выполняет ensure_model_available +
create_model в одном месте и держит модель, adapter и фактическое исполнение
на весь запуск. CLI не вызывает `get_backend` и не хранит handles.

Внутри module различаются три вещи: поддерживаемая модель, движок
распознавания (`faster-whisper`, `openvino`, `onnx-asr`) и способ исполнения.
Значение `device` из CLI/TOML сохраняется как вход: таблица `_ENGINES`
сопоставляет ему движок, а аппаратную часть (CPU, CUDA, Intel GPU, список
providers) трактует adapter. Новое исполнение — строка таблицы и его
понимание в adapter'е, не новая иерархия adapters.

### Cross-backend fallback

Fallback живёт внутри `Transcriber`, не в бэкендах и не в CLI:
- CUDA ошибка → CPU (FasterWhisper)
- OpenVINO ошибка → CPU (FasterWhisper)
- strict запуск (явный device из CLI или TOML) → ошибка без fallback

После перехода module сам держит новое состояние; следующий файл батча
использует его без участия caller. `WordTimestampsUnavailableError`
и ошибки пользовательских данных переходом не считаются.

### Диагностика исполнения

Каждый adapter отдаёт `runtime_info()` — версии runtime и фактическую
конфигурацию (для ONNX — доступные providers и те, что заданы сессиям ASR
и VAD, квантизация, бюджет потоков). `Transcriber` включает их в
`ExecutionInfo`; CLI печатает их в `--verbose`. Наличие provider в wheel не
считается доказательством выполнения графа на GPU/NPU.

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

### Зависимости бэкендов по умолчанию

faster-whisper, onnx-asr/onnxruntime и openvino-genai ставятся вместе. Модели
скачиваются только для активного бэкенда. cuBLAS подключается через extra `cuda`
для Linux/WSL x86_64 и Windows x64 (см. ADR-001). OpenVINO — conditional для x86_64/AMD64, кроме
macOS; ONNX обеспечивает автоматический CPU-путь на остальных платформах.

## Последствия

- Обратная совместимость: `transcribe()` сохранён; `load_model()` и `_transcribe_file()` удалены (2026-09-21)
- Новый бэкенд добавляется одним файлом в `backends/` + регистрацией в `__init__.py`
- Модели скачиваются по запросу — CUDA пользователь не качает OpenVINO модели, и наоборот
- ARM и macOS: OpenVINO не ставится (platform markers), auto использует ONNX

## Отклонённые альтернативы

| Альтернатива | Почему отклонена |
|---|---|
| OpenVINO как optional extra (`pip install .[openvino]`) | Теряется zero-config UX; пользователь должен знать про extras |
| whisper.cpp (pywhispercpp) | Другой движок, больший объём интеграции; OpenVINO GenAI проще |
| Единый бэкенд с OpenVINO для всего | CTranslate2 лучше оптимизирован для CUDA; OpenVINO — для CPU |
| ABC вместо Protocol | Наследование не используется в проекте; Protocol проще |
| librosa для загрузки аудио в OpenVINO | Лишняя зависимость; для видеоконтейнеров ненадёжна без системного ffmpeg |
| `--backend` как отдельный флаг | Усложняет CLI; device уже однозначно определяет бэкенд |
