# Parakeet backend — design spec

**Дата:** 2026-04-18
**Статус:** Готов к планированию реализации
**Ветка реализации:** `feature/parakeet-backend` (в отдельном git worktree)

## Контекст и гипотеза

`local-transcriber` — CLI для локальной транскрипции аудио/видео (Whisper). Архитектура pluggable backends (ADR-003): `FasterWhisperBackend` (CTranslate2, CPU/CUDA), `OpenVINOBackend` (Intel CPU/iGPU/NPU). Дефолтный язык — `ru`. Целевая аудитория — ноутбуки с Intel и AMD встройками; NVIDIA — меньшинство.

**Гипотеза.** NVIDIA Parakeet TDT 0.6B v3 (через ONNX Runtime на CPU) быстрее и/или качественнее Whisper-бэкендов на Intel/AMD встройках. Проверяется прогоном на реальных русских файлах с сравнением wall-clock и качества против OpenVINO-Whisper medium и faster-whisper CPU medium.

## Выбор технологии (зафиксирован)

- **Модель:** `nvidia/parakeet-tdt-0.6b-v3` — multilingual, 25 языков включая русский.
- **Runtime:** ONNX Runtime через Python-пакет `onnx-asr` ([`istupakov/onnx-asr`](https://github.com/istupakov/onnx-asr)). Отклонены: NeMo toolkit (~1-2GB зависимостей), OpenVINO Parakeet (AMD iGPU не поддерживается, Intel — экспериментальный порт без сообщества).
- **Execution Provider в MVP:** CPU. DirectML (Windows AMD/Intel iGPU), ORT-OpenVINO EP, `onnxruntime-gpu` (CUDA) — отдельные фазы после валидации гипотезы.
- **VAD:** Silero VAD через `asr.with_vad()` — обязательно (без VAD лимит 20-30 сек на файл; все реальные файлы длиннее).
- **Quantization:** `int8` по умолчанию (~670MB диск, ~2GB RAM при inference), `fp32` — опциональное значение `--compute-type float32`.

## Архитектура

### Инвариант

`TranscribeFileResult`, `TranscribeResult`, `Segment`, `format_transcript()` — **не меняем**. Регрессионный риск для существующих Whisper/OpenVINO-путей минимизирован.

Parakeet интегрируется через две минимальные точки:
1. `result.language = "multi"` внутри `ParakeetBackend.transcribe` — соответствует поведению Parakeet v3.
2. Атрибут backend-инстанса `effective_model_name` — аналог существующих `actual_compute_type`, `actual_ov_device`.

CLI после `_transcribe_file` читает `getattr(tfr.backend, "effective_model_name", defaults["model"])` для rich-вывода и `format_transcript(..., model_name=...)`. Rich-строка `Модель: ...` **переносится** после `_transcribe_file` (точка, где уже доступен `tfr.backend`). До транскрипции выводится только `Файл: ...` и сообщения `on_status` от загрузки модели.

**Шапка транскрипта для Parakeet:** `result.language = "multi"` + `language_mode = "detected"` (CLI форсит `"detected"` при `--device parakeet*`). Формат — `**Язык**: multi (detected)`.

### Контракт `--language`

- `result.language = "multi"` всегда.
- CLI при `--device parakeet*` форсит `language_mode = "detected"` (независимо от того, задавал ли пользователь `--language`).
- **Warning только при явном CLI-флаге `--language`.** Если `language` пришёл из `.transcriber.toml` или `HARDCODED_DEFAULTS` (без явного CLI) — warning подавлен. Источник явности: `cli_values["language"] is not None` (не `"language" in config`).
- Текст warning: `Parakeet игнорирует --language; язык определяется автоматически`.

### Контракт `--compute-type`

| CLI `--compute-type` | `onnx_asr.load_model(quantization=...)` | Примечание |
|----------------------|------------------------------------------|------------|
| `int8`               | `"int8"`                                 | Дефолт |
| `float32`            | `None`                                   | Без quantization |
| другое               | —                                        | `ValueError("Parakeet поддерживает только int8 или float32")` |

### Контракт VAD — runtime-зависимость, без prefetch

`ensure_model_available` отвечает только за Parakeet-модель. `create_model` вызывает `onnx_asr.load_vad(model="silero")`, который при первом запуске скачивает Silero VAD (~15MB).

Одна точка правды — `onnx_asr.load_vad`. Двойной prefetch (наш `snapshot_download` + внутренний `onnx-asr`) без точного знания repo/layout создавал несостыковку: `ensure_model_available` рапортует «готово», но `create_model` всё равно тянет из сети.

Документировано в ADR-005 и README: **первый запуск `--device parakeet` требует интернет для Silero VAD; повторные — оффлайн** (проверяется в long-audio gate).

### Контракт fallback — **НЕ fallback'ить Parakeet на Whisper CPU**

В `load_model` и `_transcribe_file` при `actual_device.startswith("parakeet")` — пропускаем fallback-ветку, исключение пробрасывается наружу.

Обоснование: Parakeet — другое семейство моделей с другим качеством и семантикой языка (`multi` vs `ru`). Молчаливая подмена на Whisper medium ломает интерпретацию транскрипта. Пользователь видит понятную ошибку (`Ошибка загрузки Silero VAD — проверьте интернет` / `Parakeet inference crashed — см. --verbose`) и делает осознанный выбор.

### `detect_device` (utils.py)

`--device auto` **не переключается на Parakeet.** Порядок остаётся: CUDA → OpenVINO GPU → OpenVINO CPU → CPU. Parakeet — только через явный `--device parakeet` / `--device parakeet-cpu`.

### `config.py`

```python
_VALID_DEVICES = {
    "auto", "cpu", "cuda",
    "openvino", "openvino-gpu", "openvino-cpu",
    "parakeet", "parakeet-cpu",
}

DEVICE_DEFAULTS["parakeet"] = {"model": "parakeet-tdt-0.6b-v3", "compute_type": "int8"}
DEVICE_DEFAULTS["parakeet-cpu"] = {"model": "parakeet-tdt-0.6b-v3", "compute_type": "int8"}
```

**Config-conflict на `model`.** Если в глобальном `.transcriber.toml` стоит `model = "medium"` и пользователь запускает `--device parakeet` без `--model`, `apply_device_defaults` возьмёт `"medium"` из конфига → `ParakeetBackend.ensure_model_available` кидает **понятный `ValueError`**:

> `Parakeet поддерживает только 'parakeet-tdt-0.6b-v3' или алиас 'parakeet'. Получено: 'medium'. Передайте --model parakeet или уберите 'model' из .transcriber.toml.`

Принципиально **не делаем** скрытого override в `apply_device_defaults` — это ломает инвариант «CLI > config > defaults». Пользователь получает ясную ошибку и делает явный фикс.

Алиас `"parakeet"` → `"parakeet-tdt-0.6b-v3"` допустим в `ensure_model_available` (UX-удобство).

### Новый бэкенд — `backends/parakeet.py`

```python
class ParakeetBackend:
    SUPPORTED_MODEL_NAMES = {"parakeet-tdt-0.6b-v3", "parakeet"}
    SUPPORTED_COMPUTE_TYPES = {"int8", "float32"}

    def __init__(self, compute_type_explicit: bool = True):
        self._compute_type_explicit = compute_type_explicit
        self.actual_compute_type: str | None = None
        self.effective_model_name: str = "parakeet-tdt-0.6b-v3"

    def ensure_model_available(self, model_name, compute_type, on_status) -> str:
        # 1. model_name ∈ SUPPORTED_MODEL_NAMES иначе ValueError (текст выше)
        # 2. compute_type ∈ SUPPORTED_COMPUTE_TYPES иначе ValueError
        # 3. snapshot_download("nvidia/parakeet-tdt-0.6b-v3") → локальный путь
        # 4. Silero VAD здесь НЕ трогаем (runtime-зависимость)
        # 5. self.actual_compute_type = compute_type
        # 6. Возвращаем путь к каталогу

    def create_model(self, model_path, device, compute_type, cpu_threads=0) -> Any:
        # 1. quantization = "int8" if compute_type == "int8" else None
        # 2. asr = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", path=model_path, quantization=quantization)
        # 3. vad = onnx_asr.load_vad(model="silero")   # может качать VAD при первом запуске
        # 4. asr_with_vad = asr.with_vad(vad=vad)
        # 5. cpu_threads → sess_options через providers_options
        # 6. Возвращаем asr_with_vad

    def transcribe(self, model, file_path, language, on_segment, on_status) -> TranscribeResult:
        # 1. Если language задан явно (см. «Контракт --language») — warnings.warn
        # 2. result = model.recognize(str(file_path))
        # 3. Поддержка iterator И list (детектим на лету)
        # 4. Для каждого SegmentResult → Segment(start, end, text); on_segment; on_status c heartbeat
        # 5. Возвращаем TranscribeResult(segments, language="multi", language_probability=0.0, duration, device_used="")
```

### Регистрация в `backends/__init__.py`

```python
if device in ("parakeet", "parakeet-cpu"):
    try:
        from .parakeet import ParakeetBackend
    except ImportError:
        raise ValueError(
            "Parakeet бэкенд недоступен. Установите: pip install onnx-asr[cpu,hub]"
        ) from None
    return ParakeetBackend(compute_type_explicit=compute_type_explicit)
```

### `transcriber.py` — no-fallback для Parakeet

```python
def _is_parakeet_error(exc: BaseException) -> bool:
    if isinstance(exc, RuntimeError):
        return True
    mod = type(exc).__module__ or ""
    return "onnxruntime" in mod or "onnx_asr" in mod

def _is_backend_error(exc, device):
    if device.startswith("parakeet"):
        return _is_parakeet_error(exc)
    # ... existing branches
```

В `load_model` и `_transcribe_file` добавляется ранний выход из fallback-ветки при `actual_device.startswith("parakeet")` — исключение пробрасывается (как будто `strict_device=True`).

### `cli.py` — точечные изменения

1. **Rich-строка `Модель: ...` переносится после `_transcribe_file`.** Использует `getattr(tfr.backend, "effective_model_name", defaults["model"])`.
2. **`format_transcript(..., model_name=...)`** получает `getattr(tfr.backend, "effective_model_name", defaults["model"])`.
3. **`language_mode`** при Parakeet: `if actual_device.startswith("parakeet"): language_mode = "detected"`. Иначе — существующая логика.
4. **Warning на явный `--language`** при Parakeet: проверка `cli_values["language"] is not None` (не `"language" in config`). Срабатывает в `_run_single` / `_run_batch` до `_transcribe_file` — пользователь видит сразу.
5. **`_format_device_info`**: `if device_used == "parakeet-cpu": return "Parakeet (CPU via ONNX Runtime)"`.

Для Whisper-бэкендов код не меняется.

### Progress UX — heartbeat если блокирующий вызов

Если `asr_with_vad.recognize()` возвращает iterator — итерируем с вызовами `on_status` per segment. Если блокирующий — обёртка в поток с heartbeat-таймером (паттерн `openvino.py::_generate_with_progress`). Тип возврата проверяется в long-audio gate (см. ниже); при блокирующем вызове heartbeat — **обязателен до merge**.

## Зависимости

```toml
# pyproject.toml
dependencies = [
    # ... existing
    "onnx-asr[cpu,hub]>=0.7",  # минимальная версия уточнить после uv lock
]
```

`uv lock` обязателен до любой имплементации — проверить, что `onnxruntime` не конфликтует с тем, что тянут `faster-whisper` / `openvino-genai`. Platform markers не нужны: `onnx-asr` — pure Python; `onnxruntime` CPU-wheels покрывают Linux/macOS/Windows × x86_64/arm64.

## Тесты

### `tests/test_parakeet_backend.py` — юнит с mock `onnx_asr`

1. `test_ensure_model_available_accepts_alias` — `model_name="parakeet"` → скачивает `nvidia/parakeet-tdt-0.6b-v3`.
2. `test_ensure_model_available_rejects_unknown_model` — `model_name="medium"` → `ValueError` с текстом-подсказкой.
3. `test_ensure_model_available_rejects_unknown_compute_type` — `compute_type="float16"` → `ValueError`.
4. `test_create_model_maps_compute_type` — `int8` → `quantization="int8"`; `float32` → `quantization=None`.
5. `test_transcribe_produces_segments` — mock `recognize` возвращает 3 `SegmentResult` → 3 `Segment`.
6. `test_transcribe_warns_on_explicit_cli_language` — явный `--language` → `warnings.warn`.
7. `test_transcribe_does_not_warn_on_config_language` — `language` из конфига → warning НЕ срабатывает.
8. `test_transcribe_calls_on_segment` — callback на каждый сегмент.
9. `test_transcribe_returns_multi_language` — `result.language == "multi"`, `language_probability == 0.0`.
10. `test_backend_sets_effective_model_name` — атрибут проставлен.
11. `test_transcribe_accepts_list_and_iterator_results` — оба типа возврата обрабатываются.

### `tests/test_transcriber.py`

12. `test_load_model_parakeet_no_fallback` — при ошибке Parakeet исключение пробрасывается (без fallback на CPU Whisper).

### `tests/test_cli.py` — regression + новые

13. `test_cli_whisper_transcript_header_unchanged` — `--device cpu --language ru` → шапка содержит `**Язык**: ru (forced)` (регрессия отсутствует).
14. `test_cli_whisper_auto_language_header_unchanged` — `--device cpu --language auto` → шапка отражает detected-режим.
15. `test_cli_parakeet_transcript_header` — `--device parakeet` → шапка `**Язык**: multi (detected)` и `**Модель**: parakeet-tdt-0.6b-v3`.
16. `test_cli_parakeet_model_row_shows_effective` — config `model="medium"` + `--device parakeet --model parakeet` → rich-вывод и `format_transcript` получают `parakeet-tdt-0.6b-v3`.
17. `test_cli_errors_on_config_model_mismatch_without_override` — config `model="medium"` + `--device parakeet` без `--model` → ясная ошибка (не traceback).
18. `test_cli_parakeet_warns_explicit_language` — config без `language`, явный `--language en --device parakeet` → warning.

Все юнит-тесты — чистые mocks, без реального скачивания.

## Long-audio verification gate — exit criteria, blocker для merge

**Метрики и нормы (единственный источник истины):**

| Метрика | Норма |
|---------|-------|
| Длина тестового файла | ≥30 минут, русский |
| Peak RSS | ≤4 GB |
| Первый `on_status` update | ≤10 сек от старта `_transcribe_file` |
| Максимальный интервал между `on_status` update'ами | ≤30 сек |
| Повторный прогон с `HF_HUB_OFFLINE=1` | без сетевых запросов |

**Методика измерения:**
- Peak RSS: `/usr/bin/time -v ...` (поле "Maximum resident set size") или `psutil.Process().memory_info().rss` из внешнего процесса раз в секунду.
- Интервалы `on_status`: специальный тестовый callback в CLI логирует `time.monotonic()` каждого вызова в файл; пост-анализ считает max interval.
- Offline: `HF_HUB_OFFLINE=1` через `env`; либо `strace -e trace=connect`, либо проверка отсутствия задержек на сетевых операциях в начале прогона.

**Blocker-правила (план заморожен до устранения):**
- Peak RSS > 4 GB → нужен ручной chunking; MVP не готов.
- Первый update > 10 сек ИЛИ любой интервал > 30 сек → нужен heartbeat-thread (паттерн `openvino.py::_generate_with_progress`).
- Offline-прогон делает сетевые обращения → VAD-кэш не работает, разбираться до merge.

Результаты (числа всех метрик) фиксируются в `docs/gpu.md`.

## Документация

1. **ADR-005** (`docs/adr/005-parakeet-backend.md`) — контекст, решение (ONNX Runtime + onnx-asr + Silero VAD), отклонённые альтернативы (NeMo toolkit, OpenVINO Parakeet, DirectML в MVP, VAD prefetch, скрытый override в `apply_device_defaults`, cross-backend fallback на Whisper), последствия (язык игнорируется, no fallback, `effective_model_name` через backend-атрибут, VAD runtime-зависимость с требованием интернета на первый запуск).
2. **README.md** — секция «Parakeet (экспериментально)»: установка, `--device parakeet`, ограничения (int8/fp32, язык auto, первый запуск — интернет для VAD ~15MB, память ~2GB, конфликт с `model` в конфиге).
3. **docs/gpu.md** — секция «Parakeet»: результаты long-audio gate + сравнение с OpenVINO medium / faster-whisper medium CPU на одном и том же RU-файле.

## Порядок шагов (для writing-plans)

Вся работа — в git worktree на ветке `feature/parakeet-backend` (через `superpowers:using-git-worktrees`).

1. **Scaffold + зависимости:** worktree, ветка, `uv add "onnx-asr[cpu,hub]>=0.7"`, `uv lock` → нет конфликтов `onnxruntime` с существующими deps (**блокер**). `uv sync`. Существующие `uv run pytest` зелёные.
2. **config.py:** `_VALID_DEVICES` + `DEVICE_DEFAULTS` для parakeet/parakeet-cpu. Тест на валидацию `device = "parakeet"` в конфиге.
3. **ParakeetBackend (заглушка):** файл `backends/parakeet.py` со stub'ами (`raise NotImplementedError`). Регистрация в `backends/__init__.py`. Импорт работает без ошибок.
4. **`ensure_model_available`:** валидация model_name / compute_type, `snapshot_download`, валидация каталога. Тесты 1-3.
5. **`create_model`:** `onnx_asr.load_model + load_vad + with_vad`, маппинг compute_type, cpu_threads. Тест 4.
6. **`transcribe`:** итерация `SegmentResult`, warning на язык, `language="multi"`, duration. Тесты 5-11.
7. **`transcriber.py` — no-fallback:** `_is_parakeet_error`, ветка в `_is_backend_error`, пропуск fallback при `actual_device.startswith("parakeet")`. Тест 12.
8. **cli.py:** перенос строки `Модель:` после `_transcribe_file`, чтение `effective_model_name` через `getattr`, force `language_mode="detected"` для Parakeet, warning на явный CLI `--language`, `_format_device_info` для `parakeet-cpu`. Regression-тесты 13-14, Parakeet-тесты 15-18.
9. **Long-audio verification gate:** прогон ≥30-мин RU-файла, замеры всех метрик, оффлайн-повтор. Фиксация в `docs/gpu.md`. **Blocker для merge.**
10. **ADR-005 + README** с финальными числами из шага 9.
11. **Финал:** `uv run pytest -v` зелёное, `uv run transcribe sample.mp3 --device parakeet` работает end-to-end, PR.

## Что НЕ делаем в MVP (отложено)

- DirectML EP для AMD iGPU / Intel iGPU (Windows).
- ORT-OpenVINO EP для Intel iGPU/NPU.
- Parakeet на CUDA (`onnxruntime-gpu`).
- Ручной chunking длинных файлов (доверяем `with_vad` + gate).
- `--device auto` включает Parakeet.
- Alias `parakeet-tdt-0.6b-v3` для моделей других бэкендов.
- Canary (ещё одна NeMo-модель, поддерживается `onnx-asr`).
- Prefetch Silero VAD в `ensure_model_available`.
- Изменение сигнатур `TranscribeFileResult` / `TranscribeResult` / `format_transcript`.

## Открытые риски и митигации

1. **Память на длинных файлах** — gate шага 9; провал → блокер MVP до ручного chunking.
2. **Progress UX (блокирующий `recognize`)** — gate шага 9; провал → heartbeat-thread обязателен.
3. **Качество на русском** — основная гипотеза; результат фиксируется в `docs/gpu.md`; при ухудшении по сравнению с Whisper — бэкенд остаётся с пометкой «EN лучше, чем RU».
4. **Конфликт версий `onnxruntime`** — блокер шага 1; решается через `uv lock` до любой имплементации.
5. **VAD offline** — gate шага 9 (`HF_HUB_OFFLINE=1`); провал → разбираться, при необходимости возвращаться к prefetch с реальной проверкой repo/layout.
6. **Config-conflict на `model`** — ясная ошибка, не скрытый override.
