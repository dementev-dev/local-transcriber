# Parakeet Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Добавить в `local-transcriber` новый бэкенд транскрипции на NVIDIA Parakeet TDT 0.6B v3 (через `onnx-asr` + Silero VAD) как `--device parakeet` / `--device parakeet-cpu`.

**Architecture:** Реализация класса `ParakeetBackend` в `src/local_transcriber/backends/parakeet.py` по существующему `Backend` Protocol; регистрация в `get_backend()`; точечные правки `config.py`, `transcriber.py` (no-fallback для Parakeet), `cli.py` (effective_model_name через `getattr`, force `language_mode="detected"`, warning на явный `--language`). Типы данных (`TranscribeFileResult`, `TranscribeResult`, `Segment`) и сигнатура `format_transcript` **не меняются**.

**Tech Stack:** Python 3.10+, `onnx-asr[cpu,hub]>=0.7`, ONNX Runtime CPU EP, Silero VAD (через `onnx-asr.load_vad`), `huggingface-hub`, pytest, typer/rich.

**Spec reference:** `docs/superpowers/specs/2026-04-18-parakeet-backend-design.md`

**Worktree:** `/home/dementev/sources/local-transcriber-parakeet` (ветка `feature/parakeet-backend`). Все команды выполняются из этого каталога.

---

## File Structure

**Создаются:**
- `src/local_transcriber/backends/parakeet.py` — реализация `ParakeetBackend`.
- `tests/test_backend_parakeet.py` — юнит-тесты.
- `docs/adr/005-parakeet-backend.md` — ADR.
- `scripts/measure_parakeet.py` — standalone-измеритель для long-audio gate (Task 10).

**Модифицируются:**
- `pyproject.toml` — добавление `onnx-asr[cpu,hub]>=0.7` (Task 1) и `psutil` в dev (Task 10).
- `src/local_transcriber/backends/__init__.py` — регистрация бэкенда в `get_backend()`.
- `src/local_transcriber/config.py` — `_VALID_DEVICES` и `DEVICE_DEFAULTS`.
- `src/local_transcriber/transcriber.py` — `_is_parakeet_error`, ветка в `_is_backend_error`, пропуск fallback при `actual_device.startswith("parakeet")`.
- `src/local_transcriber/cli.py` — чтение `effective_model_name` через `getattr`, force `language_mode="detected"` для Parakeet, warning на явный CLI `--language`, `_format_device_info` для `parakeet-cpu`, передача `language_explicit` в `_run_single`/`_run_batch`.
- `tests/test_transcriber.py` — тест no-fallback для Parakeet.
- `tests/test_cli.py` — regression Whisper + новые Parakeet-тесты.
- `tests/test_config.py` — валидация `device = "parakeet"`.
- `README.md` — секция «Parakeet (экспериментально)».
- `docs/gpu.md` — раздел «Parakeet» с результатами long-audio gate.

**Принципы:**
- Коммитим после каждой Task (мелкими шагами).
- TDD: сначала падающий тест, потом минимальная реализация.
- `uv run pytest` прогоняем до и после каждой Task.

---

## Task 1: Добавить зависимость `onnx-asr` и проверить отсутствие конфликтов

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock` (автоматически)

- [ ] **Step 1: Добавить зависимость через uv**

```bash
cd /home/dementev/sources/local-transcriber-parakeet
uv add "onnx-asr[cpu,hub]>=0.7"
```

Expected: `pyproject.toml` содержит новую строку в `dependencies`, `uv.lock` обновлён, пакет установлен в `.venv`.

- [ ] **Step 2: Проверить, что `onnxruntime` не конфликтует с существующими deps**

```bash
uv pip list | grep -iE "onnxruntime|faster-whisper|openvino"
```

Expected: `onnxruntime` установлен одной версией; `faster-whisper` и `openvino-genai` остались в своих версиях без downgrade. Если видно предупреждение «conflict» — **блокер**: остановиться, зафиксировать версии в issue, обсудить с пользователем.

- [ ] **Step 3: Sanity-импорт**

```bash
uv run python -c "import onnx_asr; print(onnx_asr.__version__)"
```

Expected: печатает версию без traceback.

- [ ] **Step 4: Существующие тесты всё ещё зелёные**

```bash
uv run pytest -v
```

Expected: все тесты проходят (регрессия по `onnxruntime` / numpy не проявилась).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -F- <<'EOF'
feat(deps): добавлен onnx-asr для Parakeet бэкенда

- Зачем:
  - требуется runtime для NVIDIA Parakeet TDT v3 на CPU (ONNX Runtime + Silero VAD).
- Что:
  - onnx-asr[cpu,hub]>=0.7 в dependencies.
  - проверено отсутствие конфликта onnxruntime с faster-whisper / openvino-genai.
- Проверка:
  - uv run pytest -v.
EOF
```

---

## Task 2: Расширить `_VALID_DEVICES` и `DEVICE_DEFAULTS` в `config.py`

**Files:**
- Modify: `src/local_transcriber/config.py:19-29`
- Modify: `tests/test_config.py` (новый тест)

- [ ] **Step 1: Написать падающий тест в `tests/test_config.py`**

Найти секцию тестов `load_config` (валидация `device`) и добавить:

```python
def test_load_config_accepts_parakeet_device(tmp_path):
    cfg = tmp_path / ".transcriber.toml"
    cfg.write_text('device = "parakeet"\n')
    assert load_config(cfg) == {"device": "parakeet"}


def test_load_config_accepts_parakeet_cpu_device(tmp_path):
    cfg = tmp_path / ".transcriber.toml"
    cfg.write_text('device = "parakeet-cpu"\n')
    assert load_config(cfg) == {"device": "parakeet-cpu"}
```

- [ ] **Step 2: Прогнать тесты — ожидаем FAIL**

```bash
uv run pytest tests/test_config.py::test_load_config_accepts_parakeet_device tests/test_config.py::test_load_config_accepts_parakeet_cpu_device -v
```

Expected: FAIL — `device = "parakeet"` сейчас отвергается валидацией (`Недопустимое значение device`).

- [ ] **Step 3: Обновить `_VALID_DEVICES` и `DEVICE_DEFAULTS`**

В `src/local_transcriber/config.py`:

```python
DEVICE_DEFAULTS: dict[str, dict[str, str]] = {
    "cuda": {"model": "medium", "compute_type": "float16"},
    "cpu": {"model": "medium", "compute_type": "float32"},
    "openvino": {"model": "medium", "compute_type": "int8"},
    "openvino-gpu": {"model": "medium", "compute_type": "int8"},
    "openvino-cpu": {"model": "medium", "compute_type": "int8"},
    "parakeet": {"model": "parakeet-tdt-0.6b-v3", "compute_type": "int8"},
    "parakeet-cpu": {"model": "parakeet-tdt-0.6b-v3", "compute_type": "int8"},
}

# Одно место правды для допустимых ключей конфига
_VALID_KEYS = set(HARDCODED_DEFAULTS)
_VALID_DEVICES = {
    "auto", "cpu", "cuda",
    "openvino", "openvino-gpu", "openvino-cpu",
    "parakeet", "parakeet-cpu",
}
```

- [ ] **Step 4: Прогнать тесты — ожидаем PASS**

```bash
uv run pytest tests/test_config.py -v
```

Expected: PASS для всех тестов (включая два новых).

- [ ] **Step 5: Commit**

```bash
git add src/local_transcriber/config.py tests/test_config.py
git commit -F- <<'EOF'
feat(config): допустимые device-значения parakeet/parakeet-cpu

- Зачем:
  - для интеграции Parakeet бэкенда нужно разрешить новые device-значения в .transcriber.toml и дефолтах.
- Что:
  - добавлены "parakeet" и "parakeet-cpu" в _VALID_DEVICES.
  - DEVICE_DEFAULTS для обоих значений: model=parakeet-tdt-0.6b-v3, compute_type=int8.
  - два новых теста в test_config.py.
- Проверка:
  - uv run pytest tests/test_config.py -v.
EOF
```

---

## Task 3: Scaffold `ParakeetBackend` (заглушки) + регистрация

**Files:**
- Create: `src/local_transcriber/backends/parakeet.py`
- Modify: `src/local_transcriber/backends/__init__.py`

- [ ] **Step 1: Создать заглушку `backends/parakeet.py`**

Полный файл:

```python
"""Бэкенд транскрипции на основе NVIDIA Parakeet TDT (onnx-asr + Silero VAD)."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from local_transcriber.types import Segment, TranscribeResult

MODEL_REPO = "nvidia/parakeet-tdt-0.6b-v3"
ONNX_ASR_MODEL_ALIAS = "nemo-parakeet-tdt-0.6b-v3"

SUPPORTED_MODEL_NAMES: set[str] = {"parakeet-tdt-0.6b-v3", "parakeet"}
SUPPORTED_COMPUTE_TYPES: set[str] = {"int8", "float32"}

MODEL_REQUIRED_FILES: list[str] = [
    "config.json",
]


class ParakeetBackend:
    """Бэкенд Parakeet TDT v3 через onnx-asr (CPU execution provider)."""

    def __init__(self, compute_type_explicit: bool = True):
        self._compute_type_explicit = compute_type_explicit
        self.actual_compute_type: str | None = None
        self.effective_model_name: str = "parakeet-tdt-0.6b-v3"

    def ensure_model_available(
        self,
        model_name: str,
        compute_type: str,
        on_status: Callable[[str], None] | None = None,
    ) -> str:
        raise NotImplementedError

    def create_model(
        self,
        model_path: str,
        device: str,
        compute_type: str,
        cpu_threads: int = 0,
    ) -> Any:
        raise NotImplementedError

    def transcribe(
        self,
        model: Any,
        file_path: Path,
        language: str | None,
        on_segment: Callable[[Segment], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> TranscribeResult:
        raise NotImplementedError


def _notify(on_status: Callable[[str], None] | None, message: str) -> None:
    if on_status is not None:
        on_status(message)
```

- [ ] **Step 2: Зарегистрировать в `backends/__init__.py`**

Прочитать текущий `src/local_transcriber/backends/__init__.py`, добавить ветку ДО `from .faster_whisper`:

```python
def get_backend(device: str, *, compute_type_explicit: bool = True) -> Backend:
    """Возвращает экземпляр бэкенда для указанного устройства.

    Импорты ленивые — бэкенд загружается только при запросе.
    compute_type_explicit: False если compute_type пришёл из дефолтов (влияет на fallback).
    """
    if device in ("openvino", "openvino-gpu", "openvino-cpu"):
        try:
            from .openvino import OpenVINOBackend
        except ImportError:
            raise ValueError(
                "OpenVINO бэкенд недоступен. Установите: pip install openvino-genai"
            ) from None
        return OpenVINOBackend(
            ov_device=device, compute_type_explicit=compute_type_explicit
        )

    if device in ("parakeet", "parakeet-cpu"):
        try:
            from .parakeet import ParakeetBackend
        except ImportError:
            raise ValueError(
                "Parakeet бэкенд недоступен. Установите: pip install onnx-asr[cpu,hub]"
            ) from None
        return ParakeetBackend(compute_type_explicit=compute_type_explicit)

    # cuda, cpu и всё остальное → faster-whisper
    from .faster_whisper import FasterWhisperBackend

    return FasterWhisperBackend()
```

- [ ] **Step 3: Sanity-импорт**

```bash
uv run python -c "from local_transcriber.backends import get_backend; b = get_backend('parakeet'); print(type(b).__name__, b.effective_model_name)"
```

Expected: `ParakeetBackend parakeet-tdt-0.6b-v3`.

- [ ] **Step 4: Прогнать существующие тесты**

```bash
uv run pytest -v
```

Expected: PASS (никакие существующие тесты не сломаны — мы только добавили новый код).

- [ ] **Step 5: Commit**

```bash
git add src/local_transcriber/backends/parakeet.py src/local_transcriber/backends/__init__.py
git commit -F- <<'EOF'
feat(backends): scaffold ParakeetBackend + регистрация

- Зачем:
  - подготовить каркас бэкенда с заглушками, чтобы следующие задачи реализовывали методы по TDD.
- Что:
  - новый файл backends/parakeet.py с классом ParakeetBackend (методы raise NotImplementedError).
  - регистрация в backends/__init__.py::get_backend для device=parakeet/parakeet-cpu.
- Проверка:
  - uv run python -c "from local_transcriber.backends import get_backend; print(get_backend('parakeet'))".
  - uv run pytest -v.
EOF
```

---

## Task 4: Реализовать `ensure_model_available` (TDD)

**Files:**
- Modify: `src/local_transcriber/backends/parakeet.py`
- Create: `tests/test_backend_parakeet.py`

- [ ] **Step 1: Создать `tests/test_backend_parakeet.py` с падающими тестами для валидации**

```python
"""Тесты для Parakeet бэкенда."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from local_transcriber.backends.parakeet import (
    MODEL_REPO,
    ONNX_ASR_MODEL_ALIAS,
    ParakeetBackend,
    SUPPORTED_COMPUTE_TYPES,
    SUPPORTED_MODEL_NAMES,
)


def _create_parakeet_model_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    return path


# === ensure_model_available: валидация входов ===


def test_ensure_model_available_accepts_canonical_name(tmp_path):
    backend = ParakeetBackend()
    model_dir = _create_parakeet_model_dir(tmp_path / "parakeet")
    with patch(
        "local_transcriber.backends.parakeet.snapshot_download",
        return_value=str(model_dir),
    ) as mock_dl:
        result = backend.ensure_model_available("parakeet-tdt-0.6b-v3", "int8")
    mock_dl.assert_called_once()
    args, kwargs = mock_dl.call_args
    assert (args and args[0] == MODEL_REPO) or kwargs.get("repo_id") == MODEL_REPO
    assert result == str(model_dir)


def test_ensure_model_available_accepts_alias(tmp_path):
    backend = ParakeetBackend()
    model_dir = _create_parakeet_model_dir(tmp_path / "parakeet")
    with patch(
        "local_transcriber.backends.parakeet.snapshot_download",
        return_value=str(model_dir),
    ) as mock_dl:
        result = backend.ensure_model_available("parakeet", "int8")
    args, kwargs = mock_dl.call_args
    assert (args and args[0] == MODEL_REPO) or kwargs.get("repo_id") == MODEL_REPO
    assert result == str(model_dir)


def test_ensure_model_available_rejects_unknown_model():
    backend = ParakeetBackend()
    with pytest.raises(ValueError, match="parakeet-tdt-0.6b-v3"):
        backend.ensure_model_available("medium", "int8")


def test_ensure_model_available_rejects_unknown_compute_type():
    backend = ParakeetBackend()
    with pytest.raises(ValueError, match="int8 или float32"):
        backend.ensure_model_available("parakeet-tdt-0.6b-v3", "float16")


def test_ensure_model_available_sets_actual_compute_type(tmp_path):
    backend = ParakeetBackend()
    model_dir = _create_parakeet_model_dir(tmp_path / "parakeet")
    with patch(
        "local_transcriber.backends.parakeet.snapshot_download",
        return_value=str(model_dir),
    ):
        backend.ensure_model_available("parakeet-tdt-0.6b-v3", "int8")
    assert backend.actual_compute_type == "int8"


def test_ensure_model_available_validates_model_dir(tmp_path):
    """Если snapshot_download вернул путь без config.json → ValueError."""
    backend = ParakeetBackend()
    bad_dir = tmp_path / "broken"
    bad_dir.mkdir()
    with patch(
        "local_transcriber.backends.parakeet.snapshot_download",
        return_value=str(bad_dir),
    ):
        with pytest.raises(ValueError, match="Неполная"):
            backend.ensure_model_available("parakeet-tdt-0.6b-v3", "int8")
```

- [ ] **Step 2: Запустить тесты — ожидаем FAIL**

```bash
uv run pytest tests/test_backend_parakeet.py -v
```

Expected: FAIL — `ensure_model_available` бросает `NotImplementedError` (либо `ModuleNotFoundError` на `snapshot_download`).

- [ ] **Step 3a: Добавить импорт `snapshot_download` и `LocalEntryNotFoundError`**

В `src/local_transcriber/backends/parakeet.py` вверху файла (после `from local_transcriber.types import ...`):

```python
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
```

- [ ] **Step 3b: Добавить helper `_validate_model_dir` в конце файла (после `_notify`)**

```python
def _validate_model_dir(model_dir: Path) -> None:
    missing = [f for f in MODEL_REQUIRED_FILES if not (model_dir / f).exists()]
    if missing:
        raise ValueError(
            f"Неполная Parakeet-модель в '{model_dir}': отсутствуют {', '.join(missing)}"
        )
```

- [ ] **Step 3c: Заменить тело метода `ensure_model_available`**

В классе `ParakeetBackend` тело метода `ensure_model_available` (где `raise NotImplementedError`) заменить на:

```python
    def ensure_model_available(
        self,
        model_name: str,
        compute_type: str,
        on_status: Callable[[str], None] | None = None,
    ) -> str:
        if model_name not in SUPPORTED_MODEL_NAMES:
            allowed = ", ".join(sorted(SUPPORTED_MODEL_NAMES))
            raise ValueError(
                f"Parakeet поддерживает только модели: {allowed}. "
                f"Получено: '{model_name}'. Передайте --model parakeet или уберите "
                f"'model' из .transcriber.toml."
            )
        if compute_type not in SUPPORTED_COMPUTE_TYPES:
            allowed = ", ".join(sorted(SUPPORTED_COMPUTE_TYPES))
            raise ValueError(
                f"Parakeet поддерживает только compute_type: {allowed}. "
                f"Получено: '{compute_type}' (варианты для Parakeet — int8 или float32)."
            )

        self.actual_compute_type = compute_type

        # Cache-first: сначала проверяем локальный кэш (как faster_whisper / openvino бэкенды)
        try:
            _notify(on_status, f"Проверяю кэш модели Parakeet ({MODEL_REPO})...")
            cached_path = Path(snapshot_download(MODEL_REPO, local_files_only=True))
            _validate_model_dir(cached_path)
            return str(cached_path)
        except LocalEntryNotFoundError:
            pass
        except ValueError:
            _notify(on_status, f"Кэш Parakeet неполный, докачиваю...")

        _notify(on_status, f"Скачиваю модель Parakeet ({MODEL_REPO}) из Hugging Face...")
        model_dir = Path(snapshot_download(MODEL_REPO, local_files_only=False))
        _validate_model_dir(model_dir)
        return str(model_dir)
```

- [ ] **Step 4: Запустить тесты — ожидаем PASS**

```bash
uv run pytest tests/test_backend_parakeet.py -v
```

Expected: все 6 тестов PASS.

- [ ] **Step 5: Прогнать полный test suite**

```bash
uv run pytest -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/local_transcriber/backends/parakeet.py tests/test_backend_parakeet.py
git commit -F- <<'EOF'
feat(backends): ensure_model_available для Parakeet

- Зачем:
  - гарантировать наличие Parakeet модели в локальном кэше HF перед create_model; явная ошибка при неверных model/compute_type (важно для UX при конфликте с .transcriber.toml).
- Что:
  - валидация model_name ∈ {parakeet-tdt-0.6b-v3, parakeet} и compute_type ∈ {int8, float32}.
  - snapshot_download("nvidia/parakeet-tdt-0.6b-v3") + проверка config.json.
  - установка actual_compute_type.
  - 6 юнит-тестов в test_backend_parakeet.py.
- Проверка:
  - uv run pytest tests/test_backend_parakeet.py -v.
EOF
```

---

## Task 5: Реализовать `create_model` (TDD)

**Files:**
- Modify: `src/local_transcriber/backends/parakeet.py`
- Modify: `tests/test_backend_parakeet.py`

- [ ] **Step 1: Добавить падающие тесты в `tests/test_backend_parakeet.py`**

Добавить в конец файла:

```python
# === create_model ===


def test_create_model_maps_int8_to_quantization(tmp_path):
    backend = ParakeetBackend()
    model_dir = _create_parakeet_model_dir(tmp_path / "parakeet")
    fake_asr = MagicMock(name="asr")
    fake_asr.with_vad.return_value = MagicMock(name="asr_with_vad")

    with (
        patch("local_transcriber.backends.parakeet.onnx_asr") as mock_ox,
    ):
        mock_ox.load_model.return_value = fake_asr
        mock_ox.load_vad.return_value = MagicMock(name="vad")

        backend.create_model(str(model_dir), "parakeet-cpu", "int8")

    load_model_kwargs = mock_ox.load_model.call_args.kwargs
    assert load_model_kwargs.get("quantization") == "int8"
    # Проверка алиаса модели и локального пути
    assert mock_ox.load_model.call_args.args[0] == ONNX_ASR_MODEL_ALIAS
    assert load_model_kwargs.get("path") == str(model_dir)


def test_create_model_maps_float32_to_none_quantization(tmp_path):
    backend = ParakeetBackend()
    model_dir = _create_parakeet_model_dir(tmp_path / "parakeet")
    fake_asr = MagicMock()
    fake_asr.with_vad.return_value = MagicMock()

    with patch("local_transcriber.backends.parakeet.onnx_asr") as mock_ox:
        mock_ox.load_model.return_value = fake_asr
        mock_ox.load_vad.return_value = MagicMock()

        backend.create_model(str(model_dir), "parakeet-cpu", "float32")

    assert mock_ox.load_model.call_args.kwargs.get("quantization") is None


def test_create_model_wraps_with_silero_vad(tmp_path):
    backend = ParakeetBackend()
    model_dir = _create_parakeet_model_dir(tmp_path / "parakeet")
    fake_asr = MagicMock()
    wrapped = MagicMock(name="asr_with_vad")
    fake_asr.with_vad.return_value = wrapped

    with patch("local_transcriber.backends.parakeet.onnx_asr") as mock_ox:
        mock_ox.load_model.return_value = fake_asr
        fake_vad = MagicMock(name="silero_vad")
        mock_ox.load_vad.return_value = fake_vad

        returned = backend.create_model(str(model_dir), "parakeet-cpu", "int8")

    mock_ox.load_vad.assert_called_once()
    assert mock_ox.load_vad.call_args.kwargs.get("model") == "silero" \
        or mock_ox.load_vad.call_args.args[0] == "silero"
    fake_asr.with_vad.assert_called_once_with(vad=fake_vad)
    assert returned is wrapped


def test_create_model_cpu_threads_via_sess_options(tmp_path):
    """cpu_threads > 0 → SessionOptions.intra_op_num_threads, НЕ provider_options."""
    backend = ParakeetBackend()
    model_dir = _create_parakeet_model_dir(tmp_path / "parakeet")
    fake_asr = MagicMock()
    fake_asr.with_vad.return_value = MagicMock()

    with patch("local_transcriber.backends.parakeet.onnx_asr") as mock_ox:
        mock_ox.load_model.return_value = fake_asr
        mock_ox.load_vad.return_value = MagicMock()

        backend.create_model(str(model_dir), "parakeet-cpu", "int8", cpu_threads=4)

    kwargs = mock_ox.load_model.call_args.kwargs
    assert "sess_options" in kwargs
    assert kwargs["sess_options"].intra_op_num_threads == 4
    assert "provider_options" not in kwargs


def test_create_model_zero_threads_omits_sess_options(tmp_path):
    backend = ParakeetBackend()
    model_dir = _create_parakeet_model_dir(tmp_path / "parakeet")
    fake_asr = MagicMock()
    fake_asr.with_vad.return_value = MagicMock()

    with patch("local_transcriber.backends.parakeet.onnx_asr") as mock_ox:
        mock_ox.load_model.return_value = fake_asr
        mock_ox.load_vad.return_value = MagicMock()

        backend.create_model(str(model_dir), "parakeet-cpu", "int8", cpu_threads=0)

    kwargs = mock_ox.load_model.call_args.kwargs
    assert "sess_options" not in kwargs  # 0 = дефолт библиотеки, не трогаем
```

- [ ] **Step 2: Запустить тесты — ожидаем FAIL**

```bash
uv run pytest tests/test_backend_parakeet.py -v -k "create_model"
```

Expected: FAIL — `create_model` бросает `NotImplementedError`.

- [ ] **Step 3: Реализовать `create_model`**

В `src/local_transcriber/backends/parakeet.py`:

1. Добавить вверху файла:

```python
import onnx_asr  # noqa: E402 — тяжёлый импорт; ленивая загрузка обеспечивается ленивым get_backend
```

(Важно: размещаем **после** `from local_transcriber.types import ...` — стандартный порядок, и это всё равно уже ленивый импорт на уровне `get_backend`.)

2. Заменить метод:

```python
def create_model(
    self,
    model_path: str,
    device: str,
    compute_type: str,
    cpu_threads: int = 0,
) -> Any:
    """Создаёт asr-pipeline с Silero VAD.

    cpu_threads: если > 0, передаётся через onnxruntime.SessionOptions.intra_op_num_threads.
    VAD загружается здесь (runtime-зависимость; первый запуск требует интернет).
    """
    quantization = "int8" if compute_type == "int8" else None

    load_kwargs: dict[str, Any] = {"path": model_path, "quantization": quantization}
    if cpu_threads and cpu_threads > 0:
        # intra_op_num_threads — атрибут SessionOptions, а НЕ provider_options.
        # provider_options ожидает EP-специфичные ключи (device_id и т.п.).
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = cpu_threads
        load_kwargs["sess_options"] = sess_options

    asr = onnx_asr.load_model(ONNX_ASR_MODEL_ALIAS, **load_kwargs)
    vad = onnx_asr.load_vad(model="silero")
    return asr.with_vad(vad=vad)
```

- [ ] **Step 4: Запустить тесты — ожидаем PASS**

```bash
uv run pytest tests/test_backend_parakeet.py -v
```

Expected: все тесты (включая 3 новых) PASS.

- [ ] **Step 5: Commit**

```bash
git add src/local_transcriber/backends/parakeet.py tests/test_backend_parakeet.py
git commit -F- <<'EOF'
feat(backends): create_model для Parakeet с Silero VAD

- Зачем:
  - инициализация onnx-asr pipeline Parakeet v3, обёрнутого Silero VAD; без VAD лимит 20-30 сек на файл.
- Что:
  - onnx_asr.load_model(path=..., quantization="int8"|None) + load_vad(silero) + with_vad.
  - cpu_threads → provider_options.intra_op_num_threads для ONNX Runtime.
  - 3 новых теста на маппинг compute_type и обёртку VAD.
- Проверка:
  - uv run pytest tests/test_backend_parakeet.py -v.
EOF
```

---

## Task 6: Реализовать `transcribe` (TDD)

**Files:**
- Modify: `src/local_transcriber/backends/parakeet.py`
- Modify: `tests/test_backend_parakeet.py`

- [ ] **Step 1: Добавить падающие тесты в `tests/test_backend_parakeet.py`**

```python
# === transcribe ===


def _segment_result(start, end, text):
    """Фабрика onnx-asr SegmentResult-подобного объекта."""
    seg = MagicMock()
    seg.start = start
    seg.end = end
    seg.text = text
    return seg


def test_transcribe_produces_segments():
    backend = ParakeetBackend()
    model = MagicMock()
    model.recognize.return_value = [
        _segment_result(0.0, 1.5, " Привет"),
        _segment_result(1.5, 3.0, " мир"),
        _segment_result(3.0, 5.0, " это тест"),
    ]

    result = backend.transcribe(model, Path("test.mp3"), language=None)

    assert len(result.segments) == 3
    assert result.segments[0].start == 0.0
    assert result.segments[0].text == " Привет"
    assert result.language == "multi"
    assert result.language_probability == 0.0


def test_transcribe_accepts_iterator_results():
    backend = ParakeetBackend()
    model = MagicMock()

    def gen():
        yield _segment_result(0.0, 1.0, "a")
        yield _segment_result(1.0, 2.0, "b")

    model.recognize.return_value = gen()

    result = backend.transcribe(model, Path("test.mp3"), language=None)
    assert len(result.segments) == 2


def test_transcribe_calls_on_segment():
    backend = ParakeetBackend()
    model = MagicMock()
    model.recognize.return_value = [
        _segment_result(0.0, 1.0, "one"),
        _segment_result(1.0, 2.0, "two"),
    ]
    callback = MagicMock()

    backend.transcribe(model, Path("test.mp3"), language=None, on_segment=callback)

    assert callback.call_count == 2


def test_transcribe_warns_on_explicit_language():
    backend = ParakeetBackend()
    model = MagicMock()
    model.recognize.return_value = []

    with pytest.warns(UserWarning, match="Parakeet игнорирует"):
        backend.transcribe(model, Path("test.mp3"), language="ru")


def test_transcribe_does_not_warn_on_none_language():
    backend = ParakeetBackend()
    model = MagicMock()
    model.recognize.return_value = []

    import warnings as wmod

    with wmod.catch_warnings(record=True) as caught:
        wmod.simplefilter("always")
        backend.transcribe(model, Path("test.mp3"), language=None)

    assert not any("Parakeet" in str(w.message) for w in caught)


def test_transcribe_computes_duration_from_segments():
    backend = ParakeetBackend()
    model = MagicMock()
    model.recognize.return_value = [
        _segment_result(0.0, 10.0, "a"),
        _segment_result(10.5, 25.0, "b"),
    ]

    result = backend.transcribe(model, Path("test.mp3"), language=None)
    assert result.duration == 25.0  # max(end)


def test_transcribe_empty_segments_returns_zero_duration():
    backend = ParakeetBackend()
    model = MagicMock()
    model.recognize.return_value = []

    result = backend.transcribe(model, Path("test.mp3"), language=None)
    assert result.duration == 0.0
    assert result.segments == []
    assert result.language == "multi"


def test_transcribe_calls_on_status():
    backend = ParakeetBackend()
    model = MagicMock()
    model.recognize.return_value = [
        _segment_result(0.0, 1.0, "one"),
    ]
    status_cb = MagicMock()

    backend.transcribe(model, Path("test.mp3"), language=None, on_status=status_cb)

    assert status_cb.call_count >= 1  # хотя бы одно status-сообщение
```

- [ ] **Step 2: Запустить тесты — ожидаем FAIL**

```bash
uv run pytest tests/test_backend_parakeet.py -v -k "transcribe"
```

Expected: FAIL — `transcribe` бросает `NotImplementedError`.

- [ ] **Step 3: Реализовать `transcribe`**

В `src/local_transcriber/backends/parakeet.py` заменить метод:

```python
def transcribe(
    self,
    model: Any,
    file_path: Path,
    language: str | None,
    on_segment: Callable[[Segment], None] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> TranscribeResult:
    """Транскрибирует файл через onnx-asr + Silero VAD.

    language=None/"auto" — ок; любое другое значение → warning (Parakeet игнорирует язык).
    Возвращает TranscribeResult с language="multi" (независимо от пользовательского hint).
    """
    if language is not None and language != "auto":
        warnings.warn(
            "Parakeet игнорирует --language; язык определяется автоматически",
            UserWarning,
            stacklevel=2,
        )

    _notify(on_status, "Транскрибирую (Parakeet + Silero VAD)...")

    raw_segments = model.recognize(str(file_path))

    segments: list[Segment] = []
    max_end = 0.0
    for raw in raw_segments:
        start = float(getattr(raw, "start", 0.0) or 0.0)
        end = float(getattr(raw, "end", start) or start)
        text = getattr(raw, "text", "") or ""
        seg = Segment(start=start, end=end, text=text)
        if on_segment is not None:
            on_segment(seg)
        segments.append(seg)
        if end > max_end:
            max_end = end
        _notify(on_status, f"Parakeet: {len(segments)} сегм. ({end:.1f}с)")

    # duration = конец последнего речевого сегмента после VAD, НЕ точная длина аудио-файла.
    # Для заголовка транскрипта и форматирования таймкодов этого достаточно: расхождение —
    # только на длину трейлинг-тишины. Декодировать аудио отдельно ради точного значения
    # (как это делает OpenVINO-бэкенд через decode_audio) не стоит: ONNX-порт сам грузит
    # файл внутри recognize(), повторное декодирование на 3-часовых файлах — заметная цена.
    return TranscribeResult(
        segments=segments,
        language="multi",
        language_probability=0.0,
        duration=max_end,
        device_used="",  # оркестратор проставит
    )
```

- [ ] **Step 4: Запустить тесты — ожидаем PASS**

```bash
uv run pytest tests/test_backend_parakeet.py -v
```

Expected: все тесты (всего ~17) PASS.

- [ ] **Step 5: Прогнать полный test suite**

```bash
uv run pytest -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/local_transcriber/backends/parakeet.py tests/test_backend_parakeet.py
git commit -F- <<'EOF'
feat(backends): transcribe для Parakeet

- Зачем:
  - конвертация результатов onnx-asr recognize() в общий TranscribeResult (Segment + language="multi").
- Что:
  - поддержка iterator и list возврата из recognize.
  - warning при явном language != None/"auto".
  - duration = max(segment.end).
  - on_segment/on_status вызовы.
  - 8 новых тестов (produces_segments, iterator, on_segment, warnings, duration, empty, on_status).
- Проверка:
  - uv run pytest tests/test_backend_parakeet.py -v.
EOF
```

---

## Task 7: No-fallback политика для Parakeet в `transcriber.py`

**Files:**
- Modify: `src/local_transcriber/transcriber.py`
- Modify: `tests/test_transcriber.py`

- [ ] **Step 1: Добавить падающий тест в `tests/test_transcriber.py`**

В конец файла:

```python
# === Parakeet no-fallback policy ===


@patch("local_transcriber.transcriber.get_backend")
def test_load_model_parakeet_no_fallback_on_create_error(mock_get_backend):
    """При ошибке Parakeet во время create_model исключение пробрасывается,
    без silent fallback на CPU FasterWhisper."""
    parakeet_backend = _make_backend(create_model_error=RuntimeError("onnxruntime crashed"))
    mock_get_backend.return_value = parakeet_backend

    with pytest.raises(RuntimeError, match="onnxruntime"):
        load_model(
            model_name="parakeet-tdt-0.6b-v3",
            device="parakeet-cpu",
            compute_type="int8",
        )

    # get_backend вызван только один раз (НЕ было fallback-вызова для "cpu")
    assert mock_get_backend.call_count == 1


@patch("local_transcriber.transcriber.get_backend")
def test_transcribe_file_parakeet_no_fallback_on_runtime_error(mock_get_backend):
    """Ошибка Parakeet при transcribe не переключает на CPU FasterWhisper."""
    parakeet_backend = _make_backend(transcribe_error=RuntimeError("onnxruntime inference error"))
    mock_get_backend.return_value = parakeet_backend

    with pytest.raises(RuntimeError, match="onnxruntime"):
        transcribe(
            file_path=Path("test.mp3"),
            model_name="parakeet-tdt-0.6b-v3",
            device="parakeet-cpu",
            compute_type="int8",
        )

    # Только один бэкенд создан — Parakeet; без fallback на cpu
    assert mock_get_backend.call_count == 1
```

- [ ] **Step 2: Запустить тесты — ожидаем FAIL**

```bash
uv run pytest tests/test_transcriber.py::test_load_model_parakeet_no_fallback_on_create_error tests/test_transcriber.py::test_transcribe_file_parakeet_no_fallback_on_runtime_error -v
```

Expected: FAIL — текущий код делает fallback (`_is_backend_error` возвращает True для CUDA/OpenVINO; для Parakeet `_is_backend_error` возвращает False → исключение **прокидывается**, но `get_backend` всё равно может быть вызван дважды в load_model, смотрим реальный output).

Если тест #2 случайно проходит — всё равно продолжаем реализацию: мы хотим **явный** пропуск fallback.

- [ ] **Step 3: Реализовать `_is_parakeet_error` и добавить ветку no-fallback**

В `src/local_transcriber/transcriber.py`:

1. Добавить функцию после `_is_openvino_error`:

```python
def _is_parakeet_error(exc: BaseException) -> bool:
    """Проверка ошибок Parakeet runtime (onnx-asr / onnxruntime).

    Для RuntimeError считаем backend failure (аналогично OpenVINO).
    Дополнительно проверяем module name — чтобы явно поймать onnxruntime-исключения.
    """
    if isinstance(exc, RuntimeError):
        return True
    mod = type(exc).__module__ or ""
    return "onnxruntime" in mod or "onnx_asr" in mod
```

2. Обновить `_is_backend_error`:

```python
def _is_backend_error(exc: BaseException, device: str) -> bool:
    """Определяет, связана ли ошибка с конкретным бэкендом (а не с пользовательскими данными)."""
    if device in ("cuda", "cpu"):
        return _is_cuda_error(exc)
    if device.startswith("openvino"):
        return _is_openvino_error(exc)
    if device.startswith("parakeet"):
        return _is_parakeet_error(exc)
    return False
```

3. В `load_model` — принудительно пробрасывать ошибку для Parakeet, без fallback-ветки. Найти блок `except (RuntimeError, ValueError) as exc:` и заменить условие:

```python
    except (RuntimeError, ValueError) as exc:
        if device.startswith("parakeet"):
            # Parakeet — экспериментальный бэкенд, silent fallback на Whisper меняет
            # семантику (модель/язык) — пробрасываем ошибку наружу.
            raise
        if device != "cpu" and _is_backend_error(exc, device):
            if strict_device:
                raise
            warnings.warn(
                f"Не удалось загрузить модель на {device}: {exc}. "
                "Переключение на CPU.",
                stacklevel=2,
            )
            actual_device = "cpu"
            backend = get_backend("cpu")
            model_path = backend.ensure_model_available(model_name, compute_type, on_status)
            _notify_status(on_status, "Инициализирую модель на cpu...")
            model = backend.create_model(model_path, "cpu", compute_type, cpu_threads=cpu_threads)
        else:
            raise
```

4. В `_transcribe_file` аналогично — в блоке `except (RuntimeError, ValueError) as exc:`:

```python
    except (RuntimeError, ValueError) as exc:
        if actual_device.startswith("parakeet"):
            raise
        if actual_device != "cpu" and _is_backend_error(exc, actual_device):
            if strict_device:
                raise
            # ... existing fallback body
        else:
            raise
```

- [ ] **Step 4: Запустить тесты — ожидаем PASS**

```bash
uv run pytest tests/test_transcriber.py -v
```

Expected: все тесты PASS (новые два + регрессия по существующим).

- [ ] **Step 5: Commit**

```bash
git add src/local_transcriber/transcriber.py tests/test_transcriber.py
git commit -F- <<'EOF'
feat(transcriber): no-fallback политика для Parakeet

- Зачем:
  - Parakeet — другое семейство моделей (multi-language vs ru Whisper); silent fallback на CPU Whisper меняет качество/семантику и вводит в заблуждение.
- Что:
  - _is_parakeet_error: RuntimeError или исключение из onnxruntime/onnx_asr модулей.
  - _is_backend_error возвращает этот флаг для device.startswith("parakeet").
  - load_model и _transcribe_file пробрасывают ошибку наружу при device.startswith("parakeet"), минуя cross-backend fallback.
  - 2 новых теста.
- Проверка:
  - uv run pytest tests/test_transcriber.py -v.
EOF
```

---

## Task 8: CLI — effective_model_name, force language_mode, warning, device_info

**Files:**
- Modify: `src/local_transcriber/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Добавить падающие CLI-тесты в `tests/test_cli.py`**

В конец файла:

```python
# === Regression: Whisper headers not changed ===


def test_cli_whisper_transcript_header_has_language_forced(tmp_path):
    """Регрессия: для --device cpu --language ru шапка содержит 'ru (forced)'."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")

    captured = {}
    def capture(content, path):
        captured["content"] = content

    result = _make_result(language="ru", device_used="cpu")
    model = _make_model()
    backend = _make_backend()
    backend.effective_model_name = None  # Whisper не ставит это
    tfr = _make_tfr(result=result, model=model, actual_device="cpu", backend=backend)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu", backend, "/models/medium")),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript", side_effect=capture),
    ):
        out = runner.invoke(app, [str(audio), "--language", "ru"])

    assert out.exit_code == 0
    assert "**Язык**: ru (forced)" in captured["content"]
    assert "**Модель**: medium" in captured["content"]


def test_cli_whisper_auto_language_header_has_detected(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    captured = {}
    def capture(content, path):
        captured["content"] = content

    result = _make_result(language="en", device_used="cpu")
    model = _make_model()
    backend = _make_backend()
    backend.effective_model_name = None
    tfr = _make_tfr(result=result, model=model, actual_device="cpu", backend=backend)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="cpu"),
        patch("local_transcriber.cli.load_model", return_value=(model, "cpu", backend, "/models/medium")),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript", side_effect=capture),
    ):
        out = runner.invoke(app, [str(audio), "--language", "auto"])

    assert out.exit_code == 0
    assert "**Язык**: en (detected)" in captured["content"]


# === Parakeet-specific CLI ===


def _make_parakeet_backend(effective_name="parakeet-tdt-0.6b-v3"):
    backend = MagicMock(name="ParakeetBackend")
    backend.effective_model_name = effective_name
    backend.actual_compute_type = "int8"
    return backend


def test_cli_parakeet_transcript_header_multi_detected(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    captured = {}
    def capture(content, path):
        captured["content"] = content

    result = _make_result(language="multi", device_used="parakeet-cpu")
    model = _make_model()
    backend = _make_parakeet_backend()
    tfr = _make_tfr(result=result, model=model, actual_device="parakeet-cpu", backend=backend)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="parakeet-cpu"),
        patch("local_transcriber.cli.load_model", return_value=(model, "parakeet-cpu", backend, "/models/parakeet")),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript", side_effect=capture),
    ):
        out = runner.invoke(app, [str(audio), "--device", "parakeet"])

    assert out.exit_code == 0
    assert "**Язык**: multi (detected)" in captured["content"]
    assert "**Модель**: parakeet-tdt-0.6b-v3" in captured["content"]


def test_cli_parakeet_effective_model_overrides_config_model(tmp_path):
    """Когда config имеет model=medium, но user'ская команда — --device parakeet --model parakeet,
    шапка транскрипта должна показать parakeet-tdt-0.6b-v3."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    captured = {}
    def capture(content, path):
        captured["content"] = content

    result = _make_result(language="multi", device_used="parakeet-cpu")
    model = _make_model()
    backend = _make_parakeet_backend()
    tfr = _make_tfr(result=result, model=model, actual_device="parakeet-cpu", backend=backend)

    with (
        patch("local_transcriber.cli.load_config", return_value={"model": "medium"}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="parakeet-cpu"),
        patch("local_transcriber.cli.load_model", return_value=(model, "parakeet-cpu", backend, "/models/parakeet")),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript", side_effect=capture),
    ):
        out = runner.invoke(app, [str(audio), "--device", "parakeet", "--model", "parakeet"])

    assert out.exit_code == 0
    assert "**Модель**: parakeet-tdt-0.6b-v3" in captured["content"]


def test_cli_parakeet_warns_on_explicit_cli_language(tmp_path):
    """Явный --language с --device parakeet → warning в stderr."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")

    result = _make_result(language="multi", device_used="parakeet-cpu")
    model = _make_model()
    backend = _make_parakeet_backend()
    tfr = _make_tfr(result=result, model=model, actual_device="parakeet-cpu", backend=backend)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="parakeet-cpu"),
        patch("local_transcriber.cli.load_model", return_value=(model, "parakeet-cpu", backend, "/models/parakeet")),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(audio), "--device", "parakeet", "--language", "en"])

    assert out.exit_code == 0
    # Warning печатается через Rich в stderr
    assert "Parakeet игнорирует" in out.stderr or "Parakeet игнорирует" in out.output


def test_cli_parakeet_does_not_warn_on_config_language(tmp_path):
    """Language из конфига без явного CLI — warning НЕ срабатывает."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")

    result = _make_result(language="multi", device_used="parakeet-cpu")
    model = _make_model()
    backend = _make_parakeet_backend()
    tfr = _make_tfr(result=result, model=model, actual_device="parakeet-cpu", backend=backend)

    with (
        patch("local_transcriber.cli.load_config", return_value={"language": "ru"}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="parakeet-cpu"),
        patch("local_transcriber.cli.load_model", return_value=(model, "parakeet-cpu", backend, "/models/parakeet")),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
        patch("local_transcriber.cli.write_transcript"),
    ):
        out = runner.invoke(app, [str(audio), "--device", "parakeet"])

    assert out.exit_code == 0
    assert "Parakeet игнорирует" not in out.stderr and "Parakeet игнорирует" not in out.output


def test_cli_parakeet_device_info_shows_onnx():
    assert _format_device_info("parakeet-cpu") == "Parakeet (CPU via ONNX Runtime)"


def test_cli_parakeet_config_model_mismatch_without_override_prints_friendly_error(tmp_path):
    """config: model=medium, запуск --device parakeet БЕЗ --model → ValueError из
    ensure_model_available пробрасывается; CLI должен напечатать текст ошибки
    без traceback (exit code 1)."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")

    # load_model вызывает backend.ensure_model_available — имитируем реальную цепочку:
    # с config={"model": "medium"} defaults["model"] = "medium", backend кинет ValueError.
    def failing_load_model(*args, **kwargs):
        raise ValueError(
            "Parakeet поддерживает только модели: parakeet, parakeet-tdt-0.6b-v3. "
            "Получено: 'medium'. Передайте --model parakeet или уберите "
            "'model' из .transcriber.toml."
        )

    with (
        patch("local_transcriber.cli.load_config", return_value={"model": "medium"}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="parakeet-cpu"),
        patch("local_transcriber.cli.load_model", side_effect=failing_load_model),
    ):
        out = runner.invoke(app, [str(audio), "--device", "parakeet"])

    assert out.exit_code == 1
    # Сообщение ошибки (и/или его часть) видно пользователю
    combined = (out.stderr or "") + (out.output or "")
    assert "Parakeet поддерживает только" in combined
    assert "--model parakeet" in combined
    # Traceback НЕ показан (без --verbose)
    assert "Traceback" not in combined
```

- [ ] **Step 2: Запустить тесты — ожидаем FAIL**

```bash
uv run pytest tests/test_cli.py -v -k "parakeet or whisper_transcript_header or whisper_auto_language"
```

Expected: FAIL на новых тестах.

- [ ] **Step 3: Обновить `cli.py::_format_device_info`**

В `src/local_transcriber/cli.py`:

```python
def _format_device_info(device_used: str) -> str:
    """Формирует строку устройства для шапки транскрипта."""
    if device_used == "cuda":
        gpu_name = get_gpu_name()
        return f"CUDA ({gpu_name or 'Unknown GPU'})"
    if device_used == "openvino-gpu":
        gpu_name = get_intel_gpu_name()
        return f"OpenVINO ({gpu_name or 'Intel GPU'})"
    if device_used in ("openvino", "openvino-cpu"):
        return "OpenVINO (CPU)"
    if device_used == "parakeet-cpu":
        return "Parakeet (CPU via ONNX Runtime)"
    return "CPU"
```

- [ ] **Step 4: Пробросить `language_explicit` + warning в `main()`**

В `src/local_transcriber/cli.py::main` найти строку:

```python
ct_explicit = compute_type is not None or "compute_type" in config
```

И заменить на:

```python
ct_explicit = compute_type is not None or "compute_type" in config
language_explicit_cli = language is not None  # только CLI — warning для Parakeet
```

Обновить вызовы:

```python
if is_batch:
    _run_batch(expanded, defaults, verbose, force, ct_explicit, cpu_threads=threads,
               language_explicit_cli=language_explicit_cli)
else:
    _run_single(expanded[0], defaults, output, verbose, ct_explicit, cpu_threads=threads,
                language_explicit_cli=language_explicit_cli)
```

- [ ] **Step 5: Обновить сигнатуры и тела `_run_single` / `_run_batch`**

Добавить параметр `language_explicit_cli: bool = False` в обе функции. В каждой, **перед** вызовом `load_model`, добавить блок warning и determine language_mode:

```python
# В _run_single, после resolved_device = detect_device(requested_device):
is_parakeet = resolved_device.startswith("parakeet")
if is_parakeet and language_explicit_cli:
    console.print(
        "Parakeet игнорирует --language; язык определяется автоматически",
        style="yellow",
    )
```

Найти строку:

```python
actual_ct = getattr(backend, "actual_compute_type", defaults["compute_type"]) or defaults["compute_type"]
console.print(
    f"Модель: [bold]{defaults['model']}[/bold]  "
    f"Устройство: [bold]{actual_device}[/bold]  "
    f"Compute: [bold]{actual_ct}[/bold]"
)
```

Заменить на:

```python
actual_ct = getattr(backend, "actual_compute_type", defaults["compute_type"]) or defaults["compute_type"]
effective_model = getattr(backend, "effective_model_name", None) or defaults["model"]
console.print(
    f"Модель: [bold]{effective_model}[/bold]  "
    f"Устройство: [bold]{actual_device}[/bold]  "
    f"Compute: [bold]{actual_ct}[/bold]"
)
```

Найти блок определения `language_mode`:

```python
device_info = _format_device_info(result.device_used)
language_mode = "detected" if defaults["language"] == "auto" else "forced"
```

Заменить на:

```python
device_info = _format_device_info(result.device_used)
if tfr.actual_device.startswith("parakeet"):
    language_mode = "detected"  # Parakeet всегда auto-detect (multi-language)
else:
    language_mode = "detected" if defaults["language"] == "auto" else "forced"
```

Найти вызов `format_transcript`:

```python
content = format_transcript(
    result=result,
    source_filename=validated_file.name,
    model_name=defaults["model"],
    device_info=device_info,
    language_mode=language_mode,
)
```

Заменить `model_name=defaults["model"]` на `model_name=effective_model`.

Для `_run_batch` (структура другая — model loaded once before per-file loop) — конкретные правки:

1. **Добавить параметр** `language_explicit_cli: bool = False` в сигнатуру `_run_batch`.

2. **Warning после `resolved_device = detect_device(requested_device)`** (перед `strict = requested_device != "auto"`):

```python
is_parakeet = resolved_device.startswith("parakeet")
if is_parakeet and language_explicit_cli:
    console.print(
        "Parakeet игнорирует --language; язык определяется автоматически",
        style="yellow",
    )
```

3. **После `load_model(...)` (где получаем `backend, actual_device`)** — добавить:

```python
effective_model = getattr(backend, "effective_model_name", None) or defaults["model"]
```

4. **Заменить строку `language_mode = "detected" if defaults["language"] == "auto" else "forced"`** (перед `batch_start = time.monotonic()`) на:

```python
if actual_device.startswith("parakeet"):
    language_mode = "detected"  # Parakeet всегда auto-detect (multi-language)
else:
    language_mode = "detected" if defaults["language"] == "auto" else "forced"
```

**Важно:** `language_mode` считается **один раз** вне per-file цикла (Parakeet без fallback, `actual_device` не меняется между файлами).

5. **В вызове `format_transcript(...)` внутри цикла** заменить `model_name=defaults["model"]` на `model_name=effective_model`.

- [ ] **Step 6: Запустить тесты — ожидаем PASS**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: все тесты PASS (включая новые и regression).

- [ ] **Step 7: Прогнать полный test suite**

```bash
uv run pytest -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/local_transcriber/cli.py tests/test_cli.py
git commit -F- <<'EOF'
feat(cli): интеграция Parakeet бэкенда

- Зачем:
  - шапка транскрипта и rich-вывод должны отражать реальную модель (parakeet-tdt-0.6b-v3) и режим языка (multi detected), а не дефолты из конфига.
- Что:
  - _format_device_info для "parakeet-cpu" → "Parakeet (CPU via ONNX Runtime)".
  - effective_model = getattr(backend, "effective_model_name", ...) используется в rich-выводе и format_transcript.
  - language_mode принудительно "detected" при actual_device.startswith("parakeet").
  - warning "Parakeet игнорирует --language" при явном CLI --language.
  - 2 regression-теста Whisper + 5 Parakeet-тестов.
- Проверка:
  - uv run pytest tests/test_cli.py -v.
EOF
```

---

## Task 9: Smoke-тест end-to-end (mocks)

**Files:**
- Modify: `tests/test_cli.py` (один интеграционный тест с реальным write_transcript)

- [ ] **Step 1: Добавить тест в `tests/test_cli.py`**

```python
def test_cli_parakeet_end_to_end_writes_file(tmp_path):
    """Проверка end-to-end пайплайна с mock backend: файл записан и содержит ожидаемые поля."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")

    segments = [Segment(start=0.0, end=2.0, text="Привет мир")]
    result = TranscribeResult(
        segments=segments, language="multi", language_probability=0.0,
        duration=2.0, device_used="parakeet-cpu",
    )
    model = _make_model()
    backend = _make_parakeet_backend()
    tfr = _make_tfr(result=result, model=model, actual_device="parakeet-cpu", backend=backend)

    with (
        patch("local_transcriber.cli.load_config", return_value={}),
        patch("local_transcriber.cli.validate_input_file", return_value=audio),
        patch("local_transcriber.cli.detect_device", return_value="parakeet-cpu"),
        patch("local_transcriber.cli.load_model", return_value=(model, "parakeet-cpu", backend, "/models/parakeet")),
        patch("local_transcriber.cli._transcribe_file", return_value=tfr),
    ):
        out = runner.invoke(app, [str(audio), "--device", "parakeet"])

    assert out.exit_code == 0
    output_md = tmp_path / "test-transcript.md"
    assert output_md.exists()
    content = output_md.read_text(encoding="utf-8")
    assert "# Транскрипт: test.mp3" in content
    assert "**Модель**: parakeet-tdt-0.6b-v3" in content
    assert "**Язык**: multi (detected)" in content
    assert "Привет мир" in content
```

- [ ] **Step 2: Запустить тест — ожидаем PASS**

```bash
uv run pytest tests/test_cli.py::test_cli_parakeet_end_to_end_writes_file -v
```

Expected: PASS (все изменения Task 8 уже в силе).

- [ ] **Step 3: Прогнать полный test suite**

```bash
uv run pytest -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_cli.py
git commit -F- <<'EOF'
test(cli): end-to-end smoke для --device parakeet

- Зачем:
  - убедиться, что полный пайплайн CLI записывает корректный .md для Parakeet.
- Что:
  - тест проверяет наличие шапки, модели, языка, текста в выходном файле.
- Проверка:
  - uv run pytest tests/test_cli.py::test_cli_parakeet_end_to_end_writes_file -v.
EOF
```

---

## Task 10: Long-audio verification gate (ручной прогон, blocker)

**Files:**
- Create: `scripts/measure_parakeet.py` (скрипт-измеритель, коммитится — будет полезен для будущих замеров).
- Modify: `docs/gpu.md` (новая секция с числами).

Эта Task **не автоматизирована** — требует реального аудиофайла и локальной машины. Выполняется после Task 9, перед Task 11 (доки).

- [ ] **Step 1: Подготовить тестовый RU-файл ≥30 минут**

Например, лекция, подкаст или интервью. Файл `sample-long.mp3` (или любой формат). Положить путь в переменную окружения:

```bash
export TESTFILE="/абсолютный/путь/sample-long.mp3"
```

- [ ] **Step 2: Создать `scripts/measure_parakeet.py`**

Standalone-скрипт — использует `local_transcriber.transcribe()` как библиотеку и сам ведёт лог интервалов `on_status`. **Не трогает `cli.py`** — меньше риска оставить мусор в коде.

```python
"""Измерение производительности Parakeet на длинном файле.

Usage:
    uv run scripts/measure_parakeet.py <audio-file> [--compute-type int8|float32]

Логирует:
  - peak RSS (через psutil, опрашивает раз в секунду в отдельном потоке).
  - timestamp каждого on_status callback (первый update, интервалы между).
  - wall-clock и RTFx.
Результат печатается в stdout + записывается в /tmp/parakeet-measure.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import psutil

from local_transcriber.transcriber import transcribe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--model", default="parakeet-tdt-0.6b-v3")
    parser.add_argument("--output", type=Path, default=Path("/tmp/parakeet-measure.json"))
    args = parser.parse_args()

    # RSS sampler в отдельном потоке
    stop = threading.Event()
    peak_rss = [0]
    proc = psutil.Process()

    def sample_rss() -> None:
        while not stop.is_set():
            rss = proc.memory_info().rss
            if rss > peak_rss[0]:
                peak_rss[0] = rss
            time.sleep(1.0)

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()

    # on_status log
    status_events: list[dict] = []
    start = time.monotonic()

    def on_status(msg: str) -> None:
        status_events.append({"t": time.monotonic() - start, "msg": msg})

    try:
        result = transcribe(
            file_path=args.audio,
            model_name=args.model,
            device="parakeet-cpu",
            compute_type=args.compute_type,
            on_status=on_status,
        )
        wall = time.monotonic() - start
    finally:
        stop.set()
        sampler.join(timeout=2.0)

    # Анализ интервалов
    times = [e["t"] for e in status_events]
    first_update = times[0] if times else None
    max_gap = max((b - a for a, b in zip(times, times[1:])), default=0.0)

    rtfx = (result.duration / wall) if wall > 0 else 0.0
    peak_rss_gb = peak_rss[0] / (1024 ** 3)

    report = {
        "audio": str(args.audio),
        "duration_sec": result.duration,
        "wall_sec": wall,
        "rtfx": rtfx,
        "peak_rss_gb": peak_rss_gb,
        "first_status_sec": first_update,
        "max_status_gap_sec": max_gap,
        "status_events_count": len(status_events),
        "segments": len(result.segments),
        "compute_type": args.compute_type,
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # Gate thresholds
    ok_rss = peak_rss_gb <= 4.0
    ok_first = first_update is not None and first_update <= 10.0
    ok_gap = max_gap <= 30.0

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nGate: rss={'✓' if ok_rss else '✗'}  "
          f"first_update={'✓' if ok_first else '✗'}  "
          f"max_gap={'✓' if ok_gap else '✗'}")

    return 0 if (ok_rss and ok_first and ok_gap) else 1


if __name__ == "__main__":
    sys.exit(main())
```

Добавить `psutil` в dev-deps если ещё нет:

```bash
uv add --dev psutil
```

- [ ] **Step 3: Прогон на длинном файле**

```bash
uv run scripts/measure_parakeet.py "$TESTFILE"
```

Expected: скрипт завершается; `/tmp/parakeet-measure.json` заполнен; exit code 0 если все gate-пороги пройдены, 1 если нет. В stdout виден report и строка `Gate: rss=✓ first_update=✓ max_gap=✓`.

- [ ] **Step 4: Проверка gate-порогов**

Сверить числа в `/tmp/parakeet-measure.json` с нормами:

| Метрика | Норма | Результат |
|---------|-------|-----------|
| `peak_rss_gb` | ≤4 GB | _заполнить_ |
| `first_status_sec` | ≤10 сек | _заполнить_ |
| `max_status_gap_sec` | ≤30 сек | _заполнить_ |

**Если нарушено:**
- Peak RSS > 4 GB → **БЛОКЕР**: добавить ручной chunking в `ParakeetBackend.transcribe` до продолжения. Создать Task 10b с TDD-реализацией.
- Первый update > 10 сек ИЛИ интервал > 30 сек → **БЛОКЕР**: обернуть `model.recognize` в поток с heartbeat-таймером (паттерн `openvino.py::_generate_with_progress`). Task 10b.

- [ ] **Step 5: Оффлайн-повтор**

```bash
HF_HUB_OFFLINE=1 uv run scripts/measure_parakeet.py "$TESTFILE"
```

Expected: прогон проходит без ошибок (VAD и модель уже в кэше).

Если есть ошибки типа `requires network` / `not in cache` → **БЛОКЕР**: разобраться с VAD-кэшем. Возможно нужно передать `local_files_only=True` в `onnx_asr.load_vad` или явно скачивать VAD в `ensure_model_available`.

- [ ] **Step 6: Записать результаты в `docs/gpu.md`**

Добавить секцию в `docs/gpu.md` (перед разделом «Потенциальные направления», если он есть):

```markdown
## Parakeet TDT v3 (ONNX Runtime CPU)

**Дата:** 2026-04-18
**Модель:** nvidia/parakeet-tdt-0.6b-v3 (int8)
**VAD:** Silero через onnx-asr
**Устройство:** CPU (Intel i7-11800H / Ryzen XXXX — зависит от машины)
**Тестовый файл:** `<имя>.mp3`, длительность ~<NN> минут, RU.

### Long-audio gate (из `docs/superpowers/plans/2026-04-18-parakeet-backend.md` Task 10)

| Метрика | Норма | Результат | Вердикт |
|---------|-------|-----------|---------|
| Peak RSS | ≤4 GB | _N GB_ | ✅/❌ |
| Первый on_status | ≤10 сек | _N сек_ | ✅/❌ |
| Максимальный интервал | ≤30 сек | _N сек_ | ✅/❌ |
| Offline повтор (HF_HUB_OFFLINE=1) | без сети | _ok/fail_ | ✅/❌ |

### Сравнение с Whisper

| Backend | Wall-clock | RTFx | Качество (ручная оценка 1-5) |
|---------|-----------|------|------------------------------|
| Parakeet TDT v3 (int8, CPU) | _N мин_ | _N_ | _N_ |
| OpenVINO Whisper medium (int8, CPU) | _N мин_ | _N_ | _N_ |
| faster-whisper medium (float32, CPU) | _N мин_ | _N_ | _N_ |

### Выводы

- _свободный текст_
```

- [ ] **Step 7: Убедиться, что `cli.py` не был изменён**

```bash
git diff src/local_transcriber/cli.py
```

Expected: никаких изменений (весь измерительный код — в `scripts/measure_parakeet.py`, а не патч в `cli.py`).

- [ ] **Step 8: Commit**

```bash
git add scripts/ pyproject.toml uv.lock docs/gpu.md
git commit -F- <<'EOF'
docs(gpu): long-audio gate — Parakeet TDT v3 + measure script

- Зачем:
  - Task 10 (Long-audio verification gate) — blocker для merge; скрипт для повторных замеров в будущем.
- Что:
  - scripts/measure_parakeet.py — измеритель peak RSS, on_status интервалов, RTFx.
  - psutil в dev-deps.
  - docs/gpu.md: секция «Parakeet TDT v3» с числами и вердиктом.
- Проверка:
  - uv run scripts/measure_parakeet.py <testfile>.
  - cat docs/gpu.md | grep -A 20 "Parakeet TDT v3".
EOF
```

---

## Task 11: ADR-005

**Files:**
- Create: `docs/adr/005-parakeet-backend.md`

- [ ] **Step 1: Создать `docs/adr/005-parakeet-backend.md`**

Прочитать формат `docs/adr/003-pluggable-backends.md` и выдержать стиль. Написать:

```markdown
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
качает VAD (~15MB) через свой HF-mechanism. `ensure_model_available` не trogaет VAD —
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
```

- [ ] **Step 2: Commit**

```bash
git add docs/adr/005-parakeet-backend.md
git commit -F- <<'EOF'
docs(adr): ADR-005 Parakeet backend

- Зачем:
  - зафиксировать архитектурные решения (onnx-asr runtime, Silero VAD как runtime-зависимость, no fallback, effective_model_name через атрибут).
- Что:
  - docs/adr/005-parakeet-backend.md со всеми отклонёнными альтернативами и последствиями.
- Проверка:
  - cat docs/adr/005-parakeet-backend.md.
EOF
```

---

## Task 12: README.md — секция «Parakeet (экспериментально)»

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Прочитать текущую структуру `README.md`**

```bash
uv run python -c "print(open('README.md').read()[:2000])"
```

Найти место после секции OpenVINO / CUDA — куда логично встроить Parakeet.

- [ ] **Step 2: Добавить новую секцию**

В подходящее место (перед FAQ / troubleshooting) вставить:

```markdown
## Parakeet TDT v3 (экспериментально)

Альтернативный бэкенд на NVIDIA Parakeet TDT 0.6B v3 через ONNX Runtime — multilingual
(25 европейских языков, включая русский), быстрый на CPU Intel/AMD.

### Запуск

```bash
transcribe ваш_файл.mp3 --device parakeet
```

Первый запуск скачает:
- Модель `nvidia/parakeet-tdt-0.6b-v3` (~670 MB int8 или ~2 GB fp32) в `~/.cache/huggingface/`.
- Silero VAD (~15 MB) — обязателен для файлов > 20 сек.

Последующие запуски — **оффлайн**.

### Ограничения

- `--language` **игнорируется** — Parakeet v3 определяет язык автоматически. В шапке
  транскрипта — `**Язык**: multi (detected)`. Если указать `--language` явно — CLI напомнит.
- Поддерживаемые `--compute-type`: `int8` (по умолчанию, ~670 MB) и `float32` (~2 GB).
  Другие значения отвергаются.
- Только модель `parakeet-tdt-0.6b-v3` (алиас `parakeet`). Если в вашем `.transcriber.toml`
  стоит `model = "medium"` — добавьте `--model parakeet` к команде или уберите `model` из
  конфига (иначе CLI вернёт ошибку).
- В MVP — только CPU (ONNX Runtime CPU EP). DirectML / OpenVINO EP / CUDA — следующие фазы.
- Ошибка Parakeet **не переключает** на Whisper CPU (silent fallback отключён) — вы получите
  явную ошибку и сможете осознанно выбрать другой `--device`.

### Сравнение производительности

См. `docs/gpu.md`, раздел «Parakeet TDT v3 (ONNX Runtime CPU)» — актуальные числа wall-clock,
RTFx и качества относительно Whisper medium.
```

- [ ] **Step 3: Обновить таблицу опций CLI (если есть) — добавить `parakeet` и `parakeet-cpu` в список значений `--device`**

Найти в `README.md` таблицу/перечисление device-значений, добавить новые.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -F- <<'EOF'
docs(readme): секция Parakeet (экспериментально)

- Зачем:
  - пользователю нужна инструкция по запуску, ограничениям и интерпретации шапки транскрипта.
- Что:
  - секция «Parakeet TDT v3 (экспериментально)» с запуском, ограничениями (язык, compute_type, config-conflict, no-fallback), ссылкой на docs/gpu.md.
  - обновлены значения --device.
- Проверка:
  - grep -A 5 "Parakeet TDT v3" README.md.
EOF
```

---

## Task 13: Финальная проверка и PR

**Files:** ничего не создаём; верификация.

- [ ] **Step 1: Полный прогон тестов**

```bash
uv run pytest -v
```

Expected: все тесты PASS. Если падает — чинить, не двигаться дальше.

- [ ] **Step 2: End-to-end ручной прогон с реальным коротким файлом (если есть)**

```bash
uv run transcribe /путь/к/короткому/файлу.mp3 --device parakeet
```

Expected: создаётся `файл-transcript.md` с шапкой `**Модель**: parakeet-tdt-0.6b-v3`, `**Язык**: multi (detected)`, сегменты с текстом.

- [ ] **Step 3: Проверить, что ничего лишнего не закоммичено**

```bash
git log master..HEAD --oneline
git diff master..HEAD --stat
```

Expected: список коммитов соответствует Task 1-12; изменены/созданы только файлы из списка File Structure.

- [ ] **Step 4: Явно запросить подтверждение у пользователя перед push**

Push в origin и создание PR — user-visible действия. Перед выполнением напечатать пользователю:

```
Готов к push + PR. Подтверждаешь?
  - git push -u origin feature/parakeet-backend
  - gh pr create ...

Ждать явного "да" / "push" / аналогичного подтверждения. НЕ выполнять без явного ответа.
```

После подтверждения выполнить Step 5.

- [ ] **Step 5: Push и PR (только после Step 4)**

```bash
git push -u origin feature/parakeet-backend
gh pr create --title "feat: Parakeet TDT v3 бэкенд (экспериментальный)" --body "$(cat <<'EOF'
## Summary
- Новый pluggable backend — `--device parakeet` на ONNX Runtime + Silero VAD.
- Парик-модель `nvidia/parakeet-tdt-0.6b-v3` (multilingual, 25 языков включая RU).
- В MVP только CPU EP; DirectML/OpenVINO EP/CUDA — следующие фазы.
- No cross-backend fallback — осознанный выбор для экспериментального бэкенда.

## Test plan
- [ ] `uv run pytest -v` — всё зелёное.
- [ ] `uv run transcribe short.mp3 --device parakeet` — шапка `multi (detected)`, `Parakeet (CPU via ONNX Runtime)`.
- [ ] Long-audio gate (см. docs/gpu.md) — Peak RSS ≤4 GB, first status ≤10s, max gap ≤30s, offline повтор ok.

## Спек и ADR
- docs/superpowers/specs/2026-04-18-parakeet-backend-design.md
- docs/adr/005-parakeet-backend.md
EOF
)"
```

Expected: PR создан; URL печатается в stdout.

---

## Self-review checklist (для автора плана)

После прогона всех Task'ов убедиться:

1. **Spec coverage** — каждый пункт из `docs/superpowers/specs/2026-04-18-parakeet-backend-design.md` отработан:
   - Технология (onnx-asr + Silero VAD + CPU EP) — Task 1, 5.
   - `--language` warning только на CLI — Task 8 steps 4-5, Task 6.
   - `--compute-type` маппинг — Task 4, 5.
   - VAD runtime-зависимость — Task 5.
   - Fallback отключён — Task 7.
   - `detect_device` не включает auto→parakeet — **не требует изменений** (текущий `detect_device` уже это соблюдает; добавить проверку в Task 13 Step 2).
   - Config-conflict ошибка — Task 4 (rejects_unknown_model).
   - `effective_model_name` через backend-атрибут — Task 3 (scaffold) + Task 8 (CLI использует getattr).
   - Progress UX — Task 6 (on_status); heartbeat добавляется **только** если Task 10 gate провален.
   - Long-audio gate — Task 10.
   - Тесты покрывают: model validation (Task 4), compute_type map (Task 5), VAD wrapping + `sess_options` для cpu_threads (Task 5), iterator/list results, warning/non-warning на языке, language="multi", duration (Task 6), no-fallback (Task 7), CLI regression Whisper + Parakeet shape + effective_model_name + device_info + config-mismatch error-path (Task 8), end-to-end smoke (Task 9).
   - Documentation: ADR-005 — Task 11, README — Task 12, docs/gpu.md — Task 10.

2. **Placeholder scan** — нет TBD / TODO / «implement later» в шагах (каждый шаг содержит код или точную команду).

3. **Type consistency** — `ParakeetBackend` имеет одинаковые сигнатуры методов во всех Task'ах; `effective_model_name` — str, `actual_compute_type` — Optional[str], `MODEL_REPO` — str, всюду.

Если найдено расхождение — исправить inline.
