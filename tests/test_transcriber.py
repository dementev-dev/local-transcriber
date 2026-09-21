from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from local_transcriber.transcriber import (
    ExecutionRequest,
    Segment,
    Transcriber,
    TranscribeResult,
    cuda_error_hint,
    ensure_model_available,
    transcribe,
)
from local_transcriber.types import WordTimestampsUnavailableError

# === Helpers ===


@pytest.mark.parametrize(
    "error",
    [RuntimeError("out of memory"), RuntimeError("CUDA unknown error")],
)
def test_unknown_error_does_not_suggest_cuda_installation(error):
    """CPU OOM и неизвестная причина не получают совет по установке CUDA."""
    assert cuda_error_hint(error) is None


def test_windows_loader_error_suggests_extra():
    """OSError загрузчика DLL распознаётся без обёртки RuntimeError."""
    error = OSError("[WinError 126] Could not find module 'cublas64_12.dll'")
    assert "uv sync --extra cuda" in cuda_error_hint(error)


def _make_result(
    count: int = 2,
    language: str = "ru",
    probability: float = 0.95,
    duration: float = 60.0,
    device_used: str = "cpu",
) -> TranscribeResult:
    segments = [
        Segment(start=float(i * 5), end=float(i * 5 + 4), text=f" Segment {i}")
        for i in range(count)
    ]
    return TranscribeResult(
        segments=segments,
        language=language,
        language_probability=probability,
        duration=duration,
        device_used=device_used,
    )


def _make_backend(
    model=None,
    transcribe_result=None,
    create_model_error=None,
    transcribe_error=None,
    model_path="/mock/model",
):
    """Создаёт mock-бэкенд с настраиваемым поведением."""
    backend = MagicMock()
    backend.ensure_model_available.return_value = model_path

    if create_model_error:
        backend.create_model.side_effect = create_model_error
    else:
        backend.create_model.return_value = model or MagicMock()

    if transcribe_error:
        backend.transcribe.side_effect = transcribe_error
    elif transcribe_result:
        backend.transcribe.return_value = transcribe_result
    else:
        backend.transcribe.return_value = _make_result()

    return backend


def _create_model_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    (path / "preprocessor_config.json").write_text("{}")
    (path / "tokenizer.json").write_text("{}")
    (path / "vocabulary.json").write_text("{}")
    (path / "model.bin").write_bytes(b"ok")
    return path


# === transcribe() tests ===


@patch("local_transcriber.transcriber.get_backend")
def test_transcribe_collects_segments(mock_get_backend):
    result_data = _make_result(count=3)
    backend = _make_backend(transcribe_result=result_data)
    mock_get_backend.return_value = backend

    result = transcribe(
        file_path=Path("test.mp3"),
        model_name="tiny",
        device="cpu",
    )

    assert len(result.segments) == 3
    assert result.segments[0].text == " Segment 0"
    assert result.segments[2].text == " Segment 2"
    assert result.language == "ru"
    assert result.language_probability == 0.95
    assert result.duration == 60.0


@patch("local_transcriber.transcriber.get_backend")
def test_transcribe_calls_on_segment(mock_get_backend):
    result_data = _make_result(count=3)
    backend = _make_backend(transcribe_result=result_data)
    mock_get_backend.return_value = backend

    callback = MagicMock()

    transcribe(
        file_path=Path("test.mp3"),
        model_name="tiny",
        device="cpu",
        on_segment=callback,
    )

    # on_segment is passed through to backend.transcribe
    call_args = backend.transcribe.call_args
    assert call_args.kwargs.get("on_segment") is callback or call_args[0][3] is callback


@patch("local_transcriber.transcriber.get_backend")
def test_transcribe_cuda_fallback(mock_get_backend):
    """CUDA error at init -> fallback на CPU."""
    cuda_backend = _make_backend(create_model_error=RuntimeError("CUDA out of memory"))
    cpu_backend = _make_backend(
        transcribe_result=_make_result(count=2, device_used="cpu"),
        model_path="/mock/cpu/model",
    )

    def backend_for_device(device, **kwargs):
        return cuda_backend if device == "cuda" else cpu_backend

    mock_get_backend.side_effect = backend_for_device

    with pytest.warns(UserWarning, match="Переключение на CPU"):
        result = transcribe(
            file_path=Path("test.mp3"),
            model_name="tiny",
            device="cuda",
        )

    assert result.device_used == "cpu"
    assert len(result.segments) == 2


@patch("local_transcriber.transcriber.get_backend")
def test_transcribe_device_used(mock_get_backend):
    backend = _make_backend(
        transcribe_result=_make_result(count=1, device_used="cuda"),
    )
    mock_get_backend.return_value = backend

    result = transcribe(
        file_path=Path("test.mp3"),
        model_name="tiny",
        device="cuda",
    )

    assert result.device_used == "cuda"
    backend.create_model.assert_called_once()


@patch("local_transcriber.transcriber.get_backend")
def test_transcribe_cuda_fallback_on_transcribe_call(mock_get_backend):
    """CUDA error in transcribe (not init) triggers CPU fallback."""
    cuda_backend = _make_backend(
        transcribe_error=RuntimeError("CUDA error during transcription"),
    )
    cpu_backend = _make_backend(
        transcribe_result=_make_result(count=2, device_used="cpu"),
        model_path="/mock/cpu/model",
    )

    def backend_for_device(device, **kwargs):
        return cuda_backend if device == "cuda" else cpu_backend

    mock_get_backend.side_effect = backend_for_device

    with pytest.warns(UserWarning, match="Переключение на CPU"):
        result = transcribe(
            file_path=Path("test.mp3"),
            model_name="tiny",
            device="cuda",
        )

    assert result.device_used == "cpu"
    assert len(result.segments) == 2


@patch("local_transcriber.transcriber.get_backend")
def test_transcribe_reports_missing_socksio_for_proxy(mock_get_backend):
    backend = _make_backend(
        create_model_error=ImportError(
            "Using SOCKS proxy, but the 'socksio' package is not installed."
        ),
    )
    mock_get_backend.return_value = backend

    # ImportError is not caught as backend error → propagates
    with pytest.raises(ImportError, match="socksio"):
        transcribe(
            file_path=Path("test.mp3"),
            model_name="tiny",
            device="cpu",
        )


@patch("local_transcriber.transcriber.get_backend")
def test_transcribe_reports_status_transitions(mock_get_backend):
    backend = _make_backend(transcribe_result=_make_result(count=1))
    mock_get_backend.return_value = backend

    statuses: list[str] = []

    transcribe(
        file_path=Path("test.mp3"),
        model_name="tiny",
        device="cpu",
        on_status=statuses.append,
    )

    # Статусы подготовки и распознавания идут через один callback
    assert any("Инициализирую модель" in s for s in statuses)
    assert any("Транскрибирую" in s for s in statuses)


@patch("local_transcriber.transcriber.get_backend")
def test_transcribe_strict_cuda_error(mock_get_backend):
    """strict_device=True + CUDA error -> raise, без fallback."""
    backend = _make_backend(create_model_error=RuntimeError("CUDA out of memory"))
    mock_get_backend.return_value = backend

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        transcribe(
            file_path=Path("test.mp3"),
            model_name="tiny",
            device="cuda",
            strict_device=True,
        )


@patch("local_transcriber.transcriber.get_backend")
def test_transcribe_non_strict_cuda_fallback(mock_get_backend):
    """strict_device=False + CUDA error -> fallback на CPU."""
    cuda_backend = _make_backend(create_model_error=RuntimeError("CUDA out of memory"))
    cpu_backend = _make_backend(
        transcribe_result=_make_result(count=2, device_used="cpu"),
        model_path="/mock/cpu/model",
    )

    def backend_for_device(device, **kwargs):
        return cuda_backend if device == "cuda" else cpu_backend

    mock_get_backend.side_effect = backend_for_device

    with pytest.warns(UserWarning, match="Переключение на CPU"):
        result = transcribe(
            file_path=Path("test.mp3"),
            model_name="tiny",
            device="cuda",
            strict_device=False,
        )

    assert result.device_used == "cpu"
    assert len(result.segments) == 2


@patch("local_transcriber.transcriber.get_backend")
def test_transcribe_strict_cuda_error_during_transcription(mock_get_backend):
    """strict_device=True + CUDA error during transcription -> raise."""
    backend = _make_backend(
        transcribe_error=RuntimeError("CUDA error during transcription"),
    )
    mock_get_backend.return_value = backend

    with pytest.raises(RuntimeError, match="CUDA error during transcription"):
        transcribe(
            file_path=Path("test.mp3"),
            model_name="tiny",
            device="cuda",
            strict_device=True,
        )


# === ensure_model_available() tests (через FasterWhisperBackend) ===


@patch("local_transcriber.backends.faster_whisper.snapshot_download")
def test_ensure_model_available_uses_cache_first(mock_snapshot_download, tmp_path):
    model_dir = _create_model_dir(tmp_path / "cache-model")
    mock_snapshot_download.return_value = str(model_dir)

    result = ensure_model_available("large-v3")

    assert result == str(model_dir)
    mock_snapshot_download.assert_called_once()
    assert mock_snapshot_download.call_args.kwargs["local_files_only"] is True


@patch("local_transcriber.backends.faster_whisper._validate_model_dir")
@patch("local_transcriber.backends.faster_whisper.snapshot_download")
def test_ensure_model_available_downloads_on_cache_miss(
    mock_snapshot_download, mock_validate_model_dir
):
    from huggingface_hub.errors import LocalEntryNotFoundError

    mock_snapshot_download.side_effect = [
        LocalEntryNotFoundError("not cached"),
        "/downloaded/model",
    ]
    statuses: list[str] = []

    result = ensure_model_available("large-v3", on_status=statuses.append)

    assert Path(result) == Path("/downloaded/model")
    assert mock_snapshot_download.call_args_list[0].kwargs["local_files_only"] is True
    assert mock_snapshot_download.call_args_list[1].kwargs["local_files_only"] is False
    assert "Проверяю кэш модели large-v3..." in statuses
    assert "Скачиваю модель large-v3 из Hugging Face..." in statuses


def test_ensure_model_available_accepts_local_directory(tmp_path):
    model_dir = _create_model_dir(tmp_path / "model")

    result = ensure_model_available(str(model_dir))

    assert result == str(model_dir)


def test_ensure_model_available_accepts_repo_id(tmp_path):
    model_dir = _create_model_dir(tmp_path / "repo-model")
    with patch(
        "local_transcriber.backends.faster_whisper.snapshot_download",
        return_value=str(model_dir),
    ) as mock_snapshot_download:
        result = ensure_model_available("org/model")

    assert result == str(model_dir)
    assert mock_snapshot_download.call_args.kwargs["local_files_only"] is True


def test_ensure_model_available_rejects_unsupported_alias():
    with pytest.raises(ValueError, match="Неподдерживаемая модель"):
        ensure_model_available("distil-large-v3")


@patch("local_transcriber.backends.faster_whisper.snapshot_download")
def test_ensure_model_available_redownloads_incomplete_cache(
    mock_snapshot_download, tmp_path
):
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "config.json").write_text("{}")
    (incomplete / "preprocessor_config.json").write_text("{}")
    (incomplete / "tokenizer.json").write_text("{}")
    (incomplete / "vocabulary.json").write_text("{}")

    complete = tmp_path / "complete"
    complete.mkdir()
    (complete / "config.json").write_text("{}")
    (complete / "preprocessor_config.json").write_text("{}")
    (complete / "tokenizer.json").write_text("{}")
    (complete / "vocabulary.json").write_text("{}")
    (complete / "model.bin").write_bytes(b"ok")

    mock_snapshot_download.side_effect = [
        str(incomplete),
        str(complete),
    ]
    statuses: list[str] = []

    result = ensure_model_available("large-v3", on_status=statuses.append)

    assert result == str(complete)
    assert "Кэш модели large-v3 неполный, докачиваю..." in statuses


def test_ensure_model_available_rejects_incomplete_local_directory(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    with pytest.raises(ValueError, match="Неполная локальная модель"):
        ensure_model_available(str(model_dir))


def test_ensure_model_available_openvino_default_compute_type():
    """ensure_model_available(device='openvino-cpu') без compute_type не падает."""
    from local_transcriber.backends.openvino import OpenVINOBackend

    backend = OpenVINOBackend(ov_device="openvino-cpu", compute_type_explicit=True)
    # Проверяем что _resolve_repo работает с дефолтным compute_type для openvino (int8)
    repo, ct = backend._resolve_repo("medium", "int8")
    assert repo == "OpenVINO/whisper-medium-int8-ov"
    assert ct == "int8"


# === Transcriber: module выполнения ===


def _make_run_backend(**kwargs):
    """Fake adapter для external interface: runtime_info возвращает словарь."""
    backend = _make_backend(**kwargs)
    backend.runtime_info.return_value = {}
    backend.engine = "fake-engine"
    backend.word_timestamps_available = True
    backend.actual_compute_type = None
    backend.actual_ov_device = None
    return backend


@patch("local_transcriber.transcriber.get_backend")
def test_run_loads_model_once_for_two_files(mock_get_backend):
    """Два файла переиспользуют одну загрузку модели."""
    backend = _make_run_backend()
    mock_get_backend.return_value = backend

    run = Transcriber(ExecutionRequest(device="cpu", model="tiny", compute_type="int8"))
    first = run.transcribe(Path("a.mp3"))
    second = run.transcribe(Path("b.mp3"))

    assert backend.create_model.call_count == 1
    assert backend.transcribe.call_count == 2
    assert first.device_used == "cpu"
    assert second.device_used == "cpu"


@patch("local_transcriber.transcriber.get_backend")
def test_run_does_not_load_model_before_first_file(mock_get_backend):
    """Создание module без файлов не загружает модель."""
    mock_get_backend.return_value = _make_run_backend()

    Transcriber(ExecutionRequest(device="cpu", model="tiny", compute_type="int8"))

    mock_get_backend.assert_not_called()


@patch("local_transcriber.transcriber.get_backend")
def test_auto_resolves_to_onnx_cpu_with_device_defaults(mock_get_backend):
    """auto → ONNX CPU и его модель/квантизация по умолчанию; сведения различают запрошенное и выбранное."""
    backend = _make_run_backend()
    mock_get_backend.return_value = backend

    info = Transcriber(ExecutionRequest()).prepare()

    mock_get_backend.assert_called_once_with("onnx", compute_type_explicit=False)
    backend.ensure_model_available.assert_called_once_with(
        "gigaam-v3-e2e-rnnt", "int8", None
    )
    assert info.requested_device == "auto"
    assert info.device == "onnx"
    assert info.model == "gigaam-v3-e2e-rnnt"
    assert info.compute_type == "int8"
    assert info.description == "ONNX (CPU)"


@patch("local_transcriber.transcriber.get_backend")
def test_run_falls_back_to_cpu_at_load_and_keeps_state_for_next_file(
    mock_get_backend,
):
    """Не-strict CUDA-ошибка при загрузке → CPU; следующий файл идёт на CPU без перезагрузки."""
    cuda_backend = _make_run_backend(
        create_model_error=RuntimeError("CUDA out of memory")
    )
    cpu_backend = _make_run_backend()
    mock_get_backend.side_effect = lambda device, **_: (
        cuda_backend if device == "cuda" else cpu_backend
    )

    run = Transcriber(
        ExecutionRequest(
            device="cuda",
            model="tiny",
            compute_type="int8",
            strict_device=False,
            cpu_threads=3,
        )
    )
    with pytest.warns(UserWarning, match="Переключение на CPU"):
        first = run.transcribe(Path("a.mp3"))
    second = run.transcribe(Path("b.mp3"))

    assert first.device_used == "cpu"
    assert second.device_used == "cpu"
    assert cpu_backend.create_model.call_count == 1
    assert cpu_backend.create_model.call_args.kwargs["cpu_threads"] == 3
    assert run.execution.resolved_device == "cuda"
    assert run.execution.device == "cpu"


@patch("local_transcriber.transcriber.get_backend")
def test_run_strict_device_raises_without_fallback(mock_get_backend):
    """Явный device (strict по умолчанию) не подменяется другим исполнением."""
    mock_get_backend.return_value = _make_run_backend(
        create_model_error=RuntimeError("CUDA driver version is insufficient")
    )

    with pytest.raises(RuntimeError, match="driver version"):
        Transcriber(
            ExecutionRequest(device="cuda", model="tiny", compute_type="int8")
        ).prepare()

    assert mock_get_backend.call_count == 1


@patch("local_transcriber.transcriber.get_backend")
def test_run_midstream_fallback_reloads_once_and_continues_batch(mock_get_backend):
    """CUDA-ошибка во время распознавания → повтор файла на CPU; батч продолжается на CPU."""
    cuda_backend = _make_run_backend(
        transcribe_error=RuntimeError("CUBLAS_STATUS_ALLOC_FAILED")
    )
    cpu_backend = _make_run_backend()
    mock_get_backend.side_effect = lambda device, **_: (
        cuda_backend if device == "cuda" else cpu_backend
    )

    run = Transcriber(
        ExecutionRequest(
            device="cuda", model="tiny", compute_type="int8", strict_device=False
        )
    )
    with pytest.warns(UserWarning, match="Переключение на CPU и повтор"):
        first = run.transcribe(Path("a.mp3"))
    second = run.transcribe(Path("b.mp3"))

    assert first.device_used == "cpu"
    assert second.device_used == "cpu"
    assert cpu_backend.create_model.call_count == 1
    assert cpu_backend.transcribe.call_count == 2
    assert run.execution.device == "cpu"


@patch("local_transcriber.transcriber.get_backend")
def test_run_ordinary_file_error_does_not_change_state(mock_get_backend):
    """Ошибка файла (не движка) пробрасывается, модель и исполнение остаются прежними."""
    backend = _make_run_backend()
    backend.transcribe.side_effect = [
        ValueError("Не удалось декодировать"),
        _make_result(),
    ]
    mock_get_backend.return_value = backend

    run = Transcriber(
        ExecutionRequest(
            device="cuda", model="tiny", compute_type="int8", strict_device=False
        )
    )
    with pytest.raises(ValueError, match="декодировать"):
        run.transcribe(Path("bad.mp3"))
    result = run.transcribe(Path("good.mp3"))

    assert result.device_used == "cuda"
    assert backend.create_model.call_count == 1
    assert mock_get_backend.call_count == 1


@patch("local_transcriber.transcriber.get_backend")
def test_run_word_timestamps_error_is_not_a_fallback_reason(mock_get_backend):
    """Нарушение пословного контракта не считается ошибкой движка."""
    mock_get_backend.return_value = _make_run_backend(
        transcribe_error=WordTimestampsUnavailableError("нет слов")
    )

    run = Transcriber(
        ExecutionRequest(
            device="cuda", model="tiny", compute_type="int8", strict_device=False
        )
    )
    with pytest.raises(WordTimestampsUnavailableError):
        run.transcribe(Path("a.mp3"))

    assert mock_get_backend.call_count == 1


@patch("local_transcriber.transcriber.get_backend")
def test_run_requires_word_timestamps_before_first_file(mock_get_backend):
    """Требование пословного контракта проверяется при подготовке, ASR не запускается."""
    backend = _make_run_backend()
    backend.word_timestamps_available = False
    mock_get_backend.return_value = backend

    run = Transcriber(
        ExecutionRequest(
            device="onnx", model="some/raw-model", require_word_timestamps=True
        )
    )
    with pytest.raises(ValueError, match="пословные таймкоды"):
        run.transcribe(Path("a.mp3"))

    backend.transcribe.assert_not_called()


@patch("local_transcriber.transcriber.get_backend")
def test_run_reports_actual_openvino_device(mock_get_backend):
    """openvino-gpu, выбранный до загрузки, уточняется по фактическому устройству OpenVINO."""
    backend = _make_run_backend()
    backend.actual_ov_device = "CPU"
    backend.actual_compute_type = "int8"
    mock_get_backend.return_value = backend

    with patch(
        "local_transcriber.transcriber.detect_device", return_value="openvino-gpu"
    ):
        info = Transcriber(
            ExecutionRequest(device="openvino", model="medium")
        ).prepare()

    assert info.requested_device == "openvino"
    assert info.resolved_device == "openvino-gpu"
    assert info.device == "openvino-cpu"
    assert info.compute_type == "int8"
    assert info.description == "OpenVINO (CPU)"


@patch("local_transcriber.transcriber.get_gpu_name", return_value="RTX 3060")
@patch("local_transcriber.transcriber.get_backend")
def test_run_exposes_runtime_and_gpu_name_from_module_data(mock_get_backend, _gpu):
    """Сведения о runtime берутся из adapter'а, имя GPU — из драйвера, потоки — из запроса."""
    backend = _make_run_backend()
    backend.runtime_info.return_value = {"ctranslate2": "4.5.0"}
    mock_get_backend.return_value = backend

    info = Transcriber(
        ExecutionRequest(
            device="cuda", model="tiny", compute_type="float16", cpu_threads=4
        )
    ).prepare()

    assert info.description == "CUDA (RTX 3060)"
    assert info.runtime == {"ctranslate2": "4.5.0"}
    assert info.cpu_threads == 4
    backend.create_model.assert_called_once_with(
        "/mock/model", "cuda", "float16", cpu_threads=4
    )


@patch("local_transcriber.transcriber.get_backend")
def test_run_passes_auto_language_as_none(mock_get_backend):
    backend = _make_run_backend()
    mock_get_backend.return_value = backend

    Transcriber(
        ExecutionRequest(device="cpu", model="tiny", language="auto")
    ).transcribe(Path("a.mp3"))

    assert backend.transcribe.call_args[0][2] is None


@patch("local_transcriber.transcriber.get_backend")
def test_public_transcribe_uses_the_same_execution_module(mock_get_backend):
    """transcribe() без параметров идёт тем же путём, что CLI по умолчанию: ONNX CPU."""
    backend = _make_run_backend(transcribe_result=_make_result(count=1))
    mock_get_backend.return_value = backend

    result = transcribe(Path("a.mp3"))

    mock_get_backend.assert_called_once_with("onnx", compute_type_explicit=False)
    backend.ensure_model_available.assert_called_once_with(
        "gigaam-v3-e2e-rnnt", "int8", None
    )
    assert result.device_used == "onnx"
    assert len(result.segments) == 1


@pytest.mark.parametrize(
    ("device", "ov_device", "gpu_name", "intel_name", "expected"),
    [
        ("cpu", None, None, None, "CPU"),
        ("onnx", None, "RTX 3060", None, "ONNX (CPU)"),
        ("cuda", None, "RTX 3060", None, "CUDA (RTX 3060)"),
        ("cuda", None, None, None, "CUDA (Unknown GPU)"),
        ("openvino-gpu", "GPU", None, "Intel Arc 140T", "OpenVINO (Intel Arc 140T)"),
        ("openvino-gpu", "GPU", None, None, "OpenVINO (Intel GPU)"),
        ("openvino-cpu", "CPU", None, None, "OpenVINO (CPU)"),
    ],
)
@patch("local_transcriber.transcriber.get_backend")
def test_run_description_names_hardware_only_from_driver_data(
    mock_get_backend, device, ov_device, gpu_name, intel_name, expected
):
    """Строка исполнения для шапки: ONNX не выдаётся за CPU faster-whisper, GPU — по драйверу."""
    backend = _make_run_backend()
    backend.actual_ov_device = ov_device
    mock_get_backend.return_value = backend

    with (
        patch("local_transcriber.transcriber.get_gpu_name", return_value=gpu_name),
        patch("local_transcriber.transcriber.get_intel_gpu_name", return_value=intel_name),
    ):
        info = Transcriber(ExecutionRequest(device=device, model="m")).prepare()

    assert info.description == expected


@patch("local_transcriber.transcriber.get_backend")
def test_run_openvino_runtime_error_falls_back_when_not_strict(mock_get_backend):
    """Любой RuntimeError OpenVINO считается ошибкой движка: не-strict запуск уходит на CPU."""
    ov_backend = _make_run_backend(
        create_model_error=RuntimeError("Exception from src/inference/src/core.cpp")
    )
    cpu_backend = _make_run_backend()
    mock_get_backend.side_effect = lambda device, **_: (
        ov_backend if device.startswith("openvino") else cpu_backend
    )

    run = Transcriber(
        ExecutionRequest(device="openvino-gpu", model="medium", strict_device=False)
    )
    with pytest.warns(UserWarning, match="Переключение на CPU"):
        result = run.transcribe(Path("a.mp3"))

    assert result.device_used == "cpu"


@patch("local_transcriber.transcriber.get_backend")
def test_run_openvino_strict_raises_without_fallback(mock_get_backend):
    mock_get_backend.return_value = _make_run_backend(
        create_model_error=RuntimeError("GPU plugin failed")
    )

    with pytest.raises(RuntimeError, match="GPU plugin"):
        Transcriber(ExecutionRequest(device="openvino-gpu", model="medium")).prepare()

    assert mock_get_backend.call_count == 1


@patch("local_transcriber.transcriber.get_backend")
def test_run_reports_openvino_gpu_chosen_for_cpu_request(mock_get_backend):
    """Фактическое устройство OpenVINO важнее запрошенного и в обратную сторону."""
    backend = _make_run_backend()
    backend.actual_ov_device = "GPU"
    mock_get_backend.return_value = backend

    info = Transcriber(ExecutionRequest(device="openvino-cpu", model="medium")).prepare()

    assert info.resolved_device == "openvino-cpu"
    assert info.device == "openvino-gpu"


@patch("local_transcriber.transcriber.get_backend")
def test_run_fallback_rederives_implicit_compute_type_for_cpu(mock_get_backend):
    """Неявный compute_type следует за фактическим устройством: CPU не получает float16."""
    cuda_backend = _make_run_backend(create_model_error=RuntimeError("CUDA error"))
    cpu_backend = _make_run_backend()
    mock_get_backend.side_effect = lambda device, **_: (
        cuda_backend if device == "cuda" else cpu_backend
    )

    run = Transcriber(ExecutionRequest(device="cuda", strict_device=False))
    with pytest.warns(UserWarning, match="Переключение на CPU"):
        info = run.prepare()

    cuda_backend.ensure_model_available.assert_called_once_with("medium", "float16", None)
    cpu_backend.ensure_model_available.assert_called_once_with("medium", "float32", None)
    assert info.compute_type == "float32"


@patch("local_transcriber.transcriber.get_backend")
def test_run_fallback_keeps_explicit_compute_type(mock_get_backend):
    cuda_backend = _make_run_backend(create_model_error=RuntimeError("CUDA error"))
    cpu_backend = _make_run_backend()
    mock_get_backend.side_effect = lambda device, **_: (
        cuda_backend if device == "cuda" else cpu_backend
    )

    run = Transcriber(
        ExecutionRequest(device="cuda", compute_type="int8", strict_device=False)
    )
    with pytest.warns(UserWarning):
        run.prepare()

    cpu_backend.ensure_model_available.assert_called_once_with("medium", "int8", None)


@patch("local_transcriber.transcriber.get_backend")
def test_run_failed_capability_check_does_not_mark_module_prepared(mock_get_backend):
    """После отказа по пословному контракту повторный вызов не запускает ASR."""
    backend = _make_run_backend()
    backend.word_timestamps_available = False
    mock_get_backend.return_value = backend

    run = Transcriber(
        ExecutionRequest(device="onnx", model="some/raw-model", require_word_timestamps=True)
    )
    with pytest.raises(ValueError, match="пословные таймкоды"):
        run.prepare()
    with pytest.raises(ValueError, match="пословные таймкоды"):
        run.transcribe(Path("a.mp3"))

    backend.transcribe.assert_not_called()


@pytest.mark.parametrize(
    ("device", "model", "compute_type"),
    [
        ("cpu", "medium", "float32"),
        ("cuda", "medium", "float16"),
        ("openvino-cpu", "medium", "int8"),
        ("openvino-gpu", "medium", "int8"),
        ("onnx", "gigaam-v3-e2e-rnnt", "int8"),
    ],
)
@patch("local_transcriber.transcriber.get_backend")
def test_run_applies_device_defaults_when_request_leaves_them_empty(
    mock_get_backend, device, model, compute_type
):
    backend = _make_run_backend()
    mock_get_backend.return_value = backend

    Transcriber(ExecutionRequest(device=device)).prepare()

    mock_get_backend.assert_called_once_with(device, compute_type_explicit=False)
    backend.ensure_model_available.assert_called_once_with(model, compute_type, None)


@patch("local_transcriber.transcriber.get_backend")
def test_run_explicit_values_override_device_defaults(mock_get_backend):
    backend = _make_run_backend()
    mock_get_backend.return_value = backend

    Transcriber(ExecutionRequest(device="cuda", model="large-v3", compute_type="int8")).prepare()

    mock_get_backend.assert_called_once_with("cuda", compute_type_explicit=True)
    backend.ensure_model_available.assert_called_once_with("large-v3", "int8", None)


@patch("local_transcriber.transcriber.get_backend")
def test_run_engine_comes_from_adapter(mock_get_backend):
    backend = _make_run_backend()
    backend.engine = "openvino"
    mock_get_backend.return_value = backend

    info = Transcriber(ExecutionRequest(device="openvino-cpu", model="medium")).prepare()

    assert info.engine == "openvino"


@patch("local_transcriber.transcriber.get_backend")
def test_run_diagnostics_failure_does_not_fail_the_run(mock_get_backend):
    """Сбой сбора runtime_info не роняет распознавание, а попадает в сведения."""
    backend = _make_run_backend()
    backend.runtime_info.side_effect = RuntimeError("plugin enumeration failed")
    mock_get_backend.return_value = backend

    info = Transcriber(ExecutionRequest(device="cpu", model="tiny")).prepare()

    assert "plugin enumeration failed" in info.runtime["diagnostics"]
