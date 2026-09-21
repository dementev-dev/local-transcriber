"""CLI-точка входа (typer). Single и batch режимы транскрипции."""

import sys
import time
import warnings
from collections.abc import Callable
from functools import wraps
from pathlib import Path

import typer
from rich.console import Console
from rich.status import Status

from .config import (
    CliValues,
    load_config,
    resolve_defaults,
)
from .context_menu import install_menu as install_context_menu
from .context_menu import uninstall_menu as uninstall_context_menu
from .diarization import build_speaker_transcript
from .formatter import (
    LANGUAGE_DETECTED,
    LANGUAGE_FORCED,
    LANGUAGE_FROM_MODEL,
    LANGUAGE_UNKNOWN,
    format_duration,
    format_timestamp,
    format_transcript,
    write_transcript,
)
from .quality import (
    TAIL_GAP_WARN_S,
    RepetitionBlock,
    find_repetition_blocks,
    tail_gap,
)
from .speaker_diarizer import SpeakerDiarizer, load_speaker_diarizer
from .transcriber import (
    ExecutionInfo,
    ExecutionRequest,
    Segment,
    Transcriber,
    TranscribeResult,
    cuda_error_hint,
)
from .types import (
    UNKNOWN_LANGUAGE,
    DiarizationRun,
    SpeakerTranscript,
    StatusCallback,
)
from .utils import (
    build_output_path,
    expand_globs,
    has_existing_transcript,
    validate_input_file,
)

app = typer.Typer()
console = Console(stderr=True)


def _print_cuda_hint(exc: BaseException, device: str | None = None) -> None:
    """Показывает подсказку, не заменяя исходную ошибку."""
    hint = cuda_error_hint(exc, device=device)
    if hint:
        console.print(hint, style="yellow", markup=False)


def _show_cli_warning(
    message: Warning | str,
    *_args: object,
    **_kwargs: object,
) -> None:
    """Печатает предупреждение в пользовательском формате CLI."""
    console.print(f"Внимание: {message}", style="yellow", soft_wrap=True, markup=False)


def _with_cli_warning_renderer(
    command: Callable[..., None],
) -> Callable[..., None]:
    """Устанавливает CLI-рендер предупреждений на время одного запуска."""

    @wraps(command)
    def wrapped(*args: object, **kwargs: object) -> None:
        with warnings.catch_warnings():
            warnings.showwarning = _show_cli_warning
            command(*args, **kwargs)

    return wrapped


def _format_language_mode(requested_language: str, result: TranscribeResult) -> str:
    """Описывает источник языка, не выдавая профиль модели за детектор."""
    if requested_language != "auto":
        return LANGUAGE_FORCED
    if result.language_probability > 0:
        return LANGUAGE_DETECTED
    if result.language not in {"", UNKNOWN_LANGUAGE}:
        return LANGUAGE_FROM_MODEL
    return LANGUAGE_UNKNOWN


def _format_repetition_blocks(
    blocks: list[RepetitionBlock],
    use_hours: bool,
) -> str:
    """Формирует краткое описание блоков повторов для консоли."""
    rendered = [
        f"[{format_timestamp(block.start, use_hours=use_hours)} - "
        f"{format_timestamp(block.end, use_hours=use_hours)}] ({block.count}×)"
        for block in blocks[:3]
    ]
    summary = "; ".join(rendered)
    remaining = len(blocks) - 3
    if remaining > 0:
        summary = f"{summary} (+ ещё {remaining})"
    return summary


def _print_quality_warnings(
    result: TranscribeResult, file_name: str | None = None
) -> None:
    """Печатает предупреждения о возможной потере содержания."""
    is_batch = file_name is not None
    use_hours = result.duration > 3600

    gap = tail_gap(result)
    if gap > TAIL_GAP_WARN_S:
        covered = format_duration(result.segments[-1].end)
        total = format_duration(result.duration)
        message = (
            f"транскрипт покрывает {covered} из {total} — возможна потеря хвоста записи"
        )
        if is_batch:
            console.print(f"  {file_name}: {message}", style="yellow")
        else:
            console.print(
                f"Внимание: {message}. Попробуйте другой --device.",
                style="yellow",
            )

    blocks = find_repetition_blocks(result.segments)
    if blocks:
        message = (
            f"блоки повторов: {_format_repetition_blocks(blocks, use_hours)} "
            "— возможны галлюцинации модели"
        )
        if is_batch:
            console.print(f"  {file_name}: {message}", style="yellow")
        else:
            console.print(
                f"Внимание: {message}. Попробуйте другой --device.",
                style="yellow",
            )


def _diarize_result(
    file_path: Path,
    result: TranscribeResult,
    diarizer: SpeakerDiarizer,
    on_status: StatusCallback,
) -> tuple[SpeakerTranscript | None, str | None, DiarizationRun | None]:
    """Запускает диаризацию и переводит ожидаемые сбои в деградацию вывода."""
    try:
        run = diarizer.process(file_path, on_status=on_status)
        transcript = build_speaker_transcript(
            result.words,
            run.intervals,
            result.duration,
        )
        if not run.intervals:
            warning = "Диаризатор не нашёл интервалов при непустом распознавании"
        elif transcript.cluster_count < 2:
            warning = "Найден только один голосовой кластер"
        else:
            warning = None
        return transcript, warning, run
    except Exception as exc:
        return None, f"Диаризация завершилась с ошибкой: {exc}", None


def _print_diarization_report(
    transcript: SpeakerTranscript,
    run: DiarizationRun,
    verbose: bool,
    file_name: str | None = None,
) -> None:
    """Печатает метрики verbose и обязательные предупреждения сведения."""
    if verbose:
        indent = "  " if file_name is not None else ""
        console.print(
            f"{indent}Диаризация: {transcript.cluster_count} кластеров, "
            f"{len(run.intervals)} интервалов, {run.elapsed_seconds:.1f} с"
        )

    warning_prefix = f"  {file_name}: " if file_name is not None else "Внимание: "
    if transcript.unassigned_word_count:
        console.print(
            f"{warning_prefix}{transcript.unassigned_word_count} слов "
            "без назначенного говорящего",
            style="yellow",
        )
    for cluster in transcript.small_clusters:
        label = (
            f"Speaker {cluster.speaker}"
            if cluster.speaker is not None
            else "кластер без номера"
        )
        console.print(
            f"{warning_prefix}малый кластер {label}: {cluster.duration:.1f} с",
            style="yellow",
        )


@app.command()
@_with_cli_warning_renderer
def main(
    files: list[Path] | None = typer.Argument(None, help="Пути к аудио/видеофайлам"),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        show_default=False,
        help="Модель [по умолч.: medium (CUDA) / gigaam-v3-e2e-rnnt (ONNX)]",
    ),
    language: str | None = typer.Option(
        None, "--language", "-l", show_default=False, help="Язык [по умолч.: ru]"
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Путь к выходному файлу"
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        "-d",
        show_default=False,
        help="Устройство (auto|cpu|cuda|openvino|openvino-gpu|openvino-cpu|onnx) [по умолч.: auto]",
    ),
    compute_type: str | None = typer.Option(
        None,
        "--compute-type",
        show_default=False,
        help=(
            "Тип вычислений [по умолч.: float16 (CUDA) / "
            "int8 (ONNX/OpenVINO) / float32 (CPU)]"
        ),
    ),
    threads: int = typer.Option(
        0,
        "--threads",
        "-t",
        show_default=False,
        min=0,
        help="Потоки CPU (0 = дефолт библиотеки; рекомендуется = число физ. ядер)",
    ),
    diarize: bool | None = typer.Option(
        None,
        "--diarize/--no-diarize",
        help="Разделить транскрипт на реплики говорящих",
    ),
    speakers: int | None = typer.Option(
        None,
        "--speakers",
        min=1,
        help="Известное число говорящих; автоматически включает --diarize",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Подробный вывод"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Перезаписать существующие транскрипты"
    ),
    install_menu: bool = typer.Option(
        False, "--install-menu", help="Установить пункт Transcribe в SendTo"
    ),
    uninstall_menu: bool = typer.Option(
        False, "--uninstall-menu", help="Удалить пункт Transcribe из SendTo"
    ),
) -> None:
    """Транскрибирует аудио/видеофайлы в markdown с таймкодами.

    Каскад приоритетов параметров: CLI-флаги > .transcriber.toml > device-aware дефолты.
    """
    files = [] if files is None else files

    if install_menu or uninstall_menu:
        if install_menu and uninstall_menu:
            console.print(
                "--install-menu и --uninstall-menu несовместимы.", style="red bold"
            )
            raise SystemExit(2)
        if files:
            console.print(
                "Флаги меню нельзя использовать вместе с файлами.", style="red bold"
            )
            raise SystemExit(2)
        if sys.platform != "win32":
            console.print(
                "Пункт меню SendTo доступен только на Windows.", style="red bold"
            )
            raise SystemExit(1)

        try:
            if install_menu:
                cmd_path = install_context_menu()
                console.print(f'Пункт меню установлен: "{cmd_path}"', style="green")
            else:
                cmd_path = uninstall_context_menu()
                if cmd_path is None:
                    console.print("Пункт меню не был установлен.", style="yellow")
                else:
                    console.print(f'Пункт меню удалён: "{cmd_path}"', style="green")
        except RuntimeError as exc:
            console.print(f"Ошибка: {exc}", style="red bold")
            raise SystemExit(1)
        return

    if not files:
        console.print(
            "Укажите хотя бы один файл или используйте --install-menu/--uninstall-menu.",
            style="red bold",
        )
        raise SystemExit(2)

    if diarize is False and speakers is not None:
        console.print(
            "--no-diarize и --speakers несовместимы.",
            style="red bold",
        )
        raise SystemExit(2)

    requested_device: str | None = None
    try:
        config = load_config()
        cli_values: CliValues = {
            "model": model,
            "language": language,
            "device": device,
            "compute_type": compute_type,
            "diarize": diarize,
        }
        defaults = resolve_defaults(cli_values, config)
        diarize_enabled = defaults["diarize"] is True or speakers is not None
        requested_device = defaults["device"]

        # Умолчания model/compute_type зависят от устройства — их разрешает module.
        request = ExecutionRequest(
            device=requested_device,
            model=defaults["model"],
            compute_type=defaults["compute_type"],
            language=defaults["language"],
            cpu_threads=threads,
            require_word_timestamps=diarize_enabled,
        )

        expanded = expand_globs(files)
        if not expanded:
            console.print("Файлы не найдены.", style="red bold")
            raise SystemExit(1)

        is_batch = len(expanded) > 1
        if is_batch and output is not None:
            console.print(
                "--output несовместим с несколькими файлами.", style="red bold"
            )
            raise SystemExit(1)

        if is_batch:
            _run_batch(
                expanded,
                request,
                verbose,
                force,
                diarize=diarize_enabled,
                speakers=speakers,
            )
        else:
            _run_single(
                expanded[0],
                request,
                output,
                verbose,
                diarize=diarize_enabled,
                speakers=speakers,
            )
    except KeyboardInterrupt:
        console.print("\nПрервано пользователем.", style="yellow")
        raise SystemExit(130)
    except SystemExit:
        raise
    except ValueError as exc:
        _print_cuda_hint(exc, requested_device)
        console.print(f"Ошибка: {exc}", style="red bold")
        raise SystemExit(1)
    except (FileNotFoundError,) as exc:
        _print_cuda_hint(exc, requested_device)
        console.print(f"Ошибка: {exc}", style="red bold")
        raise SystemExit(1)
    except Exception as exc:
        _print_cuda_hint(exc, requested_device)
        if verbose:
            console.print_exception()
        else:
            console.print(f"Ошибка: {exc}", style="red bold")
            console.print("Запустите с --verbose для полного traceback.", style="dim")
        raise SystemExit(1)


def _print_execution_header(info: ExecutionInfo, verbose: bool) -> None:
    """Печатает выбранное исполнение; в --verbose — версии runtime для диагностики."""
    console.print(
        f"Модель: [bold]{info.model}[/bold]  "
        f"Устройство: [bold]{info.device}[/bold]  "
        f"Compute: [bold]{info.compute_type}[/bold]"
    )
    if info.device == "openvino-gpu" and info.model != "large-v3":
        console.print(
            "Совет: --model large-v3 даёт лучшее качество на GPU (~2x дольше)",
            style="dim",
        )
    if info.device != info.resolved_device:
        console.print(
            f"Запрошено {info.requested_device}, используется {info.device}",
            style="yellow",
        )
    if verbose:
        threads = info.cpu_threads or "по умолчанию библиотеки"
        console.print(
            f"Движок: {info.engine}  Потоки (запрошено): {threads}", style="dim"
        )
        for key, value in info.runtime.items():
            console.print(f"  {key}: {value}", style="dim", markup=False)


def _load_diarizer(
    diarize: bool, speakers: int | None, cpu_threads: int
) -> SpeakerDiarizer | None:
    """Готовит диаризатор до первого ASR; пословный контракт уже проверен module."""
    if not diarize:
        return None
    return load_speaker_diarizer(
        speakers=speakers,
        threads=cpu_threads,
        on_status=lambda message: console.print(message),
    )


def _run_single(
    file: Path,
    request: ExecutionRequest,
    output: Path | None,
    verbose: bool,
    diarize: bool = False,
    speakers: int | None = None,
) -> None:
    """Пайплайн одного файла: валидация → модель → транскрипция → запись."""
    start = time.monotonic()

    validated_file = validate_input_file(file)
    output_path = build_output_path(validated_file, output)

    console.print(f"Файл: [bold]{validated_file.name}[/bold]")

    def on_segment(seg: Segment) -> None:
        console.print(f"  [{seg.start:.2f}s] {seg.text.strip()}")

    transcriber = Transcriber(request)
    info = transcriber.prepare(on_status=lambda msg: console.print(msg))
    _print_execution_header(info, verbose)
    speaker_diarizer = _load_diarizer(diarize, speakers, request.cpu_threads)

    with Status("Подготавливаю запуск...", console=console) as status:
        result = transcriber.transcribe(
            validated_file,
            on_segment=on_segment if verbose else None,
            on_status=status.update,
        )

    speaker_transcript = None
    diarization_warning = None
    diarization_degraded = False
    if speaker_diarizer is not None and result.segments:
        with Status("Определяю говорящих...", console=console) as status:
            speaker_transcript, diarization_warning, diarization_run = _diarize_result(
                validated_file,
                result,
                speaker_diarizer,
                on_status=(
                    (lambda message: console.print(message))
                    if verbose
                    else status.update
                ),
            )
        diarization_degraded = diarization_warning is not None
        if diarization_run is not None and speaker_transcript is not None:
            _print_diarization_report(
                speaker_transcript,
                diarization_run,
                verbose,
            )

        if diarization_warning is not None:
            console.print(f"Внимание: {diarization_warning}", style="yellow")

    if len(result.segments) == 0:
        message = f"Речь не обнаружена в файле {validated_file.name}"
        if speaker_diarizer is not None:
            message += "; диаризация не запускалась"
        console.print(message, style="yellow")

    language_mode = _format_language_mode(request.language or "auto", result)

    content = format_transcript(
        result=result,
        source_filename=validated_file.name,
        model_name=info.model,
        device_info=info.description,
        language_mode=language_mode,
        speaker_transcript=speaker_transcript,
        diarization_warning=diarization_warning,
    )
    write_transcript(content, output_path)

    elapsed = time.monotonic() - start
    console.print(f'Транскрипт сохранён: "{output_path}"', style="green")
    console.print(f"  Сегментов: {len(result.segments)}  Время: {elapsed:.1f}с")
    _print_quality_warnings(result)
    if diarization_degraded:
        raise SystemExit(1)


def _run_batch(
    files: list[Path],
    request: ExecutionRequest,
    verbose: bool,
    force: bool,
    diarize: bool = False,
    speakers: int | None = None,
) -> None:
    """Трёхфазный батч-пайплайн: prescan → загрузка модели → транскрипция."""
    # Phase 1: Prescan — fail-fast + skip до загрузки модели (экономим ~2-5 сек)
    to_process: list[Path] = []
    skipped = 0
    invalid = 0
    for file in files:
        try:
            validated = validate_input_file(file)
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"  Ошибка: {file.name}: {exc}", style="red")
            invalid += 1
            continue
        if not force and has_existing_transcript(validated):
            console.print(
                f"  Пропуск: {file.name} (транскрипт существует)", style="dim"
            )
            skipped += 1
            continue
        to_process.append(validated)

    if not to_process:
        console.print(f"\nИтого: 0 обработано, {skipped} пропущено, {invalid} ошибок")
        if invalid > 0:
            raise SystemExit(1)
        return

    # Phase 2: module выполнения создаётся, когда есть работа
    transcriber = Transcriber(request)
    info = transcriber.prepare(on_status=lambda msg: console.print(msg))
    _print_execution_header(info, verbose)
    speaker_diarizer = _load_diarizer(diarize, speakers, request.cpu_threads)

    # Phase 3: Transcribe
    processed = 0
    degraded = 0
    failed = 0
    batch_start = time.monotonic()

    for i, file in enumerate(to_process, 1):
        try:
            prefix = f"[{i}/{len(to_process)}] {file.name}"
            console.print(f"{prefix}", style="bold")
            file_start = time.monotonic()

            def on_segment(seg: Segment) -> None:
                console.print(f"  [{seg.start:.2f}s] {seg.text.strip()}")

            with Status(f"{prefix}...", console=console) as status:
                result = transcriber.transcribe(
                    file,
                    on_segment=on_segment if verbose else None,
                    on_status=status.update
                    if not verbose
                    else lambda msg: console.print(msg),
                )

            language_mode = _format_language_mode(request.language or "auto", result)
            speaker_transcript = None
            diarization_warning = None
            file_degraded = False

            if speaker_diarizer is not None and result.segments:
                with Status("Определяю говорящих...", console=console) as status:
                    speaker_transcript, diarization_warning, diarization_run = (
                        _diarize_result(
                            file,
                            result,
                            speaker_diarizer,
                            on_status=(
                                (lambda message: console.print(message))
                                if verbose
                                else status.update
                            ),
                        )
                    )
                file_degraded = diarization_warning is not None
                if diarization_run is not None and speaker_transcript is not None:
                    _print_diarization_report(
                        speaker_transcript,
                        diarization_run,
                        verbose,
                        file_name=file.name,
                    )

                if diarization_warning is not None:
                    console.print(
                        f"  {file.name}: {diarization_warning}",
                        style="yellow",
                    )

            if len(result.segments) == 0:
                message = f"  Речь не обнаружена: {file.name}"
                if speaker_diarizer is not None:
                    message += "; диаризация не запускалась"
                console.print(message, style="yellow")

            content = format_transcript(
                result=result,
                source_filename=file.name,
                model_name=info.model,
                device_info=info.description,
                language_mode=language_mode,
                speaker_transcript=speaker_transcript,
                diarization_warning=diarization_warning,
            )
            write_transcript(content, build_output_path(file))
            file_elapsed = time.monotonic() - file_start
            console.print(
                f"  Готово: {file.name}  "
                f"Сегментов: {len(result.segments)}  Время: {file_elapsed:.1f}с",
                style="green",
            )
            processed += 1
            if file_degraded:
                degraded += 1
            _print_quality_warnings(result, file.name)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            _print_cuda_hint(exc, request.device)
            if verbose:
                console.print_exception()
            else:
                console.print(f"  Ошибка: {file.name}: {exc}", style="red")
            failed += 1

    total_failed = invalid + failed
    batch_elapsed = time.monotonic() - batch_start
    console.print(
        f"\nИтого: {processed} обработано, {skipped} пропущено, "
        f"{degraded} с деградацией, {total_failed} ошибок"
        f"  Время: {batch_elapsed:.1f}с"
    )
    if total_failed > 0 or degraded > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    app()
