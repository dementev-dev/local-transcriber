"""CLI-точка входа (typer). Single и batch режимы транскрипции."""

import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.status import Status

from .config import apply_device_defaults, load_config, resolve_defaults
from .formatter import format_transcript, write_transcript
from .transcriber import (
    Segment,
    _is_cuda_error,
    _transcribe_file,
    load_model,
)
from .utils import (
    build_output_path,
    detect_device,
    expand_globs,
    get_gpu_name,
    has_existing_transcript,
    validate_input_file,
)

app = typer.Typer()
console = Console(stderr=True)


def _format_device_info(device_used: str) -> str:
    """Формирует строку устройства для шапки транскрипта."""
    if device_used == "cuda":
        gpu_name = get_gpu_name()
        return f"CUDA ({gpu_name or 'Unknown GPU'})"
    if device_used == "openvino":
        return "OpenVINO (CPU)"
    return "CPU"


@app.command()
def main(
    files: list[Path] = typer.Argument(..., help="Пути к аудио/видеофайлам"),
    model: str | None = typer.Option(
        None, "--model", "-m", show_default=False, help="Модель Whisper [по умолч.: medium]"
    ),
    language: str | None = typer.Option(
        None, "--language", "-l", show_default=False, help="Язык [по умолч.: ru]"
    ),
    output: Path | None = typer.Option(None, "--output", "-o", help="Путь к выходному файлу"),
    device: str | None = typer.Option(
        None, "--device", "-d", show_default=False,
        help="Устройство (auto|cpu|cuda|openvino) [по умолч.: auto]"
    ),
    compute_type: str | None = typer.Option(
        None, "--compute-type", show_default=False,
        help="Тип вычислений [по умолч.: float16 (CUDA) / int8 (OpenVINO) / float32 (CPU)]"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Подробный вывод"),
    force: bool = typer.Option(False, "--force", "-f", help="Перезаписать существующие транскрипты"),
) -> None:
    """Транскрибирует аудио/видеофайлы в markdown с таймкодами.

    Каскад приоритетов параметров: CLI-флаги > .transcriber.toml > device-aware дефолты.
    """
    try:
        config = load_config()
        cli_values = {"model": model, "language": language, "device": device, "compute_type": compute_type}
        defaults = resolve_defaults(cli_values, config)

        resolved_device = detect_device(defaults["device"])
        defaults = apply_device_defaults(defaults, resolved_device, cli_values, config)

        ct_explicit = compute_type is not None or "compute_type" in config

        expanded = expand_globs(files)
        if not expanded:
            console.print("Файлы не найдены.", style="red bold")
            raise SystemExit(1)

        is_batch = len(expanded) > 1
        if is_batch and output is not None:
            console.print("--output несовместим с несколькими файлами.", style="red bold")
            raise SystemExit(1)

        if is_batch:
            _run_batch(expanded, defaults, verbose, force, ct_explicit)
        else:
            _run_single(expanded[0], defaults, output, verbose, ct_explicit)
    except KeyboardInterrupt:
        console.print("\nПрервано пользователем.", style="yellow")
        raise SystemExit(130)
    except SystemExit:
        raise
    except ValueError as exc:
        console.print(f"Ошибка: {exc}", style="red bold")
        raise SystemExit(1)
    except (FileNotFoundError,) as exc:
        console.print(f"Ошибка: {exc}", style="red bold")
        raise SystemExit(1)
    except Exception as exc:
        if _is_cuda_error(exc) and sys.platform == "win32":
            console.print(
                "GPU на Windows требует CUDA toolkit (включает cuBLAS).\n"
                "Установите одним из способов:\n"
                "  choco install cuda\n"
                "  winget install -e --id Nvidia.CUDA\n"
                "После установки перезапустите терминал.",
                style="yellow",
            )
        if verbose:
            console.print_exception()
        else:
            console.print(f"Ошибка: {exc}", style="red bold")
            console.print(
                "Запустите с --verbose для полного traceback.", style="dim"
            )
        raise SystemExit(1)


def _run_single(
    file: Path,
    defaults: dict[str, str],
    output: Path | None,
    verbose: bool,
    compute_type_explicit: bool = False,
) -> None:
    """Пайплайн одного файла: валидация → модель → транскрипция → запись."""
    start = time.monotonic()

    validated_file = validate_input_file(file)
    requested_device = defaults["device"]
    resolved_device = detect_device(requested_device)
    strict = requested_device != "auto"
    output_path = build_output_path(validated_file, output)

    console.print(f"Файл: [bold]{validated_file.name}[/bold]")

    def on_segment(seg: Segment) -> None:
        console.print(f"  [{seg.start:.2f}s] {seg.text.strip()}")

    model_obj, actual_device, backend, model_path = load_model(
        defaults["model"], resolved_device, defaults["compute_type"],
        on_status=lambda msg: console.print(msg), strict_device=strict,
        compute_type_explicit=compute_type_explicit,
    )
    actual_ct = getattr(backend, "actual_compute_type", defaults["compute_type"]) or defaults["compute_type"]
    console.print(
        f"Модель: [bold]{defaults['model']}[/bold]  "
        f"Устройство: [bold]{actual_device}[/bold]  "
        f"Compute: [bold]{actual_ct}[/bold]"
    )

    with Status("Подготавливаю запуск...", console=console) as status:
        tfr = _transcribe_file(
            model=model_obj,
            actual_device=actual_device,
            backend=backend,
            model_path=model_path,
            file_path=validated_file,
            model_name=defaults["model"],
            compute_type=defaults["compute_type"],
            language=defaults["language"] if defaults["language"] != "auto" else None,
            on_segment=on_segment if verbose else None,
            on_status=status.update,
            strict_device=strict,
        )

    result = tfr.result

    if tfr.actual_device != resolved_device:
        if requested_device == "auto":
            console.print(
                f"Определено устройство {resolved_device}, "
                f"но использовано {tfr.actual_device} (fallback)",
                style="yellow",
            )
        else:
            console.print(
                f"Запрошено {requested_device}, использовано {tfr.actual_device}",
                style="yellow",
            )

    if len(result.segments) == 0:
        console.print(
            f"Речь не обнаружена в файле {validated_file.name}", style="yellow"
        )

    device_info = _format_device_info(result.device_used)
    language_mode = "detected" if defaults["language"] == "auto" else "forced"

    content = format_transcript(
        result=result,
        source_filename=validated_file.name,
        model_name=defaults["model"],
        device_info=device_info,
        language_mode=language_mode,
    )
    write_transcript(content, output_path)

    elapsed = time.monotonic() - start
    console.print(f"Транскрипт сохранён: [bold]{output_path}[/bold]", style="green")
    console.print(f"  Сегментов: {len(result.segments)}  Время: {elapsed:.1f}с")


def _run_batch(
    files: list[Path],
    defaults: dict[str, str],
    verbose: bool,
    force: bool,
    compute_type_explicit: bool = False,
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
        console.print(
            f"\nИтого: 0 обработано, {skipped} пропущено, {invalid} ошибок"
        )
        if invalid > 0:
            raise SystemExit(1)
        return

    # Phase 2: Load model (ensure + create в одном вызове)
    requested_device = defaults["device"]
    resolved_device = detect_device(requested_device)
    strict = requested_device != "auto"
    model_obj, actual_device, backend, model_path = load_model(
        defaults["model"], resolved_device, defaults["compute_type"],
        on_status=lambda msg: console.print(msg), strict_device=strict,
        compute_type_explicit=compute_type_explicit,
    )

    if actual_device != resolved_device:
        if requested_device == "auto":
            console.print(
                f"Определено устройство {resolved_device}, "
                f"но используется {actual_device} (fallback)",
                style="yellow",
            )
        else:
            console.print(
                f"Запрошено {requested_device}, используется {actual_device}",
                style="yellow",
            )

    # Phase 3: Transcribe
    processed = 0
    failed = 0
    language_mode = "detected" if defaults["language"] == "auto" else "forced"

    batch_start = time.monotonic()

    for i, file in enumerate(to_process, 1):
        try:
            prefix = f"[{i}/{len(to_process)}] {file.name}"
            console.print(f"{prefix}", style="bold")
            file_start = time.monotonic()

            def on_segment(seg: Segment) -> None:
                console.print(f"  [{seg.start:.2f}s] {seg.text.strip()}")

            with Status(f"{prefix}...", console=console) as status:
                tfr = _transcribe_file(
                    model=model_obj,
                    actual_device=actual_device,
                    backend=backend,
                    model_path=model_path,
                    file_path=file,
                    model_name=defaults["model"],
                    compute_type=defaults["compute_type"],
                    language=defaults["language"] if defaults["language"] != "auto" else None,
                    on_segment=on_segment if verbose else None,
                    on_status=status.update if not verbose else lambda msg: console.print(msg),
                    strict_device=strict,
                )

            if tfr.actual_device != actual_device:
                console.print(
                    f"  {file.name}: fallback на {tfr.actual_device} при транскрипции",
                    style="yellow",
                )
            # Обновляем после возможного mid-stream fallback
            model_obj = tfr.model
            actual_device = tfr.actual_device
            backend = tfr.backend
            model_path = tfr.model_path

            result = tfr.result

            if len(result.segments) == 0:
                console.print(
                    f"  Речь не обнаружена: {file.name}", style="yellow"
                )

            device_info = _format_device_info(result.device_used)

            content = format_transcript(
                result=result,
                source_filename=file.name,
                model_name=defaults["model"],
                device_info=device_info,
                language_mode=language_mode,
            )
            write_transcript(content, build_output_path(file))
            file_elapsed = time.monotonic() - file_start
            console.print(
                f"  Готово: {file.name}  "
                f"Сегментов: {len(result.segments)}  Время: {file_elapsed:.1f}с",
                style="green",
            )
            processed += 1
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if verbose:
                console.print_exception()
            else:
                console.print(f"  Ошибка: {file.name}: {exc}", style="red")
            failed += 1

    total_failed = invalid + failed
    batch_elapsed = time.monotonic() - batch_start
    console.print(
        f"\nИтого: {processed} обработано, {skipped} пропущено, {total_failed} ошибок"
        f"  Время: {batch_elapsed:.1f}с"
    )
    if total_failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    app()
