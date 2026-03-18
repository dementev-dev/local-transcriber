import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.status import Status

from .formatter import format_transcript, write_transcript
from .transcriber import Segment, _is_cuda_error, ensure_model_available, transcribe
from .utils import build_output_path, check_ffmpeg, detect_device, get_gpu_name, validate_input_file

app = typer.Typer()
console = Console(stderr=True)


@app.command()
def main(
    file: Path = typer.Argument(..., help="Путь к аудио- или видеофайлу"),
    model: str = typer.Option("large-v3", "--model", "-m", help="Модель Whisper"),
    language: str = typer.Option("auto", "--language", "-l", help="Язык (ru|en|auto)"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Путь к выходному файлу"),
    device: str = typer.Option("auto", "--device", "-d", help="Устройство (auto|cpu|cuda)"),
    compute_type: str = typer.Option("int8", "--compute-type", help="Тип вычислений"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Подробный вывод"),
) -> None:
    try:
        _run(file, model, language, output, device, compute_type, verbose)
    except KeyboardInterrupt:
        console.print("\nПрервано пользователем.", style="yellow")
        raise SystemExit(130)
    except SystemExit:
        raise
    except (FileNotFoundError, ValueError) as exc:
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


def _run(
    file: Path,
    model: str,
    language: str,
    output: Path | None,
    device: str,
    compute_type: str,
    verbose: bool,
) -> None:
    start = time.monotonic()

    check_ffmpeg()
    validated_file = validate_input_file(file)
    requested_device = device
    resolved_device = detect_device(device)
    strict = requested_device != "auto"
    output_path = build_output_path(validated_file, output)

    console.print(f"Файл: [bold]{validated_file.name}[/bold]")
    console.print(f"Модель: [bold]{model}[/bold]  Устройство: [bold]{resolved_device}[/bold]  Compute: [bold]{compute_type}[/bold]")

    model_path = ensure_model_available(model, on_status=lambda message: console.print(message))

    def on_segment(seg: Segment) -> None:
        console.print(f"  [{seg.start:.2f}s] {seg.text.strip()}")

    with Status("Подготавливаю запуск...", console=console) as status:
        result = transcribe(
            file_path=validated_file,
            model_name=model_path,
            device=resolved_device,
            compute_type=compute_type,
            language=language if language != "auto" else None,
            on_segment=on_segment if verbose else None,
            on_status=status.update,
            strict_device=strict,
        )

    if result.device_used != resolved_device:
        if requested_device == "auto":
            console.print(
                f"Определено устройство {resolved_device}, "
                f"но использовано {result.device_used} (fallback)",
                style="yellow",
            )
        else:
            console.print(
                f"Запрошено {requested_device}, использовано {result.device_used}",
                style="yellow",
            )

    if len(result.segments) == 0:
        console.print(f"Речь не обнаружена в файле {validated_file.name}", style="yellow")

    if result.device_used == "cuda":
        gpu_name = get_gpu_name()
        device_info = f"CUDA ({gpu_name or 'Unknown GPU'})"
    else:
        device_info = "CPU"

    language_mode = "detected" if language == "auto" else "forced"

    content = format_transcript(
        result=result,
        source_filename=validated_file.name,
        model_name=model,
        device_info=device_info,
        language_mode=language_mode,
    )
    write_transcript(content, output_path)

    elapsed = time.monotonic() - start
    console.print(f"Транскрипт сохранён: [bold]{output_path}[/bold]", style="green")
    console.print(f"  Сегментов: {len(result.segments)}  Время: {elapsed:.1f}с")


if __name__ == "__main__":
    app()
