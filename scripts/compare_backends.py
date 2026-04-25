"""Compare onnx-asr vs OpenVINO backends on real audio files.

Usage: python scripts/compare_backends.py /path/to/audio.mp3
"""
import sys
import time
from pathlib import Path
from local_transcriber.transcriber import load_model, _transcribe_file
from local_transcriber.backends import get_backend
from local_transcriber.types import Segment


def transcribe_with_backend(file_path: Path, device: str, model_name: str,
                            compute_type: str, language: str | None) -> tuple[float, int, str]:
    """Run transcription and return (elapsed_sec, segment_count, transcript_text)."""
    start = time.monotonic()
    model_obj, actual_device, backend, model_path = load_model(
        model_name, device, compute_type,
        strict_device=True,
    )
    tfr = _transcribe_file(
        model_obj, actual_device, backend, model_path,
        file_path, model_name, compute_type,
        language=language,
    )
    elapsed = time.monotonic() - start
    text = " ".join(s.text for s in tfr.result.segments)
    return elapsed, len(tfr.result.segments), text


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/compare_backends.py <audio_file>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    models_to_test = [
        ("gigaam-v3", "onnx", "int8"),
        ("parakeet-v3", "onnx", "int8"),
        ("medium", "openvino-cpu", "int8"),
    ]

    print(f"File: {file_path.name} ({file_path.stat().st_size / 1e6:.1f} MB)")
    print()

    for model_name, device, ct in models_to_test:
        print(f"--- {model_name} on {device} (compute={ct}) ---")
        try:
            elapsed, seg_count, text = transcribe_with_backend(
                file_path, device, model_name, ct, language="ru" if "gigaam" in model_name else None,
            )
            print(f"  Time: {elapsed:.1f}s")
            print(f"  Segments: {seg_count}")
            print(f"  Text preview: {text[:200]}...")
            print()
        except Exception as e:
            print(f"  ERROR: {e}")
            print()


if __name__ == "__main__":
    main()
