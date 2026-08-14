"""Воспроизводимый свип параметров офлайн-диаризации sherpa-onnx."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import subprocess
import time
import wave
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sherpa_onnx

TURN_RE = re.compile(r"^\*\*\[(\d{2}):(\d{2})(?::(\d{2}))?\] Speaker (\d+):\*\*")


@dataclass(frozen=True)
class Recording:
    name: str
    path: Path
    start: float
    duration: float
    expected_speakers: int
    reference: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def decode_clip(recording: Recording, work_dir: Path) -> Path:
    output = work_dir / f"{recording.name}.wav"
    if output.exists():
        return output

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(recording.start),
        "-t",
        str(recording.duration),
        "-i",
        str(recording.path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output),
    ]
    subprocess.run(command, check=True)
    return output


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError(f"Ожидался mono PCM16 WAV: {path}")
        if source.getframerate() != 16000:
            raise ValueError(f"Ожидалась частота 16 кГц: {path}")
        samples = np.frombuffer(source.readframes(source.getnframes()), np.int16)
    return samples.astype(np.float32) / 32768.0


def timestamp_seconds(match: re.Match[str]) -> float:
    first, second, third = match.group(1), match.group(2), match.group(3)
    if third is None:
        return int(first) * 60 + int(second)
    return int(first) * 3600 + int(second) * 60 + int(third)


def read_reference_turns(recording: Recording) -> list[dict[str, Any]]:
    if recording.reference is None:
        return []

    starts: list[tuple[float, str]] = []
    for line in recording.reference.read_text(encoding="utf-8").splitlines():
        match = TURN_RE.match(line)
        if match:
            starts.append((timestamp_seconds(match), match.group(4)))

    clip_end = recording.start + recording.duration
    turns: list[dict[str, Any]] = []
    for index, (start, speaker) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else clip_end
        overlap_start = max(start, recording.start)
        overlap_end = min(end, clip_end)
        if overlap_end > overlap_start:
            turns.append(
                {
                    "speaker": speaker,
                    "start": overlap_start - recording.start,
                    "end": overlap_end - recording.start,
                }
            )
    return turns


def interval_overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
    return max(0.0, min(left["end"], right["end"]) - max(left["start"], right["start"]))


def best_mapping(
    segments: list[dict[str, Any]],
    reference_turns: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not reference_turns or not segments:
        return None

    predicted = sorted({str(segment["speaker"]) for segment in segments})
    reference = sorted({str(turn["speaker"]) for turn in reference_turns})
    overlap: defaultdict[tuple[str, str], float] = defaultdict(float)
    total = 0.0
    for segment in segments:
        predicted_speaker = str(segment["speaker"])
        for turn in reference_turns:
            value = interval_overlap(segment, turn)
            if value:
                reference_speaker = str(turn["speaker"])
                overlap[(predicted_speaker, reference_speaker)] += value
                total += value

    best_score = -1.0
    best_pairs: list[tuple[str, str]] = []
    if len(predicted) >= len(reference):
        for candidate in itertools.permutations(predicted, len(reference)):
            pairs = list(zip(candidate, reference, strict=True))
            score = sum(overlap[pair] for pair in pairs)
            if score > best_score:
                best_score, best_pairs = score, pairs
    else:
        for candidate in itertools.permutations(reference, len(predicted)):
            pairs = list(zip(predicted, candidate, strict=True))
            score = sum(overlap[pair] for pair in pairs)
            if score > best_score:
                best_score, best_pairs = score, pairs

    return {
        "mapped_speaker_purity": best_score / total if total else None,
        "mapped_overlap_seconds": best_score,
        "total_overlap_seconds": total,
        "mapping": {predicted: reference for predicted, reference in best_pairs},
    }


def make_config(
    segmentation_model: Path,
    embedding_model: Path,
    threshold: float,
    num_clusters: int,
    threads: int,
) -> sherpa_onnx.OfflineSpeakerDiarizationConfig:
    pyannote = sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
        model=str(segmentation_model)
    )
    segmentation = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
        pyannote=pyannote,
        num_threads=threads,
    )
    embedding = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(embedding_model),
        num_threads=threads,
    )
    clustering = sherpa_onnx.FastClusteringConfig(
        num_clusters=num_clusters,
        threshold=threshold,
    )
    return sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=segmentation,
        embedding=embedding,
        clustering=clustering,
    )


def summarize_segments(
    segments: list[dict[str, Any]],
    recording: Recording,
) -> dict[str, Any]:
    durations: defaultdict[str, float] = defaultdict(float)
    for segment in segments:
        durations[str(segment["speaker"])] += segment["end"] - segment["start"]

    ordered = sorted(durations.items(), key=lambda item: item[1], reverse=True)
    total = sum(durations.values())
    residual = sum(duration for _, duration in ordered[recording.expected_speakers :])
    substantial_threshold = max(5.0, recording.duration * 0.02)
    return {
        "clusters": len(ordered),
        "substantial_clusters": sum(
            duration >= substantial_threshold for _, duration in ordered
        ),
        "substantial_threshold_seconds": substantial_threshold,
        "cluster_durations_seconds": dict(ordered),
        "speaker_time_seconds": total,
        "residual_seconds_after_expected": residual,
        "residual_share_after_expected": residual / total if total else None,
    }


def save_output(path: Path, output: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def manifest_shape(manifest: dict[str, Any]) -> dict[str, Any]:
    """Отделить параметры эксперимента от машинно-зависимых путей."""
    return {
        "models": [item["name"] for item in manifest["models"]],
        "recordings": [
            {
                key: item[key]
                for key in ("name", "start", "duration", "expected_speakers")
            }
            for item in manifest["recordings"]
        ],
        "runs": manifest["runs"],
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.work_dir.mkdir(parents=True, exist_ok=True)

    recordings = [
        Recording(
            name=item["name"],
            path=Path(item["path"]),
            start=float(item["start"]),
            duration=float(item["duration"]),
            expected_speakers=int(item["expected_speakers"]),
            reference=Path(item["reference"]) if item.get("reference") else None,
        )
        for item in manifest["recordings"]
    ]
    if args.output.exists():
        output = json.loads(args.output.read_text(encoding="utf-8"))
        if (
            manifest_shape(output["manifest"]) != manifest_shape(manifest)
            or output["threads"] != args.threads
        ):
            raise ValueError("Существующий output создан с другим manifest/threads")
        output["manifest"] = manifest
    else:
        output = {
            "manifest": manifest,
            "sherpa_onnx_version": sherpa_onnx.__version__,
            "threads": args.threads,
            "results": [],
        }
    completed = {
        (item["recording"], item["model"], item["run"]) for item in output["results"]
    }

    segmentation_model = Path(manifest["segmentation_model"])
    for recording in recordings:
        print(f"Декодирование {recording.name}", flush=True)
        wav_path = decode_clip(recording, args.work_dir)
        samples = read_wav(wav_path)
        reference_turns = read_reference_turns(recording)
        source_hash = file_sha256(recording.path)

        for model in manifest["models"]:
            embedding_model = Path(model["path"])
            for run in manifest["runs"]:
                run_key = (recording.name, model["name"], run["name"])
                if run_key in completed:
                    print(f"Пропуск готового прогона: {run_key}", flush=True)
                    continue
                num_clusters = run["num_clusters"]
                if num_clusters == "expected":
                    num_clusters = recording.expected_speakers
                threshold = float(run["threshold"])
                print(
                    f"{recording.name}: {model['name']} / {run['name']}",
                    flush=True,
                )
                config = make_config(
                    segmentation_model=segmentation_model,
                    embedding_model=embedding_model,
                    threshold=threshold,
                    num_clusters=int(num_clusters),
                    threads=args.threads,
                )
                diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
                started = time.perf_counter()
                result = diarizer.process(samples)
                elapsed = time.perf_counter() - started
                segments = [
                    {
                        "speaker": int(segment.speaker),
                        "start": float(segment.start),
                        "end": float(segment.end),
                    }
                    for segment in result.sort_by_start_time()
                ]
                item = {
                    "recording": recording.name,
                    "source": recording.path.name,
                    "source_sha256": source_hash,
                    "clip_start": recording.start,
                    "clip_duration": recording.duration,
                    "expected_speakers": recording.expected_speakers,
                    "reference": recording.reference.name
                    if recording.reference
                    else None,
                    "model": model["name"],
                    "model_file": embedding_model.name,
                    "run": run["name"],
                    "threshold": threshold,
                    "num_clusters": int(num_clusters),
                    "elapsed_seconds": elapsed,
                    "rtf": elapsed / recording.duration,
                    "summary": summarize_segments(segments, recording),
                    "reference_mapping": best_mapping(segments, reference_turns),
                    "segments": segments,
                }
                output["results"].append(item)
                completed.add(run_key)
                save_output(args.output, output)


if __name__ == "__main__":
    main()
