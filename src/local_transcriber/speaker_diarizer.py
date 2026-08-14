"""Адаптер офлайн-диаризации через sherpa-onnx."""

import shutil
import tarfile
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Any

from .types import DiarizationRun, SpeakerInterval, StatusCallback

_SAMPLE_RATE = 16_000
_CLUSTERING_THRESHOLD = 0.89
_SEGMENTATION_FILENAME = "pyannote-segmentation-3.0.onnx"
_EMBEDDING_FILENAME = "wespeaker_en_voxceleb_resnet34_LM.onnx"
_SEGMENTATION_SHA256 = (
    "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079"
)
_EMBEDDING_SHA256 = "e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012"
_SEGMENTATION_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/"
    "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
_SEGMENTATION_ARCHIVE_MEMBER = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
_EMBEDDING_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx"
)


class SpeakerDiarizer:
    """Переиспользуемый в пределах команды диаризатор."""

    def __init__(self, engine: Any):
        self._engine = engine

    def process(
        self,
        file_path: Path,
        on_status: StatusCallback = None,
    ) -> DiarizationRun:
        """Строит разметку говорящих для одного файла."""
        from faster_whisper import decode_audio

        if on_status is not None:
            on_status("Загружаю аудио для диаризации...")
        samples = decode_audio(str(file_path), sampling_rate=_SAMPLE_RATE)
        if isinstance(samples, tuple):
            raise TypeError("Декодер неожиданно вернул раздельные стереоканалы")
        if on_status is not None:
            on_status("Определяю говорящих...")

        started = perf_counter()
        if on_status is None:
            result = self._engine.process(samples)
        else:

            def report_progress(processed: int, total: int) -> int:
                on_status(f"Определяю говорящих... {processed} / {total}")
                return 0

            result = self._engine.process(samples, report_progress)
        elapsed = perf_counter() - started
        intervals = [
            SpeakerInterval(
                start=float(segment.start),
                end=float(segment.end),
                cluster=int(segment.speaker),
            )
            for segment in result.sort_by_start_time()
        ]
        return DiarizationRun(intervals=intervals, elapsed_seconds=elapsed)


def load_speaker_diarizer(
    speakers: int | None,
    threads: int = 0,
    on_status: StatusCallback = None,
) -> SpeakerDiarizer:
    """Проверяет модели и создаёт batch-owned диаризатор."""
    import sherpa_onnx
    from huggingface_hub import cached_assets_path

    cache_dir = cached_assets_path(
        library_name="local-transcriber",
        namespace="diarization",
        subfolder="models-v1",
    )
    segmentation_path = cache_dir / _SEGMENTATION_FILENAME
    embedding_path = cache_dir / _EMBEDDING_FILENAME
    _ensure_cached_model(
        segmentation_path,
        _SEGMENTATION_SHA256,
        _SEGMENTATION_URL,
        on_status,
        archive_member=_SEGMENTATION_ARCHIVE_MEMBER,
    )
    _ensure_cached_model(
        embedding_path,
        _EMBEDDING_SHA256,
        _EMBEDDING_URL,
        on_status,
    )

    if on_status is not None:
        on_status("Инициализирую диаризатор...")

    segmentation_kwargs: dict[str, Any] = {
        "pyannote": sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
            model=str(segmentation_path)
        ),
        "provider": "cpu",
    }
    embedding_kwargs: dict[str, Any] = {
        "model": str(embedding_path),
        "provider": "cpu",
    }
    if threads > 0:
        segmentation_kwargs["num_threads"] = threads
        embedding_kwargs["num_threads"] = threads

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            **segmentation_kwargs
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(**embedding_kwargs),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=speakers if speakers is not None else -1,
            threshold=_CLUSTERING_THRESHOLD,
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError("Конфигурация диаризатора недействительна")

    engine = sherpa_onnx.OfflineSpeakerDiarization(config)
    if engine.sample_rate != _SAMPLE_RATE:
        raise RuntimeError(
            f"Диаризатор ожидает частоту {engine.sample_rate} Гц вместо {_SAMPLE_RATE} Гц"
        )
    return SpeakerDiarizer(engine)


def _ensure_cached_model(
    path: Path,
    expected_sha256: str,
    url: str,
    on_status: StatusCallback,
    archive_member: str | None = None,
) -> None:
    if path.is_file() and _file_sha256(path) == expected_sha256:
        return

    import httpx

    path.parent.mkdir(parents=True, exist_ok=True)
    if on_status is not None:
        on_status(f"Скачиваю модель диаризации {path.name}...")

    download_path = _temporary_path(path)
    extracted_path: Path | None = None
    try:
        with (
            httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response,
            download_path.open("wb") as output,
        ):
            response.raise_for_status()
            for chunk in response.iter_bytes():
                output.write(chunk)

        candidate = download_path
        if archive_member is not None:
            extracted_path = _temporary_path(path)
            with tarfile.open(download_path, mode="r:bz2") as archive:
                try:
                    member = archive.getmember(archive_member)
                except KeyError as exc:
                    raise RuntimeError(
                        f"В архиве модели отсутствует {archive_member}"
                    ) from exc
                if not member.isfile():
                    raise RuntimeError(
                        f"Элемент архива модели не является файлом: {archive_member}"
                    )
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Не удалось прочитать {archive_member}")
                with source, extracted_path.open("wb") as output:
                    shutil.copyfileobj(source, output)
            candidate = extracted_path

        actual_sha256 = _file_sha256(candidate)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Контрольная сумма модели {path.name} не совпала: {actual_sha256}"
            )
        candidate.replace(path)
    finally:
        download_path.unlink(missing_ok=True)
        if extracted_path is not None:
            extracted_path.unlink(missing_ok=True)


def _temporary_path(target: Path) -> Path:
    with NamedTemporaryFile(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        return Path(temporary.name)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
