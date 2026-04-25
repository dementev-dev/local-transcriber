"""Бэкенд транскрипции на основе onnx-asr (GigaAM, Parakeet, FastConformer)."""

from __future__ import annotations

MODEL_ALIASES: dict[str, str] = {
    "gigaam-v3": "gigaam-v3-ctc",
    "parakeet-v3": "nemo-parakeet-tdt-0.6b-v3",
}

SUPPORTED_ALIASES = ", ".join(MODEL_ALIASES)


class OnnxAsrBackend:
    """Бэкенд транскрипции через onnx-asr (ONNX Runtime)."""

    def _resolve_model(self, model_name: str) -> str:
        """Resolve alias to onnx-asr model name. Raw names pass through."""
        if model_name in MODEL_ALIASES:
            return MODEL_ALIASES[model_name]
        if "/" in model_name or model_name.count("-") >= 2:
            # Looks like a raw onnx-asr name — allow passthrough
            return model_name
        raise ValueError(
            f"Неподдерживаемая модель '{model_name}'. "
            f"Доступные алиасы: {SUPPORTED_ALIASES}. "
            f"Либо укажите полное имя модели onnx-asr."
        )
