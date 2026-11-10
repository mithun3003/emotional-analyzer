from __future__ import annotations

from typing import Any, Dict
import torch


def compute_modality_importance(
    modality_weights: torch.Tensor,
) -> Dict[str, float]:
    """Compute normalized percentage contributions across Text, Audio, Video."""
    weights = modality_weights.detach().cpu().numpy()
    total = max(1e-6, weights.sum())
    pcts = (weights / total * 100).round(1)
    return {
        "text_percent": float(pcts[0]),
        "audio_percent": float(pcts[1]),
        "video_percent": float(pcts[2]),
    }
