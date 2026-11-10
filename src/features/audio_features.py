from __future__ import annotations

import torch


def extract_audio_placeholder(num_samples: int, audio_dim: int = 128) -> torch.Tensor:
    """Generate audio representation with missing audio mask flag."""
    return torch.zeros(num_samples, audio_dim, dtype=torch.float32)
