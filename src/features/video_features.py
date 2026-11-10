from __future__ import annotations

import torch


def extract_video_placeholder(num_samples: int, video_dim: int = 128) -> torch.Tensor:
    """Generate video representation with missing video mask flag."""
    return torch.zeros(num_samples, video_dim, dtype=torch.float32)
