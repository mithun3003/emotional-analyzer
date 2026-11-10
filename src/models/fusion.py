from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEnhancedAttention(nn.Module):
    """Text-enhanced cross-modal multi-head attention module (CIME-inspired).

    Text serves as Query, while concatenated [Text; Audio; Video] serve as Key and Value.
    Returns fused multimodal representation and modality attention distribution.
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        video_feat: torch.Tensor,
        missing_audio: Optional[torch.Tensor] = None,
        missing_video: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:

            text_feat: [N, d_model]
            audio_feat: [N, d_model]
            video_feat: [N, d_model]
            missing_audio: [N] bool mask
            missing_video: [N] bool mask

        Returns:
            fused_feat: [N, d_model]
            modality_weights: [N, 3] (Text, Audio, Video percentage contributions)
        """
        N = text_feat.size(0)

        # Q: [N, 1, d_model]
        q = text_feat.unsqueeze(1)
        # K, V: [N, 3, d_model]
        kv = torch.stack([text_feat, audio_feat, video_feat], dim=1)

        # Multi-head projection
        # [N, 1, n_heads, d_k] -> [N, n_heads, 1, d_k]
        q_proj = self.w_q(q).view(N, 1, self.n_heads, self.d_k).transpose(1, 2)
        # [N, 3, n_heads, d_k] -> [N, n_heads, 3, d_k]
        k_proj = self.w_k(kv).view(N, 3, self.n_heads, self.d_k).transpose(1, 2)
        v_proj = self.w_v(kv).view(N, 3, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores: [N, n_heads, 1, 3]
        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / (self.d_k ** 0.5)

        # Modality masking
        mask = torch.zeros(N, 1, 1, 3, device=text_feat.device, dtype=torch.float32)
        if missing_audio is not None and missing_audio.any():
            mask[missing_audio, :, :, 1] = -1e9
        if missing_video is not None and missing_video.any():
            mask[missing_video, :, :, 2] = -1e9

        scores = scores + mask
        attn_weights = F.softmax(scores, dim=-1) # [N, n_heads, 1, 3]
        attn_weights_dropped = self.dropout(attn_weights)

        # Context vector: [N, n_heads, 1, d_k] -> [N, 1, d_model] -> [N, d_model]
        context = torch.matmul(attn_weights_dropped, v_proj)
        context = context.transpose(1, 2).contiguous().view(N, self.d_model)

        # Residual connection + LayerNorm
        fused = self.norm1(text_feat + self.out_proj(context))
        # FeedForward + Residual + LayerNorm
        fused = self.norm2(fused + self.ffn(fused))

        # Modality weights averaged over heads: [N, 3] (0: Text, 1: Audio, 2: Video)
        modality_weights = attn_weights.squeeze(2).mean(dim=1)

        return fused, modality_weights
