from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.fusion import TextEnhancedAttention
from src.models.graph_context import DialogueGraphContext


class MultimodalEmotionModel(nn.Module):
    """Context-Aware Multimodal Emotion Recognition Model for Conversational Therapy Support.

    Components:
        1. Unimodal Encoders: Projects text (768-d), audio (128-d), and video (128-d) to d_model.
        2. Text-Enhanced Cross-Modal Attention (TE): Fuses modalities with query=text.
        3. Dialogue Graph Context (GED): RGCN + Graph Transformer capturing conversational context.
        4. Classification Head: Predicts 7 emotion classes per utterance.
    """

    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 128,
        video_dim: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        num_classes: int = 7,
        dropout: float = 0.2,
        use_te_attention: bool = True,
        use_dialogue_graph: bool = True,
        modalities: str = "T+A+V", # "T", "T+A", or "T+A+V"
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.use_te_attention = use_te_attention
        self.use_dialogue_graph = use_dialogue_graph
        self.modalities = modalities

        # 1. Unimodal projection encoders
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.video_encoder = nn.Sequential(
            nn.Linear(video_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 2. Text-Enhanced Attention Fusion
        if self.use_te_attention:
            self.fusion = TextEnhancedAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        else:
            # Baseline concatenation projection
            concat_factor = 3 if "V" in modalities else (2 if "A" in modalities else 1)
            self.concat_proj = nn.Sequential(
                nn.Linear(d_model * concat_factor, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        # 3. Dialogue Context Graph (RGCN + Graph Transformer)
        if self.use_dialogue_graph:
            self.graph_context = DialogueGraphContext(
                d_model=d_model, n_heads=n_heads, num_relations=2, dropout=dropout
            )

        # 4. Final Utterance Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self,
        text_feats: torch.Tensor,
        audio_feats: torch.Tensor,
        video_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        missing_audio: Optional[torch.Tensor] = None,
        missing_video: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Forward pass for a dialogue of N sequential utterances.

        Args:
            text_feats: [N, text_dim]
            audio_feats: [N, audio_dim]
            video_feats: [N, video_dim]
            edge_index: [2, E]
            edge_type: [E]
            missing_audio: [N] bool mask
            missing_video: [N] bool mask

        Returns:
            dict containing:
                - logits: [N, num_classes]
                - probs: [N, num_classes]
                - modality_weights: [N, 3] (Text, Audio, Video contributions)
                - edge_attn: [E] attention weights over context edges
                - edge_index: [2, E]
                - node_repr: [N, d_model]
        """
        N = text_feats.size(0)

        # Project modalities
        h_text = self.text_encoder(text_feats)
        h_audio = self.audio_encoder(audio_feats)
        h_video = self.video_encoder(video_feats)

        # Apply modality configuration
        if self.modalities == "T":
            h_audio = torch.zeros_like(h_audio)
            h_video = torch.zeros_like(h_video)
            missing_audio = torch.ones(N, dtype=torch.bool, device=text_feats.device)
            missing_video = torch.ones(N, dtype=torch.bool, device=text_feats.device)
        elif self.modalities == "T+A":
            h_video = torch.zeros_like(h_video)
            missing_video = torch.ones(N, dtype=torch.bool, device=text_feats.device)

        # Modality Fusion
        if self.use_te_attention:
            h_fused, modality_weights = self.fusion(
                h_text, h_audio, h_video, missing_audio=missing_audio, missing_video=missing_video
            )
        else:
            if self.modalities == "T":
                h_fused = h_text
                modality_weights = torch.tensor([[1.0, 0.0, 0.0]], device=text_feats.device).repeat(N, 1)
            elif self.modalities == "T+A":
                h_fused = self.concat_proj(torch.cat([h_text, h_audio], dim=-1))
                modality_weights = torch.tensor([[0.5, 0.5, 0.0]], device=text_feats.device).repeat(N, 1)
            else:
                h_fused = self.concat_proj(torch.cat([h_text, h_audio, h_video], dim=-1))
                modality_weights = torch.tensor([[0.33, 0.33, 0.34]], device=text_feats.device).repeat(N, 1)

        # Context Graph Modeling
        if self.use_dialogue_graph and edge_index.size(1) > 0:
            h_context, edge_attn = self.graph_context(h_fused, edge_index, edge_type)
        else:
            h_context = h_fused
            edge_attn = torch.empty(0, device=text_feats.device)

        # Classification
        logits = self.classifier(h_context)
        probs = F.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "modality_weights": modality_weights,
            "edge_attn": edge_attn,
            "edge_index": edge_index,
            "edge_type": edge_type,
            "node_repr": h_context,
        }
