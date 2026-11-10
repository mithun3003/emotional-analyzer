from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationalGraphConvLayer(nn.Module):
    """Relational Graph Convolutional Network (RGCN) layer for dialogue graphs.

    Handles relation types:
        0: Intra-speaker relation (same speaker)
        1: Inter-speaker relation (different speakers)
    """

    def __init__(self, d_model: int = 128, num_relations: int = 2, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.num_relations = num_relations

        self.w_self = nn.Linear(d_model, d_model, bias=False)
        self.w_rel = nn.Parameter(torch.empty(num_relations, d_model, d_model))
        nn.init.xavier_uniform_(self.w_rel)

        self.bias = nn.Parameter(torch.zeros(d_model))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """Args:

            x: [N, d_model] node features
            edge_index: [2, E] (src, dst)
            edge_type: [E] relation index (0 or 1)

        Returns:
            out: [N, d_model] updated node representations
        """
        N = x.size(0)
        out = self.w_self(x) # self-loop contribution

        if edge_index.size(1) > 0:
            src, dst = edge_index[0], edge_index[1]

            for r in range(self.num_relations):
                rel_mask = (edge_type == r)
                if not rel_mask.any():
                    continue

                r_src = src[rel_mask]
                r_dst = dst[rel_mask]

                # Project source features through relation matrix
                msg = torch.matmul(x[r_src], self.w_rel[r]) # [E_r, d_model]

                # Aggregate messages into destination nodes with in-degree normalization
                deg = torch.bincount(r_dst, minlength=N).float().unsqueeze(1).clamp(min=1.0)
                agg = torch.zeros(N, self.d_model, device=x.device, dtype=x.dtype)
                agg.index_add_(0, r_dst, msg)
                out = out + (agg / deg)

        out = self.norm(x + self.dropout(F.relu(out + self.bias)))
        return out


class GraphTransformerLayer(nn.Module):
    """Graph Transformer layer with relational attention over conversational context.

    Computes per-edge attention weights between target utterance and conversational neighbors.
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4, num_relations: int = 2, dropout: float = 0.2):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.rel_embed = nn.Embedding(num_relations, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:

            x: [N, d_model]
            edge_index: [2, E]
            edge_type: [E]

        Returns:
            out: [N, d_model]
            edge_attn: [E] attention weight for each edge (for explainability)
        """
        N = x.size(0)
        E = edge_index.size(1)

        if E == 0:
            return x, torch.empty(0, device=x.device)

        src, dst = edge_index[0], edge_index[1]

        # Multi-head representations
        q = self.w_q(x).view(N, self.n_heads, self.d_k)
        k = self.w_k(x).view(N, self.n_heads, self.d_k)
        v = self.w_v(x).view(N, self.n_heads, self.d_k)

        # Relational bias
        rel_bias = self.rel_embed(edge_type).view(E, self.n_heads, self.d_k)

        # Compute dot-product attention per edge
        q_dst = q[dst]                           # [E, n_heads, d_k]
        k_src = k[src] + rel_bias                # [E, n_heads, d_k]
        v_src = v[src]                           # [E, n_heads, d_k]

        scores = (q_dst * k_src).sum(dim=-1) / (self.d_k ** 0.5) # [E, n_heads]

        # Softmax over incoming edges per destination node
        # Compute max for numerical stability
        max_scores = torch.zeros(N, self.n_heads, device=x.device).fill_(-1e9)
        max_scores.scatter_reduce_(0, dst.unsqueeze(1).expand(-1, self.n_heads), scores, reduce="amax", include_self=False)
        scores_stable = scores - max_scores[dst]
        exp_scores = torch.exp(scores_stable)

        sum_exp = torch.zeros(N, self.n_heads, device=x.device)
        sum_exp.scatter_add_(0, dst.unsqueeze(1).expand(-1, self.n_heads), exp_scores)
        sum_exp = sum_exp.clamp(min=1e-9)

        edge_attn_heads = exp_scores / sum_exp[dst] # [E, n_heads]
        edge_attn = edge_attn_heads.mean(dim=-1)    # [E] averaged over heads

        # Weighted message aggregation
        weights_expanded = self.dropout(edge_attn_heads).unsqueeze(-1) # [E, n_heads, 1]
        msg = (v_src * weights_expanded).view(E, self.d_model)

        out = torch.zeros(N, self.d_model, device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, msg)

        out = self.norm(x + self.out_proj(out))
        return out, edge_attn


class DialogueGraphContext(nn.Module):
    """Complete conversational dialogue graph module: RGCN followed by Graph Transformer."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, num_relations: int = 2, dropout: float = 0.2):
        super().__init__()
        self.rgcn = RelationalGraphConvLayer(d_model=d_model, num_relations=num_relations, dropout=dropout)
        self.gt = GraphTransformerLayer(d_model=d_model, n_heads=n_heads, num_relations=num_relations, dropout=dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_rgcn = self.rgcn(x, edge_index, edge_type)
        h_out, edge_attn = self.gt(h_rgcn, edge_index, edge_type)
        return h_out, edge_attn
