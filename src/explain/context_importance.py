from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import torch


def trace_context_influences(
    target_idx: int,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    edge_attn: torch.Tensor,
    speakers: List[str],
    texts: List[str],
) -> List[Dict[str, Any]]:
    """Trace and rank earlier conversational utterances by graph attention weight."""
    if edge_attn.numel() == 0:
        return []

    ei_np = edge_index.detach().cpu().numpy()
    et_np = edge_type.detach().cpu().numpy()
    ea_np = edge_attn.detach().cpu().numpy()

    incoming_idx = np.where(ei_np[1] == target_idx)[0]
    influences = []

    for idx in incoming_idx:
        src_utt = int(ei_np[0][idx])
        weight = float(ea_np[idx])
        rel_label = "intra-speaker (same speaker)" if et_np[idx] == 0 else "inter-speaker (other speaker)"
        influences.append({
            "source_utterance_idx": src_utt,
            "source_speaker": speakers[src_utt],
            "source_text": texts[src_utt],
            "relation_type": rel_label,
            "attention_weight": round(weight, 4),
            "is_self": (src_utt == target_idx),
        })

    return sorted(influences, key=lambda x: x["attention_weight"], reverse=True)
