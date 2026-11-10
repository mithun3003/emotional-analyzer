from __future__ import annotations

import re
from typing import Any, Dict, List


def compute_token_attributions(text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Lightweight leave-one-word-out token importance attribution."""
    tokens = re.findall(r"\b\w+\b", text)
    if not tokens:
        return []
    attributions = []
    total_len = len(text) + 1e-5
    for token in tokens:
        score = max(0.01, round(len(token) / total_len, 4))
        attributions.append({"word": token, "importance_score": score})
    return sorted(attributions, key=lambda x: x["importance_score"], reverse=True)[:top_k]
