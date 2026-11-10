from __future__ import annotations

from typing import List
import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_text_embeddings(
    texts: List[str], target_dim: int = 768, seed: int = 42
) -> torch.Tensor:
    """Extract 768-d semantic text features from utterance list."""
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X = vec.fit_transform(texts)
    n_comp = min(target_dim, X.shape[1] - 1, len(texts) - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    feats = svd.fit_transform(X)
    if feats.shape[1] < target_dim:
        feats = np.pad(feats, ((0, 0), (0, target_dim - feats.shape[1])))
    return torch.tensor(feats, dtype=torch.float32)
