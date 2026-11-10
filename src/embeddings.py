from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

DATA_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = DATA_DIR / "archive" / "train_sent_emo.csv"
DEFAULT_OUTPUT = DATA_DIR / "audi_embeddings.pkl"


def build_audio_embeddings(output_path: Path | None = None) -> Dict[str, Any]:
    """Create a lightweight embedding artifact from the available text dataset.

    Since the workspace only contains text utterances, this generates deterministic
    audio-style embeddings using a hashing-based vectorizer as a placeholder asset.
    """
    df = pd.read_csv(DATASET_PATH)
    df = df[["Utterance", "Emotion"]].dropna()
    df = df[df["Utterance"].astype(str).str.strip() != ""]

    texts = df["Utterance"].astype(str).tolist()
    labels = df["Emotion"].astype(str).tolist()

    vectorizer = HashingVectorizer(n_features=32, ngram_range=(1, 2), alternate_sign=False)
    embeddings = vectorizer.fit_transform(texts).toarray()

    payload = {
        "embedding_dim": 32,
        "rows": len(texts),
        "embeddings": embeddings,
        "labels": labels,
        "texts": texts,
    }

    target_path = output_path or DEFAULT_OUTPUT
    with target_path.open("wb") as handle:
        pickle.dump(payload, handle)

    return payload
