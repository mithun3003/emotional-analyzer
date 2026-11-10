from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

DATA_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = DATA_DIR / "archive" / "train_sent_emo.csv"


def load_dataset() -> Dict[str, List[Tuple[str, str]]]:
    df = pd.read_csv(DATASET_PATH)
    df = df[["Utterance", "Emotion"]].dropna()
    df = df[df["Utterance"].astype(str).str.strip() != ""]

    texts = df["Utterance"].astype(str).tolist()
    labels = df["Emotion"].astype(str).tolist()

    label_names = sorted(set(labels))
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    encoded_labels = [label_to_id[label] for label in labels]

    total = len(texts)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    splits: Dict[str, List[Tuple[str, str]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for i, (text, label_id) in enumerate(zip(texts, encoded_labels)):
        sample = (text, label_names[label_id])
        if i < train_end:
            splits["train"].append(sample)
        elif i < val_end:
            splits["val"].append(sample)
        else:
            splits["test"].append(sample)

    return splits
