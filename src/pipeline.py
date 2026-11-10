from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split
except ModuleNotFoundError as exc:
    raise SystemExit(
        "scikit-learn is required. Install it with: pip install -r requirements.txt"
    ) from exc

DATA_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = DATA_DIR / "train_sent_emo.csv"
TEXT_EMBEDDINGS_PATH = DATA_DIR / "text_embeddings.pkl"
TEXT_REDUCED_PATH = DATA_DIR / "text_encodings_reduced.pkl"
AUDIO_EMBEDDINGS_PATH = DATA_DIR / "audi_embeddings.pkl"
CONFUSION_MATRIX_PATH = DATA_DIR / "confusion_matrix.csv"


def _load_embedding_dataset() -> tuple[np.ndarray, List[str]]:
    with TEXT_EMBEDDINGS_PATH.open("rb") as handle:
        text_embeddings = pickle.load(handle)
    with TEXT_REDUCED_PATH.open("rb") as handle:
        reduced_embeddings = pickle.load(handle)
    with AUDIO_EMBEDDINGS_PATH.open("rb") as handle:
        audio_embeddings = pickle.load(handle)

    if not (len(text_embeddings) == len(reduced_embeddings) == len(audio_embeddings)):
        raise ValueError("The embedding pickle files do not contain the same number of rows")

    frame = pd.read_csv(CSV_PATH)
    frame = frame[["Utterance", "Emotion"]].dropna()
    frame = frame[frame["Utterance"].astype(str).str.strip() != ""]
    frame = frame.iloc[: len(text_embeddings)].copy()

    text_vectors = np.asarray(text_embeddings, dtype=np.float32)
    reduced_vectors = np.asarray(reduced_embeddings, dtype=np.float32)
    audio_vectors = np.asarray(audio_embeddings, dtype=np.float32)

    # Use the audio embeddings directly alongside reduced text embeddings.
    features = np.hstack([audio_vectors, reduced_vectors])
    labels = frame["Emotion"].astype(str).tolist()
    return features, labels


def _save_confusion_matrix(labels: List[str], predictions: List[str]) -> List[List[int]]:
    label_set = sorted(set(labels))
    matrix = confusion_matrix(labels, predictions, labels=label_set)

    with CONFUSION_MATRIX_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", *label_set])
        for index, row in enumerate(matrix):
            writer.writerow([label_set[index], *row.tolist()])

    return matrix.tolist()


def run_step_by_step(verbose: bool = True) -> Dict[str, Any]:
    steps: List[str] = []

    features, labels = _load_embedding_dataset()
    steps.append("Loaded embeddings and labels from the provided pickle files")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    steps.append("Split the dataset into train and test sets")

    model = ExtraTreesClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    steps.append("Trained an Extra Trees classifier on the audio and reduced-text embedding features")

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    confusion = _save_confusion_matrix(y_test, predictions)
    steps.append("Evaluated performance and saved a confusion matrix")

    if verbose:
        print("Embedding-based pipeline")
        for index, step in enumerate(steps, 1):
            print(f"{index}. {step}")
        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Test macro F1: {macro_f1:.4f}")
        print(classification_report(y_test, predictions, zero_division=0))
        print("Confusion matrix saved to", CONFUSION_MATRIX_PATH)

    return {
        "steps": steps,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classification_report": classification_report(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion,
        "labels": sorted(set(labels)),
    }
