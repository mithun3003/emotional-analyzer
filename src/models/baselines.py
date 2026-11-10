from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from src.features.extract_features import extract_and_cache_features
from src.utils import get_device, get_logger, load_config, set_seed

logger = get_logger("baselines")


def compute_class_weights(labels: torch.Tensor, num_classes: int = 7) -> torch.Tensor:
    """Compute normalized inverse class frequency weights for cost-sensitive loss."""
    counts = torch.bincount(labels, minlength=num_classes).float()
    total = labels.numel()
    weights = total / (num_classes * torch.clamp(counts, min=1.0))
    weights = weights / weights.mean() # normalize around 1.0
    return weights


class SimpleMLP(nn.Module):
    """Linear/MLP classification head for baseline evaluation."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 7, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def extract_flat_arrays(
    split_dict: Dict[Tuple[int, int], Dict[str, Any]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract flat numpy arrays for Text, Audio, Video, and Emotion labels."""
    utts = list(split_dict.values())
    text_arr = np.stack([u["text"].numpy() for u in utts])
    audio_arr = np.stack([u["audio"].numpy() for u in utts])
    video_arr = np.stack([u["video"].numpy() for u in utts])
    labels_arr = np.array([u["emotion_idx"] for u in utts], dtype=np.int64)
    return text_arr, audio_arr, video_arr, labels_arr


def train_eval_mlp_baseline(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    model_name: str,
    class_weights: torch.Tensor,
    epochs: int = 20,
    lr: float = 0.001,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    input_dim = X_tr.shape[1]
    model = SimpleMLP(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    tr_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True,
    )

    best_dev_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        # Dev evaluation for early stopping / epoch selection
        model.eval()
        with torch.no_grad():
            dev_logits = model(torch.tensor(X_dev, dtype=torch.float32).to(device))
            dev_preds = dev_logits.argmax(dim=-1).cpu().numpy()
            dev_f1 = f1_score(y_dev, dev_preds, average="weighted", zero_division=0)

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best dev checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on dev
    model.eval()
    with torch.no_grad():
        dev_logits = model(torch.tensor(X_dev, dtype=torch.float32).to(device))
        dev_preds = dev_logits.argmax(dim=-1).cpu().numpy()
        test_logits = model(torch.tensor(X_te, dtype=torch.float32).to(device))
        test_preds = test_logits.argmax(dim=-1).cpu().numpy()

    return {
        "model": model_name,
        "dev_acc": round(float(accuracy_score(y_dev, dev_preds)), 4),
        "dev_weighted_f1": round(float(f1_score(y_dev, dev_preds, average="weighted", zero_division=0)), 4),
        "dev_macro_f1": round(float(f1_score(y_dev, dev_preds, average="macro", zero_division=0)), 4),
        "test_acc": round(float(accuracy_score(y_te, test_preds)), 4),
        "test_weighted_f1": round(float(f1_score(y_te, test_preds, average="weighted", zero_division=0)), 4),
        "test_macro_f1": round(float(f1_score(y_te, test_preds, average="macro", zero_division=0)), 4),
        "test_precision": round(float(precision_score(y_te, test_preds, average="weighted", zero_division=0)), 4),
        "test_recall": round(float(recall_score(y_te, test_preds, average="weighted", zero_division=0)), 4),
    }


def run_all_baselines(config_path: str = "configs/default.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    set_seed(config["train"]["seed"])
    device = get_device(config["train"].get("device", "auto"))
    metrics_dir = Path(config["outputs"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    data_splits = extract_and_cache_features(config_path=config_path)
    X_tr_t, X_tr_a, X_tr_v, y_tr = extract_flat_arrays(data_splits["train"])
    X_dev_t, X_dev_a, X_dev_v, y_dev = extract_flat_arrays(data_splits["dev"])
    X_te_t, X_te_a, X_te_v, y_te = extract_flat_arrays(data_splits["test"])

    class_weights = compute_class_weights(torch.tensor(y_tr, dtype=torch.long), num_classes=7)

    results: List[Dict[str, Any]] = []

    # 1. Majority class baseline
    logger.info("Evaluating Baseline 1: Majority Class...")
    majority_class = int(np.bincount(y_tr).argmax())
    dev_maj_pred = np.full_like(y_dev, majority_class)
    test_maj_pred = np.full_like(y_te, majority_class)

    results.append({
        "model": "Majority",
        "modalities": "-",
        "context": "No",
        "dev_acc": round(float(accuracy_score(y_dev, dev_maj_pred)), 4),
        "dev_weighted_f1": round(float(f1_score(y_dev, dev_maj_pred, average="weighted", zero_division=0)), 4),
        "dev_macro_f1": round(float(f1_score(y_dev, dev_maj_pred, average="macro", zero_division=0)), 4),
        "test_acc": round(float(accuracy_score(y_te, test_maj_pred)), 4),
        "test_weighted_f1": round(float(f1_score(y_te, test_maj_pred, average="weighted", zero_division=0)), 4),
        "test_macro_f1": round(float(f1_score(y_te, test_maj_pred, average="macro", zero_division=0)), 4),
        "test_precision": round(float(precision_score(y_te, test_maj_pred, average="weighted", zero_division=0)), 4),
        "test_recall": round(float(recall_score(y_te, test_maj_pred, average="weighted", zero_division=0)), 4),
    })

    # 2. Text-only (T)
    logger.info("Evaluating Baseline 2: Text-only (T)...")
    res_t = train_eval_mlp_baseline(
        X_tr_t, y_tr, X_dev_t, y_dev, X_te_t, y_te,
        model_name="BERT (Text-only)", class_weights=class_weights, device=device
    )
    res_t.update({"modalities": "T", "context": "No"})
    results.append(res_t)

    # 3. Audio-only (A)
    logger.info("Evaluating Baseline 3: Audio-only (A)...")
    res_a = train_eval_mlp_baseline(
        X_tr_a, y_tr, X_dev_a, y_dev, X_te_a, y_te,
        model_name="Audio-only", class_weights=class_weights, device=device
    )
    res_a.update({"modalities": "A", "context": "No"})
    results.append(res_a)

    # 4. Video-only (V)
    logger.info("Evaluating Baseline 4: Video-only (V)...")
    res_v = train_eval_mlp_baseline(
        X_tr_v, y_tr, X_dev_v, y_dev, X_te_v, y_te,
        model_name="Video-only", class_weights=class_weights, device=device
    )
    res_v.update({"modalities": "V", "context": "No"})
    results.append(res_v)

    # 5. Concat (T+A)
    logger.info("Evaluating Baseline 5: Concat (T+A)...")
    X_tr_ta = np.hstack([X_tr_t, X_tr_a])
    X_dev_ta = np.hstack([X_dev_t, X_dev_a])
    X_te_ta = np.hstack([X_te_t, X_te_a])
    res_ta = train_eval_mlp_baseline(
        X_tr_ta, y_tr, X_dev_ta, y_dev, X_te_ta, y_te,
        model_name="Concat (T+A)", class_weights=class_weights, device=device
    )
    res_ta.update({"modalities": "T+A", "context": "No"})
    results.append(res_ta)

    # 6. Concat (T+A+V)
    logger.info("Evaluating Baseline 6: Concat (T+A+V)...")
    X_tr_tav = np.hstack([X_tr_t, X_tr_a, X_tr_v])
    X_dev_tav = np.hstack([X_dev_t, X_dev_a, X_dev_v])
    X_te_tav = np.hstack([X_te_t, X_te_a, X_te_v])
    res_tav = train_eval_mlp_baseline(
        X_tr_tav, y_tr, X_dev_tav, y_dev, X_te_tav, y_te,
        model_name="Concat (T+A+V)", class_weights=class_weights, device=device
    )
    res_tav.update({"modalities": "T+A+V", "context": "No"})
    results.append(res_tav)

    df_results = pd.DataFrame(results)
    out_csv = metrics_dir / "baselines.csv"
    df_results.to_csv(out_csv, index=False)
    logger.info(f"Baseline results saved to {out_csv}:")
    logger.info("\n" + df_results.to_string(index=False))

    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    run_all_baselines(config_path=args.config)
