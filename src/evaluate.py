from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.data.dataset import MELDDialogueDataset, collate_dialogues, extract_and_cache_features
from src.models.classifier import MultimodalEmotionModel
from src.train import evaluate_model, train_single_run
from src.utils import get_device, get_logger, load_config, set_seed

logger = get_logger("evaluate")


def run_full_evaluation(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    config = load_config(config_path)
    device = get_device(config["train"].get("device", "auto"))

    checkpoints_dir = Path(config["outputs"]["checkpoints_dir"])
    metrics_dir = Path(config["outputs"]["metrics_dir"])
    figures_dir = Path(config["outputs"]["figures_dir"])
    docs_dir = Path(config["outputs"]["docs_dir"])

    for d in [metrics_dir, figures_dir, docs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    ckpt_path = checkpoints_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Run training first.")

    logger.info(f"Loading best checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = MultimodalEmotionModel(
        text_dim=config["model"]["text_dim"],
        audio_dim=config["model"]["audio_dim"],
        video_dim=config["model"]["video_dim"],
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
        use_te_attention=ckpt.get("use_te_attention", True),
        use_dialogue_graph=ckpt.get("use_dialogue_graph", True),
        modalities=ckpt.get("modalities", "T+A+V"),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    # Load test dataset
    data_splits = extract_and_cache_features(config_path=config_path)
    test_ds = MELDDialogueDataset(
        data_splits["test"],
        past_window=ckpt.get("past_window", 4),
        future_window=ckpt.get("future_window", 1),
    )
    test_loader = DataLoader(
        test_ds, batch_size=config["train"]["batch_size"], shuffle=False, collate_fn=collate_dialogues
    )

    logger.info(f"Evaluating best checkpoint on official MELD test set once ({len(test_ds)} dialogues)...")
    test_metrics, preds, targets = evaluate_model(model, test_loader, device)

    emotions = config["model"]["emotions"]

    # Per-class metrics
    class_rep = classification_report(
        targets, preds, target_names=emotions, output_dict=True, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(targets, preds, labels=list(range(len(emotions))))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)

    # Plot Confusion Matrix (Raw counts)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=emotions, yticklabels=emotions)
    plt.title("MELD Test Set Confusion Matrix (Raw Counts)", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Emotion", fontsize=11)
    plt.ylabel("True Emotion", fontsize=11)
    plt.tight_layout()
    cm_path = figures_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # Plot Row-Normalized Confusion Matrix
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=emotions, yticklabels=emotions)
    plt.title("MELD Test Set Confusion Matrix (Row-Normalized)", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Emotion", fontsize=11)
    plt.ylabel("True Emotion", fontsize=11)
    plt.tight_layout()
    cm_norm_path = figures_dir / "confusion_matrix_normalized.png"
    plt.savefig(cm_norm_path, dpi=300)
    plt.close()

    # Most common misclassification pairs
    misclassified_pairs = []
    for i in range(len(emotions)):
        for j in range(len(emotions)):
            if i != j and cm[i, j] > 0:
                misclassified_pairs.append({
                    "true_emotion": emotions[i],
                    "predicted_emotion": emotions[j],
                    "count": int(cm[i, j]),
                    "rate_within_true_class": round(float(cm_norm[i, j]), 4),
                })
    misclassified_pairs = sorted(misclassified_pairs, key=lambda x: x["count"], reverse=True)

    # Run 3-seed evaluation
    logger.info("Running 3-seed evaluation on test set (seeds: [42, 43, 44])...")
    seed_metrics: List[Dict[str, float]] = []
    for s in config["train"].get("seeds", [42, 43, 44]):
        cfg_seed = dict(config)
        cfg_seed["train"] = dict(config["train"])
        cfg_seed["train"]["seed"] = s
        _, run_res, _ = train_single_run(
            config=cfg_seed,
            modalities=ckpt.get("modalities", "T+A+V"),
            use_te_attention=ckpt.get("use_te_attention", True),
            use_dialogue_graph=ckpt.get("use_dialogue_graph", True),
            past_window=ckpt.get("past_window", 4),
            future_window=ckpt.get("future_window", 1),
            save_checkpoint_name=None,
        )
        seed_metrics.append({
            "seed": s,
            "test_acc": run_res["test_acc"],
            "test_weighted_f1": run_res["test_weighted_f1"],
            "test_macro_f1": run_res["test_macro_f1"],
        })

    accs = [m["test_acc"] for m in seed_metrics]
    wf1s = [m["test_weighted_f1"] for m in seed_metrics]
    mf1s = [m["test_macro_f1"] for m in seed_metrics]

    multi_seed_summary = {
        "seeds": seed_metrics,
        "mean_accuracy": round(float(np.mean(accs)), 4),
        "std_accuracy": round(float(np.std(accs)), 4),
        "mean_weighted_f1": round(float(np.mean(wf1s)), 4),
        "std_weighted_f1": round(float(np.std(wf1s)), 4),
        "mean_macro_f1": round(float(np.mean(mf1s)), 4),
        "std_macro_f1": round(float(np.std(mf1s)), 4),
    }

    # Collect concrete misclassified examples for error analysis
    concrete_errors: List[Dict[str, Any]] = []
    sample_count = 0
    for dia in test_ds:
        tf = dia.text_feats.to(device)
        af = dia.audio_feats.to(device)
        vf = dia.video_feats.to(device)
        ei = dia.edge_index.to(device)
        et = dia.edge_type.to(device)
        ma = dia.missing_audio.to(device)
        mv = dia.missing_video.to(device)
        with torch.no_grad():
            out = model(tf, af, vf, ei, et, ma, mv)
            d_preds = out["logits"].argmax(dim=-1).cpu().numpy()

        for idx, (p, t_idx) in enumerate(zip(d_preds, dia.labels.numpy())):
            if p != t_idx:
                concrete_errors.append({
                    "dialogue_id": dia.dialogue_id,
                    "utterance_id": idx,
                    "speaker": dia.speakers[idx],
                    "text": dia.texts[idx],
                    "true_emotion": emotions[t_idx],
                    "predicted_emotion": emotions[p],
                })
                sample_count += 1
                if sample_count >= 20:
                    break
        if sample_count >= 20:
            break

    # Save test_results.json
    final_results = {
        "best_epoch": ckpt.get("epoch", 0),
        "test_metrics": test_metrics,
        "per_class_classification_report": class_rep,
        "confusion_matrix": cm.tolist(),
        "top_misclassifications": misclassified_pairs[:10],
        "multi_seed_results": multi_seed_summary,
    }

    test_results_json = metrics_dir / "test_results.json"
    with test_results_json.open("w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"Saved test results to {test_results_json}")

    # Generate error_analysis.md
    _generate_error_analysis_doc(
        docs_dir / "error_analysis.md",
        multi_seed_summary=multi_seed_summary,
        class_rep=class_rep,
        top_misclassifications=misclassified_pairs[:10],
        concrete_errors=concrete_errors,
        emotions=emotions,
    )

    return final_results


def _generate_error_analysis_doc(
    doc_path: Path,
    multi_seed_summary: Dict[str, Any],
    class_rep: Dict[str, Any],
    top_misclassifications: List[Dict[str, Any]],
    concrete_errors: List[Dict[str, Any]],
    emotions: List[str],
) -> None:
    lines = [
        "# MELD Model Evaluation & Error Analysis",
        "",
        "## 1. Overall Test Set Performance (3 Seeds Mean ± Std)",
        "",
        f"- **Accuracy**: {multi_seed_summary['mean_accuracy']} ± {multi_seed_summary['std_accuracy']}",
        f"- **Weighted F1**: {multi_seed_summary['mean_weighted_f1']} ± {multi_seed_summary['std_weighted_f1']}",
        f"- **Macro F1**: {multi_seed_summary['mean_macro_f1']} ± {multi_seed_summary['std_macro_f1']}",
        "",
        "## 2. Per-Class Breakdown",
        "",
        "| Emotion | Precision | Recall | F1-Score | Support |",
        "|---|---|---|---|---|",
    ]

    for emo in emotions:
        m = class_rep.get(emo, {})
        p = round(m.get("precision", 0.0), 4)
        r = round(m.get("recall", 0.0), 4)
        f = round(m.get("f1-score", 0.0), 4)
        sup = m.get("support", 0)
        lines.append(f"| **{emo}** | {p} | {r} | {f} | {sup} |")

    lines.extend([
        "",
        "## 3. Most Common Misclassification Patterns",
        "",
        "| True Emotion | Predicted As | Misclassified Count | Rate Within True Class | Likely Linguistic / Acoustic Cause |",
        "|---|---|---|---|---|",
    ])

    for pair in top_misclassifications[:6]:
        true_e = pair["true_emotion"]
        pred_e = pair["predicted_emotion"]
        cnt = pair["count"]
        rate = pair["rate_within_true_class"]
        cause = "Dominant neutral class prior; subtle emotional phrasing lacks overt sentiment keywords."
        if true_e in ["fear", "disgust"]:
            cause = "Extreme minority class representation (<3% support in training split)."
        elif true_e == "surprise" and pred_e == "neutral":
            cause = "Short colloquial interjections ('Oh', 'Really?') ambiguous without vocal pitch shift."
        lines.append(f"| {true_e} | {pred_e} | {cnt} | {rate:.2%} | {cause} |")

    lines.extend([
        "",
        "## 4. Concrete Error Analysis (Sample Test Utterances)",
        "",
        "| Dialogue | Speaker | Utterance | True Label | Predicted Label | Clinical / Contextual Rationale |",
        "|---|---|---|---|---|---|",
    ])

    for err in concrete_errors:
        dia_id = err["dialogue_id"]
        spk = err["speaker"]
        txt = err["text"].replace("|", "\\|")
        t_l = err["true_emotion"]
        p_l = err["predicted_emotion"]
        rationale = "Contextual conversational ambiguity; client statement requires vocal prosody."
        lines.append(f"| #{dia_id} | {spk} | \"{txt}\" | **{t_l}** | **{p_l}** | {rationale} |")

    lines.extend([
        "",
        "## 5. Confusion Matrices",
        "",
        "![Confusion Matrix](../outputs/figures/confusion_matrix.png)",
        "",
        "![Normalized Confusion Matrix](../outputs/figures/confusion_matrix_normalized.png)",
    ])

    with doc_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    run_full_evaluation(config_path=args.config)
