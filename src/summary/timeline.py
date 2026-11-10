from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.dataset import DialogueSample


def analyze_dialogue_timeline(
    dialogue: DialogueSample,
    predictions: List[str],
    confidences: List[float],
) -> Dict[str, Any]:
    """Extract timeline dynamics, emotion transitions, and per-speaker dominant emotions."""
    records = []
    for idx in range(dialogue.num_utterances):
        records.append({
            "utterance_idx": idx,
            "speaker": dialogue.speakers[idx],
            "text": dialogue.texts[idx],
            "predicted_emotion": predictions[idx],
            "confidence": confidences[idx],
        })

    df = pd.DataFrame(records)

    # 1. Per-speaker dominant emotions
    speaker_stats: Dict[str, Any] = {}
    for spk, group in df.groupby("speaker"):
        counts = group["predicted_emotion"].value_counts().to_dict()
        dominant_emo = group["predicted_emotion"].mode()[0] if not group.empty else "neutral"
        speaker_stats[spk] = {
            "total_utterances": len(group),
            "dominant_emotion": dominant_emo,
            "emotion_counts": counts,
            "avg_confidence": round(float(group["confidence"].mean()), 4),
        }

    # 2. Detect emotion shifts per speaker
    emotion_shifts: List[Dict[str, Any]] = []
    for spk, group in df.groupby("speaker"):
        group_utts = group.sort_values("utterance_idx")
        prev_emo = None
        prev_idx = None
        for _, row in group_utts.iterrows():
            curr_emo = row["predicted_emotion"]
            curr_idx = row["utterance_idx"]
            if prev_emo is not None and curr_emo != prev_emo:
                emotion_shifts.append({
                    "speaker": spk,
                    "from_utterance": prev_idx,
                    "to_utterance": curr_idx,
                    "from_emotion": prev_emo,
                    "to_emotion": curr_emo,
                })
            prev_emo = curr_emo
            prev_idx = curr_idx

    return {
        "dialogue_id": dialogue.dialogue_id,
        "utterances": records,
        "speaker_stats": speaker_stats,
        "emotion_shifts": emotion_shifts,
    }


def plot_emotion_timeline(timeline_data: Dict[str, Any], output_path: Path | str) -> None:
    """Generate visual conversational emotion timeline plot."""
    df = pd.DataFrame(timeline_data["utterances"])
    emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    emo_to_y = {emo: i for i, emo in enumerate(emotions)}

    df["y_val"] = df["predicted_emotion"].map(emo_to_y)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2.5, 1]})

    # Scatter & line timeline
    speakers = df["speaker"].unique()
    palette = sns.color_palette("tab10", len(speakers))
    spk_colors = {s: palette[i] for i, s in enumerate(speakers)}

    for spk in speakers:
        spk_df = df[df["speaker"] == spk]
        ax1.plot(spk_df["utterance_idx"], spk_df["y_val"], linestyle="--", alpha=0.5, color=spk_colors[spk])
        ax1.scatter(
            spk_df["utterance_idx"],
            spk_df["y_val"],
            s=spk_df["confidence"] * 180,
            color=spk_colors[spk],
            label=spk,
            edgecolors="black",
            zorder=3,
        )

    ax1.set_yticks(range(len(emotions)))
    ax1.set_yticklabels([e.capitalize() for e in emotions], fontsize=11)
    ax1.set_xlabel("Utterance Index (Conversational Progression)", fontsize=11)
    ax1.set_ylabel("Predicted Emotion", fontsize=11)
    ax1.set_title(f"Conversational Emotion Progression (Dialogue #{timeline_data['dialogue_id']})", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(title="Speakers", bbox_to_anchor=(1.02, 1), loc="upper left")

    # Emotion shifts markers
    for shift in timeline_data["emotion_shifts"]:
        ax1.annotate(
            "Shift",
            xy=(shift["to_utterance"], emo_to_y[shift["to_emotion"]]),
            xytext=(shift["to_utterance"], emo_to_y[shift["to_emotion"]] + 0.3),
            arrowprops=dict(facecolor="red", arrowstyle="->", lw=1),
            fontsize=8,
            color="darkred",
        )

    # Per-speaker emotion distribution bar chart in lower subplot
    spk_dist = df.groupby(["speaker", "predicted_emotion"]).size().unstack(fill_value=0)
    spk_dist.plot(kind="bar", stacked=True, ax=ax2, colormap="viridis", alpha=0.85)
    ax2.set_title("Per-Speaker Emotion Breakdown", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Speaker", fontsize=10)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.tick_params(axis="x", rotation=0)
    ax2.legend(title="Emotion", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
