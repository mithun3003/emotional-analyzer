from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.data.dataset import DialogueSample
from src.explain.explain import EmotionExplainer
from src.summary.timeline import analyze_dialogue_timeline, plot_emotion_timeline

MANDATORY_DISCLAIMER = (
    "Decision-support information only. Not a diagnosis or a replacement for a qualified mental-health professional."
)


def generate_therapy_summary_report(
    dialogue: DialogueSample,
    explainer: EmotionExplainer,
    output_dir: str | Path = "outputs/reports",
) -> Dict[str, Any]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Run inference & explanations on each utterance
    utterance_explanations: List[Dict[str, Any]] = []
    predictions: List[str] = []
    confidences: List[float] = []

    for i in range(dialogue.num_utterances):
        exp = explainer.explain_utterance(dialogue, i)
        utterance_explanations.append(exp)
        predictions.append(exp["predicted_emotion"])
        confidences.append(exp["confidence"])

    # 2. Timeline analysis
    timeline_data = analyze_dialogue_timeline(dialogue, predictions, confidences)

    # 3. Low-confidence segments
    low_confidence_segments = [
        exp for exp in utterance_explanations if exp["is_low_confidence"]
    ]

    # 4. Generate Timeline Figure
    fig_path = out_path / f"dialogue_{dialogue.dialogue_id}_timeline.png"
    plot_emotion_timeline(timeline_data, fig_path)

    # 5. Build Structured Report Payload
    report_payload = {
        "dialogue_id": dialogue.dialogue_id,
        "disclaimer": MANDATORY_DISCLAIMER,
        "total_utterances": dialogue.num_utterances,
        "speaker_dynamics": timeline_data["speaker_stats"],
        "emotion_transitions": timeline_data["emotion_shifts"],
        "low_confidence_segments": [
            {
                "utterance_idx": seg["utterance_idx"],
                "speaker": seg["speaker"],
                "text": seg["text"],
                "predicted_emotion": seg["predicted_emotion"],
                "confidence": seg["confidence"],
            }
            for seg in low_confidence_segments
        ],
        "utterances": utterance_explanations,
        "timeline_plot_path": str(fig_path),
    }

    # Save JSON Report
    json_path = out_path / f"dialogue_{dialogue.dialogue_id}_report.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    # Save Markdown Report
    md_path = out_path / f"dialogue_{dialogue.dialogue_id}_report.md"
    _write_markdown_report(report_payload, md_path)

    return report_payload


def _write_markdown_report(report: Dict[str, Any], output_path: Path) -> None:
    lines = [
        f"# Conversational Emotion Timeline & Clinical Review Report (Dialogue #{report['dialogue_id']})",
        "",
        "> [!IMPORTANT]",
        f"> **CLINICAL DISCLAIMER**: {report['disclaimer']}",
        "",
        "## 1. Conversational Dynamics & Dominant Affect",
        "",
        "| Speaker | Total Utterances | Dominant Emotion | Average Confidence | Emotion Breakdown |",
        "|---|---|---|---|---|",
    ]

    for spk, stats in report["speaker_dynamics"].items():
        breakdown = ", ".join([f"{k}: {v}" for k, v in stats["emotion_counts"].items()])
        lines.append(
            f"| **{spk}** | {stats['total_utterances']} | **{stats['dominant_emotion'].capitalize()}** | {stats['avg_confidence']:.2%} | {breakdown} |"
        )

    lines.extend([
        "",
        "## 2. Emotional State Transitions (Shifts)",
        "",
    ])

    if report["emotion_transitions"]:
        lines.extend([
            "| Speaker | From Utterance | To Utterance | Transition |",
            "|---|---|---|---|",
        ])
        for shift in report["emotion_transitions"]:
            lines.append(
                f"| {shift['speaker']} | #{shift['from_utterance']} | #{shift['to_utterance']} | **{shift['from_emotion']}** ➔ **{shift['to_emotion']}** |"
            )
    else:
        lines.append("*No significant emotional transitions detected in this dialogue sequence.*")

    lines.extend([
        "",
        "## 3. Low-Confidence Predictions Requiring Clinical Attention",
        "",
    ])

    if report["low_confidence_segments"]:
        lines.extend([
            "| Utterance | Speaker | Text | Predicted Emotion | Confidence | Clinician Note |",
            "|---|---|---|---|---|---|",
        ])
        for seg in report["low_confidence_segments"]:
            txt = seg["text"].replace("|", "\\|")
            lines.append(
                f"| #{seg['utterance_idx']} | {seg['speaker']} | \"{txt}\" | {seg['predicted_emotion']} | {seg['confidence']:.2%} | Review tone & non-verbal cues |"
            )
    else:
        lines.append("*All predicted utterances exceeded the clinical confidence threshold.*")

    lines.extend([
        "",
        "## 4. Multimodal Interpretability Highlights (Sample Utterances)",
        "",
    ])

    for utt in report["utterances"][:5]:
        lines.extend([
            f"### Utterance #{utt['utterance_idx']} - {utt['speaker']}: \"{utt['text']}\"",
            f"- **Interpretation**: {utt['explanation_text']}",
            f"- **Modality Cue Breakdown**: Text {utt['modality_attribution']['text_percent']:.1f}% | Audio {utt['modality_attribution']['audio_percent']:.1f}% | Video {utt['modality_attribution']['video_percent']:.1f}%",
            "",
        ])

    lines.extend([
        "## 5. Visual Timeline",
        "",
        f"![Conversational Emotion Progression]({Path(report['timeline_plot_path']).name})",
        "",
    ])

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
