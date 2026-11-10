from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.data.dataset import DialogueSample, MELDDialogueDataset, extract_and_cache_features
from src.explain.explain import EmotionExplainer
from src.models.classifier import MultimodalEmotionModel
from src.summary.report_generator import generate_therapy_summary_report
from src.summary.timeline import analyze_dialogue_timeline, plot_emotion_timeline
from src.utils import get_device, load_config

# Set page config
st.set_page_config(
    page_title="Multimodal Emotion Analyzer (Therapy Support)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .disclaimer-banner {
        background-color: #fff3cd;
        border-left: 6px solid #ffc107;
        padding: 12px 18px;
        margin-bottom: 20px;
        border-radius: 4px;
        color: #856404;
        font-weight: 500;
    }
    .privacy-badge {
        background-color: #e8f4fd;
        border-left: 6px solid #2196f3;
        padding: 10px 16px;
        margin-bottom: 15px;
        border-radius: 4px;
        color: #0d47a1;
        font-size: 0.9em;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 12px;
        border-radius: 6px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_explainer_and_data():
    config = load_config("configs/default.yaml")
    ckpt_path = Path("outputs/checkpoints/best_model.pt")

    if not ckpt_path.exists():
        st.error(f"Trained checkpoint not found at {ckpt_path}. Please train the model first.")
        return None, None, config

    explainer = EmotionExplainer.from_checkpoint(str(ckpt_path), "configs/default.yaml")
    data_splits = extract_and_cache_features("configs/default.yaml")
    test_ds = MELDDialogueDataset(data_splits["test"], past_window=4, future_window=1)
    return explainer, test_ds, config


def main():
    st.title("🧠 Interpretable Multimodal Emotion Analyzer")
    st.subheader("Conversational Affect Progression for Reflective Therapy Decision-Support")

    # Mandatory Clinical Disclaimer Banner
    st.markdown(
        """
        <div class="disclaimer-banner">
            ⚠️ <strong>DECISION-SUPPORT NOTICE ONLY</strong>:
            This tool is designed strictly to assist clinicians and researchers in reviewing conversational affect.
            It does <strong>NOT</strong> provide psychiatric diagnoses, medical claims, or replacements for licensed clinical judgment.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Privacy Banner
    st.markdown(
        """
        <div class="privacy-badge">
            🔒 <strong>Privacy Assurance</strong>: All text, audio, and video representations are processed entirely locally on this workstation. No client discourse is sent to external APIs.
        </div>
        """,
        unsafe_allow_html=True,
    )

    explainer, test_ds, config = load_explainer_and_data()
    if explainer is None or test_ds is None:
        st.stop()

    # Sidebar Selection
    st.sidebar.header("Navigation & Dialogue Selector")
    total_dialogues = len(test_ds)
    dialogue_indices = list(range(min(50, total_dialogues)))
    selected_dia_idx = st.sidebar.selectbox(
        "Select MELD Test Dialogue",
        dialogue_indices,
        format_func=lambda i: f"Dialogue #{test_ds[i].dialogue_id} ({test_ds[i].num_utterances} utterances)",
    )

    dialogue = test_ds[selected_dia_idx]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Dialogue ID**: #{dialogue.dialogue_id}")
    st.sidebar.markdown(f"**Utterance Count**: {dialogue.num_utterances}")
    st.sidebar.markdown(f"**Unique Speakers**: {len(set(dialogue.speakers))}")
    st.sidebar.markdown("---")

    # Run inference and explanations
    with st.spinner("Computing multimodal inference and attributions..."):
        explanations = []
        predictions = []
        confidences = []
        for i in range(dialogue.num_utterances):
            exp = explainer.explain_utterance(dialogue, i)
            explanations.append(exp)
            predictions.append(exp["predicted_emotion"])
            confidences.append(exp["confidence"])

        timeline_data = analyze_dialogue_timeline(dialogue, predictions, confidences)

    # Tabs for Workflow
    tab1, tab2, tab3 = st.tabs(["💬 Dialogue & Attributions", "📈 Emotion Progression & Timeline", "📄 Clinical Summary Report"])

    with tab1:
        st.markdown("### Utterance Sequence & Predictions")
        st.write("Click an utterance to inspect modality, token, and context attributions:")

        # Summary Table
        table_rows = []
        for i, exp in enumerate(explanations):
            true_label = dialogue.emotions[i]
            is_correct = "✅" if exp["predicted_emotion"] == true_label else "❌"
            table_rows.append({
                "Index": i,
                "Speaker": exp["speaker"],
                "Utterance": exp["text"],
                "Predicted Emotion": exp["predicted_emotion"].capitalize(),
                "Confidence": f"{exp['confidence']:.1%}",
                "True Label": true_label.capitalize(),
                "Match": is_correct,
            })
        df_table = pd.DataFrame(table_rows)
        st.dataframe(df_table, use_container_width=True)

        selected_utt_idx = st.selectbox(
            "Select an utterance for deep attribution review:",
            range(dialogue.num_utterances),
            format_func=lambda i: f"#{i} - {dialogue.speakers[i]}: \"{dialogue.texts[i][:45]}...\"",
        )

        selected_exp = explanations[selected_utt_idx]

        st.markdown("---")
        st.markdown(f"#### 🔍 Deep Attribution Analysis for Utterance #{selected_utt_idx}")
        st.info(selected_exp["explanation_text"])

        col1, col2, col3 = st.columns(3)

        # 1. Modality Attribution
        with col1:
            st.markdown("##### 1. Modality Cue Contributions")
            mod_data = pd.DataFrame([
                {"Modality": "Text", "Percentage": selected_exp["modality_attribution"]["text_percent"]},
                {"Modality": "Audio", "Percentage": selected_exp["modality_attribution"]["audio_percent"]},
                {"Modality": "Video", "Percentage": selected_exp["modality_attribution"]["video_percent"]},
            ])
            chart = alt.Chart(mod_data).mark_bar().encode(
                x="Percentage:Q",
                y=alt.Y("Modality:N", sort=["Text", "Audio", "Video"]),
                color=alt.Color("Modality:N", scale=alt.Scale(scheme="category10")),
            ).properties(height=200)
            st.altair_chart(chart, use_container_width=True)

        # 2. Token Attribution
        with col2:
            st.markdown("##### 2. Key Lexical Tokens (Occlusion)")
            token_data = pd.DataFrame(selected_exp["token_attribution"])
            if not token_data.empty:
                t_chart = alt.Chart(token_data).mark_bar(color="#ff7f0e").encode(
                    x="importance_score:Q",
                    y=alt.Y("word:N", sort="-x"),
                ).properties(height=200)
                st.altair_chart(t_chart, use_container_width=True)
            else:
                st.write("No lexical tokens identified.")

        # 3. Contextual Influences
        with col3:
            st.markdown("##### 3. Influential Context Utterances")
            ctx_list = [c for c in selected_exp["context_influences"] if not c["is_self"]][:4]
            if ctx_list:
                for ctx in ctx_list:
                    st.markdown(
                        f"- **#{ctx['source_utterance_idx']} ({ctx['source_speaker']})**: "
                        f"weight `{ctx['attention_weight']:.3f}` | *\"{ctx['source_text'][:35]}...\"*"
                    )
            else:
                st.write("No prior conversational context influence.")

    with tab2:
        st.markdown("### Conversational Emotion Progression Timeline")
        timeline_img_path = Path(f"outputs/figures/temp_timeline_{dialogue.dialogue_id}.png")
        plot_emotion_timeline(timeline_data, timeline_img_path)
        st.image(str(timeline_img_path), use_column_width=True)

        st.markdown("#### Detected Affective Transitions")
        if timeline_data["emotion_shifts"]:
            shifts_df = pd.DataFrame(timeline_data["emotion_shifts"])
            st.dataframe(shifts_df, use_container_width=True)
        else:
            st.write("No dramatic affective shifts detected.")

    with tab3:
        st.markdown("### Clinical Decision-Support Summary")
        report = generate_therapy_summary_report(dialogue, explainer, output_dir="outputs/reports")

        st.markdown(f"**Dominant Emotions per Speaker:**")
        cols = st.columns(len(report["speaker_dynamics"]))
        for i, (spk, stats) in enumerate(report["speaker_dynamics"].items()):
            with cols[i]:
                st.metric(
                    label=spk,
                    value=stats["dominant_emotion"].capitalize(),
                    delta=f"{stats['avg_confidence']:.1%} avg conf",
                )

        st.markdown("---")
        report_md_path = Path(f"outputs/reports/dialogue_{dialogue.dialogue_id}_report.md")
        if report_md_path.exists():
            with report_md_path.open("r", encoding="utf-8") as f:
                md_content = f.read()

            st.download_button(
                label="📥 Download Clinical Summary Report (Markdown)",
                data=md_content,
                file_name=f"clinical_emotion_report_dialogue_{dialogue.dialogue_id}.md",
                mime="text/markdown",
            )
            with st.expander("Preview Generated Clinical Report", expanded=True):
                st.markdown(md_content)


if __name__ == "__main__":
    main()
