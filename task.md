# TASK: Interpretable Contextual Semantic Analysis of Multimodal Emotions for Therapy Support

> Instructions for the IDE agent. Work phase by phase. Finish and verify each phase (see "Acceptance") before starting the next. After each phase, update `PROGRESS.md` with what was done, commands to reproduce, and any problems.

---

## 0. Project Summary

Build an end-to-end system that:

1. Takes conversational data (text + audio + video) from the **MELD** dataset (`MELD.Raw.tar`).
2. Predicts one of **7 emotions** per utterance: `anger, disgust, fear, joy, neutral, sadness, surprise`.
3. Uses **dialogue context** (previous utterances and speaker relations) to improve predictions.
4. Produces **interpretable output**: which modality and which context utterances/words drove each prediction.
5. Generates a **conversation-level emotion timeline and summary** for therapy-support review.
6. Is evaluated with **accuracy, precision, recall, weighted F1, per-class results, and confusion matrix**.

**Hard rule:** this is a *therapy-support tool only*. It must never output a diagnosis or clinical claim. Show a disclaimer in the UI and in every generated summary.

---

## 1. Design Reference (from the three papers)

Take these ideas. Do not copy numbers as targets.

| Source | What to reuse |
|---|---|
| **Voloshina & Makhnytkina (2023)**, 3-Modal Cross-BERT / Multimodal Interaction | Text: `bert-base-uncased` (768-d). Audio: 40-d MFCC (librosa) + 88-d eGeMAPS (openSMILE), concatenated, then Conv1D. Video: every 5th frame, face crop (OpenCV/dlib), grayscale 64x64, groups of 10 frames with overlap 5, Conv3D + LSTM. Masked attention fusion. Known issue: Neutral is confused with other classes on MELD. |
| **Wang et al. (2025), CIME** | Text-centric cross-modal attention (text = Query; concatenated audio+video+text = Key/Value; multi-head; residual + LayerNorm). Dialogue graph: nodes = utterances, edges = intra-speaker and inter-speaker relations; **RGCN** then **Graph Transformer**; mean-pool/classifier; cross-entropy + L2. Ablate with/without graph (GED) and with/without text-enhanced attention (TE). |
| **Kachhara et al. (2024)** | Practical tooling only: OpenCV + DeepFace for faces, librosa for audio features, MLP head. Do **not** copy its depression-detection framing. |

### Important methodology warnings
- Use the **official MELD train/dev/test split**. Do not use a random 80/20 split.
- Select the best epoch and tune hyperparameters on **dev only**. Report test results once, at the end. (Choosing the best epoch on the test set is leakage.)
- Numbers from the papers (62.2% on MELD, 71.7% on IEMOCAP, 77.9% depression) are on different splits or tasks. Use them only as loose context. Report **our own measured results**.
- If a modality file is missing or unusable, say so clearly. Never describe a modality as processed if it was not.

---

## 2. Tech Stack

- Python 3.10+
- PyTorch, `transformers` (HuggingFace), `scikit-learn`, `pandas`, `numpy`
- `torch_geometric` (RGCNConv, TransformerConv). Fall back to plain-PyTorch if install fails.
- `librosa`, `opensmile`, `ffmpeg` (audio extraction)
- `opencv-python`, `mediapipe` or `facenet-pytorch (MTCNN)` or `dlib` for face detection (pick one that installs cleanly; document the choice)
- `captum` (token-level attribution), `matplotlib`, `seaborn`, `plotly`
- `streamlit` for the UI
- `pytest`, `pyyaml`, `tqdm`

Check for GPU at start. If none, keep models small, cache all features to disk, and train only on cached features (the BERT/audio/video encoders can be frozen).

---

## 3. Repository Layout

```
emotion-therapy-support/
├── README.md
├── PROGRESS.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/                 # MELD.Raw.tar extracted here (git-ignored)
│   ├── interim/             # extracted wav, frames, face crops
│   └── features/            # cached .pt / .npy features
├── src/
│   ├── data/
│   │   ├── inspect_meld.py      # dataset audit
│   │   ├── dataset.py           # MELD dataset + dialogue batching
│   │   └── preprocess_text.py
│   ├── features/
│   │   ├── text_features.py
│   │   ├── audio_features.py
│   │   └── video_features.py
│   ├── models/
│   │   ├── fusion.py            # text-centric cross-modal attention
│   │   ├── graph_context.py     # dialogue graph + RGCN + graph transformer
│   │   ├── classifier.py
│   │   └── baselines.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain/
│   │   ├── modality_importance.py
│   │   ├── token_attribution.py
│   │   └── context_importance.py
│   ├── summary/
│   │   ├── timeline.py
│   │   └── report_generator.py
│   └── app/
│       └── streamlit_app.py
├── tests/
├── notebooks/
├── outputs/
│   ├── checkpoints/
│   ├── metrics/
│   └── figures/
└── docs/
    └── ethics_and_limitations.md
```

---

## 4. Phases and Tasks

### Phase 1: Setup and Dataset Audit
- [ ] Create the repo layout, `requirements.txt`, `configs/default.yaml` (paths, seeds, hyperparameters).
- [ ] Set global seed utility (Python, NumPy, torch) and use it everywhere.
- [ ] Extract `MELD.Raw.tar` into `data/raw/`.
- [ ] Write `src/data/inspect_meld.py` that reports:
  - Folder structure and which files exist (CSV label files, train/dev/test video folders).
  - Number of dialogues and utterances per split.
  - Emotion class distribution per split (print table and save bar chart).
  - Count of missing / corrupt / zero-length video files.
  - Whether audio can be extracted from each video.
  - Save to `outputs/metrics/dataset_audit.json` and `docs/dataset_audit.md`.
- [ ] Be explicit about which modalities are actually available.

**Acceptance:** audit file exists; class imbalance is documented (neutral dominates; fear and disgust are rare); the agent states which modalities are usable.

---

### Phase 2: Preprocessing and Feature Extraction (cache everything)

**Text**
- [ ] Clean text, keep speaker name, utterance order, dialogue ID, utterance ID.
- [ ] Extract BERT features (`bert-base-uncased`), `[CLS]` embedding (768-d) per utterance. Max length 50. Save per split.

**Audio**
- [ ] Extract mono 16 kHz `.wav` from each video with ffmpeg.
- [ ] Extract 40-d MFCC (mean + std pooled over time or time-series, document choice) and 88-d eGeMAPS. Concatenate to one audio vector.
- [ ] Handle failures: log the utterance and use a zero vector plus a `missing_audio` mask flag.

**Video**
- [ ] Sample every 5th frame. Detect and crop faces. Convert to grayscale 64x64.
- [ ] Build windows of 10 frames with overlap 5; use the first 25 sampled frames max.
- [ ] If multiple faces appear, pick the largest or most central face and document this (MELD has multi-person frames and speaker ID is hard).
- [ ] If no face is detected, set a `missing_video` mask flag.
- [ ] Optional lighter alternative: pretrained frame embeddings (e.g., ResNet/ViT on face crops), averaged over frames. Make this a config switch.

**All**
- [ ] Save features keyed by `(dialogue_id, utterance_id)`.
- [ ] Add a `--limit N` flag to every extraction script for fast testing on a small subset.
- [ ] Features must be resumable (skip already-computed files).

**Acceptance:** features exist for train/dev/test; a short report lists how many utterances have missing audio or video; a smoke test loads one batch with all modalities.

---

### Phase 3: Baselines (needed for honest comparison)
- [ ] Majority-class baseline.
- [ ] Text-only (BERT features + linear/MLP head).
- [ ] Audio-only, Video-only (small MLP/CNN heads).
- [ ] Text+Audio, Text+Audio+Video with simple concatenation fusion.
- [ ] Use class-weighted cross-entropy (or focal loss) to handle imbalance. Make it a config option.
- [ ] Log accuracy, weighted F1, macro F1, per-class P/R/F1.

**Acceptance:** `outputs/metrics/baselines.csv` with all rows; selection was done on dev only.

---

### Phase 4: Proposed Model (Context-Aware Multimodal)

- [ ] **Unimodal encoders**: Conv1D / MLP for audio, Conv3D+LSTM (or frame-embedding MLP) for video, linear projection for text. Project all to a common dimension `d`.
- [ ] **Text-enhanced cross-modal attention (TE)** in `fusion.py`:
  - Query = text features; Key/Value = concatenated [text; audio; video]; multi-head attention; residual + LayerNorm.
  - **Return the attention weights** (needed for explainability).
  - Support modality masking for missing audio/video.
- [ ] **Dialogue graph (GED)** in `graph_context.py`:
  - Node = utterance (fused feature).
  - Edges within a context window of `w` past (and optionally `f` future) utterances in the same dialogue.
  - Two relation types: **intra-speaker** and **inter-speaker**. Make `w` and `f` configurable.
  - `RGCNConv` then `TransformerConv` (graph transformer), with residual + LayerNorm and dropout.
  - **Keep per-edge attention weights** from the graph transformer for explanations.
- [ ] **Classifier**: linear layer over node features to 7 classes (per-utterance prediction, not a single pooled vector, since we need one emotion per utterance).
- [ ] **Loss**: cross-entropy (class-weighted) + L2 weight decay. AdamW, early stopping on dev weighted F1.
- [ ] Training script `train.py` with config file, checkpointing best-on-dev, TensorBoard or CSV logging, mixed precision when GPU is available.
- [ ] **Ablations** (run all, save to `outputs/metrics/ablation.csv`):
  - T, T+A, T+A+V
  - with / without graph context (GED)
  - with / without text-enhanced attention (TE)
  - context window sizes (e.g., 0, 2, 4, 6)

**Acceptance:** model trains end-to-end; ablation table exists; training curves saved; best checkpoint reloads and reproduces dev metrics.

---

### Phase 5: Evaluation
- [ ] `evaluate.py`: run best checkpoint on **test once**. Save:
  - Accuracy, weighted precision/recall/F1, macro F1
  - Per-class precision/recall/F1 and support
  - Confusion matrix (counts and row-normalized) as PNG
  - A list of the most common misclassification pairs (expect Neutral confusions, and Fear/Disgust being rare)
- [ ] Short error analysis in `docs/error_analysis.md` with 10-20 concrete misclassified examples and likely reasons.
- [ ] Run 3 seeds and report mean ± std.

**Acceptance:** `outputs/metrics/test_results.json`, confusion matrix figure, error analysis doc.

---

### Phase 6: Interpretability
Implement all three, expose them through one function `explain_utterance(dialogue, utterance_idx)` that returns a structured dict.

- [ ] **Modality importance**: from TE attention weights and from a modality-ablation check (zero or mask one modality, measure change in predicted probability). Output: text/audio/video contribution percentages.
- [ ] **Token importance**: Captum (Integrated Gradients or attention-based) on the text branch. Output: top-k words with scores. If fine-tuning BERT is skipped (frozen features), use a lightweight alternative (leave-one-word-out occlusion) and document it.
- [ ] **Context importance**: from graph attention weights, list which previous utterances (and speaker relation: intra/inter) influenced the prediction most.
- [ ] Plain-language explanation generator (template-based, no LLM required), e.g. "Predicted *sadness* (72%). Main cues: text (60%), audio (30%). Key words: 'alone', 'sorry'. Most influential earlier utterance: #3 by another speaker."
- [ ] Always show the prediction confidence, and flag low-confidence predictions (e.g., max prob < threshold).
- [ ] Sanity check: explanations must change when inputs change (add a test).

**Acceptance:** `explain_utterance` works on any test dialogue; saved example explanation figures in `outputs/figures/`.

---

### Phase 7: Timeline and Therapy-Support Summary
- [ ] `timeline.py`: per dialogue, produce utterance-level emotion sequence with confidence, per speaker.
- [ ] Plot emotion timeline (x = utterance order, y = emotion, color by speaker) and emotion distribution per speaker.
- [ ] Detect **emotion shifts** (change between consecutive utterances of a speaker) and **dominant emotion** per speaker.
- [ ] `report_generator.py`: produce a structured summary (JSON + Markdown/HTML):
  - Dominant emotions, emotion transitions, low-confidence segments, key explanations
  - Fixed disclaimer: *"Decision-support information only. Not a diagnosis or a replacement for a qualified mental-health professional."*
- [ ] Never include wording like "depressed", "disorder", or "diagnosis" as a model output.

**Acceptance:** a sample dialogue produces a timeline figure and a summary file.

---

### Phase 8: User Interface (Streamlit)
- [ ] Select a MELD test dialogue (or upload text + optional audio/video if supported).
- [ ] Show per-utterance table: speaker, text, predicted emotion, confidence, true label (when known).
- [ ] Click an utterance: show modality contribution bars, highlighted key words, influential context utterances.
- [ ] Show emotion timeline and conversation summary.
- [ ] Download button for the summary report.
- [ ] Disclaimer banner always visible. Privacy note: data processed locally, nothing stored by default.
- [ ] Handle missing modalities gracefully (text-only mode).

**Acceptance:** `streamlit run src/app/streamlit_app.py` works on a clean checkout with the trained checkpoint.

---

### Phase 9: Testing, Docs, Packaging
- [ ] `pytest` tests: dataset loader shapes, missing-modality masks, graph construction (edge counts and relation types), model forward pass shapes, metric computation, explanation output schema.
- [ ] `README.md`: setup, data download/extraction, one-command pipeline (`make features`, `make train`, `make eval`, `make app`), results table, screenshots.
- [ ] `docs/ethics_and_limitations.md`: dataset bias (TV show *Friends*, scripted acting, limited demographic diversity), acted vs real emotion, not clinically validated, privacy, misclassification risk, not for diagnosis.
- [ ] Pin versions in `requirements.txt`. Everything reproducible from seed and config.

---

## 5. Metrics Table to Fill (final report)

| Model | Modalities | Context | Acc | Weighted F1 | Macro F1 |
|---|---|---|---|---|---|
| Majority | - | - | | | |
| BERT | T | No | | | |
| Concat | T+A+V | No | | | |
| Proposed (no graph) | T+A+V | No | | | |
| Proposed (full) | T+A+V | Yes | | | |

Also include the per-class table and confusion matrix. Report honestly even if the proposed model does not beat a baseline.

---

## 6. Coding Rules for the Agent
- Config-driven (YAML); no hard-coded paths.
- Type hints and short docstrings on every function.
- Log with `logging`, not bare `print`.
- Every long script supports `--limit` and resumes from cache.
- Do not commit data, features, or checkpoints (add to `.gitignore`).
- Do not fabricate results. If something fails, record it in `PROGRESS.md` and propose an alternative.
- Prefer simple, working code over complicated architectures. Get Phase 3 baselines running **before** the proposed model.
- Ask me before changing the target emotion set, dataset, or the therapy-support-only scope.

---

## 7. Fallbacks if Resources Are Limited
- No GPU: freeze all encoders, train only fusion + graph + classifier on cached features.
- Video too slow or unreliable: use frame-embedding mode or run text+audio first, then add video; document the limitation.
- `torch_geometric` install problems: implement a simple relational graph layer in plain PyTorch with adjacency matrices per relation.
- Faces not detected in many clips: rely on the `missing_video` mask and report the percentage.

---

## 8. Definition of Done
- [ ] Dataset audit, features, baselines, proposed model, and ablations all complete and reproducible.
- [ ] Test-set metrics (accuracy, precision, recall, weighted F1, per-class, confusion matrix) reported with 3-seed mean ± std.
- [ ] Explanations (modality, word, context) available for every prediction.
- [ ] Timeline + summary generated and shown in the Streamlit app.
- [ ] Disclaimers and ethics documentation present.
- [ ] README lets a new user run everything from scratch.