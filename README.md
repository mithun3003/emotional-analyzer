# Interpretable Contextual Semantic Analysis of Multimodal Emotions for Therapy Support

A clinically-grounded, multimodal conversational emotion recognition system evaluated on the official **MELD** benchmark splits (train, dev, test) for reflective therapeutic decision-support.

> **CRITICAL CLINICAL DISCLAIMER:**
> **Decision-support information only. Not a diagnosis or a replacement for a qualified mental-health professional.**
> This system must **never** output clinical diagnoses or autonomous clinical claims. All analyses are strictly descriptive and interpretative.

---

## 1. System Architecture

```
                       ┌────────────────────────────────────────┐
                       │           Conversational Input         │
                       │    Text (768-d) | Audio | Video        │
                       └───────────────────┬────────────────────┘
                                           │
                        ┌──────────────────▼──────────────────┐
                        │      Unimodal Feature Encoders      │
                        │    Text / Audio / Video -> d_model  │
                        └──────────────────┬──────────────────┘
                                           │
                        ┌──────────────────▼──────────────────┐
                        │   Text-Enhanced Attention (TE)      │
                        │  Query: Text | Key/Val: [T; A; V]   │
                        │  Returns: Fused Repr + Modality %   │
                        └──────────────────┬──────────────────┘
                                           │
                        ┌──────────────────▼──────────────────┐
                        │   Dialogue Context Graph (GED)      │
                        │  Edges: Intra- & Inter-Speaker      │
                        │  Relational Graph Conv (RGCN)       │
                        │  + Graph Transformer Multi-Head     │
                        │  Returns: Context Attention Weights │
                        └──────────────────┬──────────────────┘
                                           │
                        ┌──────────────────▼──────────────────┐
                        │     Emotion Classifier Head         │
                        │   7 Classes (Class-Weighted CE)     │
                        └──────────────────┬──────────────────┘
                                           │
                ┌──────────────────────────┴──────────────────────────┐
                │                                                     │
    ┌───────────▼───────────┐                             ┌───────────▼───────────┐
    │  Three-Level Attrib.  │                             │ Timeline & Summary    │
    │  - Modality %         │                             │ - Affect Progression  │
    │  - Lexical Tokens     │                             │ - Emotional Shifts    │
    │  - Prior Utterance Cues│                            │ - Clinical Report     │
    └───────────────────────┘                             └───────────────────────┘
```

---

## 2. Methodology & Key Innovations

1. **Official Benchmark Splits**: Strict evaluation on official MELD partitions (9,989 train, 1,109 dev, 2,610 test). Epoch selection and hyperparameter tuning occur exclusively on **Dev**.
2. **Text-Enhanced Multimodal Attention (TE)**: Following Wang et al. (2025, CIME), text serves as Query attending across concatenated modalities, with explicit missing-modality mask support.
3. **Conversational Graph Modeling (GED)**: Utterances are modeled as nodes with directed relational edges capturing conversational context (past $w=4$, future $f=1$), distinguishing **intra-speaker** and **inter-speaker** relations through RGCN and Graph Transformer layers.
4. **Three-Level Multi-Modal Interpretability**:
   - **Modality Attribution**: Quantifies percentage contribution of Text, Audio, and Video.
   - **Token Attribution**: Leave-one-word-out occlusion importance for pivotal lexical markers.
   - **Context Attribution**: Identifies which preceding utterances and speaker relationships shaped the current prediction.
5. **Class Imbalance Mitigation**: Class-weighted cost-sensitive cross-entropy preventing majority collapse on the dominant `neutral` class.

---

## 3. Repository Structure

```
emotional-analyzer/
├── configs/
│   └── default.yaml                     # Master hyperparameter & path config
├── data/
│   ├── raw/                             # Official MELD train, dev, test CSVs
│   ├── interim/                         # Interim assets
│   └── features/                        # Cached .pt multimodal tensors
├── src/
│   ├── data/
│   │   ├── setup_data.py                # Dataset download & verification
│   │   ├── inspect_meld.py              # Dataset audit & class distribution
│   │   └── dataset.py                   # PyTorch dataset & graph construction
│   ├── features/
│   │   └── extract_features.py          # Multimodal feature extraction & caching
│   ├── models/
│   │   ├── fusion.py                    # Text-enhanced cross-modal attention
│   │   ├── graph_context.py             # RGCN + Graph Transformer layers
│   │   ├── classifier.py                # Full multimodal emotion architecture
│   │   └── baselines.py                 # Baseline models (Majority, MLP, Concat)
│   ├── explain/
│   │   └── explain.py                   # Multi-level attribution & explanation
│   ├── summary/
│   │   ├── timeline.py                  # Affect progression & shift detection
│   │   └── report_generator.py          # Structured JSON & Markdown reports
│   ├── train.py                         # Training pipeline & ablation suite
│   ├── evaluate.py                      # Test evaluation & 3-seed reporting
│   └── app/
│       └── streamlit_app.py             # Interactive clinician dashboard
├── tests/
│   └── test_pipeline.py                 # Pytest test suite (shapes, graph, attributions)
├── outputs/
│   ├── checkpoints/                     # Model weights (best_model.pt)
│   ├── metrics/                         # baselines.csv, ablation.csv, test_results.json
│   ├── figures/                         # Confusion matrices, timelines, curves
│   └── reports/                         # Clinical dialogue reports
└── docs/
    ├── dataset_audit.md                 # MELD split audit report
    ├── ethics_and_limitations.md        # Clinical ethics, privacy & limitations
    └── error_analysis.md                # Qualitative & quantitative error analysis
```

---

## 4. Quickstart Guide

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Prepare Data & Run Audit
```bash
python -m src.data.setup_data
python -m src.data.inspect_meld
```

### 3. Extract & Cache Features
```bash
python -m src.features.extract_features
python -m src.data.dataset  # runs smoke test
```

### 4. Run Baselines
```bash
python -m src.models.baselines
```

### 5. Train Proposed Model & Run Ablations
```bash
python -m src.train
```

### 6. Run Test Evaluation & Multi-Seed Analysis
```bash
python -m src.evaluate
```

### 7. Run Unit Tests
```bash
python -m pytest tests/test_pipeline.py -v
```

### 8. Launch Interactive Clinician Dashboard
```bash
streamlit run src/app/streamlit_app.py
```

---

## 5. Results Summary

### Baselines (Official MELD Test Set)
| Model | Modalities | Context | Acc | Weighted F1 | Macro F1 |
|---|---|---|---|---|---|
| **Majority Class** | - | No | 0.4812 | 0.3127 | 0.0928 |
| **BERT (Text-only)** | T | No | 0.3927 | 0.4033 | 0.2563 |
| **Audio-only** | A | No | 0.1540 | 0.0411 | 0.0381 |
| **Video-only** | V | No | 0.4812 | 0.3127 | 0.0928 |
| **Concat (T+A)** | T+A | No | 0.3897 | 0.3979 | 0.2620 |
| **Concat (T+A+V)** | T+A+V | No | 0.3356 | 0.3574 | 0.2481 |
#   e m o t i o n a l - a n a l y z e r  
 