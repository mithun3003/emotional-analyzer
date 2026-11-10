# 🧠 Interpretable Contextual Semantic Analysis of Multimodal Emotions for Therapy Support

A clinically grounded, multimodal conversational emotion recognition system evaluated on the official **MELD benchmark dataset** (Train, Dev, and Test splits) for reflective therapeutic decision-support.

The system integrates **Text, Audio, and Video** modalities with conversational context modeling to recognize seven emotion categories and generate interpretable emotional summaries.

> [!IMPORTANT]
> **CLINICAL DISCLAIMER**
>
> This project provides decision-support information only. It is not a diagnostic tool and is not a replacement for a qualified mental-health professional.
>
> The system must never generate clinical diagnoses or autonomous clinical claims. All outputs are strictly descriptive and interpretative.

---

## 📌 1. Project Overview

Human emotions are complex and cannot always be understood through text alone. Facial expressions, vocal characteristics, and conversational context provide additional emotional information.

This project proposes an interpretable multimodal emotion recognition framework that analyzes conversational utterances using:

- **Text:** Semantic and contextual information from spoken sentences.
- **Audio:** Vocal characteristics, pitch, tone, and speech patterns.
- **Video:** Facial expressions and visual features.
- **Context:** Previous utterances and speaker relationships.

The proposed system combines these modalities using Text-Enhanced Attention and Dialogue Context Graph modeling to produce emotion predictions with interpretable explanations.

### 🎯 Objectives

1. Develop a multimodal emotion recognition system using Text, Audio, and Video.
2. Integrate conversational context using graph-based learning.
3. Improve emotion classification using class-weighted loss.
4. Provide modality, token, and context-level interpretability.
5. Generate emotional progression timelines and structured summaries.
6. Evaluate performance using Accuracy, Weighted F1-score, and Macro F1-score.

---

## 🏗️ 2. System Architecture

The proposed architecture consists of the following major components:

```text
                  ┌─────────────────────────────────────┐
                  │       Conversational Input          │
                  │                                     │
                  │      Text | Audio | Video           │
                  └──────────────────┬──────────────────┘
                                     │
                  ┌──────────────────▼──────────────────┐
                  │       Unimodal Feature Encoders     │
                  │                                     │
                  │     Text / Audio / Video → d_model  │
                  └──────────────────┬──────────────────┘
                                     │
                  ┌──────────────────▼──────────────────┐
                  │      Text-Enhanced Attention        │
                  │                                     │
                  │   Query: Text                       │
                  │   Key / Value: Text + Audio + Video │
                  │                                     │
                  │   Fused Representation              │
                  └──────────────────┬──────────────────┘
                                     │
                  ┌──────────────────▼──────────────────┐
                  │       Dialogue Context Graph        │
                  │                                     │
                  │   Intra-Speaker Relations           │
                  │   Inter-Speaker Relations           │
                  │   RGCN + Graph Transformer          │
                  └──────────────────┬──────────────────┘
                                     │
                  ┌──────────────────▼──────────────────┐
                  │        Emotion Classifier           │
                  │                                     │
                  │      Seven Emotion Classes          │
                  │      Class-Weighted CE Loss         │
                  └──────────────────┬──────────────────┘
                                     │
                ┌────────────────────┴────────────────────┐
                │                                         │
       ┌────────▼─────────┐                     ┌─────────▼────────┐
       │ Interpretability │                     │ Emotion Timeline│
       │                  │                     │                 │
       │ Modality Scores  │                     │ Affect Changes  │
       │ Token Importance │                     │ Emotional Shifts│
       │ Context Cues     │                     │ Structured Report│
       └──────────────────┘                     └─────────────────┘
```

---

## 🔬 3. Methodology and Key Innovations

### 3.1 Official MELD Benchmark Evaluation

The system uses the official MELD dataset partitions:

| Dataset Split | Number of Utterances |
|---|---:|
| Training | 9,989 |
| Development | 1,109 |
| Testing | 2,610 |
| **Total** | **13,708** |

- Training data is used for model learning.
- Development data is used for hyperparameter tuning and model selection.
- Test data is reserved for final evaluation.

### 3.2 Text-Enhanced Multimodal Attention (TE)

Text is used as the query, while Text, Audio, and Video representations are used as keys and values.

This mechanism allows the model to identify relevant information across different modalities.

**Key features:**
- Cross-modal attention.
- Multimodal feature fusion.
- Missing-modality mask support.
- Modality contribution analysis.

### 3.3 Dialogue Context Graph (GED)

Each conversational utterance is represented as a graph node.

The graph captures:
- Intra-speaker relationships.
- Inter-speaker relationships.
- Previous conversational context.
- Relevant future context.

The context window uses:
- Past utterances: `w = 4`
- Future utterances: `f = 1`

Relational Graph Convolutional Networks (RGCN) and Graph Transformer layers are used to model conversational dependencies.

### 3.4 Three-Level Multimodal Interpretability

The proposed framework provides three levels of explanation.

| Level | Technique | Purpose |
|---|---|---|
| Modality Attribution | Modality contribution scores | Measures Text, Audio, and Video contributions |
| Token Attribution | Leave-one-word-out occlusion | Identifies important lexical tokens |
| Context Attribution | Graph attention analysis | Identifies influential previous utterances and speaker relationships |

### 3.5 Class Imbalance Mitigation

The MELD dataset contains an imbalanced distribution of emotion classes.

To address this issue, class-weighted Cross-Entropy Loss is implemented to reduce majority-class bias and improve minority-class learning.

---

## 📂 4. Repository Structure

```text
emotional-analyzer/
│
├── configs/
│   └── default.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── features/
│
├── src/
│   ├── data/
│   │   ├── setup_data.py
│   │   ├── inspect_meld.py
│   │   └── dataset.py
│   │
│   ├── features/
│   │   └── extract_features.py
│   │
│   ├── models/
│   │   ├── fusion.py
│   │   ├── graph_context.py
│   │   ├── classifier.py
│   │   └── baselines.py
│   │
│   ├── explain/
│   │   └── explain.py
│   │
│   ├── summary/
│   │   ├── timeline.py
│   │   └── report_generator.py
│   │
│   ├── train.py
│   ├── evaluate.py
│   │
│   └── app/
│       └── streamlit_app.py
│
├── tests/
│   └── test_pipeline.py
│
├── outputs/
│   ├── checkpoints/
│   ├── metrics/
│   ├── figures/
│   └── reports/
│
├── docs/
│   ├── dataset_audit.md
│   ├── ethics_and_limitations.md
│   └── error_analysis.md
│
├── requirements.txt
└── README.md
```

---

## ⚙️ 5. Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| NLP | BERT / Transformer |
| Audio Processing | Librosa |
| Video Processing | OpenCV |
| Graph Learning | RGCN / Graph Transformer |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Evaluation | Scikit-learn |
| Testing | Pytest |

---

## 🚀 6. Installation and Execution

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/emotional-analyzer.git
cd emotional-analyzer
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Prepare the Dataset

```bash
python -m src.data.setup_data
```

### Step 4: Inspect and Audit the Dataset

```bash
python -m src.data.inspect_meld
```

### Step 5: Extract Multimodal Features

```bash
python -m src.features.extract_features
```

### Step 6: Run Dataset Smoke Test

```bash
python -m src.data.dataset
```

### Step 7: Run Baseline Models

```bash
python -m src.models.baselines
```

### Step 8: Train the Proposed Model

```bash
python -m src.train
```

### Step 9: Evaluate the Model

```bash
python -m src.evaluate
```

### Step 10: Run Unit Tests

```bash
python -m pytest tests/test_pipeline.py -v
```

### Step 11: Launch the Streamlit Dashboard

```bash
streamlit run src/app/streamlit_app.py
```

---

## 📊 7. Experimental Results

### 7.1 Baseline Model Performance

The following results represent the reported baseline performance on the official MELD test set.

| Model | Modalities | Context | Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---:|---:|---:|
| Majority Class | - | No | 0.4812 | 0.3127 | 0.0928 |
| BERT (Text-only) | Text | No | 0.3927 | 0.4033 | 0.2563 |
| Audio-only | Audio | No | 0.1540 | 0.0411 | 0.0381 |
| Video-only | Video | No | 0.4812 | 0.3127 | 0.0928 |
| Concat (T+A) | Text + Audio | No | 0.3897 | 0.3979 | 0.2620 |
| Concat (T+A+V) | Text + Audio + Video | No | 0.3356 | 0.3574 | 0.2481 |

### 7.2 Evaluation Metrics

The proposed system is evaluated using:

- **Accuracy:** Measures the overall percentage of correctly classified utterances.
- **Precision:** Measures the correctness of positive emotion predictions.
- **Recall:** Measures the ability to identify actual emotion categories.
- **Weighted F1-score:** Calculates F1-score considering class support.
- **Macro F1-score:** Measures average F1-score across all emotion classes equally.

Additional evaluation includes:
- Confusion matrix.
- Class-wise precision and recall.
- Three-seed performance analysis.
- Ablation studies.
- Qualitative error analysis.

> **Note:** The baseline values above are reported results. Proposed-model results should be added only after completing the actual training and test evaluation. Do not treat baseline results as proposed-model performance.

---

## 📈 8. Expected System Outputs

The system is designed to generate:

1. Predicted emotion category for each conversational utterance.
2. Text, Audio, and Video modality contribution percentages.
3. Important lexical tokens influencing predictions.
4. Contextual utterances influencing the current prediction.
5. Emotional progression timeline.
6. Detected descriptive emotional shifts.
7. Structured JSON and Markdown conversation summaries.

### Seven Emotion Categories

The MELD emotion labels are:

| Emotion | Description |
|---|---|
| Neutral | Emotionally neutral expression |
| Happiness | Positive emotional expression |
| Sadness | Sad emotional expression |
| Anger | Angry emotional expression |
| Fear | Fear-related emotional expression |
| Disgust | Disgust-related emotional expression |
| Surprise | Unexpected emotional expression |

---

## 🖥️ 9. Interactive Dashboard

The Streamlit dashboard is designed to provide an interactive interface for exploring conversational emotion analysis.

**Dashboard features:**

- Conversational utterance input.
- Multimodal emotion predictions.
- Modality contribution visualization.
- Token-level explanation.
- Context-level explanation.
- Emotion progression timeline.
- Structured conversation reports.

---

## 🔐 10. Ethics, Privacy and Limitations

This project is developed for research and reflective decision-support purposes.

Important limitations include:

- Emotion recognition does not reveal a person's actual internal emotional state.
- Facial expressions and vocal patterns may vary across individuals and cultures.
- MELD is based on acted conversational data and may not represent real therapeutic conversations.
- Model predictions can contain errors and biases.
- Missing or poor-quality modalities may reduce prediction reliability.
- Interpretability scores indicate model behavior, not clinical causation.

**Privacy considerations:**
- Avoid storing personally identifiable information.
- Obtain appropriate consent before processing real conversations.
- Protect audio and video recordings.
- Restrict access to generated reports.
- Require qualified professional interpretation of any decision-support information.

The system must not provide diagnoses, treatment recommendations, or autonomous mental-health decisions.

---

## 📚 11. Dataset and References

### Dataset

**MELD: Multimodal EmotionLines Dataset**

Poria, S., et al. (2019). *MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations.*

- Dataset: https://github.com/SenticNet/MELD
- Research Paper: https://aclanthology.org/P19-1050/

### Additional Research

- Wang et al. (2025) – CIME and text-enhanced multimodal emotion modeling.
- Research on Relational Graph Convolutional Networks.
- Research on Graph Transformers for conversational emotion recognition.
- Research on explainable multimodal machine learning.

---

## 🔮 12. Future Enhancements

- Improve minority emotion classification.
- Explore advanced multimodal fusion mechanisms.
- Improve robustness to missing modalities.
- Integrate real-time conversational analysis.
- Enhance explanation quality and reliability.
- Evaluate on additional conversational datasets.
- Conduct expert-led usability and ethical evaluations.

---

## 👩‍💻 13. Project Information

**Project Title:** Interpretable Contextual Semantic Analysis of Multimodal Emotions for Therapy Support

**Domain:** Artificial Intelligence | Deep Learning | Multimodal Emotion Recognition | Explainable AI

**Primary Dataset:** MELD

**Framework:** PyTorch and Streamlit

---

## ⭐ Acknowledgements

We acknowledge the researchers who developed the MELD dataset and the open-source communities supporting multimodal machine learning, conversational AI, and explainable artificial intelligence.

---

<p align="center">
  <b>Interpretable AI for Better Understanding of Conversational Emotions</b>
</p>

<p align="center">
  ⚠️ Research and decision-support only — Not a clinical diagnostic system.
</p>
