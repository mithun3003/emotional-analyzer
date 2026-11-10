# Ethics and Limitations Statement

## Clinical Purpose and Explicit Boundaries
This project is engineered strictly as a **therapy-support and conversational emotion analysis tool** for reflective review by human clinicians or researchers.

> **CRITICAL CLINICAL DISCLAIMER:**
> **Decision-support information only. Not a diagnosis or a replacement for a qualified mental-health professional.**
> This system must **never** output clinical diagnoses, disease labels (such as "major depressive disorder", "bipolar disorder", or "generalized anxiety disorder"), or any autonomous clinical claims.

---

## Dataset Characteristics and Bias
The underlying conversational model is evaluated on the Multimodal EmotionLines Dataset (**MELD**), derived from the television situation comedy series *Friends*. Consequently, several inherent biases and limitations apply:

1. **Acted vs. Natural Emotion**:
   - The dialogues are scripted, staged, and performed by professional actors for dramatic or comedic entertainment.
   - Pacing, exaggerated vocal modulations, theatrical facial expressions, and laughter cues deviate substantially from authentic, unscripted therapeutic conversations.

2. **Demographic and Cultural Homogeneity**:
   - The cast reflects a narrow demographic profile (predominantly young adult, Caucasian, North American, 1990s urban culture).
   - Cross-cultural, generational, socio-economic, and neurodivergent variations in affective expression are not represented.

3. **Acoustic and Visual Artifacts**:
   - The original videos feature background laughter tracks (laugh tracks), incidental studio applause, overlapping acoustic channels, dynamic camera zooms, and multi-speaker frames.
   - Such artifacts can confound unimodal audio and visual feature extraction.

---

## Modality Availability and Masking
- In multi-speaker television clips or real-world clinical sessions, specific modalities can be corrupted, occluded, or missing (e.g. client looking away from camera, microphone clipping or noise, multiple speakers in frame).
- Our architecture explicitly incorporates modality masking (`missing_audio`, `missing_video`), ensuring the system degrades gracefully into text-only or text-audio modes rather than hallucinating features.
- If a modality is absent or unusable, the system transparently reports the missing status rather than masking failure.

---

## Misclassification Risks & Imbalance
- MELD exhibits significant class imbalance: `neutral` accounts for nearly 50% of utterances, while `disgust` and `fear` represent less than 3% each.
- In therapeutic settings, confusing a subtle `sadness` or `fear` utterance for `neutral` could miss critical client vulnerability.
- Clinicians must never rely on automated predictions without reviewing the raw transcript and audio/video recordings. Confidence scores and multi-level explanations (token attributions, context relations) must always be inspected.

---

## Data Privacy and Local Processing
- Patient mental health discourse is sensitive, protected health information (PHI).
- All model inference, feature processing, and explanation generation in this system are executed locally on the client's machine.
- No client audio, video, or conversational transcripts are transmitted to external cloud APIs or stored remotely.
