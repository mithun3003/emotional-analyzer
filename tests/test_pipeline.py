from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.dataset import DialogueSample, MELDDialogueDataset
from src.explain.explain import EmotionExplainer
from src.models.classifier import MultimodalEmotionModel
from src.models.fusion import TextEnhancedAttention
from src.models.graph_context import DialogueGraphContext, RelationalGraphConvLayer


def _create_mock_dialogue() -> DialogueSample:
    utts = [
        {
            "utterance_id": 0,
            "speaker": "Rachel",
            "text": "Hey, what are you doing here?",
            "emotion": "surprise",
            "emotion_idx": 6,
            "text": torch.randn(768),
            "audio": torch.randn(128),
            "video": torch.randn(128),
            "missing_audio": False,
            "missing_video": True,
            "utterance": "Hey, what are you doing here?",
        },
        {
            "utterance_id": 1,
            "speaker": "Ross",
            "text": "I was just waiting for you.",
            "emotion": "neutral",
            "emotion_idx": 4,
            "text": torch.randn(768),
            "audio": torch.randn(128),
            "video": torch.randn(128),
            "missing_audio": False,
            "missing_video": True,
            "utterance": "I was just waiting for you.",
        },
        {
            "utterance_id": 2,
            "speaker": "Rachel",
            "text": "I am so happy to see you!",
            "emotion": "joy",
            "emotion_idx": 3,
            "text": torch.randn(768),
            "audio": torch.randn(128),
            "video": torch.randn(128),
            "missing_audio": False,
            "missing_video": True,
            "utterance": "I am so happy to see you!",
        },
    ]
    return DialogueSample(dialogue_id=999, utterances=utts, past_window=4, future_window=1)


def test_dialogue_sample_shapes():
    sample = _create_mock_dialogue()
    assert sample.num_utterances == 3
    assert sample.text_feats.shape == (3, 768)
    assert sample.audio_feats.shape == (3, 128)
    assert sample.video_feats.shape == (3, 128)
    assert sample.missing_audio.shape == (3,)
    assert sample.missing_video.shape == (3,)
    assert sample.labels.shape == (3,)


def test_graph_construction_relations():
    sample = _create_mock_dialogue()
    edge_index, edge_type = sample.edge_index, sample.edge_type

    assert edge_index.ndim == 2
    assert edge_index.shape[0] == 2
    assert edge_type.shape[0] == edge_index.shape[1]

    # At least self-loops (relation 0) must exist
    assert 0 in edge_type.tolist()
    # Inter-speaker edge between Rachel and Ross (relation 1) must exist
    assert 1 in edge_type.tolist()


def test_text_enhanced_attention_fusion():
    tea = TextEnhancedAttention(d_model=128, n_heads=4)
    t = torch.randn(3, 128)
    a = torch.randn(3, 128)
    v = torch.randn(3, 128)
    ma = torch.tensor([False, False, True])
    mv = torch.tensor([True, True, True])

    fused, weights = tea(t, a, v, missing_audio=ma, missing_video=mv)
    assert fused.shape == (3, 128)
    assert weights.shape == (3, 3)
    # Video contribution must be near zero when masked
    assert weights[0, 2].item() < 0.05


def test_dialogue_graph_context():
    gc = DialogueGraphContext(d_model=128, n_heads=4)
    x = torch.randn(3, 128)
    edge_index = torch.tensor([[0, 1, 2, 0, 1], [0, 1, 2, 1, 2]], dtype=torch.long)
    edge_type = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

    out, edge_attn = gc(x, edge_index, edge_type)
    assert out.shape == (3, 128)
    assert edge_attn.shape == (5,)


def test_multimodal_model_forward():
    model = MultimodalEmotionModel(
        text_dim=768, audio_dim=128, video_dim=128, d_model=128, num_classes=7
    )
    sample = _create_mock_dialogue()
    out = model(
        text_feats=sample.text_feats,
        audio_feats=sample.audio_feats,
        video_feats=sample.video_feats,
        edge_index=sample.edge_index,
        edge_type=sample.edge_type,
        missing_audio=sample.missing_audio,
        missing_video=sample.missing_video,
    )

    assert out["logits"].shape == (3, 7)
    assert out["probs"].shape == (3, 7)
    assert out["modality_weights"].shape == (3, 3)
    # Probabilities must sum to 1
    assert torch.allclose(out["probs"].sum(dim=-1), torch.ones(3), atol=1e-4)


def test_explanation_schema_and_sensitivity():
    model = MultimodalEmotionModel(
        text_dim=768, audio_dim=128, video_dim=128, d_model=128, num_classes=7
    )
    config = {
        "model": {"emotions": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]},
        "project": {"disclaimer": "Test disclaimer"},
    }
    explainer = EmotionExplainer(model=model, config=config)

    sample = _create_mock_dialogue()
    exp = explainer.explain_utterance(sample, 0)

    # Schema checks
    assert "predicted_emotion" in exp
    assert "confidence" in exp
    assert "modality_attribution" in exp
    assert "token_attribution" in exp
    assert "context_influences" in exp
    assert "explanation_text" in exp
    assert "disclaimer" in exp

    # Sensitivity check: change text features and check that explanation responds
    sample_perturbed = _create_mock_dialogue()
    sample_perturbed.text_feats[0] = sample_perturbed.text_feats[0] + 5.0
    exp_perturbed = explainer.explain_utterance(sample_perturbed, 0)

    # Confidence or logits should change
    assert (exp["confidence"] != exp_perturbed["confidence"]) or (
        exp["predicted_emotion"] != exp_perturbed["predicted_emotion"]
    )
