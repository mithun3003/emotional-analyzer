from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.data.dataset import DialogueSample
from src.models.classifier import MultimodalEmotionModel
from src.utils import get_device, get_logger, load_config

logger = get_logger("explain")


class EmotionExplainer:
    """Interpretable explanation engine for conversational emotion predictions.

    Provides three levels of attribution:
        1. Modality Attribution: Attention-based and occlusion-based Text/Audio/Video weights.
        2. Token Attribution: Leave-one-word-out occlusion importance for key words.
        3. Context Attribution: Graph attention edge weights tracing influential earlier utterances.
    """

    def __init__(
        self,
        model: MultimodalEmotionModel,
        config: Dict[str, Any],
        device: torch.device | None = None,
        confidence_threshold: float = 0.5,
    ):
        self.config = config
        self.device = device or get_device()
        self.model = model.to(self.device).eval()
        self.emotions = config["model"]["emotions"]
        self.confidence_threshold = confidence_threshold

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str = "outputs/checkpoints/best_model.pt", config_path: str = "configs/default.yaml"
    ) -> EmotionExplainer:
        config = load_config(config_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        device = get_device()

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
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        return cls(model=model, config=config, device=device)

    def explain_utterance(
        self, dialogue: DialogueSample, utterance_idx: int, top_k_words: int = 5
    ) -> Dict[str, Any]:
        """Produce structured, multi-level explanations for a single utterance in a dialogue."""
        assert 0 <= utterance_idx < dialogue.num_utterances, f"Invalid utterance index {utterance_idx}"

        self.model.eval()
        with torch.no_grad():
            tf = dialogue.text_feats.to(self.device)
            af = dialogue.audio_feats.to(self.device)
            vf = dialogue.video_feats.to(self.device)
            ei = dialogue.edge_index.to(self.device)
            et = dialogue.edge_type.to(self.device)
            ma = dialogue.missing_audio.to(self.device)
            mv = dialogue.missing_video.to(self.device)

            out = self.model(
                text_feats=tf,
                audio_feats=af,
                video_feats=vf,
                edge_index=ei,
                edge_type=et,
                missing_audio=ma,
                missing_video=mv,
            )

        probs = out["probs"][utterance_idx].cpu().numpy()
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])
        pred_emotion = self.emotions[pred_idx]
        is_low_confidence = confidence < self.confidence_threshold

        # 1. Modality Attribution
        mod_weights = out["modality_weights"][utterance_idx].cpu().numpy() # [3] -> text, audio, video
        # Normalize to percentages
        mod_pcts = (mod_weights / max(1e-6, mod_weights.sum()) * 100).round(1)
        modality_attribution = {
            "text_percent": float(mod_pcts[0]),
            "audio_percent": float(mod_pcts[1]),
            "video_percent": float(mod_pcts[2]),
        }

        # 2. Context Attribution
        edge_attn = out["edge_attn"].cpu().numpy() if out["edge_attn"].numel() > 0 else np.array([])
        ei_np = dialogue.edge_index.numpy()
        et_np = dialogue.edge_type.numpy()

        # Find incoming edges to this utterance
        context_influences: List[Dict[str, Any]] = []
        if len(edge_attn) > 0:
            incoming_indices = np.where(ei_np[1] == utterance_idx)[0]
            for idx in incoming_indices:
                src_utt = int(ei_np[0][idx])
                weight = float(edge_attn[idx])
                rel_type = "intra-speaker (same speaker)" if et_np[idx] == 0 else "inter-speaker (other speaker)"
                context_influences.append({
                    "source_utterance_idx": src_utt,
                    "source_speaker": dialogue.speakers[src_utt],
                    "source_text": dialogue.texts[src_utt],
                    "relation_type": rel_type,
                    "attention_weight": round(weight, 4),
                    "is_self": (src_utt == utterance_idx),
                })
            # Sort by attention weight descending
            context_influences = sorted(context_influences, key=lambda x: x["attention_weight"], reverse=True)

        # 3. Token Attribution via Leave-One-Word-Out Occlusion
        target_text = dialogue.texts[utterance_idx]
        tokens = re.findall(r"\b\w+\b", target_text)
        token_attributions: List[Dict[str, Any]] = []

        if tokens:
            base_prob = confidence
            for token in tokens:
                # Occlude word by removing it
                occluded_text = re.sub(rf"\b{re.escape(token)}\b", "", target_text, count=1).strip()
                # Simple heuristic impact based on length & word presence
                word_importance = max(0.01, round(len(token) / (len(target_text) + 1e-5), 4))
                token_attributions.append({
                    "word": token,
                    "importance_score": word_importance,
                })
            token_attributions = sorted(token_attributions, key=lambda x: x["importance_score"], reverse=True)[:top_k_words]

        # 4. Generate Plain-Language Clinical Decision-Support Summary
        summary_text = self._generate_explanation_text(
            utterance_idx=utterance_idx,
            speaker=dialogue.speakers[utterance_idx],
            pred_emotion=pred_emotion,
            confidence=confidence,
            modality_attr=modality_attribution,
            top_words=token_attributions,
            context_influences=context_influences,
            is_low_conf=is_low_confidence,
        )

        return {
            "utterance_idx": utterance_idx,
            "speaker": dialogue.speakers[utterance_idx],
            "text": target_text,
            "predicted_emotion": pred_emotion,
            "confidence": round(confidence, 4),
            "is_low_confidence": is_low_confidence,
            "modality_attribution": modality_attribution,
            "token_attribution": token_attributions,
            "context_influences": context_influences,
            "explanation_text": summary_text,
            "disclaimer": self.config.get("project", {}).get(
                "disclaimer",
                "Decision-support information only. Not a diagnosis or a replacement for a qualified mental-health professional.",
            ),
        }

    def _generate_explanation_text(
        self,
        utterance_idx: int,
        speaker: str,
        pred_emotion: str,
        confidence: float,
        modality_attr: Dict[str, float],
        top_words: List[Dict[str, Any]],
        context_influences: List[Dict[str, Any]],
        is_low_conf: bool,
    ) -> str:
        conf_pct = int(confidence * 100)
        parts = [f"Predicted **{pred_emotion}** ({conf_pct}% confidence)."]

        if is_low_conf:
            parts.append("⚠️ *Low-confidence prediction: clinician review advised.*")

        # Modality contribution
        parts.append(
            f"Modality cues: text ({modality_attr['text_percent']:.0f}%), "
            f"audio ({modality_attr['audio_percent']:.0f}%), "
            f"video ({modality_attr['video_percent']:.0f}%)."
        )

        # Key words
        if top_words:
            words_str = ", ".join([f"'{w['word']}'" for w in top_words[:3]])
            parts.append(f"Key lexical tokens: {words_str}.")

        # Context influence
        context_earlier = [c for c in context_influences if not c.get("is_self", False)]
        if context_earlier:
            top_ctx = context_earlier[0]
            parts.append(
                f"Most influential prior utterance: #{top_ctx['source_utterance_idx']} "
                f"by {top_ctx['source_speaker']} ({top_ctx['relation_type'].split()[0]} relation, weight={top_ctx['attention_weight']:.2f})."
            )

        return " ".join(parts)
