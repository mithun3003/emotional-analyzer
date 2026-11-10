from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from src.features.extract_features import extract_and_cache_features
from src.utils import get_logger, load_config

logger = get_logger("dataset")


class DialogueSample:
    """Represents a single conversation dialogue containing sequential utterances and context graph."""

    def __init__(
        self,
        dialogue_id: int,
        utterances: List[Dict[str, Any]],
        past_window: int = 4,
        future_window: int = 1,
    ):
        self.dialogue_id = dialogue_id
        self.utterances = utterances
        self.num_utterances = len(utterances)
        self.past_window = past_window
        self.future_window = future_window

        # Build tensors
        self.text_feats = torch.stack([u["text"] for u in utterances])   # [N, text_dim]
        self.audio_feats = torch.stack([u["audio"] for u in utterances]) # [N, audio_dim]
        self.video_feats = torch.stack([u["video"] for u in utterances]) # [N, video_dim]

        self.missing_audio = torch.tensor([u["missing_audio"] for u in utterances], dtype=torch.bool)
        self.missing_video = torch.tensor([u["missing_video"] for u in utterances], dtype=torch.bool)
        self.labels = torch.tensor([u["emotion_idx"] for u in utterances], dtype=torch.long)
        self.speakers = [u["speaker"] for u in utterances]
        self.texts = [u["utterance"] for u in utterances]
        self.emotions = [u["emotion"] for u in utterances]

        # Build context graph edges
        self.edge_index, self.edge_type = self._build_graph()

    def _build_graph(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct intra-speaker and inter-speaker edges within the context window.

        Edge Types:
            0: Intra-speaker relation (same speaker)
            1: Inter-speaker relation (different speaker)
        """
        src_nodes: List[int] = []
        dst_nodes: List[int] = []
        edge_types: List[int] = []

        N = self.num_utterances
        for i in range(N):
            # Self-loop: always intra-speaker
            src_nodes.append(i)
            dst_nodes.append(i)
            edge_types.append(0)

            # Window of past and future utterances
            start_j = max(0, i - self.past_window)
            end_j = min(N, i + self.future_window + 1)

            for j in range(start_j, end_j):
                if i == j:
                    continue
                src_nodes.append(j)
                dst_nodes.append(i) # edge from context j to target i
                # Intra-speaker vs inter-speaker
                if self.speakers[i] == self.speakers[j]:
                    edge_types.append(0) # intra-speaker
                else:
                    edge_types.append(1) # inter-speaker

        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        return edge_index, edge_type


class MELDDialogueDataset(Dataset):
    """Dataset of MELD dialogues with multimodal features and conversational graphs."""

    def __init__(
        self,
        features_dict: Dict[Tuple[int, int], Dict[str, Any]],
        past_window: int = 4,
        future_window: int = 1,
    ):
        self.past_window = past_window
        self.future_window = future_window

        # Group by dialogue_id
        dialogues_map: Dict[int, List[Dict[str, Any]]] = {}
        for (dia_id, utt_id), utt_data in features_dict.items():
            if dia_id not in dialogues_map:
                dialogues_map[dia_id] = []
            dialogues_map[dia_id].append(utt_data)

        # Sort utterances by utterance_id within each dialogue
        self.dialogues: List[DialogueSample] = []
        for dia_id in sorted(dialogues_map.keys()):
            utts = sorted(dialogues_map[dia_id], key=lambda x: x["utterance_id"])
            if len(utts) > 0:
                self.dialogues.append(
                    DialogueSample(
                        dialogue_id=dia_id,
                        utterances=utts,
                        past_window=past_window,
                        future_window=future_window,
                    )
                )

    def __len__(self) -> int:
        return len(self.dialogues)

    def __getitem__(self, idx: int) -> DialogueSample:
        return self.dialogues[idx]


def collate_dialogues(batch: List[DialogueSample]) -> List[DialogueSample]:
    """Collate list of DialogueSample objects."""
    return batch


def get_dataloaders(
    config_path: str = "configs/default.yaml",
    batch_size: int | None = None,
    limit: int | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    config = load_config(config_path)
    b_size = batch_size or config["train"]["batch_size"]
    past_w = config["model"]["past_window"]
    future_w = config["model"]["future_window"]

    data_splits = extract_and_cache_features(config_path=config_path, limit=limit)

    train_ds = MELDDialogueDataset(data_splits["train"], past_window=past_w, future_window=future_w)
    dev_ds = MELDDialogueDataset(data_splits["dev"], past_window=past_w, future_window=future_w)
    test_ds = MELDDialogueDataset(data_splits["test"], past_window=past_w, future_window=future_w)

    train_loader = DataLoader(train_ds, batch_size=b_size, shuffle=True, collate_fn=collate_dialogues)
    dev_loader = DataLoader(dev_ds, batch_size=b_size, shuffle=False, collate_fn=collate_dialogues)
    test_loader = DataLoader(test_ds, batch_size=b_size, shuffle=False, collate_fn=collate_dialogues)

    logger.info(
        f"Created DataLoaders: Train dialogues={len(train_ds)}, Dev dialogues={len(dev_ds)}, Test dialogues={len(test_ds)}"
    )
    return train_loader, dev_loader, test_loader, config


def smoke_test(config_path: str = "configs/default.yaml") -> bool:
    """Smoke test to verify loading one batch with all modalities."""
    logger.info("Executing Phase 2 smoke test...")
    train_loader, dev_loader, test_loader, config = get_dataloaders(config_path=config_path, limit=20)

    batch = next(iter(train_loader))
    assert len(batch) > 0, "Batch is empty"
    sample = batch[0]

    logger.info(f"Sample Dialogue ID: {sample.dialogue_id}")
    logger.info(f"Num Utterances: {sample.num_utterances}")
    logger.info(f"Text tensor shape: {sample.text_feats.shape}")
    logger.info(f"Audio tensor shape: {sample.audio_feats.shape}")
    logger.info(f"Video tensor shape: {sample.video_feats.shape}")
    logger.info(f"Missing audio flags: {sample.missing_audio.tolist()[:5]}")
    logger.info(f"Missing video flags: {sample.missing_video.tolist()[:5]}")
    logger.info(f"Graph edge_index shape: {sample.edge_index.shape}")
    logger.info(f"Graph edge_type distribution: {torch.bincount(sample.edge_type).tolist()}")

    assert sample.text_feats.ndim == 2, "Text features must be 2D [N, text_dim]"
    assert sample.audio_feats.ndim == 2, "Audio features must be 2D [N, audio_dim]"
    assert sample.video_feats.ndim == 2, "Video features must be 2D [N, video_dim]"
    assert sample.edge_index.shape[0] == 2, "edge_index must have shape [2, E]"
    assert sample.edge_type.shape[0] == sample.edge_index.shape[1], "edge_type must match edge count"

    logger.info("Phase 2 smoke test PASSED successfully!")
    return True


if __name__ == "__main__":
    smoke_test()
