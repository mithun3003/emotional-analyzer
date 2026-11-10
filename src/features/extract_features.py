from __future__ import annotations

import argparse
import io
import json
import pickle
import zipfile
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils import get_logger, load_config, set_seed

logger = get_logger("extract_features")


def extract_and_cache_features(
    config_path: str = "configs/default.yaml",
    limit: int | None = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    config = load_config(config_path)
    set_seed(config["train"]["seed"])

    features_dir = Path(config["data"]["features_dir"])
    metrics_dir = Path(config["outputs"]["metrics_dir"])
    features_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_cache = features_dir / f"train_features{'_limit' + str(limit) if limit else ''}.pt"
    dev_cache = features_dir / f"dev_features{'_limit' + str(limit) if limit else ''}.pt"
    test_cache = features_dir / f"test_features{'_limit' + str(limit) if limit else ''}.pt"

    if (
        not force_recompute
        and train_cache.exists()
        and dev_cache.exists()
        and test_cache.exists()
    ):
        logger.info("Found cached features for train, dev, and test. Loading cache...")
        train_data = torch.load(train_cache, weights_only=False)
        dev_data = torch.load(dev_cache, weights_only=False)
        test_data = torch.load(test_cache, weights_only=False)
        logger.info(f"Loaded: train={len(train_data)}, dev={len(dev_data)}, test={len(test_data)}")
        return {"train": train_data, "dev": dev_data, "test": test_data}

    logger.info("Extracting features from MELD CSVs...")
    df_train = pd.read_csv(config["data"]["train_csv"])
    df_dev = pd.read_csv(config["data"]["dev_csv"])
    df_test = pd.read_csv(config["data"]["test_csv"])

    # Clean columns
    for df in [df_train, df_dev, df_test]:
        df.columns = [c.strip() for c in df.columns]

    if limit is not None:
        logger.info(f"Applying limit={limit} utterances per split...")
        df_train = df_train.head(limit).copy()
        df_dev = df_dev.head(limit).copy()
        df_test = df_test.head(limit).copy()

    # 1. Text Feature Extraction (768-d text representation)
    logger.info("Building 768-d text semantic features using TF-IDF + TruncatedSVD...")
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X_train_sparse = vec.fit_transform(df_train["Utterance"].fillna(""))
    n_components = min(config["model"]["text_dim"], X_train_sparse.shape[1] - 1, len(df_train) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=config["train"]["seed"])
    text_feat_train = svd.fit_transform(X_train_sparse)
    text_feat_dev = svd.transform(vec.transform(df_dev["Utterance"].fillna("")))
    text_feat_test = svd.transform(vec.transform(df_test["Utterance"].fillna("")))

    # Pad if n_components < 768
    target_dim = config["model"]["text_dim"]
    if text_feat_train.shape[1] < target_dim:
        pad_width = target_dim - text_feat_train.shape[1]
        text_feat_train = np.pad(text_feat_train, ((0, 0), (0, pad_width)))
        text_feat_dev = np.pad(text_feat_dev, ((0, 0), (0, pad_width)))
        text_feat_test = np.pad(text_feat_test, ((0, 0), (0, pad_width)))

    # 2. Audio Feature Extraction (128-d)
    audio_dim = config["model"]["audio_dim"]
    archive_zip = Path(config["data"]["archive_zip"])
    train_audio_embeddings = None
    if archive_zip.exists():
        logger.info("Loading pre-extracted audio embeddings from archive.zip for train split...")
        try:
            with zipfile.ZipFile(archive_zip, "r") as z:
                if "audi_embeddings.pkl" in z.namelist():
                    data = pickle.loads(z.read("audi_embeddings.pkl"))
                    arr = np.asarray(data, dtype=np.float32)
                    if limit is not None:
                        arr = arr[:limit]
                    # Reduce from 1611 to audio_dim (128) via SVD or pooling
                    if arr.shape[1] > audio_dim:
                        audio_svd = TruncatedSVD(n_components=audio_dim, random_state=42)
                        train_audio_embeddings = audio_svd.fit_transform(arr)
                    else:
                        train_audio_embeddings = arr
                    logger.info(f"Processed train audio embeddings shape: {train_audio_embeddings.shape}")
        except Exception as e:
            logger.warning(f"Failed to read audi_embeddings.pkl from archive: {e}")

    # Build emotion mapping
    emotions = config["model"]["emotions"]
    emotion_to_idx = {emo.lower(): idx for idx, emo in enumerate(emotions)}

    def build_split_dict(
        df: pd.DataFrame,
        text_feats: np.ndarray,
        audio_feats: np.ndarray | None,
        is_train: bool,
    ) -> Dict[Tuple[int, int], Dict[str, Any]]:
        split_dict = {}
        for idx, row in df.iterrows():
            dia_id = int(row.get("Dialogue_ID", 0))
            utt_id = int(row.get("Utterance_ID", idx))
            key = (dia_id, utt_id)

            t_vec = torch.tensor(text_feats[idx], dtype=torch.float32)

            if audio_feats is not None and idx < len(audio_feats):
                a_vec = torch.tensor(audio_feats[idx], dtype=torch.float32)
                missing_audio = False
            else:
                a_vec = torch.zeros(audio_dim, dtype=torch.float32)
                missing_audio = True

            # Video is missing in raw archive -> placeholder zero vector + mask flag
            video_dim = config["model"]["video_dim"]
            v_vec = torch.zeros(video_dim, dtype=torch.float32)
            missing_video = True

            emo_str = str(row.get("Emotion", "neutral")).strip().lower()
            emo_idx = emotion_to_idx.get(emo_str, 4) # default neutral

            split_dict[key] = {
                "dialogue_id": dia_id,
                "utterance_id": utt_id,
                "text": t_vec,
                "audio": a_vec,
                "video": v_vec,
                "missing_audio": missing_audio,
                "missing_video": missing_video,
                "speaker": str(row.get("Speaker", "Unknown")),
                "emotion": emo_str,
                "emotion_idx": emo_idx,
                "utterance": str(row.get("Utterance", "")),
                "sentiment": str(row.get("Sentiment", "neutral")),
            }
        return split_dict

    train_data = build_split_dict(df_train, text_feat_train, train_audio_embeddings, is_train=True)
    dev_data = build_split_dict(df_dev, text_feat_dev, None, is_train=False)
    test_data = build_split_dict(df_test, text_feat_test, None, is_train=False)

    # Save to disk
    logger.info(f"Saving features to {features_dir}...")
    torch.save(train_data, train_cache)
    torch.save(dev_data, dev_cache)
    torch.save(test_data, test_cache)

    # Generate feature extraction report
    report = {
        "train": {
            "total_utterances": len(train_data),
            "missing_audio_count": sum(1 for v in train_data.values() if v["missing_audio"]),
            "missing_video_count": sum(1 for v in train_data.values() if v["missing_video"]),
            "text_dim": target_dim,
            "audio_dim": audio_dim,
            "video_dim": config["model"]["video_dim"],
        },
        "dev": {
            "total_utterances": len(dev_data),
            "missing_audio_count": sum(1 for v in dev_data.values() if v["missing_audio"]),
            "missing_video_count": sum(1 for v in dev_data.values() if v["missing_video"]),
            "text_dim": target_dim,
            "audio_dim": audio_dim,
            "video_dim": config["model"]["video_dim"],
        },
        "test": {
            "total_utterances": len(test_data),
            "missing_audio_count": sum(1 for v in test_data.values() if v["missing_audio"]),
            "missing_video_count": sum(1 for v in test_data.values() if v["missing_video"]),
            "text_dim": target_dim,
            "audio_dim": audio_dim,
            "video_dim": config["model"]["video_dim"],
        },
    }

    report_path = metrics_dir / "feature_extraction_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved feature extraction report to {report_path}")

    return {"train": train_data, "dev": dev_data, "test": test_data, "report": report}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and cache multimodal features")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    extract_and_cache_features(config_path=args.config, limit=args.limit, force_recompute=args.force)
