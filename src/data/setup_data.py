from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from pathlib import Path

from src.utils import get_logger

logger = get_logger("setup_data")

MELD_DEV_URL = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv"
MELD_TEST_URL = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv"


def prepare_raw_data(data_root: str | Path = "data") -> None:
    data_path = Path(data_root)
    raw_path = data_path / "raw"
    interim_path = data_path / "interim"
    features_path = data_path / "features"
    outputs_path = Path("outputs")

    for p in [raw_path, interim_path, features_path, outputs_path / "metrics", outputs_path / "figures", outputs_path / "checkpoints"]:
        p.mkdir(parents=True, exist_ok=True)

    # 1. Extract train_sent_emo.csv from archive.zip if needed
    archive_zip = Path("archive.zip")
    target_train_csv = raw_path / "train_sent_emo.csv"
    if not target_train_csv.exists() and archive_zip.exists():
        logger.info(f"Extracting train_sent_emo.csv from {archive_zip}...")
        with zipfile.ZipFile(archive_zip, "r") as z:
            if "train_sent_emo.csv" in z.namelist():
                content = z.read("train_sent_emo.csv")
                with target_train_csv.open("wb") as f:
                    f.write(content)
                logger.info(f"Extracted train_sent_emo.csv ({len(content)} bytes) to {target_train_csv}")
    elif target_train_csv.exists():
        logger.info(f"train_sent_emo.csv already exists at {target_train_csv}")

    # 2. Download dev_sent_emo.csv if needed
    target_dev_csv = raw_path / "dev_sent_emo.csv"
    if not target_dev_csv.exists():
        logger.info(f"Downloading official MELD dev split from {MELD_DEV_URL}...")
        try:
            with urllib.request.urlopen(MELD_DEV_URL, timeout=30) as resp:
                data = resp.read()
                with target_dev_csv.open("wb") as f:
                    f.write(data)
            logger.info(f"Downloaded dev_sent_emo.csv ({len(data)} bytes) to {target_dev_csv}")
        except Exception as e:
            logger.error(f"Failed to download dev_sent_emo.csv: {e}")
    else:
        logger.info(f"dev_sent_emo.csv already exists at {target_dev_csv}")

    # 3. Download test_sent_emo.csv if needed
    target_test_csv = raw_path / "test_sent_emo.csv"
    if not target_test_csv.exists():
        logger.info(f"Downloading official MELD test split from {MELD_TEST_URL}...")
        try:
            with urllib.request.urlopen(MELD_TEST_URL, timeout=30) as resp:
                data = resp.read()
                with target_test_csv.open("wb") as f:
                    f.write(data)
            logger.info(f"Downloaded test_sent_emo.csv ({len(data)} bytes) to {target_test_csv}")
        except Exception as e:
            logger.error(f"Failed to download test_sent_emo.csv: {e}")
    else:
        logger.info(f"test_sent_emo.csv already exists at {target_test_csv}")

    logger.info("Raw data setup completed.")


if __name__ == "__main__":
    prepare_raw_data()
