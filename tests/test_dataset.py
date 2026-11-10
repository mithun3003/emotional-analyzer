import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import load_dataset
from src.pipeline import run_step_by_step


def test_dataset_loading_and_split():
    dataset = load_dataset()
    assert len(dataset["train"]) > 0
    assert len(dataset["val"]) > 0
    assert len(dataset["test"]) > 0
    assert dataset["train"][0][0] != ""
    assert dataset["train"][0][1] in {"neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"}


def test_step_by_step_pipeline():
    summary = run_step_by_step(verbose=False)
    assert len(summary["steps"]) == 4
    assert summary["accuracy"] >= 0
    assert summary["macro_f1"] >= 0
