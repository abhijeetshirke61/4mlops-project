from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocess import download_data, preprocess
from train import train


def test_training_pipeline():
    download_data()
    preprocess()
    accuracy = train()
    assert 0.0 <= accuracy <= 1.0
    assert (ROOT / "models" / "model.joblib").exists()
    assert (ROOT / "models" / "metrics.json").exists()
