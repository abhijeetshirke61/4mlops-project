from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"


def download_data():
    iris = load_iris(as_frame=True)
    df = iris.frame.rename(columns={"target": "target"})
    df["target"] = df["target"].astype(int)
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA / "data.csv", index=False)
    print(f"Saved {len(df)} rows to data/data.csv")


def preprocess(raw_path=None, output_dir=None):
    raw_path = Path(raw_path) if raw_path else DATA / "data.csv"
    output_dir = Path(output_dir) if output_dir else PROCESSED
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw_path).dropna()
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train = X_train.copy()
    train["target"] = y_train
    test = X_test.copy()
    test["target"] = y_test
    train.to_csv(output_dir / "train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    print(f"Preprocessing complete. Train: {len(train)}, Test: {len(test)}")


if __name__ == "__main__":
    download_data()
    preprocess()
