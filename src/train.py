from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]


def train(processed_dir=None, model_dir=None):
    processed_dir = Path(processed_dir) if processed_dir else ROOT / "data" / "processed"
    model_dir = Path(model_dir) if model_dir else ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(processed_dir / "train.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))

    joblib.dump(model, model_dir / "model.joblib")
    metrics = {
        "accuracy": accuracy,
        "report": classification_report(y_test, predictions, output_dict=True),
    }
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(f"Model trained. Accuracy: {accuracy:.4f}")
    return accuracy


if __name__ == "__main__":
    train()
