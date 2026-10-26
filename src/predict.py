from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"


def predict(sepal_length, sepal_width, petal_length, petal_width):
    model = joblib.load(MODEL_PATH)
    sample = pd.DataFrame([{
        "sepal length (cm)": sepal_length,
        "sepal width (cm)": sepal_width,
        "petal length (cm)": petal_length,
        "petal width (cm)": petal_width,
    }])
    return int(model.predict(sample)[0])


if __name__ == "__main__":
    print("Predicted Iris class:", predict(5.1, 3.5, 1.4, 0.2))
