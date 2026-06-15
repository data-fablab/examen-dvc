import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "metrics"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "prediction.csv"


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    X_test = pd.read_csv(PROCESSED_DIR / "X_test_scaled.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")
    model = joblib.load(MODELS_DIR / "my_model.pkl")

    y_pred = model.predict(X_test)

    predictions = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": y_pred,
        }
    )
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    mse = mean_squared_error(y_test, y_pred)
    scores = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

    with (METRICS_DIR / "scores.json").open("w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    print(f"Evaluation scores saved to {METRICS_DIR / 'scores.json'}: {scores}")


if __name__ == "__main__":
    main()
