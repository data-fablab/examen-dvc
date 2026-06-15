from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42


def main() -> None:
    X_train = pd.read_csv(PROCESSED_DIR / "X_train_scaled.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze("columns")
    search_result = joblib.load(MODELS_DIR / "best_params.pkl")

    model = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **search_result["best_params"],
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODELS_DIR / "my_model.pkl")

    print(f"Model trained and saved to {MODELS_DIR / 'my_model.pkl'}")


if __name__ == "__main__":
    main()
