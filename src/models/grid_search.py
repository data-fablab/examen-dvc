from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with PARAMS_PATH.open("r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    X_train = pd.read_csv(PROCESSED_DIR / "X_train_scaled.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze("columns")

    model = RandomForestRegressor(
        random_state=params["model"]["random_state"],
        n_jobs=-1,
    )

    search = GridSearchCV(
        estimator=model,
        param_grid=params["model"]["param_grid"],
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    result = {
        "best_params": search.best_params_,
        "best_score_neg_mse": search.best_score_,
    }
    joblib.dump(result, MODELS_DIR / "best_params.pkl")

    print(f"Best parameters saved: {result}")


if __name__ == "__main__":
    main()
