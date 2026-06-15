from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "raw.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
TARGET = "silica_concentrate"


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with PARAMS_PATH.open("r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    split_params = params["split"]
    df = pd.read_csv(RAW_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in {RAW_PATH}")

    X = df.drop(columns=[TARGET]).select_dtypes(include="number")
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_params["test_size"],
        random_state=split_params["random_state"],
    )

    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    print(
        "Split complete: "
        f"X_train={X_train.shape}, X_test={X_test.shape}, "
        f"y_train={y_train.shape}, y_test={y_test.shape}"
    )


if __name__ == "__main__":
    main()
