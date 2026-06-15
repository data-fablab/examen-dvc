from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
    )

    X_train_scaled.to_csv(PROCESSED_DIR / "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(PROCESSED_DIR / "X_test_scaled.csv", index=False)

    print(
        "Normalization complete: "
        f"X_train_scaled={X_train_scaled.shape}, X_test_scaled={X_test_scaled.shape}"
    )


if __name__ == "__main__":
    main()
