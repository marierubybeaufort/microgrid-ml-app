import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from backend.app.ml.fault_detection.preprocessing import (
    FEATURE_COLUMNS,
    prepare_fault_data,
)


DATA_PATH = Path("data/community_households.csv")

ARTIFACT_DIR = Path(
    "backend/artifacts/fault_detection"
)

MODEL_PATH = ARTIFACT_DIR / "model.joblib"
CONFIG_PATH = ARTIFACT_DIR / "config.json"

RANDOM_STATE = 42
N_ESTIMATORS = 500
NORMAL_ALERT_QUANTILE = 0.99


def main() -> None:
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = prepare_fault_data(DATA_PATH)

    x = df[
        list(FEATURE_COLUMNS)
    ].to_numpy()

    y = df["fault"].to_numpy()

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        bootstrap=True,
        oob_score=True,
    )

    model.fit(x, y)

    oob_probabilities = (
        model.oob_decision_function_[:, 1]
    )

    normal_probabilities = (
        oob_probabilities[y == 0]
    )

    threshold = float(
        np.quantile(
            normal_probabilities,
            NORMAL_ALERT_QUANTILE,
        )
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    with open(
        CONFIG_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "features": list(FEATURE_COLUMNS),
                "threshold": threshold,
                "threshold_strategy": (
                    "99th percentile of OOB "
                    "normal-class probabilities"
                ),
                "normal_alert_quantile": (
                    NORMAL_ALERT_QUANTILE
                ),
                "n_estimators": N_ESTIMATORS,
                "random_state": RANDOM_STATE,
                "training_rows": int(len(df)),
                "training_faults": int(y.sum()),
                "training_fault_rate": float(
                    y.mean()
                ),
            },
            file,
            indent=2,
        )

    print("Training rows:", len(df))
    print("Faults:", int(y.sum()))
    print(
        "Threshold:",
        round(threshold, 6),
    )
    print("Saved:", MODEL_PATH)
    print("Saved:", CONFIG_PATH)


if __name__ == "__main__":
    main()
