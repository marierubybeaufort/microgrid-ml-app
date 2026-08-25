from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from backend.app.ml.fault_detection.preprocessing import (
    FEATURE_COLUMNS,
    prepare_fault_data,
)


FAULTY_HOUSEHOLDS = (
    "House_01",
    "House_07",
    "House_13",
)


def choose_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Set threshold from the training normal-score distribution.

    The 99th percentile limits the nominal false-positive
    alert budget to roughly 1% on training normal observations.
    """

    normal_probabilities = probabilities[y_true == 0]

    if len(normal_probabilities) == 0:
        return 0.5

    return float(
        np.quantile(normal_probabilities, 0.99)
    )


def evaluate_fault_model(
    path: str | Path,
) -> list[dict[str, float | str]]:
    """Evaluate on unseen faulty households with training-only threshold tuning."""

    df = prepare_fault_data(path)

    results = []

    for test_household in FAULTY_HOUSEHOLDS:
        train_df = df[
            df["household_id"] != test_household
        ]

        test_df = df[
            df["household_id"] == test_household
        ]

        x_train = train_df[
            list(FEATURE_COLUMNS)
        ].to_numpy()

        y_train = train_df["fault"].to_numpy()

        x_test = test_df[
            list(FEATURE_COLUMNS)
        ].to_numpy()

        y_test = test_df["fault"].to_numpy()

        model = RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            oob_score=True,
        )

        model.fit(x_train, y_train)

        # OOB probabilities are generated using trees that did not
        # train on each respective training observation.
        oob_probabilities = (
            model.oob_decision_function_[:, 1]
        )

        threshold = choose_threshold(
            y_train,
            oob_probabilities,
        )

        probabilities = model.predict_proba(
            x_test
        )[:, 1]

        predictions = (
            probabilities >= threshold
        ).astype(int)

        results.append(
            {
                "household": test_household,
                "threshold": threshold,
                "precision": precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "pr_auc": average_precision_score(
                    y_test,
                    probabilities,
                ),
                "faults": int(y_test.sum()),
                "predicted_faults": int(
                    predictions.sum()
                ),
                "max_probability": float(
                    probabilities.max()
                ),
            }
        )

    return results


if __name__ == "__main__":
    results = evaluate_fault_model(
        "data/community_households.csv"
    )

    for result in results:
        print()
        print("===", result["household"], "===")
        print("Faults:", result["faults"])
        print(
            "Predicted faults:",
            result["predicted_faults"],
        )
        print(
            "Threshold:",
            round(result["threshold"], 4),
        )
        print(
            "Max probability:",
            round(result["max_probability"], 4),
        )
        print(
            "Precision:",
            round(result["precision"], 3),
        )
        print(
            "Recall:",
            round(result["recall"], 3),
        )
        print(
            "F1:",
            round(result["f1"], 3),
        )
        print(
            "PR-AUC:",
            round(result["pr_auc"], 3),
        )

    print()
    print(
        "Mean F1:",
        round(
            np.mean(
                [r["f1"] for r in results]
            ),
            3,
        ),
    )

    print(
        "Mean PR-AUC:",
        round(
            np.mean(
                [r["pr_auc"] for r in results]
            ),
            3,
        ),
    )
