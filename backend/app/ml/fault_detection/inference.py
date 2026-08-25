import json
from pathlib import Path

import joblib
import numpy as np


ARTIFACT_DIR = Path(
    "backend/artifacts/fault_detection"
)


class FaultPredictor:
    """Load the saved fault detector and score current fault risk."""

    def __init__(
        self,
        artifact_dir: str | Path = ARTIFACT_DIR,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)

        with open(
            self.artifact_dir / "config.json",
            encoding="utf-8",
        ) as file:
            self.config = json.load(file)

        self.model = joblib.load(
            self.artifact_dir / "model.joblib"
        )

        self.features = tuple(
            self.config["features"]
        )

        self.threshold = float(
            self.config["threshold"]
        )

    def _build_features(
        self,
        generation_history: np.ndarray,
    ) -> np.ndarray:
        history = np.asarray(
            generation_history,
            dtype=np.float64,
        )

        if history.shape != (13,):
            raise ValueError(
                "Expected exactly 13 generation readings "
                f"(12 historical + current), got {history.shape}"
            )

        if np.any(history < 0):
            raise ValueError(
                "Generation readings must be non-negative"
            )

        current = history[-1]
        lag_1 = history[-2]

        previous_4 = history[-5:-1]
        previous_12 = history[-13:-1]

        rolling_mean_4 = float(
            previous_4.mean()
        )

        rolling_mean_12 = float(
            previous_12.mean()
        )

        rolling_std_4 = float(
            previous_4.std(ddof=1)
        )

        rolling_std_12 = float(
            previous_12.std(ddof=1)
        )

        delta_1 = current - lag_1

        drop_from_4 = (
            rolling_mean_4 - current
        )

        drop_from_12 = (
            rolling_mean_12 - current
        )

        lag_scale = max(abs(lag_1), 0.1)
        mean_4_scale = max(
            abs(rolling_mean_4),
            0.1,
        )
        mean_12_scale = max(
            abs(rolling_mean_12),
            0.1,
        )

        std_4_scale = max(
            rolling_std_4,
            0.05,
        )

        std_12_scale = max(
            rolling_std_12,
            0.05,
        )

        relative_delta_1 = np.clip(
            delta_1 / lag_scale,
            -10.0,
            10.0,
        )

        relative_drop_4 = np.clip(
            drop_from_4 / mean_4_scale,
            -10.0,
            10.0,
        )

        relative_drop_12 = np.clip(
            drop_from_12 / mean_12_scale,
            -10.0,
            10.0,
        )

        z_drop_4 = np.clip(
            drop_from_4 / std_4_scale,
            -10.0,
            10.0,
        )

        z_drop_12 = np.clip(
            drop_from_12 / std_12_scale,
            -10.0,
            10.0,
        )

        values = {
            "delta_1": delta_1,
            "drop_from_4": drop_from_4,
            "drop_from_12": drop_from_12,
            "relative_delta_1": relative_delta_1,
            "relative_drop_4": relative_drop_4,
            "relative_drop_12": relative_drop_12,
            "z_drop_4": z_drop_4,
            "z_drop_12": z_drop_12,
        }

        return np.asarray(
            [
                values[name]
                for name in self.features
            ],
            dtype=np.float64,
        ).reshape(1, -1)

    def predict(
        self,
        generation_history: np.ndarray,
    ) -> dict[str, float | bool]:
        features = self._build_features(
            generation_history
        )

        probability = float(
            self.model.predict_proba(
                features
            )[0, 1]
        )

        return {
            "fault_risk": probability,
            "threshold": self.threshold,
            "alert": probability >= self.threshold,
        }
