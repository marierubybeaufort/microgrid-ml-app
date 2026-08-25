import json
from pathlib import Path

import numpy as np
import torch

from backend.app.ml.forecasting.model import LSTMForecaster
from backend.app.ml.forecasting.scaling import TargetScaler


ARTIFACT_DIR = Path("backend/artifacts/forecasting")


class ForecastPredictor:
    """Load saved forecasting artifacts and run next-hour inference."""

    def __init__(
        self,
        artifact_dir: str | Path = ARTIFACT_DIR,
        device: str | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)

        with open(
            self.artifact_dir / "config.json",
            encoding="utf-8",
        ) as file:
            self.config = json.load(file)

        with open(
            self.artifact_dir / "scaler.json",
            encoding="utf-8",
        ) as file:
            scaler_data = json.load(file)

        self.scaler = TargetScaler(
            mean=float(scaler_data["mean"]),
            std=float(scaler_data["std"]),
            target_column=scaler_data["target_column"],
        )

        self.features = tuple(self.config["features"])
        self.sequence_length = int(
            self.config["sequence_length"]
        )

        self.device = torch.device(
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.model = LSTMForecaster(
            input_size=len(self.features),
            hidden_size=int(self.config["hidden_size"]),
        ).to(self.device)

        state_dict = torch.load(
            self.artifact_dir / "model.pt",
            map_location=self.device,
            weights_only=True,
        )

        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        sequence: np.ndarray,
    ) -> float:
        """Predict next-hour consumption in kWh per premise."""

        sequence = np.asarray(
            sequence,
            dtype=np.float32,
        )

        expected_shape = (
            self.sequence_length,
            len(self.features),
        )

        if sequence.shape != expected_shape:
            raise ValueError(
                f"Expected sequence shape "
                f"{expected_shape}, got {sequence.shape}"
            )

        sequence = sequence.copy()

        # Feature 0 is consumption_per_premise_kwh.
        sequence[:, 0] = self.scaler.transform(
            sequence[:, 0]
        )

        x = torch.from_numpy(
            sequence
        ).unsqueeze(0).to(self.device)

        prediction_scaled = self.model(x)

        prediction_real = self.scaler.inverse_transform(
            prediction_scaled
        )

        return float(
            prediction_real.squeeze().cpu().item()
        )
