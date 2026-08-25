import numpy as np

from backend.app.api.schemas.forecasting import ForecastRequest
from backend.app.ml.forecasting.inference import ForecastPredictor


class ForecastService:
    """Prepare API observations and run next-hour forecasting."""

    def __init__(self) -> None:
        self.predictor = ForecastPredictor()

    def predict(self, request: ForecastRequest) -> float:
        rows = []

        for observation in request.observations:
            timestamp = observation.timestamp

            hour = timestamp.hour
            day_of_week = timestamp.weekday()

            hour_angle = 2.0 * np.pi * hour / 24.0
            dow_angle = 2.0 * np.pi * day_of_week / 7.0

            rows.append(
                [
                    observation.consumption_per_premise_kwh,
                    np.sin(hour_angle),
                    np.cos(hour_angle),
                    np.sin(dow_angle),
                    np.cos(dow_angle),
                ]
            )

        sequence = np.asarray(
            rows,
            dtype=np.float32,
        )

        return self.predictor.predict(sequence)
