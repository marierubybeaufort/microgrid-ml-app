import numpy as np

from backend.app.api.schemas.fault_detection import FaultRequest
from backend.app.ml.fault_detection.inference import FaultPredictor


class FaultDetectionService:
    """Prepare API observations and run fault-risk inference."""

    def __init__(self) -> None:
        self.predictor = FaultPredictor()

    def predict(
        self,
        request: FaultRequest,
    ) -> dict[str, float | bool]:
        generation_history = np.asarray(
            [
                observation.generation_kw
                for observation in request.observations
            ],
            dtype=np.float64,
        )

        return self.predictor.predict(
            generation_history
        )
