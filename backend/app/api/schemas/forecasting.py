from datetime import datetime, timedelta

from pydantic import BaseModel, Field, field_validator


class ForecastObservation(BaseModel):
    timestamp: datetime
    consumption_per_premise_kwh: float = Field(
        ge=0.0,
    )


class ForecastRequest(BaseModel):
    observations: list[ForecastObservation] = Field(
        min_length=24,
        max_length=24,
    )

    @field_validator("observations")
    @classmethod
    def validate_chronological_order(
        cls,
        observations: list[ForecastObservation],
    ) -> list[ForecastObservation]:
        timestamps = [
            observation.timestamp
            for observation in observations
        ]

        if timestamps != sorted(timestamps):
            raise ValueError(
                "Observations must be ordered chronologically"
            )

        if len(set(timestamps)) != len(timestamps):
            raise ValueError(
                "Observation timestamps must be unique"
            )

        for previous, current in zip(
            timestamps,
            timestamps[1:],
        ):
            if current - previous != timedelta(hours=1):
                raise ValueError(
                    "Forecast observations must be exactly 1 hour apart"
                )

        return observations


class ForecastResponse(BaseModel):
    prediction_kwh_per_premise: float