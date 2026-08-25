from datetime import datetime, timedelta

from pydantic import BaseModel, Field, field_validator


class FaultObservation(BaseModel):
    timestamp: datetime
    generation_kw: float = Field(
        ge=0.0,
    )


class FaultRequest(BaseModel):
    observations: list[FaultObservation] = Field(
        min_length=13,
        max_length=13,
    )

    @field_validator("observations")
    @classmethod
    def validate_observations(
        cls,
        observations: list[FaultObservation],
    ) -> list[FaultObservation]:
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
            if current - previous != timedelta(minutes=15):
                raise ValueError(
                    "Fault observations must be exactly 15 minutes apart"
                )

        return observations


class FaultResponse(BaseModel):
    fault_risk: float
    threshold: float
    alert: bool
