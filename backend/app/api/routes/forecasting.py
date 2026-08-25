from fastapi import APIRouter

from backend.app.api.schemas.forecasting import (
    ForecastRequest,
    ForecastResponse,
)
from backend.app.services.forecasting import ForecastService


router = APIRouter(
    prefix="/forecast",
    tags=["forecasting"],
)

service = ForecastService()


@router.post(
    "",
    response_model=ForecastResponse,
)
def forecast(
    request: ForecastRequest,
) -> ForecastResponse:
    prediction = service.predict(request)

    return ForecastResponse(
        prediction_kwh_per_premise=prediction,
    )
