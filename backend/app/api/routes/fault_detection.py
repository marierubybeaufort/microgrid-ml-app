from fastapi import APIRouter

from backend.app.api.schemas.fault_detection import (
    FaultRequest,
    FaultResponse,
)
from backend.app.services.fault_detection import (
    FaultDetectionService,
)


router = APIRouter(
    prefix="/fault",
    tags=["fault-detection"],
)

service = FaultDetectionService()


@router.post(
    "",
    response_model=FaultResponse,
)
def detect_fault(
    request: FaultRequest,
) -> FaultResponse:
    result = service.predict(request)

    return FaultResponse(
        fault_risk=float(result["fault_risk"]),
        threshold=float(result["threshold"]),
        alert=bool(result["alert"]),
    )
