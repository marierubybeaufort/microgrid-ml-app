from fastapi import APIRouter

from backend.app.api.routes.fault_detection import (
    router as fault_detection_router,
)
from backend.app.api.routes.forecasting import (
    router as forecasting_router,
)


router = APIRouter()

router.include_router(forecasting_router)
router.include_router(fault_detection_router)


@router.get("/health", tags=["system"])
def health():
    return {"status": "healthy"}
