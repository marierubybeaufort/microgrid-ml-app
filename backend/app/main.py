from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.router import router


app = FastAPI(
    title="Microgrid ML API",
    description=(
        "Backend API for electricity-consumption forecasting "
        "and microgrid fault-risk detection."
    ),
    version="0.2.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router)


@app.get("/", tags=["system"])
def root():
    return {
        "name": "Microgrid ML API",
        "version": "0.2.0",
        "status": "running",
    }