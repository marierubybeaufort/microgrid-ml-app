from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_forecast_endpoint() -> None:
    start = datetime(2025, 5, 29, 1, 0)

    observations = [
        {
            "timestamp": (start + timedelta(hours=i)).isoformat(),
            "consumption_per_premise_kwh": 0.8,
        }
        for i in range(24)
    ]

    response = client.post(
        "/forecast",
        json={"observations": observations},
    )

    assert response.status_code == 200

    body = response.json()

    assert "prediction_kwh_per_premise" in body
    assert isinstance(
        body["prediction_kwh_per_premise"],
        float,
    )


def test_forecast_requires_24_observations() -> None:
    start = datetime(2025, 5, 29, 1, 0)

    observations = [
        {
            "timestamp": (start + timedelta(hours=i)).isoformat(),
            "consumption_per_premise_kwh": 0.8,
        }
        for i in range(23)
    ]

    response = client.post(
        "/forecast",
        json={"observations": observations},
    )

    assert response.status_code == 422


def test_fault_endpoint() -> None:
    start = datetime(2025, 9, 12, 8, 30)

    generation = [
        2.8,
        2.7,
        2.6,
        2.5,
        2.4,
        2.3,
        2.2,
        2.1,
        2.0,
        1.9,
        1.8,
        1.7,
        0.2,
    ]

    observations = [
        {
            "timestamp": (
                start + timedelta(minutes=15 * i)
            ).isoformat(),
            "generation_kw": value,
        }
        for i, value in enumerate(generation)
    ]

    response = client.post(
        "/fault",
        json={"observations": observations},
    )

    assert response.status_code == 200

    body = response.json()

    assert "fault_risk" in body
    assert "threshold" in body
    assert "alert" in body

    assert 0.0 <= body["fault_risk"] <= 1.0
    assert isinstance(body["alert"], bool)


def test_fault_rejects_negative_generation() -> None:
    start = datetime(2025, 9, 12, 8, 30)

    observations = [
        {
            "timestamp": (
                start + timedelta(minutes=15 * i)
            ).isoformat(),
            "generation_kw": -1.0 if i == 12 else 1.0,
        }
        for i in range(13)
    ]

    response = client.post(
        "/fault",
        json={"observations": observations},
    )

    assert response.status_code == 422


def test_forecast_rejects_irregular_intervals() -> None:
    start = datetime(2025, 5, 29, 1, 0)

    observations = [
        {
            "timestamp": (
                start
                + timedelta(hours=i)
                + (timedelta(minutes=5) if i == 12 else timedelta())
            ).isoformat(),
            "consumption_per_premise_kwh": 0.8,
        }
        for i in range(24)
    ]

    response = client.post(
        "/forecast",
        json={"observations": observations},
    )

    assert response.status_code == 422


def test_fault_rejects_irregular_intervals() -> None:
    start = datetime(2025, 9, 12, 8, 30)

    observations = [
        {
            "timestamp": (
                start
                + timedelta(minutes=15 * i)
                + (timedelta(minutes=1) if i == 6 else timedelta())
            ).isoformat(),
            "generation_kw": 1.0,
        }
        for i in range(13)
    ]

    response = client.post(
        "/fault",
        json={"observations": observations},
    )

    assert response.status_code == 422
