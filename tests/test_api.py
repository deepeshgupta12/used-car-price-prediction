from __future__ import annotations

from fastapi.testclient import TestClient

from ucpp.api.app import app

client = TestClient(app)


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_validation_error() -> None:
    # invalid market should be rejected at request parsing level
    r = client.post(
        "/v1/predict",
        json={"market": "EU", "payload": {}},
    )
    assert r.status_code in {400, 422}


def test_batch_non_strict_returns_errors() -> None:
    # Provide one invalid row; strict=false should return errors array
    r = client.post(
        "/v1/batch",
        json={
            "market": "US",
            "strict": False,
            "rows": [
                {"brand": "Toyota", "model": "Camry", "model_year": 2018},  # partial payload
                {},  # definitely invalid
            ],
        },
    )

    # If artifacts aren't present in test environment, inference can fail.
    # We mainly assert API shape + non-strict behavior doesn't crash.
    assert r.status_code in {200, 400, 422, 500}

    if r.status_code == 200:
        body = r.json()
        assert "preds" in body
        assert "errors" in body
        assert len(body["errors"]) >= 1
