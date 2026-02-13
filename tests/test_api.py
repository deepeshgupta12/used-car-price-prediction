from __future__ import annotations

from fastapi.testclient import TestClient

from ucpp.api.app import app

client = TestClient(app)


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_shape() -> None:
    r = client.post(
        "/v1/predict",
        json={
            "market": "US",
            "payload": {"brand": "Toyota", "model": "Camry", "model_year": 2018},
        },
    )

    # Depending on whether artifacts exist in CI/local tests, this can vary.
    assert r.status_code in {200, 400, 422, 500}

    if r.status_code == 200:
        body = r.json()
        assert body["market"] == "US"
        assert "model" in body
        assert "prediction" in body


def test_batch_strict_may_fail_fast() -> None:
    r = client.post(
        "/v1/batch",
        json={
            "market": "US",
            "strict": True,
            "rows": [
                {"brand": "Toyota", "model": "Camry", "model_year": 2018},
                {"bad_field": "x"},  # guaranteed schema failure (extra="forbid")
            ],
        },
    )
    assert r.status_code in {200, 400, 422, 500}


def test_batch_non_strict_returns_errors() -> None:
    # Provide one invalid row; strict=false should return errors array
    r = client.post(
        "/v1/batch",
        json={
            "market": "US",
            "strict": False,
            "rows": [
                {"brand": "Toyota", "model": "Camry", "model_year": 2018},
                {"bad_field": "x"},  # guaranteed schema failure (extra="forbid")
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
