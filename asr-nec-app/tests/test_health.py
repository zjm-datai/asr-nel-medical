from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel


@pytest.fixture()
def client() -> TestClient:
    from app import create_app
    from extensions import ext_database

    SQLModel.metadata.create_all(ext_database.engine)
    return TestClient(create_app())


def test_health(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["database"] == "ok"
    assert payload["model_loaded"] is False
    assert payload["device"] == "cpu"
