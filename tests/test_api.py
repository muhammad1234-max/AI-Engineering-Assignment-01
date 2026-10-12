"""Unit and integration tests for FastAPI REST inference service."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from zaid_lm.api import app, load_model_state


@pytest.fixture(scope="module")
def client():
    load_model_state()
    return TestClient(app)


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "endpoints" in data
    assert "/generate" in data["endpoints"]


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_model_info(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["parameters"] == 809984
    assert data["vocab_size"] == 65
    assert data["n_layers"] == 4
    assert data["n_heads"] == 4


def test_generate_endpoint_success(client):
    payload = {
        "prompt": "ROMEO:\n",
        "max_new_tokens": 20,
        "temperature": 0.8,
        "top_k": 40,
        "seed": 42,
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["tokens_generated"] == 20
    assert data["prompt"] == "ROMEO:\n"
    assert len(data["full_text"]) > len(data["prompt"])
    assert data["tokens_per_second"] > 0.0


def test_generate_endpoint_validation_error(client):
    # Invalid max_new_tokens (too high)
    payload = {
        "prompt": "TEST",
        "max_new_tokens": 1000,  # Max allowed is 512
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 422
