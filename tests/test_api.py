"""FastAPI endpoint tests using TestClient."""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from deeptrace.api.app import app


@pytest.fixture()
def client():
    return TestClient(app)


def test_root_index_endpoint(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "DeepTrace AI" in res.text
    assert "/docs" in res.text


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "vision_model_status" in data


def test_history_endpoint(client):
    res = client.get("/api/history")
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_analyze_image_api(client, png_file):
    with open(png_file, "rb") as f:
        res = client.post(
            "/api/analyze/image",
            files={"file": ("sample.png", f, "image/png")},
        )
    assert res.status_code == 200
    data = res.json()
    assert data["media_type"] == "image"
    assert "verdict" in data
    assert "report_id" in data


def test_analyze_audio_api(client, wav_file):
    with open(wav_file, "rb") as f:
        res = client.post(
            "/api/analyze/audio",
            files={"file": ("sample.wav", f, "audio/wav")},
        )
    assert res.status_code == 200
    data = res.json()
    assert data["media_type"] == "audio"
    assert "verdict" in data
