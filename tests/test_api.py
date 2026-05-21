import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

import deployment.api.main as api


@pytest.fixture(autouse=True)
def reset_api_state(monkeypatch):
    api.model = None
    monkeypatch.setattr(api, "API_ALERTS_WEBHOOK", None)
    monkeypatch.setattr(api, "AI_PREDICTION_WEBHOOK", None)
    yield
    api.model = None


@pytest.fixture
def client():
    return TestClient(api.app)


def image_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color="white").save(buffer, format="JPEG")
    return buffer.getvalue()


class DummyModel:
    def __call__(self, input_tensor):
        return torch.tensor([[0.1, 0.2, 0.9, 0.3, 0.4]], dtype=torch.float32)


class DummyTransform:
    def __call__(self, image):
        return torch.zeros((3, 224, 224), dtype=torch.float32)


def install_predict_mocks(monkeypatch):
    api.model = DummyModel()
    monkeypatch.setattr(api, "get_transforms", lambda: {"val": DummyTransform()})
    monkeypatch.setattr(
        api,
        "get_heatmap",
        lambda model, input_tensor, image: np.zeros((224, 224, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(api, "is_valid_knee_xray", lambda image: (True, 0.99))


def test_health_returns_503_when_model_not_loaded(client):
    response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "unhealthy", "detail": "Model not loaded"}


def test_predict_returns_503_when_model_not_loaded(client):
    response = client.post(
        "/predict",
        files={"file": ("scan.jpg", image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Model not loaded"


def test_predict_rejects_invalid_extension(client):
    response = client.post(
        "/predict",
        files={"file": ("scan.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


def test_predict_rejects_invalid_image(client):
    response = client.post(
        "/predict",
        files={"file": ("scan.jpg", b"not an image", "image/jpeg")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image file"


def test_predict_rejects_oversized_file(client):
    oversized_content = b"0" * (api.MAX_FILE_SIZE + 1)

    response = client.post(
        "/predict",
        files={"file": ("scan.jpg", oversized_content, "image/jpeg")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "File too large"


def test_predict_succeeds_with_mocked_model(client, monkeypatch):
    install_predict_mocks(monkeypatch)

    response = client.post(
        "/predict",
        files={"file": ("scan.jpg", image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["prediction"] == "2"
    assert payload["confidence"] == "32.2%"
    assert payload["heatmap_base64"]


def test_predict_rejects_invalid_knee_xray(client, monkeypatch):
    install_predict_mocks(monkeypatch)
    # Simulate an invalid image (e.g., cat, dog, non-xray)
    monkeypatch.setattr(api, "is_valid_knee_xray", lambda image: (False, 0.98))

    response = client.post(
        "/predict",
        files={"file": ("fake_image.jpg", image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 400
    assert "Ảnh tải lên không phải là ảnh chụp X-quang khớp gối" in response.json()["detail"]
