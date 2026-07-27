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


def test_predict_returns_sanitized_uploaded_filename(client, monkeypatch, tmp_path):
    install_predict_mocks(monkeypatch)
    temp_temp_dir = tmp_path / "feedback" / "temp"
    temp_temp_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api, "TEMP_FEEDBACK_DIR", str(temp_temp_dir))

    response = client.post(
        "/predict",
        files={"file": ("../scan.jpg", image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["filename"] == "scan.jpg"


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


def test_feedback_returns_404_when_no_temp_image(client):
    response = client.post(
        "/feedback",
        json={
            "request_id": "non-existent-uuid",
            "feedback": "correct",
            "prediction": "2"
        }
    )
    assert response.status_code == 404
    assert "Không tìm thấy dữ liệu yêu cầu" in response.json()["detail"]


def test_feedback_rejects_unsafe_request_id(client):
    response = client.post(
        "/feedback",
        json={
            "request_id": "../escape",
            "feedback": "correct",
            "prediction": "2"
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid request_id"


def test_feedback_rejects_invalid_feedback_value(client):
    response = client.post(
        "/feedback",
        json={
            "request_id": "test-uuid-789",
            "feedback": "maybe",
            "prediction": "2"
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid feedback"


def test_feedback_rejects_invalid_prediction_grade(client):
    response = client.post(
        "/feedback",
        json={
            "request_id": "test-uuid-789",
            "feedback": "correct",
            "prediction": "9"
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid prediction"


def test_feedback_succeeds_for_correct_prediction(client, monkeypatch, tmp_path):
    temp_dir = tmp_path / "feedback"
    temp_temp_dir = temp_dir / "temp"
    temp_temp_dir.mkdir(parents=True, exist_ok=True)
    
    monkeypatch.setattr(api, "FEEDBACK_DIR", str(temp_dir))
    monkeypatch.setattr(api, "TEMP_FEEDBACK_DIR", str(temp_temp_dir))
    
    request_id = "test-uuid-123"
    dummy_img = temp_temp_dir / f"{request_id}.jpg"
    dummy_img.write_bytes(b"dummy image data")
    
    response = client.post(
        "/feedback",
        json={
            "request_id": request_id,
            "feedback": "correct",
            "prediction": "2"
        }
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    expected_path = temp_dir / "correct" / "2" / f"{request_id}.jpg"
    assert expected_path.exists()
    assert expected_path.read_bytes() == b"dummy image data"
    assert not dummy_img.exists()


def test_feedback_requires_corrected_grade_for_incorrect_prediction(client, monkeypatch, tmp_path):
    temp_dir = tmp_path / "feedback"
    temp_temp_dir = temp_dir / "temp"
    temp_temp_dir.mkdir(parents=True, exist_ok=True)
    
    monkeypatch.setattr(api, "FEEDBACK_DIR", str(temp_dir))
    monkeypatch.setattr(api, "TEMP_FEEDBACK_DIR", str(temp_temp_dir))
    
    request_id = "test-uuid-missing-grade"
    dummy_img = temp_temp_dir / f"{request_id}.jpg"
    dummy_img.write_bytes(b"dummy image data")
    
    response = client.post(
        "/feedback",
        json={
            "request_id": request_id,
            "feedback": "incorrect",
            "prediction": "2"
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "corrected_grade is required when feedback is incorrect"
    assert dummy_img.exists()


def test_feedback_succeeds_for_incorrect_prediction(client, monkeypatch, tmp_path):
    temp_dir = tmp_path / "feedback"
    temp_temp_dir = temp_dir / "temp"
    temp_temp_dir.mkdir(parents=True, exist_ok=True)
    
    monkeypatch.setattr(api, "FEEDBACK_DIR", str(temp_dir))
    monkeypatch.setattr(api, "TEMP_FEEDBACK_DIR", str(temp_temp_dir))
    
    request_id = "test-uuid-456"
    dummy_img = temp_temp_dir / f"{request_id}.jpg"
    dummy_img.write_bytes(b"dummy image data")
    
    response = client.post(
        "/feedback",
        json={
            "request_id": request_id,
            "feedback": "incorrect",
            "prediction": "2",
            "corrected_grade": "3"
        }
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    expected_path = temp_dir / "incorrect" / "grade_3" / f"{request_id}.jpg"
    assert expected_path.exists()
    assert expected_path.read_bytes() == b"dummy image data"
    assert not dummy_img.exists()
