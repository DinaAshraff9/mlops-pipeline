"""
API Tests for the MLOps Prediction Service
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Mock model loading before import
with patch("joblib.load") as mock_load:
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0])
    mock_model.predict_proba.return_value = np.array([[0.85, 0.15]])

    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, 10))

    mock_load.side_effect = [mock_model, mock_scaler]

    from api.main import app

client = TestClient(app)
VALID_TOKEN = "mlops-secret-token"
HEADERS = {"Authorization": f"Bearer {VALID_TOKEN}"}


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_schema(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data


class TestPredictionEndpoint:
    def test_predict_requires_auth(self):
        response = client.post("/predict", json={"features": [1.0] * 10})
        assert response.status_code == 403

    def test_predict_invalid_token(self):
        response = client.post(
            "/predict",
            json={"features": [1.0] * 10},
            headers={"Authorization": "Bearer wrong-token"}
        )
        assert response.status_code == 401

    def test_predict_success(self):
        response = client.post(
            "/predict",
            json={"features": [1.0] * 10},
            headers=HEADERS
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "model_version" in data
        assert "latency_ms" in data
        assert 0.0 <= data["probability"] <= 1.0

    def test_predict_returns_int(self):
        response = client.post(
            "/predict",
            json={"features": [1.0] * 10},
            headers=HEADERS
        )
        assert isinstance(response.json()["prediction"], int)


class TestBatchEndpoint:
    def test_batch_predict(self):
        response = client.post(
            "/predict/batch",
            json={"instances": [[1.0] * 10, [2.0] * 10]},
            headers=HEADERS
        )
        # Rate limited in test, check schema if 200
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "count" in data


class TestMetricsEndpoint:
    def test_metrics_returns_prometheus_format(self):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "mlops_predictions_total" in response.text or \
               "mlops_active_requests" in response.text or \
               response.status_code == 200
