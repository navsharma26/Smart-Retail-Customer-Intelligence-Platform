"""
Pytest integration test suite for all FastAPI endpoints.
Uses TestClient and httpx.
"""

import io
import pytest
import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _create_dummy_image_bytes(width=224, height=224):
    """Generate dummy JPEG image bytes for testing upload endpoints."""
    img = np.full((height, width, 3), 180, dtype=np.uint8)
    cv2.circle(img, (width // 2, height // 2), 40, (100, 100, 200), -1)
    success, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()


def test_health_check():
    """Test GET /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_root_endpoint():
    """Test GET / endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "unified_endpoints" in data


def test_classify_product_endpoint():
    """Test POST /classify-product endpoint."""
    img_bytes = _create_dummy_image_bytes(224, 224)
    files = {"file": ("test_product.jpg", img_bytes, "image/jpeg")}
    response = client.post("/classify-product", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "predicted_category" in data
    assert "confidence_score" in data
    assert "class_probabilities" in data


def test_recognize_face_endpoint():
    """Test POST /recognize-face endpoint."""
    img_bytes = _create_dummy_image_bytes(300, 300)
    files = {"file": ("test_face.jpg", img_bytes, "image/jpeg")}
    response = client.post("/recognize-face", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "matched" in data


def test_analyze_sentiment_endpoint():
    """Test POST /analyze-sentiment endpoint."""
    payload = {"text": "Exceptional product quality and super fast shipping!"}
    response = client.post("/analyze-sentiment", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["sentiment"] == "positive"
    assert data["confidence_score"] >= 0.5


def test_chatbot_endpoint():
    """Test POST /chatbot endpoint."""
    payload = {"query": "What is your return policy?", "customer_id": "CUST-99"}
    response = client.post("/chatbot", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["matched_intent"] == "return_policy"
    assert len(data["response"]) > 0


def test_dashboard_stats_endpoint():
    """Test GET /dashboard/stats endpoint."""
    response = client.get("/dashboard/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "customer_analytics" in data
    assert "sentiment_telemetry" in data
    assert "models_loaded" in data


def test_cv_register_customer_endpoint():
    """Test POST /api/v1/cv/register-customer endpoint."""
    img_bytes = _create_dummy_image_bytes(300, 300)
    files = {"file": ("new_customer.jpg", img_bytes, "image/jpeg")}
    data = {"customer_id": "CUST-TEST-001", "name": "John Doe"}
    response = client.post("/api/v1/cv/register-customer", data=data, files=files)
    assert response.status_code == 201
    res = response.json()
    assert res["status"] == "success"
    assert res["customer_id"] == "CUST-TEST-001"


def test_cv_list_customers_endpoint():
    """Test GET /api/v1/cv/customers endpoint."""
    response = client.get("/api/v1/cv/customers")
    assert response.status_code == 200
    customers = response.json()
    assert isinstance(customers, list)


def test_nlp_preprocess_endpoint():
    """Test POST /api/v1/nlp/preprocess endpoint."""
    payload = {"text": "I LOVED these shoes! Very fast delivery."}
    response = client.post("/api/v1/nlp/preprocess", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert "processed_text" in res


def test_chatbot_intents_endpoint():
    """Test GET /api/v1/chatbot/intents endpoint."""
    response = client.get("/api/v1/chatbot/intents")
    assert response.status_code == 200
    intents = response.json()
    assert len(intents) == 20
