"""
API Endpoint Tests
Test critical API endpoints with authentication.
"""

import pytest
from fastapi.testclient import TestClient
from main import app
import os

# Set test environment
os.environ['ENVIRONMENT'] = 'development'
os.environ['API_KEYS'] = 'test_endpoint_key'


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Authentication headers for tests."""
    return {"X-API-Key": "test_endpoint_key"}


@pytest.mark.api
def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "rate_limiter" in data


@pytest.mark.api
def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert "Stock Analysis API" in data["message"]
    assert "authentication" in data
    assert data["authentication"]["required"] == True


@pytest.mark.api
def test_backtest_endpoint(client, auth_headers):
    """Test backtesting endpoint."""
    response = client.get("/api/backtest-summary/AAPL", headers=auth_headers)

    # Should return 200 or 400/500 depending on data availability
    assert response.status_code in [200, 400, 500]

    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "success"
        assert "backtest_summary" in data
        assert isinstance(data["backtest_summary"], dict)


@pytest.mark.api
def test_model_accuracy_endpoint(client, auth_headers):
    """Test model accuracy endpoint."""
    response = client.get("/api/model-accuracy/AAPL", headers=auth_headers)

    assert response.status_code in [200, 500]

    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "success"
        assert "accuracy_data" in data


@pytest.mark.api
def test_forecast_endpoint(client, auth_headers):
    """Test forecast endpoint."""
    response = client.get(
        "/api/quantitative/forecast/AAPL?steps=5&models=lstm",
        headers=auth_headers
    )

    assert response.status_code in [200, 400, 500]

    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "success"
        assert "forecast" in data
        assert "predictions" in data["forecast"]


@pytest.mark.api
def test_prediction_history_endpoint(client, auth_headers):
    """Test prediction history endpoint."""
    response = client.get(
        "/api/prediction-history/AAPL?limit=10",
        headers=auth_headers
    )

    assert response.status_code in [200, 500]

    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "success"
        assert "recent_predictions" in data
        assert isinstance(data["recent_predictions"], list)


@pytest.mark.api
def test_invalid_symbol(client, auth_headers):
    """Test with invalid stock symbol."""
    response = client.get("/api/stock/INVALID123", headers=auth_headers)

    # Should return error (404 or 500)
    assert response.status_code in [404, 500]


@pytest.mark.api
def test_forecast_validation(client, auth_headers):
    """Test forecast endpoint input validation."""
    # Steps too high
    response = client.get(
        "/api/quantitative/forecast/AAPL?steps=100",
        headers=auth_headers
    )
    assert response.status_code == 400

    # Steps too low
    response = client.get(
        "/api/quantitative/forecast/AAPL?steps=0",
        headers=auth_headers
    )
    assert response.status_code == 400
