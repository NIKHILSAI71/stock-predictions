"""
Authentication Tests
Test API key authentication middleware.
"""

import pytest
from fastapi.testclient import TestClient
from main import app
import os

# Set test environment
os.environ['ENVIRONMENT'] = 'development'
os.environ['API_KEYS'] = 'test_key_12345,another_test_key'


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.mark.security
def test_public_endpoints_no_auth(client):
    """Public endpoints should work without API key."""
    # Health check
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

    # Root endpoint
    response = client.get("/")
    assert response.status_code == 200
    assert "Stock Analysis API" in response.json()["message"]

    # Docs (if in development mode)
    response = client.get("/docs")
    assert response.status_code in [200, 404]  # 404 if production mode


@pytest.mark.security
def test_protected_endpoint_without_key(client):
    """Protected endpoints should require API key."""
    response = client.get("/api/stock/AAPL")
    assert response.status_code == 401
    assert "API key required" in str(response.json())


@pytest.mark.security
def test_protected_endpoint_with_invalid_key(client):
    """Invalid API key should be rejected."""
    response = client.get(
        "/api/stock/AAPL",
        headers={"X-API-Key": "invalid_key_wrong"}
    )
    assert response.status_code == 403
    assert "Invalid" in str(response.json())


@pytest.mark.security
def test_protected_endpoint_with_valid_key(client):
    """Valid API key should grant access."""
    response = client.get(
        "/api/stock/AAPL",
        headers={"X-API-Key": "test_key_12345"}
    )
    # Should not be 401 or 403 (might be 200, 404, or 500 depending on data availability)
    assert response.status_code not in [401, 403]


@pytest.mark.security
def test_multiple_valid_keys(client):
    """Multiple API keys should all work."""
    # First key
    response1 = client.get(
        "/api/stock/AAPL",
        headers={"X-API-Key": "test_key_12345"}
    )
    assert response1.status_code not in [401, 403]

    # Second key
    response2 = client.get(
        "/api/stock/AAPL",
        headers={"X-API-Key": "another_test_key"}
    )
    assert response2.status_code not in [401, 403]


@pytest.mark.security
def test_api_key_header_case_sensitivity(client):
    """API key header should be case-insensitive."""
    # Standard case
    response1 = client.get(
        "/api/stock/AAPL",
        headers={"X-API-Key": "test_key_12345"}
    )

    # Different case (FastAPI normalizes headers)
    response2 = client.get(
        "/api/stock/AAPL",
        headers={"x-api-key": "test_key_12345"}
    )

    # Both should have same auth result
    assert response1.status_code == response2.status_code
