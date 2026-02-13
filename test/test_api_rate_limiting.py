"""
Rate Limiting Tests
Test per-IP rate limiting middleware.
"""

import pytest
from fastapi.testclient import TestClient
from main import app
import os
import time

# Set test environment
os.environ['ENVIRONMENT'] = 'development'
os.environ['API_KEYS'] = 'test_key_rl'
os.environ['RATE_LIMIT_PER_MINUTE'] = '10'  # Low limit for testing


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.mark.security
@pytest.mark.slow
def test_rate_limiting_enforcement(client):
    """Test that rate limiting is enforced."""
    headers = {"X-API-Key": "test_key_rl"}

    # Make requests up to the limit
    responses = []
    for i in range(12):  # Limit is 10, so 11th should fail
        response = client.get("/api/stock/AAPL", headers=headers)
        responses.append(response)

        if response.status_code == 429:
            break  # Hit rate limit

    # Check that we got rate limited
    status_codes = [r.status_code for r in responses]
    assert 429 in status_codes, "Rate limit should have been hit"


@pytest.mark.security
def test_rate_limit_headers_present(client):
    """Rate limit headers should be present in responses."""
    headers = {"X-API-Key": "test_key_rl"}

    response = client.get("/health")  # Exempt from rate limiting
    # Health endpoint might not have rate limit headers

    # Try protected endpoint
    response = client.get("/api/stock/AAPL", headers=headers)

    if response.status_code not in [401, 403]:
        # If we got past auth, check for rate limit headers
        assert "x-ratelimit-limit" in response.headers or "X-RateLimit-Limit" in response.headers


@pytest.mark.security
def test_exempt_paths_not_rate_limited(client):
    """Exempt paths should not be rate limited."""
    # Make many requests to health endpoint
    success_count = 0
    for i in range(20):
        response = client.get("/health")
        if response.status_code == 200:
            success_count += 1

    # All should succeed (health is exempt)
    assert success_count == 20, "Health endpoint should not be rate limited"


@pytest.mark.security
def test_rate_limit_error_message(client):
    """Rate limit error should return proper message."""
    headers = {"X-API-Key": "test_key_rl"}

    # Trigger rate limit
    for i in range(15):
        response = client.get("/api/stock/AAPL", headers=headers)
        if response.status_code == 429:
            data = response.json()
            assert "error" in data or "detail" in data
            assert "Rate limit" in str(data) or "Too many" in str(data)
            break
