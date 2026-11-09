import asyncio

import httpx
import pytest

from app.core.config import FRONTEND_URL
from app.main import app


def request(method: str, path: str, **kwargs) -> httpx.Response:
    async def send_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send_request())


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        FRONTEND_URL,
    ],
)
def test_login_preflight_allows_configured_frontend_origins(origin):
    response = request(
        "OPTIONS",
        "/api/auth/login",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    assert response.headers["access-control-allow-credentials"] == "true"
    assert "POST" in response.headers["access-control-allow-methods"]


def test_login_preflight_rejects_unapproved_origin():
    response = request(
        "OPTIONS",
        "/api/auth/login",
        headers={
            "Origin": "http://unapproved.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers

