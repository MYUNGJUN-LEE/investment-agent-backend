from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


client = TestClient(app)


def test_naver_news_endpoint_requires_api_key_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "backend_api_key", "secret-key")
    monkeypatch.setattr(
        "app.main.search_naver_news",
        lambda query, display=10: {"connected": True, "query": query, "items": []},
    )

    unauthorized = client.get("/naver/news", params={"query": "005930"})
    assert unauthorized.status_code == 401

    authorized = client.get(
        "/naver/news",
        params={"query": "005930"},
        headers={"X-API-Key": "secret-key"},
    )
    assert authorized.status_code == 200
    assert authorized.json()["connected"] is True
