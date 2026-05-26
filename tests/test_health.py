from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["message"] == "Backend health check passed"


def test_health_allows_head_and_options():
    assert client.head("/health").status_code == 200
    response = client.options("/health")
    assert response.status_code == 204
    assert "GET" in response.headers["allow"]


def test_health_tolerates_gateway_variants():
    assert client.get("/health/").status_code == 200
    assert client.head("/health/").status_code == 200
    assert client.options("/health/").status_code == 204
    assert client.post("/health").status_code == 200
    assert client.post("/health/").status_code == 200
    assert client.get("/healthz").status_code == 200


def test_gpt_health_endpoint_uses_fresh_operation_path():
    assert client.get("/gpt/health").status_code == 200
    assert client.get("/gpt/health/").status_code == 200
    assert client.post("/gpt/health").status_code == 200
    assert client.post("/gpt/health/").status_code == 200
    assert client.head("/gpt/health").status_code == 200
    assert client.options("/gpt/health").status_code == 204


def test_action_schema_is_served_for_custom_gpt():
    response = client.get("/action-schema.yaml")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/yaml")
    assert response.headers["cache-control"] == "no-store"
    assert "openapi: 3.1.0" in response.text
    assert "version: 1.0.3" in response.text
    assert "\n  /gpt/health:\n    post:\n      operationId: healthCheck" in response.text
    assert "operationId: healthCheck" in response.text
    assert "\n  /health:" not in response.text
    assert "\n      operationId: gptBackendHealthCheck\n" not in response.text
    assert "gptBackendHealthCheckPost" not in response.text
    assert "/gpt/workers/status" in response.text
    assert "/gpt/broker/kis/account-probe" in response.text
    assert "/gpt/auto-trading/status" in response.text
    assert "/gpt/auto-trading/start-paper" in response.text
    assert "nullable:" not in response.text
    assert "anyOf:" not in response.text
    assert 'type: "null"' not in response.text
    assert "GptStartPaperRequest" not in response.text
    assert "StartPaperRequest" in response.text
    assert '$ref: "#/components/schemas/GenericJsonResponse"' in response.text
    assert "  - url: https://api.autoinvestmentkorea.online" in response.text


def test_action_schema_uses_forwarded_public_host(monkeypatch):
    monkeypatch.setattr("app.main.settings.action_schema_public_url", None)

    response = client.get(
        "/action-schema.yaml",
        headers={
            "x-forwarded-proto": "https",
            "x-forwarded-host": "api.autoinvestmentkorea.online",
        },
    )

    assert response.status_code == 200
    assert "  - url: https://api.autoinvestmentkorea.online" in response.text


def test_naver_news_endpoint(monkeypatch):
    def fake_search_naver_news(query: str, display: int = 10):
        return {
            "connected": True,
            "source": "Naver Search API",
            "query": query,
            "total": 1,
            "items": [
                {
                    "title": "SK하이닉스 테스트 뉴스",
                    "description": "테스트 설명",
                    "originallink": "https://example.com/original",
                    "link": "https://example.com/news",
                    "pubDate": "Wed, 20 May 2026 09:00:00 +0900",
                }
            ],
            "display": display,
        }

    monkeypatch.setattr("app.main.search_naver_news", fake_search_naver_news)

    response = client.get("/naver/news", params={"query": "SK하이닉스", "display": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["connected"] is True
    assert body["query"] == "SK하이닉스"
    assert body["display"] == 3
    assert body["items"][0]["title"] == "SK하이닉스 테스트 뉴스"
