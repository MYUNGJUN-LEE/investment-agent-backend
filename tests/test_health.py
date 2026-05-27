from fastapi.testclient import TestClient
from app.config import settings
from app.main import app

client = TestClient(app)


def _auth_headers() -> dict[str, str]:
    if settings.backend_api_key:
        return {"X-API-Key": settings.backend_api_key}
    return {}


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


def test_admin_dashboard_is_served():
    response = client.get("/admin.html")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Investment Agent Admin" in response.text
    assert "/gpt/auto-trading/start-paper" in response.text
    assert "/gpt/broker/kis/account-probe" in response.text
    assert "Start Live Auto" in response.text
    assert "/live/orders" in response.text
    assert "Universe Scan" in response.text
    assert "Runtime Overview" in response.text
    assert "Auto refresh" in response.text
    assert "/admin/runtime-status" in response.text
    assert "Edge Samples" in response.text
    assert "Top10 Expectancy" in response.text
    assert "Avg Expected Return" in response.text
    assert "MAE Edge Error" in response.text
    assert "Scan + Refresh Samples" in response.text
    assert "Refresh Samples" in response.text
    assert "/edge-calibration/samples" in response.text
    assert "/edge-calibration/refresh-samples" in response.text

    alias_response = client.get("/admin")
    assert alias_response.status_code == 200
    assert "Investment Agent Admin" in alias_response.text


def test_edge_training_samples_endpoint_returns_summary():
    response = client.get("/edge-calibration/samples", headers=_auth_headers())
    assert response.status_code == 200
    body = response.json()
    assert "sample_count" in body
    assert "summary" in body
    assert "diagnostics" in body
    assert "total_return_bps" in body["summary"]
    assert "total_risk_bps" in body["summary"]
    assert "avg_expected_return_bps" in body["summary"]
    assert "avg_realized_net_edge_bps" in body["summary"]
    assert "net_edge_win_rate" in body["summary"]
    assert "mae_net_edge_error_bps" in body["summary"]
    assert "top10_performance" in body
    assert "label_policy" in body
    assert "universe_scan_count" in body["diagnostics"]
    assert "total_auto_trading_cycle_count" in body["diagnostics"]
    assert "recent_samples" in body


def test_admin_runtime_status_endpoint_returns_compact_summary():
    response = client.get("/admin/runtime-status", headers=_auth_headers())
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert "generated_at" in body
    assert "summary" in body
    assert "auto_trading" in body
    assert "latest_universe" in body
    assert "samples" in body
    assert "workers" in body
    assert "scanner_state" in body["summary"]
    assert "sample_state" in body["summary"]
    assert "universe_scan_count" in body["summary"]
    assert "latest_scan_age_seconds" in body["summary"]
    assert "scanner_stale_after_seconds" in body["summary"]
    assert "locked_session_count" in body["summary"]


def test_edge_training_samples_refresh_endpoint_returns_diagnostics(monkeypatch):
    monkeypatch.setattr(
        "app.main.refresh_edge_training_samples",
        lambda: {"status": "success", "inserted_count": 0},
    )
    monkeypatch.setattr("app.main.initialize_universe_db", lambda: None)
    monkeypatch.setattr(
        "app.main.refresh_top10_performance_if_due",
        lambda force=False: {"status": "success", "top10_performance": {"sample_count": 0}},
    )
    monkeypatch.setattr(
        "app.main.get_edge_training_sample_summary",
        lambda limit=20: {"status": "empty", "diagnostics": {}},
    )

    response = client.post("/edge-calibration/refresh-samples", headers=_auth_headers())
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert "refresh" in body
    assert "top10_performance_refresh" in body
    assert "samples" in body
    assert "diagnostics" in body["samples"]


def test_admin_reset_data_endpoint_requires_confirmation():
    response = client.post(
        "/admin/reset-data",
        headers=_auth_headers(),
        json={"confirm": "wrong", "dry_run": True},
    )
    assert response.status_code == 400


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
