from __future__ import annotations

import sqlite3

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.trading import market_monitor


client = TestClient(app)


def test_kis_market_watch_records_surge_volume_and_stop_alerts(tmp_path, monkeypatch):
    db_path = tmp_path / "monitor.sqlite3"
    monkeypatch.setattr(settings, "market_monitor_db_path", str(db_path))
    monkeypatch.setattr(settings, "alert_db_path", str(tmp_path / "alerts.sqlite3"))
    monkeypatch.setattr(settings, "monitor_surge_change_pct", 5.0)
    monkeypatch.setattr(settings, "monitor_volume_spike_ratio", 3.0)
    monkeypatch.setattr(
        market_monitor,
        "_resolve_watchlist",
        lambda: {"005930": {"symbol": "005930", "name": "삼성전자"}},
    )
    monkeypatch.setattr(
        market_monitor,
        "_load_broker_positions",
        lambda: {
            "005930": {
                "symbol": "005930",
                "quantity": 1,
                "avg_price": 110,
            }
        },
    )

    def fake_fetch_price_data(symbol):
        return {
            "status": "connected",
            "symbol": symbol,
            "current_price": 100,
            "change_rate": 6.2,
            "volume_ratio": 3.5,
        }

    monkeypatch.setattr(market_monitor, "fetch_price_data", fake_fetch_price_data)

    result = market_monitor.run_kis_market_watch(db_path=db_path)

    assert result["alert_count"] == 3
    assert {alert["alert_type"] for alert in result["alerts"]} == {
        "price_surge",
        "volume_spike",
        "stop_loss_hit",
    }
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM monitor_alerts").fetchone()[0]
    assert count == 3


def test_naver_news_watch_dedupes_news_across_queries(tmp_path, monkeypatch):
    db_path = tmp_path / "monitor.sqlite3"
    monkeypatch.setattr(settings, "alert_db_path", str(tmp_path / "alerts.sqlite3"))
    monkeypatch.setattr(settings, "monitor_market_keywords", "코스피,코스피,환율")
    monkeypatch.setattr(
        market_monitor,
        "_resolve_watchlist",
        lambda: {"005930": {"symbol": "005930", "name": "삼성전자"}},
    )

    def fake_search_naver_news(query, display=10, sort="date"):
        return {
            "connected": True,
            "source": "Naver Search API",
            "query": query,
            "items": [
                {
                    "title": "반도체 투자 확대",
                    "description": "AI 수요 증가",
                    "originallink": "https://example.com/same-news",
                    "link": "https://naver.example.com/same-news",
                    "pubDate": "Fri, 22 May 2026 09:00:00 +0900",
                }
            ],
        }

    monkeypatch.setattr(market_monitor, "search_naver_news", fake_search_naver_news)

    first = market_monitor.run_naver_news_watch(db_path=db_path)
    second = market_monitor.run_naver_news_watch(db_path=db_path)

    assert first["query_count"] == 3
    assert first["new_count"] == 1
    assert second["new_count"] == 0


def test_opendart_watch_filters_watchlist_and_dedupes(tmp_path, monkeypatch):
    db_path = tmp_path / "monitor.sqlite3"
    monkeypatch.setattr(settings, "alert_db_path", str(tmp_path / "alerts.sqlite3"))
    monkeypatch.setattr(
        market_monitor,
        "_resolve_watchlist",
        lambda: {"005930": {"symbol": "005930", "name": "삼성전자"}},
    )

    def fake_fetch_opendart_disclosures(symbol, lookback_hours):
        return [
            {
                "source": "OpenDART",
                "date": "20260522",
                "title": "주요사항보고서",
                "url": "https://dart.fss.or.kr/123",
                "impact_direction": "uncertain",
                "impact_strength": 10,
                "raw": {"rcept_no": "20260522000123"},
            }
        ]

    monkeypatch.setattr(
        market_monitor,
        "fetch_opendart_disclosures",
        fake_fetch_opendart_disclosures,
    )

    first = market_monitor.run_opendart_disclosure_watch(db_path=db_path)
    second = market_monitor.run_opendart_disclosure_watch(db_path=db_path)

    assert first["symbol_count"] == 1
    assert first["new_count"] == 1
    assert second["new_count"] == 0


def test_process_due_monitor_jobs_uses_configured_intervals(tmp_path, monkeypatch):
    db_path = tmp_path / "monitor.sqlite3"
    monkeypatch.setattr(settings, "market_monitor_enabled", True)
    monkeypatch.setattr(settings, "monitor_price_interval_seconds", 60)
    monkeypatch.setattr(settings, "monitor_disclosure_interval_seconds", 300)
    monkeypatch.setattr(settings, "monitor_news_interval_seconds", 600)
    calls = []

    def fake_run_monitor_job(name, db_path=None):
        calls.append(name)
        return {"status": "success", "job": name}

    monkeypatch.setattr(market_monitor, "run_monitor_job", fake_run_monitor_job)

    processed = market_monitor.process_due_monitor_jobs(db_path=db_path)
    status = market_monitor.get_monitor_status(db_path=db_path)

    assert [item["job"] for item in processed] == [
        market_monitor.JOB_KIS_MARKET,
        market_monitor.JOB_NAVER_NEWS,
        market_monitor.JOB_OPENDART,
    ]
    intervals = {job["name"]: job["interval_seconds"] for job in status["jobs"]}
    assert intervals[market_monitor.JOB_KIS_MARKET] == 60
    assert intervals[market_monitor.JOB_OPENDART] == 300
    assert intervals[market_monitor.JOB_NAVER_NEWS] == 600


def test_monitor_status_endpoint_reports_sqlite_not_supabase(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "market_monitor_db_path", str(tmp_path / "monitor.sqlite3"))

    response = client.get("/monitor/status")

    assert response.status_code == 200
    body = response.json()
    assert body["database"] == "sqlite"
    assert body["uses_supabase"] is False
