from __future__ import annotations

import sqlite3

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.trading import paper_trading


client = TestClient(app)


def _auth_headers() -> dict[str, str]:
    if settings.backend_api_key:
        return {"X-API-Key": settings.backend_api_key}
    return {}


def test_position_sizing_blocks_oversized_order(tmp_path, monkeypatch):
    db_path = tmp_path / "paper.sqlite3"
    monkeypatch.setattr(paper_trading, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr(paper_trading, "_now", lambda: "2026-05-20T10:00:00")

    response = client.post(
        "/paper/run-once",
        headers=_auth_headers(),
        json={
            "symbol": "005930",
            "market": "KR",
            "strategy_type": "daytrade",
            "risk_level": "medium",
            "signal_type": "entry",
            "price": 100,
            "quantity": 150,
            "confidence": 0.9,
            "account_equity": 100000,
            "risk_per_trade": 0.005,
            "stop_loss": 95,
        },
    )

    body = response.json()
    assert body["status"] == "rejected"
    assert "position_size_exceeded" in body["message"]


def test_duplicate_entry_order_is_blocked(tmp_path, monkeypatch):
    db_path = tmp_path / "paper.sqlite3"
    monkeypatch.setattr(paper_trading, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr(paper_trading, "_now", lambda: "2026-05-20T10:00:00")
    payload = {
        "symbol": "005930",
        "market": "KR",
        "strategy_type": "daytrade",
        "signal_type": "entry",
        "price": 1000,
        "quantity": 1,
        "confidence": 0.9,
    }

    first = client.post("/paper/run-once", headers=_auth_headers(), json=payload)
    second = client.post("/paper/run-once", headers=_auth_headers(), json=payload)

    assert first.json()["status"] == "success"
    assert second.json()["status"] == "rejected"
    assert "duplicate_order_detected" in second.json()["message"]


def test_required_decision_logs_are_persisted(tmp_path, monkeypatch):
    db_path = tmp_path / "paper.sqlite3"
    monkeypatch.setattr(paper_trading, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr(paper_trading, "_now", lambda: "2026-05-20T10:00:00")

    response = client.post(
        "/paper/run-once",
        headers=_auth_headers(),
        json={
            "symbol": "005930",
            "market": "KR",
            "strategy_type": "daytrade",
            "signal_type": "entry",
            "price": 1000,
            "quantity": 1,
            "confidence": 0.9,
            "signal_time": "2026-05-20T09:59:58",
            "decision_price": 998,
            "order_price": 1000,
            "signal_score": 82,
            "stop_loss": 970,
            "take_profit": 1060,
            "market_regime": "uptrend",
            "reason": "VWAP breakout",
            "model_version": "rule_based_v2",
            "sector": "반도체",
        },
    )

    assert response.json()["status"] == "success"
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT signal_time, decision_price, order_price, fill_price,
                   slippage_bps, signal_score, position_size, stop_loss,
                   take_profit, market_regime, reason, model_version, sector
            FROM paper_orders
            WHERE symbol = '005930'
            """
        ).fetchone()

    assert row == (
        "2026-05-20T09:59:58",
        998,
        1000,
        1000,
        0,
        82,
        0.0001,
        970,
        1060,
        "uptrend",
        "VWAP breakout",
        "rule_based_v2",
        "반도체",
    )
