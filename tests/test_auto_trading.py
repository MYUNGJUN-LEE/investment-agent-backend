from __future__ import annotations

import time
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.trading import auto_trading
from app.trading import auto_trading_store


client = TestClient(app)


def _auth_headers() -> dict[str, str]:
    if settings.backend_api_key:
        return {"X-API-Key": settings.backend_api_key}
    return {}


def _auto_trade_payload(**overrides):
    payload = {
        "execution_mode": "paper",
        "symbols": [
            {
                "symbol": "005930",
                "name": "Samsung Electronics",
                "market": "KR",
                "strategy_type": "daytrade",
                "requested_action": "entry",
                "price": 10000,
                "quantity": 2,
                "stop_loss": 9500,
                "expected_gross_edge_bps": 100,
                "expected_win_bps": 100,
                "expected_loss_bps": 50,
            }
        ],
    }
    payload.update(overrides)
    return payload


def test_auto_trading_run_once_confirms_pending_paper_preview(monkeypatch):
    preview_calls = []
    confirm_calls = []

    def fake_create_order_preview(req):
        preview_calls.append(req)
        return {
            "status": "pending",
            "preview_id": 7,
            "preview_token": "preview-token",
            "symbol": req.symbol,
            "signal_type": "entry",
            "side": "BUY",
            "price": req.price,
            "quantity": req.quantity,
            "amount": req.price * req.quantity,
            "recommended_quantity": None,
            "message": "ready",
            "strategy_decision": {},
            "risk_decision": {"approved": True},
            "cost_edge_decision": None,
        }

    def fake_confirm_order_preview(req):
        confirm_calls.append(req)
        return {
            "status": "confirmed",
            "preview_id": req.preview_id,
            "execution_mode": "paper",
            "paper_result": {"order_status": "FILLED"},
        }

    monkeypatch.setattr(auto_trading, "create_order_preview", fake_create_order_preview)
    monkeypatch.setattr(auto_trading, "confirm_order_preview", fake_confirm_order_preview)

    response = client.post(
        "/auto-trading/run-once",
        headers=_auth_headers(),
        json=_auto_trade_payload(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["execution_mode"] == "paper"
    assert body["results"][0]["status"] == "confirmed"
    assert body["results"][0]["execution"]["paper_result"]["order_status"] == "FILLED"
    assert preview_calls[0].symbol == "005930"
    assert confirm_calls[0].preview_id == 7


def test_auto_trading_start_status_and_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "auto_trading_db_path", str(tmp_path / "auto.sqlite3"))

    response = client.post(
        "/auto-trading/start",
        headers=_auth_headers(),
        json=_auto_trade_payload(
            run_immediately=False,
            interval_seconds=10,
        ),
    )

    assert response.status_code == 200
    started = response.json()
    assert started["status"] == "active"
    session_id = started["session_id"]

    status_response = client.get(
        f"/auto-trading/status/{session_id}",
        headers=_auth_headers(),
    )
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "active"

    stop_response = client.post(
        f"/auto-trading/stop/{session_id}",
        headers=_auth_headers(),
    )
    assert stop_response.status_code == 200
    assert stop_response.json()["status"] == "stopped"

    list_response = client.get(
        "/auto-trading/sessions",
        headers=_auth_headers(),
    )
    assert list_response.status_code == 200
    assert list_response.json()["count"] == 1

    events_response = client.get(
        f"/auto-trading/events/{session_id}",
        headers=_auth_headers(),
    )
    assert events_response.status_code == 200
    assert events_response.json()["count"] >= 2

    restart_response = client.post(
        f"/auto-trading/restart/{session_id}",
        headers=_auth_headers(),
    )
    assert restart_response.status_code == 200
    assert restart_response.json()["status"] == "active"


def test_gpt_auto_trading_control_start_status_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "auto_trading_db_path", str(tmp_path / "auto.sqlite3"))

    start_response = client.post(
        "/gpt/auto-trading/control",
        headers=_auth_headers(),
        json={"command": "start", "execution_mode": "paper"},
    )

    assert start_response.status_code == 200
    started = start_response.json()
    assert started["status"] == "started"
    assert started["started_session"]["status"] == "active"

    duplicate_response = client.post(
        "/gpt/auto-trading/control",
        headers=_auth_headers(),
        json={"command": "start", "execution_mode": "paper"},
    )
    assert duplicate_response.status_code == 200
    assert duplicate_response.json()["status"] == "already_active"

    status_response = client.post(
        "/gpt/auto-trading/control",
        headers=_auth_headers(),
        json={"command": "status"},
    )
    assert status_response.status_code == 200
    assert status_response.json()["active_sessions"]

    stop_response = client.post(
        "/gpt/auto-trading/control",
        headers=_auth_headers(),
        json={"command": "stop"},
    )
    assert stop_response.status_code == 200
    assert stop_response.json()["stopped_sessions"][0]["status"] == "stopped"


def test_embedded_worker_status_endpoint_reports_disabled(monkeypatch):
    monkeypatch.setattr(settings, "embedded_workers_enabled", False)

    response = client.get("/workers/status", headers=_auth_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is False
    assert "workers" in body


def test_auto_trading_dashboard_serves_html():
    response = client.get("/dashboard/auto-trading")

    assert response.status_code == 200
    assert "Auto Trading Dashboard" in response.text


def test_auto_trading_worker_processes_persisted_session(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "auto_trading_db_path", str(tmp_path / "auto.sqlite3"))
    monkeypatch.setattr(
        auto_trading,
        "_run_cycle",
        lambda req, session_id=None: [{"symbol": "005930", "status": "mocked"}],
    )

    started = client.post(
        "/auto-trading/start",
        headers=_auth_headers(),
        json=_auto_trade_payload(max_cycles=1),
    ).json()

    processed = auto_trading.process_due_sessions(worker_id="test-worker")

    assert processed == [
        {
            "session_id": started["session_id"],
            "status": "stopped",
            "results": [{"symbol": "005930", "status": "mocked"}],
        }
    ]
    session = auto_trading_store.get_session(started["session_id"])
    assert session["cycle_count"] == 1
    assert session["status"] == "stopped"
    assert session["last_results"] == [{"symbol": "005930", "status": "mocked"}]


def test_auto_trading_runs_symbols_in_parallel(monkeypatch):
    monkeypatch.setattr(settings, "auto_trading_symbol_workers", 2)

    def slow_run_symbol(req, symbol_cfg, session_id=None):
        time.sleep(0.2)
        return {"symbol": symbol_cfg.symbol, "status": "done"}

    monkeypatch.setattr(auto_trading, "_run_symbol", slow_run_symbol)
    payload = _auto_trade_payload()
    payload["symbols"] = [
        {**payload["symbols"][0], "symbol": "005930"},
        {**payload["symbols"][0], "symbol": "000660"},
    ]
    req = auto_trading.AutoTradeStartRequest(**payload)

    started_at = time.perf_counter()
    results = auto_trading._run_cycle(req)
    elapsed = time.perf_counter() - started_at

    assert [result["symbol"] for result in results] == ["005930", "000660"]
    assert elapsed < 0.35


def test_live_auto_trading_requires_live_gate(monkeypatch):
    monkeypatch.setattr(settings, "enable_live_trading", False)
    monkeypatch.setattr(settings, "kis_is_paper", False)
    monkeypatch.setattr(settings, "live_trading_confirm_token", "confirm-live")

    response = client.post(
        "/auto-trading/start",
        headers=_auth_headers(),
        json=_auto_trade_payload(
            execution_mode="live",
            live_confirm_token="confirm-live",
        ),
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Live trading is disabled"
