from __future__ import annotations

import sqlite3
import time
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.models import AutoTradeStartRequest
from app.strategies.rule_based import build_strategy_decision
from app.workers import manager as worker_manager
from app.trading import auto_trading
from app.trading import auto_trading_store
from app.trading import paper_trading
from app.trading import trade_orchestrator


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


def test_auto_trading_applies_request_sizing_defaults(monkeypatch):
    preview_calls = []

    def fake_create_order_preview(req):
        preview_calls.append(req)
        return {
            "status": "blocked",
            "preview_id": 7,
            "preview_token": None,
            "symbol": req.symbol,
            "signal_type": "entry",
            "side": "BUY",
            "price": req.price,
            "quantity": req.quantity,
            "amount": 0,
            "recommended_quantity": None,
            "message": "captured",
            "strategy_decision": {},
            "risk_decision": None,
            "cost_edge_decision": None,
        }

    monkeypatch.setattr(auto_trading, "create_order_preview", fake_create_order_preview)

    payload = _auto_trade_payload(
        account_equity=20_000_000,
        risk_per_trade=0.004,
        cash_available=500_000,
    )
    payload["symbols"][0].pop("quantity")
    payload["symbols"][0].pop("stop_loss")

    response = client.post(
        "/auto-trading/run-once",
        headers=_auth_headers(),
        json=payload,
    )

    assert response.status_code == 200
    assert preview_calls[0].account_equity == 20_000_000
    assert preview_calls[0].risk_per_trade == 0.004
    assert preview_calls[0].cash_available == 500_000
    assert preview_calls[0].stop_loss == 9700


def test_auto_trading_blocks_before_preview_when_cash_is_insufficient(monkeypatch):
    preview_calls = []
    monkeypatch.setattr(
        auto_trading,
        "create_order_preview",
        lambda req: preview_calls.append(req),
    )

    payload = _auto_trade_payload(account_equity=5_000, cash_available=5_000)
    payload["symbols"][0].pop("quantity")

    response = client.post(
        "/auto-trading/run-once",
        headers=_auth_headers(),
        json=payload,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["status"] == "blocked"
    assert "Insufficient available cash" in body["results"][0]["message"]
    assert preview_calls == []


def test_auto_trading_blocks_new_entries_above_max_open_positions(tmp_path, monkeypatch):
    paper_db = tmp_path / "paper.sqlite3"
    monkeypatch.setattr(paper_trading, "DEFAULT_DB_PATH", paper_db)
    monkeypatch.setattr(settings, "auto_trading_max_open_positions", 1)
    preview_calls = []
    monkeypatch.setattr(
        auto_trading,
        "create_order_preview",
        lambda req: preview_calls.append(req),
    )
    with sqlite3.connect(paper_db) as conn:
        conn.row_factory = sqlite3.Row
        paper_trading.initialize_db(conn)
        conn.execute(
            """
            INSERT INTO positions (
                symbol, name, market, quantity, avg_price, cost_basis,
                realized_pnl, updated_at
            )
            VALUES ('005930', 'Samsung Electronics', 'KR', 1, 10000, 10000, 0, '2026-05-23T09:00:00')
            """
        )

    payload = _auto_trade_payload()
    payload["symbols"][0]["symbol"] = "000660"

    response = client.post(
        "/auto-trading/run-once",
        headers=_auth_headers(),
        json=payload,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["status"] == "blocked"
    assert "Maximum open-position count reached" in body["results"][0]["message"]
    assert preview_calls == []


def test_position_expires_when_symbol_disappears_from_scanner_candidates(
    tmp_path,
    monkeypatch,
):
    paper_db = tmp_path / "paper.sqlite3"
    monkeypatch.setattr(paper_trading, "DEFAULT_DB_PATH", paper_db)
    with sqlite3.connect(paper_db) as conn:
        conn.row_factory = sqlite3.Row
        paper_trading.initialize_db(conn)
        conn.execute(
            """
            INSERT INTO positions (
                symbol, name, market, quantity, avg_price, cost_basis,
                realized_pnl, updated_at
            )
            VALUES ('005930', 'Samsung Electronics', 'KR', 3, 70000, 210000, 0, '2026-05-23T09:00:00')
            """
        )

    exits = auto_trading._expired_candidate_exit_symbols(
        req=auto_trading.AutoTradeStartRequest(),
        active_candidate_symbols={"000660"},
    )
    still_valid = auto_trading._expired_candidate_exit_symbols(
        req=auto_trading.AutoTradeStartRequest(),
        active_candidate_symbols={"005930"},
    )

    assert len(exits) == 1
    assert exits[0].symbol == "005930"
    assert exits[0].requested_action == "exit"
    assert exits[0].quantity == 3
    assert still_valid == []


def test_trade_orchestrator_executes_exit_for_removed_holding(
    tmp_path,
    monkeypatch,
):
    auto_db = tmp_path / "auto.sqlite3"
    paper_db = tmp_path / "paper.sqlite3"
    monkeypatch.setattr(settings, "auto_trading_db_path", str(auto_db))
    monkeypatch.setattr(paper_trading, "DEFAULT_DB_PATH", paper_db)

    with sqlite3.connect(paper_db) as conn:
        conn.row_factory = sqlite3.Row
        paper_trading.initialize_db(conn)
        conn.execute(
            """
            INSERT INTO positions (
                symbol, name, market, quantity, avg_price, cost_basis,
                realized_pnl, updated_at
            )
            VALUES ('005930', 'Samsung Electronics', 'KR', 3, 70000, 210000, 0, '2026-05-23T09:00:00')
            """
        )

    session = auto_trading_store.create_session(
        AutoTradeStartRequest(run_immediately=False)
    )
    run_calls = []
    monkeypatch.setattr(
        trade_orchestrator,
        "get_active_scanner_candidates",
        lambda limit, include_expired: [
            {
                "symbol": "000660",
                "name": "SK hynix",
                "rank": 1,
                "status": "READY",
                "current_price": 180000,
                "net_edge": 120,
                "composite_score": 80,
                "expires_at": "2999-01-01T00:00:00",
            }
        ],
    )

    def fake_run_symbol(req, symbol_cfg, session_id=None):
        run_calls.append((symbol_cfg.symbol, symbol_cfg.requested_action, symbol_cfg.quantity))
        return {
            "symbol": symbol_cfg.symbol,
            "status": "confirmed",
            "action": symbol_cfg.requested_action,
        }

    monkeypatch.setattr(auto_trading, "_run_symbol", fake_run_symbol)

    result = trade_orchestrator.run_trade_orchestrator_once(worker_id="orch-test")
    events = auto_trading_store.list_events(session["session_id"])

    assert result["status"] == "executed"
    assert ("005930", "exit", 3) in run_calls
    assert any(event["event_type"] == "orchestrator_completed" for event in events)


def test_orchestrated_entries_follow_net_edge_priority(monkeypatch):
    run_calls = []
    monkeypatch.setattr(
        auto_trading,
        "_open_positions",
        lambda mode: {},
    )
    monkeypatch.setattr(
        auto_trading,
        "_prepare_symbols_for_account_balance",
        lambda req, symbols: {"symbols": symbols, "results": []},
    )
    monkeypatch.setattr(
        auto_trading,
        "_run_symbol",
        lambda req, symbol_cfg, session_id=None: run_calls.append(symbol_cfg.symbol)
        or {"symbol": symbol_cfg.symbol, "status": "confirmed"},
    )
    monkeypatch.setattr(
        auto_trading,
        "edge_entry_gate",
        lambda candidates=None: {
            "status": "approved",
            "approved": True,
            "message": "test gate approved",
        },
    )

    result = auto_trading.run_orchestrated_candidates_once(
        AutoTradeStartRequest(),
        active_candidates=[
            {
                "symbol": "005930",
                "rank": 2,
                "status": "READY",
                "current_price": 70000,
                "net_edge": 80,
                "composite_score": 75,
                "expires_at": "2999-01-01T00:00:00",
            },
            {
                "symbol": "000660",
                "rank": 1,
                "status": "READY",
                "current_price": 180000,
                "net_edge": 140,
                "composite_score": 70,
                "expires_at": "2999-01-01T00:00:00",
            },
        ],
        session_id="session-test",
    )

    assert result["status"] == "executed"
    assert run_calls[:2] == ["000660", "005930"]


def test_start_reuses_active_session_for_same_account(tmp_path, monkeypatch):
    db_path = tmp_path / "auto.sqlite3"
    monkeypatch.setattr(settings, "auto_trading_db_path", str(db_path))
    monkeypatch.setattr(settings, "auto_trading_one_session_per_account", True)

    first = auto_trading.start_auto_trading(AutoTradeStartRequest())
    second = auto_trading.start_auto_trading(AutoTradeStartRequest())
    active = auto_trading_store.list_sessions(status="active", db_path=db_path)

    assert first["session_id"] == second["session_id"]
    assert len(active) == 1
    assert "no duplicate session" in second["message"]


def test_live_orchestrator_waits_for_exit_fill_before_entry(monkeypatch):
    run_calls = []
    monkeypatch.setattr(settings, "live_exit_confirm_before_entry", True)
    monkeypatch.setattr(
        auto_trading,
        "_open_positions",
        lambda mode: {
            "005930": {
                "symbol": "005930",
                "quantity": 3,
                "current_price": 70000,
            }
        },
    )
    monkeypatch.setattr(
        auto_trading,
        "_run_symbol",
        lambda req, symbol_cfg, session_id=None: run_calls.append(symbol_cfg.symbol)
        or {"symbol": symbol_cfg.symbol, "status": "submitted"},
    )
    monkeypatch.setattr(
        auto_trading.broker_sync,
        "sync_kis_account",
        lambda: {"status": "success", "account_no": "acct"},
    )
    monkeypatch.setattr(
        auto_trading.order_state,
        "reconcile_after_broker_sync",
        lambda **kwargs: {
            "symbol": kwargs["symbol"],
            "state": "EXIT_PENDING",
            "current_quantity": 3,
            "raw": {"message": "Fill not confirmed yet"},
        },
    )
    monkeypatch.setattr(
        auto_trading,
        "edge_entry_gate",
        lambda candidates=None: {
            "status": "approved",
            "approved": True,
            "message": "test gate approved",
        },
    )
    monkeypatch.setattr(
        auto_trading,
        "_prepare_symbols_for_account_balance",
        lambda req, symbols: {"symbols": symbols, "results": []},
    )

    result = auto_trading.run_orchestrated_candidates_once(
        AutoTradeStartRequest(execution_mode="live", live_confirm_token="token"),
        active_candidates=[
            {
                "symbol": "000660",
                "rank": 1,
                "status": "READY",
                "current_price": 180000,
                "net_edge": 140,
                "composite_score": 70,
                "expires_at": "2999-01-01T00:00:00",
            }
        ],
        session_id="session-test",
    )

    assert result["status"] == "blocked"
    assert run_calls == ["005930"]
    assert result["planned_entry_count"] == 1
    assert result["results"][-1]["symbol"] == "__exit_confirmation__"
    assert result["results"][-1]["status"] == "blocked"


def test_auto_trading_uses_live_broker_cash_before_order_preview(monkeypatch):
    preview_calls = []
    monkeypatch.setattr(settings, "enable_live_trading", True)
    monkeypatch.setattr(settings, "kis_is_paper", False)
    monkeypatch.setattr(settings, "live_trading_confirm_token", "confirm-live")
    monkeypatch.setattr(
        auto_trading.broker_sync,
        "sync_kis_account",
        lambda: {
            "status": "success",
            "total_cash": 40_000,
            "total_value": 1_000_000,
        },
    )

    def fake_create_order_preview(req):
        preview_calls.append(req)
        return {
            "status": "blocked",
            "preview_id": 7,
            "preview_token": None,
            "symbol": req.symbol,
            "signal_type": "entry",
            "side": "BUY",
            "price": req.price,
            "quantity": req.quantity,
            "amount": 0,
            "recommended_quantity": None,
            "message": "captured",
            "strategy_decision": {},
            "risk_decision": None,
            "cost_edge_decision": None,
        }

    monkeypatch.setattr(auto_trading, "create_order_preview", fake_create_order_preview)

    payload = _auto_trade_payload(
        execution_mode="live",
        live_confirm_token="confirm-live",
        account_equity=10_000_000,
    )
    payload["symbols"][0].pop("quantity")

    response = client.post(
        "/auto-trading/run-once",
        headers=_auth_headers(),
        json=payload,
    )

    assert response.status_code == 200
    assert preview_calls[0].account_equity == 1_000_000
    assert preview_calls[0].cash_available == 40_000


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


def test_gpt_status_and_start_paper_compatibility_routes(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "auto_trading_db_path", str(tmp_path / "auto.sqlite3"))

    status_response = client.get(
        "/gpt/auto-trading/status",
        headers=_auth_headers(),
    )
    assert status_response.status_code == 200
    assert status_response.json()["command"] == "status"

    start_response = client.post(
        "/gpt/auto-trading/start-paper",
        headers=_auth_headers(),
        json={},
    )
    assert start_response.status_code == 200
    assert start_response.json()["status"] == "started"
    assert start_response.json()["started_session"]["execution_mode"] == "paper"


def test_gpt_control_returns_json_diagnostic_for_missing_api_key(monkeypatch):
    monkeypatch.setattr(settings, "backend_api_key", "secret-key")

    response = client.post(
        "/gpt/auto-trading/control",
        json={"command": "status"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert body["error_type"] == "http_error"
    assert body["http_status"] == 401
    assert "API key" in body["message"]


def test_gpt_kis_routes_return_json_diagnostic_for_missing_api_key(monkeypatch):
    monkeypatch.setattr(settings, "backend_api_key", "secret-key")

    config_response = client.get("/gpt/broker/kis/config-status")
    preflight_response = client.post("/gpt/broker/kis/paper-preflight")

    assert config_response.status_code == 200
    assert config_response.json()["status"] == "error"
    assert config_response.json()["http_status"] == 401
    assert preflight_response.status_code == 200
    assert preflight_response.json()["status"] == "error"
    assert preflight_response.json()["http_status"] == 401


def test_gpt_control_returns_json_diagnostic_for_invalid_body():
    response = client.post(
        "/gpt/auto-trading/control",
        json={"command": "bad-command"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert body["error_type"] == "validation_error"
    assert body["http_status"] == 422


def test_embedded_worker_status_endpoint_reports_disabled(monkeypatch):
    monkeypatch.setattr(settings, "embedded_workers_enabled", False)

    response = client.get("/workers/status", headers=_auth_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is False
    assert "workers" in body


def test_worker_status_compatibility_routes_are_reachable(monkeypatch):
    monkeypatch.setattr(settings, "embedded_workers_enabled", False)

    assert client.get("/workers/status").status_code == 200
    assert client.get("/worker/status").status_code == 200
    assert client.get("/gpt/workers/status").status_code == 200
    assert client.get("/gpt/worker/status").status_code == 200


def test_embedded_worker_specs_include_orchestrator(monkeypatch):
    monkeypatch.setattr(settings, "trade_orchestrator_enabled", True)

    names = [spec.name for spec in worker_manager._worker_specs()]

    assert "orchestrator_worker" in names


def test_api_key_query_fallback_for_gpt_routes(monkeypatch):
    monkeypatch.setattr(settings, "backend_api_key", "secret-key")

    response = client.post(
        "/gpt/auto-trading/control?api_key=secret-key",
        json={"command": "status"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"


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


def test_requested_exit_forces_sell_strategy_decision():
    decision = build_strategy_decision(
        {
            "entry_signal": False,
            "exit_signal": False,
            "confidence": 0.1,
            "scores": {"final_score": 10, "risk_score": 90},
            "summary": "candidate removed from scanner table",
        },
        requested_action="exit",
        risk_level="medium",
    )

    assert decision["approved"] is True
    assert decision["action"] == "exit"
    assert decision["side"] == "SELL"


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
