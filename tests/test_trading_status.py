from __future__ import annotations

import sqlite3

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.models import PaperRunRequest
from app.trading import auto_trading, execution_status, paper_trading, risk_manager
from app.trading import auto_trading_store
from app.trading import universe_scanner
from app.workers import trading_worker


client = TestClient(app)


def _auth_headers() -> dict[str, str]:
    if settings.backend_api_key:
        return {"X-API-Key": settings.backend_api_key}
    return {}


def _use_tmp_data_dir(tmp_path, monkeypatch) -> None:
    settings.clear_storage_cache()
    for key in (
        "EXECUTION_MODE",
        "TRADING_EXECUTION_MODE",
        "AUTO_TRADING_EXECUTION_MODE",
        "DEFAULT_EXECUTION_MODE",
        "BROKER_PROVIDER",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(settings, "render_disk_mount_path", None)


def test_paper_execution_creates_internal_paper_order(tmp_path, monkeypatch):
    _use_tmp_data_dir(tmp_path, monkeypatch)

    monkeypatch.setattr(
        paper_trading.risk_manager,
        "approve_order",
        lambda *args, **kwargs: risk_manager.RiskDecision(
            approved=True,
            code="approved",
            message="test approved",
            checks={},
        ),
    )

    captured = {}

    def fake_create_order_preview(req):
        captured.update(
            {
                "symbol": req.symbol,
                "name": req.name,
                "market": req.market,
                "strategy_type": req.strategy_type,
                "risk_level": req.risk_level,
                "price": req.price,
                "quantity": req.quantity,
            }
        )
        return {
            "status": "pending",
            "preview_id": 77,
            "preview_token": "preview-token",
            "symbol": req.symbol,
            "signal_type": "entry",
            "side": "BUY",
            "price": req.price,
            "quantity": req.quantity,
            "amount": req.price * req.quantity,
            "recommended_quantity": None,
            "message": "eligible paper preview",
            "strategy_decision": {"approved": True},
            "risk_decision": {"approved": True},
            "cost_edge_decision": None,
        }

    def fake_confirm_order_preview(req):
        paper_result = paper_trading.run_paper_once(
            PaperRunRequest(
                symbol=captured["symbol"],
                name=captured.get("name"),
                market=captured["market"],
                strategy_type=captured["strategy_type"],
                risk_level=captured["risk_level"],
                signal_type="entry",
                price=float(captured["price"]),
                quantity=int(captured["quantity"]),
                confidence=0.9,
                reason="eligible paper candidate",
                source="auto_trading_test",
            )
        )
        return {
            "status": "confirmed",
            "preview_id": req.preview_id,
            "execution_mode": "paper",
            "paper_result": paper_result,
        }

    monkeypatch.setattr(auto_trading, "create_order_preview", fake_create_order_preview)
    monkeypatch.setattr(auto_trading, "confirm_order_preview", fake_confirm_order_preview)

    response = client.post(
        "/auto-trading/run-once",
        headers=_auth_headers(),
        json={
            "execution_mode": "paper",
            "auto_discover_symbols": False,
            "symbols": [
                {
                    "symbol": "005930",
                    "name": "Samsung Electronics",
                    "market": "KR",
                    "strategy_type": "daytrade",
                    "requested_action": "entry",
                    "price": 10000,
                    "quantity": 1,
                    "expected_gross_edge_bps": 120,
                    "expected_win_bps": 120,
                    "expected_loss_bps": 40,
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["status"] == "confirmed"

    paper_db_path = settings.storage_path(paper_trading.DEFAULT_DB_PATH)
    with sqlite3.connect(paper_db_path) as conn:
        order_count = conn.execute("SELECT COUNT(*) FROM paper_orders").fetchone()[0]

    assert order_count >= 1


def test_trading_worker_run_once_initializes_execution_dbs(tmp_path, monkeypatch):
    _use_tmp_data_dir(tmp_path, monkeypatch)
    monkeypatch.setattr(
        trading_worker,
        "process_due_sessions",
        lambda worker_id=None, limit=10: [],
    )

    assert trading_worker.run_once(limit=1) == []
    assert settings.storage_path(settings.auto_trading_db_path).exists()
    assert settings.storage_path(settings.order_state_db_path).exists()


def test_trading_status_exposes_missing_auto_trading_db(tmp_path, monkeypatch):
    _use_tmp_data_dir(tmp_path, monkeypatch)

    response = client.get("/trading/status", headers=_auth_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["auto_trading_db_missing"] is True
    assert body["active_session_count"] == 0
    assert body["resolved_data_dir"] == str(tmp_path)
    assert body["data_dir_writable"] is True
    assert body["storage_root_fallback_used"] is False


def test_execution_mode_env_alias_is_exposed_in_trading_status(tmp_path, monkeypatch):
    _use_tmp_data_dir(tmp_path, monkeypatch)
    monkeypatch.setenv("TRADING_EXECUTION_MODE", "broker_paper")
    monkeypatch.setenv("BROKER_PROVIDER", "kis")
    monkeypatch.setattr(settings, "kis_is_paper", True)
    monkeypatch.setattr(settings, "kis_app_key", "app-key")
    monkeypatch.setattr(settings, "kis_app_secret", "app-secret")
    monkeypatch.setattr(settings, "kis_account_no", "12345678")
    monkeypatch.setattr(settings, "kis_account_product_code", "01")

    status = execution_status.trading_status_snapshot()

    assert status["configured_execution_mode"] == "broker_paper"
    assert status["resolved_execution_mode"] == "broker_paper"
    assert status["execution_mode_source"] == "TRADING_EXECUTION_MODE"
    assert status["execution_mode"] == "broker_paper"
    assert status["configured_broker_provider"] == "kis"
    assert status["broker_provider_source"] == "BROKER_PROVIDER"
    assert status["broker_submit_enabled"] is True


def test_active_paper_session_mismatch_requires_restart(tmp_path, monkeypatch):
    _use_tmp_data_dir(tmp_path, monkeypatch)
    auto_trading_store.create_session(
        auto_trading.AutoTradeStartRequest(
            execution_mode="paper",
            auto_discover_symbols=True,
            run_immediately=True,
        )
    )
    monkeypatch.setenv("EXECUTION_MODE", "broker_paper")
    monkeypatch.setattr(settings, "kis_is_paper", True)
    monkeypatch.setattr(settings, "kis_app_key", "app-key")
    monkeypatch.setattr(settings, "kis_app_secret", "app-secret")
    monkeypatch.setattr(settings, "kis_account_no", "12345678")
    monkeypatch.setattr(settings, "kis_account_product_code", "01")

    status = execution_status.trading_status_snapshot()

    assert status["resolved_execution_mode"] == "broker_paper"
    assert status["active_session_execution_mode"] == "paper"
    assert status["session_mode_mismatch"] is True
    assert status["requires_session_restart"] is True
    assert status["broker_submit_enabled"] is False


def test_trading_restart_stops_stale_paper_session_and_starts_broker_paper(
    tmp_path,
    monkeypatch,
):
    _use_tmp_data_dir(tmp_path, monkeypatch)
    old = auto_trading_store.create_session(
        auto_trading.AutoTradeStartRequest(
            execution_mode="paper",
            auto_discover_symbols=True,
            run_immediately=True,
        )
    )
    monkeypatch.setenv("EXECUTION_MODE", "broker_paper")
    monkeypatch.setenv("BROKER_PROVIDER", "kis")
    monkeypatch.setattr(settings, "kis_is_paper", True)
    monkeypatch.setattr(settings, "kis_app_key", "app-key")
    monkeypatch.setattr(settings, "kis_app_secret", "app-secret")
    monkeypatch.setattr(settings, "kis_account_no", "12345678")
    monkeypatch.setattr(settings, "kis_account_product_code", "01")
    monkeypatch.setattr(
        auto_trading,
        "broker_paper_safety_check",
        lambda **kwargs: {"approved": True, "broker_submit_blocked": False},
    )

    response = client.post("/trading/restart", headers=_auth_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["started_session"]["execution_mode"] == "broker_paper"
    assert body["started_session"]["broker_provider"] == "kis"
    assert body["stopped_sessions"][0]["old_session_id"] == old["session_id"]
    assert auto_trading_store.get_session(old["session_id"])["status"] == "stopped"

    status = execution_status.trading_status_snapshot()
    assert status["active_session_count"] == 1
    assert status["active_session_execution_mode"] == "broker_paper"
    assert status["session_mode_mismatch"] is False


def test_zero_paper_orders_do_not_promote_candidate_label_win_rate(
    tmp_path,
    monkeypatch,
):
    _use_tmp_data_dir(tmp_path, monkeypatch)
    edge_path = settings.storage_path(settings.edge_calibration_db_path)
    edge_path.parent.mkdir(parents=True, exist_ok=True)
    edge_path.touch()

    monkeypatch.setattr(
        execution_status.edge_calibration,
        "edge_entry_gate",
        lambda **kwargs: {"status": "approved", "sample_count": 10},
    )
    monkeypatch.setattr(
        execution_status.edge_calibration,
        "get_edge_training_sample_summary",
        lambda **kwargs: {
            "unit_performance": {
                execution_status.edge_calibration.CANDIDATE_LABEL_UNIT: {
                    "unit": execution_status.edge_calibration.CANDIDATE_LABEL_UNIT,
                    "sample_count": 10,
                    "win_rate": 0.754,
                    "status": "ready",
                }
            }
        },
    )

    status = execution_status.trading_status_snapshot(execution_mode="paper")

    assert status["paper_orders_count"] == 0
    assert status["candidate_label_win_rate"] == 0.754
    assert status["paper_order_win_rate"] is None
    assert status["actual_trading_win_rate"] is None
    assert status["actual_trading_win_rate_display"] == "N/A"
    assert status["candidate_label_win_rate_is_actual_trading_win_rate"] is False


def test_claimed_candidate_without_submission_is_not_submitted(
    tmp_path,
    monkeypatch,
):
    _use_tmp_data_dir(tmp_path, monkeypatch)
    universe_path = settings.storage_path(settings.universe_scanner_db_path)
    universe_scanner.initialize_universe_db(universe_path)
    with sqlite3.connect(universe_path) as conn:
        conn.execute(
            """
            INSERT INTO scanner_candidates (
                scan_id, scan_time, symbol, name, raw_score, expected_return,
                expected_risk, trading_cost, slippage_cost, net_edge,
                composite_score, rank, reason, status, decision, current_price,
                expires_at, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "scan-claimed",
                "2026-06-06T09:00:00",
                "005930",
                "Samsung Electronics",
                80.0,
                120.0,
                40.0,
                5.0,
                3.0,
                112.0,
                75.0,
                1,
                "sample_count 0/600",
                "CLAIMED",
                "BUY",
                10000.0,
                "2999-01-01T00:00:00",
                "{}",
            ),
        )

    status = execution_status.trading_status_snapshot(execution_mode="paper")

    assert status["claimed_candidate_count"] == 1
    assert status["submitted_order_count"] == 0
    assert status["order_status"] == "not_submitted"
