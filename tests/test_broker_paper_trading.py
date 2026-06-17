from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.models import LiveOrderRequest
from app.trading import auto_trading, broker_sync, live_trading, order_state, risk_manager
from app.trading import paper_trading


client = TestClient(app)


def _auth_headers() -> dict[str, str]:
    if settings.backend_api_key:
        return {"X-API-Key": settings.backend_api_key}
    return {}


def _use_tmp_kis_paper_config(tmp_path, monkeypatch) -> None:
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
    monkeypatch.setattr(settings, "kis_is_paper", True)
    monkeypatch.setattr(settings, "kis_app_key", "app-key")
    monkeypatch.setattr(settings, "kis_app_secret", "app-secret")
    monkeypatch.setattr(settings, "kis_account_no", "12345678")
    monkeypatch.setattr(settings, "kis_account_product_code", "01")


def _approve_all_orders(monkeypatch) -> None:
    monkeypatch.setattr(
        live_trading.risk_manager,
        "approve_order",
        lambda *args, **kwargs: risk_manager.RiskDecision(
            approved=True,
            code="approved",
            message="ok",
            checks={},
        ),
    )


class FakeKisPaperClient:
    calls: list[dict[str, object]] = []

    def __init__(self, is_paper: bool = True):
        self.is_paper = is_paper
        self.base_url = live_trading.KIS_PAPER_BASE_URL
        self.app_key = "app-key"
        self.app_secret = "app-secret"
        self.account_no = "12345678"
        self.account_product_code = "01"

    def runtime_diagnostics(self):
        return {
            "base_url": self.base_url,
            "is_paper": self.is_paper,
            "app_key_configured": True,
            "app_secret_configured": True,
            "account_no_configured": True,
            "account_product_code": "01",
        }

    def get_current_price(self, symbol):
        return {"output": {"stck_prpr": "75000"}}

    def get_balance(self):
        return {"output2": [{"dnca_tot_amt": "10000000", "tot_evlu_amt": "10000000"}]}

    def place_domestic_limit_order(self, symbol, side, price, quantity):
        self.calls.append(
            {
                "symbol": symbol,
                "side": side,
                "price": price,
                "quantity": quantity,
                "is_paper": self.is_paper,
            }
        )
        return {
            "rt_cd": "0",
            "msg_cd": "APBK0013",
            "msg1": "paper order accepted",
            "output": {"ODNO": "PAPER123"},
        }


class RejectingKisPaperClient(FakeKisPaperClient):
    def place_domestic_limit_order(self, symbol, side, price, quantity):
        self.calls.append(
            {
                "symbol": symbol,
                "side": side,
                "price": price,
                "quantity": quantity,
                "is_paper": self.is_paper,
            }
        )
        return {
            "rt_cd": "1",
            "msg_cd": "REJECT",
            "msg1": "mock rejected",
            "output": {},
        }


class ZeroCashKisPaperClient(FakeKisPaperClient):
    def get_balance(self):
        return {"output2": [{"dnca_tot_amt": "0", "ord_psbl_cash": "0"}]}


def test_paper_mode_does_not_call_kis_submit(tmp_path, monkeypatch):
    _use_tmp_kis_paper_config(tmp_path, monkeypatch)

    def fail_broker_submit(*args, **kwargs):
        raise AssertionError("paper mode must not submit to KIS")

    monkeypatch.setattr(auto_trading, "execute_broker_paper_order", fail_broker_submit)
    monkeypatch.setattr(
        auto_trading,
        "create_order_preview",
        lambda req: {
            "status": "pending",
            "preview_id": 1,
            "preview_token": "token",
            "symbol": req.symbol,
            "signal_type": "entry",
            "side": "BUY",
            "price": req.price,
            "quantity": req.quantity,
            "amount": req.price * req.quantity,
            "recommended_quantity": None,
            "message": "ok",
            "strategy_decision": {},
            "risk_decision": {"approved": True},
            "cost_edge_decision": None,
        },
    )
    monkeypatch.setattr(
        auto_trading,
        "confirm_order_preview",
        lambda req: {
            "status": "confirmed",
            "preview_id": req.preview_id,
            "execution_mode": "paper",
            "paper_result": {"order_status": "FILLED"},
        },
    )

    response = client.post(
        "/auto-trading/run-once",
        headers=_auth_headers(),
        json={
            "execution_mode": "paper",
            "auto_discover_symbols": False,
            "symbols": [
                {
                    "symbol": "005930",
                    "price": 75000,
                    "quantity": 1,
                    "expected_gross_edge_bps": 100,
                    "expected_win_bps": 100,
                    "expected_loss_bps": 40,
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["execution_mode"] == "paper"
    assert response.json()["results"][0]["status"] == "confirmed"


def test_broker_paper_submits_kis_order_and_records_event(tmp_path, monkeypatch):
    _use_tmp_kis_paper_config(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "broker_paper_max_order_krw", 100_000.0)
    _approve_all_orders(monkeypatch)
    FakeKisPaperClient.calls = []
    monkeypatch.setattr(live_trading, "KisClient", FakeKisPaperClient)
    monkeypatch.setattr(
        auto_trading,
        "broker_paper_safety_check",
        lambda **kwargs: {"approved": True, "broker_submit_blocked": False},
    )
    monkeypatch.setattr(
        auto_trading.broker_sync,
        "sync_kis_account",
        lambda **kwargs: {
            "status": "success",
            "account_no": "12345678",
            "total_cash": 10_000_000,
            "total_value": 10_000_000,
            "execution_count": 0,
            "synced_at": "2026-06-06T09:00:00",
        },
    )
    monkeypatch.setattr(
        auto_trading,
        "create_order_preview",
        lambda req: {
            "status": "pending",
            "preview_id": 1,
            "preview_token": "token",
            "symbol": req.symbol,
            "signal_type": "entry",
            "side": "BUY",
            "price": req.price,
            "quantity": req.quantity,
            "amount": req.price * req.quantity,
            "recommended_quantity": None,
            "message": "ok",
            "strategy_decision": {},
            "risk_decision": {"approved": True},
            "cost_edge_decision": None,
        },
    )

    response = client.post(
        "/auto-trading/run-once",
        headers=_auth_headers(),
        json={
            "execution_mode": "broker_paper",
            "broker_provider": "kis",
            "auto_discover_symbols": False,
            "symbols": [
                {
                    "symbol": "005930",
                    "price": 75000,
                    "quantity": 3,
                    "expected_gross_edge_bps": 100,
                    "expected_win_bps": 100,
                    "expected_loss_bps": 40,
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["status"] == "submitted"
    assert FakeKisPaperClient.calls == [
        {
            "symbol": "005930",
            "side": "buy",
            "price": 75000.0,
            "quantity": 3,
            "is_paper": True,
        }
    ]

    state_path = settings.storage_path(settings.order_state_db_path)
    with sqlite3.connect(state_path) as conn:
        row = conn.execute(
            """
            SELECT symbol, side, qty, execution_mode, broker_order_id, order_status
            FROM broker_order_events
            """
        ).fetchone()

    assert row == ("005930", "buy", 3, "broker_paper", "PAPER123", "submitted")
    with sqlite3.connect(state_path) as conn:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(broker_order_events)").fetchall()
        }
        event_row = conn.execute(
            """
            SELECT session_id, scan_id, name, notional_krw
            FROM broker_order_events
            """
        ).fetchone()

    assert {"session_id", "scan_id", "name", "notional_krw"}.issubset(columns)
    assert event_row[3] == 225000.0

    status_response = client.get(
        "/trading/status?execution_mode=broker_paper",
        headers=_auth_headers(),
    )
    assert status_response.status_code == 200
    status = status_response.json()
    assert status["execution_mode"] == "broker_paper"
    assert status["submits_to_broker"] is True
    assert status["uses_internal_paper_orders"] is False
    assert status["submitted_order_count"] > 0
    assert status["latest_broker_order_event"]["broker_order_id"] == "PAPER123"


def test_broker_paper_blocks_when_account_probe_cash_is_zero(tmp_path, monkeypatch):
    _use_tmp_kis_paper_config(tmp_path, monkeypatch)
    _approve_all_orders(monkeypatch)
    ZeroCashKisPaperClient.calls = []
    monkeypatch.setattr(live_trading, "KisClient", ZeroCashKisPaperClient)
    monkeypatch.setattr(
        auto_trading,
        "broker_paper_safety_check",
        lambda **kwargs: {"approved": True, "broker_submit_blocked": False},
    )
    monkeypatch.setattr(
        auto_trading.broker_sync,
        "sync_kis_account",
        lambda **kwargs: {
            "status": "success",
            "account_no": "12345678",
            "total_cash": 10_000_000,
            "total_value": 10_000_000,
            "execution_count": 0,
            "synced_at": "2026-06-06T09:00:00",
        },
    )
    monkeypatch.setattr(
        auto_trading,
        "create_order_preview",
        lambda req: {
            "status": "pending",
            "preview_id": 1,
            "preview_token": "token",
            "symbol": req.symbol,
            "signal_type": "entry",
            "side": "BUY",
            "price": req.price,
            "quantity": req.quantity,
            "amount": req.price * req.quantity,
            "recommended_quantity": None,
            "message": "ok",
            "strategy_decision": {},
            "risk_decision": {"approved": True},
            "cost_edge_decision": None,
        },
    )

    response = client.post(
        "/auto-trading/run-once",
        headers=_auth_headers(),
        json={
            "execution_mode": "broker_paper",
            "broker_provider": "kis",
            "auto_discover_symbols": False,
            "symbols": [
                {
                    "symbol": "005930",
                    "price": 75000,
                    "quantity": 1,
                    "expected_gross_edge_bps": 100,
                    "expected_win_bps": 100,
                    "expected_loss_bps": 40,
                }
            ],
        },
    )

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["status"] == "error"
    assert result["broker_submit_blocked"] is True
    assert result["broker_submit_block_code"] == "cash_or_buying_power_zero"
    assert "cash/buying_power is zero" in result["message"]
    assert ZeroCashKisPaperClient.calls == []


def test_broker_paper_prevents_second_buy_while_active_order_is_open(tmp_path, monkeypatch):
    _use_tmp_kis_paper_config(tmp_path, monkeypatch)
    _approve_all_orders(monkeypatch)
    FakeKisPaperClient.calls = []
    monkeypatch.setattr(live_trading, "KisClient", FakeKisPaperClient)
    monkeypatch.setattr(
        auto_trading,
        "broker_paper_safety_check",
        lambda **kwargs: {"approved": True, "broker_submit_blocked": False},
    )
    monkeypatch.setattr(
        auto_trading.broker_sync,
        "sync_kis_account",
        lambda **kwargs: {
            "status": "success",
            "account_no": "12345678",
            "total_cash": 10_000_000,
            "total_value": 10_000_000,
            "execution_count": 0,
            "synced_at": "2026-06-06T09:00:00",
        },
    )
    monkeypatch.setattr(
        auto_trading,
        "create_order_preview",
        lambda req: {
            "status": "pending",
            "preview_id": 1,
            "preview_token": "token",
            "symbol": req.symbol,
            "signal_type": "entry",
            "side": "BUY",
            "price": req.price,
            "quantity": req.quantity,
            "amount": req.price * req.quantity,
            "recommended_quantity": None,
            "message": "ok",
            "strategy_decision": {},
            "risk_decision": {"approved": True},
            "cost_edge_decision": None,
        },
    )
    payload = {
        "execution_mode": "broker_paper",
        "broker_provider": "kis",
        "auto_discover_symbols": False,
        "symbols": [
            {
                "symbol": "005930",
                "price": 75000,
                "quantity": 1,
                "scan_id": "scan-dup",
                "expected_gross_edge_bps": 100,
                "expected_win_bps": 100,
                "expected_loss_bps": 40,
            }
        ],
    }

    first = client.post(
        "/auto-trading/run-once",
        headers=_auth_headers(),
        json=payload,
    )
    second = client.post(
        "/auto-trading/run-once",
        headers=_auth_headers(),
        json=payload,
    )

    assert first.status_code == 200
    assert first.json()["results"][0]["status"] == "submitted"
    assert second.status_code == 200
    blocked = second.json()["results"][0]
    assert blocked["status"] == "error"
    assert blocked["broker_submit_blocked"] is True
    assert blocked["broker_submit_block_code"] == "open_broker_order_exists"
    assert len(FakeKisPaperClient.calls) == 1

    status = client.get(
        "/trading/status?execution_mode=broker_paper",
        headers=_auth_headers(),
    ).json()
    assert status["open_broker_order_count"] == 1
    assert status["last_order_by_symbol"]["005930"]["scan_id"] == "scan-dup"


def test_broker_paper_artificial_limits_do_not_block_final_prior_orders(
    tmp_path,
    monkeypatch,
):
    _use_tmp_kis_paper_config(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "broker_paper_max_order_krw", 0.0)
    monkeypatch.setattr(settings, "broker_paper_max_daily_orders", 0)
    monkeypatch.setattr(settings, "broker_paper_max_daily_orders_per_symbol", 0)
    monkeypatch.setattr(settings, "broker_paper_symbol_cooldown_days", 0)
    monkeypatch.setattr(settings, "broker_paper_max_daily_notional_krw", 0.0)

    for index, (symbol, status) in enumerate(
        (
            ("111111", "filled"),
            ("222222", "filled"),
            ("333333", "filled"),
            ("444444", "filled"),
            ("444444", "rejected"),
        ),
        start=1,
    ):
        order_state.record_broker_order_event(
            {
                "session_id": f"session-{index}",
                "scan_id": f"scan-{index}",
                "symbol": symbol,
                "side": "buy",
                "qty": 1,
                "order_type": "limit",
                "limit_price": 100_000,
                "submitted_price": 100_000,
                "broker_provider": "kis",
                "kis_is_paper": True,
                "execution_mode": "broker_paper",
                "broker_order_id": f"FILLED{index}",
                "order_status": status,
                "raw_response": {"rt_cd": "0"},
            }
        )

    req = LiveOrderRequest(
        symbol="444444",
        market="KR",
        broker_provider="kis",
        side="buy",
        order_type="limit",
        price=1_000_000,
        quantity=10,
        confirm_token="broker_paper",
        decision_price=1_000_000,
        scan_id="scan-new",
    )
    guard = order_state.validate_broker_paper_order(req)

    assert guard["approved"] is True
    assert guard["broker_order_risk_limits"]["max_order_krw"] == 0.0
    assert guard["broker_order_risk_limits"]["max_daily_orders"] == 0
    assert guard["broker_order_risk_limits"]["max_daily_orders_per_symbol"] == 0
    assert guard["broker_order_risk_limits"]["symbol_cooldown_days"] == 0
    assert guard["broker_order_risk_limits"]["max_daily_notional_krw"] == 0.0

    assert guard["broker_submit_blocked"] is False
    assert guard["broker_submit_block_code"] is None


def test_broker_paper_blocks_active_broker_order_for_same_symbol(
    tmp_path,
    monkeypatch,
):
    _use_tmp_kis_paper_config(tmp_path, monkeypatch)
    order_state.record_broker_order_event(
        {
            "session_id": "session-open",
            "scan_id": "scan-open",
            "symbol": "444444",
            "side": "buy",
            "qty": 1,
            "order_type": "limit",
            "limit_price": 1_000_000,
            "submitted_price": 1_000_000,
            "broker_provider": "kis",
            "kis_is_paper": True,
            "execution_mode": "broker_paper",
            "broker_order_id": "OPEN1",
            "order_status": "submitted",
            "raw_response": {"rt_cd": "0"},
        }
    )
    req = LiveOrderRequest(
        symbol="444444",
        market="KR",
        broker_provider="kis",
        side="buy",
        order_type="limit",
        price=1_000_000,
        quantity=1,
        confirm_token="broker_paper",
        decision_price=1_000_000,
        scan_id="scan-new",
    )

    blocked = order_state.validate_broker_paper_order(req)

    assert blocked["approved"] is False
    assert blocked["broker_submit_block_code"] == "open_broker_order_exists"
    assert blocked["broker_submit_block_reason"] == (
        "Open broker order already exists for this symbol"
    )


def test_broker_paper_blocks_pending_order_intent_for_same_symbol(
    tmp_path,
    monkeypatch,
):
    _use_tmp_kis_paper_config(tmp_path, monkeypatch)
    pending_req = LiveOrderRequest(
        symbol="555555",
        market="KR",
        broker_provider="kis",
        side="buy",
        order_type="limit",
        price=50_000,
        quantity=1,
        confirm_token="broker_paper",
        decision_price=50_000,
        client_order_id="pending-intent",
    )
    order_state.begin_order_intent(pending_req)
    next_req = pending_req.model_copy(
        update={"client_order_id": "next-intent", "scan_id": "scan-next"}
    )

    blocked = order_state.validate_broker_paper_order(next_req)

    assert blocked["approved"] is False
    assert blocked["broker_submit_block_code"] == "open_order_intent_exists"
    assert blocked["broker_submit_block_reason"] == (
        "Open order intent already exists for this symbol"
    )


def test_order_state_blocks_duplicate_idempotency_key(tmp_path, monkeypatch):
    _use_tmp_kis_paper_config(tmp_path, monkeypatch)
    req = LiveOrderRequest(
        symbol="555555",
        market="KR",
        broker_provider="kis",
        side="buy",
        order_type="limit",
        price=50_000,
        quantity=1,
        confirm_token="broker_paper",
        decision_price=50_000,
        client_order_id="same-key",
    )
    order_state.begin_order_intent(req)

    with pytest.raises(order_state.OrderStateError) as exc:
        order_state.begin_order_intent(req.model_copy(update={"symbol": "666666"}))

    assert exc.value.code == "duplicate_idempotency_key"


def test_broker_paper_blocks_existing_long_position_and_flat_sell(
    tmp_path,
    monkeypatch,
):
    _use_tmp_kis_paper_config(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "allow_position_additions", False)
    broker_db = settings.storage_path(settings.broker_sync_db_path)
    broker_sync.record_kis_sync(
        balance={
            "output1": [
                {
                    "pdno": "777777",
                    "prdt_name": "Held",
                    "hldg_qty": "2",
                    "pchs_avg_pric": "50000",
                }
            ],
            "output2": [{"tot_evlu_amt": "100000"}],
        },
        executions={"output1": []},
        account_no="12345678",
        db_path=broker_db,
    )
    order_state.reconcile_after_broker_sync(
        symbol="777777",
        market="KR",
        account_no="12345678",
        broker_db_path=broker_db,
    )
    buy_req = LiveOrderRequest(
        symbol="777777",
        market="KR",
        broker_provider="kis",
        side="buy",
        order_type="limit",
        price=50_000,
        quantity=1,
        confirm_token="broker_paper",
        decision_price=50_000,
        scan_id="scan-long",
    )

    blocked = order_state.validate_broker_paper_order(buy_req)

    assert blocked["approved"] is False
    assert blocked["broker_submit_block_code"] == "already_position_exists"

    sell_req = LiveOrderRequest(
        symbol="888888",
        market="KR",
        broker_provider="kis",
        side="sell",
        order_type="limit",
        price=50_000,
        quantity=1,
        confirm_token="broker_paper",
        decision_price=50_000,
        client_order_id="flat-sell",
    )
    with pytest.raises(order_state.OrderStateError) as exc:
        order_state.begin_order_intent(sell_req)

    assert exc.value.code == "position_already_flat"


def test_claimed_broker_paper_candidate_records_no_order_reason(
    tmp_path,
    monkeypatch,
):
    _use_tmp_kis_paper_config(tmp_path, monkeypatch)
    monkeypatch.setattr(
        auto_trading,
        "broker_paper_safety_check",
        lambda **kwargs: {"approved": True, "broker_submit_blocked": False},
    )
    monkeypatch.setattr(
        auto_trading.broker_sync,
        "sync_kis_account",
        lambda **kwargs: {
            "status": "success",
            "account_no": "12345678",
            "total_cash": 10_000_000,
            "total_value": 10_000_000,
            "execution_count": 0,
            "synced_at": "2026-06-06T09:00:00",
        },
    )
    monkeypatch.setattr(
        auto_trading,
        "create_order_preview",
        lambda req: {
            "status": "blocked",
            "preview_id": 1,
            "preview_token": None,
            "symbol": req.symbol,
            "signal_type": "entry",
            "side": "BUY",
            "price": req.price,
            "quantity": req.quantity,
            "amount": req.price * req.quantity,
            "recommended_quantity": None,
            "message": "Risk blocked this order preview",
            "strategy_decision": {"approved": True},
            "risk_decision": {
                "approved": False,
                "code": "risk_cap",
                "message": "risk cap",
            },
            "cost_edge_decision": None,
        },
    )

    response = client.post(
        "/auto-trading/run-once",
        headers=_auth_headers(),
        json={
            "execution_mode": "broker_paper",
            "broker_provider": "kis",
            "auto_discover_symbols": False,
            "symbols": [
                {
                    "symbol": "005930",
                    "price": 75000,
                    "quantity": 1,
                    "claimed": True,
                    "claim_time": "2026-06-06T09:00:00",
                    "candidate_status": "CLAIMED",
                    "candidate_decision": "buy_candidate",
                    "expected_gross_edge_bps": 100,
                    "expected_win_bps": 100,
                    "expected_loss_bps": 40,
                }
            ],
        },
    )

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["claimed"] is True
    assert result["broker_submit_attempted"] is False
    assert result["claimed_no_order_reason"] == "risk_manager_rejected"
    assert result["post_claim_diagnostics"]["risk_approved"] is False


def test_claimed_broker_paper_candidate_blocked_by_open_order_has_diagnostics(
    tmp_path,
    monkeypatch,
):
    _use_tmp_kis_paper_config(tmp_path, monkeypatch)
    _approve_all_orders(monkeypatch)
    FakeKisPaperClient.calls = []
    monkeypatch.setattr(live_trading, "KisClient", FakeKisPaperClient)
    monkeypatch.setattr(
        auto_trading,
        "broker_paper_safety_check",
        lambda **kwargs: {"approved": True, "broker_submit_blocked": False},
    )
    monkeypatch.setattr(
        auto_trading.broker_sync,
        "sync_kis_account",
        lambda **kwargs: {
            "status": "success",
            "account_no": "12345678",
            "total_cash": 10_000_000,
            "total_value": 10_000_000,
            "execution_count": 0,
            "synced_at": "2026-06-06T09:00:00",
        },
    )
    monkeypatch.setattr(
        auto_trading,
        "create_order_preview",
        lambda req: {
            "status": "pending",
            "preview_id": 1,
            "preview_token": "token",
            "symbol": req.symbol,
            "signal_type": "entry",
            "side": "BUY",
            "price": req.price,
            "quantity": req.quantity,
            "amount": req.price * req.quantity,
            "recommended_quantity": None,
            "message": "ok",
            "strategy_decision": {"approved": True},
            "risk_decision": {"approved": True},
            "cost_edge_decision": None,
        },
    )
    order_state.record_broker_order_event(
        {
            "session_id": "session-open",
            "scan_id": "scan-open",
            "symbol": "005930",
            "side": "buy",
            "qty": 1,
            "order_type": "limit",
            "limit_price": 75000,
            "submitted_price": 75000,
            "broker_provider": "kis",
            "kis_is_paper": True,
            "execution_mode": "broker_paper",
            "broker_order_id": "OPEN1",
            "order_status": "submitted",
            "raw_response": {"rt_cd": "0"},
        }
    )

    response = client.post(
        "/auto-trading/run-once",
        headers=_auth_headers(),
        json={
            "execution_mode": "broker_paper",
            "broker_provider": "kis",
            "auto_discover_symbols": False,
            "symbols": [
                {
                    "symbol": "005930",
                    "price": 75000,
                    "quantity": 1,
                    "claimed": True,
                    "claim_time": "2026-06-06T09:00:00",
                    "candidate_status": "CLAIMED",
                    "candidate_decision": "buy_candidate",
                    "expected_gross_edge_bps": 100,
                    "expected_win_bps": 100,
                    "expected_loss_bps": 40,
                }
            ],
        },
    )

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["claimed"] is True
    assert result["broker_submit_blocked"] is True
    assert result["broker_submit_block_code"] == "open_broker_order_exists"
    assert result["claimed_no_order_reason"] == "open_order_exists"
    assert result["order_state_code"] == "open_broker_order_exists"
    assert result["order_state_message"] == (
        "Open broker order already exists for this symbol"
    )
    diagnostics = result["post_claim_diagnostics"]
    assert diagnostics["broker_submit_blocked"] is True
    assert diagnostics["broker_submit_block_code"] == "open_broker_order_exists"
    assert diagnostics["broker_submit_block_reason"] == (
        "Open broker order already exists for this symbol"
    )
    assert diagnostics["order_state_code"] == "open_broker_order_exists"
    assert diagnostics["claimed_no_order_reason"] == "open_order_exists"
    assert FakeKisPaperClient.calls == []


def test_broker_paper_scan_symbol_side_dedupe_blocks_filled_prior_event(
    tmp_path,
    monkeypatch,
):
    _use_tmp_kis_paper_config(tmp_path, monkeypatch)
    order_state.record_broker_order_event(
        {
            "session_id": "session-1",
            "scan_id": "scan-1",
            "symbol": "005930",
            "name": "Samsung Electronics",
            "side": "buy",
            "qty": 1,
            "order_type": "limit",
            "limit_price": 75000,
            "submitted_price": 75000,
            "broker_provider": "kis",
            "kis_is_paper": True,
            "execution_mode": "broker_paper",
            "broker_order_id": "FILLED1",
            "order_status": "filled",
            "raw_response": {"rt_cd": "0"},
        }
    )
    req = LiveOrderRequest(
        symbol="005930",
        name="Samsung Electronics",
        market="KR",
        broker_provider="kis",
        side="buy",
        order_type="limit",
        price=75000,
        quantity=1,
        confirm_token="broker_paper",
        decision_price=75000,
        scan_id="scan-1",
    )

    guard = order_state.validate_broker_paper_order(req)

    assert guard["approved"] is False
    assert guard["broker_submit_block_code"] == "duplicate_scan_symbol_side"
    assert guard["broker_submit_blocked"] is True


def test_broker_paper_refuses_when_kis_is_not_paper(tmp_path, monkeypatch):
    _use_tmp_kis_paper_config(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "kis_is_paper", False)
    called = False

    def fake_submit(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(auto_trading, "execute_broker_paper_order", fake_submit)

    response = client.post(
        "/auto-trading/run-once",
        headers=_auth_headers(),
        json={
            "execution_mode": "broker_paper",
            "auto_discover_symbols": False,
            "symbols": [{"symbol": "005930", "price": 75000, "quantity": 1}],
        },
    )

    assert response.status_code == 403
    assert "KIS_IS_PAPER=true" in response.json()["detail"]
    assert called is False

    status_response = client.get(
        "/trading/status?execution_mode=broker_paper",
        headers=_auth_headers(),
    )
    status = status_response.json()
    assert status["broker_submit_blocked"] is True
    assert "KIS_IS_PAPER=true" in status["broker_submit_block_reason"]
    assert status["submits_to_broker"] is False


def test_broker_paper_claimed_candidate_without_submit_is_not_submitted(
    tmp_path,
    monkeypatch,
):
    _use_tmp_kis_paper_config(tmp_path, monkeypatch)
    from app.trading import universe_scanner

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
                "sample_count 0/100",
                "CLAIMED",
                "BUY",
                10000.0,
                "2999-01-01T00:00:00",
                "{}",
            ),
        )

    status = client.get(
        "/trading/status?execution_mode=broker_paper",
        headers=_auth_headers(),
    ).json()

    assert status["claimed_candidate_count"] == 1
    assert status["submitted_order_count"] == 0
    assert status["order_status"] == "not_submitted"


def test_broker_paper_rejected_response_records_rejected_event(tmp_path, monkeypatch):
    _use_tmp_kis_paper_config(tmp_path, monkeypatch)
    _approve_all_orders(monkeypatch)
    RejectingKisPaperClient.calls = []
    req = LiveOrderRequest(
        symbol="005930",
        market="KR",
        broker_provider="kis",
        side="buy",
        order_type="limit",
        price=75000,
        quantity=1,
        confirm_token="broker_paper",
        decision_price=75000,
    )

    result = live_trading.execute_broker_paper_order(
        req,
        client=RejectingKisPaperClient(is_paper=True),
    )

    assert result["status"] == "rejected"
    assert result["order_event"]["order_status"] == "rejected"
    assert result["order_event"]["reject_reason"] == "mock rejected"
    latest = order_state.latest_broker_order_event()
    assert latest["order_status"] == "rejected"
    assert latest["reject_reason"] == "mock rejected"
