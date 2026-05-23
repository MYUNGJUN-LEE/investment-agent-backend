from __future__ import annotations

import pytest

from app.config import settings
from app.models import LiveOrderRequest
from app.trading import broker_sync, order_state


def _live_req(**overrides) -> LiveOrderRequest:
    payload = {
        "symbol": "005930",
        "market": "KR",
        "risk_level": "medium",
        "side": "buy",
        "order_type": "limit",
        "price": 75000,
        "quantity": 1,
        "confirm_token": "confirm-live",
        "decision_price": 75000,
    }
    payload.update(overrides)
    return LiveOrderRequest(**payload)


def test_order_state_blocks_second_order_while_transition_is_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "order_state_db_path", str(tmp_path / "state.sqlite3"))

    first = order_state.begin_order_intent(_live_req(client_order_id="first"))

    assert first["status"] == "PENDING"
    assert first["position_state_after"] == "ENTRY_PENDING"

    with pytest.raises(order_state.OrderStateError) as exc:
        order_state.begin_order_intent(_live_req(client_order_id="second"))

    assert exc.value.code == "position_transition_pending"


def test_order_state_reconciles_broker_position_and_blocks_duplicate_entry(
    tmp_path,
    monkeypatch,
):
    state_db = tmp_path / "state.sqlite3"
    broker_db = tmp_path / "broker.sqlite3"
    monkeypatch.setattr(settings, "order_state_db_path", str(state_db))
    monkeypatch.setattr(settings, "broker_sync_db_path", str(broker_db))
    monkeypatch.setattr(settings, "allow_position_additions", False)

    broker_sync.record_kis_sync(
        balance={
            "output1": [
                {
                    "pdno": "005930",
                    "prdt_name": "Samsung Electronics",
                    "hldg_qty": "3",
                    "pchs_avg_pric": "75000",
                }
            ],
            "output2": [{"tot_evlu_amt": "225000"}],
        },
        executions={"output1": []},
        account_no="12345678",
        db_path=broker_db,
    )

    state = order_state.reconcile_after_broker_sync(
        symbol="005930",
        market="KR",
        account_no="12345678",
        state_db_path=state_db,
        broker_db_path=broker_db,
    )

    assert state["state"] == "LONG"
    assert state["current_quantity"] == 3

    with pytest.raises(order_state.OrderStateError) as exc:
        order_state.begin_order_intent(_live_req(client_order_id="add-entry"))

    assert exc.value.code == "position_already_long"


def test_order_state_keeps_entry_pending_when_sync_does_not_confirm_fill(
    tmp_path,
    monkeypatch,
):
    state_db = tmp_path / "state.sqlite3"
    broker_db = tmp_path / "broker.sqlite3"
    monkeypatch.setattr(settings, "order_state_db_path", str(state_db))
    monkeypatch.setattr(settings, "broker_sync_db_path", str(broker_db))

    intent = order_state.begin_order_intent(_live_req(client_order_id="pending-buy"))
    order_state.mark_order_submitted(intent["id"], {"output": {"ODNO": "12345"}})
    broker_sync.record_kis_sync(
        balance={"output1": [], "output2": [{"tot_evlu_amt": "1000000"}]},
        executions={"output1": []},
        account_no="12345678",
        db_path=broker_db,
    )

    state = order_state.reconcile_after_broker_sync(
        symbol="005930",
        market="KR",
        account_no="12345678",
        state_db_path=state_db,
        broker_db_path=broker_db,
    )
    stored_intent = order_state.get_order_intent(intent["id"], db_path=state_db)

    assert state["state"] == "ENTRY_PENDING"
    assert state["pending_order_intent_id"] == intent["id"]
    assert stored_intent["status"] == "SUBMITTED"


def test_order_state_failed_order_restores_previous_quantity(tmp_path, monkeypatch):
    state_db = tmp_path / "state.sqlite3"
    broker_db = tmp_path / "broker.sqlite3"
    monkeypatch.setattr(settings, "order_state_db_path", str(state_db))
    monkeypatch.setattr(settings, "broker_sync_db_path", str(broker_db))

    broker_sync.record_kis_sync(
        balance={
            "output1": [
                {
                    "pdno": "005930",
                    "hldg_qty": "7",
                    "pchs_avg_pric": "75000",
                }
            ],
            "output2": [{"tot_evlu_amt": "525000"}],
        },
        executions={"output1": []},
        account_no="12345678",
        db_path=broker_db,
    )
    order_state.reconcile_after_broker_sync(
        symbol="005930",
        account_no="12345678",
        state_db_path=state_db,
        broker_db_path=broker_db,
    )

    intent = order_state.begin_order_intent(
        _live_req(side="sell", quantity=2, client_order_id="sell-fail")
    )
    failed = order_state.mark_order_failed(intent["id"], "forced failure", db_path=state_db)
    state = order_state.get_position_state("005930", db_path=state_db)

    assert failed["status"] == "FAILED"
    assert state["state"] == "LONG"
    assert state["current_quantity"] == 7


def test_order_state_marks_partial_sell_from_broker_quantity(tmp_path, monkeypatch):
    state_db = tmp_path / "state.sqlite3"
    broker_db = tmp_path / "broker.sqlite3"
    monkeypatch.setattr(settings, "order_state_db_path", str(state_db))
    monkeypatch.setattr(settings, "broker_sync_db_path", str(broker_db))

    broker_sync.record_kis_sync(
        balance={
            "output1": [
                {"pdno": "005930", "hldg_qty": "10", "pchs_avg_pric": "75000"}
            ],
            "output2": [{"tot_evlu_amt": "750000"}],
        },
        executions={"output1": []},
        account_no="12345678",
        db_path=broker_db,
    )
    order_state.reconcile_after_broker_sync(
        symbol="005930",
        account_no="12345678",
        state_db_path=state_db,
        broker_db_path=broker_db,
    )
    intent = order_state.begin_order_intent(
        _live_req(side="sell", quantity=5, client_order_id="partial-sell")
    )
    order_state.mark_order_submitted(intent["id"], {"output": {"ODNO": "12345"}}, db_path=state_db)

    broker_sync.record_kis_sync(
        balance={
            "output1": [
                {"pdno": "005930", "hldg_qty": "8", "pchs_avg_pric": "75000"}
            ],
            "output2": [{"tot_evlu_amt": "600000"}],
        },
        executions={"output1": []},
        account_no="12345678",
        db_path=broker_db,
    )
    state = order_state.reconcile_after_broker_sync(
        symbol="005930",
        account_no="12345678",
        state_db_path=state_db,
        broker_db_path=broker_db,
    )
    stored_intent = order_state.get_order_intent(intent["id"], db_path=state_db)

    assert state["state"] == "PARTIAL"
    assert state["current_quantity"] == 8
    assert stored_intent["status"] == "PARTIALLY_FILLED"
    assert stored_intent["filled_quantity"] == 2
    assert stored_intent["remaining_quantity"] == 3
