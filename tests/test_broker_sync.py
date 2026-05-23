from __future__ import annotations

import sqlite3

from app.brokers.kis_client import KisApiError
from app.trading import broker_sync_worker
from app.trading.broker_sync import record_kis_sync, sync_kis_account


def test_record_kis_sync_stores_positions_and_executions(tmp_path):
    db_path = tmp_path / "broker.sqlite3"
    balance = {
        "output1": [
            {
                "pdno": "005930",
                "prdt_name": "Samsung Electronics",
                "hldg_qty": "3",
                "pchs_avg_pric": "75000",
                "prpr": "76000",
                "evlu_amt": "228000",
                "evlu_pfls_amt": "3000",
            }
        ],
        "output2": [
            {
                "dnca_tot_amt": "1000000",
                "tot_evlu_amt": "1228000",
            }
        ],
    }
    executions = {
        "output1": [
            {
                "odno": "12345",
                "pdno": "005930",
                "sll_buy_dvsn_cd_name": "BUY",
                "ord_qty": "3",
                "tot_ccld_qty": "3",
                "rmn_qty": "0",
                "ord_unpr": "75000",
                "avg_prvs": "75000",
                "ord_tmd": "093000",
            }
        ]
    }

    result = record_kis_sync(
        balance=balance,
        executions=executions,
        account_no="12345678",
        db_path=db_path,
    )

    assert result["status"] == "success"
    assert result["position_count"] == 1
    assert result["execution_count"] == 1
    assert result["total_cash"] == 1_000_000

    with sqlite3.connect(db_path) as conn:
        position = conn.execute(
            "SELECT symbol, quantity, avg_price FROM broker_positions"
        ).fetchone()
        execution = conn.execute(
            "SELECT order_no, symbol, status FROM broker_order_executions"
        ).fetchone()

    assert position == ("005930", 3, 75000)
    assert execution == ("12345", "005930", "FILLED")


def test_record_kis_sync_closes_positions_missing_from_latest_balance(tmp_path):
    db_path = tmp_path / "broker.sqlite3"
    first_balance = {
        "output1": [
            {
                "pdno": "005930",
                "prdt_name": "Samsung Electronics",
                "hldg_qty": "3",
                "pchs_avg_pric": "75000",
                "prpr": "76000",
            },
            {
                "pdno": "000660",
                "prdt_name": "SK Hynix",
                "hldg_qty": "2",
                "pchs_avg_pric": "120000",
                "prpr": "121000",
            },
        ],
        "output2": [{"dnca_tot_amt": "1000000", "tot_evlu_amt": "1468000"}],
    }
    second_balance = {
        "output1": [
            {
                "pdno": "005930",
                "prdt_name": "Samsung Electronics",
                "hldg_qty": "3",
                "pchs_avg_pric": "75000",
                "prpr": "76000",
            }
        ],
        "output2": [{"dnca_tot_amt": "1000000", "tot_evlu_amt": "1228000"}],
    }

    record_kis_sync(
        balance=first_balance,
        executions={"output1": []},
        account_no="12345678",
        db_path=db_path,
        complete_snapshot=True,
    )
    result = record_kis_sync(
        balance=second_balance,
        executions={"output1": []},
        account_no="12345678",
        db_path=db_path,
        complete_snapshot=True,
    )

    assert result["closed_position_count"] == 1

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT symbol, quantity
            FROM broker_positions
            ORDER BY symbol
            """
        ).fetchall()

    assert rows == [("000660", 0), ("005930", 3)]


def test_record_kis_sync_does_not_close_missing_positions_without_complete_snapshot(tmp_path):
    db_path = tmp_path / "broker.sqlite3"
    first_balance = {
        "output1": [
            {
                "pdno": "005930",
                "hldg_qty": "3",
                "pchs_avg_pric": "75000",
            },
            {
                "pdno": "000660",
                "hldg_qty": "2",
                "pchs_avg_pric": "120000",
            },
        ],
    }
    partial_balance = {
        "output1": [
            {
                "pdno": "005930",
                "hldg_qty": "3",
                "pchs_avg_pric": "75000",
            }
        ],
    }

    record_kis_sync(
        balance=first_balance,
        executions={"output1": []},
        account_no="12345678",
        db_path=db_path,
        complete_snapshot=True,
    )
    result = record_kis_sync(
        balance=partial_balance,
        executions={"output1": []},
        account_no="12345678",
        db_path=db_path,
    )

    assert result["closed_position_count"] == 0
    assert result["complete_snapshot"] is False

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT symbol, quantity
            FROM broker_positions
            ORDER BY symbol
            """
        ).fetchall()

    assert rows == [("000660", 2), ("005930", 3)]


def test_sync_kis_account_returns_config_error_for_invalid_kis_account():
    class FakeClient:
        account_no = "50189471"
        account_product_code = "01"
        is_paper = True

        def get_balance(self):
            raise KisApiError(
                "KIS API error OPSQ2000: ERROR : INPUT INVALID_CHECK_ACNO",
                error_code="OPSQ2000",
                error_description="ERROR : INPUT INVALID_CHECK_ACNO",
            )

    result = sync_kis_account(client=FakeClient())

    assert result["status"] == "config_error"
    assert result["kis_error_code"] == "OPSQ2000"
    assert result["account_no_last4"] == "9471"
    assert "paper/live KIS app key" in result["message"]


def test_broker_sync_once_does_not_reconcile_when_sync_has_config_error(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        broker_sync_worker.broker_sync,
        "sync_kis_account",
        lambda: {"status": "config_error", "account_no": "50189471"},
    )
    monkeypatch.setattr(
        broker_sync_worker.order_state,
        "reconcile_all_after_broker_sync",
        lambda account_no: calls.append(account_no),
    )

    result = broker_sync_worker.run_broker_sync_once()

    assert result["status"] == "config_error"
    assert result["order_state_reconcile"] is None
    assert calls == []
