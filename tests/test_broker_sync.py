from __future__ import annotations

import sqlite3

from app.brokers.kis_client import KisApiError
from app.trading import broker_sync_worker
from app.trading.broker_sync import (
    broker_account_check_from_sync_result,
    record_kis_sync,
    sync_kis_account,
)


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


def test_sync_kis_account_returns_token_backoff_for_egw00133():
    class FakeClient:
        account_no = "50189471"
        account_product_code = "01"
        is_paper = True

        def get_balance(self):
            raise KisApiError(
                "KIS HTTP error: 403 EGW00133",
                status_code=403,
                error_code="EGW00133",
                error_description="token issuance limited",
            )

        def token_status(self):
            return {
                "kis_token_cached": False,
                "kis_token_status": "backoff",
                "kis_token_refresh_blocked_by_rate_limit": True,
            }

    result = sync_kis_account(client=FakeClient())

    assert result["status"] == "token_backoff"
    assert result["message"] == "kis_token_rate_limited"
    assert result["recoverable"] is True
    assert result["kis_token"]["kis_token_status"] == "backoff"


def test_sync_kis_account_returns_account_backoff_for_egw00201():
    class FakeClient:
        account_no = "50189471"
        account_product_code = "01"
        is_paper = True

        def get_balance(self):
            raise KisApiError(
                "KIS API error EGW00201: ledger TPS exceeded",
                status_code=403,
                error_code="EGW00201",
                error_description="ledger TPS exceeded",
            )

        def account_status(self):
            return {
                "kis_account_rate_limited": True,
                "kis_account_next_probe_allowed_at": "2026-06-11T09:01:10",
            }

    result = sync_kis_account(client=FakeClient())

    assert result["status"] == "account_backoff"
    assert result["message"] == "kis_account_rate_limited"
    assert result["recoverable"] is True
    assert result["kis_account"]["kis_account_rate_limited"] is True
    assert "config" not in result["status"]


def test_broker_account_check_marks_account_rate_limit():
    check = broker_account_check_from_sync_result(
        {"status": "account_backoff", "message": "kis_account_rate_limited"}
    )

    assert check["status"] == "rate_limited"
    assert check["rate_limited"] is True
    assert check["account_rate_limited"] is True
    assert check["block_reason"] == "account_rate_limited"


def test_broker_account_check_marks_total_cash_zero():
    check = broker_account_check_from_sync_result(
        {
            "status": "success",
            "account_no": "50189471",
            "total_cash": 0,
            "total_value": 0,
            "raw_cash_fields": {"dnca_tot_amt": "0"},
            "synced_at": "2026-06-11T09:00:00",
        }
    )

    assert check["status"] == "blocked"
    assert check["connected"] is True
    assert check["block_reason"] == "total_cash_zero"


def test_broker_account_check_marks_cash_fields_missing():
    check = broker_account_check_from_sync_result(
        {
            "status": "success",
            "account_no": "50189471",
            "position_count": 0,
            "execution_count": 0,
            "raw_cash_fields": {},
            "synced_at": "2026-06-11T09:00:00",
        }
    )

    assert check["status"] == "blocked"
    assert check["block_reason"] == "cash_unavailable"


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


def test_broker_sync_worker_backs_off_on_token_backoff(monkeypatch):
    monkeypatch.setattr(
        broker_sync_worker.settings,
        "broker_sync_config_error_backoff_seconds",
        900,
    )

    assert broker_sync_worker._next_sleep_seconds(
        {"status": "token_backoff"},
        60,
    ) == 900
    assert broker_sync_worker._next_sleep_seconds(
        {"status": "account_backoff"},
        60,
    ) == 900
