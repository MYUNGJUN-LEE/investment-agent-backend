from __future__ import annotations

import sqlite3

from app.trading.broker_sync import record_kis_sync


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
