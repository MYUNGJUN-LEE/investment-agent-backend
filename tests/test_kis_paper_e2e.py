from __future__ import annotations

import os
import sqlite3

import pytest

from app.brokers.kis_client import KisApiError
from app.trading.kis_paper_e2e import preflight_kis_paper_e2e, run_kis_paper_order_e2e


class FakeKisPaperClient:
    is_paper = True
    account_no = "12345678"
    placed_order = False

    def get_current_price(self, symbol):
        return {"rt_cd": "0", "output": {"stck_prpr": "75000"}}

    def get_balance(self):
        return {
            "output1": [
                {
                    "pdno": "005930",
                    "prdt_name": "Samsung Electronics",
                    "hldg_qty": "1",
                    "pchs_avg_pric": "75000",
                    "prpr": "75000",
                }
            ],
            "output2": [{"tot_evlu_amt": "75000"}],
        }

    def get_daily_order_executions(self, start_date, end_date, symbol="", **kwargs):
        return {
            "output1": [
                {
                    "odno": "99999",
                    "pdno": "005930",
                    "ord_qty": "1",
                    "tot_ccld_qty": "1",
                    "rmn_qty": "0",
                    "ord_unpr": "75000",
                }
            ]
        }

    def place_domestic_limit_order(self, symbol, side, price, quantity):
        self.placed_order = True
        return {
            "rt_cd": "0",
            "output": {"ODNO": "99999"},
        }


class FakeKisPaperClientWithStaleToken(FakeKisPaperClient):
    refresh_count = 0
    balance_attempts = 0

    def get_balance(self):
        self.balance_attempts += 1
        if self.balance_attempts == 1:
            raise KisApiError(
                "KIS API error OPSQ2000: ERROR : INPUT INVALID_CHECK_ACNO",
                error_code="OPSQ2000",
                error_description="ERROR : INPUT INVALID_CHECK_ACNO",
            )
        return super().get_balance()

    def issue_access_token(self, force_refresh=False):
        if force_refresh:
            self.refresh_count += 1
        return "fresh-token"


def test_kis_paper_e2e_records_filled_order_with_fake_client(tmp_path):
    db_path = tmp_path / "broker.sqlite3"

    result = run_kis_paper_order_e2e(
        client=FakeKisPaperClient(),
        db_path=db_path,
        poll_seconds=0,
        timeout_seconds=0,
        require_fill=True,
    )

    assert result["status"] == "filled"
    assert result["order_no"] == "99999"
    with sqlite3.connect(db_path) as conn:
        execution = conn.execute(
            "SELECT order_no, symbol, status FROM broker_order_executions"
        ).fetchone()
    assert execution == ("99999", "005930", "FILLED")


def test_kis_paper_preflight_does_not_place_order(tmp_path):
    fake_client = FakeKisPaperClient()

    result = preflight_kis_paper_e2e(
        client=fake_client,
        db_path=tmp_path / "broker.sqlite3",
    )

    assert result["status"] == "ready"
    assert result["current_price"] == 75000
    assert fake_client.placed_order is False


def test_kis_paper_preflight_refreshes_token_once_on_invalid_account(tmp_path):
    fake_client = FakeKisPaperClientWithStaleToken()

    result = preflight_kis_paper_e2e(
        client=fake_client,
        db_path=tmp_path / "broker.sqlite3",
    )

    assert result["status"] == "ready"
    assert fake_client.refresh_count == 1
    assert fake_client.balance_attempts == 2


def test_real_kis_paper_e2e_is_opt_in(tmp_path):
    if os.getenv("RUN_KIS_PAPER_E2E") != "1":
        pytest.skip("Set RUN_KIS_PAPER_E2E=1 with KIS paper credentials to run")

    result = run_kis_paper_order_e2e(
        symbol=os.getenv("KIS_E2E_SYMBOL", "005930"),
        side=os.getenv("KIS_E2E_SIDE", "buy"),
        quantity=int(os.getenv("KIS_E2E_QUANTITY", "1")),
        price=float(os.getenv("KIS_E2E_PRICE")) if os.getenv("KIS_E2E_PRICE") else None,
        poll_seconds=float(os.getenv("KIS_E2E_POLL_SECONDS", "5")),
        timeout_seconds=float(os.getenv("KIS_E2E_TIMEOUT_SECONDS", "60")),
        require_fill=os.getenv("KIS_E2E_REQUIRE_FILL", "0") == "1",
        db_path=tmp_path / "broker.sqlite3",
    )

    assert result["status"] in {"filled", "submitted_not_filled"}
