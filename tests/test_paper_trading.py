from __future__ import annotations

import sqlite3

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.trading import paper_trading
from app.trading import risk_manager


client = TestClient(app)


def _auth_headers() -> dict[str, str]:
    if settings.backend_api_key:
        return {"X-API-Key": settings.backend_api_key}
    return {}


def test_paper_run_once_entry_records_signal_order_and_position(tmp_path, monkeypatch):
    db_path = tmp_path / "paper.sqlite3"
    monkeypatch.setattr(paper_trading, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr(paper_trading, "_now", lambda: "2026-05-20T10:00:00")

    response = client.post(
        "/paper/run-once",
        headers=_auth_headers(),
        json={
            "symbol": "005930",
            "name": "삼성전자",
            "market": "KR",
            "strategy_type": "daytrade",
            "signal_type": "entry",
            "price": 75000,
            "quantity": 3,
            "confidence": 0.82,
            "reason": "VWAP 돌파",
            "source": "pytest",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["order_status"] == "FILLED"
    assert body["side"] == "BUY"
    assert body["quantity"] == 3
    assert body["amount"] == 225000
    assert body["total_cost"] == 33.75
    assert body["position"]["quantity"] == 3
    assert body["position"]["avg_price"] == 75011.25
    assert body["performance_metrics"]["turnover"] > 0

    with sqlite3.connect(db_path) as conn:
        signal_count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        order_count = conn.execute("SELECT COUNT(*) FROM paper_orders").fetchone()[0]
        position = conn.execute(
            "SELECT quantity, avg_price, cost_basis FROM positions WHERE symbol = ?",
            ("005930",),
        ).fetchone()
        performance_count = conn.execute(
            "SELECT COUNT(*) FROM performance_snapshots"
        ).fetchone()[0]

    assert signal_count == 1
    assert order_count == 1
    assert position == (3, 75011.25, 225033.75)
    assert performance_count == 1


def test_paper_run_once_exit_closes_position_and_records_pnl(tmp_path, monkeypatch):
    db_path = tmp_path / "paper.sqlite3"
    monkeypatch.setattr(paper_trading, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr(paper_trading, "_now", lambda: "2026-05-20T10:00:00")

    entry_response = client.post(
        "/paper/run-once",
        headers=_auth_headers(),
        json={
            "symbol": "005930",
            "name": "삼성전자",
            "market": "KR",
            "strategy_type": "daytrade",
            "signal_type": "entry",
            "price": 75000,
            "quantity": 2,
            "confidence": 0.8,
        },
    )
    assert entry_response.status_code == 200

    exit_response = client.post(
        "/paper/run-once",
        headers=_auth_headers(),
        json={
            "symbol": "005930",
            "name": "삼성전자",
            "market": "KR",
            "strategy_type": "daytrade",
            "signal_type": "exit",
            "price": 76000,
            "confidence": 0.75,
            "reason": "익절",
        },
    )

    assert exit_response.status_code == 200
    body = exit_response.json()
    assert body["status"] == "success"
    assert body["side"] == "SELL"
    assert body["quantity"] == 2
    assert body["amount"] == 152000
    assert body["total_cost"] == 326.8
    assert body["position"]["quantity"] == 0
    assert body["position"]["avg_price"] == 0
    assert body["position"]["realized_pnl"] == 1650.7

    with sqlite3.connect(db_path) as conn:
        signal_count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        order_count = conn.execute("SELECT COUNT(*) FROM paper_orders").fetchone()[0]
        performance_count = conn.execute(
            "SELECT COUNT(*) FROM performance_snapshots"
        ).fetchone()[0]

    assert signal_count == 2
    assert order_count == 2
    assert performance_count == 2

    with sqlite3.connect(db_path) as conn:
        realized_pnl = conn.execute(
            "SELECT realized_pnl FROM paper_orders WHERE side = 'SELL'"
        ).fetchone()[0]

    assert realized_pnl == 1650.7


def test_paper_run_once_exit_without_position_is_rejected(tmp_path, monkeypatch):
    db_path = tmp_path / "paper.sqlite3"
    monkeypatch.setattr(paper_trading, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr(paper_trading, "_now", lambda: "2026-05-20T10:00:00")

    response = client.post(
        "/paper/run-once",
        headers=_auth_headers(),
        json={
            "symbol": "000660",
            "name": "SK하이닉스",
            "market": "KR",
            "strategy_type": "daytrade",
            "signal_type": "exit",
            "price": 250000,
            "confidence": 0.7,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "rejected"
    assert body["order_status"] == "REJECTED"
    assert body["side"] == "SELL"
    assert body["quantity"] == 0
    assert body["position"] is None


def test_paper_run_once_rejects_order_amount_limit(tmp_path, monkeypatch):
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
            "price": 75000,
            "quantity": 20,
            "confidence": 0.8,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "rejected"
    assert "max_order_amount_exceeded" in body["message"]
    assert body["position"] is None


def test_paper_run_once_rejects_symbol_weight_limit(tmp_path, monkeypatch):
    db_path = tmp_path / "paper.sqlite3"
    monkeypatch.setattr(paper_trading, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr(paper_trading, "_now", lambda: "2026-05-20T10:00:00")
    monkeypatch.setattr(
        risk_manager,
        "DEFAULT_LIMITS",
        risk_manager.RiskLimits(
            max_order_amount=10_000_000,
            portfolio_value=1_000_000,
            max_symbol_weight=0.5,
        ),
    )

    response = client.post(
        "/paper/run-once",
        headers=_auth_headers(),
        json={
            "symbol": "005930",
            "market": "KR",
            "strategy_type": "daytrade",
            "signal_type": "entry",
            "price": 200000,
            "quantity": 3,
            "confidence": 0.8,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "rejected"
    assert "symbol_weight_limit_exceeded" in body["message"]


def test_paper_run_once_rejects_daily_loss_limit(tmp_path, monkeypatch):
    db_path = tmp_path / "paper.sqlite3"
    monkeypatch.setattr(paper_trading, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr(paper_trading, "_now", lambda: "2026-05-20T10:00:00")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        paper_trading.initialize_db(conn)
        conn.execute(
            """
            INSERT INTO paper_orders (
                signal_id, created_at, symbol, side, price, quantity,
                amount, realized_pnl, status, message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (1, "2026-05-20T09:30:00", "005930", "SELL", 1000, 1, 1000, -400, "FILLED", "loss"),
        )
        conn.commit()

    monkeypatch.setattr(
        risk_manager,
        "DEFAULT_LIMITS",
        risk_manager.RiskLimits(max_daily_loss_amount=300),
    )

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
            "confidence": 0.8,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "rejected"
    assert "daily_loss_limit_exceeded" in body["message"]


def test_paper_run_once_rejects_consecutive_stop_losses(tmp_path, monkeypatch):
    db_path = tmp_path / "paper.sqlite3"
    monkeypatch.setattr(paper_trading, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr(paper_trading, "_now", lambda: "2026-05-20T10:00:00")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        paper_trading.initialize_db(conn)
        for idx in range(3):
            conn.execute(
                """
                INSERT INTO paper_orders (
                    signal_id, created_at, symbol, side, price, quantity,
                    amount, realized_pnl, status, message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (idx + 1, "2026-05-20T09:30:00", "005930", "SELL", 1000, 1, 1000, -100, "FILLED", "loss"),
            )
        conn.commit()

    monkeypatch.setattr(
        risk_manager,
        "DEFAULT_LIMITS",
        risk_manager.RiskLimits(max_daily_loss_amount=10_000),
    )

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
            "confidence": 0.8,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "rejected"
    assert "consecutive_stop_loss_limit_reached" in body["message"]


def test_paper_run_once_rejects_new_entry_outside_entry_windows(tmp_path, monkeypatch):
    db_path = tmp_path / "paper.sqlite3"
    monkeypatch.setattr(paper_trading, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr(paper_trading, "_now", lambda: "2026-05-20T13:00:00")

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
            "confidence": 0.8,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "rejected"
    assert "entry_time_window_blocked" in body["message"]


def test_paper_run_once_allows_new_entry_at_afternoon_window_end(tmp_path, monkeypatch):
    db_path = tmp_path / "paper.sqlite3"
    monkeypatch.setattr(paper_trading, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr(paper_trading, "_now", lambda: "2026-05-20T15:20:00")

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
            "confidence": 0.8,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"


def test_paper_run_once_must_pass_approve_order(tmp_path, monkeypatch):
    db_path = tmp_path / "paper.sqlite3"
    monkeypatch.setattr(paper_trading, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr(paper_trading, "_now", lambda: "2026-05-20T10:00:00")

    def reject_all_orders(*args, **kwargs):
        return risk_manager.RiskDecision(
            approved=False,
            code="forced_rejection",
            message="forced rejection for test",
            checks={},
        )

    monkeypatch.setattr(paper_trading.risk_manager, "approve_order", reject_all_orders)

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
            "confidence": 0.8,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "rejected"
    assert "forced_rejection" in body["message"]

    with sqlite3.connect(db_path) as conn:
        position_count = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]

    assert position_count == 0
