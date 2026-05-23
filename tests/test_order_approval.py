from __future__ import annotations

import sqlite3

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.trading import order_approval, paper_trading


client = TestClient(app)


def _auth_headers() -> dict[str, str]:
    if settings.backend_api_key:
        return {"X-API-Key": settings.backend_api_key}
    return {}


def _pipeline_result(
    entry_signal: bool = True,
    exit_signal: bool = False,
    final_grade: str = "공격",
    final_score: float = 82,
    risk_score: float = 35,
    confidence: float = 0.82,
) -> dict:
    return {
        "symbol": "005930",
        "name": "삼성전자",
        "market": "KR",
        "strategy_type": "daytrade",
        "final_grade": final_grade,
        "entry_signal": entry_signal,
        "exit_signal": exit_signal,
        "confidence": confidence,
        "summary": "mock pipeline summary",
        "scores": {
            "final_score": final_score,
            "risk_score": risk_score,
        },
        "entry_conditions": [],
        "avoid_conditions": [],
        "stop_loss_candidates": [],
        "take_profit_candidates": [],
        "time_exit_rule": "mock",
        "research_result": {},
        "financial_result": {},
        "chart_flow_result": {},
        "devils_advocate_result": {},
    }


def test_order_preview_blocks_when_strategy_has_no_entry_signal(tmp_path, monkeypatch):
    db_path = tmp_path / "paper.sqlite3"
    monkeypatch.setattr(paper_trading, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr(order_approval, "_now", lambda: "2026-05-20T10:00:00")
    monkeypatch.setattr(
        order_approval,
        "run_full_pipeline",
        lambda req: _pipeline_result(entry_signal=False, final_grade="관망", final_score=55),
    )

    response = client.post(
        "/orders/preview",
        headers=_auth_headers(),
        json={
            "symbol": "005930",
            "name": "삼성전자",
            "market": "KR",
            "strategy_type": "daytrade",
            "requested_action": "auto",
            "price": 75000,
            "quantity": 3,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "blocked"
    assert body["preview_token"] is None
    assert body["strategy_decision"]["action"] == "hold"
    assert "Pipeline entry_signal is false" in body["message"]


def test_order_preview_and_confirm_execute_paper_order(tmp_path, monkeypatch):
    db_path = tmp_path / "paper.sqlite3"
    monkeypatch.setattr(paper_trading, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr(order_approval, "_now", lambda: "2026-05-20T10:00:00")
    monkeypatch.setattr(paper_trading, "_now", lambda: "2026-05-20T10:00:01")
    monkeypatch.setattr(order_approval, "run_full_pipeline", lambda req: _pipeline_result())

    preview_response = client.post(
        "/orders/preview",
        headers=_auth_headers(),
        json={
            "symbol": "005930",
            "name": "삼성전자",
            "market": "KR",
            "strategy_type": "daytrade",
            "requested_action": "auto",
            "price": 75000,
            "quantity": 3,
        },
    )

    assert preview_response.status_code == 200
    preview = preview_response.json()
    assert preview["status"] == "pending"
    assert preview["preview_token"]
    assert preview["signal_type"] == "entry"
    assert preview["side"] == "BUY"
    assert preview["risk_decision"]["approved"] is True

    confirm_response = client.post(
        "/orders/confirm",
        headers=_auth_headers(),
        json={
            "preview_id": preview["preview_id"],
            "preview_token": preview["preview_token"],
            "execution_mode": "paper",
        },
    )

    assert confirm_response.status_code == 200
    body = confirm_response.json()
    assert body["status"] == "confirmed"
    assert body["execution_mode"] == "paper"
    assert body["paper_result"]["order_status"] == "FILLED"
    assert body["paper_result"]["position"]["quantity"] == 3

    with sqlite3.connect(db_path) as conn:
        preview_status = conn.execute(
            "SELECT status FROM order_previews WHERE id = ?",
            (preview["preview_id"],),
        ).fetchone()[0]
        position = conn.execute(
            "SELECT quantity, avg_price FROM positions WHERE symbol = ?",
            ("005930",),
        ).fetchone()

    assert preview_status == "confirmed"
    assert position == (3, 75011.25)


def test_order_preview_blocks_when_expected_edge_does_not_cover_costs(tmp_path, monkeypatch):
    db_path = tmp_path / "paper.sqlite3"
    monkeypatch.setattr(paper_trading, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr(order_approval, "_now", lambda: "2026-05-20T10:00:00")
    monkeypatch.setattr(order_approval, "run_full_pipeline", lambda req: _pipeline_result())

    response = client.post(
        "/orders/preview",
        headers=_auth_headers(),
        json={
            "symbol": "005930",
            "name": "삼성전자",
            "market": "KR",
            "strategy_type": "daytrade",
            "requested_action": "auto",
            "price": 75000,
            "quantity": 3,
            "expected_gross_edge_bps": 10,
            "expected_win_bps": 20,
            "expected_loss_bps": 30,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "blocked"
    assert body["risk_decision"]["code"] == "edge_requirement_not_met"
    assert body["cost_edge_decision"]["approved"] is False


def test_order_confirm_rejects_wrong_preview_token(tmp_path, monkeypatch):
    db_path = tmp_path / "paper.sqlite3"
    monkeypatch.setattr(paper_trading, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr(order_approval, "_now", lambda: "2026-05-20T10:00:00")
    monkeypatch.setattr(order_approval, "run_full_pipeline", lambda req: _pipeline_result())

    preview_response = client.post(
        "/orders/preview",
        headers=_auth_headers(),
        json={
            "symbol": "005930",
            "market": "KR",
            "strategy_type": "daytrade",
            "requested_action": "auto",
            "price": 75000,
            "quantity": 3,
        },
    )
    assert preview_response.status_code == 200
    preview = preview_response.json()

    confirm_response = client.post(
        "/orders/confirm",
        headers=_auth_headers(),
        json={
            "preview_id": preview["preview_id"],
            "preview_token": "wrong-token",
            "execution_mode": "paper",
        },
    )

    assert confirm_response.status_code == 403
    assert confirm_response.json()["detail"] == "Invalid preview_token"
