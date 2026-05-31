from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.models import PaperRunRequest
from app.trading import cost_model
from app.trading import order_approval, paper_trading
from app.trading import risk_manager


client = TestClient(app)


def _auth_headers() -> dict[str, str]:
    if settings.backend_api_key:
        return {"X-API-Key": settings.backend_api_key}
    return {}


def _pipeline_result() -> dict:
    return {
        "symbol": "005930",
        "name": "Samsung Electronics",
        "market": "KR",
        "strategy_type": "daytrade",
        "final_grade": "\uacf5\uaca9",
        "entry_signal": True,
        "exit_signal": False,
        "confidence": 0.82,
        "summary": "mock pipeline summary",
        "scores": {
            "final_score": 82,
            "risk_score": 35,
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


def test_order_preview_recommends_quantity_when_quantity_is_omitted(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(paper_trading, "DEFAULT_DB_PATH", tmp_path / "paper.sqlite3")
    monkeypatch.setattr(order_approval, "_now", lambda: "2026-05-20T10:00:00")
    monkeypatch.setattr(order_approval, "run_full_pipeline", lambda req: _pipeline_result())

    recommendation = risk_manager.recommend_order_quantity(
        price=10000,
        stop_loss=9500,
        account_equity=1_000_000,
        risk_per_trade=0.01,
    )
    edge_probe = cost_model.evaluate_entry_edge(
        PaperRunRequest(
            symbol="005930",
            market="KR",
            strategy_type="daytrade",
            signal_type="entry",
            price=10000,
            quantity=recommendation["recommended_quantity"],
            expected_gross_edge_bps=0,
            expected_win_bps=300,
            expected_loss_bps=50,
        ),
        quantity=recommendation["recommended_quantity"],
    )
    required_gross_edge_bps = round(
        (
            edge_probe["required_net_edge_bps"]
            + edge_probe["round_trip_cost"]["total_cost_bps"]
        )
        / settings.default_fill_probability
        + 1.0,
        4,
    )

    response = client.post(
        "/orders/preview",
        headers=_auth_headers(),
        json={
            "symbol": "005930",
            "market": "KR",
            "strategy_type": "daytrade",
            "requested_action": "entry",
            "price": 10000,
            "account_equity": 1_000_000,
            "risk_per_trade": 0.01,
            "stop_loss": 9500,
            "expected_gross_edge_bps": required_gross_edge_bps,
            "expected_win_bps": 300,
            "expected_loss_bps": 50,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pending"
    assert body["quantity"] == recommendation["recommended_quantity"]
    assert body["amount"] == recommendation["estimated_amount"]
    assert body["recommended_quantity"]["recommended_quantity"] == recommendation["recommended_quantity"]
    assert body["recommended_quantity"]["symbol_weight_limit"] == recommendation["symbol_weight_limit"]
    assert body["recommended_quantity"]["used_recommended_quantity"] is True
    assert body["strategy_decision"]["recommended_quantity"]["formula"] == (
        "min(account risk quantity, cash quantity, max order quantity, symbol weight quantity)"
    )
    edge_decision = body["risk_decision"]["checks"]["edge"]
    assert edge_decision["approved"] is True
    assert edge_decision["round_trip_cost"]["total_cost_bps"] == edge_probe["round_trip_cost"]["total_cost_bps"]
    assert edge_decision["round_trip_cost"]["sell"]["tax"] == round(
        10000
        * recommendation["recommended_quantity"]
        * settings.kr_stock_sell_tax_rate,
        6,
    )


def test_order_preview_requires_quantity_or_sizing_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(paper_trading, "DEFAULT_DB_PATH", tmp_path / "paper.sqlite3")

    response = client.post(
        "/orders/preview",
        headers=_auth_headers(),
        json={
            "symbol": "005930",
            "market": "KR",
            "strategy_type": "daytrade",
            "requested_action": "entry",
            "price": 10000,
        },
    )

    assert response.status_code == 400
    assert "quantity is required" in response.json()["detail"]
