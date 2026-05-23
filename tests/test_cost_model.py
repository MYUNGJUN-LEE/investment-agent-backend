from __future__ import annotations

from app.models import PaperRunRequest
from app.trading.cost_model import estimate_order_cost, evaluate_entry_edge


def test_korean_sell_cost_includes_commission_tax_spread_and_slippage():
    req = PaperRunRequest(
        symbol="005930",
        market="KR",
        strategy_type="daytrade",
        signal_type="exit",
        price=100_000,
        quantity=10,
        commission_rate=0.00015,
        sell_tax_rate=0.002,
        spread_bps=2,
        slippage_bps=3,
    )

    cost = estimate_order_cost(req, side="SELL", quantity=10)

    assert cost.effective_price == 99950
    assert cost.commission == 149.925
    assert cost.tax == 1999.0
    assert cost.spread_cost == 200
    assert cost.slippage_cost == 300
    assert cost.total_cost == 2648.925


def test_edge_gate_rejects_trade_when_net_edge_is_too_small():
    req = PaperRunRequest(
        symbol="005930",
        market="KR",
        strategy_type="daytrade",
        risk_level="medium",
        signal_type="entry",
        price=100_000,
        quantity=10,
        confidence=0.9,
        expected_gross_edge_bps=20,
        expected_win_bps=50,
        expected_loss_bps=40,
    )

    decision = evaluate_entry_edge(req, quantity=10)

    assert decision["approved"] is False
    assert decision["code"] == "edge_requirement_not_met"
    assert decision["net_edge_bps"] < decision["required_net_edge_bps"]


def test_edge_gate_accepts_trade_with_cost_adjusted_edge_and_reward_risk():
    req = PaperRunRequest(
        symbol="005930",
        market="KR",
        strategy_type="daytrade",
        risk_level="medium",
        signal_type="entry",
        price=100_000,
        quantity=10,
        confidence=0.9,
        expected_gross_edge_bps=120,
        expected_win_bps=90,
        expected_loss_bps=50,
        expected_sharpe=1.2,
    )

    decision = evaluate_entry_edge(req, quantity=10)

    assert decision["approved"] is True
    assert decision["reward_risk_ratio"] == 1.8
