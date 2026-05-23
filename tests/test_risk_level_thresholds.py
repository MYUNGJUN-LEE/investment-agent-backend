from __future__ import annotations

from app.agents import run_chart_flow_agent, run_final_check_agent
from app.models import PipelineRequest
from app.strategies.rule_based import build_strategy_decision


def test_high_risk_strategy_uses_less_conservative_thresholds():
    pipeline_result = {
        "risk_level": "high",
        "final_grade": "공격",
        "entry_signal": True,
        "exit_signal": False,
        "confidence": 0.65,
        "scores": {
            "final_score": 72,
            "risk_score": 58,
        },
    }

    medium_decision = build_strategy_decision(
        pipeline_result={**pipeline_result, "risk_level": "medium"},
        requested_action="entry",
    )
    high_decision = build_strategy_decision(
        pipeline_result=pipeline_result,
        requested_action="entry",
    )

    assert medium_decision["approved"] is False
    assert high_decision["approved"] is True
    assert high_decision["thresholds"]["min_final_score"] == 70
    assert high_decision["thresholds"]["max_risk_score"] == 60


def test_final_check_thresholds_follow_request_risk_level():
    research_result = {"research_score": 80}
    financial_result = {"financial_score": 70, "data_needed": []}
    chart_flow_result = {
        "chart_flow_score": 80,
        "entry_signal": True,
        "exit_signal": False,
        "entry_conditions": [],
        "avoid_conditions": [],
        "stop_loss_candidates": [],
        "take_profit_candidates": [],
        "data_needed": [],
        "price_data": {"status": "connected"},
    }
    devils_result = {
        "risk_score": 58,
        "do_not_enter_conditions": [],
        "counterarguments": [],
    }

    low_req = PipelineRequest(
        symbol="005930",
        market="KR",
        strategy_type="daytrade",
        risk_level="low",
    )
    high_req = PipelineRequest(
        symbol="005930",
        market="KR",
        strategy_type="daytrade",
        risk_level="high",
    )

    low_result = run_final_check_agent(
        low_req,
        research_result,
        financial_result,
        chart_flow_result,
        devils_result,
    )
    high_result = run_final_check_agent(
        high_req,
        research_result,
        financial_result,
        chart_flow_result,
        devils_result,
    )

    assert low_result["entry_signal"] is False
    assert high_result["entry_signal"] is True


def test_chart_flow_uses_less_conservative_intraday_profile_for_high_risk(monkeypatch):
    monkeypatch.setattr(
        "app.agents.fetch_price_data",
        lambda symbol: {
            "status": "connected",
            "symbol": symbol,
            "change_rate": 0.3,
            "volume_ratio": 1.3,
            "trend": "sideways",
            "overheated": False,
            "support_levels": [],
            "resistance_levels": [],
            "price_position": {
                "above_ma5": False,
                "above_ma20": True,
            },
            "intraday": {
                "intraday_score": 57,
                "execution_strength": 95,
                "orderbook_imbalance": -0.1,
                "spread_pct": 0.5,
                "above_minute_vwap": True,
                "minute_volume_ratio": 1.8,
                "smart_money_net_buy_5d": 3000,
            },
            "optional_errors": [],
        },
    )
    research_result = {"research_score": 70}
    financial_result = {"financial_score": 70, "data_needed": []}

    low_req = PipelineRequest(
        symbol="005930",
        market="KR",
        strategy_type="daytrade",
        risk_level="low",
    )
    high_req = PipelineRequest(
        symbol="005930",
        market="KR",
        strategy_type="daytrade",
        risk_level="high",
    )

    low_result = run_chart_flow_agent(low_req, research_result, financial_result)
    high_result = run_chart_flow_agent(high_req, research_result, financial_result)

    assert low_result["entry_signal"] is False
    assert high_result["entry_signal"] is True
