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


def test_swing_chart_flow_uses_daily_profile_over_intraday_orderbook(monkeypatch):
    price_data = {
        "status": "connected",
        "symbol": "005930",
        "current_price": 70000,
        "change_rate": -0.8,
        "volume_ratio": 1.4,
        "trend": "sideways",
        "overheated": False,
        "support_levels": [],
        "resistance_levels": [],
        "price_position": {
            "above_ma5": False,
            "above_ma20": True,
        },
        "latest_technical_features": {
            "close": 70000,
            "return_5d": 0.03,
            "return_20d": 0.07,
            "return_60d": 0.16,
            "high_breakout_20d": True,
            "low_breakdown_20d": False,
            "ma20_slope": 300,
            "ma60": 64000,
            "atr_14": 1200,
            "atr_14_pct": 1200 / 70000,
        },
        "technical_features": [
            {"close": 69000, "atr_14": 1200},
            {"close": 71000, "atr_14": 1200},
        ],
        "intraday": {
            "intraday_score": 45,
            "execution_strength": 65,
            "orderbook_imbalance": -0.4,
            "spread_pct": 1.1,
            "above_minute_vwap": False,
            "minute_momentum_pct": -1.2,
        },
        "optional_errors": [],
    }
    market_context = {
        "status": "connected",
        "market_regime": "bull",
        "risk_on_score": 67,
        "selected_sector_relative_strength": {
            "sector": "semiconductor",
            "score": 72,
        },
    }
    monkeypatch.setattr("app.agents.fetch_price_data", lambda symbol: dict(price_data))
    monkeypatch.setattr(
        "app.agents.fetch_market_context",
        lambda **kwargs: dict(market_context),
    )

    research_result = {"research_score": 70}
    financial_result = {"financial_score": 70, "data_needed": []}
    daytrade_req = PipelineRequest(
        symbol="005930",
        market="KR",
        strategy_type="daytrade",
        risk_level="medium",
        sector="semiconductor",
    )
    swing_req = PipelineRequest(
        symbol="005930",
        market="KR",
        strategy_type="swing",
        risk_level="medium",
        sector="semiconductor",
    )

    daytrade_result = run_chart_flow_agent(daytrade_req, research_result, financial_result)
    swing_result = run_chart_flow_agent(swing_req, research_result, financial_result)

    assert daytrade_result["entry_signal"] is False
    assert swing_result["entry_signal"] is True
    assert "entry - 1.8 * ATR14" in swing_result["stop_loss_candidates"][0]
