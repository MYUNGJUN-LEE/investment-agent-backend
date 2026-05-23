from __future__ import annotations

from app.scoring import classify_text_impact, score_chart_flow


def test_expanded_keyword_pool_detects_short_term_positive_catalysts():
    impact = classify_text_impact("HBM 장기공급 계약과 자사주 소각 발표")

    assert impact["direction"] == "positive"
    assert "HBM" in impact["matched_keywords"]
    assert "장기공급" in impact["matched_keywords"]
    assert "자사주 소각" in impact["matched_keywords"]


def test_expanded_keyword_pool_detects_material_negative_events():
    impact = classify_text_impact("유상증자와 관리종목 지정, 감사의견 거절 우려")

    assert impact["direction"] == "negative"
    assert "유상증자" in impact["matched_keywords"]
    assert "관리종목" in impact["matched_keywords"]
    assert "감사의견 거절" in impact["matched_keywords"]


def test_chart_flow_score_uses_intraday_execution_and_orderbook():
    base = {
        "status": "connected",
        "change_rate": 1.2,
        "volume_ratio": 1.8,
        "trend": "uptrend",
        "price_position": {"above_ma5": True, "above_ma20": True},
        "overheated": False,
    }
    strong_intraday = {
        **base,
        "intraday": {
            "intraday_score": 82,
            "execution_strength": 135,
            "orderbook_imbalance": 0.25,
            "spread_pct": 0.04,
            "above_minute_vwap": True,
            "minute_momentum_pct": 0.8,
            "smart_money_net_buy_5d": 5000,
        },
    }
    weak_intraday = {
        **base,
        "intraday": {
            "intraday_score": 35,
            "execution_strength": 70,
            "orderbook_imbalance": -0.3,
            "spread_pct": 0.9,
            "above_minute_vwap": False,
            "minute_momentum_pct": -1.4,
            "smart_money_net_buy_5d": -5000,
        },
    }

    assert score_chart_flow(strong_intraday) > score_chart_flow(weak_intraday)
