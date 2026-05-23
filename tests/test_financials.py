from __future__ import annotations

from app.data_sources.financials import (
    calculate_growth_metrics,
    parse_financial_metrics,
)
from app.scoring import score_financial_data


def test_parse_financial_metrics_extracts_core_accounts():
    data = {
        "list": [
            {"account_nm": "매출액", "thstrm_amount": "1,000"},
            {"account_nm": "영업이익", "thstrm_amount": "150"},
            {"account_nm": "당기순이익", "thstrm_amount": "100"},
            {"account_nm": "자산총계", "thstrm_amount": "2,000"},
            {"account_nm": "부채총계", "thstrm_amount": "800"},
            {"account_nm": "자본총계", "thstrm_amount": "1,200"},
        ]
    }

    metrics = parse_financial_metrics(data)

    assert metrics["revenue"] == 1000
    assert metrics["operating_income"] == 150
    assert metrics["net_income"] == 100
    assert metrics["operating_margin"] == 15
    assert metrics["net_margin"] == 10
    assert metrics["roe"] == 8.33
    assert metrics["debt_ratio"] == 66.67


def test_calculate_growth_metrics_and_financial_score():
    current = {
        "revenue": 1200,
        "operating_income": 180,
        "net_income": 120,
        "operating_margin": 15,
        "net_margin": 10,
        "roe": 15,
        "debt_ratio": 80,
    }
    previous = {
        "revenue": 1000,
        "operating_income": 100,
        "net_income": 80,
    }
    growth = calculate_growth_metrics(current, previous)

    assert growth["revenue_growth"] == 20
    assert growth["operating_income_growth"] == 80

    score = score_financial_data(
        {
            "status": "connected",
            "metrics": current,
            "growth_metrics": growth,
        }
    )

    assert score >= 80
