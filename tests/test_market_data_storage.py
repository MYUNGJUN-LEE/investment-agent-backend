from __future__ import annotations

import sqlite3

from app.storage import market_data


def test_record_market_data_snapshots(tmp_path, monkeypatch):
    db_path = tmp_path / "market.sqlite3"
    monkeypatch.setattr(market_data, "_now", lambda: "2026-05-21T10:00:00")

    market_data.record_price_snapshot(
        {
            "symbol": "005930",
            "current_price": 75000,
            "change_rate": 1.2,
            "volume": 1000000,
            "volume_ratio": 2.0,
            "turnover_value": 75000000000,
            "trend": "uptrend",
            "intraday": {
                "minute_momentum_pct": 0.8,
                "execution_strength": 130,
                "orderbook_imbalance": 0.2,
                "spread_pct": 0.04,
                "smart_money_net_buy": 1500,
                "smart_money_net_buy_5d": 7500,
                "intraday_score": 82,
            },
        },
        db_path=db_path,
    )
    market_data.record_news_events(
        "삼성전자",
        [
            {
                "source": "Naver Search API",
                "date": "2026-05-21T09:00:00+00:00",
                "title": "AI 공급 계약",
                "url": "https://example.com",
                "impact_direction": "positive",
                "impact_strength": 65,
            }
        ],
        db_path=db_path,
    )
    market_data.record_financial_snapshot(
        "005930",
        {
            "business_year": "2025",
            "metrics": {
                "revenue": 1000,
                "operating_income": 150,
                "net_income": 100,
                "total_assets": 2000,
                "total_liabilities": 800,
                "total_equity": 1200,
                "operating_margin": 15,
                "net_margin": 10,
                "roe": 8.33,
                "debt_ratio": 66.67,
            },
        },
        db_path=db_path,
    )

    with sqlite3.connect(db_path) as conn:
        price_count = conn.execute("SELECT COUNT(*) FROM price_snapshots").fetchone()[0]
        intraday_row = conn.execute(
            """
            SELECT execution_strength, orderbook_imbalance, intraday_score
            FROM price_snapshots
            """
        ).fetchone()
        news_count = conn.execute("SELECT COUNT(*) FROM news_events").fetchone()[0]
        financial_count = conn.execute(
            "SELECT COUNT(*) FROM financial_snapshots"
        ).fetchone()[0]

    assert price_count == 1
    assert intraday_row == (130, 0.2, 82)
    assert news_count == 1
    assert financial_count == 1
