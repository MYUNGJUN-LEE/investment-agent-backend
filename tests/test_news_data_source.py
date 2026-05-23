from __future__ import annotations

from datetime import datetime, timezone
from email.utils import format_datetime

from app.data_sources import news


def test_fetch_news_maps_naver_items_to_scored_events(monkeypatch):
    recent_pub_date = format_datetime(datetime.now(timezone.utc))
    monkeypatch.setattr(news, "record_news_events", lambda query, events: None)
    monkeypatch.setattr(
        news,
        "search_naver_news",
        lambda query, display=20, sort="date": {
            "connected": True,
            "items": [
                {
                    "title": "삼성전자 AI 반도체 공급 계약",
                    "description": "대형 공급 계약 체결",
                    "originallink": "https://example.com/news",
                    "link": "https://example.com/naver",
                    "pubDate": recent_pub_date,
                }
            ],
        },
    )

    result = news.fetch_news("삼성전자", lookback_hours=48)

    assert len(result) == 1
    assert result[0]["source"] == "Naver Search API"
    assert result[0]["impact_direction"] == "positive"
    assert "계약" in result[0]["matched_keywords"]


def test_fetch_news_returns_empty_when_naver_is_not_connected(monkeypatch):
    monkeypatch.setattr(
        news,
        "search_naver_news",
        lambda query, display=20, sort="date": {"connected": False, "items": []},
    )

    assert news.fetch_news("삼성전자", lookback_hours=48) == []
