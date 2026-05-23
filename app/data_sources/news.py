from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

from app.scoring import classify_text_impact
from app.services.naver_news import search_naver_news
from app.storage.market_data import record_news_events


def fetch_news(query: str, lookback_hours: int) -> list[dict]:
    if not query:
        return []

    result = search_naver_news(query=query, display=20, sort="date")
    if not result.get("connected"):
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    events: list[dict] = []
    for item in result.get("items", []):
        published_at = _parse_pub_date(item.get("pubDate"))
        if published_at and published_at < cutoff:
            continue

        title = item.get("title") or ""
        description = item.get("description") or ""
        impact = classify_text_impact(f"{title} {description}")
        events.append(
            {
                "source": "Naver Search API",
                "date": published_at.isoformat() if published_at else item.get("pubDate"),
                "title": title,
                "summary": description,
                "url": item.get("originallink") or item.get("link"),
                "event_type": "news",
                "impact_direction": impact["direction"],
                "impact_strength": impact["strength"],
                "matched_keywords": impact["matched_keywords"],
                "confidence": _confidence_from_impact(impact["strength"]),
                "raw": item,
            }
        )

    record_news_events(query, events)
    return events


def _parse_pub_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _confidence_from_impact(strength: float) -> int:
    if strength >= 60:
        return 75
    if strength >= 45:
        return 65
    return 50
