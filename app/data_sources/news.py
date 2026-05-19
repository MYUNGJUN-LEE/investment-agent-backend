from __future__ import annotations

from app.config import settings


def fetch_news(query: str, lookback_hours: int) -> list[dict]:
    """
    News API placeholder.

    Later options:
    - Naver Search News API
    - paid news APIs
    - your own curated RSS/news DB

    This function intentionally returns an empty list until configured.
    """
    if not query:
        return []

    if not settings.naver_client_id or not settings.naver_client_secret:
        return []

    # TODO: Implement Naver News API integration.
    # Keep this stub safe so the server runs without credentials.
    return []
