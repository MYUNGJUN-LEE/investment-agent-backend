from __future__ import annotations

from app.config import settings


def fetch_price_data(symbol: str) -> dict:
    """
    KIS price API placeholder.

    Later:
    - token issuance
    - current price
    - minute candle
    - volume/value
    - order book
    - investor flow

    This returns a safe placeholder so the full pipeline can run immediately.
    """
    if not settings.kis_app_key or not settings.kis_app_secret:
        return {
            "status": "not_connected_yet",
            "symbol": symbol,
            "current_price": None,
            "change_rate": None,
            "volume": None,
            "volume_ratio": None,
            "minute_candles": [],
            "message": "KIS API credentials are not configured.",
        }

    # TODO: Implement KIS quote API integration.
    return {
        "status": "not_implemented_yet",
        "symbol": symbol,
        "current_price": None,
        "change_rate": None,
        "volume": None,
        "volume_ratio": None,
        "minute_candles": [],
    }
