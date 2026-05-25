from __future__ import annotations

from typing import Any


ATR_STOP_MULTIPLIER = 1.8
ATR_TARGET_R_MULTIPLIER = 2.5
ATR_TRAILING_MULTIPLIER = 2.0


def atr_exit_levels(
    *,
    entry_price: float | None,
    atr14: float | None,
    highest_close: float | None = None,
) -> dict[str, float | None]:
    entry = _to_float(entry_price)
    atr = _to_float(atr14)
    if entry is None or entry <= 0 or atr is None or atr <= 0:
        return {
            "stop_loss": None,
            "take_profit": None,
            "trailing_stop": None,
            "risk_per_share": None,
            "atr_14": atr,
        }

    stop_loss = max(0.01, entry - ATR_STOP_MULTIPLIER * atr)
    risk_per_share = entry - stop_loss
    take_profit = entry + ATR_TARGET_R_MULTIPLIER * risk_per_share

    trailing_stop = None
    highest = _to_float(highest_close)
    if highest is not None and highest > entry:
        trailing_stop = max(0.01, highest - ATR_TRAILING_MULTIPLIER * atr)

    return {
        "stop_loss": _round(stop_loss),
        "take_profit": _round(take_profit),
        "trailing_stop": _round(trailing_stop),
        "risk_per_share": _round(risk_per_share),
        "atr_14": _round(atr),
    }


def atr_exit_levels_from_price_data(
    *,
    entry_price: float | None,
    price_data: dict[str, Any],
) -> dict[str, float | None]:
    return atr_exit_levels(
        entry_price=entry_price,
        atr14=atr14_from_price_data(price_data, entry_price=entry_price),
        highest_close=highest_close_from_price_data(price_data),
    )


def atr14_from_price_data(
    price_data: dict[str, Any],
    *,
    entry_price: float | None = None,
) -> float | None:
    latest = price_data.get("latest_technical_features") or {}
    atr = _to_float(latest.get("atr_14"))
    if atr is not None and atr > 0:
        return atr

    atr_pct = _to_float(latest.get("atr_14_pct"))
    entry = _to_float(entry_price) or _to_float(price_data.get("current_price"))
    if atr_pct is not None and atr_pct > 0 and entry is not None and entry > 0:
        return entry * atr_pct

    features = price_data.get("technical_features") or []
    for feature in reversed(features):
        if not isinstance(feature, dict):
            continue
        atr = _to_float(feature.get("atr_14"))
        if atr is not None and atr > 0:
            return atr
        atr_pct = _to_float(feature.get("atr_14_pct"))
        close = _to_float(feature.get("close")) or entry
        if atr_pct is not None and atr_pct > 0 and close is not None and close > 0:
            return close * atr_pct

    return None


def highest_close_from_price_data(price_data: dict[str, Any]) -> float | None:
    closes: list[float] = []
    for key in ("technical_features", "daily_candles"):
        rows = price_data.get(key) or []
        for row in rows:
            if not isinstance(row, dict):
                continue
            close = _to_float(row.get("close"))
            if close is not None and close > 0:
                closes.append(close)
    current_price = _to_float(price_data.get("current_price"))
    if current_price is not None and current_price > 0:
        closes.append(current_price)
    return max(closes) if closes else None


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None
