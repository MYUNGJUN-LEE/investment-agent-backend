from __future__ import annotations

from typing import Any

from app.config import settings

BPS = 10_000
NET_STOP_LOSS_BPS = 300.0
NET_TAKE_PROFIT_BPS = 500.0
MIN_TARGET_R_MULTIPLIER = 1.5
MAX_TARGET_R_MULTIPLIER = 2.5
ATR_TRAILING_MULTIPLIER = 2.0


def atr_exit_levels(
    *,
    entry_price: float | None,
    atr14: float | None,
    highest_close: float | None = None,
    round_trip_cost_bps: float | None = None,
) -> dict[str, float | None]:
    entry = _to_float(entry_price)
    atr = _to_float(atr14)
    if entry is None or entry <= 0:
        return {
            "stop_loss": None,
            "take_profit": None,
            "trailing_stop": None,
            "risk_per_share": None,
            "atr_14": atr,
            "net_stop_loss_bps": NET_STOP_LOSS_BPS,
            "net_take_profit_bps": NET_TAKE_PROFIT_BPS,
            "round_trip_cost_bps": _round_trip_cost_bps(round_trip_cost_bps),
            "reward_risk_ratio": None,
        }

    cost_bps = _round_trip_cost_bps(round_trip_cost_bps)
    gross_stop_bps = max(50.0, NET_STOP_LOSS_BPS - cost_bps)
    stop_loss = max(0.01, entry * (1 - gross_stop_bps / BPS))
    risk_per_share = entry - stop_loss
    gross_target_bps = max(
        NET_TAKE_PROFIT_BPS + cost_bps,
        gross_stop_bps * MIN_TARGET_R_MULTIPLIER,
    )
    gross_target_bps = min(gross_target_bps, gross_stop_bps * MAX_TARGET_R_MULTIPLIER)
    take_profit = entry * (1 + gross_target_bps / BPS)

    trailing_stop = None
    highest = _to_float(highest_close)
    if highest is not None and highest > entry:
        trailing_candidates = [highest * (1 - gross_stop_bps / BPS)]
        if atr is not None and atr > 0:
            trailing_candidates.append(highest - ATR_TRAILING_MULTIPLIER * atr)
        trailing_stop = max(0.01, max(trailing_candidates))

    return {
        "stop_loss": _round(stop_loss),
        "take_profit": _round(take_profit),
        "trailing_stop": _round(trailing_stop),
        "risk_per_share": _round(risk_per_share),
        "atr_14": _round(atr),
        "net_stop_loss_bps": NET_STOP_LOSS_BPS,
        "net_take_profit_bps": NET_TAKE_PROFIT_BPS,
        "round_trip_cost_bps": _round(cost_bps),
        "reward_risk_ratio": _round((take_profit - entry) / risk_per_share),
    }


def atr_exit_levels_from_price_data(
    *,
    entry_price: float | None,
    price_data: dict[str, Any],
    round_trip_cost_bps: float | None = None,
) -> dict[str, float | None]:
    return atr_exit_levels(
        entry_price=entry_price,
        atr14=atr14_from_price_data(price_data, entry_price=entry_price),
        highest_close=highest_close_from_price_data(price_data),
        round_trip_cost_bps=round_trip_cost_bps,
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


def _round_trip_cost_bps(value: float | None = None) -> float:
    explicit = _to_float(value)
    if explicit is not None:
        return max(0.0, explicit)
    commission = max(0.0, float(settings.commission_rate or 0.0)) * 2
    sell_tax = max(0.0, float(settings.kr_stock_sell_tax_rate or 0.0))
    spread = max(0.0, float(settings.universe_scanner_default_spread_bps or 0.0))
    slippage = max(0.0, float(settings.universe_scanner_default_slippage_bps or 0.0))
    return (commission + sell_tax) * BPS + spread + slippage


def _round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None
