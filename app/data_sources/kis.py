from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable

from app.brokers.kis_client import KisApiError, KisClient, KisConfigError
from app.config import settings
from app.features.technical import build_technical_features, latest_technical_features
from app.storage.market_data import record_price_snapshot


def fetch_price_data(symbol: str) -> dict[str, Any]:
    """
    Fetch KIS market data and derive trading indicators.

    Current price and daily candles are treated as critical. Intraday minute,
    orderbook, execution, and investor-flow APIs are optional enrichments so
    one delayed KIS endpoint does not break the whole pipeline.
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
            "orderbook": {},
            "executions": {},
            "investor_flow": {},
            "intraday": {},
            "message": "KIS API credentials are not configured.",
        }

    try:
        client = KisClient()
        now = datetime.now()
        data = client.get_current_price(symbol)
        daily_data = client.get_daily_prices(
            symbol=symbol,
            start_date=(now - timedelta(days=90)).strftime("%Y%m%d"),
            end_date=now.strftime("%Y%m%d"),
        )
    except (KisApiError, KisConfigError, Exception) as exc:
        return {
            "status": "error",
            "symbol": symbol,
            "current_price": None,
            "change_rate": None,
            "volume": None,
            "volume_ratio": None,
            "minute_candles": [],
            "orderbook": {},
            "executions": {},
            "investor_flow": {},
            "intraday": {},
            "message": str(exc),
        }

    optional_errors: list[str] = []
    minute_data = _optional_api_call(
        "minute_prices",
        lambda: client.get_minute_prices(symbol=symbol),
        optional_errors,
    )
    orderbook_data = _optional_api_call(
        "orderbook",
        lambda: client.get_orderbook(symbol=symbol),
        optional_errors,
    )
    execution_data = _optional_api_call(
        "executions",
        lambda: client.get_executions(symbol=symbol),
        optional_errors,
    )
    investor_data = _optional_api_call(
        "investor_flow",
        lambda: client.get_investor_flow(symbol=symbol),
        optional_errors,
    )
    investor_daily_data = _optional_api_call(
        "investor_daily",
        lambda: client.get_investor_daily(
            symbol=symbol,
            start_date=(datetime.now() - timedelta(days=14)).strftime("%Y%m%d"),
        ),
        optional_errors,
    )

    output = data.get("output") or {}
    daily_candles = _parse_daily_candles(daily_data)
    minute_candles = _parse_minute_candles(minute_data)
    technical_features = build_technical_features(daily_candles)
    orderbook = _parse_orderbook(orderbook_data)
    executions = _parse_executions(execution_data)
    investor_flow = _parse_investor_flow(investor_data, investor_daily_data)

    current_price = _to_float(output.get("stck_prpr"))
    change_rate = _to_float(output.get("prdy_ctrt"))
    volume = _to_int(output.get("acml_vol"))
    turnover_value = _to_float(output.get("acml_tr_pbmn"))
    indicators = _calculate_indicators(
        current_price=current_price,
        change_rate=change_rate,
        volume=volume,
        turnover_value=turnover_value,
        daily_candles=daily_candles,
    )
    intraday = _calculate_intraday_indicators(
        current_price=current_price,
        minute_candles=minute_candles,
        orderbook=orderbook,
        executions=executions,
        investor_flow=investor_flow,
    )

    result = {
        "status": "connected",
        "symbol": symbol,
        "current_price": current_price,
        "change_rate": change_rate,
        "volume": volume,
        "volume_ratio": indicators.get("volume_ratio"),
        "turnover_value": turnover_value,
        "trend": indicators.get("trend"),
        "overheated": _is_overheated(
            indicators.get("overheated"),
            change_rate=change_rate,
            volume_ratio=indicators.get("volume_ratio"),
            intraday=intraday,
        ),
        "moving_averages": indicators.get("moving_averages"),
        "support_levels": indicators.get("support_levels"),
        "resistance_levels": indicators.get("resistance_levels"),
        "price_position": indicators.get("price_position"),
        "minute_candles": minute_candles[:30],
        "daily_candles": daily_candles[:30],
        "technical_features": technical_features[-30:],
        "latest_technical_features": latest_technical_features(daily_candles),
        "orderbook": orderbook,
        "executions": executions,
        "investor_flow": investor_flow,
        "intraday": intraday,
        "optional_errors": optional_errors,
        "source": "KIS Open API",
        "raw": output,
    }
    record_price_snapshot(result)
    return result


def _optional_api_call(
    name: str,
    call: Callable[[], dict[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    try:
        return call()
    except Exception as exc:
        errors.append(f"{name}: {exc}")
        return {}


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value).replace(",", "").strip()))
    except ValueError:
        return None


def _parse_daily_candles(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _rows(data, "output2", "output")
    candles: list[dict[str, Any]] = []
    for row in rows:
        close = _to_float(row.get("stck_clpr") or row.get("stck_prpr"))
        if close is None:
            continue
        candles.append(
            {
                "date": row.get("stck_bsop_date"),
                "open": _to_float(row.get("stck_oprc")),
                "high": _to_float(row.get("stck_hgpr")),
                "low": _to_float(row.get("stck_lwpr")),
                "close": close,
                "volume": _to_int(row.get("acml_vol") or row.get("cntg_vol")),
                "turnover_value": _to_float(row.get("acml_tr_pbmn")),
            }
        )
    return candles


def _parse_minute_candles(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _rows(data, "output2", "output")
    candles: list[dict[str, Any]] = []
    for row in rows:
        close = _to_float(row.get("stck_prpr") or row.get("stck_clpr"))
        if close is None:
            continue
        candles.append(
            {
                "date": row.get("stck_bsop_date"),
                "time": row.get("stck_cntg_hour"),
                "open": _to_float(row.get("stck_oprc")),
                "high": _to_float(row.get("stck_hgpr")),
                "low": _to_float(row.get("stck_lwpr")),
                "close": close,
                "volume": _to_int(row.get("cntg_vol") or row.get("acml_vol")),
                "turnover_value": _to_float(row.get("acml_tr_pbmn")),
            }
        )
    return sorted(candles, key=lambda row: (str(row.get("date") or ""), str(row.get("time") or "")))


def _parse_orderbook(data: dict[str, Any]) -> dict[str, Any]:
    row = _first_dict(data, "output1", "output")
    expected = _first_dict(data, "output2")
    ask_levels: list[dict[str, Any]] = []
    bid_levels: list[dict[str, Any]] = []

    for level in range(1, 11):
        ask_price = _to_float(row.get(f"askp{level}"))
        bid_price = _to_float(row.get(f"bidp{level}"))
        ask_size = _to_int(row.get(f"askp_rsqn{level}"))
        bid_size = _to_int(row.get(f"bidp_rsqn{level}"))
        if ask_price is not None or ask_size is not None:
            ask_levels.append(
                {"level": level, "price": ask_price, "size": ask_size or 0}
            )
        if bid_price is not None or bid_size is not None:
            bid_levels.append(
                {"level": level, "price": bid_price, "size": bid_size or 0}
            )

    total_ask_size = (
        _to_int(row.get("total_askp_rsqn"))
        or _to_int(row.get("askp_rsqn"))
        or sum(level["size"] for level in ask_levels)
    )
    total_bid_size = (
        _to_int(row.get("total_bidp_rsqn"))
        or _to_int(row.get("bidp_rsqn"))
        or sum(level["size"] for level in bid_levels)
    )
    best_ask = ask_levels[0]["price"] if ask_levels else None
    best_bid = bid_levels[0]["price"] if bid_levels else None
    spread = best_ask - best_bid if best_ask and best_bid else None
    mid_price = (best_ask + best_bid) / 2 if best_ask and best_bid else None
    spread_pct = round((spread / mid_price) * 100, 3) if spread and mid_price else None
    depth_total = total_ask_size + total_bid_size
    imbalance = (
        round((total_bid_size - total_ask_size) / depth_total, 4)
        if depth_total
        else None
    )

    return {
        "best_ask": best_ask,
        "best_bid": best_bid,
        "spread": spread,
        "spread_pct": spread_pct,
        "total_ask_size": total_ask_size,
        "total_bid_size": total_bid_size,
        "orderbook_imbalance": imbalance,
        "ask_levels": ask_levels,
        "bid_levels": bid_levels,
        "expected_price": _to_float(
            expected.get("antc_cnpr") or row.get("antc_cnpr")
        ),
        "expected_volume": _to_int(
            expected.get("antc_cntg_vol") or row.get("antc_cntg_vol")
        ),
    }


def _parse_executions(data: dict[str, Any]) -> dict[str, Any]:
    rows = _rows(data, "output", "output2", "output1")
    parsed_rows: list[dict[str, Any]] = []
    for row in rows[:30]:
        parsed_rows.append(
            {
                "time": row.get("stck_cntg_hour"),
                "price": _to_float(row.get("stck_prpr")),
                "change_rate": _to_float(row.get("prdy_ctrt")),
                "volume": _to_int(row.get("cntg_vol")),
                "execution_strength": _pick_number(row, "tday_rltv", "cttr", "cnqn"),
                "sign": row.get("prdy_vrss_sign"),
            }
        )

    strengths = [
        row["execution_strength"]
        for row in parsed_rows
        if isinstance(row.get("execution_strength"), (int, float))
    ]
    latest_strength = strengths[0] if strengths else None
    avg_strength = _average(strengths[:10]) if strengths else None
    buy_ticks = len([row for row in parsed_rows[:10] if row.get("sign") in ("1", "2")])
    sell_ticks = len([row for row in parsed_rows[:10] if row.get("sign") in ("4", "5")])

    return {
        "latest_execution_strength": latest_strength,
        "average_execution_strength": round(avg_strength, 2) if avg_strength else None,
        "recent_buy_ticks": buy_ticks,
        "recent_sell_ticks": sell_ticks,
        "rows": parsed_rows,
    }


def _parse_investor_flow(
    investor_data: dict[str, Any],
    investor_daily_data: dict[str, Any],
) -> dict[str, Any]:
    rows = _rows(investor_data, "output", "output2", "output1")
    daily_rows = _rows(investor_daily_data, "output2", "output", "output1")
    source_rows = rows or daily_rows
    latest = source_rows[0] if source_rows else {}

    foreign = _investor_value(latest, "frgn_ntby_qty", "frgn_ntby_tr_pbmn")
    institution = _investor_value(latest, "orgn_ntby_qty", "inst_ntby_qty", "orgn_ntby_tr_pbmn")
    individual = _investor_value(latest, "prsn_ntby_qty", "ivid_ntby_qty")
    program = _investor_value(latest, "pgtr_ntby_qty", "program_ntby_qty", "pgm_ntby_qty")

    history_rows = daily_rows or rows
    history: list[dict[str, Any]] = []
    for row in history_rows[:10]:
        row_foreign = _investor_value(row, "frgn_ntby_qty", "frgn_ntby_tr_pbmn")
        row_institution = _investor_value(row, "orgn_ntby_qty", "inst_ntby_qty", "orgn_ntby_tr_pbmn")
        row_individual = _investor_value(row, "prsn_ntby_qty", "ivid_ntby_qty")
        row_program = _investor_value(row, "pgtr_ntby_qty", "program_ntby_qty", "pgm_ntby_qty")
        smart_money = _sum_optional(row_foreign, row_institution)
        history.append(
            {
                "date": row.get("stck_bsop_date"),
                "foreign_net_buy": row_foreign,
                "institution_net_buy": row_institution,
                "individual_net_buy": row_individual,
                "program_net_buy": row_program,
                "smart_money_net_buy": smart_money,
            }
        )

    smart_money_values = [
        row["smart_money_net_buy"]
        for row in history[:5]
        if isinstance(row.get("smart_money_net_buy"), (int, float))
    ]
    smart_money_5d = sum(smart_money_values) if smart_money_values else None
    smart_money_days_positive = (
        len([value for value in smart_money_values if value > 0])
        if smart_money_values
        else None
    )

    return {
        "foreign_net_buy": foreign,
        "institution_net_buy": institution,
        "individual_net_buy": individual,
        "program_net_buy": program,
        "smart_money_net_buy": _sum_optional(foreign, institution),
        "smart_money_net_buy_5d": smart_money_5d,
        "smart_money_days_positive": smart_money_days_positive,
        "history": history,
    }


def _calculate_indicators(
    current_price: float | None,
    change_rate: float | None,
    volume: int | None,
    turnover_value: float | None,
    daily_candles: list[dict[str, Any]],
) -> dict[str, Any]:
    closes = [c["close"] for c in daily_candles if isinstance(c.get("close"), (int, float))]
    volumes = [c["volume"] for c in daily_candles[1:21] if isinstance(c.get("volume"), int)]
    price = current_price or (closes[0] if closes else None)
    ma5 = _average(closes[:5])
    ma20 = _average(closes[:20])
    avg_volume20 = _average(volumes)
    volume_ratio = round(volume / avg_volume20, 2) if volume and avg_volume20 else None

    support_levels = sorted(
        {c["low"] for c in daily_candles[:20] if isinstance(c.get("low"), (int, float))}
    )[:3]
    resistance_levels = sorted(
        {c["high"] for c in daily_candles[:20] if isinstance(c.get("high"), (int, float))},
        reverse=True,
    )[:3]

    if price and ma5 and ma20 and price >= ma5 >= ma20:
        trend = "uptrend"
    elif price and ma5 and ma20 and price <= ma5 <= ma20:
        trend = "downtrend"
    else:
        trend = "sideways"

    overheated = bool(
        (change_rate is not None and change_rate >= 8)
        or (volume_ratio is not None and volume_ratio >= 5 and change_rate and change_rate > 5)
    )

    return {
        "trend": trend,
        "overheated": overheated,
        "volume_ratio": volume_ratio,
        "moving_averages": {
            "ma5": round(ma5, 2) if ma5 else None,
            "ma20": round(ma20, 2) if ma20 else None,
        },
        "support_levels": support_levels,
        "resistance_levels": resistance_levels,
        "price_position": {
            "above_ma5": bool(price and ma5 and price >= ma5),
            "above_ma20": bool(price and ma20 and price >= ma20),
            "distance_from_ma20_pct": round(((price - ma20) / ma20) * 100, 2)
            if price and ma20
            else None,
            "turnover_value": turnover_value,
        },
    }


def _calculate_intraday_indicators(
    current_price: float | None,
    minute_candles: list[dict[str, Any]],
    orderbook: dict[str, Any],
    executions: dict[str, Any],
    investor_flow: dict[str, Any],
) -> dict[str, Any]:
    closes = [
        candle["close"]
        for candle in minute_candles
        if isinstance(candle.get("close"), (int, float))
    ]
    volumes = [
        candle["volume"]
        for candle in minute_candles
        if isinstance(candle.get("volume"), int)
    ]
    latest_close = current_price or (closes[-1] if closes else None)
    oldest_close = closes[0] if closes else None
    minute_momentum_pct = (
        round(((latest_close - oldest_close) / oldest_close) * 100, 2)
        if latest_close and oldest_close
        else None
    )
    vwap_numerator = sum(
        candle["close"] * candle["volume"]
        for candle in minute_candles
        if isinstance(candle.get("close"), (int, float))
        and isinstance(candle.get("volume"), int)
        and candle["volume"] > 0
    )
    vwap_denominator = sum(volume for volume in volumes if volume > 0)
    minute_vwap = (
        round(vwap_numerator / vwap_denominator, 2)
        if vwap_denominator
        else None
    )
    above_minute_vwap = (
        bool(latest_close and minute_vwap and latest_close >= minute_vwap)
        if minute_vwap
        else None
    )
    previous_volume_avg = _average(volumes[:-1])
    minute_volume_ratio = (
        round(volumes[-1] / previous_volume_avg, 2)
        if volumes and previous_volume_avg
        else None
    )
    execution_strength = executions.get("latest_execution_strength")
    orderbook_imbalance = orderbook.get("orderbook_imbalance")
    spread_pct = orderbook.get("spread_pct")
    smart_money_net_buy = investor_flow.get("smart_money_net_buy")
    smart_money_net_buy_5d = investor_flow.get("smart_money_net_buy_5d")
    smart_money_days_positive = investor_flow.get("smart_money_days_positive")
    program_net_buy = investor_flow.get("program_net_buy")

    score = 50.0
    if above_minute_vwap is True:
        score += 10
    elif above_minute_vwap is False:
        score -= 8

    if isinstance(minute_momentum_pct, (int, float)):
        if 0.2 <= minute_momentum_pct <= 3:
            score += 12
        elif 0 < minute_momentum_pct < 0.2:
            score += 4
        elif minute_momentum_pct > 3:
            score += 4
            score -= 6
        elif minute_momentum_pct <= -1.5:
            score -= 15
        elif minute_momentum_pct < 0:
            score -= 8

    if isinstance(minute_volume_ratio, (int, float)):
        if 1.5 <= minute_volume_ratio <= 4:
            score += 8
        elif minute_volume_ratio > 4:
            score += 3
            score -= 5
        elif minute_volume_ratio < 0.7:
            score -= 6

    if isinstance(execution_strength, (int, float)):
        if execution_strength >= 150:
            score += 15
        elif execution_strength >= 120:
            score += 10
        elif execution_strength >= 100:
            score += 5
        elif execution_strength < 80:
            score -= 15
        elif execution_strength < 95:
            score -= 8

    if isinstance(orderbook_imbalance, (int, float)):
        if orderbook_imbalance >= 0.2:
            score += 12
        elif orderbook_imbalance >= 0.05:
            score += 5
        elif orderbook_imbalance <= -0.2:
            score -= 12
        elif orderbook_imbalance <= -0.05:
            score -= 5

    if isinstance(spread_pct, (int, float)):
        if spread_pct <= 0.15:
            score += 4
        elif spread_pct > 0.8:
            score -= 15
        elif spread_pct > 0.4:
            score -= 6

    smart_money_signal = (
        smart_money_net_buy_5d
        if isinstance(smart_money_net_buy_5d, (int, float))
        else smart_money_net_buy
    )
    if isinstance(smart_money_signal, (int, float)):
        if smart_money_signal > 0:
            score += 8
        elif smart_money_signal < 0:
            score -= 8
    if isinstance(smart_money_days_positive, int):
        if smart_money_days_positive >= 3:
            score += 5
        elif smart_money_days_positive == 0:
            score -= 4

    if isinstance(program_net_buy, (int, float)):
        if program_net_buy > 0:
            score += 3
        elif program_net_buy < 0:
            score -= 3

    return {
        "minute_momentum_pct": minute_momentum_pct,
        "minute_vwap": minute_vwap,
        "above_minute_vwap": above_minute_vwap,
        "minute_volume_ratio": minute_volume_ratio,
        "execution_strength": execution_strength,
        "average_execution_strength": executions.get("average_execution_strength"),
        "orderbook_imbalance": orderbook_imbalance,
        "spread_pct": spread_pct,
        "smart_money_net_buy": smart_money_net_buy,
        "smart_money_net_buy_5d": smart_money_net_buy_5d,
        "smart_money_days_positive": smart_money_days_positive,
        "program_net_buy": program_net_buy,
        "intraday_score": _clamp_score(score),
    }


def _is_overheated(
    base_overheated: Any,
    change_rate: float | None,
    volume_ratio: Any,
    intraday: dict[str, Any],
) -> bool:
    if base_overheated:
        return True
    minute_momentum = intraday.get("minute_momentum_pct")
    minute_volume_ratio = intraday.get("minute_volume_ratio")
    execution_strength = intraday.get("execution_strength")
    return bool(
        isinstance(change_rate, (int, float))
        and change_rate >= 7
        and (
            (isinstance(volume_ratio, (int, float)) and volume_ratio >= 4.5)
            or (isinstance(minute_momentum, (int, float)) and minute_momentum >= 3)
            or (isinstance(minute_volume_ratio, (int, float)) and minute_volume_ratio >= 4)
            or (isinstance(execution_strength, (int, float)) and execution_strength >= 180)
        )
    )


def _rows(data: dict[str, Any], *keys: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            rows.extend([row for row in value if isinstance(row, dict)])
        elif isinstance(value, dict):
            rows.append(value)
    return rows


def _first_dict(data: dict[str, Any], *keys: str) -> dict[str, Any]:
    rows = _rows(data, *keys)
    return rows[0] if rows else {}


def _pick_number(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _to_float(row.get(key))
        if value is not None:
            return value
    return None


def _investor_value(row: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = _to_int(row.get(key))
        if value is not None:
            return value
    return None


def _sum_optional(*values: int | float | None) -> int | float | None:
    numeric = [value for value in values if isinstance(value, (int, float))]
    if not numeric:
        return None
    return sum(numeric)


def _average(values: list[float | int]) -> float | None:
    cleaned = [float(v) for v in values if isinstance(v, (int, float))]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def _clamp_score(value: float) -> float:
    return max(0.0, min(100.0, round(value, 1)))
