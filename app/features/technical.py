from __future__ import annotations

import math
from typing import Any


def build_technical_features(candles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Build technical features using only current and historical rows.

    The function sorts candles ascending by date/time and never references rows
    after the row currently being calculated.
    """
    rows = sorted(
        [row for row in candles if _num(row.get("close")) is not None],
        key=lambda row: (str(row.get("date") or ""), str(row.get("time") or "")),
    )
    closes = [_num(row.get("close")) or 0.0 for row in rows]
    opens = [_num(row.get("open")) for row in rows]
    highs = [_num(row.get("high")) for row in rows]
    lows = [_num(row.get("low")) for row in rows]
    volumes = [_num(row.get("volume")) for row in rows]
    returns_1d = [_pct_change(closes, i, 1) for i in range(len(rows))]
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd = [
        _sub_optional(ema12[i], ema26[i])
        for i in range(len(rows))
    ]
    macd_signal = _ema([value or 0.0 for value in macd], 9)

    features: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        ma5 = _mean(closes, i, 5)
        ma20 = _mean(closes, i, 20)
        ma60 = _mean(closes, i, 60)
        prev_ma20 = _mean(closes, i - 1, 20) if i > 0 else None
        atr14 = _atr(highs, lows, closes, i, 14)
        realized_vol_20 = _realized_volatility(returns_1d, i, 20)
        bollinger_width = _bollinger_width(closes, i, 20)
        previous_high_20 = _max_previous(highs, i, 20)
        previous_low_20 = _min_previous(lows, i, 20)
        previous_close = closes[i - 1] if i > 0 else None
        volume_avg20 = _mean_previous(volumes, i, 20)
        volume_prev = volumes[i - 1] if i > 0 else None

        feature = {
            "date": row.get("date"),
            "time": row.get("time"),
            "close": closes[i],
            "return_1d": _round(_pct_change(closes, i, 1)),
            "return_5d": _round(_pct_change(closes, i, 5)),
            "return_20d": _round(_pct_change(closes, i, 20)),
            "return_60d": _round(_pct_change(closes, i, 60)),
            "high_breakout_20d": bool(
                previous_high_20 is not None and closes[i] > previous_high_20
            ),
            "low_breakdown_20d": bool(
                previous_low_20 is not None and closes[i] < previous_low_20
            ),
            "gap_up_pct": _round(
                ((opens[i] - previous_close) / previous_close)
                if opens[i] is not None and previous_close
                else None
            ),
            "volume_change_rate": _round(
                ((volumes[i] - volume_prev) / volume_prev)
                if volumes[i] is not None and volume_prev
                else None
            ),
            "volume_ratio_20d": _round(
                volumes[i] / volume_avg20
                if volumes[i] is not None and volume_avg20
                else None
            ),
            "atr_14": _round(atr14),
            "atr_14_pct": _round(atr14 / closes[i] if atr14 and closes[i] else None),
            "realized_volatility_20d": _round(realized_vol_20),
            "bollinger_width_20d": _round(bollinger_width),
            "ma5": _round(ma5),
            "ma20": _round(ma20),
            "ma60": _round(ma60),
            "ma20_slope": _round(
                ma20 - prev_ma20 if ma20 is not None and prev_ma20 is not None else None
            ),
            "macd": _round(macd[i]),
            "macd_signal": _round(macd_signal[i]),
            "macd_histogram": _round(
                macd[i] - macd_signal[i]
                if macd[i] is not None and macd_signal[i] is not None
                else None
            ),
            "adx_14": _round(_adx(highs, lows, closes, i, 14)),
        }
        features.append(feature)
    return features


def latest_technical_features(candles: list[dict[str, Any]]) -> dict[str, Any]:
    features = build_technical_features(candles)
    return features[-1] if features else {}


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct_change(values: list[float], i: int, periods: int) -> float | None:
    if i - periods < 0:
        return None
    previous = values[i - periods]
    if previous == 0:
        return None
    return values[i] / previous - 1


def _mean(values: list[float | None], i: int, window: int) -> float | None:
    if i < 0:
        return None
    start = max(0, i - window + 1)
    sample = [value for value in values[start : i + 1] if value is not None]
    return sum(sample) / len(sample) if sample else None


def _mean_previous(values: list[float | None], i: int, window: int) -> float | None:
    if i <= 0:
        return None
    start = max(0, i - window)
    sample = [value for value in values[start:i] if value is not None]
    return sum(sample) / len(sample) if sample else None


def _max_previous(values: list[float | None], i: int, window: int) -> float | None:
    if i <= 0:
        return None
    sample = [value for value in values[max(0, i - window) : i] if value is not None]
    return max(sample) if sample else None


def _min_previous(values: list[float | None], i: int, window: int) -> float | None:
    if i <= 0:
        return None
    sample = [value for value in values[max(0, i - window) : i] if value is not None]
    return min(sample) if sample else None


def _ema(values: list[float], period: int) -> list[float | None]:
    alpha = 2 / (period + 1)
    result: list[float | None] = []
    ema_value: float | None = None
    for value in values:
        ema_value = value if ema_value is None else value * alpha + ema_value * (1 - alpha)
        result.append(ema_value)
    return result


def _atr(
    highs: list[float | None],
    lows: list[float | None],
    closes: list[float],
    i: int,
    window: int,
) -> float | None:
    true_ranges: list[float] = []
    for idx in range(max(0, i - window + 1), i + 1):
        if highs[idx] is None or lows[idx] is None:
            continue
        previous_close = closes[idx - 1] if idx > 0 else closes[idx]
        true_ranges.append(
            max(
                highs[idx] - lows[idx],
                abs(highs[idx] - previous_close),
                abs(lows[idx] - previous_close),
            )
        )
    return sum(true_ranges) / len(true_ranges) if true_ranges else None


def _realized_volatility(
    returns: list[float | None],
    i: int,
    window: int,
) -> float | None:
    sample = [
        value
        for value in returns[max(0, i - window + 1) : i + 1]
        if value is not None
    ]
    if len(sample) < 2:
        return None
    mean_return = sum(sample) / len(sample)
    variance = sum((value - mean_return) ** 2 for value in sample) / (len(sample) - 1)
    return math.sqrt(variance) * math.sqrt(252)


def _bollinger_width(values: list[float], i: int, window: int) -> float | None:
    sample = values[max(0, i - window + 1) : i + 1]
    if len(sample) < 2:
        return None
    mean_value = sum(sample) / len(sample)
    if mean_value == 0:
        return None
    variance = sum((value - mean_value) ** 2 for value in sample) / (len(sample) - 1)
    std = math.sqrt(variance)
    return (4 * std) / mean_value


def _adx(
    highs: list[float | None],
    lows: list[float | None],
    closes: list[float],
    i: int,
    window: int,
) -> float | None:
    if i < 1:
        return None
    dx_values: list[float] = []
    for idx in range(max(1, i - window + 1), i + 1):
        if highs[idx] is None or lows[idx] is None or highs[idx - 1] is None or lows[idx - 1] is None:
            continue
        up_move = highs[idx] - highs[idx - 1]
        down_move = lows[idx - 1] - lows[idx]
        plus_dm = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm = down_move if down_move > up_move and down_move > 0 else 0
        true_range = max(
            highs[idx] - lows[idx],
            abs(highs[idx] - closes[idx - 1]),
            abs(lows[idx] - closes[idx - 1]),
        )
        if true_range == 0:
            continue
        plus_di = 100 * plus_dm / true_range
        minus_di = 100 * minus_dm / true_range
        denominator = plus_di + minus_di
        if denominator:
            dx_values.append(100 * abs(plus_di - minus_di) / denominator)
    return sum(dx_values) / len(dx_values) if dx_values else None


def _sub_optional(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)
