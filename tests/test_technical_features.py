from __future__ import annotations

from app.features.technical import build_technical_features


def _candles(count: int = 80) -> list[dict]:
    return [
        {
            "date": f"2026-01-{idx + 1:02d}",
            "open": 100 + idx,
            "high": 102 + idx,
            "low": 99 + idx,
            "close": 101 + idx,
            "volume": 1000 + idx * 10,
        }
        for idx in range(count)
    ]


def test_technical_features_include_requested_categories():
    features = build_technical_features(_candles())
    latest = features[-1]

    assert latest["return_5d"] is not None
    assert latest["return_20d"] is not None
    assert latest["volume_ratio_20d"] is not None
    assert latest["atr_14"] is not None
    assert latest["realized_volatility_20d"] is not None
    assert latest["bollinger_width_20d"] is not None
    assert latest["ma20_slope"] is not None
    assert latest["macd"] is not None
    assert latest["adx_14"] is not None


def test_technical_features_do_not_look_ahead():
    candles = _candles()
    baseline = build_technical_features(candles)[30]
    changed_future = _candles()
    changed_future[60]["close"] = 1_000_000

    after_future_change = build_technical_features(changed_future)[30]

    assert after_future_change["return_5d"] == baseline["return_5d"]
    assert after_future_change["volume_ratio_20d"] == baseline["volume_ratio_20d"]
    assert after_future_change["ma20_slope"] == baseline["ma20_slope"]
