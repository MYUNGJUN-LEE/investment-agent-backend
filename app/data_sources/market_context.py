from __future__ import annotations

from datetime import date
from typing import Any, Protocol

import httpx

from app.config import settings
from app.scoring import clamp_score
from app.storage.market_data import record_market_context_snapshot


class MarketContextProvider(Protocol):
    def fetch(self) -> dict[str, Any]:
        """Return a daily market-context payload."""


class StaticMarketContextProvider:
    """Test/manual provider for already-fetched market context data."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def fetch(self) -> dict[str, Any]:
        return self.payload


class HttpMarketContextProvider:
    """
    Fetch a JSON market-context payload from MARKET_CONTEXT_URL.

    Expected payload shape is intentionally flexible. It may contain:
    indices.KOSPI/KOSDAQ, fx.usdkrw, vix, rates.KR/US, and sectors.
    Each time series can be a list of {date, close} rows or an object with
    series/history/prices/data rows.
    """

    def __init__(
        self,
        url: str | None = None,
        timeout: float | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.url = url or settings.market_context_url
        self.timeout = timeout or settings.market_context_timeout
        self.transport = transport

    def fetch(self) -> dict[str, Any]:
        if not self.url:
            return {
                "status": "not_connected_yet",
                "message": "MARKET_CONTEXT_URL is not configured.",
                "data_needed": [
                    "Daily KOSPI/KOSDAQ time series",
                    "Daily USD/KRW, VIX, and interest-rate series",
                    "Daily sector ETF or industry-index time series",
                ],
            }

        with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
            response = client.get(self.url)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {"data": payload}


def fetch_market_context(
    symbol: str | None = None,
    sector: str | None = None,
    provider: MarketContextProvider | None = None,
    persist: bool = True,
    db_path: str | None = None,
) -> dict[str, Any]:
    provider = provider or HttpMarketContextProvider()

    try:
        raw = provider.fetch()
    except httpx.HTTPError as exc:
        return _not_connected(f"Market context HTTP error: {exc}")
    except Exception as exc:
        return _not_connected(f"Market context provider error: {exc}")

    if raw.get("status") in {"not_connected_yet", "error"} and not _has_market_payload(raw):
        return _not_connected(
            raw.get("message", "Market context provider is not connected."),
            data_needed=raw.get("data_needed"),
        )

    context = calculate_market_context(raw=raw, symbol=symbol, sector=sector)
    if persist and context["status"] in {"connected", "partial"}:
        context["snapshot_id"] = record_market_context_snapshot(context, db_path=db_path)
    return context


def calculate_market_context(
    raw: dict[str, Any],
    symbol: str | None = None,
    sector: str | None = None,
) -> dict[str, Any]:
    indices = raw.get("indices") if isinstance(raw.get("indices"), dict) else {}
    fx = raw.get("fx") if isinstance(raw.get("fx"), dict) else {}
    rates = raw.get("rates") if isinstance(raw.get("rates"), dict) else {}
    sectors = raw.get("sectors") if isinstance(raw.get("sectors"), dict) else {}

    kospi_source = _first_present(indices, "KOSPI", "kospi", "kospi_index")
    kosdaq_source = _first_present(indices, "KOSDAQ", "kosdaq", "kosdaq_index")
    usdkrw_source = _first_present(fx, "USD/KRW", "USDKRW", "USD_KRW", "usdkrw")
    kospi_series = _normalize_series(kospi_source if kospi_source is not None else raw.get("kospi"))
    kosdaq_series = _normalize_series(kosdaq_source if kosdaq_source is not None else raw.get("kosdaq"))
    usdkrw_series = _normalize_series(usdkrw_source if usdkrw_source is not None else raw.get("usdkrw"))
    vix_series = _normalize_series(raw.get("vix") or raw.get("VIX"))

    kospi = _summarize_series(kospi_series)
    kosdaq = _summarize_series(kosdaq_series)
    usdkrw = _summarize_series(usdkrw_series)
    vix = _summarize_series(vix_series)
    rate_metrics = _rate_metrics(raw=raw, rates=rates)
    sector_relative_strength = calculate_sector_relative_strength(
        sectors=sectors,
        benchmark=kospi_series,
        benchmark_name=raw.get("benchmark") or "KOSPI",
        selected_sector=sector,
    )

    risk_on_score = calculate_risk_on_score(
        kospi=kospi,
        kosdaq=kosdaq,
        usdkrw=usdkrw,
        vix=vix,
        rates=rate_metrics,
    )
    market_regime = calculate_market_regime(
        kospi=kospi,
        kosdaq=kosdaq,
        usdkrw=usdkrw,
        vix=vix,
        risk_on_score=risk_on_score,
    )
    missing = _missing_fields(kospi, kosdaq, usdkrw, vix, rate_metrics, sectors)

    return {
        "status": "connected" if "KOSPI" not in missing and "KOSDAQ" not in missing else "partial",
        "symbol": symbol,
        "sector": sector,
        "trade_date": raw.get("trade_date") or _latest_date(
            kospi_series,
            kosdaq_series,
            usdkrw_series,
            vix_series,
        )
        or date.today().isoformat(),
        "source": raw.get("source") or "market_context_provider",
        "market_regime": market_regime,
        "risk_on_score": risk_on_score,
        "kospi": kospi,
        "kosdaq": kosdaq,
        "usdkrw": usdkrw,
        "vix": vix,
        "rates": rate_metrics,
        "sector_relative_strength": sector_relative_strength,
        "selected_sector_relative_strength": _selected_sector_strength(
            sector_relative_strength,
            sector,
        ),
        "data_quality": {
            "missing": missing,
            "sample_counts": {
                "kospi": len(kospi_series),
                "kosdaq": len(kosdaq_series),
                "usdkrw": len(usdkrw_series),
                "vix": len(vix_series),
                "sectors": len(sectors),
            },
        },
        "raw": raw,
    }


def calculate_risk_on_score(
    kospi: dict[str, Any],
    kosdaq: dict[str, Any],
    usdkrw: dict[str, Any],
    vix: dict[str, Any],
    rates: dict[str, Any],
) -> float:
    score = 50.0
    score += _trend_points(kospi, above_points=10, slope_points=7)
    score += _trend_points(kosdaq, above_points=8, slope_points=6)

    vix_close = vix.get("close")
    if isinstance(vix_close, (int, float)):
        if vix_close <= 18:
            score += 10
        elif vix_close <= 22:
            score += 4
        elif vix_close >= 30:
            score -= 20
        elif vix_close >= 25:
            score -= 12

    vix_change = vix.get("change_pct")
    if isinstance(vix_change, (int, float)):
        if vix_change >= 10:
            score -= 8
        elif vix_change <= -5:
            score += 4

    usdkrw_change = usdkrw.get("change_pct")
    if isinstance(usdkrw_change, (int, float)):
        if usdkrw_change >= 1.0:
            score -= 8
        elif usdkrw_change <= -0.5:
            score += 4

    rate_spread = rates.get("rate_spread")
    if isinstance(rate_spread, (int, float)):
        if rate_spread >= 1.5:
            score -= 3
        elif rate_spread <= 0:
            score += 2

    return clamp_score(score)


def calculate_market_regime(
    kospi: dict[str, Any],
    kosdaq: dict[str, Any],
    usdkrw: dict[str, Any],
    vix: dict[str, Any],
    risk_on_score: float,
) -> str:
    vix_close = vix.get("close")
    usdkrw_change = usdkrw.get("change_pct")
    both_below_ma20 = kospi.get("above_ma20") is False and kosdaq.get("above_ma20") is False
    both_positive_slope = _positive(kospi.get("ma20_slope")) and _positive(kosdaq.get("ma20_slope"))
    vix_stress = isinstance(vix_close, (int, float)) and vix_close >= 25
    fx_stress = isinstance(usdkrw_change, (int, float)) and usdkrw_change >= 1.0

    if risk_on_score < 35 or (both_below_ma20 and (vix_stress or fx_stress)):
        return "bear"
    if risk_on_score >= 65 and both_positive_slope and not vix_stress:
        return "bull"
    return "sideways"


def calculate_sector_relative_strength(
    sectors: dict[str, Any],
    benchmark: list[dict[str, Any]],
    benchmark_name: str = "KOSPI",
    selected_sector: str | None = None,
) -> dict[str, Any]:
    benchmark_return_5d = _period_return(benchmark, 5)
    benchmark_return_20d = _period_return(benchmark, 20)
    rows: list[dict[str, Any]] = []

    for sector_name, sector_series_raw in sectors.items():
        series = _normalize_series(sector_series_raw)
        return_5d = _period_return(series, 5)
        return_20d = _period_return(series, 20)
        relative_5d = _diff_or_none(return_5d, benchmark_return_5d)
        relative_20d = _diff_or_none(return_20d, benchmark_return_20d)
        rows.append(
            {
                "sector": sector_name,
                "return_5d": _round(return_5d),
                "return_20d": _round(return_20d),
                "relative_strength_5d": _round(relative_5d),
                "relative_strength_20d": _round(relative_20d),
                "score": _sector_score(relative_5d, relative_20d),
                "sample_count": len(series),
            }
        )

    rows.sort(
        key=lambda row: (
            row["relative_strength_20d"] is not None,
            row["relative_strength_20d"] or -9999,
            row["relative_strength_5d"] or -9999,
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["leader"] = bool(
            isinstance(row.get("relative_strength_20d"), (int, float))
            and row["relative_strength_20d"] > 0
            and rank <= max(1, len(rows) // 3 or 1)
        )

    return {
        "benchmark": benchmark_name,
        "benchmark_return_5d": _round(benchmark_return_5d),
        "benchmark_return_20d": _round(benchmark_return_20d),
        "selected_sector": selected_sector,
        "sectors": rows,
    }


def _summarize_series(series: list[dict[str, Any]]) -> dict[str, Any]:
    closes = [point["close"] for point in series]
    close = closes[-1] if closes else None
    ma20 = _mean(closes[-20:]) if len(closes) >= 20 else None
    previous_ma20 = _mean(closes[-21:-1]) if len(closes) >= 21 else None
    ma20_slope = _pct_change(ma20, previous_ma20)
    return {
        "close": _round(close),
        "change_pct": _round(_change_pct(series)),
        "return_5d": _round(_period_return(series, 5)),
        "return_20d": _round(_period_return(series, 20)),
        "ma20": _round(ma20),
        "ma20_slope": _round(ma20_slope),
        "above_ma20": bool(close >= ma20) if close is not None and ma20 is not None else None,
    }


def _rate_metrics(raw: dict[str, Any], rates: dict[str, Any]) -> dict[str, Any]:
    korea_rate = _latest_value(
        raw.get("korea_rate")
        if "korea_rate" in raw
        else _first_present(rates, "KR", "KOR", "KOREA", "korea", "korea_rate")
    )
    us_rate = _latest_value(
        raw.get("us_rate")
        if "us_rate" in raw
        else _first_present(rates, "US", "USA", "us", "us_rate")
    )
    rate_spread = us_rate - korea_rate if us_rate is not None and korea_rate is not None else None
    return {
        "korea_rate": _round(korea_rate),
        "us_rate": _round(us_rate),
        "rate_spread": _round(rate_spread),
    }


def _normalize_series(value: Any) -> list[dict[str, Any]]:
    points = _raw_points(value)
    series: list[dict[str, Any]] = []
    for point in points:
        row = point if isinstance(point, dict) else {"close": point}
        close = _pick_float(
            row,
            "close",
            "value",
            "price",
            "last",
            "stck_prpr",
            "stck_clpr",
            "bas_idx",
        )
        if close is None:
            continue
        series.append(
            {
                "date": _pick_text(row, "date", "trade_date", "stck_bsop_date", "time"),
                "close": close,
                "change_pct": _pick_float(row, "change_pct", "change_rate", "prdy_ctrt"),
                "raw": row,
            }
        )

    if any(point.get("date") for point in series):
        return sorted(series, key=lambda point: str(point.get("date") or ""))
    return series


def _raw_points(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("series", "history", "prices", "candles", "data", "rows", "output2", "output"):
            nested = value.get(key)
            if isinstance(nested, list):
                return nested
        return [value]
    return [value]


def _change_pct(series: list[dict[str, Any]]) -> float | None:
    if not series:
        return None
    latest_change = series[-1].get("change_pct")
    if isinstance(latest_change, (int, float)):
        return float(latest_change)
    if len(series) < 2:
        return None
    return _pct_change(series[-1]["close"], series[-2]["close"])


def _period_return(series: list[dict[str, Any]], days: int) -> float | None:
    if len(series) <= days:
        return None
    return _pct_change(series[-1]["close"], series[-days - 1]["close"])


def _pct_change(current: float | None, previous: float | None) -> float | None:
    if current is None or previous in (None, 0):
        return None
    return (current - previous) / previous * 100


def _latest_value(value: Any) -> float | None:
    series = _normalize_series(value)
    return series[-1]["close"] if series else _to_float(value)


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _latest_date(*series_list: list[dict[str, Any]]) -> str | None:
    dates = [
        str(point["date"])
        for series in series_list
        for point in series
        if point.get("date")
    ]
    if not dates:
        return None
    latest = max(dates)
    if len(latest) == 8 and latest.isdigit():
        return f"{latest[:4]}-{latest[4:6]}-{latest[6:]}"
    return latest


def _missing_fields(
    kospi: dict[str, Any],
    kosdaq: dict[str, Any],
    usdkrw: dict[str, Any],
    vix: dict[str, Any],
    rates: dict[str, Any],
    sectors: dict[str, Any],
) -> list[str]:
    missing = []
    if kospi.get("close") is None:
        missing.append("KOSPI")
    if kosdaq.get("close") is None:
        missing.append("KOSDAQ")
    if usdkrw.get("close") is None:
        missing.append("USD/KRW")
    if vix.get("close") is None:
        missing.append("VIX")
    if rates.get("korea_rate") is None:
        missing.append("korea_rate")
    if rates.get("us_rate") is None:
        missing.append("us_rate")
    if not sectors:
        missing.append("sectors")
    return missing


def _trend_points(
    summary: dict[str, Any],
    above_points: float,
    slope_points: float,
) -> float:
    points = 0.0
    above_ma20 = summary.get("above_ma20")
    if above_ma20 is True:
        points += above_points
    elif above_ma20 is False:
        points -= above_points

    slope = summary.get("ma20_slope")
    if isinstance(slope, (int, float)):
        if slope > 0:
            points += slope_points
        elif slope < 0:
            points -= slope_points
    return points


def _sector_score(relative_5d: float | None, relative_20d: float | None) -> float | None:
    values = []
    if isinstance(relative_20d, (int, float)):
        values.append(50 + relative_20d * 2)
    if isinstance(relative_5d, (int, float)):
        values.append(50 + relative_5d)
    if not values:
        return None
    return clamp_score(sum(values) / len(values))


def _selected_sector_strength(
    sector_relative_strength: dict[str, Any],
    sector: str | None,
) -> dict[str, Any] | None:
    if not sector:
        return None
    for row in sector_relative_strength.get("sectors", []):
        if row.get("sector") == sector:
            return row
    return None


def _has_market_payload(raw: dict[str, Any]) -> bool:
    return any(key in raw for key in ("indices", "kospi", "kosdaq", "fx", "usdkrw", "vix", "rates", "sectors"))


def _not_connected(
    message: str,
    data_needed: Any = None,
) -> dict[str, Any]:
    return {
        "status": "not_connected_yet",
        "market_regime": "unknown",
        "risk_on_score": None,
        "sector_relative_strength": {"benchmark": "KOSPI", "sectors": []},
        "message": message,
        "data_needed": data_needed
        or [
            "Set MARKET_CONTEXT_URL or pass a payload to /market-context/run-once.",
        ],
    }


def _pick_float(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in row:
            value = _to_float(row.get(key))
            if value is not None:
                return value
    return None


def _pick_text(row: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _to_float(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


def _mean(values: list[float]) -> float | None:
    cleaned = [float(value) for value in values if isinstance(value, (int, float))]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def _diff_or_none(value: float | None, benchmark: float | None) -> float | None:
    if value is None or benchmark is None:
        return None
    return value - benchmark


def _positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and value > 0


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)
