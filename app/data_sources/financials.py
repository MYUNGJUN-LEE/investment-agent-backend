from __future__ import annotations

from datetime import datetime
import httpx

from app.config import settings
from app.data_sources.opendart import map_symbol_to_corp_code
from app.storage.market_data import record_financial_snapshot


def fetch_financial_data(symbol: str) -> dict:
    """
    Minimal OpenDART financial statement fetcher.

    This version fetches annual consolidated financial statements for the previous year.
    It returns raw data. Parsing key accounts can be added later.
    """
    if not settings.opendart_api_key:
        return {"status": "missing_opendart_api_key"}

    corp_code = map_symbol_to_corp_code(symbol)
    if not corp_code:
        return {"status": "missing_corp_code"}

    current_year = datetime.now().year
    business_year = str(current_year - 1)
    previous_year = str(current_year - 2)

    try:
        current_data = _fetch_financial_statement(corp_code, business_year)
        previous_data = _fetch_financial_statement(corp_code, previous_year)
    except Exception as exc:
        return {"error": str(exc)}

    current_metrics = parse_financial_metrics(current_data)
    previous_metrics = parse_financial_metrics(previous_data)
    growth_metrics = calculate_growth_metrics(current_metrics, previous_metrics)
    result = {
        "status": "connected" if current_metrics else "no_financial_metrics",
        "symbol": symbol,
        "corp_code": corp_code,
        "business_year": business_year,
        "previous_business_year": previous_year,
        "metrics": current_metrics,
        "previous_metrics": previous_metrics,
        "growth_metrics": growth_metrics,
        "valuation_metrics": {
            "per": None,
            "pbr": None,
            "ev_ebitda": None,
            "source": "not_connected",
        },
        "raw": current_data,
        "previous_raw": previous_data,
    }
    record_financial_snapshot(symbol, result)
    return result


def _fetch_financial_statement(corp_code: str, business_year: str) -> dict:
    url = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
    params = {
        "crtfc_key": settings.opendart_api_key,
        "corp_code": corp_code,
        "bsns_year": business_year,
        "reprt_code": "11011",  # annual report
        "fs_div": "CFS",        # consolidated financial statements
    }

    with httpx.Client(timeout=10) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        return response.json()


def parse_financial_metrics(data: dict) -> dict:
    rows = data.get("list") or []
    values = {
        "revenue": _find_account_value(rows, ["매출액", "영업수익", "수익(매출액)"]),
        "operating_income": _find_account_value(rows, ["영업이익"]),
        "net_income": _find_account_value(rows, ["당기순이익", "연결당기순이익"]),
        "total_assets": _find_account_value(rows, ["자산총계"]),
        "total_liabilities": _find_account_value(rows, ["부채총계"]),
        "total_equity": _find_account_value(rows, ["자본총계"]),
    }
    revenue = values["revenue"]
    operating_income = values["operating_income"]
    net_income = values["net_income"]
    equity = values["total_equity"]
    liabilities = values["total_liabilities"]

    values.update(
        {
            "operating_margin": _ratio(operating_income, revenue),
            "net_margin": _ratio(net_income, revenue),
            "roe": _ratio(net_income, equity),
            "debt_ratio": _ratio(liabilities, equity),
        }
    )
    return values


def calculate_growth_metrics(current: dict, previous: dict) -> dict:
    return {
        "revenue_growth": _growth(current.get("revenue"), previous.get("revenue")),
        "operating_income_growth": _growth(
            current.get("operating_income"),
            previous.get("operating_income"),
        ),
        "net_income_growth": _growth(current.get("net_income"), previous.get("net_income")),
    }


def _find_account_value(rows: list[dict], account_names: list[str]) -> float | None:
    for name in account_names:
        for row in rows:
            account_nm = row.get("account_nm") or ""
            if account_nm.strip() == name:
                return _to_float(row.get("thstrm_amount"))
    for name in account_names:
        for row in rows:
            account_nm = row.get("account_nm") or ""
            if name in account_nm:
                return _to_float(row.get("thstrm_amount"))
    return None


def _to_float(value: str | int | float | None) -> float | None:
    if value in (None, "", "-"):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return round((numerator / denominator) * 100, 2)


def _growth(current: float | None, previous: float | None) -> float | None:
    if current is None or previous in (None, 0):
        return None
    return round(((current - previous) / abs(previous)) * 100, 2)
