from __future__ import annotations

from datetime import datetime
import httpx

from app.config import settings
from app.data_sources.opendart import map_symbol_to_corp_code


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

    url = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
    params = {
        "crtfc_key": settings.opendart_api_key,
        "corp_code": corp_code,
        "bsns_year": business_year,
        "reprt_code": "11011",  # annual report
        "fs_div": "CFS",        # consolidated financial statements
    }

    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        return {"error": str(exc)}

    return data
