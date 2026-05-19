from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import csv
import httpx

from app.config import settings
from app.scoring import classify_text_impact


CORP_MAP_PATH = Path("data/corp_map.csv")


def load_corp_map() -> dict[str, str]:
    """
    data/corp_map.csv columns:
    symbol,corp_code,name

    For production, download and maintain OpenDART corp code data regularly.
    """
    mapping: dict[str, str] = {}
    if not CORP_MAP_PATH.exists():
        return mapping

    with CORP_MAP_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = (row.get("symbol") or "").strip()
            corp_code = (row.get("corp_code") or "").strip()
            if symbol and corp_code:
                mapping[symbol] = corp_code
    return mapping


def map_symbol_to_corp_code(symbol: str) -> str | None:
    return load_corp_map().get(symbol)


def fetch_opendart_disclosures(symbol: str, lookback_hours: int) -> list[dict]:
    """
    Fetch recent disclosures from OpenDART.

    Returns an empty list if OPENDART_API_KEY or corp_code is missing.
    """
    if not settings.opendart_api_key:
        return []

    corp_code = map_symbol_to_corp_code(symbol)
    if not corp_code:
        return []

    end_date = datetime.now()
    start_date = end_date - timedelta(hours=lookback_hours)

    url = "https://opendart.fss.or.kr/api/list.json"
    params = {
        "crtfc_key": settings.opendart_api_key,
        "corp_code": corp_code,
        "bgn_de": start_date.strftime("%Y%m%d"),
        "end_de": end_date.strftime("%Y%m%d"),
        "sort": "date",
        "sort_mth": "desc",
        "page_no": "1",
        "page_count": "20",
    }

    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        return [
            {
                "source": "OpenDART",
                "title": "OpenDART request error",
                "summary": str(exc),
                "impact_direction": "uncertain",
                "impact_strength": 0,
                "confidence": 0,
            }
        ]

    if data.get("status") not in (None, "000"):
        return [
            {
                "source": "OpenDART",
                "title": "OpenDART API status error",
                "summary": data.get("message", "Unknown OpenDART error"),
                "impact_direction": "uncertain",
                "impact_strength": 0,
                "confidence": 0,
                "raw": data,
            }
        ]

    result: list[dict] = []
    for item in data.get("list", []):
        title = item.get("report_nm") or ""
        impact = classify_text_impact(title)
        rcept_no = item.get("rcept_no")

        result.append(
            {
                "source": "OpenDART",
                "date": item.get("rcept_dt"),
                "title": title,
                "summary": f"{item.get('corp_name', '')} 공시: {title}",
                "url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}" if rcept_no else None,
                "event_type": "disclosure",
                "impact_direction": impact["direction"],
                "impact_strength": impact["strength"],
                "confidence": 85,
                "raw": item,
            }
        )

    return result
