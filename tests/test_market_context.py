from __future__ import annotations

from datetime import date, timedelta
import sqlite3

from fastapi.testclient import TestClient

from app.config import settings
from app.data_sources import market_context
from app.main import app
from app.storage import market_data


client = TestClient(app)


def _auth_headers() -> dict[str, str]:
    if settings.backend_api_key:
        return {"X-API-Key": settings.backend_api_key}
    return {}


def _series(start: float, step: float, days: int = 30) -> list[dict[str, float | str]]:
    first_day = date(2026, 4, 1)
    return [
        {
            "date": (first_day + timedelta(days=idx)).isoformat(),
            "close": round(start + step * idx, 4),
        }
        for idx in range(days)
    ]


def _payload() -> dict:
    return {
        "source": "pytest",
        "trade_date": "2026-04-30",
        "indices": {
            "KOSPI": _series(2500, 5),
            "KOSDAQ": _series(850, 2),
        },
        "fx": {
            "usdkrw": _series(1390, -1.2),
        },
        "vix": _series(19, -0.08),
        "rates": {
            "KR": {"close": 3.25},
            "US": {"close": 4.35},
        },
        "sectors": {
            "semiconductor": _series(100, 1.2),
            "battery": _series(100, 0.2),
            "bio": _series(100, -0.1),
        },
    }


def test_calculate_market_context_builds_regime_and_sector_strength():
    result = market_context.calculate_market_context(
        raw=_payload(),
        symbol="005930",
        sector="semiconductor",
    )

    assert result["status"] == "connected"
    assert result["market_regime"] == "bull"
    assert result["risk_on_score"] >= 65
    assert result["kospi"]["above_ma20"] is True
    assert result["usdkrw"]["change_pct"] < 0
    assert result["rates"]["rate_spread"] == 1.1

    sectors = result["sector_relative_strength"]["sectors"]
    assert sectors[0]["sector"] == "semiconductor"
    assert sectors[0]["rank"] == 1
    assert sectors[0]["relative_strength_20d"] > 0
    assert result["selected_sector_relative_strength"]["sector"] == "semiconductor"


def test_fetch_market_context_persists_snapshot(tmp_path, monkeypatch):
    db_path = tmp_path / "market.sqlite3"
    monkeypatch.setattr(market_data, "_now", lambda: "2026-04-30T16:00:00")

    result = market_context.fetch_market_context(
        provider=market_context.StaticMarketContextProvider(_payload()),
        sector="semiconductor",
        db_path=db_path,
    )

    assert result["snapshot_id"] == 1
    latest = market_data.get_latest_market_context(db_path=db_path)
    assert latest is not None
    assert latest["market_regime"] == "bull"
    assert latest["kospi"]["close"] == result["kospi"]["close"]
    assert latest["sector_relative_strength"]["sectors"][0]["sector"] == "semiconductor"

    with sqlite3.connect(db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM market_context_snapshots"
        ).fetchone()[0]

    assert count == 1


def test_market_context_endpoint_accepts_manual_payload():
    response = client.post(
        "/market-context/run-once",
        headers=_auth_headers(),
        json={
            "sector": "semiconductor",
            "payload": _payload(),
            "persist": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "connected"
    assert body["market_regime"] == "bull"
    assert body["sector_relative_strength"]["sectors"][0]["sector"] == "semiconductor"
