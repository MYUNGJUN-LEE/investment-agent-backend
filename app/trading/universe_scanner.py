from __future__ import annotations

from datetime import datetime, time
import json
from pathlib import Path
import sqlite3
import time as time_module
from typing import Any
from uuid import uuid4

from app.config import settings
from app.data_sources.kis import fetch_price_data
from app.data_sources.opendart import fetch_opendart_disclosures
from app.models import AutoTradeStartRequest, AutoTradeSymbolConfig
from app.services.naver_news import search_naver_news


DEFAULT_UNIVERSE = {
    "005930": "Samsung Electronics",
    "000660": "SK hynix",
    "373220": "LG Energy Solution",
    "207940": "Samsung Biologics",
    "005380": "Hyundai Motor",
    "000270": "Kia",
    "068270": "Celltrion",
    "105560": "KB Financial",
    "055550": "Shinhan Financial",
    "035420": "NAVER",
    "035720": "Kakao",
    "012330": "Hyundai Mobis",
    "005490": "POSCO Holdings",
    "028260": "Samsung C&T",
    "051910": "LG Chem",
    "006400": "Samsung SDI",
    "032830": "Samsung Life",
    "086790": "Hana Financial",
    "015760": "KEPCO",
    "034020": "Doosan Enerbility",
    "009540": "HD Korea Shipbuilding",
    "010140": "Samsung Heavy Industries",
    "267260": "HD Hyundai Electric",
    "042660": "Hanwha Ocean",
    "012450": "Hanwha Aerospace",
    "329180": "HD Hyundai Heavy Industries",
    "064350": "Hyundai Rotem",
    "000810": "Samsung Fire",
    "003550": "LG",
    "066570": "LG Electronics",
    "096770": "SK Innovation",
    "247540": "ECOPRO BM",
    "086520": "ECOPRO",
    "196170": "Alteogen",
    "091990": "Celltrion Healthcare",
    "352820": "HYBE",
    "259960": "KRAFTON",
    "034730": "SK",
    "017670": "SK Telecom",
    "033780": "KT&G",
}


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS universe_scan_runs (
    scan_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    source_symbol_count INTEGER NOT NULL,
    candidate_limit INTEGER NOT NULL,
    final_limit INTEGER NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    raw_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS universe_price_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    name TEXT,
    current_price REAL,
    change_rate REAL,
    volume INTEGER,
    volume_ratio REAL,
    turnover_value REAL,
    trend TEXT,
    source TEXT,
    raw_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS universe_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    name TEXT,
    rank INTEGER NOT NULL,
    score REAL NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    current_price REAL,
    change_rate REAL,
    volume INTEGER,
    volume_ratio REAL,
    turnover_value REAL,
    news_count INTEGER DEFAULT 0,
    disclosure_count INTEGER DEFAULT 0,
    raw_json TEXT NOT NULL
);
"""


def scan_universe_for_auto_trade(
    req: AutoTradeStartRequest,
    *,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Discover symbols, store the scan, and return final auto-trading configs."""
    candidate_limit = min(
        int(req.universe_candidate_limit or settings.universe_scanner_candidate_limit),
        int(settings.universe_scanner_candidate_limit or 20),
    )
    final_limit = min(
        int(req.universe_final_limit or settings.universe_scanner_final_limit),
        int(settings.universe_scanner_final_limit or 10),
        candidate_limit,
    )
    source_symbols = _resolve_source_symbols(req)
    scan_id = f"scan-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"
    created_at = _now()
    path = _db_path(db_path)
    initialize_universe_db(path)

    snapshots = _collect_price_snapshots(source_symbols)
    _record_snapshots(path, scan_id, created_at, snapshots)

    ranked = sorted(
        (_score_snapshot(item) for item in snapshots),
        key=lambda item: item["score"],
        reverse=True,
    )
    candidates = ranked[:candidate_limit]
    enriched = [_enrich_candidate(candidate) for candidate in candidates]
    final_candidates = [
        item
        for item in sorted(enriched, key=lambda row: row["score"], reverse=True)
        if item["decision"] != "exclude"
    ][:final_limit]

    symbols = [
        _to_symbol_config(req, candidate)
        for candidate in final_candidates
        if candidate.get("current_price")
    ]
    result = {
        "scan_id": scan_id,
        "status": "success",
        "created_at": created_at,
        "source_symbol_count": len(source_symbols),
        "snapshot_count": len(snapshots),
        "candidate_limit": candidate_limit,
        "final_limit": final_limit,
        "candidate_count": len(candidates),
        "final_count": len(symbols),
        "candidates": _compact_candidates(enriched),
        "final_candidates": _compact_candidates(final_candidates),
        "symbols": symbols,
    }
    _record_candidates(path, scan_id, created_at, enriched)
    _record_scan_run(path, result)
    return result


def get_latest_universe_scan(db_path: Path | str | None = None) -> dict[str, Any]:
    path = _db_path(db_path)
    initialize_universe_db(path)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        run = conn.execute(
            """
            SELECT * FROM universe_scan_runs
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()
        if not run:
            return {"status": "empty", "message": "No universe scan has run yet"}
        candidates = conn.execute(
            """
            SELECT symbol, name, rank, score, decision, reason, current_price,
                   change_rate, volume, volume_ratio, turnover_value,
                   news_count, disclosure_count
            FROM universe_candidates
            WHERE scan_id = ?
            ORDER BY rank ASC
            """,
            (run["scan_id"],),
        ).fetchall()
    raw = _parse_json(run["raw_json"], {})
    raw["candidates"] = [dict(row) for row in candidates]
    return raw


def initialize_universe_db(db_path: Path | str | None = None) -> None:
    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)


def _resolve_source_symbols(req: AutoTradeStartRequest) -> dict[str, str | None]:
    symbols: dict[str, str | None] = {}
    for item in req.symbols:
        symbols[_normalize_symbol(item.symbol)] = item.name
    for raw in req.universe_seed_symbols:
        symbols.setdefault(_normalize_symbol(raw), None)
    for raw in _comma_list(settings.universe_scanner_seed_symbols):
        symbols.setdefault(_normalize_symbol(raw), None)
    for raw in _comma_list(settings.monitor_watchlist_symbols):
        symbols.setdefault(_normalize_symbol(raw), None)
    for symbol, name in DEFAULT_UNIVERSE.items():
        symbols.setdefault(symbol, name)
    max_symbols = max(1, int(settings.universe_scanner_max_source_symbols or 15))
    return {
        symbol: name
        for symbol, name in list(symbols.items())[:max_symbols]
        if symbol
    }


def _collect_price_snapshots(source_symbols: dict[str, str | None]) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    symbol_interval_seconds = max(
        0.0,
        float(settings.universe_scanner_symbol_interval_seconds or 0),
    )
    for index, (symbol, name) in enumerate(source_symbols.items()):
        if index and symbol_interval_seconds:
            time_module.sleep(symbol_interval_seconds)
        try:
            price_data = fetch_price_data(symbol)
        except Exception as exc:
            price_data = {
                "status": "error",
                "symbol": symbol,
                "message": str(exc),
            }
        price_data["symbol"] = _normalize_symbol(price_data.get("symbol") or symbol)
        price_data["name"] = name or DEFAULT_UNIVERSE.get(symbol)
        snapshots.append(price_data)
    return snapshots


def _score_snapshot(item: dict[str, Any]) -> dict[str, Any]:
    price = _to_float(item.get("current_price"))
    fallback_price = _latest_close_from_daily(item)
    market_open = _is_kr_regular_market_open()
    change_rate = _to_float(item.get("change_rate"))
    volume_ratio = _to_float(item.get("volume_ratio"))
    turnover_value = _to_float(item.get("turnover_value"))
    volume = _to_int(item.get("volume"))
    score = 0.0
    reasons: list[str] = []

    if price and price > 0:
        score += 15
    elif fallback_price and fallback_price > 0 and not market_open:
        price = fallback_price
        item = {
            **item,
            "current_price": price,
            "price_source": "latest_close",
            "market_phase": "closed",
        }
        score += 35
        reasons.append("market closed; latest close used for watchlist only")
    else:
        phase = "market closed" if not market_open else "market open"
        return {
            **item,
            "score": 0.0,
            "decision": "exclude",
            "reason": f"no price ({phase})",
            "market_phase": "closed" if not market_open else "open",
        }

    if change_rate is not None:
        if -1.5 <= change_rate <= 7:
            score += 25
            reasons.append("healthy intraday change")
        elif 7 < change_rate <= 12:
            score += 12
            reasons.append("strong but extended move")
        elif change_rate < -4:
            score -= 20
            reasons.append("sharp drop")

    if volume_ratio is not None:
        if 1.5 <= volume_ratio <= 5:
            score += 30
            reasons.append("volume expansion")
        elif volume_ratio > 5:
            score += 16
            reasons.append("possible volume exhaustion")

    if turnover_value is not None and turnover_value > 0:
        score += min(20, turnover_value / 50_000_000_000 * 20)
        reasons.append("turnover present")
    elif volume:
        score += min(10, volume / 5_000_000 * 10)
        reasons.append("volume present")

    intraday = item.get("intraday") or {}
    minute_volume_ratio = _to_float(intraday.get("minute_volume_ratio"))
    if minute_volume_ratio is not None and minute_volume_ratio >= 1.5:
        score += min(15, minute_volume_ratio * 3)
        reasons.append("minute volume expansion")

    if bool(item.get("overheated")):
        score -= 15
        reasons.append("overheated risk")

    decision = "exclude"
    if item.get("price_source") == "latest_close":
        decision = "watch"
    elif score >= 65:
        decision = "buy_candidate"
    elif score >= 40:
        decision = "watch"
    return {
        **item,
        "score": round(max(0.0, min(100.0, score)), 2),
        "decision": decision,
        "reason": ", ".join(reasons) or "insufficient signal",
    }


def _enrich_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    symbol = str(candidate.get("symbol") or "")
    query = candidate.get("name") or symbol
    news_count = 0
    disclosure_count = 0
    score = float(candidate.get("score") or 0)
    enrichment_errors: list[str] = []

    try:
        news = search_naver_news(query=query, display=5, sort="date")
        news_count = len(news.get("items") or [])
        if news_count:
            score += min(5, news_count)
    except Exception as exc:
        enrichment_errors.append(f"news: {exc}")

    try:
        disclosures = fetch_opendart_disclosures(symbol=symbol, lookback_hours=24)
        disclosure_count = len(disclosures)
        for event in disclosures[:3]:
            direction = event.get("impact_direction")
            strength = _to_float(event.get("impact_strength")) or 0
            if direction == "positive":
                score += min(8, strength / 12)
            elif direction == "negative":
                score -= min(12, strength / 8)
    except Exception as exc:
        enrichment_errors.append(f"opendart: {exc}")

    score = round(max(0.0, min(100.0, score)), 2)
    decision = "exclude"
    if candidate.get("price_source") == "latest_close":
        decision = "watch"
    elif score >= 70:
        decision = "buy_candidate"
    elif score >= 45:
        decision = "watch"
    return {
        **candidate,
        "score": score,
        "decision": decision,
        "news_count": news_count,
        "disclosure_count": disclosure_count,
        "enrichment_errors": enrichment_errors,
    }


def _to_symbol_config(
    req: AutoTradeStartRequest,
    candidate: dict[str, Any],
) -> AutoTradeSymbolConfig:
    price = _to_float(candidate.get("current_price"))
    stop_loss = price * (1 - settings.monitor_default_stop_loss_pct / 100) if price else None
    take_profit = price * (1 + settings.monitor_default_take_profit_pct / 100) if price else None
    return AutoTradeSymbolConfig(
        symbol=str(candidate["symbol"]),
        name=candidate.get("name"),
        market="KR",
        strategy_type="daytrade",
        lookback_hours=24,
        risk_level="medium",
        requested_action="auto",
        price=price,
        decision_price=price,
        order_price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        signal_score=_to_float(candidate.get("score")),
        account_equity=_first_symbol_attr(
            req,
            "account_equity",
            fallback=req.account_equity,
        ),
        risk_per_trade=_first_symbol_attr(
            req,
            "risk_per_trade",
            fallback=req.risk_per_trade,
        ),
        cash_available=_first_symbol_attr(
            req,
            "cash_available",
            fallback=req.cash_available,
        ),
    )


def _first_symbol_attr(
    req: AutoTradeStartRequest,
    name: str,
    fallback: Any = None,
) -> Any:
    for item in req.symbols:
        value = getattr(item, name, None)
        if value is not None:
            return value
    return fallback


def _compact_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for index, item in enumerate(candidates, start=1):
        compact.append(
            {
                "rank": index,
                "symbol": item.get("symbol"),
                "name": item.get("name"),
                "score": item.get("score"),
                "decision": item.get("decision"),
                "reason": item.get("reason"),
                "current_price": item.get("current_price"),
                "change_rate": item.get("change_rate"),
                "volume": item.get("volume"),
                "volume_ratio": item.get("volume_ratio"),
                "turnover_value": item.get("turnover_value"),
                "news_count": item.get("news_count", 0),
                "disclosure_count": item.get("disclosure_count", 0),
            }
        )
    return compact


def _record_snapshots(
    path: Path,
    scan_id: str,
    created_at: str,
    snapshots: list[dict[str, Any]],
) -> None:
    with sqlite3.connect(path) as conn:
        conn.executemany(
            """
            INSERT INTO universe_price_snapshots (
                scan_id, created_at, symbol, name, current_price, change_rate,
                volume, volume_ratio, turnover_value, trend, source, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    scan_id,
                    created_at,
                    item.get("symbol"),
                    item.get("name"),
                    _to_float(item.get("current_price")),
                    _to_float(item.get("change_rate")),
                    _to_int(item.get("volume")),
                    _to_float(item.get("volume_ratio")),
                    _to_float(item.get("turnover_value")),
                    item.get("trend"),
                    item.get("source"),
                    _json(item),
                )
                for item in snapshots
            ],
        )


def _record_candidates(
    path: Path,
    scan_id: str,
    created_at: str,
    candidates: list[dict[str, Any]],
) -> None:
    with sqlite3.connect(path) as conn:
        conn.executemany(
            """
            INSERT INTO universe_candidates (
                scan_id, created_at, symbol, name, rank, score, decision, reason,
                current_price, change_rate, volume, volume_ratio, turnover_value,
                news_count, disclosure_count, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    scan_id,
                    created_at,
                    item.get("symbol"),
                    item.get("name"),
                    index,
                    _to_float(item.get("score")) or 0.0,
                    item.get("decision", "exclude"),
                    item.get("reason", ""),
                    _to_float(item.get("current_price")),
                    _to_float(item.get("change_rate")),
                    _to_int(item.get("volume")),
                    _to_float(item.get("volume_ratio")),
                    _to_float(item.get("turnover_value")),
                    _to_int(item.get("news_count")) or 0,
                    _to_int(item.get("disclosure_count")) or 0,
                    _json(item),
                )
                for index, item in enumerate(candidates, start=1)
            ],
        )


def _record_scan_run(path: Path, result: dict[str, Any]) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO universe_scan_runs (
                scan_id, created_at, source_symbol_count, candidate_limit,
                final_limit, status, error, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result["scan_id"],
                result["created_at"],
                result["source_symbol_count"],
                result["candidate_limit"],
                result["final_limit"],
                result["status"],
                result.get("error"),
                _json(_serializable_result(result)),
            ),
        )


def _serializable_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key != "symbols"
    }


def _normalize_symbol(value: Any) -> str:
    raw = str(value or "").strip()
    if raw.isdigit():
        return raw.zfill(6)
    return raw


def _comma_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value).replace(",", "").strip()))
    except (TypeError, ValueError):
        return None


def _latest_close_from_daily(item: dict[str, Any]) -> float | None:
    for candle in item.get("daily_candles") or []:
        close = _to_float(
            candle.get("close")
            or candle.get("stck_clpr")
            or candle.get("price")
        )
        if close and close > 0:
            return close
    return None


def _is_kr_regular_market_open(now: datetime | None = None) -> bool:
    now = now or datetime.now()
    if now.weekday() >= 5:
        return False
    current = now.time()
    return time(9, 0) <= current < time(15, 30)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _parse_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _db_path(db_path: Path | str | None = None) -> Path:
    return Path(db_path or settings.universe_scanner_db_path)
