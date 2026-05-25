from __future__ import annotations

from datetime import datetime, time, timedelta
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
from app.storage.market_data import get_latest_market_context
from app.trading.edge_calibration import (
    edge_entry_gate,
    estimate_expected_edges,
    load_edge_model,
)
from app.trading.atr_exits import atr_exit_levels_from_price_data
from app.trading import paper_trading


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

CREATE TABLE IF NOT EXISTS scanner_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id TEXT NOT NULL,
    scan_time TEXT NOT NULL,
    symbol TEXT NOT NULL,
    name TEXT,
    raw_score REAL NOT NULL,
    expected_return REAL NOT NULL,
    expected_risk REAL NOT NULL,
    trading_cost REAL NOT NULL,
    slippage_cost REAL NOT NULL,
    net_edge REAL NOT NULL,
    composite_score REAL NOT NULL,
    rank INTEGER NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL,
    decision TEXT NOT NULL,
    current_price REAL,
    change_rate REAL,
    volume INTEGER,
    volume_ratio REAL,
    turnover_value REAL,
    news_count INTEGER DEFAULT 0,
    disclosure_count INTEGER DEFAULT 0,
    claimed_by_worker TEXT,
    expires_at TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    UNIQUE(symbol)
);

CREATE INDEX IF NOT EXISTS idx_scanner_candidates_ready
ON scanner_candidates(status, net_edge, expires_at);

CREATE TABLE IF NOT EXISTS scanner_candidate_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id TEXT NOT NULL,
    scan_time TEXT NOT NULL,
    symbol TEXT NOT NULL,
    name TEXT,
    raw_score REAL NOT NULL,
    expected_return REAL NOT NULL,
    expected_risk REAL NOT NULL,
    trading_cost REAL NOT NULL,
    slippage_cost REAL NOT NULL,
    net_edge REAL NOT NULL,
    composite_score REAL NOT NULL,
    rank INTEGER NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL,
    decision TEXT NOT NULL,
    current_price REAL,
    change_rate REAL,
    volume INTEGER,
    volume_ratio REAL,
    turnover_value REAL,
    news_count INTEGER DEFAULT 0,
    disclosure_count INTEGER DEFAULT 0,
    claimed_by_worker TEXT,
    expires_at TEXT,
    raw_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scanner_candidate_history_scan
ON scanner_candidate_history(scan_id, rank);
"""


def scan_universe_for_auto_trade(
    req: AutoTradeStartRequest,
    *,
    db_path: Path | str | None = None,
    worker_id: str | None = None,
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
    hurdle_rate = _worker_hurdle_rate_bps(req.execution_mode)
    ranked_candidates = _rank_execution_candidates(
        enriched,
        scan_time=created_at,
        final_limit=final_limit,
        hurdle_rate=hurdle_rate,
    )
    _record_candidates(path, scan_id, created_at, ranked_candidates)
    _record_scanner_candidates(
        path=path,
        scan_id=scan_id,
        scan_time=created_at,
        candidates=ranked_candidates,
        execution_limit=final_limit,
    )

    active_candidates = ranked_candidates[:final_limit]
    ready_candidates = get_ready_execution_candidates(
        db_path=path,
        worker_id=worker_id,
        limit=final_limit,
        worker_hurdle_rate=hurdle_rate,
    )

    symbols = [
        _to_symbol_config(req, candidate)
        for candidate in ready_candidates
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
        "final_count": len(active_candidates),
        "executable_count": len(symbols),
        "worker_hurdle_rate": hurdle_rate,
        "entry_gate": ranked_candidates[0].get("entry_gate") if ranked_candidates else None,
        "active_candidate_symbols": [
            item.get("symbol")
            for item in active_candidates
        ],
        "candidates": _compact_candidates(active_candidates),
        "final_candidates": _compact_candidates(active_candidates),
        "ready_candidates": _compact_candidates(ready_candidates),
        "symbols": symbols,
    }
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
            SELECT symbol, name, rank, composite_score AS score, raw_score,
                   expected_return, expected_risk, trading_cost, slippage_cost,
                   net_edge, composite_score, status, claimed_by_worker,
                   expires_at, decision, reason, current_price, change_rate,
                   volume, volume_ratio, turnover_value, news_count,
                   disclosure_count
            FROM scanner_candidates
            WHERE scan_id = ?
            ORDER BY rank ASC
            """,
            (run["scan_id"],),
        ).fetchall()
    raw = _parse_json(run["raw_json"], {})
    raw["candidates"] = [dict(row) for row in candidates]
    return raw


def get_ready_execution_candidates(
    *,
    db_path: Path | str | None = None,
    worker_id: str | None = None,
    limit: int | None = None,
    worker_hurdle_rate: float | None = None,
) -> list[dict[str, Any]]:
    path = _db_path(db_path)
    initialize_universe_db(path)
    limit = max(1, min(int(limit or settings.universe_scanner_final_limit or 10), 10))
    hurdle_rate = (
        float(settings.universe_scanner_worker_hurdle_rate_bps)
        if worker_hurdle_rate is None
        else float(worker_hurdle_rate)
    )
    now = _now()
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT *
            FROM scanner_candidates
            WHERE status = 'READY'
              AND net_edge > ?
              AND expires_at > ?
            ORDER BY net_edge DESC, composite_score DESC, rank ASC
            LIMIT ?
            """,
            (hurdle_rate, now, limit),
        ).fetchall()
        if worker_id and rows:
            conn.executemany(
                """
                UPDATE scanner_candidates
                SET status = 'CLAIMED', claimed_by_worker = ?
                WHERE id = ? AND status = 'READY'
                """,
                [(worker_id, row["id"]) for row in rows],
            )
    return [dict(row) for row in rows]


def get_active_scanner_candidates(
    *,
    db_path: Path | str | None = None,
    limit: int | None = None,
    include_expired: bool = True,
) -> list[dict[str, Any]]:
    path = _db_path(db_path)
    initialize_universe_db(path)
    limit = max(1, min(int(limit or settings.universe_scanner_final_limit or 10), 10))
    where = ""
    params: list[Any] = []
    if not include_expired:
        where = "WHERE expires_at > ?"
        params.append(_now())
    params.append(limit)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT *
            FROM scanner_candidates
            {where}
            ORDER BY rank ASC
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def scanner_candidate_to_symbol_config(
    req: AutoTradeStartRequest,
    candidate: dict[str, Any],
) -> AutoTradeSymbolConfig:
    return _to_symbol_config(req, candidate)


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
    symbol_interval_seconds = _scanner_symbol_interval_seconds()
    for index, (symbol, name) in enumerate(source_symbols.items()):
        if index and symbol_interval_seconds:
            time_module.sleep(symbol_interval_seconds)
        try:
            price_data = _fetch_scanner_price_data(symbol)
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


def _scanner_symbol_interval_seconds() -> float:
    configured = max(
        0.0,
        float(settings.universe_scanner_symbol_interval_seconds or 0.0),
    )
    cap = max(
        0.0,
        float(settings.universe_scanner_symbol_interval_cap_seconds or 0.0),
    )
    if cap <= 0:
        return configured
    return min(configured, cap)


def _fetch_scanner_price_data(symbol: str) -> dict[str, Any]:
    try:
        return fetch_price_data(
            symbol,
            include_intraday=bool(settings.universe_scanner_intraday_enrichment_enabled),
        )
    except TypeError as exc:
        if "include_intraday" not in str(exc):
            raise
        return fetch_price_data(symbol)


def _score_snapshot(item: dict[str, Any]) -> dict[str, Any]:
    price = _to_float(item.get("current_price"))
    fallback_price = _latest_close_from_daily(item)
    market_open = _is_kr_regular_market_open()
    change_rate = _to_float(item.get("change_rate"))
    volume_ratio = _to_float(item.get("volume_ratio"))
    turnover_value = _to_float(item.get("turnover_value"))
    volume = _to_int(item.get("volume"))
    technical = item.get("latest_technical_features") or {}
    score = 0.0
    reasons: list[str] = []

    if price and price > 0:
        score += 10
    elif fallback_price and fallback_price > 0 and not market_open:
        price = fallback_price
        item = {
            **item,
            "current_price": price,
            "price_source": "latest_close",
            "market_phase": "closed",
        }
        score += 25
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

    momentum_hits = 0
    for key, label, weight in (
        ("return_5d", "5d momentum", 12),
        ("return_20d", "20d momentum", 16),
        ("return_60d", "60d momentum", 14),
    ):
        value = _to_float(technical.get(key))
        if value is None:
            continue
        if value >= 0:
            momentum_hits += 1
            score += min(weight, value * 100 * 1.2 + weight * 0.35)
            reasons.append(label)
        elif value < -0.06:
            score -= weight * 0.7

    if technical.get("high_breakout_20d"):
        momentum_hits += 1
        score += 16
        reasons.append("20d high breakout")
    if technical.get("low_breakdown_20d"):
        score -= 24
        reasons.append("20d low breakdown")

    ma20_slope = _to_float(technical.get("ma20_slope"))
    if ma20_slope is not None:
        if ma20_slope > 0:
            score += 12
            reasons.append("positive MA20 slope")
        else:
            score -= 12

    atr_pct = _to_float(technical.get("atr_14_pct"))
    if atr_pct is not None:
        if 0.015 <= atr_pct <= 0.085:
            score += 8
            reasons.append("ATR risk band")
        elif atr_pct > 0.11:
            score -= 18
            reasons.append("ATR volatility too high")

    if momentum_hits == 0 and change_rate is not None:
        if -1.5 <= change_rate <= 7:
            score += 12
            reasons.append("intraday change fallback")
        elif change_rate < -4:
            score -= 16
            reasons.append("sharp drop")

    if volume_ratio is not None:
        if 1.5 <= volume_ratio <= 5:
            score += 12
            reasons.append("volume expansion")
        elif volume_ratio > 5:
            score += 4
            reasons.append("possible volume exhaustion")

    if turnover_value is not None and turnover_value > 0:
        score += min(12, turnover_value / 50_000_000_000 * 12)
        reasons.append("turnover present")
    elif volume:
        score += min(8, volume / 5_000_000 * 8)
        reasons.append("volume present")

    intraday = item.get("intraday") or {}
    minute_volume_ratio = _to_float(intraday.get("minute_volume_ratio"))
    if minute_volume_ratio is not None and minute_volume_ratio >= 1.5:
        score += min(5, minute_volume_ratio)
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

    if settings.universe_scanner_news_enrichment_enabled:
        try:
            news = search_naver_news(query=query, display=5, sort="date")
            news_count = len(news.get("items") or [])
            if news_count:
                score += min(5, news_count)
        except Exception as exc:
            enrichment_errors.append(f"news: {exc}")

    if settings.universe_scanner_disclosure_enrichment_enabled:
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


def _rank_execution_candidates(
    candidates: list[dict[str, Any]],
    *,
    scan_time: str,
    final_limit: int,
    hurdle_rate: float,
) -> list[dict[str, Any]]:
    expires_at = _plus_seconds(
        scan_time,
        int(settings.universe_scanner_candidate_ttl_seconds or 3600),
    )
    edge_model = load_edge_model()
    scored = [
        _with_expected_value_scores(
            item,
            expires_at=expires_at,
            edge_model=edge_model,
        )
        for item in candidates
    ]
    entry_gate = edge_entry_gate(scored)
    ranked = sorted(
        scored,
        key=lambda item: (
            float(item.get("composite_score") or 0),
            float(item.get("net_edge") or 0),
            float(item.get("raw_score") or 0),
        ),
        reverse=True,
    )
    prepared: list[dict[str, Any]] = []
    for rank, item in enumerate(ranked, start=1):
        status, reason = _execution_status_for_candidate(
            item,
            rank=rank,
            execution_limit=final_limit,
            hurdle_rate=hurdle_rate,
            entry_gate=entry_gate,
        )
        prepared.append(
            {
                **item,
                "rank": rank,
                "score": item["composite_score"],
                "status": status,
                "reason": reason,
                "worker_hurdle_rate": round(hurdle_rate, 4),
                "entry_gate": entry_gate,
            }
        )
    return prepared


def _with_expected_value_scores(
    candidate: dict[str, Any],
    *,
    expires_at: str,
    edge_model: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    raw_score = float(candidate.get("score") or 0)
    heuristic_return = _estimate_expected_return_bps(candidate, raw_score)
    heuristic_risk = _estimate_expected_risk_penalty_bps(candidate, raw_score)
    calibrated = estimate_expected_edges(candidate, raw_score, model=edge_model)
    edge_model_name = (calibrated or {}).get("edge_model", "heuristic_v1")
    expected_return = (
        max(float(calibrated["expected_return"]), heuristic_return * 0.65)
        if calibrated
        else heuristic_return
    )
    expected_risk = (
        min(float(calibrated["expected_risk"]), max(35.0, heuristic_risk * 1.10))
        if calibrated
        else heuristic_risk
    )
    if calibrated:
        edge_model_name = f"{edge_model_name}+heuristic_floor"
    quality = _swing_edge_quality(candidate, raw_score)
    reward_risk = _atr_reward_risk_estimate(candidate, raw_score, quality["score"])
    if reward_risk and quality["score"] >= 58:
        expected_return = max(
            expected_return,
            float(reward_risk["gross_return_floor_bps"]),
        )
        expected_risk = min(
            expected_risk,
            float(reward_risk["loss_risk_floor_bps"])
            + float(quality["risk_buffer_bps"]),
        )
        edge_model_name = f"{edge_model_name}+atr_rr"
    expected_return = min(
        500.0,
        expected_return + float(quality["return_uplift_bps"]),
    )
    expected_risk = max(
        25.0,
        min(
            500.0,
            expected_risk
            - float(quality["risk_discount_bps"])
            + float(quality["risk_penalty_bps"]),
        ),
    )
    trading_cost = _estimate_round_trip_trading_cost_bps()
    slippage_cost = max(
        4.0,
        _estimate_slippage_cost_bps(candidate)
        - float(quality["slippage_discount_bps"]),
    )
    net_edge = expected_return - expected_risk - trading_cost - slippage_cost
    expected_return_score = _score_bps(expected_return, cap_bps=350)
    net_edge_score = _score_bps(net_edge, cap_bps=250)
    expected_risk_score = _score_bps(expected_risk, cap_bps=350)
    composite_score = (
        raw_score * 0.25
        + expected_return_score * 0.30
        + net_edge_score * 0.35
        - expected_risk_score * 0.10
    )
    return {
        **candidate,
        "raw_score": round(raw_score, 4),
        "expected_return": round(expected_return, 4),
        "expected_risk": round(expected_risk, 4),
        "trading_cost": round(trading_cost, 4),
        "slippage_cost": round(slippage_cost, 4),
        "net_edge": round(net_edge, 4),
        "composite_score": round(max(0.0, min(100.0, composite_score)), 4),
        "expires_at": expires_at,
        "edge_model": edge_model_name,
        "edge_quality_score": quality["score"],
        "edge_quality_reasons": quality["reasons"],
        "edge_reward_risk": reward_risk,
    }


def _execution_status_for_candidate(
    candidate: dict[str, Any],
    *,
    rank: int,
    execution_limit: int,
    hurdle_rate: float,
    entry_gate: dict[str, Any],
) -> tuple[str, str]:
    reasons = [str(candidate.get("reason") or "")]
    if rank > execution_limit:
        reasons.append("archived below execution top 10")
        return "ARCHIVED", _join_reasons(reasons)
    if candidate.get("decision") == "exclude" or not candidate.get("current_price"):
        reasons.append("not executable")
        return "EXCLUDED", _join_reasons(reasons)
    if candidate.get("decision") != "buy_candidate":
        reasons.append("not a buy candidate")
        return "SKIPPED", _join_reasons(reasons)
    net_edge = float(candidate.get("net_edge") or 0)
    if net_edge <= hurdle_rate:
        reasons.append(
            f"net_edge {net_edge:.2f}bps is below worker hurdle {hurdle_rate:.2f}bps"
        )
        return "SKIPPED", _join_reasons(reasons)
    if not entry_gate.get("approved", False):
        reasons.append(str(entry_gate.get("message") or "entry gate blocked"))
        return "SKIPPED", _join_reasons(reasons)
    return "READY", _join_reasons(reasons)


def _estimate_expected_return_bps(candidate: dict[str, Any], raw_score: float) -> float:
    change_rate = _to_float(candidate.get("change_rate")) or 0.0
    volume_ratio = _to_float(candidate.get("volume_ratio")) or 0.0
    minute_volume_ratio = _to_float((candidate.get("intraday") or {}).get("minute_volume_ratio")) or 0.0
    news_count = _to_int(candidate.get("news_count")) or 0
    disclosure_count = _to_int(candidate.get("disclosure_count")) or 0
    technical = candidate.get("latest_technical_features") or {}
    return_5d = _to_float(technical.get("return_5d")) or 0.0
    return_20d = _to_float(technical.get("return_20d")) or 0.0
    return_60d = _to_float(technical.get("return_60d")) or 0.0

    score_component = raw_score * 1.8
    daily_momentum_component = max(
        -80.0,
        min(
            140.0,
            return_5d * 250.0
            + return_20d * 180.0
            + return_60d * 110.0,
        ),
    )
    breakout_component = 35.0 if technical.get("high_breakout_20d") else 0.0
    momentum_component = max(-25.0, min(45.0, change_rate * 4.0))
    volume_component = min(35.0, max(0.0, volume_ratio - 1.0) * 10.0)
    minute_component = min(10.0, max(0.0, minute_volume_ratio - 1.0) * 4.0)
    news_component = min(25.0, news_count * 5.0)
    disclosure_component = min(24.0, disclosure_count * 8.0)
    expected = (
        35.0
        + score_component
        + daily_momentum_component
        + breakout_component
        + momentum_component
        + volume_component
        + minute_component
        + news_component
        + disclosure_component
    )
    if candidate.get("price_source") == "latest_close":
        expected *= 0.55
    return max(0.0, min(500.0, expected))


def _estimate_expected_risk_penalty_bps(
    candidate: dict[str, Any],
    raw_score: float,
) -> float:
    change_rate = _to_float(candidate.get("change_rate")) or 0.0
    volume_ratio = _to_float(candidate.get("volume_ratio")) or 0.0
    price = _to_float(candidate.get("current_price"))
    levels = atr_exit_levels_from_price_data(entry_price=price, price_data=candidate)
    risk_per_share = _to_float(levels.get("risk_per_share"))
    risk = (
        max(55.0, min(420.0, (risk_per_share / price) * 10_000))
        if price and risk_per_share
        else 140.0
    )
    if raw_score < 55:
        risk += (55.0 - raw_score) * 2.0
    if change_rate > 7:
        risk += min(90.0, (change_rate - 7.0) * 18.0 + 35.0)
    if change_rate < -4:
        risk += min(100.0, abs(change_rate + 4.0) * 18.0 + 45.0)
    if volume_ratio > 5:
        risk += min(70.0, (volume_ratio - 5.0) * 12.0 + 25.0)
    if bool(candidate.get("overheated")):
        risk += 90.0
    if candidate.get("price_source") == "latest_close":
        risk += 35.0
    return max(0.0, min(500.0, risk))


def _estimate_round_trip_trading_cost_bps() -> float:
    commission = max(0.0, float(settings.commission_rate or 0.0)) * 2
    sell_tax = max(0.0, float(settings.kr_stock_sell_tax_rate or 0.0))
    return (commission + sell_tax) * 10_000


def _estimate_slippage_cost_bps(candidate: dict[str, Any]) -> float:
    base = (
        max(0.0, float(settings.universe_scanner_default_spread_bps or 0.0))
        + max(0.0, float(settings.universe_scanner_default_slippage_bps or 0.0))
    )
    volume_ratio = _to_float(candidate.get("volume_ratio"))
    turnover = _to_float(candidate.get("turnover_value"))
    if volume_ratio is not None and volume_ratio < 1:
        base += 8.0
    if turnover is not None and turnover < 20_000_000_000:
        base += 7.0
    return base


def _swing_edge_quality(candidate: dict[str, Any], raw_score: float) -> dict[str, Any]:
    technical = candidate.get("latest_technical_features") or {}
    change_rate = _to_float(candidate.get("change_rate")) or 0.0
    volume_ratio = _to_float(candidate.get("volume_ratio"))
    turnover = _to_float(candidate.get("turnover_value"))
    atr_pct = _to_float(technical.get("atr_14_pct"))
    reasons: list[str] = []
    score = max(0.0, min(25.0, raw_score * 0.25))

    return_5d = _to_float(technical.get("return_5d"))
    return_20d = _to_float(technical.get("return_20d"))
    return_60d = _to_float(technical.get("return_60d"))
    positive_momentum = 0
    for value, threshold, points, label in (
        (return_5d, 0.02, 10.0, "5d momentum"),
        (return_20d, 0.05, 14.0, "20d momentum"),
        (return_60d, 0.08, 12.0, "60d momentum"),
    ):
        if value is None:
            continue
        if value >= threshold:
            score += points
            positive_momentum += 1
            reasons.append(label)
        elif value < -threshold:
            score -= points * 0.8
    if positive_momentum >= 3:
        score += 10.0
        reasons.append("stacked momentum")

    if technical.get("high_breakout_20d"):
        score += 14.0
        reasons.append("20d high breakout")
    if technical.get("low_breakdown_20d"):
        score -= 28.0

    ma20_slope = _to_float(technical.get("ma20_slope"))
    if ma20_slope is not None:
        if ma20_slope > 0:
            score += 8.0
            reasons.append("positive MA slope")
        else:
            score -= 10.0

    if atr_pct is not None:
        if 0.015 <= atr_pct <= 0.055:
            score += 12.0
            reasons.append("ATR sweet spot")
        elif 0.055 < atr_pct <= 0.085:
            score += 5.0
            reasons.append("tradable ATR")
        elif atr_pct > 0.095:
            score -= 22.0
        elif atr_pct < 0.008:
            score -= 6.0

    if 0.0 <= change_rate <= 5.0:
        score += 6.0
    elif change_rate > 8.0:
        score -= 14.0
    elif change_rate < -3.0:
        score -= 16.0

    if volume_ratio is not None:
        if 1.3 <= volume_ratio <= 4.0:
            score += 8.0
            reasons.append("healthy volume expansion")
        elif volume_ratio > 5.5:
            score -= 10.0
        elif volume_ratio < 0.8:
            score -= 8.0

    if turnover is not None:
        if turnover >= 100_000_000_000:
            score += 10.0
            reasons.append("strong liquidity")
        elif turnover >= 50_000_000_000:
            score += 6.0
            reasons.append("liquid")
        elif turnover < 20_000_000_000:
            score -= 12.0

    if bool(candidate.get("overheated")):
        score -= 25.0

    score += _market_context_quality_points(candidate, reasons)
    score = round(max(0.0, min(100.0, score)), 4)
    return {
        "score": score,
        "reasons": reasons[:8],
        "return_uplift_bps": round(max(0.0, min(150.0, (score - 55.0) * 3.0)), 4),
        "risk_discount_bps": round(max(0.0, min(80.0, (score - 60.0) * 1.5)), 4),
        "risk_penalty_bps": round(max(0.0, min(100.0, (45.0 - score) * 2.0)), 4),
        "risk_buffer_bps": round(max(15.0, min(95.0, 95.0 - score * 0.7)), 4),
        "slippage_discount_bps": round(max(0.0, min(8.0, (score - 60.0) * 0.25)), 4),
    }


def _market_context_quality_points(
    candidate: dict[str, Any],
    reasons: list[str],
) -> float:
    context = _candidate_market_context(candidate)
    if not context:
        return 0.0
    points = 0.0
    regime = str(context.get("market_regime") or "unknown").lower()
    if regime in {"bull", "risk_on", "uptrend", "strong"}:
        points += 10.0
        reasons.append("supportive market regime")
    elif regime in {"bear", "risk_off", "downtrend", "stress"}:
        points -= 20.0

    risk_on_score = _to_float(context.get("risk_on_score"))
    if risk_on_score is not None:
        if risk_on_score >= 65:
            points += 10.0
            reasons.append("risk-on market")
        elif risk_on_score >= 55:
            points += 6.0
        elif risk_on_score < 35:
            points -= 15.0

    sector_score = _sector_strength_score(candidate, context)
    if sector_score is not None:
        if sector_score >= 65:
            points += 10.0
            reasons.append("sector relative strength")
        elif sector_score >= 55:
            points += 6.0
        elif sector_score <= 40:
            points -= 12.0
    return points


def _candidate_market_context(candidate: dict[str, Any]) -> dict[str, Any]:
    context = candidate.get("market_context") or candidate.get("market_context_snapshot")
    if isinstance(context, dict) and context:
        return context
    try:
        return get_latest_market_context() or {}
    except Exception:
        return {}


def _sector_strength_score(
    candidate: dict[str, Any],
    context: dict[str, Any],
) -> float | None:
    selected = context.get("selected_sector_relative_strength") or {}
    value = _to_float(selected.get("score"))
    if value is not None:
        return value
    sector = candidate.get("sector")
    if not sector:
        return None
    relative_strength = context.get("sector_relative_strength") or {}
    if not isinstance(relative_strength, dict):
        return None
    sector_rows = relative_strength.get("sectors") or []
    for row in sector_rows:
        if row.get("sector") == sector:
            return _to_float(row.get("score"))
    return None


def _atr_reward_risk_estimate(
    candidate: dict[str, Any],
    raw_score: float,
    quality_score: float,
) -> dict[str, float] | None:
    price = _to_float(candidate.get("current_price"))
    if not price or price <= 0:
        return None
    levels = atr_exit_levels_from_price_data(entry_price=price, price_data=candidate)
    risk_per_share = _to_float(levels.get("risk_per_share"))
    target = _to_float(levels.get("take_profit"))
    if not risk_per_share or risk_per_share <= 0 or not target or target <= price:
        return None
    risk_bps = risk_per_share / price * 10_000
    target_bps = (target - price) / price * 10_000
    win_probability = _expected_win_probability(candidate, raw_score, quality_score)
    gross_return_floor = win_probability * target_bps
    loss_risk_floor = (1.0 - win_probability) * risk_bps
    return {
        "win_probability": round(win_probability, 4),
        "target_bps": round(target_bps, 4),
        "risk_bps": round(risk_bps, 4),
        "gross_return_floor_bps": round(max(0.0, min(500.0, gross_return_floor)), 4),
        "loss_risk_floor_bps": round(max(25.0, min(500.0, loss_risk_floor)), 4),
    }


def _expected_win_probability(
    candidate: dict[str, Any],
    raw_score: float,
    quality_score: float,
) -> float:
    technical = candidate.get("latest_technical_features") or {}
    probability = 0.38 + max(0.0, min(1.0, raw_score / 100.0)) * 0.10
    probability += max(0.0, min(1.0, quality_score / 100.0)) * 0.14
    if technical.get("high_breakout_20d"):
        probability += 0.035
    if technical.get("low_breakdown_20d"):
        probability -= 0.06
    if bool(candidate.get("overheated")):
        probability -= 0.055
    return max(0.32, min(0.62, probability))


def _score_bps(value: float, *, cap_bps: float) -> float:
    if cap_bps <= 0:
        return 0.0
    return max(0.0, min(100.0, value / cap_bps * 100.0))


def _join_reasons(reasons: list[str]) -> str:
    return "; ".join(reason for reason in reasons if reason)


def _to_symbol_config(
    req: AutoTradeStartRequest,
    candidate: dict[str, Any],
) -> AutoTradeSymbolConfig:
    candidate_payload = _candidate_payload(candidate)
    price = _to_float(candidate_payload.get("current_price"))
    levels = atr_exit_levels_from_price_data(
        entry_price=price,
        price_data=candidate_payload,
    )
    stop_loss = levels["stop_loss"]
    take_profit = levels["take_profit"]
    trailing_stop = levels["trailing_stop"]
    risk_per_share = _to_float(levels.get("risk_per_share"))
    expected_loss_bps = (
        risk_per_share / price * 10_000
        if price and risk_per_share
        else _to_float(candidate.get("expected_risk"))
    )
    expected_win_bps = (
        (take_profit - price) / price * 10_000
        if price and take_profit
        else _to_float(candidate.get("expected_return"))
    )
    return AutoTradeSymbolConfig(
        symbol=str(candidate_payload["symbol"]),
        name=candidate_payload.get("name"),
        market="KR",
        strategy_type="swing",
        lookback_hours=24 * 10,
        risk_level="medium",
        requested_action="entry",
        price=price,
        decision_price=price,
        order_price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        trailing_stop=trailing_stop,
        signal_score=_to_float(candidate.get("composite_score") or candidate.get("score")),
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
        expected_gross_edge_bps=_to_float(candidate.get("expected_return")),
        expected_win_bps=expected_win_bps,
        expected_loss_bps=expected_loss_bps,
        spread_bps=max(0.0, float(settings.universe_scanner_default_spread_bps or 0.0)),
        slippage_bps=max(
            0.0,
            float(settings.universe_scanner_default_slippage_bps or 0.0),
        ),
        expected_holding_days=5.0,
    )


def _candidate_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    raw = _parse_json(candidate.get("raw_json"), {})
    return {**raw, **candidate} if isinstance(raw, dict) else candidate


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
                "rank": item.get("rank", index),
                "symbol": item.get("symbol"),
                "name": item.get("name"),
                "score": item.get("score"),
                "raw_score": item.get("raw_score"),
                "expected_return": item.get("expected_return"),
                "expected_risk": item.get("expected_risk"),
                "trading_cost": item.get("trading_cost"),
                "slippage_cost": item.get("slippage_cost"),
                "net_edge": item.get("net_edge"),
                "composite_score": item.get("composite_score"),
                "decision": item.get("decision"),
                "status": item.get("status"),
                "reason": item.get("reason"),
                "current_price": item.get("current_price"),
                "change_rate": item.get("change_rate"),
                "volume": item.get("volume"),
                "volume_ratio": item.get("volume_ratio"),
                "turnover_value": item.get("turnover_value"),
                "news_count": item.get("news_count", 0),
                "disclosure_count": item.get("disclosure_count", 0),
                "edge_model": item.get("edge_model"),
                "edge_quality_score": item.get("edge_quality_score"),
                "edge_quality_reasons": item.get("edge_quality_reasons"),
                "edge_reward_risk": item.get("edge_reward_risk"),
                "claimed_by_worker": item.get("claimed_by_worker"),
                "expires_at": item.get("expires_at"),
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
                    _to_int(item.get("rank")) or index,
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


def _record_scanner_candidates(
    *,
    path: Path,
    scan_id: str,
    scan_time: str,
    candidates: list[dict[str, Any]],
    execution_limit: int,
) -> None:
    active_rows = [
        item
        for item in candidates
        if int(item.get("rank") or 0) <= execution_limit
    ]
    with sqlite3.connect(path) as conn:
        conn.execute("DELETE FROM scanner_candidates")
        conn.executemany(
            """
            INSERT INTO scanner_candidate_history (
                scan_id, scan_time, symbol, name, raw_score, expected_return,
                expected_risk, trading_cost, slippage_cost, net_edge,
                composite_score, rank, reason, status, decision, current_price,
                change_rate, volume, volume_ratio, turnover_value, news_count,
                disclosure_count, claimed_by_worker, expires_at, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                _candidate_row_values(
                    scan_id=scan_id,
                    scan_time=scan_time,
                    item=item if int(item.get("rank") or 0) <= execution_limit else {
                        **item,
                        "status": "ARCHIVED",
                    },
                )
                for item in candidates
            ],
        )
        conn.executemany(
            """
            INSERT INTO scanner_candidates (
                scan_id, scan_time, symbol, name, raw_score, expected_return,
                expected_risk, trading_cost, slippage_cost, net_edge,
                composite_score, rank, reason, status, decision, current_price,
                change_rate, volume, volume_ratio, turnover_value, news_count,
                disclosure_count, claimed_by_worker, expires_at, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                _candidate_row_values(
                    scan_id=scan_id,
                    scan_time=scan_time,
                    item=item,
                )
                for item in active_rows
            ],
        )


def _candidate_row_values(
    *,
    scan_id: str,
    scan_time: str,
    item: dict[str, Any],
) -> tuple[Any, ...]:
    return (
        scan_id,
        scan_time,
        item.get("symbol"),
        item.get("name"),
        _to_float(item.get("raw_score")) or 0.0,
        _to_float(item.get("expected_return")) or 0.0,
        _to_float(item.get("expected_risk")) or 0.0,
        _to_float(item.get("trading_cost")) or 0.0,
        _to_float(item.get("slippage_cost")) or 0.0,
        _to_float(item.get("net_edge")) or 0.0,
        _to_float(item.get("composite_score")) or 0.0,
        _to_int(item.get("rank")) or 0,
        item.get("reason", ""),
        item.get("status", "SKIPPED"),
        item.get("decision", "exclude"),
        _to_float(item.get("current_price")),
        _to_float(item.get("change_rate")),
        _to_int(item.get("volume")),
        _to_float(item.get("volume_ratio")),
        _to_float(item.get("turnover_value")),
        _to_int(item.get("news_count")) or 0,
        _to_int(item.get("disclosure_count")) or 0,
        item.get("claimed_by_worker"),
        item.get("expires_at"),
        _json(item),
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


def _plus_seconds(value: str, seconds: int | float) -> str:
    return (
        datetime.fromisoformat(value) + timedelta(seconds=float(seconds))
    ).isoformat(timespec="seconds")


def _worker_hurdle_rate_bps(execution_mode: str = "paper") -> float:
    configured = max(
        0.0,
        float(settings.universe_scanner_worker_hurdle_rate_bps or 0.0),
    )
    if execution_mode != "paper":
        return configured
    achieved = _paper_average_realized_return_bps()
    if achieved is None:
        return configured
    return max(configured, achieved)


def _paper_average_realized_return_bps(limit: int = 20) -> float | None:
    path = settings.storage_path(paper_trading.DEFAULT_DB_PATH)
    if not path.exists():
        return None
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT requested_amount, amount, net_realized_pnl, realized_pnl
                FROM paper_orders
                WHERE side = 'SELL'
                  AND status IN ('FILLED', 'PARTIALLY_FILLED')
                ORDER BY id DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
    except sqlite3.Error:
        return None
    returns: list[float] = []
    for row in rows:
        amount = _to_float(row["requested_amount"]) or _to_float(row["amount"])
        pnl = _to_float(row["net_realized_pnl"]) or _to_float(row["realized_pnl"])
        if amount and pnl is not None:
            returns.append(float(pnl) / float(amount) * 10_000)
    if not returns:
        return None
    return sum(returns) / len(returns)


def _db_path(db_path: Path | str | None = None) -> Path:
    return settings.storage_path(db_path or settings.universe_scanner_db_path)
