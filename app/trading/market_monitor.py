from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import json
import re
import sqlite3
import time
from typing import Any

from app.config import settings
from app.data_sources.kis import fetch_price_data
from app.data_sources.opendart import fetch_opendart_disclosures
from app.scoring import classify_text_impact
from app.services.naver_news import search_naver_news
from app.trading import alerting, auto_trading_store, broker_sync


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS monitor_jobs (
    name TEXT PRIMARY KEY,
    interval_seconds INTEGER NOT NULL,
    last_run_at TEXT,
    next_run_at TEXT NOT NULL,
    last_status TEXT,
    last_error TEXT,
    last_result_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS monitor_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    job_name TEXT NOT NULL,
    symbol TEXT,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    current_price REAL,
    change_rate REAL,
    volume_ratio REAL,
    raw_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS monitor_news (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    query TEXT NOT NULL,
    dedupe_key TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL,
    published_at TEXT,
    title TEXT NOT NULL,
    url TEXT,
    impact_direction TEXT NOT NULL,
    impact_strength REAL NOT NULL,
    raw_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS monitor_disclosures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    dedupe_key TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL,
    published_at TEXT,
    title TEXT NOT NULL,
    url TEXT,
    impact_direction TEXT NOT NULL,
    impact_strength REAL NOT NULL,
    raw_json TEXT NOT NULL
);
"""


JOB_KIS_MARKET = "kis_market_watch"
JOB_OPENDART = "opendart_disclosures"
JOB_NAVER_NEWS = "naver_news"


def initialize_monitor_db(db_path: Path | str | None = None) -> None:
    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
        _ensure_default_jobs(conn)


def get_monitor_status(db_path: Path | str | None = None) -> dict[str, Any]:
    path = _db_path(db_path)
    if not path.exists():
        initialize_monitor_db(path)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_SQL)
        _ensure_default_jobs(conn)
        rows = conn.execute(
            """
            SELECT *
            FROM monitor_jobs
            ORDER BY next_run_at ASC, name ASC
            """
        ).fetchall()
    return {
        "enabled": settings.market_monitor_enabled,
        "db_path": str(path),
        "jobs": [_job_to_dict(row) for row in rows],
        "watchlist": list(_resolve_watchlist().keys()),
        "market_keywords": _market_keywords(),
        "uses_supabase": False,
        "database": "sqlite",
    }


def process_due_monitor_jobs(
    db_path: Path | str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Run due 1m/5m/10m monitor jobs once."""
    if not settings.market_monitor_enabled:
        return []

    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = _now()
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_SQL)
        _ensure_default_jobs(conn)
        rows = conn.execute(
            """
            SELECT *
            FROM monitor_jobs
            WHERE next_run_at <= ?
            ORDER BY next_run_at ASC, name ASC
            LIMIT ?
            """,
            (now, max(1, int(limit))),
        ).fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        name = row["name"]
        try:
            result = run_monitor_job(name, db_path=path)
            _complete_job(path, name, "success", result)
            results.append({"job": name, "status": "success", "result": result})
        except Exception as exc:
            result = {"status": "error", "message": str(exc)}
            _complete_job(path, name, "error", result, error=str(exc))
            results.append({"job": name, "status": "error", "error": str(exc)})
    return results


def run_monitor_job(
    name: str,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    if name == JOB_KIS_MARKET:
        return run_kis_market_watch(db_path=db_path)
    if name == JOB_OPENDART:
        return run_opendart_disclosure_watch(db_path=db_path)
    if name == JOB_NAVER_NEWS:
        return run_naver_news_watch(db_path=db_path)
    raise ValueError(f"Unknown monitor job: {name}")


def run_monitor_worker_forever(
    poll_seconds: float | None = None,
    db_path: Path | str | None = None,
) -> None:
    poll_seconds = (
        settings.auto_trading_worker_poll_seconds
        if poll_seconds is None
        else poll_seconds
    )
    initialize_monitor_db(db_path)
    while True:
        process_due_monitor_jobs(db_path=db_path)
        time.sleep(float(poll_seconds))


def run_kis_market_watch(db_path: Path | str | None = None) -> dict[str, Any]:
    """Every minute: KIS price/volume/change and position stop/take checks."""
    path = _db_path(db_path)
    watchlist = _resolve_watchlist()
    positions = _load_broker_positions()
    broker_snapshot = _latest_broker_sync_status()
    for position in positions.values():
        watchlist.setdefault(
            position["symbol"],
            {
                "symbol": position["symbol"],
                "name": position.get("name"),
                "stop_loss": None,
                "take_profit": None,
            },
        )

    alerts: list[dict[str, Any]] = []
    checked_symbols: list[str] = []
    for symbol, config in watchlist.items():
        price_data = fetch_price_data(symbol)
        checked_symbols.append(symbol)
        alerts.extend(_market_alerts(symbol=symbol, config=config, price_data=price_data))
        position = positions.get(symbol)
        if position:
            alerts.extend(
                _position_risk_alerts(
                    symbol=symbol,
                    config=config,
                    price_data=price_data,
                    position=position,
                )
            )

    inserted_alerts = _record_alerts(path, alerts)
    return {
        "status": "success",
        "job": JOB_KIS_MARKET,
        "checked_symbol_count": len(checked_symbols),
        "checked_symbols": checked_symbols,
        "alert_count": len(inserted_alerts),
        "alerts": inserted_alerts,
        "broker_sync": broker_snapshot,
    }


def run_naver_news_watch(db_path: Path | str | None = None) -> dict[str, Any]:
    """Every ten minutes: NAVER news, market keywords, and dedupe."""
    path = _db_path(db_path)
    watchlist = _resolve_watchlist()
    symbol_queries = [
        config.get("name") or symbol
        for symbol, config in watchlist.items()
    ]
    queries = _dedupe_strings(symbol_queries + _market_keywords())

    inserted: list[dict[str, Any]] = []
    query_results: list[dict[str, Any]] = []
    for query in queries:
        response = search_naver_news(
            query=query,
            display=max(1, min(100, int(settings.monitor_news_display))),
            sort="date",
        )
        items = _news_events_from_response(query, response)
        new_rows = _record_monitor_news(path, query, items)
        inserted.extend(new_rows)
        query_results.append(
            {
                "query": query,
                "connected": response.get("connected"),
                "item_count": len(response.get("items") or []),
                "new_count": len(new_rows),
            }
        )

    _notify_event_alerts(inserted, source=JOB_NAVER_NEWS)
    return {
        "status": "success",
        "job": JOB_NAVER_NEWS,
        "query_count": len(queries),
        "new_count": len(inserted),
        "queries": query_results,
        "new_items": inserted[:20],
    }


def run_opendart_disclosure_watch(db_path: Path | str | None = None) -> dict[str, Any]:
    """Every five minutes: latest OpenDART disclosures filtered to watchlist."""
    path = _db_path(db_path)
    watchlist = _resolve_watchlist()
    inserted: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for symbol in watchlist:
        disclosures = fetch_opendart_disclosures(symbol=symbol, lookback_hours=24)
        new_rows = _record_monitor_disclosures(path, symbol, disclosures)
        inserted.extend(new_rows)
        results.append(
            {
                "symbol": symbol,
                "fetched_count": len(disclosures),
                "new_count": len(new_rows),
            }
        )
    _notify_event_alerts(inserted, source=JOB_OPENDART)
    return {
        "status": "success",
        "job": JOB_OPENDART,
        "symbol_count": len(watchlist),
        "new_count": len(inserted),
        "symbols": results,
        "new_items": inserted[:20],
    }


def _resolve_watchlist() -> dict[str, dict[str, Any]]:
    symbols: dict[str, dict[str, Any]] = {}
    for raw_symbol in _comma_list(settings.monitor_watchlist_symbols):
        symbols[_normalize_symbol(raw_symbol)] = {
            "symbol": _normalize_symbol(raw_symbol),
            "name": None,
            "stop_loss": None,
            "take_profit": None,
        }

    for session in auto_trading_store.list_sessions(status="active", limit=500):
        payload = session.get("request_payload") or {}
        for item in payload.get("symbols") or []:
            symbol = _normalize_symbol(item.get("symbol"))
            if not symbol:
                continue
            symbols[symbol] = {
                "symbol": symbol,
                "name": item.get("name"),
                "stop_loss": item.get("stop_loss"),
                "take_profit": item.get("take_profit"),
            }
    return symbols


def _market_alerts(
    symbol: str,
    config: dict[str, Any],
    price_data: dict[str, Any],
) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    change_rate = _to_float(price_data.get("change_rate"))
    volume_ratio = _to_float(price_data.get("volume_ratio"))
    current_price = _to_float(price_data.get("current_price"))

    if change_rate is not None and change_rate >= settings.monitor_surge_change_pct:
        alerts.append(
            _alert(
                symbol=symbol,
                alert_type="price_surge",
                severity="high",
                message=f"{symbol} 급등 감지: 등락률 {change_rate:.2f}%",
                current_price=current_price,
                change_rate=change_rate,
                volume_ratio=volume_ratio,
                raw=price_data,
            )
        )
    if change_rate is not None and change_rate <= settings.monitor_drop_change_pct:
        alerts.append(
            _alert(
                symbol=symbol,
                alert_type="price_drop",
                severity="high",
                message=f"{symbol} 급락 감지: 등락률 {change_rate:.2f}%",
                current_price=current_price,
                change_rate=change_rate,
                volume_ratio=volume_ratio,
                raw=price_data,
            )
        )
    if volume_ratio is not None and volume_ratio >= settings.monitor_volume_spike_ratio:
        alerts.append(
            _alert(
                symbol=symbol,
                alert_type="volume_spike",
                severity="medium",
                message=f"{symbol} 거래량 폭증 감지: 평균 대비 {volume_ratio:.2f}배",
                current_price=current_price,
                change_rate=change_rate,
                volume_ratio=volume_ratio,
                raw=price_data,
            )
        )
    return alerts


def _position_risk_alerts(
    symbol: str,
    config: dict[str, Any],
    price_data: dict[str, Any],
    position: dict[str, Any],
) -> list[dict[str, Any]]:
    current_price = _to_float(price_data.get("current_price"))
    if current_price is None:
        return []

    avg_price = _to_float(position.get("avg_price"))
    stop_loss = _to_float(config.get("stop_loss"))
    take_profit = _to_float(config.get("take_profit"))
    if stop_loss is None and avg_price:
        stop_loss = avg_price * (1 - settings.monitor_default_stop_loss_pct / 100)
    if take_profit is None and avg_price:
        take_profit = avg_price * (1 + settings.monitor_default_take_profit_pct / 100)

    alerts: list[dict[str, Any]] = []
    if stop_loss and current_price <= stop_loss:
        alerts.append(
            _alert(
                symbol=symbol,
                alert_type="stop_loss_hit",
                severity="critical",
                message=f"{symbol} 손절 조건 도달: 현재가 {current_price:.2f}, 기준 {stop_loss:.2f}",
                current_price=current_price,
                change_rate=_to_float(price_data.get("change_rate")),
                volume_ratio=_to_float(price_data.get("volume_ratio")),
                raw={"price_data": price_data, "position": position, "stop_loss": stop_loss},
            )
        )
    if take_profit and current_price >= take_profit:
        alerts.append(
            _alert(
                symbol=symbol,
                alert_type="take_profit_hit",
                severity="high",
                message=f"{symbol} 익절 조건 도달: 현재가 {current_price:.2f}, 기준 {take_profit:.2f}",
                current_price=current_price,
                change_rate=_to_float(price_data.get("change_rate")),
                volume_ratio=_to_float(price_data.get("volume_ratio")),
                raw={"price_data": price_data, "position": position, "take_profit": take_profit},
            )
        )
    return alerts


def _load_broker_positions() -> dict[str, dict[str, Any]]:
    path = Path(settings.broker_sync_db_path)
    if not path.exists():
        return {}
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            conn.executescript(broker_sync.SCHEMA_SQL)
            rows = conn.execute(
                """
                SELECT symbol, name, quantity, avg_price, current_price, pnl, synced_at
                FROM broker_positions
                WHERE quantity > 0
                """
            ).fetchall()
    except sqlite3.Error:
        return {}
    return {row["symbol"]: dict(row) for row in rows}


def _latest_broker_sync_status() -> dict[str, Any]:
    path = Path(settings.broker_sync_db_path)
    if not path.exists():
        return {
            "status": "not_synced",
            "message": "Run app.trading.broker_sync_worker separately.",
        }
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            conn.executescript(broker_sync.SCHEMA_SQL)
            row = conn.execute(
                """
                SELECT created_at, account_no, total_cash, total_value
                FROM broker_balance_snapshots
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
    except sqlite3.Error as exc:
        return {"status": "error", "message": str(exc)}
    if not row:
        return {
            "status": "not_synced",
            "message": "No broker balance snapshot found.",
        }
    return {
        "status": "synced",
        "synced_at": row["created_at"],
        "account_no": row["account_no"],
        "total_cash": row["total_cash"],
        "total_value": row["total_value"],
    }


def _news_events_from_response(
    query: str,
    response: dict[str, Any],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for item in response.get("items") or []:
        title = item.get("title") or ""
        description = item.get("description") or ""
        impact = classify_text_impact(f"{title} {description}")
        events.append(
            {
                "query": query,
                "source": response.get("source", "Naver Search API"),
                "published_at": item.get("pubDate"),
                "title": title,
                "summary": description,
                "url": item.get("originallink") or item.get("link"),
                "impact_direction": impact["direction"],
                "impact_strength": impact["strength"],
                "raw": item,
            }
        )
    return events


def _notify_event_alerts(events: list[dict[str, Any]], source: str) -> None:
    alerts: list[dict[str, Any]] = []
    for event in events:
        impact_strength = abs(float(event.get("impact_strength") or 0))
        if impact_strength < settings.alert_min_impact_strength:
            continue
        severity = "critical" if impact_strength >= 90 else "high"
        alerts.append(
            {
                "severity": severity,
                "alert_type": "disclosure_event"
                if source == JOB_OPENDART
                else "news_event",
                "symbol": event.get("symbol"),
                "title": event.get("title"),
                "message": event.get("title") or "New market event",
                "impact_strength": impact_strength,
                "raw": event,
            }
        )
    alerting.notify_alerts(alerts, source=source)


def _record_alerts(
    db_path: Path,
    alerts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not alerts:
        return []
    now = _now()
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.executemany(
            """
            INSERT INTO monitor_alerts (
                created_at, job_name, symbol, alert_type, severity, message,
                current_price, change_rate, volume_ratio, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    now,
                    alert["job_name"],
                    alert.get("symbol"),
                    alert["alert_type"],
                    alert["severity"],
                    alert["message"],
                    alert.get("current_price"),
                    alert.get("change_rate"),
                    alert.get("volume_ratio"),
                    _json(alert.get("raw", alert)),
                )
                for alert in alerts
            ],
        )
    alerting.notify_alerts(alerts, source=JOB_KIS_MARKET)
    return alerts


def _record_monitor_news(
    db_path: Path,
    query: str,
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    inserted: list[dict[str, Any]] = []
    now = _now()
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        for event in events:
            key = _dedupe_key(event.get("url"), event.get("title"))
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO monitor_news (
                    created_at, query, dedupe_key, source, published_at, title,
                    url, impact_direction, impact_strength, raw_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    query,
                    key,
                    event.get("source", "Naver Search API"),
                    event.get("published_at"),
                    event.get("title", ""),
                    event.get("url"),
                    event.get("impact_direction", "uncertain"),
                    event.get("impact_strength", 0),
                    _json(event),
                ),
            )
            if cursor.rowcount:
                inserted.append({**event, "query": query, "dedupe_key": key})
    return inserted


def _record_monitor_disclosures(
    db_path: Path,
    symbol: str,
    disclosures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    inserted: list[dict[str, Any]] = []
    now = _now()
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        for event in disclosures:
            raw = event.get("raw") or {}
            key = _dedupe_key(raw.get("rcept_no") or event.get("url"), event.get("title"))
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO monitor_disclosures (
                    created_at, symbol, dedupe_key, source, published_at, title,
                    url, impact_direction, impact_strength, raw_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    symbol,
                    key,
                    event.get("source", "OpenDART"),
                    event.get("date") or event.get("published_at"),
                    event.get("title", ""),
                    event.get("url"),
                    event.get("impact_direction", "uncertain"),
                    event.get("impact_strength", 0),
                    _json(event),
                ),
            )
            if cursor.rowcount:
                inserted.append({**event, "symbol": symbol, "dedupe_key": key})
    return inserted


def _ensure_default_jobs(conn: sqlite3.Connection) -> None:
    now = _now()
    defaults = {
        JOB_KIS_MARKET: settings.monitor_price_interval_seconds,
        JOB_OPENDART: settings.monitor_disclosure_interval_seconds,
        JOB_NAVER_NEWS: settings.monitor_news_interval_seconds,
    }
    for name, interval in defaults.items():
        conn.execute(
            """
            INSERT INTO monitor_jobs (
                name, interval_seconds, last_run_at, next_run_at, updated_at
            )
            VALUES (?, ?, NULL, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                interval_seconds = excluded.interval_seconds
            """,
            (name, int(interval), now, now),
        )


def _complete_job(
    db_path: Path,
    name: str,
    status: str,
    result: dict[str, Any],
    error: str | None = None,
) -> None:
    now = _now_dt()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_SQL)
        row = conn.execute(
            "SELECT interval_seconds FROM monitor_jobs WHERE name = ?",
            (name,),
        ).fetchone()
        interval = int(row["interval_seconds"]) if row else 60
        conn.execute(
            """
            UPDATE monitor_jobs
            SET last_run_at = ?, next_run_at = ?, last_status = ?,
                last_error = ?, last_result_json = ?, updated_at = ?
            WHERE name = ?
            """,
            (
                _format_dt(now),
                _format_dt(now + timedelta(seconds=interval)),
                status,
                error,
                _json(result),
                _format_dt(now),
                name,
            ),
        )


def _job_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["last_result"] = _parse_json(data.pop("last_result_json", None), {})
    return data


def _alert(
    symbol: str,
    alert_type: str,
    severity: str,
    message: str,
    current_price: float | None,
    change_rate: float | None,
    volume_ratio: float | None,
    raw: dict[str, Any],
) -> dict[str, Any]:
    return {
        "job_name": JOB_KIS_MARKET,
        "symbol": symbol,
        "alert_type": alert_type,
        "severity": severity,
        "message": message,
        "current_price": current_price,
        "change_rate": change_rate,
        "volume_ratio": volume_ratio,
        "raw": raw,
    }


def _market_keywords() -> list[str]:
    return _comma_list(settings.monitor_market_keywords)


def _comma_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = re.sub(r"\s+", " ", value.strip()).lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(value.strip())
    return result


def _normalize_symbol(value: Any) -> str:
    return str(value or "").strip()


def _dedupe_key(*values: Any) -> str:
    joined = "|".join(
        re.sub(r"\s+", " ", str(value or "").strip().lower())
        for value in values
        if value
    )
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _to_float(value: Any) -> float | None:
    if value in (None, "", "-"):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _parse_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _db_path(db_path: Path | str | None = None) -> Path:
    return Path(db_path or settings.market_monitor_db_path)


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _now_dt() -> datetime:
    return datetime.now()


def _now() -> str:
    return _format_dt(_now_dt())


def _format_dt(value: datetime) -> str:
    return value.isoformat(timespec="seconds")
