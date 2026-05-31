from __future__ import annotations

from datetime import date, datetime, timedelta
import json
from pathlib import Path
import sqlite3
from typing import Any

from app.config import settings


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS corporate_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    market TEXT,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    event_date TEXT NOT NULL,
    effective_date TEXT,
    expires_at TEXT,
    title TEXT,
    reason TEXT,
    source TEXT,
    updated_at TEXT NOT NULL,
    raw_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_corporate_events_symbol_date
ON corporate_events(symbol, event_date);

CREATE INDEX IF NOT EXISTS idx_corporate_events_updated
ON corporate_events(updated_at);

CREATE TABLE IF NOT EXISTS corporate_event_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


SEVERE_EVENT_TYPES = {
    "capital_reduction",
    "delisting_review",
    "trading_halt_related",
    "liquidation",
    "merger",
    "spinoff",
}

WARNING_EVENT_TYPES = {
    "paid_in_capital_increase",
    "rights_offering",
    "ex_rights",
    "convertible_bond",
    "bond_with_warrant",
    "block_deal",
    "stock_split",
    "bonus_issue",
}

INFO_EVENT_TYPES = {
    "earnings",
    "dividend",
    "general_disclosure",
}


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _today() -> date:
    return datetime.utcnow().date()


def _db_path(db_path: Path | str | None = None) -> Path:
    return settings.storage_path(db_path or settings.corporate_event_db_path)


def initialize_corporate_event_db(db_path: Path | str | None = None) -> None:
    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)


def _normalize_symbol(value: Any) -> str:
    raw = str(value or "").strip()
    return raw.zfill(6) if raw.isdigit() else raw


def _normalize_event_type(value: Any) -> str:
    raw = str(value or "").strip().lower()

    mapping = {
        "감자": "capital_reduction",
        "유상증자": "paid_in_capital_increase",
        "무상증자": "bonus_issue",
        "주주배정": "rights_offering",
        "권리락": "ex_rights",
        "액면분할": "stock_split",
        "주식분할": "stock_split",
        "합병": "merger",
        "분할": "spinoff",
        "회사분할": "spinoff",
        "cb": "convertible_bond",
        "전환사채": "convertible_bond",
        "bw": "bond_with_warrant",
        "신주인수권부사채": "bond_with_warrant",
        "블록딜": "block_deal",
        "대량매매": "block_deal",
        "실적": "earnings",
        "실적발표": "earnings",
        "상장적격성": "delisting_review",
        "상장적격성 실질심사": "delisting_review",
        "상장적격성실질심사": "delisting_review",
        "정리매매": "liquidation",
    }

    return mapping.get(raw, raw.replace(" ", "_"))


def _severity_for_event(event_type: str) -> str:
    if event_type in SEVERE_EVENT_TYPES:
        return "severe"
    if event_type in WARNING_EVENT_TYPES:
        return "warning"
    if event_type in INFO_EVENT_TYPES:
        return "info"
    return "info"


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    try:
        raw = str(value).strip()
        if not raw:
            return None
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except Exception:
        try:
            return datetime.strptime(str(value), "%Y-%m-%d").date()
        except Exception:
            return None


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        raw = str(value).strip()
        if not raw:
            return None
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def upsert_corporate_events(
    events: list[dict[str, Any]],
    *,
    source: str = "manual",
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Upsert corporate event cache from manual upload or a low-frequency source adapter."""
    initialize_corporate_event_db(db_path)
    path = _db_path(db_path)

    now = _now()
    inserted = 0
    skipped = 0

    with sqlite3.connect(path) as conn:
        for item in events:
            if not isinstance(item, dict):
                skipped += 1
                continue

            symbol = _normalize_symbol(item.get("symbol"))
            event_date = _parse_date(item.get("event_date"))

            if not symbol or event_date is None:
                skipped += 1
                continue

            event_type = _normalize_event_type(item.get("event_type"))
            severity = str(item.get("severity") or _severity_for_event(event_type))

            conn.execute(
                """
                INSERT INTO corporate_events (
                    symbol, market, event_type, severity, event_date,
                    effective_date, expires_at, title, reason, source,
                    updated_at, raw_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    item.get("market"),
                    event_type,
                    severity,
                    event_date.isoformat(),
                    item.get("effective_date"),
                    item.get("expires_at"),
                    item.get("title"),
                    item.get("reason"),
                    source,
                    now,
                    json.dumps(item.get("raw_json") or item, ensure_ascii=False, default=str),
                ),
            )
            inserted += 1

        conn.execute(
            """
            INSERT INTO corporate_event_meta (key, value, updated_at)
            VALUES ('last_refresh_at', ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (now, now),
        )

    return {
        "status": "success",
        "inserted": inserted,
        "skipped": skipped,
        "source": source,
        "updated_at": now,
    }


def corporate_event_cache_status(
    *,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    if not bool(settings.corporate_event_cache_enabled):
        return {"status": "disabled", "enabled": False}

    path = _db_path(db_path)
    if not path.exists():
        return {"status": "missing", "enabled": True}

    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            count = conn.execute(
                "SELECT COUNT(*) FROM corporate_events"
            ).fetchone()[0]
            row = conn.execute(
                "SELECT value FROM corporate_event_meta WHERE key = 'last_refresh_at'"
            ).fetchone()
    except sqlite3.Error as exc:
        return {"status": "error", "enabled": True, "message": str(exc)}

    last_refresh = str(row["value"]) if row else None
    stale = True
    age_seconds = None

    if last_refresh:
        dt = _parse_datetime(last_refresh)
        if dt is not None:
            age_seconds = (datetime.utcnow() - dt).total_seconds()
            stale = age_seconds > float(settings.corporate_event_cache_ttl_seconds or 86400)

    return {
        "status": "ready",
        "enabled": True,
        "event_count": int(count or 0),
        "last_refresh_at": last_refresh,
        "age_seconds": round(age_seconds, 3) if age_seconds is not None else None,
        "stale": stale,
    }


def get_relevant_corporate_events(
    symbol: str,
    *,
    reference_date: date | None = None,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    if not bool(settings.corporate_event_cache_enabled):
        return []

    symbol = _normalize_symbol(symbol)
    if not symbol:
        return []

    path = _db_path(db_path)
    if not path.exists():
        return []

    reference_date = reference_date or _today()
    pre_days = int(settings.corporate_event_pre_event_window_days or 3)
    post_days = int(settings.corporate_event_post_event_window_days or 2)
    earnings_days = int(settings.corporate_event_earnings_window_days or 1)
    max_days = max(pre_days, post_days, earnings_days)
    start_date = reference_date - timedelta(days=max_days)
    end_date = reference_date + timedelta(days=max_days)

    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT *
                FROM corporate_events
                WHERE symbol = ?
                  AND event_date >= ?
                  AND event_date <= ?
                ORDER BY event_date ASC, severity DESC, id ASC
                """,
                (symbol, start_date.isoformat(), end_date.isoformat()),
            ).fetchall()
    except sqlite3.Error:
        return []

    events: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)

        expires_at = _parse_datetime(item.get("expires_at"))
        if expires_at is not None and datetime.utcnow() > expires_at:
            continue

        event_date = _parse_date(item.get("event_date"))
        if event_date is None:
            continue

        event_type = str(item.get("event_type") or "").lower()
        before_days = earnings_days if event_type == "earnings" else pre_days
        after_days = earnings_days if event_type == "earnings" else post_days
        if not (
            reference_date - timedelta(days=before_days)
            <= event_date
            <= reference_date + timedelta(days=after_days)
        ):
            continue

        events.append(item)

    return events


def corporate_event_check(
    symbol: str,
    *,
    execution_mode: str = "paper",
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    if not bool(settings.corporate_event_cache_enabled):
        return {
            "enabled": False,
            "approved": True,
            "block": False,
            "penalty_bps": 0.0,
            "events": [],
            "message": "corporate event cache disabled",
        }

    paper = str(execution_mode or "paper").lower() == "paper"
    cache = corporate_event_cache_status(db_path=db_path)

    if cache.get("status") in {"missing", "error"} or cache.get("stale") is True:
        penalty = 0.0 if paper else float(settings.corporate_event_missing_cache_live_penalty_bps or 0.0)
        message = (
            "corporate event cache stale; neutral or mild penalty"
            if cache.get("stale") is True
            else "corporate event cache unavailable; neutral or mild penalty"
        )
        return {
            "enabled": True,
            "approved": True,
            "block": False,
            "penalty_bps": round(penalty, 4),
            "events": [],
            "cache": cache,
            "message": message,
        }

    events = get_relevant_corporate_events(symbol, db_path=db_path)

    block = False
    penalty = 0.0
    reasons: list[str] = []

    for event in events:
        event_type = str(event.get("event_type") or "").lower()
        severity = str(event.get("severity") or "").lower()

        if severity == "severe" or event_type in SEVERE_EVENT_TYPES:
            penalty += float(settings.corporate_event_severe_penalty_bps or 60.0)
            if (paper and bool(settings.corporate_event_block_paper_severe)) or (
                not paper and bool(settings.corporate_event_block_live_severe)
            ):
                block = True
                reasons.append(f"severe corporate event: {event_type}")

        elif severity == "warning" or event_type in WARNING_EVENT_TYPES:
            penalty += float(settings.corporate_event_warning_penalty_bps or 30.0)
            reasons.append(f"warning corporate event: {event_type}")

        else:
            penalty += float(settings.corporate_event_info_penalty_bps or 10.0)
            reasons.append(f"info corporate event: {event_type}")

    approved = not block

    return {
        "enabled": True,
        "approved": approved,
        "block": block,
        "penalty_bps": round(penalty, 4),
        "events": events,
        "cache": cache,
        "message": "; ".join(reasons) if reasons else "corporate event check passed",
    }
