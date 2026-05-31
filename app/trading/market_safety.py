from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any

from app.config import settings


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS market_safety_flags (
    symbol TEXT NOT NULL,
    market TEXT,
    flag_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    reason TEXT,
    source TEXT,
    effective_date TEXT,
    expires_at TEXT,
    updated_at TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    PRIMARY KEY(symbol, flag_type)
);

CREATE INDEX IF NOT EXISTS idx_market_safety_symbol
ON market_safety_flags(symbol);

CREATE INDEX IF NOT EXISTS idx_market_safety_updated
ON market_safety_flags(updated_at);

CREATE TABLE IF NOT EXISTS market_safety_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


SEVERE_FLAG_TYPES = {
    "trading_halt",
    "delisting",
    "liquidation",
    "managed",
    "investment_risk",
}

WARNING_FLAG_TYPES = {
    "investment_warning",
    "unfaithful_disclosure",
    "substantive_review",
}

CAUTION_FLAG_TYPES = {
    "investment_caution",
    "caution",
    "watch",
}


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _db_path(db_path: Path | str | None = None) -> Path:
    return settings.storage_path(db_path or settings.market_safety_db_path)


def initialize_market_safety_db(db_path: Path | str | None = None) -> None:
    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)


def _normalize_symbol(value: Any) -> str:
    raw = str(value or "").strip()
    return raw.zfill(6) if raw.isdigit() else raw


def _normalize_flag_type(value: Any) -> str:
    raw = str(value or "").strip().lower()

    mapping = {
        "관리종목": "managed",
        "투자주의": "investment_caution",
        "투자경고": "investment_warning",
        "투자위험": "investment_risk",
        "매매거래정지": "trading_halt",
        "거래정지": "trading_halt",
        "상장폐지": "delisting",
        "상장폐지 위험": "delisting",
        "상장폐지위험": "delisting",
        "정리매매": "liquidation",
        "불성실공시": "unfaithful_disclosure",
        "불성실공시법인": "unfaithful_disclosure",
        "불성실공시 법인": "unfaithful_disclosure",
        "실질심사": "substantive_review",
        "상장적격성 실질심사": "substantive_review",
        "상장적격성실질심사": "substantive_review",
    }

    return mapping.get(raw, raw.replace(" ", "_"))


def _severity_for_flag(flag_type: str) -> str:
    if flag_type in SEVERE_FLAG_TYPES:
        return "severe"
    if flag_type in WARNING_FLAG_TYPES:
        return "warning"
    if flag_type in CAUTION_FLAG_TYPES:
        return "caution"
    return "info"


def _parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        raw = str(value).strip()
        if not raw:
            return None
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def _is_expired(expires_at: Any) -> bool:
    dt = _parse_time(expires_at)
    if dt is None:
        return False
    return datetime.utcnow() > dt


def upsert_market_safety_flags(
    flags: list[dict[str, Any]],
    *,
    source: str = "manual",
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Upsert cached official safety flags from any trusted manual/source adapter."""
    initialize_market_safety_db(db_path)
    path = _db_path(db_path)

    now = _now()
    inserted = 0
    skipped = 0

    with sqlite3.connect(path) as conn:
        for item in flags:
            if not isinstance(item, dict):
                skipped += 1
                continue

            symbol = _normalize_symbol(item.get("symbol"))
            if not symbol:
                skipped += 1
                continue

            flag_type = _normalize_flag_type(item.get("flag_type"))
            if not flag_type:
                skipped += 1
                continue

            severity = str(item.get("severity") or _severity_for_flag(flag_type))

            conn.execute(
                """
                INSERT INTO market_safety_flags (
                    symbol, market, flag_type, severity, reason,
                    source, effective_date, expires_at, updated_at, raw_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, flag_type) DO UPDATE SET
                    market = excluded.market,
                    severity = excluded.severity,
                    reason = excluded.reason,
                    source = excluded.source,
                    effective_date = excluded.effective_date,
                    expires_at = excluded.expires_at,
                    updated_at = excluded.updated_at,
                    raw_json = excluded.raw_json
                """,
                (
                    symbol,
                    item.get("market"),
                    flag_type,
                    severity,
                    item.get("reason"),
                    source,
                    item.get("effective_date"),
                    item.get("expires_at"),
                    now,
                    json.dumps(item.get("raw_json") or item, ensure_ascii=False, default=str),
                ),
            )
            inserted += 1

        conn.execute(
            """
            INSERT INTO market_safety_meta (key, value, updated_at)
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


def get_market_safety_flags(
    symbol: str,
    *,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    if not bool(settings.market_safety_cache_enabled):
        return []

    symbol = _normalize_symbol(symbol)
    if not symbol:
        return []

    path = _db_path(db_path)
    if not path.exists():
        return []

    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT *
                FROM market_safety_flags
                WHERE symbol = ?
                ORDER BY severity DESC, updated_at DESC
                """,
                (symbol,),
            ).fetchall()
    except sqlite3.Error:
        return []

    flags: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if _is_expired(item.get("expires_at")):
            continue
        flags.append(item)

    return flags


def market_safety_cache_status(
    *,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    if not bool(settings.market_safety_cache_enabled):
        return {"status": "disabled", "enabled": False}

    path = _db_path(db_path)
    if not path.exists():
        return {"status": "missing", "enabled": True}

    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            count = conn.execute(
                "SELECT COUNT(*) FROM market_safety_flags"
            ).fetchone()[0]
            row = conn.execute(
                "SELECT value FROM market_safety_meta WHERE key = 'last_refresh_at'"
            ).fetchone()
    except sqlite3.Error as exc:
        return {"status": "error", "enabled": True, "message": str(exc)}

    last_refresh = str(row["value"]) if row else None
    stale = True
    age = None

    if last_refresh:
        dt = _parse_time(last_refresh)
        if dt is not None:
            age = (datetime.utcnow() - dt).total_seconds()
            stale = age > float(settings.market_safety_cache_ttl_seconds or 86400)

    return {
        "status": "ready",
        "enabled": True,
        "flag_count": int(count or 0),
        "last_refresh_at": last_refresh,
        "age_seconds": round(age, 3) if age is not None else None,
        "stale": stale,
    }


def market_safety_check(
    symbol: str,
    *,
    execution_mode: str = "paper",
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    if not bool(settings.market_safety_cache_enabled):
        return {
            "enabled": False,
            "approved": True,
            "block": False,
            "penalty_bps": 0.0,
            "flags": [],
            "message": "market safety cache disabled",
        }

    cache = market_safety_cache_status(db_path=db_path)
    paper = str(execution_mode or "paper").lower() == "paper"

    if cache.get("status") in {"missing", "error"} or cache.get("stale") is True:
        penalty = 0.0 if paper else float(settings.market_safety_missing_cache_live_penalty_bps or 5.0)
        status = str(cache.get("status") or "missing")
        message = (
            "market safety cache stale; neutral in paper, mild penalty in live"
            if cache.get("stale") is True
            else "market safety cache unavailable; neutral in paper, mild penalty in live"
        )
        return {
            "enabled": True,
            "approved": True,
            "block": False,
            "penalty_bps": round(penalty, 4),
            "flags": [],
            "cache": cache,
            "status": status,
            "message": message,
        }

    flags = get_market_safety_flags(symbol, db_path=db_path)

    block = False
    penalty = 0.0
    reasons: list[str] = []

    for flag in flags:
        flag_type = str(flag.get("flag_type") or "").lower()
        severity = str(flag.get("severity") or "").lower()

        if flag_type == "trading_halt":
            if (paper and bool(settings.market_safety_paper_block_halt)) or (
                not paper and bool(settings.market_safety_live_block_halt)
            ):
                block = True
                reasons.append("trading halt")

        elif flag_type in {"delisting", "liquidation"}:
            if (paper and bool(settings.market_safety_paper_block_delisting)) or (
                not paper and (
                    (flag_type == "delisting" and bool(settings.market_safety_live_block_delisting))
                    or (
                        flag_type == "liquidation"
                        and bool(settings.market_safety_live_block_liquidation)
                    )
                )
            ):
                block = True
                reasons.append(flag_type)

        elif flag_type == "managed":
            if not paper and bool(settings.market_safety_live_block_managed):
                block = True
                reasons.append("managed issue")
            else:
                penalty += float(settings.market_safety_warning_penalty_bps or 40.0)

        elif flag_type == "investment_risk":
            if not paper and bool(settings.market_safety_live_block_investment_risk):
                block = True
                reasons.append("investment risk")
            else:
                penalty += float(settings.market_safety_warning_penalty_bps or 40.0)

        elif flag_type == "investment_warning":
            if not paper and bool(settings.market_safety_live_block_investment_warning):
                block = True
                reasons.append("investment warning")
            else:
                penalty += float(settings.market_safety_warning_penalty_bps or 40.0)

        elif flag_type == "investment_caution":
            if not paper and bool(settings.market_safety_live_block_investment_caution):
                block = True
                reasons.append("investment caution")
            else:
                penalty += float(settings.market_safety_caution_penalty_bps or 15.0)

        elif severity in {"warning", "severe"}:
            penalty += float(settings.market_safety_warning_penalty_bps or 40.0)
        elif severity == "caution":
            penalty += float(settings.market_safety_caution_penalty_bps or 15.0)

    approved = not block

    return {
        "enabled": True,
        "approved": approved,
        "block": block,
        "penalty_bps": round(penalty, 4),
        "flags": flags,
        "cache": cache,
        "status": "ready",
        "message": "; ".join(reasons) if reasons else "market safety check passed",
    }
