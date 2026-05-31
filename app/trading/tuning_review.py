from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from typing import Any

from app.config import settings
from app.trading.auto_tuning import latest_auto_tuning_recommendation


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tuning_review_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    recommendation_key TEXT NOT NULL,
    current_setting TEXT,
    suggested_setting TEXT,
    decision TEXT NOT NULL,
    reason TEXT,
    reviewer TEXT,
    source_created_at TEXT,
    raw_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tuning_review_key_time
ON tuning_review_decisions(recommendation_key, created_at);

CREATE INDEX IF NOT EXISTS idx_tuning_review_decision_time
ON tuning_review_decisions(decision, created_at);
"""


def _now() -> str:
    return datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds")


def _db_path(db_path: Path | str | None = None) -> Path:
    return settings.storage_path(db_path or settings.tuning_review_db_path)


def initialize_tuning_review_db(db_path: Path | str | None = None) -> None:
    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


def _allowed_keys() -> set[str]:
    raw = str(settings.tuning_review_allowed_keys or "")
    return {item.strip().upper() for item in raw.split(",") if item.strip()}


def _normalize_decision(value: Any) -> str:
    decision = str(value or "").strip().lower()
    if decision in {"approve", "approved"}:
        return "approved"
    if decision in {"reject", "rejected"}:
        return "rejected"
    if decision in {"defer", "deferred"}:
        return "deferred"
    return "deferred"


def list_latest_tuning_recommendations() -> dict[str, Any]:
    if not bool(settings.tuning_review_enabled):
        return {"status": "disabled", "recommendations": []}

    latest = latest_auto_tuning_recommendation()
    payload = latest.get("payload") if isinstance(latest, dict) else {}
    if not isinstance(payload, dict):
        payload = {}

    recommendations = payload.get("recommendations") or []
    if not isinstance(recommendations, list):
        recommendations = []

    allowed = _allowed_keys()

    safe_recommendations: list[dict[str, Any]] = []
    for item in recommendations:
        if not isinstance(item, dict):
            continue

        key = str(item.get("key") or "").strip().upper()
        if key not in allowed:
            continue

        safe_recommendations.append(
            {
                "key": key,
                "current_setting": item.get("current_setting"),
                "suggested_setting": item.get("suggested_setting"),
                "reason": item.get("reason"),
                "severity": item.get("severity"),
                "confidence": item.get("confidence"),
                "apply_automatically": False,
            }
        )

    return {
        "status": "ready",
        "source_status": latest.get("status") if isinstance(latest, dict) else None,
        "source_created_at": latest.get("created_at") if isinstance(latest, dict) else None,
        "recommendation_count": len(safe_recommendations),
        "recommendations": safe_recommendations,
    }


def record_tuning_decision(
    *,
    recommendation_key: str,
    decision: str,
    current_setting: Any = None,
    suggested_setting: Any = None,
    reason: str | None = None,
    reviewer: str | None = None,
    raw_json: dict[str, Any] | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    if not bool(settings.tuning_review_enabled):
        return {"status": "disabled"}

    key = str(recommendation_key or "").strip().upper()
    if key not in _allowed_keys():
        return {
            "status": "rejected",
            "message": f"Recommendation key {key} is not allowlisted",
        }

    normalized = _normalize_decision(decision)

    latest = latest_auto_tuning_recommendation()
    source_created_at = latest.get("created_at") if isinstance(latest, dict) else None

    payload = {
        "recommendation_key": key,
        "decision": normalized,
        "current_setting": current_setting,
        "suggested_setting": suggested_setting,
        "reason": reason,
        "reviewer": reviewer,
        "source_created_at": source_created_at,
        "raw_json": raw_json or {},
    }

    initialize_tuning_review_db(db_path)
    path = _db_path(db_path)

    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            """
            INSERT INTO tuning_review_decisions (
                created_at, recommendation_key, current_setting,
                suggested_setting, decision, reason, reviewer,
                source_created_at, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now(),
                key,
                str(current_setting) if current_setting is not None else None,
                str(suggested_setting) if suggested_setting is not None else None,
                normalized,
                reason,
                reviewer,
                source_created_at,
                json.dumps(payload, ensure_ascii=False, default=str),
            ),
        )
        _prune_old_records(conn)
        conn.commit()

    return {
        "status": "recorded",
        "recommendation_key": key,
        "decision": normalized,
        "suggested_setting": suggested_setting,
    }


def _prune_old_records(conn: sqlite3.Connection) -> None:
    max_records = int(settings.tuning_review_max_records or 1000)
    if max_records <= 0:
        return

    conn.execute(
        """
        DELETE FROM tuning_review_decisions
        WHERE id NOT IN (
            SELECT id
            FROM tuning_review_decisions
            ORDER BY created_at DESC, id DESC
            LIMIT ?
        )
        """,
        (max_records,),
    )


def latest_approved_tuning_decisions(
    *,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    if not bool(settings.tuning_review_enabled):
        return {"status": "disabled", "approved": []}

    path = _db_path(db_path)
    if not path.exists():
        return {"status": "empty", "approved": []}

    allowed = _allowed_keys()

    try:
        with closing(sqlite3.connect(path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT *
                FROM tuning_review_decisions
                WHERE decision = 'approved'
                ORDER BY created_at DESC, id DESC
                LIMIT 200
                """
            ).fetchall()
    except sqlite3.Error as exc:
        return {"status": "error", "message": str(exc), "approved": []}

    latest_by_key: dict[str, dict[str, Any]] = {}

    for row in rows:
        key = str(row["recommendation_key"] or "").strip().upper()
        if key not in allowed:
            continue
        if key in latest_by_key:
            continue

        latest_by_key[key] = {
            "key": key,
            "suggested_setting": row["suggested_setting"],
            "current_setting": row["current_setting"],
            "approved_at": row["created_at"],
            "reviewer": row["reviewer"],
            "reason": row["reason"],
        }

    return {
        "status": "ready",
        "approved_count": len(latest_by_key),
        "approved": list(latest_by_key.values()),
    }


def export_approved_tuning_env_lines(
    *,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    approved = latest_approved_tuning_decisions(db_path=db_path)
    items = approved.get("approved") or []

    lines: list[str] = []
    for item in items:
        key = str(item.get("key") or "").strip().upper()
        value = item.get("suggested_setting")
        if key and value is not None:
            lines.append(f"{key}={value}")

    return {
        "status": approved.get("status"),
        "line_count": len(lines),
        "env_lines": lines,
        "message": "Copy these lines into Render Environment manually after review.",
    }
