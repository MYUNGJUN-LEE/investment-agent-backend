from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import sqlite3
from typing import Any

import httpx

from app.config import settings




SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS alert_deliveries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,
    severity TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    symbol TEXT,
    title TEXT,
    message TEXT NOT NULL,
    delivery_status TEXT NOT NULL,
    webhook_status_code INTEGER,
    error TEXT,
    raw_json TEXT NOT NULL
);
"""


SEVERITY_RANK = {
    "low": 10,
    "medium": 20,
    "high": 30,
    "critical": 40,
}


def initialize_alert_db(db_path: Path | str | None = None) -> None:
    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)


def notify_alerts(
    alerts: list[dict[str, Any]],
    source: str,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Persist alert deliveries and send them to a webhook when configured."""
    eligible = [
        alert
        for alert in alerts
        if _severity_allowed(str(alert.get("severity") or "medium"))
    ]
    if not eligible:
        return []

    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deliveries: list[dict[str, Any]] = []
    for alert in eligible:
        delivery = _deliver_one(alert=alert, source=source)
        deliveries.append(delivery)

    now = _now()
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.executemany(
            """
            INSERT INTO alert_deliveries (
                created_at, source, severity, alert_type, symbol, title,
                message, delivery_status, webhook_status_code, error, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    now,
                    delivery["source"],
                    delivery["severity"],
                    delivery["alert_type"],
                    delivery.get("symbol"),
                    delivery.get("title"),
                    delivery["message"],
                    delivery["delivery_status"],
                    delivery.get("webhook_status_code"),
                    delivery.get("error"),
                    _json(delivery.get("raw", delivery)),
                )
                for delivery in deliveries
            ],
        )
    return deliveries

def _json_bytes(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")


def _deliver_one(alert: dict[str, Any], source: str) -> dict[str, Any]:
    payload = {
        "source": source,
        "severity": str(alert.get("severity") or "medium"),
        "alert_type": str(alert.get("alert_type") or alert.get("event_type") or "alert"),
        "symbol": alert.get("symbol"),
        "title": alert.get("title"),
        "message": str(alert.get("message") or alert.get("title") or "Alert"),
        "raw": alert,
    }
    if not settings.alert_webhook_url:
        return {
            **payload,
            "delivery_status": "stored_no_webhook",
            "webhook_status_code": None,
            "error": None,
        }

    try:
        webhook_url = str(settings.alert_webhook_url)

        outbound_payload = (
            _build_discord_payload(payload)
            if _is_discord_webhook(webhook_url)
            else payload
        )

        response = httpx.post(
            webhook_url,
            content=_json_bytes(outbound_payload),
            timeout=settings.alert_webhook_timeout,
            headers={"Content-Type": "application/json; charset=utf-8"},
        )

        return {
            **payload,
            "delivery_status": "sent" if response.status_code < 400 else "failed",
            "webhook_status_code": response.status_code,
            "error": response.text[:500] if response.status_code >= 400 else None,
        }
    except Exception as exc:
        return {
            **payload,
            "delivery_status": "failed",
            "webhook_status_code": None,
            "error": str(exc),
        }


def _severity_allowed(severity: str) -> bool:
    minimum = SEVERITY_RANK.get(settings.alert_min_severity, 30)
    return SEVERITY_RANK.get(severity, 20) >= minimum


def _db_path(db_path: Path | str | None = None) -> Path:
    return settings.storage_path(db_path or settings.alert_db_path)


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
