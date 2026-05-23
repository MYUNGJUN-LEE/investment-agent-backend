from __future__ import annotations

import sqlite3

from app.config import settings
from app.trading import alerting


def test_notify_alerts_stores_without_webhook(tmp_path, monkeypatch):
    db_path = tmp_path / "alerts.sqlite3"
    monkeypatch.setattr(settings, "alert_db_path", str(db_path))
    monkeypatch.setattr(settings, "alert_webhook_url", None)
    monkeypatch.setattr(settings, "alert_min_severity", "high")

    deliveries = alerting.notify_alerts(
        [
            {
                "severity": "high",
                "alert_type": "price_surge",
                "symbol": "005930",
                "message": "surge",
            },
            {
                "severity": "medium",
                "alert_type": "volume_spike",
                "symbol": "005930",
                "message": "volume",
            },
        ],
        source="test",
    )

    assert len(deliveries) == 1
    assert deliveries[0]["delivery_status"] == "stored_no_webhook"
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT source, severity, alert_type, delivery_status FROM alert_deliveries"
        ).fetchone()
    assert row == ("test", "high", "price_surge", "stored_no_webhook")
