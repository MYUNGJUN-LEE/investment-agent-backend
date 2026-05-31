from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from typing import Any

from app.config import settings


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS auto_tuning_recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    sample_count INTEGER NOT NULL,
    win_rate REAL,
    avg_realized_net_edge_bps REAL,
    recommendation_count INTEGER NOT NULL,
    raw_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_auto_tuning_recommendations_time
ON auto_tuning_recommendations(created_at);
"""


def _now() -> str:
    return datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds")


def _db_path(db_path: Path | str | None = None) -> Path:
    return settings.storage_path(db_path or settings.auto_tuning_db_path)


def initialize_auto_tuning_db(db_path: Path | str | None = None) -> None:
    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _load_recent_outcomes(limit: int | None = None) -> list[dict[str, Any]]:
    path = settings.storage_path(settings.outcome_attribution_db_path)
    if not path.exists():
        return []

    limit = max(
        10,
        min(
            int(limit or settings.auto_tuning_recent_limit or 200),
            1000,
        ),
    )

    try:
        with closing(sqlite3.connect(path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT *
                FROM outcome_attribution_events
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
    except sqlite3.Error:
        return []

    return [dict(row) for row in rows]


def _component_average(rows: list[dict[str, Any]], key: str) -> float:
    values = [
        _to_float(row.get(key), 0.0)
        for row in rows
        if row.get(key) is not None
    ]
    return _mean(values)


def _loss_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("outcome_label") or "").lower() == "loss"
        or _to_float(row.get("realized_net_edge_bps"), 0.0) < 0
    ]


def _bounded_recommendation(
    *,
    key: str,
    current_setting: Any,
    suggested_setting: Any,
    reason: str,
    severity: str = "medium",
    confidence: str = "low",
) -> dict[str, Any]:
    return {
        "key": key,
        "current_setting": current_setting,
        "suggested_setting": suggested_setting,
        "reason": reason,
        "severity": severity,
        "confidence": confidence,
        "apply_automatically": False,
    }


def generate_auto_tuning_recommendations(
    *,
    db_path: Path | str | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    if not bool(settings.auto_tuning_enabled):
        return {
            "status": "disabled",
            "enabled": False,
            "recommendations": [],
        }

    mode = str(settings.auto_tuning_mode or "recommend").strip().lower()
    if mode not in {"recommend", "disabled"}:
        mode = "recommend"

    if mode == "disabled":
        return {
            "status": "disabled",
            "enabled": False,
            "recommendations": [],
        }

    rows = _load_recent_outcomes()
    sample_count = len(rows)
    min_samples = int(settings.auto_tuning_min_samples or 30)

    if sample_count < min_samples:
        result = {
            "status": "collecting",
            "enabled": True,
            "mode": mode,
            "sample_count": sample_count,
            "min_samples": min_samples,
            "message": f"Need at least {min_samples} outcome records for auto-tuning recommendations",
            "recommendations": [],
        }
        if persist:
            _record_recommendations(result, db_path=db_path)
        return result

    losses = _loss_rows(rows)
    loss_count = len(losses)
    wins = sample_count - loss_count
    win_rate = wins / sample_count if sample_count else 0.0

    avg_realized = _component_average(rows, "realized_net_edge_bps")
    avg_signal = _component_average(rows, "signal_component_bps")
    avg_regime = _component_average(rows, "market_regime_component_bps")
    avg_execution = _component_average(rows, "execution_component_bps")
    avg_sizing = _component_average(rows, "sizing_component_bps")
    avg_time_decay = _component_average(rows, "time_decay_component_bps")
    avg_unexplained = _component_average(rows, "unexplained_component_bps")

    loss_avg_execution = (
        _component_average(losses, "execution_component_bps") if losses else 0.0
    )
    loss_avg_regime = (
        _component_average(losses, "market_regime_component_bps") if losses else 0.0
    )
    loss_avg_time_decay = (
        _component_average(losses, "time_decay_component_bps") if losses else 0.0
    )
    loss_avg_unexplained = (
        _component_average(losses, "unexplained_component_bps") if losses else 0.0
    )

    recommendations: list[dict[str, Any]] = []

    if loss_count >= int(settings.auto_tuning_min_loss_samples or 10):
        bad_execution_threshold = float(
            settings.auto_tuning_bad_execution_component_bps or -25.0
        )
        if loss_avg_execution <= bad_execution_threshold:
            step = float(settings.auto_tuning_slippage_step_bps or 5.0)
            current_slippage = float(settings.universe_scanner_default_slippage_bps or 10.0)
            suggested_slippage = min(50.0, current_slippage + step)

            recommendations.append(
                _bounded_recommendation(
                    key="UNIVERSE_SCANNER_DEFAULT_SLIPPAGE_BPS",
                    current_setting=current_slippage,
                    suggested_setting=round(suggested_slippage, 4),
                    reason=(
                        f"Average execution component on losses is {loss_avg_execution:.2f}bps, "
                        "suggesting realized fills/slippage are worse than expected."
                    ),
                    severity="high",
                    confidence="medium",
                )
            )

    bad_time_decay_threshold = float(
        settings.auto_tuning_bad_time_decay_component_bps or -15.0
    )
    if (
        loss_count >= int(settings.auto_tuning_min_loss_samples or 10)
        and loss_avg_time_decay <= bad_time_decay_threshold
    ):
        scale_step = float(settings.auto_tuning_signal_decay_scale_step or 0.20)
        current_half_life = int(settings.signal_decay_half_life_seconds or 1800)
        suggested_half_life = max(600, int(current_half_life * (1.0 - scale_step)))

        current_live_age = int(settings.signal_decay_live_max_candidate_age_seconds or 3600)
        suggested_live_age = max(900, int(current_live_age * (1.0 - scale_step)))

        recommendations.append(
            _bounded_recommendation(
                key="SIGNAL_DECAY_HALF_LIFE_SECONDS",
                current_setting=current_half_life,
                suggested_setting=suggested_half_life,
                reason=(
                    f"Average time-decay component on losses is {loss_avg_time_decay:.2f}bps, "
                    "suggesting stale candidates are hurting outcomes."
                ),
                severity="medium",
                confidence="medium",
            )
        )

        recommendations.append(
            _bounded_recommendation(
                key="SIGNAL_DECAY_LIVE_MAX_CANDIDATE_AGE_SECONDS",
                current_setting=current_live_age,
                suggested_setting=suggested_live_age,
                reason="Losses are associated with stale signals; consider reducing live candidate max age.",
                severity="medium",
                confidence="medium",
            )
        )

    bad_regime_threshold = float(settings.auto_tuning_bad_regime_component_bps or -20.0)
    if (
        loss_count >= int(settings.auto_tuning_min_loss_samples or 10)
        and loss_avg_regime <= bad_regime_threshold
    ):
        hurdle_step = float(settings.auto_tuning_hurdle_step_bps or 10.0)

        current_bear_hurdle = float(settings.regime_gate_bear_hurdle_add_bps or 30.0)
        suggested_bear_hurdle = min(100.0, current_bear_hurdle + hurdle_step)

        recommendations.append(
            _bounded_recommendation(
                key="REGIME_GATE_BEAR_HURDLE_ADD_BPS",
                current_setting=current_bear_hurdle,
                suggested_setting=round(suggested_bear_hurdle, 4),
                reason=(
                    f"Average regime component on losses is {loss_avg_regime:.2f}bps, "
                    "suggesting weak-market entries need stricter hurdle."
                ),
                severity="medium",
                confidence="medium",
            )
        )

    bad_avg_threshold = float(settings.auto_tuning_bad_avg_net_edge_bps or -20.0)
    bad_unexplained_threshold = float(
        settings.auto_tuning_bad_unexplained_component_bps or -35.0
    )

    if avg_realized <= bad_avg_threshold or loss_avg_unexplained <= bad_unexplained_threshold:
        scale_step = float(settings.auto_tuning_position_scale_step or 0.10)

        current_max_mult = float(settings.position_sizing_max_multiplier or 1.0)
        suggested_max_mult = max(0.35, current_max_mult * (1.0 - scale_step))

        current_symbol_weight = float(settings.position_sizing_max_symbol_weight or 0.10)
        suggested_symbol_weight = max(0.03, current_symbol_weight * (1.0 - scale_step))

        recommendations.append(
            _bounded_recommendation(
                key="POSITION_SIZING_MAX_MULTIPLIER",
                current_setting=current_max_mult,
                suggested_setting=round(suggested_max_mult, 4),
                reason=(
                    f"Average realized net edge is {avg_realized:.2f}bps and/or unexplained loss is high; "
                    "consider reducing sizing aggressiveness."
                ),
                severity="medium",
                confidence="low",
            )
        )

        recommendations.append(
            _bounded_recommendation(
                key="POSITION_SIZING_MAX_SYMBOL_WEIGHT",
                current_setting=current_symbol_weight,
                suggested_setting=round(suggested_symbol_weight, 4),
                reason="Recent outcomes suggest reducing maximum per-symbol exposure.",
                severity="medium",
                confidence="low",
            )
        )

    if avg_realized <= bad_avg_threshold and win_rate < 0.45:
        hurdle_step = float(settings.auto_tuning_hurdle_step_bps or 10.0)
        current_hurdle = float(settings.universe_scanner_worker_hurdle_rate_bps or 0.0)
        suggested_hurdle = min(120.0, max(0.0, current_hurdle) + hurdle_step)

        recommendations.append(
            _bounded_recommendation(
                key="UNIVERSE_SCANNER_WORKER_HURDLE_RATE_BPS",
                current_setting=current_hurdle,
                suggested_setting=round(suggested_hurdle, 4),
                reason=(
                    f"Recent win rate {win_rate:.2%} and avg realized net edge {avg_realized:.2f}bps "
                    "suggest entry hurdle may be too loose."
                ),
                severity="high",
                confidence="medium",
            )
        )

    max_recs = int(settings.auto_tuning_max_recommendations or 20)
    recommendations = recommendations[:max_recs]

    result = {
        "status": "ready",
        "enabled": True,
        "mode": mode,
        "sample_count": sample_count,
        "loss_count": loss_count,
        "win_rate": round(win_rate, 6),
        "avg_realized_net_edge_bps": round(avg_realized, 4),
        "avg_signal_component_bps": round(avg_signal, 4),
        "avg_market_regime_component_bps": round(avg_regime, 4),
        "avg_execution_component_bps": round(avg_execution, 4),
        "avg_sizing_component_bps": round(avg_sizing, 4),
        "avg_time_decay_component_bps": round(avg_time_decay, 4),
        "avg_unexplained_component_bps": round(avg_unexplained, 4),
        "loss_avg_execution_component_bps": round(loss_avg_execution, 4),
        "loss_avg_regime_component_bps": round(loss_avg_regime, 4),
        "loss_avg_time_decay_component_bps": round(loss_avg_time_decay, 4),
        "loss_avg_unexplained_component_bps": round(loss_avg_unexplained, 4),
        "recommendation_count": len(recommendations),
        "recommendations": recommendations,
    }

    if persist:
        _record_recommendations(result, db_path=db_path)

    return result


def _record_recommendations(
    result: dict[str, Any],
    *,
    db_path: Path | str | None = None,
) -> None:
    try:
        initialize_auto_tuning_db(db_path)
        path = _db_path(db_path)
        with closing(sqlite3.connect(path)) as conn:
            conn.execute(
                """
                INSERT INTO auto_tuning_recommendations (
                    created_at, status, sample_count, win_rate,
                    avg_realized_net_edge_bps, recommendation_count, raw_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _now(),
                    str(result.get("status") or ""),
                    int(result.get("sample_count") or 0),
                    result.get("win_rate"),
                    result.get("avg_realized_net_edge_bps"),
                    int(result.get("recommendation_count") or 0),
                    json.dumps(result, ensure_ascii=False, default=str),
                ),
            )
            conn.commit()
    except sqlite3.Error:
        return


def latest_auto_tuning_recommendation(
    *,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    path = _db_path(db_path)
    if not path.exists():
        return {"status": "empty"}

    try:
        with closing(sqlite3.connect(path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT *
                FROM auto_tuning_recommendations
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """
            ).fetchone()
    except sqlite3.Error as exc:
        return {"status": "error", "message": str(exc)}

    if not row:
        return {"status": "empty"}

    try:
        payload = json.loads(row["raw_json"])
    except Exception:
        payload = {}

    return {
        "status": "ready",
        "created_at": row["created_at"],
        "sample_count": row["sample_count"],
        "recommendation_count": row["recommendation_count"],
        "payload": payload,
    }
