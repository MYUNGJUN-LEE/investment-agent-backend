from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import json
import sqlite3
from typing import Any, Iterator

from app.config import settings
from app.trading import paper_trading


FEATURE_NAMES = [
    "bias",
    "raw_score",
    "change_rate",
    "volume_ratio",
    "minute_volume_ratio",
    "news_count",
    "disclosure_count",
    "overheated",
    "latest_close",
    "low_turnover",
]

DEFAULT_RETURN_COEFFICIENTS = {
    "bias": 35.0,
    "raw_score": 180.0,
    "change_rate": 80.0,
    "volume_ratio": 90.0,
    "minute_volume_ratio": 36.0,
    "news_count": 25.0,
    "disclosure_count": 24.0,
    "overheated": -120.0,
    "latest_close": -100.0,
    "low_turnover": -20.0,
}

DEFAULT_RISK_COEFFICIENTS = {
    "bias": 109.0,
    "raw_score": -80.0,
    "change_rate": 20.0,
    "volume_ratio": 15.0,
    "minute_volume_ratio": 10.0,
    "news_count": 0.0,
    "disclosure_count": 0.0,
    "overheated": 90.0,
    "latest_close": 35.0,
    "low_turnover": 20.0,
}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS edge_calibration_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    sample_count INTEGER NOT NULL,
    skipped_count INTEGER NOT NULL,
    horizon_seconds INTEGER NOT NULL,
    mae_return_bps REAL,
    mae_risk_bps REAL,
    gate_approved INTEGER,
    oos_sample_count INTEGER NOT NULL DEFAULT 0,
    top10_avg_return_bps REAL,
    top10_win_rate REAL,
    raw_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS edge_coefficients (
    target TEXT NOT NULL,
    feature TEXT NOT NULL,
    coefficient REAL NOT NULL,
    updated_at TEXT NOT NULL,
    run_id INTEGER,
    PRIMARY KEY(target, feature)
);

CREATE TABLE IF NOT EXISTS edge_calibration_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS edge_training_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_candidate_id INTEGER NOT NULL UNIQUE,
    symbol TEXT NOT NULL,
    scan_time TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    entry_price REAL NOT NULL,
    observed_price REAL NOT NULL,
    features_json TEXT NOT NULL,
    realized_return_bps REAL NOT NULL,
    realized_risk_bps REAL NOT NULL,
    rank INTEGER,
    status TEXT,
    created_at TEXT NOT NULL,
    raw_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_edge_training_samples_observed
ON edge_training_samples(observed_at);

CREATE INDEX IF NOT EXISTS idx_edge_training_samples_symbol
ON edge_training_samples(symbol, scan_time);

CREATE TABLE IF NOT EXISTS top_candidate_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluated_at TEXT NOT NULL,
    sample_count INTEGER NOT NULL,
    top_count INTEGER NOT NULL,
    avg_return_bps REAL,
    win_rate REAL,
    raw_json TEXT NOT NULL
);
"""


def initialize_edge_calibration_db(db_path: Path | str | None = None) -> None:
    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
        _ensure_schema_migrations(conn)


def calibrate_edge_model_if_due(
    *,
    universe_db_path: Path | str | None = None,
    calibration_db_path: Path | str | None = None,
) -> dict[str, Any]:
    if not settings.edge_calibration_enabled:
        return {"status": "disabled", "message": "Edge calibration is disabled"}
    path = _db_path(calibration_db_path)
    initialize_edge_calibration_db(path)
    interval = max(60, int(settings.edge_calibration_interval_seconds or 3600))
    now = _now()
    last_run_at = _read_meta(path, "last_success_at") or _read_meta(path, "last_attempt_at")
    if last_run_at:
        try:
            elapsed = (datetime.fromisoformat(now) - datetime.fromisoformat(last_run_at)).total_seconds()
        except ValueError:
            elapsed = interval
        if elapsed < interval:
            return {
                "status": "not_due",
                "last_run_at": last_run_at,
                "next_run_at": (
                    datetime.fromisoformat(last_run_at) + timedelta(seconds=interval)
                ).isoformat(timespec="seconds"),
            }
    return calibrate_edge_model(
        universe_db_path=universe_db_path,
        calibration_db_path=path,
    )


def calibrate_edge_model(
    *,
    universe_db_path: Path | str | None = None,
    calibration_db_path: Path | str | None = None,
    max_samples: int | None = None,
    min_samples: int | None = None,
) -> dict[str, Any]:
    """Fit tiny ridge models from scanner history without loading large frames."""
    calibration_path = _db_path(calibration_db_path)
    universe_path = Path(universe_db_path or settings.universe_scanner_db_path)
    initialize_edge_calibration_db(calibration_path)
    now = _now()
    _write_meta(calibration_path, "last_attempt_at", now)

    if not universe_path.exists():
        result = {
            "status": "skipped",
            "message": "Universe scanner DB does not exist yet",
            "sample_count": 0,
        }
        _record_run(calibration_path, result)
        return result

    max_samples = max(1, int(max_samples or settings.edge_calibration_max_samples or 1000))
    min_samples = max(1, int(min_samples or settings.edge_calibration_min_samples or 30))
    horizon_seconds = max(60, int(settings.edge_calibration_horizon_seconds or 86400))
    ridge_lambda = max(0.0, float(settings.edge_calibration_ridge_lambda or 0.0))
    blend = max(0.0, min(1.0, float(settings.edge_calibration_blend or 0.35)))
    target_samples = max(
        min_samples,
        int(settings.edge_calibration_target_samples or 1000),
    )
    training_limit = max(max_samples, target_samples)

    refresh_result = _refresh_training_samples(
        calibration_path=calibration_path,
        universe_path=universe_path,
        horizon_seconds=horizon_seconds,
        candidate_limit=max(
            training_limit,
            int(settings.edge_calibration_sample_retention_limit or 10_000),
        ),
    )
    samples = list(
        _iter_training_samples_from_store(
            calibration_path=calibration_path,
            limit=training_limit,
        )
    )
    stored_sample_count = _stored_sample_count(calibration_path)
    sample_count = len(samples)
    skipped_count = int(refresh_result.get("skipped_count") or 0)

    dimension = len(FEATURE_NAMES)
    return_matrix = _ridge_matrix(dimension, ridge_lambda)
    risk_matrix = _ridge_matrix(dimension, ridge_lambda)
    return_vector = [0.0] * dimension
    risk_vector = [0.0] * dimension

    train_samples, oos_samples = _performance_split(samples)
    for sample in train_samples:
        features, realized_return_bps, realized_risk_bps = sample
        _accumulate(return_matrix, return_vector, features, realized_return_bps)
        _accumulate(risk_matrix, risk_vector, features, realized_risk_bps)

    if sample_count < min_samples:
        gate = _default_gate(
            status="collecting",
            approved=False,
            message=f"Stored calibration samples {sample_count}/{min_samples}",
        )
        result = {
            "status": "skipped",
            "message": f"Not enough calibration samples: {sample_count}/{min_samples}",
            "sample_count": sample_count,
            "skipped_count": skipped_count,
            "stored_sample_count": stored_sample_count,
            "horizon_seconds": horizon_seconds,
            "refresh": refresh_result,
            "gate": gate,
        }
        _record_run(calibration_path, result)
        return result

    fitted_return = _coefficients_from_solution(_solve(return_matrix, return_vector))
    fitted_risk = _coefficients_from_solution(_solve(risk_matrix, risk_vector))
    previous = load_edge_model(calibration_db_path=calibration_path, include_defaults=True)
    blended_return = _blend_coefficients(
        previous.get("expected_return", DEFAULT_RETURN_COEFFICIENTS),
        fitted_return,
        blend=blend,
    )
    blended_risk = _blend_coefficients(
        previous.get("expected_risk", DEFAULT_RISK_COEFFICIENTS),
        fitted_risk,
        blend=blend,
    )
    train_metrics = _model_metrics_for_samples(
        train_samples,
        return_coefficients=blended_return,
        risk_coefficients=blended_risk,
    )
    oos_metrics = _model_metrics_for_samples(
        oos_samples,
        return_coefficients=blended_return,
        risk_coefficients=blended_risk,
    )
    top10_performance = _top10_performance_from_store(
        calibration_path=calibration_path,
        limit=target_samples,
    )
    _record_top10_performance(calibration_path, top10_performance)
    fill_adjustment = record_fill_adjustment_from_fills(
        calibration_db_path=calibration_path,
    )
    gate = _gate_from_metrics(
        sample_count=sample_count,
        oos_sample_count=len(oos_samples),
        mae_return_bps=oos_metrics["mae_return_bps"],
        mae_risk_bps=oos_metrics["mae_risk_bps"],
        top10_performance=top10_performance,
        fill_adjustment=fill_adjustment,
    )
    result = {
        "status": "success",
        "sample_count": sample_count,
        "train_sample_count": len(train_samples),
        "oos_sample_count": len(oos_samples),
        "skipped_count": skipped_count,
        "stored_sample_count": stored_sample_count,
        "horizon_seconds": horizon_seconds,
        "mae_return_bps": oos_metrics["mae_return_bps"],
        "mae_risk_bps": oos_metrics["mae_risk_bps"],
        "train_metrics": train_metrics,
        "oos_metrics": oos_metrics,
        "top10_performance": top10_performance,
        "fill_adjustment": fill_adjustment,
        "gate": gate,
        "refresh": refresh_result,
        "blend": blend,
        "coefficient_count": len(FEATURE_NAMES) * 2,
    }
    run_id = _record_run(calibration_path, result)
    _store_coefficients(
        calibration_path,
        run_id=run_id,
        return_coefficients=blended_return,
        risk_coefficients=blended_risk,
        updated_at=now,
    )
    _write_meta(calibration_path, "last_success_at", now)
    return {**result, "run_id": run_id}


def get_edge_calibration_status(
    *,
    calibration_db_path: Path | str | None = None,
) -> dict[str, Any]:
    path = _db_path(calibration_db_path)
    if not path.exists():
        return {
            "status": "empty",
            "message": "No edge calibration DB has been created yet",
            "enabled": settings.edge_calibration_enabled,
        }
    initialize_edge_calibration_db(path)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        run = conn.execute(
            """
            SELECT *
            FROM edge_calibration_runs
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
        coefficient_count = conn.execute(
            "SELECT COUNT(*) FROM edge_coefficients"
        ).fetchone()[0]
        stored_sample_count = conn.execute(
            "SELECT COUNT(*) FROM edge_training_samples"
        ).fetchone()[0]
    return {
        "status": "ready" if coefficient_count else "empty",
        "enabled": settings.edge_calibration_enabled,
        "coefficient_count": int(coefficient_count or 0),
        "stored_sample_count": int(stored_sample_count or 0),
        "last_run": dict(run) if run else None,
        "latest_gate": _latest_raw_json(path, "gate"),
        "latest_top10_performance": _latest_raw_json(path, "top10_performance"),
        "last_success_at": _read_meta(path, "last_success_at"),
        "interval_seconds": settings.edge_calibration_interval_seconds,
        "horizon_seconds": settings.edge_calibration_horizon_seconds,
        "max_samples": settings.edge_calibration_max_samples,
        "min_samples": settings.edge_calibration_min_samples,
    }


def load_edge_model(
    *,
    calibration_db_path: Path | str | None = None,
    include_defaults: bool = False,
) -> dict[str, dict[str, float]]:
    path = _db_path(calibration_db_path)
    model: dict[str, dict[str, float]] = {}
    if path.exists():
        initialize_edge_calibration_db(path)
        with sqlite3.connect(path) as conn:
            rows = conn.execute(
                """
                SELECT target, feature, coefficient
                FROM edge_coefficients
                """
            ).fetchall()
        for target, feature, coefficient in rows:
            model.setdefault(str(target), {})[str(feature)] = float(coefficient)
    if include_defaults:
        if "expected_return" not in model:
            model["expected_return"] = dict(DEFAULT_RETURN_COEFFICIENTS)
        if "expected_risk" not in model:
            model["expected_risk"] = dict(DEFAULT_RISK_COEFFICIENTS)
    return model


def estimate_expected_edges(
    candidate: dict[str, Any],
    raw_score: float,
    *,
    model: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any] | None:
    model = model or load_edge_model()
    return_coefficients = model.get("expected_return")
    risk_coefficients = model.get("expected_risk")
    if not return_coefficients or not risk_coefficients:
        return None
    feature_map = candidate_feature_map(candidate, raw_score)
    fill_adjustment = load_fill_adjustment()
    expected_return = _predict(return_coefficients, feature_map) * fill_adjustment
    expected_risk = _predict(risk_coefficients, feature_map)
    return {
        "expected_return": round(max(0.0, min(500.0, expected_return)), 4),
        "expected_risk": round(max(0.0, min(500.0, expected_risk)), 4),
        "edge_model": "calibrated_ridge_v1",
        "fill_adjustment": round(fill_adjustment, 4),
    }


def candidate_feature_map(
    candidate: dict[str, Any],
    raw_score: float,
) -> dict[str, float]:
    intraday = candidate.get("intraday") or {}
    volume_ratio = _to_float(candidate.get("volume_ratio")) or 0.0
    minute_volume_ratio = _to_float(intraday.get("minute_volume_ratio")) or 0.0
    turnover = _to_float(candidate.get("turnover_value"))
    return {
        "bias": 1.0,
        "raw_score": max(0.0, min(1.0, float(raw_score) / 100.0)),
        "change_rate": max(-1.0, min(1.5, (_to_float(candidate.get("change_rate")) or 0.0) / 10.0)),
        "volume_ratio": max(0.0, min(2.0, volume_ratio / 5.0)),
        "minute_volume_ratio": max(0.0, min(2.0, minute_volume_ratio / 5.0)),
        "news_count": max(0.0, min(1.0, float(_to_int(candidate.get("news_count")) or 0) / 5.0)),
        "disclosure_count": max(0.0, min(1.0, float(_to_int(candidate.get("disclosure_count")) or 0) / 3.0)),
        "overheated": 1.0 if bool(candidate.get("overheated")) else 0.0,
        "latest_close": 1.0 if candidate.get("price_source") == "latest_close" else 0.0,
        "low_turnover": 1.0 if turnover is not None and turnover < 20_000_000_000 else 0.0,
    }


def edge_entry_gate(
    candidates: list[dict[str, Any]] | None = None,
    *,
    calibration_db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Return whether calibrated edge quality is strong enough for new entries."""
    if not settings.edge_calibration_enabled:
        return _default_gate(
            status="disabled",
            approved=True,
            message="Edge calibration is disabled",
        )

    path = _db_path(calibration_db_path)
    if not path.exists():
        return _default_gate(
            status="collecting",
            approved=False,
            message="No edge calibration DB has been created yet",
        )
    initialize_edge_calibration_db(path)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        stored_sample_count = conn.execute(
            "SELECT COUNT(*) FROM edge_training_samples"
        ).fetchone()[0]
        run = conn.execute(
            """
            SELECT *
            FROM edge_calibration_runs
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    if not run:
        return _default_gate(
            status="collecting",
            approved=False,
            message="No edge calibration run has completed yet",
            sample_count=int(stored_sample_count or 0),
        )

    raw = _parse_json(run["raw_json"], {})
    top10_performance = raw.get("top10_performance") or _top10_performance_from_store(
        calibration_path=path,
        limit=int(settings.edge_calibration_target_samples or 1000),
    )
    fill_adjustment = raw.get("fill_adjustment") or {
        "multiplier": load_fill_adjustment(calibration_db_path=path),
    }
    return _gate_from_metrics(
        sample_count=int(stored_sample_count or run["sample_count"] or 0),
        oos_sample_count=int(raw.get("oos_sample_count") or run["oos_sample_count"] or 0),
        mae_return_bps=_to_float(run["mae_return_bps"]),
        mae_risk_bps=_to_float(run["mae_risk_bps"]),
        top10_performance=top10_performance,
        fill_adjustment=fill_adjustment,
        candidates=candidates,
    )


def load_fill_adjustment(
    *,
    calibration_db_path: Path | str | None = None,
) -> float:
    path = _db_path(calibration_db_path)
    if not path.exists():
        return 1.0
    try:
        raw = _read_meta(path, "fill_adjustment_multiplier")
    except sqlite3.Error:
        return 1.0
    value = _to_float(raw)
    if value is None:
        return 1.0
    return max(0.5, min(1.05, value))


def record_fill_adjustment_from_fills(
    *,
    calibration_db_path: Path | str | None = None,
    paper_db_path: Path | str | None = None,
    broker_db_path: Path | str | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    """Estimate execution-quality drag from paper and broker fills, then persist it."""
    path = _db_path(calibration_db_path)
    initialize_edge_calibration_db(path)
    fill_events: list[dict[str, float]] = []
    fill_events.extend(
        _paper_fill_quality_events(
            Path(paper_db_path or paper_trading.DEFAULT_DB_PATH),
            limit=limit,
        )
    )
    fill_events.extend(
        _broker_fill_quality_events(
            Path(broker_db_path or settings.broker_sync_db_path),
            limit=limit,
        )
    )
    if not fill_events:
        multiplier = load_fill_adjustment(calibration_db_path=path)
        return {
            "status": "empty",
            "multiplier": multiplier,
            "sample_count": 0,
            "message": "No paper or broker fill samples available yet",
        }

    fill_rate = sum(item["fill_ratio"] for item in fill_events) / len(fill_events)
    adverse_slippage_bps = (
        sum(item["adverse_slippage_bps"] for item in fill_events) / len(fill_events)
    )
    multiplier = max(
        0.5,
        min(1.05, fill_rate * max(0.5, 1.0 - adverse_slippage_bps / 500.0)),
    )
    _write_meta(path, "fill_adjustment_multiplier", f"{multiplier:.6f}")
    _write_meta(path, "fill_adjustment_updated_at", _now())
    return {
        "status": "ready",
        "multiplier": round(multiplier, 6),
        "sample_count": len(fill_events),
        "avg_fill_rate": round(fill_rate, 6),
        "avg_adverse_slippage_bps": round(adverse_slippage_bps, 4),
    }


def _iter_calibration_samples(
    *,
    universe_path: Path,
    max_samples: int,
    horizon_seconds: int,
) -> Iterator[tuple[list[float], float, float] | None]:
    with sqlite3.connect(universe_path) as conn:
        conn.row_factory = sqlite3.Row
        candidate_cursor = conn.execute(
            """
            SELECT *
            FROM scanner_candidate_history
            WHERE current_price IS NOT NULL
              AND current_price > 0
            ORDER BY scan_time DESC, id DESC
            LIMIT ?
            """,
            (max_samples,),
        )
        while True:
            rows = candidate_cursor.fetchmany(50)
            if not rows:
                break
            for row in rows:
                yield _sample_from_candidate_row(
                    conn=conn,
                    row=row,
                    horizon_seconds=horizon_seconds,
                )


def _sample_from_candidate_row(
    *,
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    horizon_seconds: int,
) -> tuple[list[float], float, float] | None:
    payload = _training_payload_from_candidate_row(
        conn=conn,
        row=row,
        horizon_seconds=horizon_seconds,
    )
    if payload is None:
        return None
    return (
        payload["features"],
        payload["realized_return_bps"],
        payload["realized_risk_bps"],
    )


def _training_payload_from_candidate_row(
    *,
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    horizon_seconds: int,
) -> dict[str, Any] | None:
    entry_price = _to_float(row["current_price"])
    if entry_price is None or entry_price <= 0:
        return None
    scan_time = str(row["scan_time"])
    try:
        end_time = (
            datetime.fromisoformat(scan_time) + timedelta(seconds=horizon_seconds)
        ).isoformat(timespec="seconds")
    except ValueError:
        return None
    future_rows = conn.execute(
        """
        SELECT created_at, current_price
        FROM universe_price_snapshots
        WHERE symbol = ?
          AND created_at > ?
          AND created_at <= ?
          AND current_price IS NOT NULL
          AND current_price > 0
        ORDER BY created_at ASC
        LIMIT ?
        """,
        (
            row["symbol"],
            scan_time,
            end_time,
            max(1, int(settings.edge_calibration_future_price_limit or 96)),
        ),
    ).fetchall()
    if not future_rows:
        return None
    final_price = _to_float(future_rows[-1]["current_price"])
    observed_at = str(future_rows[-1]["created_at"])
    prices = [_to_float(item["current_price"]) for item in future_rows]
    usable_prices = [price for price in prices if price is not None and price > 0]
    if final_price is None or not usable_prices:
        return None
    realized_return_bps = (final_price - entry_price) / entry_price * 10_000
    min_price = min(usable_prices)
    realized_risk_bps = max(0.0, (entry_price - min_price) / entry_price * 10_000)
    raw_score = _to_float(row["raw_score"]) or 0.0
    raw_payload = _parse_json(row["raw_json"], {})
    candidate = {
        **raw_payload,
        "raw_score": raw_score,
        "change_rate": row["change_rate"],
        "volume_ratio": row["volume_ratio"],
        "turnover_value": row["turnover_value"],
        "news_count": row["news_count"],
        "disclosure_count": row["disclosure_count"],
    }
    feature_map = candidate_feature_map(candidate, raw_score)
    return {
        "source_candidate_id": int(row["id"]),
        "symbol": str(row["symbol"]),
        "scan_time": scan_time,
        "observed_at": observed_at,
        "entry_price": entry_price,
        "observed_price": final_price,
        "features": [feature_map[name] for name in FEATURE_NAMES],
        "realized_return_bps": realized_return_bps,
        "realized_risk_bps": realized_risk_bps,
        "rank": _to_int(row["rank"]),
        "status": row["status"],
        "raw_json": {
            "candidate": raw_payload,
            "feature_map": feature_map,
            "source": "scanner_candidate_history",
        },
    }


def _refresh_training_samples(
    *,
    calibration_path: Path,
    universe_path: Path,
    horizon_seconds: int,
    candidate_limit: int,
) -> dict[str, int]:
    inserted_count = 0
    skipped_count = 0
    examined_count = 0
    if not universe_path.exists():
        return {
            "examined_count": 0,
            "inserted_count": 0,
            "skipped_count": 0,
        }
    initialize_edge_calibration_db(calibration_path)
    try:
        with sqlite3.connect(universe_path) as source_conn, sqlite3.connect(calibration_path) as target_conn:
            source_conn.row_factory = sqlite3.Row
            target_conn.executescript(SCHEMA_SQL)
            _ensure_schema_migrations(target_conn)
            cursor = source_conn.execute(
                """
                SELECT *
                FROM scanner_candidate_history
                WHERE current_price IS NOT NULL
                  AND current_price > 0
                ORDER BY scan_time DESC, id DESC
                LIMIT ?
                """,
                (max(1, int(candidate_limit)),),
            )
            while True:
                rows = cursor.fetchmany(50)
                if not rows:
                    break
                for row in rows:
                    examined_count += 1
                    payload = _training_payload_from_candidate_row(
                        conn=source_conn,
                        row=row,
                        horizon_seconds=horizon_seconds,
                    )
                    if payload is None:
                        skipped_count += 1
                        continue
                    inserted_count += _store_training_sample(target_conn, payload)
            _prune_training_samples(target_conn)
    except sqlite3.Error:
        return {
            "examined_count": examined_count,
            "inserted_count": inserted_count,
            "skipped_count": skipped_count,
        }
    return {
        "examined_count": examined_count,
        "inserted_count": inserted_count,
        "skipped_count": skipped_count,
    }


def _store_training_sample(
    conn: sqlite3.Connection,
    payload: dict[str, Any],
) -> int:
    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO edge_training_samples (
            source_candidate_id, symbol, scan_time, observed_at, entry_price,
            observed_price, features_json, realized_return_bps,
            realized_risk_bps, rank, status, created_at, raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload["source_candidate_id"],
            payload["symbol"],
            payload["scan_time"],
            payload["observed_at"],
            payload["entry_price"],
            payload["observed_price"],
            _json(payload["features"]),
            payload["realized_return_bps"],
            payload["realized_risk_bps"],
            payload.get("rank"),
            payload.get("status"),
            _now(),
            _json(payload.get("raw_json") or {}),
        ),
    )
    return int(cursor.rowcount or 0)


def _prune_training_samples(conn: sqlite3.Connection) -> None:
    retention = max(1000, int(settings.edge_calibration_sample_retention_limit or 10_000))
    conn.execute(
        """
        DELETE FROM edge_training_samples
        WHERE id NOT IN (
            SELECT id
            FROM edge_training_samples
            ORDER BY observed_at DESC, id DESC
            LIMIT ?
        )
        """,
        (retention,),
    )


def _iter_training_samples_from_store(
    *,
    calibration_path: Path,
    limit: int,
) -> Iterator[tuple[list[float], float, float]]:
    initialize_edge_calibration_db(calibration_path)
    with sqlite3.connect(calibration_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT features_json, realized_return_bps, realized_risk_bps
            FROM (
                SELECT *
                FROM edge_training_samples
                ORDER BY observed_at DESC, id DESC
                LIMIT ?
            )
            ORDER BY observed_at ASC, id ASC
            """,
            (max(1, int(limit)),),
        )
        while True:
            chunk = rows.fetchmany(100)
            if not chunk:
                break
            for row in chunk:
                sample = _training_sample_from_row(row)
                if sample is not None:
                    yield sample


def _training_sample_from_row(
    row: sqlite3.Row,
) -> tuple[list[float], float, float] | None:
    features = _parse_json(row["features_json"], None)
    if not isinstance(features, list) or len(features) != len(FEATURE_NAMES):
        return None
    try:
        return (
            [float(value) for value in features],
            float(row["realized_return_bps"]),
            float(row["realized_risk_bps"]),
        )
    except (TypeError, ValueError):
        return None


def _evaluate_model(
    *,
    universe_path: Path,
    max_samples: int,
    horizon_seconds: int,
    return_coefficients: dict[str, float],
    risk_coefficients: dict[str, float],
) -> dict[str, float | None]:
    count = 0
    return_error = 0.0
    risk_error = 0.0
    for sample in _iter_calibration_samples(
        universe_path=universe_path,
        max_samples=max_samples,
        horizon_seconds=horizon_seconds,
    ):
        if sample is None:
            continue
        features, realized_return_bps, realized_risk_bps = sample
        feature_map = dict(zip(FEATURE_NAMES, features, strict=True))
        return_error += abs(_predict(return_coefficients, feature_map) - realized_return_bps)
        risk_error += abs(_predict(risk_coefficients, feature_map) - realized_risk_bps)
        count += 1
    return {
        "mae_return_bps": round(return_error / count, 4) if count else None,
        "mae_risk_bps": round(risk_error / count, 4) if count else None,
    }


def _performance_split(
    samples: list[tuple[list[float], float, float]],
) -> tuple[
    list[tuple[list[float], float, float]],
    list[tuple[list[float], float, float]],
]:
    if len(samples) <= 1:
        return samples, []
    split_index = int(len(samples) * 0.8)
    split_index = max(1, min(split_index, len(samples) - 1))
    return samples[:split_index], samples[split_index:]


def _model_metrics_for_samples(
    samples: list[tuple[list[float], float, float]],
    *,
    return_coefficients: dict[str, float],
    risk_coefficients: dict[str, float],
) -> dict[str, float | None]:
    count = 0
    return_error = 0.0
    risk_error = 0.0
    for features, realized_return_bps, realized_risk_bps in samples:
        feature_map = dict(zip(FEATURE_NAMES, features, strict=True))
        return_error += abs(_predict(return_coefficients, feature_map) - realized_return_bps)
        risk_error += abs(_predict(risk_coefficients, feature_map) - realized_risk_bps)
        count += 1
    return {
        "sample_count": count,
        "mae_return_bps": round(return_error / count, 4) if count else None,
        "mae_risk_bps": round(risk_error / count, 4) if count else None,
    }


def _top10_performance_from_store(
    *,
    calibration_path: Path,
    limit: int,
) -> dict[str, Any]:
    initialize_edge_calibration_db(calibration_path)
    with sqlite3.connect(calibration_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT realized_return_bps, rank, symbol, scan_time
            FROM edge_training_samples
            WHERE rank IS NOT NULL
              AND rank <= 10
            ORDER BY observed_at DESC, id DESC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        ).fetchall()
    returns = [
        float(row["realized_return_bps"])
        for row in rows
        if _to_float(row["realized_return_bps"]) is not None
    ]
    if not returns:
        return {
            "status": "empty",
            "sample_count": 0,
            "top_count": 10,
            "avg_return_bps": None,
            "win_rate": None,
        }
    wins = sum(1 for value in returns if value > 0)
    return {
        "status": "ready",
        "sample_count": len(returns),
        "top_count": 10,
        "avg_return_bps": round(sum(returns) / len(returns), 4),
        "win_rate": round(wins / len(returns), 6),
    }


def _record_top10_performance(
    path: Path,
    performance: dict[str, Any],
) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
        _ensure_schema_migrations(conn)
        conn.execute(
            """
            INSERT INTO top_candidate_performance (
                evaluated_at, sample_count, top_count, avg_return_bps,
                win_rate, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                _now(),
                int(performance.get("sample_count") or 0),
                int(performance.get("top_count") or 10),
                performance.get("avg_return_bps"),
                performance.get("win_rate"),
                _json(performance),
            ),
        )


def _stored_sample_count(path: Path) -> int:
    initialize_edge_calibration_db(path)
    with sqlite3.connect(path) as conn:
        return int(
            conn.execute("SELECT COUNT(*) FROM edge_training_samples").fetchone()[0]
            or 0
        )


def _accumulate(
    matrix: list[list[float]],
    vector: list[float],
    features: list[float],
    target: float,
) -> None:
    for row_index, row_value in enumerate(features):
        vector[row_index] += row_value * target
        for col_index, col_value in enumerate(features):
            matrix[row_index][col_index] += row_value * col_value


def _ridge_matrix(dimension: int, ridge_lambda: float) -> list[list[float]]:
    return [
        [ridge_lambda if row == col else 0.0 for col in range(dimension)]
        for row in range(dimension)
    ]


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for pivot_index in range(size):
        pivot_row = max(
            range(pivot_index, size),
            key=lambda row_index: abs(augmented[row_index][pivot_index]),
        )
        if abs(augmented[pivot_row][pivot_index]) < 1e-12:
            continue
        if pivot_row != pivot_index:
            augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]
        pivot = augmented[pivot_index][pivot_index]
        augmented[pivot_index] = [value / pivot for value in augmented[pivot_index]]
        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            if factor == 0:
                continue
            augmented[row_index] = [
                value - factor * augmented[pivot_index][col_index]
                for col_index, value in enumerate(augmented[row_index])
            ]
    return [augmented[index][-1] for index in range(size)]


def _coefficients_from_solution(values: list[float]) -> dict[str, float]:
    return {
        feature: max(-750.0, min(750.0, float(values[index])))
        for index, feature in enumerate(FEATURE_NAMES)
    }


def _blend_coefficients(
    previous: dict[str, float],
    fitted: dict[str, float],
    *,
    blend: float,
) -> dict[str, float]:
    return {
        feature: round(
            float(previous.get(feature, 0.0)) * (1.0 - blend)
            + float(fitted.get(feature, 0.0)) * blend,
            6,
        )
        for feature in FEATURE_NAMES
    }


def _store_coefficients(
    path: Path,
    *,
    run_id: int,
    return_coefficients: dict[str, float],
    risk_coefficients: dict[str, float],
    updated_at: str,
) -> None:
    rows = []
    for target, coefficients in (
        ("expected_return", return_coefficients),
        ("expected_risk", risk_coefficients),
    ):
        rows.extend(
            (target, feature, float(coefficients[feature]), updated_at, run_id)
            for feature in FEATURE_NAMES
        )
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
        _ensure_schema_migrations(conn)
        conn.executemany(
            """
            INSERT INTO edge_coefficients (
                target, feature, coefficient, updated_at, run_id
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(target, feature) DO UPDATE SET
                coefficient = excluded.coefficient,
                updated_at = excluded.updated_at,
                run_id = excluded.run_id
            """,
            rows,
        )


def _record_run(path: Path, result: dict[str, Any]) -> int:
    now = _now()
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
        _ensure_schema_migrations(conn)
        cursor = conn.execute(
            """
            INSERT INTO edge_calibration_runs (
                created_at, status, sample_count, skipped_count, horizon_seconds,
                mae_return_bps, mae_risk_bps, gate_approved, oos_sample_count,
                top10_avg_return_bps, top10_win_rate, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                result.get("status", "unknown"),
                int(result.get("sample_count") or 0),
                int(result.get("skipped_count") or 0),
                int(result.get("horizon_seconds") or settings.edge_calibration_horizon_seconds),
                result.get("mae_return_bps"),
                result.get("mae_risk_bps"),
                1 if (result.get("gate") or {}).get("approved") else 0,
                int(result.get("oos_sample_count") or 0),
                (result.get("top10_performance") or {}).get("avg_return_bps"),
                (result.get("top10_performance") or {}).get("win_rate"),
                _json(result),
            ),
        )
        return int(cursor.lastrowid)


def _default_gate(
    *,
    status: str,
    approved: bool,
    message: str,
    sample_count: int = 0,
) -> dict[str, Any]:
    return {
        "status": status,
        "approved": approved,
        "message": message,
        "sample_count": sample_count,
        "required": {
            "min_samples": settings.edge_calibration_gate_min_samples,
            "min_oos_samples": settings.edge_calibration_gate_min_oos_samples,
            "max_mae_return_bps": settings.edge_calibration_gate_max_mae_return_bps,
            "max_mae_risk_bps": settings.edge_calibration_gate_max_mae_risk_bps,
            "min_top10_avg_return_bps": settings.edge_calibration_gate_min_top10_avg_return_bps,
            "min_top10_win_rate": settings.edge_calibration_gate_min_top10_win_rate,
            "min_fill_adjusted_edge_bps": settings.edge_calibration_gate_min_fill_adjusted_edge_bps,
        },
    }


def _gate_from_metrics(
    *,
    sample_count: int,
    oos_sample_count: int,
    mae_return_bps: float | None,
    mae_risk_bps: float | None,
    top10_performance: dict[str, Any],
    fill_adjustment: dict[str, Any],
    candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    min_samples = int(settings.edge_calibration_gate_min_samples or 1000)
    min_oos = int(settings.edge_calibration_gate_min_oos_samples or 200)
    max_return_mae = float(settings.edge_calibration_gate_max_mae_return_bps or 180.0)
    max_risk_mae = float(settings.edge_calibration_gate_max_mae_risk_bps or 180.0)
    min_top10_return = float(settings.edge_calibration_gate_min_top10_avg_return_bps or 20.0)
    min_top10_win_rate = float(settings.edge_calibration_gate_min_top10_win_rate or 0.52)
    min_fill_adjusted_edge = float(
        settings.edge_calibration_gate_min_fill_adjusted_edge_bps or 30.0
    )
    multiplier = _to_float(fill_adjustment.get("multiplier")) or 1.0
    best_fill_adjusted_edge: float | None = None

    if sample_count < min_samples:
        failures.append(f"sample_count {sample_count}/{min_samples}")
    if oos_sample_count < min_oos:
        failures.append(f"oos_sample_count {oos_sample_count}/{min_oos}")
    if mae_return_bps is None or mae_return_bps > max_return_mae:
        failures.append(f"mae_return_bps {mae_return_bps} > {max_return_mae}")
    if mae_risk_bps is None or mae_risk_bps > max_risk_mae:
        failures.append(f"mae_risk_bps {mae_risk_bps} > {max_risk_mae}")

    top10_avg_return = _to_float(top10_performance.get("avg_return_bps"))
    top10_win_rate = _to_float(top10_performance.get("win_rate"))
    if top10_avg_return is None or top10_avg_return < min_top10_return:
        failures.append(f"top10_avg_return_bps {top10_avg_return} < {min_top10_return}")
    if top10_win_rate is None or top10_win_rate < min_top10_win_rate:
        failures.append(f"top10_win_rate {top10_win_rate} < {min_top10_win_rate}")

    if candidates:
        candidate_edges = [
            float(candidate.get("net_edge") or 0.0) * multiplier
            for candidate in candidates
            if candidate.get("net_edge") is not None
        ]
        if candidate_edges:
            best_fill_adjusted_edge = max(candidate_edges)
            if best_fill_adjusted_edge < min_fill_adjusted_edge:
                failures.append(
                    "best_fill_adjusted_edge_bps "
                    f"{best_fill_adjusted_edge:.2f} < {min_fill_adjusted_edge:.2f}"
                )

    approved = not failures
    return {
        "status": "approved" if approved else "blocked",
        "approved": approved,
        "message": (
            "Calibration performance gate passed"
            if approved
            else "Calibration performance gate blocked entries: " + "; ".join(failures)
        ),
        "sample_count": sample_count,
        "oos_sample_count": oos_sample_count,
        "mae_return_bps": mae_return_bps,
        "mae_risk_bps": mae_risk_bps,
        "top10_performance": top10_performance,
        "fill_adjustment": fill_adjustment,
        "best_fill_adjusted_edge_bps": (
            round(best_fill_adjusted_edge, 4)
            if best_fill_adjusted_edge is not None
            else None
        ),
        "required": {
            "min_samples": min_samples,
            "min_oos_samples": min_oos,
            "max_mae_return_bps": max_return_mae,
            "max_mae_risk_bps": max_risk_mae,
            "min_top10_avg_return_bps": min_top10_return,
            "min_top10_win_rate": min_top10_win_rate,
            "min_fill_adjusted_edge_bps": min_fill_adjusted_edge,
        },
    }


def _paper_fill_quality_events(path: Path, *, limit: int) -> list[dict[str, float]]:
    if not path.exists():
        return []
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT side, requested_quantity, filled_quantity,
                       order_price, fill_price, slippage_bps
                FROM paper_orders
                ORDER BY id DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
    except sqlite3.Error:
        return []
    events: list[dict[str, float]] = []
    for row in rows:
        event = _fill_quality_from_row(row)
        if event:
            events.append(event)
    return events


def _broker_fill_quality_events(path: Path, *, limit: int) -> list[dict[str, float]]:
    if not path.exists():
        return []
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT side, order_qty AS requested_quantity,
                       filled_qty AS filled_quantity, order_price,
                       avg_fill_price AS fill_price
                FROM broker_order_executions
                ORDER BY synced_at DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
    except sqlite3.Error:
        return []
    events: list[dict[str, float]] = []
    for row in rows:
        event = _fill_quality_from_row(row)
        if event:
            events.append(event)
    return events


def _fill_quality_from_row(row: sqlite3.Row) -> dict[str, float] | None:
    requested = _to_float(row["requested_quantity"])
    filled = _to_float(row["filled_quantity"])
    order_price = _to_float(row["order_price"])
    fill_price = _to_float(row["fill_price"])
    if requested is None or requested <= 0 or filled is None:
        return None
    fill_ratio = max(0.0, min(1.0, filled / requested))
    adverse_slippage = 0.0
    if order_price and order_price > 0 and fill_price and fill_price > 0:
        side = str(row["side"] or "").upper()
        if side == "SELL":
            adverse_slippage = max(0.0, (order_price - fill_price) / order_price * 10_000)
        else:
            adverse_slippage = max(0.0, (fill_price - order_price) / order_price * 10_000)
    elif "slippage_bps" in row.keys():
        side = str(row["side"] or "").upper()
        slippage = _to_float(row["slippage_bps"]) or 0.0
        adverse_slippage = max(0.0, -slippage if side == "SELL" else slippage)
    return {
        "fill_ratio": fill_ratio,
        "adverse_slippage_bps": adverse_slippage,
    }


def _latest_raw_json(path: Path, key: str) -> Any:
    initialize_edge_calibration_db(path)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT raw_json
            FROM edge_calibration_runs
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    if not row:
        return None
    raw = _parse_json(row["raw_json"], {})
    return raw.get(key)


def _predict(coefficients: dict[str, float], feature_map: dict[str, float]) -> float:
    return sum(
        float(coefficients.get(feature, 0.0)) * float(feature_map.get(feature, 0.0))
        for feature in FEATURE_NAMES
    )


def _read_meta(path: Path, key: str) -> str | None:
    initialize_edge_calibration_db(path)
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT value FROM edge_calibration_meta WHERE key = ?",
            (key,),
        ).fetchone()
    return str(row[0]) if row else None


def _write_meta(path: Path, key: str, value: str) -> None:
    now = _now()
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
        _ensure_schema_migrations(conn)
        conn.execute(
            """
            INSERT INTO edge_calibration_meta (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (key, value, now),
        )


def _ensure_schema_migrations(conn: sqlite3.Connection) -> None:
    _ensure_column(conn, "edge_calibration_runs", "gate_approved", "INTEGER")
    _ensure_column(
        conn,
        "edge_calibration_runs",
        "oos_sample_count",
        "INTEGER NOT NULL DEFAULT 0",
    )
    _ensure_column(conn, "edge_calibration_runs", "top10_avg_return_bps", "REAL")
    _ensure_column(conn, "edge_calibration_runs", "top10_win_rate", "REAL")


def _ensure_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    columns = {
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    return int(number) if number is not None else None


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
    return Path(db_path or settings.edge_calibration_db_path)
