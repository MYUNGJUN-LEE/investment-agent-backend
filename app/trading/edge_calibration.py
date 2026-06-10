from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
import sqlite3
from typing import Any, Iterator

from app.config import settings
from app.storage.sqlite import connect_sqlite
from app.trading import paper_trading

logger = logging.getLogger(__name__)


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

     # v2 features for 1~2 day swing trading
    "market_regime",
    "market_breadth",
    "index_return_1d",
    "relative_strength_3d",
    "relative_strength_5d",
    "atr_pct",
    "volatility_10d",
    "close_position",
    "intraday_recovery",
    "overheat_score",
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

     # v2 return priors
    "market_regime": 60.0,
    "market_breadth": 40.0,
    "index_return_1d": 30.0,
    "relative_strength_3d": 70.0,
    "relative_strength_5d": 50.0,
    "atr_pct": -20.0,
    "volatility_10d": -25.0,
    "close_position": 45.0,
    "intraday_recovery": 50.0,
    "overheat_score": -100.0,
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

     # v2 risk priors
    "market_regime": -30.0,
    "market_breadth": -20.0,
    "index_return_1d": -10.0,
    "relative_strength_3d": -20.0,
    "relative_strength_5d": -10.0,
    "atr_pct": 80.0,
    "volatility_10d": 90.0,
    "close_position": -20.0,
    "intraday_recovery": -20.0,
    "overheat_score": 120.0,
}


REALIZED_RISK_WEIGHT = 0.10

# ---------------------------------------------------------------------
# Trading cost / slippage policy
# ---------------------------------------------------------------------
# Unit convention:
# - percent: 0.147 means 0.147%
# - bps: 1% = 100 bps, 0.147% = 14.7 bps
#
# Current policy:
# - Buy fee: 0.147%
# - Sell fee: 0.147%
# - Max allowed slippage: 0.10%
# - Sell tax is not included here. Add it to SELL_TAX_PCT if needed.
# ---------------------------------------------------------------------

EXPECTED_RISK_WEIGHT = REALIZED_RISK_WEIGHT

BUY_FEE_PCT = 0.147
SELL_FEE_PCT = 0.147
SELL_TAX_PCT = 0.2

MAX_ALLOWED_SLIPPAGE_PCT = 0.10


def percent_to_bps(percent: float) -> float:
    """Convert percent value to bps. Example: 0.147% -> 14.7 bps."""
    return float(percent) * 100.0


def estimate_round_trip_trading_cost_bps(
    *,
    buy_fee_pct: float = BUY_FEE_PCT,
    sell_fee_pct: float = SELL_FEE_PCT,
    sell_tax_pct: float = SELL_TAX_PCT,
) -> float:
    """Return round-trip trading cost in bps."""
    return percent_to_bps(buy_fee_pct + sell_fee_pct + sell_tax_pct)


DEFAULT_ROUND_TRIP_TRADING_COST_BPS = estimate_round_trip_trading_cost_bps()
MAX_ALLOWED_SLIPPAGE_BPS = percent_to_bps(MAX_ALLOWED_SLIPPAGE_PCT)


EXPECTED_NET_EDGE_FORMULA = (
    f"expected_return_bps - expected_risk_bps * {EXPECTED_RISK_WEIGHT:.2f} "
    "- trading_cost_bps - slippage_cost_bps - liquidity_drag_bps"
)


def estimate_slippage_cost_bps(
    candidate: dict[str, Any],
    *,
    default_slippage_bps: float = MAX_ALLOWED_SLIPPAGE_BPS,
) -> float:
    """
    Estimate slippage cost in bps.

    If candidate already has slippage_cost or slippage_cost_bps, use it.
    Otherwise use the max allowed slippage policy: 0.10% = 10 bps.
    """
    raw_slippage = (
        _to_float(candidate.get("slippage_cost"))
        if candidate.get("slippage_cost") is not None
        else _to_float(candidate.get("slippage_cost_bps"))
    )

    if raw_slippage is None:
        return float(default_slippage_bps)

    return max(0.0, float(raw_slippage))


def is_slippage_allowed(
    slippage_cost_bps: float,
    *,
    max_allowed_slippage_bps: float = MAX_ALLOWED_SLIPPAGE_BPS,
) -> bool:
    """Return whether estimated slippage is within the allowed limit."""
    return float(slippage_cost_bps) <= float(max_allowed_slippage_bps)


def estimate_liquidity_drag_bps(
    candidate: dict[str, Any],
    *,
    default_order_value: float | None = None,
    impact_k_bps: float = 50.0,
) -> float:
    """
    Estimate liquidity/market-impact drag in bps.

    Formula:
        participation_rate = order_value / turnover_value
        liquidity_drag_bps = impact_k_bps * sqrt(participation_rate)

    If order_value or turnover_value is missing, return 0.
    """
    turnover = _to_float(candidate.get("turnover_value"))
    order_value = (
        _to_float(candidate.get("order_value"))
        or _to_float(candidate.get("position_value"))
        or default_order_value
        or 0.0
    )

    if turnover is None or turnover <= 0 or order_value <= 0:
        return 0.0

    participation = max(0.0, min(0.05, order_value / turnover))
    return impact_k_bps * (participation ** 0.5)


def estimate_expected_net_edge_bps(
    *,
    expected_return_bps: float,
    expected_risk_bps: float,
    trading_cost_bps: float = DEFAULT_ROUND_TRIP_TRADING_COST_BPS,
    slippage_cost_bps: float = MAX_ALLOWED_SLIPPAGE_BPS,
    liquidity_drag_bps: float = 0.0,
) -> float:
    """
    Estimate net edge after risk penalty, trading cost, slippage, and liquidity drag.

    Formula:
        expected_net_edge
        = expected_return
        - expected_risk * EXPECTED_RISK_WEIGHT
        - trading_cost
        - slippage_cost
        - liquidity_drag
    """
    return (
        float(expected_return_bps)
        - float(expected_risk_bps) * EXPECTED_RISK_WEIGHT
        - float(trading_cost_bps)
        - float(slippage_cost_bps)
        - float(liquidity_drag_bps)
    )

EXPECTED_RISK_WEIGHT = 0.10

EXPECTED_NET_EDGE_FORMULA = (
    f"expected_return_bps - expected_risk_bps * {EXPECTED_RISK_WEIGHT:.2f} "
    "- trading_cost_bps - slippage_cost_bps - liquidity_drag_bps"
)

ELIGIBLE_EDGE_SAMPLE_STATUSES = {
    "BUY",
    "BUY_CANDIDATE",
    "WATCH",
    "CANDIDATE",
    "EXECUTABLE",
    "READY",
    "CLAIMED",
}

INELIGIBLE_EDGE_SAMPLE_STATUSES = {
    "EXCLUDED",
    "SKIPPED",
    "ARCHIVED",
    "BLOCKED",
    "NOT_EXECUTABLE",
}

ELIGIBLE_EDGE_SAMPLE_STATUS_SQL = (
    "UPPER(COALESCE(status, '')) IN "
    "('BUY', 'BUY_CANDIDATE', 'WATCH', 'CANDIDATE', 'EXECUTABLE', 'READY', 'CLAIMED')"
)


REALIZED_NET_EDGE_SQL = (
    f"realized_return_bps - (realized_risk_bps * {REALIZED_RISK_WEIGHT:.2f}) "
    "- COALESCE(trading_cost_bps, 0) - COALESCE(slippage_cost_bps, 0)"
)

REALIZED_NET_EDGE_FORMULA = (
    f"realized_return_bps - realized_risk_bps * {REALIZED_RISK_WEIGHT:.2f} "
    "- trading_cost_bps - slippage_cost_bps"
)

COMPOSITE_SCORE_FORMULA = (
    "raw_score * 0.10 + expected_return_score * 0.20 "
    "+ net_edge_score * 0.45 - expected_risk_score * 0.20 "
    "+ executable_status_bonus"
)

TOP10_SAMPLE_SOURCE_SCAN_RUN = "scan_run_deduped_top10_samples"
CANDIDATE_LABEL_UNIT = "candidate_label"
PAPER_ORDER_UNIT = "paper_order"
ACTUAL_BROKER_FILL_UNIT = "actual_broker_fill"
BROKER_PAPER_OBSERVE_ONLY_REASON = (
    "broker_paper bootstrap observe-only candidate-label calibration"
)

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
    scan_id TEXT,
    label_horizon_key TEXT,
    symbol TEXT NOT NULL,
    scan_time TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    entry_price REAL NOT NULL,
    observed_price REAL NOT NULL,
    features_json TEXT NOT NULL,
    realized_return_bps REAL NOT NULL,
    realized_risk_bps REAL NOT NULL,
    label_observation_span_seconds INTEGER,
    raw_score REAL,
    expected_return_bps REAL,
    expected_risk_bps REAL,
    trading_cost_bps REAL,
    slippage_cost_bps REAL,
    net_edge_bps REAL,
    composite_score REAL,
    rank INTEGER,
    status TEXT,
    market_segment TEXT,
    sector TEXT,
    theme TEXT,
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

    # Fixed round-trip trading cost:
    # Buy fee 0.147% + sell fee 0.147% = 0.294% = 29.4 bps
    trading_cost_bps = DEFAULT_ROUND_TRIP_TRADING_COST_BPS

    # Slippage policy:
    # Default/worst allowed slippage = 0.10% = 10 bps.
    # If candidate already has slippage estimate, use that estimate.
    slippage_cost_bps = estimate_slippage_cost_bps(candidate)
    slippage_allowed = is_slippage_allowed(slippage_cost_bps)

    # Optional liquidity drag.
    # If candidate has order_value/position_value and turnover_value,
    # this adds market-impact cost.
    liquidity_drag_bps = estimate_liquidity_drag_bps(candidate)

    expected_return = max(0.0, min(500.0, expected_return))
    expected_risk = max(0.0, min(500.0, expected_risk))

    expected_net_edge = estimate_expected_net_edge_bps(
        expected_return_bps=expected_return,
        expected_risk_bps=expected_risk,
        trading_cost_bps=trading_cost_bps,
        slippage_cost_bps=slippage_cost_bps,
        liquidity_drag_bps=liquidity_drag_bps,
    )

    total_cost_bps = trading_cost_bps + slippage_cost_bps + liquidity_drag_bps

    return {
        "expected_return": round(expected_return, 4),
        "expected_risk": round(expected_risk, 4),

        # Net edge after cost/risk adjustment
        "expected_net_edge": round(expected_net_edge, 4),

        # Cost breakdown
        "trading_cost_bps": round(trading_cost_bps, 4),
        "buy_fee_pct": BUY_FEE_PCT,
        "sell_fee_pct": SELL_FEE_PCT,
        "sell_tax_pct": SELL_TAX_PCT,
        "round_trip_fee_pct": round(BUY_FEE_PCT + SELL_FEE_PCT + SELL_TAX_PCT, 6),

        "slippage_cost_bps": round(slippage_cost_bps, 4),
        "max_allowed_slippage_bps": round(MAX_ALLOWED_SLIPPAGE_BPS, 4),
        "max_allowed_slippage_pct": MAX_ALLOWED_SLIPPAGE_PCT,
        "slippage_allowed": slippage_allowed,

        "liquidity_drag_bps": round(liquidity_drag_bps, 4),
        "total_cost_bps": round(total_cost_bps, 4),

        "expected_net_edge_formula": EXPECTED_NET_EDGE_FORMULA,
        "edge_model": "calibrated_ridge_v2_cost_adjusted",
        "fill_adjustment": round(fill_adjustment, 4),
    }

def _normalize_sample_status(value: Any) -> str:
    return str(value or "").strip().upper()


def _is_edge_training_sample_status_allowed(value: Any) -> bool:
    status = _normalize_sample_status(value)
    if status in INELIGIBLE_EDGE_SAMPLE_STATUSES:
        return False
    return status in ELIGIBLE_EDGE_SAMPLE_STATUSES


def _row_value(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError):
        return default


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
    _purge_invalid_label_samples_by_path(path)
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


def refresh_top10_performance_if_due(
    *,
    calibration_db_path: Path | str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    path = _db_path(calibration_db_path)
    initialize_edge_calibration_db(path)
    interval = max(
        60,
        int(settings.edge_calibration_top10_performance_interval_seconds or 600),
    )
    now = _now()
    last_run_at = _read_meta(path, "last_top10_performance_at")
    if not force and last_run_at:
        try:
            elapsed = (
                datetime.fromisoformat(now) - datetime.fromisoformat(last_run_at)
            ).total_seconds()
        except ValueError:
            elapsed = interval
        if elapsed < interval:
            return {
                "status": "not_due",
                "last_run_at": last_run_at,
                "next_run_at": (
                    datetime.fromisoformat(last_run_at) + timedelta(seconds=interval)
                ).isoformat(timespec="seconds"),
                "interval_seconds": interval,
                "top10_performance": (
                    _latest_top10_performance_from_store(path)
                    or _top10_performance_from_store(calibration_path=path)
                ),
            }

    performance = _top10_performance_from_store(calibration_path=path)
    _record_top10_performance(path, performance)
    _write_meta(path, "last_top10_performance_at", now)
    return {
        "status": "success",
        "last_run_at": now,
        "interval_seconds": interval,
        "top10_performance": performance,
    }


def refresh_edge_training_samples(
    *,
    universe_db_path: Path | str | None = None,
    calibration_db_path: Path | str | None = None,
    horizon_seconds: int | None = None,
    candidate_limit: int | None = None,
) -> dict[str, Any]:
    """Persist realized labels for scanner predictions without refitting the model."""
    if not settings.edge_calibration_enabled:
        return {"status": "disabled", "message": "Edge calibration is disabled"}

    universe_path = settings.storage_path(
        universe_db_path or settings.universe_scanner_db_path
    )
    calibration_path = _db_path(calibration_db_path)
    if not universe_path.exists():
        return {
            "status": "empty",
            "message": "Universe scanner DB does not exist yet",
            "examined_count": 0,
            "inserted_count": 0,
            "skipped_count": 0,
            "stored_sample_count": _stored_sample_count(calibration_path),
        }

    resolved_horizon = max(
        60,
        int(horizon_seconds or settings.edge_calibration_horizon_seconds or 86400),
    )
    snapshot_sync = _append_labeling_price_snapshots(
        universe_path=universe_path,
        horizon_seconds=resolved_horizon,
    )
    refresh_result = _refresh_training_samples(
        calibration_path=calibration_path,
        universe_path=universe_path,
        horizon_seconds=resolved_horizon,
        candidate_limit=candidate_limit,
    )
    return {
        "status": "success",
        **refresh_result,
        "label_snapshot_sync": snapshot_sync,
        "label_policy": label_policy_summary(),
        "stored_sample_count": _stored_sample_count(calibration_path),
    }


def calibrate_edge_model(
    *,
    universe_db_path: Path | str | None = None,
    calibration_db_path: Path | str | None = None,
    max_samples: int | None = None,
    min_samples: int | None = None,
) -> dict[str, Any]:
    """Fit tiny ridge models from scanner history without loading large frames."""
    calibration_path = _db_path(calibration_db_path)
    universe_path = settings.storage_path(
        universe_db_path or settings.universe_scanner_db_path
    )
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

    max_samples = max(1, int(max_samples or settings.edge_calibration_max_samples or 5000))
    min_samples = max(1, int(min_samples or settings.edge_calibration_min_samples or 30))
    horizon_seconds = max(60, int(settings.edge_calibration_horizon_seconds or 86400))
    ridge_lambda = max(0.0, float(settings.edge_calibration_ridge_lambda or 0.0))
    blend = max(0.0, min(1.0, float(settings.edge_calibration_blend or 0.35)))
    target_samples = max(
        min_samples,
        int(settings.edge_calibration_target_samples or 3000),
    )
    training_limit = max(max_samples, target_samples)

    snapshot_sync = _append_labeling_price_snapshots(
        universe_path=universe_path,
        horizon_seconds=horizon_seconds,
    )
    refresh_result = _refresh_training_samples(
        calibration_path=calibration_path,
        universe_path=universe_path,
        horizon_seconds=horizon_seconds,
        candidate_limit=None,
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
            "label_snapshot_sync": snapshot_sync,
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
    top10_performance = _top10_performance_from_store(calibration_path=calibration_path)
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
        "label_snapshot_sync": snapshot_sync,
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
            f"""
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
            f"SELECT COUNT(*) FROM edge_training_samples WHERE {ELIGIBLE_EDGE_SAMPLE_STATUS_SQL}"
        ).fetchone()[0]
    return {
        "status": "ready" if coefficient_count else "empty",
        "enabled": settings.edge_calibration_enabled,
        "coefficient_count": int(coefficient_count or 0),
        "stored_sample_count": int(stored_sample_count or 0),
        "last_run": dict(run) if run else None,
        "latest_gate": _latest_raw_json(path, "gate"),
        "latest_top10_performance": (
            _latest_top10_performance_from_store(path)
            or _latest_raw_json(path, "top10_performance")
        ),
        "last_top10_performance_at": _read_meta(path, "last_top10_performance_at"),
        "last_success_at": _read_meta(path, "last_success_at"),
        "interval_seconds": settings.edge_calibration_interval_seconds,
        "horizon_seconds": settings.edge_calibration_horizon_seconds,
        "max_samples": settings.edge_calibration_max_samples,
        "min_samples": settings.edge_calibration_min_samples,
    }


def get_edge_training_sample_summary(
    *,
    calibration_db_path: Path | str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    path = _db_path(calibration_db_path)
    if not path.exists():
        diagnostics = get_edge_data_diagnostics(calibration_db_path=path)
        return {
            "status": "empty",
            "message": _empty_sample_message(diagnostics),
            "sample_count": 0,
            "summary": _empty_sample_summary(),
            "top10_performance": _empty_top10_performance(),
            "unit_performance": _empty_unit_performance(),
            "label_policy": label_policy_summary(),
            "recent_samples": [],
            "symbol_summary": [],
            "diagnostics": diagnostics,
        }

    initialize_edge_calibration_db(path)
    _purge_invalid_label_samples_by_path(path)
    limit = max(1, min(int(limit or 20), 100))
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        aggregate = conn.execute(
            f"""
            WITH canonical AS (
                SELECT *
                FROM (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (
                            PARTITION BY symbol, COALESCE(label_horizon_key, symbol || ':' || scan_time)
                            ORDER BY observed_at ASC, id ASC
                        ) AS symbol_horizon_rank
                    FROM edge_training_samples
                    WHERE {ELIGIBLE_EDGE_SAMPLE_STATUS_SQL}
                )
                WHERE symbol_horizon_rank = 1
            )
            SELECT
                COUNT(*) AS sample_count,
                COALESCE(SUM(realized_return_bps), 0) AS total_return_bps,
                COALESCE(SUM(realized_risk_bps), 0) AS total_risk_bps,
                COALESCE(SUM(
                    realized_return_bps - (realized_risk_bps * 0.10)
                    - COALESCE(trading_cost_bps, 0)
                    - COALESCE(slippage_cost_bps, 0)
                ), 0) AS total_realized_net_edge_bps,
                COALESCE(SUM(expected_return_bps), 0) AS total_expected_return_bps,
                COALESCE(SUM(expected_risk_bps), 0) AS total_expected_risk_bps,
                COALESCE(SUM(net_edge_bps), 0) AS total_predicted_net_edge_bps,
                COALESCE(SUM(COALESCE(trading_cost_bps, 0) + COALESCE(slippage_cost_bps, 0)), 0) AS total_cost_bps,
                AVG(realized_return_bps) AS avg_return_bps,
                AVG(realized_risk_bps) AS avg_risk_bps,
                AVG(expected_return_bps) AS avg_expected_return_bps,
                AVG(expected_risk_bps) AS avg_expected_risk_bps,
                AVG(net_edge_bps) AS avg_predicted_net_edge_bps,
                AVG(raw_score) AS avg_raw_score,
                AVG(composite_score) AS avg_composite_score,
                AVG(COALESCE(trading_cost_bps, 0) + COALESCE(slippage_cost_bps, 0)) AS avg_cost_bps,
                AVG(realized_return_bps - expected_return_bps) AS avg_return_error_bps,
                AVG(ABS(realized_return_bps - expected_return_bps)) AS mae_return_error_bps,
                AVG(realized_risk_bps - expected_risk_bps) AS avg_risk_error_bps,
                AVG(ABS(realized_risk_bps - expected_risk_bps)) AS mae_risk_error_bps,
                AVG(
                    realized_return_bps - (realized_risk_bps * 0.10)
                    - COALESCE(trading_cost_bps, 0)
                    - COALESCE(slippage_cost_bps, 0)
                    - net_edge_bps
                ) AS avg_net_edge_error_bps,
                AVG(ABS(
                    realized_return_bps - (realized_risk_bps * 0.10)
                    - COALESCE(trading_cost_bps, 0)
                    - COALESCE(slippage_cost_bps, 0)
                    - net_edge_bps
                )) AS mae_net_edge_error_bps,
                COUNT(expected_return_bps) AS metric_sample_count,
                MIN(realized_return_bps) AS min_return_bps,
                MAX(realized_return_bps) AS max_return_bps,
                SUM(CASE WHEN realized_return_bps > 0 THEN 1 ELSE 0 END) AS win_count,
                SUM(CASE WHEN realized_return_bps <= 0 THEN 1 ELSE 0 END) AS loss_count,
                AVG(
                    realized_return_bps - (realized_risk_bps * 0.10)
                    - COALESCE(trading_cost_bps, 0)
                    - COALESCE(slippage_cost_bps, 0)
                ) AS avg_realized_net_edge_bps,
                SUM(CASE
                    WHEN (
                        realized_return_bps - (realized_risk_bps * 0.10)
                        - COALESCE(trading_cost_bps, 0)
                        - COALESCE(slippage_cost_bps, 0)
                    ) > 0 THEN 1 ELSE 0
                END) AS net_edge_win_count,
                SUM(CASE
                    WHEN (
                        realized_return_bps - (realized_risk_bps * 0.10)
                        - COALESCE(trading_cost_bps, 0)
                        - COALESCE(slippage_cost_bps, 0)
                    ) <= 0 THEN 1 ELSE 0
                END) AS net_edge_loss_count
            FROM canonical
            """
        ).fetchone()
        recent = conn.execute(
            f"""
            WITH canonical AS (
                SELECT *
                FROM (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (
                            PARTITION BY symbol, COALESCE(label_horizon_key, symbol || ':' || scan_time)
                            ORDER BY observed_at ASC, id ASC
                        ) AS symbol_horizon_rank
                    FROM edge_training_samples
                    WHERE {ELIGIBLE_EDGE_SAMPLE_STATUS_SQL}
                )
                WHERE symbol_horizon_rank = 1
            )
            SELECT
                id, scan_id, label_horizon_key, symbol, scan_time, observed_at, entry_price, observed_price,
                realized_return_bps, realized_risk_bps, label_observation_span_seconds,
                raw_score,
                expected_return_bps, expected_risk_bps, trading_cost_bps,
                slippage_cost_bps, net_edge_bps, composite_score, rank, status,
                market_segment, sector, theme
            FROM canonical
            ORDER BY observed_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        symbols = conn.execute(
            f"""
            WITH canonical AS (
                SELECT *
                FROM (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (
                            PARTITION BY symbol, COALESCE(label_horizon_key, symbol || ':' || scan_time)
                            ORDER BY observed_at ASC, id ASC
                        ) AS symbol_horizon_rank
                    FROM edge_training_samples
                    WHERE {ELIGIBLE_EDGE_SAMPLE_STATUS_SQL}
                )
                WHERE symbol_horizon_rank = 1
            )
            SELECT
                symbol,
                COUNT(*) AS sample_count,
                COALESCE(SUM(realized_return_bps), 0) AS total_return_bps,
                COALESCE(SUM(net_edge_bps), 0) AS total_predicted_net_edge_bps,
                AVG(realized_return_bps) AS avg_return_bps,
                AVG(realized_risk_bps) AS avg_risk_bps,
                AVG(expected_return_bps) AS avg_expected_return_bps,
                AVG(net_edge_bps) AS avg_predicted_net_edge_bps,
                SUM(CASE WHEN realized_return_bps > 0 THEN 1 ELSE 0 END) AS win_count
            FROM canonical
            GROUP BY symbol
            ORDER BY sample_count DESC, avg_return_bps DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    summary = _sample_summary_from_row(dict(aggregate or {}))
    diagnostics = get_edge_data_diagnostics(calibration_db_path=path)
    top10_performance = _top10_performance_from_store(calibration_path=path)
    unit_performance = _unit_performance_summary(calibration_path=path)
    return {
        "status": "ready" if summary["sample_count"] else "empty",
        "message": (
            "Edge training samples are available"
            if summary["sample_count"]
            else _empty_sample_message(diagnostics)
        ),
        "sample_count": summary["sample_count"],
        "summary": summary,
        "top10_performance": top10_performance,
        "unit_performance": unit_performance,
        "label_policy": label_policy_summary(),
        "recent_samples": [_sample_row(row) for row in recent],
        "symbol_summary": [_symbol_sample_row(row) for row in symbols],
        "diagnostics": diagnostics,
    }


def get_edge_data_diagnostics(
    *,
    calibration_db_path: Path | str | None = None,
) -> dict[str, Any]:
    calibration_path = _db_path(calibration_db_path)
    universe_path = settings.storage_path(settings.universe_scanner_db_path)
    auto_path = settings.storage_path(settings.auto_trading_db_path)
    return {
        "calibration_db_path": str(calibration_path),
        "calibration_db_exists": calibration_path.exists(),
        "universe_db_path": str(universe_path),
        "universe_db_exists": universe_path.exists(),
        "auto_trading_db_path": str(auto_path),
        "auto_trading_db_exists": auto_path.exists(),
        "edge_training_sample_count": _safe_sqlite_scalar(
            calibration_path,
            f"SELECT COUNT(*) FROM edge_training_samples WHERE {ELIGIBLE_EDGE_SAMPLE_STATUS_SQL}",
        ),
        "edge_top10_sample_count": _safe_sqlite_scalar(
            calibration_path,
            f"""
            SELECT COUNT(*)
            FROM edge_training_samples
            WHERE rank IS NOT NULL
              AND rank <= 10
              AND {ELIGIBLE_EDGE_SAMPLE_STATUS_SQL}
            """,
        ),
        "invalid_label_sample_count": _safe_sqlite_scalar(
            calibration_path,
            f"""
            SELECT COUNT(*)
            FROM edge_training_samples
            WHERE label_observation_span_seconds IS NOT NULL
              AND label_observation_span_seconds < {int(settings.edge_calibration_horizon_seconds or 0)}
              AND {ELIGIBLE_EDGE_SAMPLE_STATUS_SQL}
            """,
        ),
        "last_training_sample_at": _safe_sqlite_scalar(
            calibration_path,
            "SELECT MAX(observed_at) FROM edge_training_samples",
        ),
        "universe_scan_count": _safe_sqlite_scalar(
            universe_path,
            "SELECT COUNT(*) FROM universe_scan_runs",
        ),
        "scanner_candidate_history_count": _safe_sqlite_scalar(
            universe_path,
            "SELECT COUNT(*) FROM scanner_candidate_history",
        ),
        "eligible_candidate_history_count": _safe_sqlite_scalar(
            universe_path,
            """
            SELECT COUNT(*)
            FROM scanner_candidate_history
            WHERE current_price IS NOT NULL AND current_price > 0
            """,
        ),
        "candidate_with_future_snapshot_count": _safe_sqlite_scalar(
            universe_path,
            """
            SELECT COUNT(*)
            FROM scanner_candidate_history c
            WHERE c.current_price IS NOT NULL
              AND c.current_price > 0
              AND EXISTS (
                  SELECT 1
                  FROM universe_price_snapshots p
                  WHERE p.symbol = c.symbol
                    AND p.scan_id != c.scan_id
                    AND p.current_price IS NOT NULL
                    AND p.current_price > 0
              )
            """,
        ),
        "universe_price_snapshot_count": _safe_sqlite_scalar(
            universe_path,
            "SELECT COUNT(*) FROM universe_price_snapshots",
        ),
        "latest_scan_time": _safe_sqlite_scalar(
            universe_path,
            "SELECT MAX(created_at) FROM universe_scan_runs",
        ),
        "latest_candidate_scan_time": _safe_sqlite_scalar(
            universe_path,
            "SELECT MAX(scan_time) FROM scanner_candidate_history",
        ),
        "latest_price_snapshot_time": _safe_sqlite_scalar(
            universe_path,
            "SELECT MAX(created_at) FROM universe_price_snapshots",
        ),
        "auto_trading_session_count": _safe_sqlite_scalar(
            auto_path,
            "SELECT COUNT(*) FROM auto_trading_sessions",
        ),
        "active_auto_trading_session_count": _safe_sqlite_scalar(
            auto_path,
            "SELECT COUNT(*) FROM auto_trading_sessions WHERE status = 'active'",
        ),
        "total_auto_trading_cycle_count": _safe_sqlite_scalar(
            auto_path,
            "SELECT COALESCE(SUM(cycle_count), 0) FROM auto_trading_sessions",
        ),
        "latest_auto_trading_update": _safe_sqlite_scalar(
            auto_path,
            "SELECT MAX(updated_at) FROM auto_trading_sessions",
        ),
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
        with connect_sqlite(path) as conn:
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

def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "ok", "bull", "bullish", "positive"}


def _clip_bps(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def estimate_overheat_score(candidate: dict[str, Any]) -> float:
    """
    Continuous overheat score.

    Formula:
        overheat_score =
            max(0, (change_rate - 7) / 8)
          + max(0, (rsi - 70) / 30)
          + max(0, gap_up_pct / 5)

    0 = not overheated
    1~2 = increasingly overheated
    """
    change_rate = _to_float(candidate.get("change_rate")) or 0.0
    rsi = _to_float(candidate.get("rsi_14")) or 50.0
    gap_up_pct = _to_float(candidate.get("gap_up_pct")) or 0.0

    score = 0.0
    score += max(0.0, (change_rate - 7.0) / 8.0)
    score += max(0.0, (rsi - 70.0) / 30.0)
    score += max(0.0, gap_up_pct / 5.0)

    return _clamp(score, 0.0, 2.0)


def estimate_profit_factor_from_top10(
    top10_performance: dict[str, Any],
) -> float | None:
    """
    Estimate Profit Factor from existing top10 performance fields.

    Formula:
        profit_factor = gross_profit / gross_loss

    Because _top10_performance_from_store already returns:
        win_rate, loss_rate, avg_win_bps, avg_loss_bps

    We can estimate:
        gross_profit_per_sample = win_rate * avg_win_bps
        gross_loss_per_sample = loss_rate * avg_loss_bps
    """
    win_rate = _to_float(top10_performance.get("win_rate"))
    loss_rate = _to_float(top10_performance.get("loss_rate"))
    avg_win = _to_float(top10_performance.get("avg_win_bps"))
    avg_loss = _to_float(top10_performance.get("avg_loss_bps"))

    if win_rate is None or loss_rate is None or avg_win is None or avg_loss is None:
        return None

    gross_profit = win_rate * avg_win
    gross_loss = loss_rate * avg_loss

    if gross_loss <= 0:
        return None

    return gross_profit / gross_loss


def _pearson_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None

    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)

    if var_x <= 0 or var_y <= 0:
        return None

    return cov / ((var_x ** 0.5) * (var_y ** 0.5))


def _recent_ic_from_store(
    *,
    calibration_path: Path,
    limit: int = 300,
) -> dict[str, Any]:
    """
    Information Coefficient.

    Formula:
        IC = corr(predicted_net_edge_bps, realized_net_edge_bps)

    If IC is below 0, the model's ranking may be working backwards.
    """
    initialize_edge_calibration_db(calibration_path)

    with sqlite3.connect(calibration_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT
                net_edge_bps,
                realized_return_bps,
                realized_risk_bps,
                trading_cost_bps,
                slippage_cost_bps
            FROM edge_training_samples
            WHERE net_edge_bps IS NOT NULL
              AND {ELIGIBLE_EDGE_SAMPLE_STATUS_SQL}
            ORDER BY observed_at DESC, id DESC
            LIMIT ?
            """,
            (max(10, int(limit)),),
        ).fetchall()

    predicted: list[float] = []
    realized: list[float] = []

    for row in rows:
        pred = _to_float(row["net_edge_bps"])
        if pred is None:
            continue

        realized_return = _to_float(row["realized_return_bps"]) or 0.0
        realized_risk = _to_float(row["realized_risk_bps"]) or 0.0
        trading_cost = _to_float(row["trading_cost_bps"]) or 0.0
        slippage_cost = _to_float(row["slippage_cost_bps"]) or 0.0

        realized_net = (
            realized_return
            - realized_risk * REALIZED_RISK_WEIGHT
            - trading_cost
            - slippage_cost
        )

        predicted.append(pred)
        realized.append(realized_net)

    ic = _pearson_corr(predicted, realized)

    return {
        "sample_count": len(predicted),
        "ic": round(ic, 6) if ic is not None else None,
        "ic_formula": "corr(predicted_net_edge_bps, realized_net_edge_bps)",
    }


def candidate_cost_coverage(candidate: dict[str, Any]) -> float | None:
    """
    Cost coverage ratio.

    Formula:
        cost_coverage = expected_return_bps / total_cost_bps

    This version uses default policy costs when candidate does not provide costs.
    """
    expected_return = _to_float(
        candidate.get("expected_return")
        or candidate.get("expected_return_bps")
    )

    trading_cost = _to_float(
        candidate.get("trading_cost")
        or candidate.get("trading_cost_bps")
    )
    if trading_cost is None:
        trading_cost = DEFAULT_ROUND_TRIP_TRADING_COST_BPS

    slippage_cost = _to_float(
        candidate.get("slippage_cost")
        or candidate.get("slippage_cost_bps")
    )
    if slippage_cost is None:
        slippage_cost = estimate_slippage_cost_bps(candidate)

    liquidity_drag = _to_float(candidate.get("liquidity_drag_bps"))
    if liquidity_drag is None:
        liquidity_drag = estimate_liquidity_drag_bps(candidate)

    total_cost = trading_cost + slippage_cost + liquidity_drag

    if expected_return is None or total_cost <= 0:
        return None

    return expected_return / total_cost


def candidate_feature_map(
    candidate: dict[str, Any],
    raw_score: float,
) -> dict[str, float]:
    intraday = candidate.get("intraday") or {}
    market = candidate.get("market") or candidate.get("market_context") or {}

    volume_ratio = _to_float(candidate.get("volume_ratio")) or 0.0
    minute_volume_ratio = _to_float(intraday.get("minute_volume_ratio")) or 0.0
    turnover = _to_float(candidate.get("turnover_value"))

    market_regime_raw = (
        market.get("regime_ok")
        if isinstance(market, dict) and "regime_ok" in market
        else candidate.get("market_regime_ok")
    )
    market_regime = 1.0 if _to_bool(market_regime_raw) else 0.0

    market_breadth = (
        _to_float(market.get("breadth"))
        if isinstance(market, dict)
        else None
    )
    if market_breadth is None:
        market_breadth = _to_float(candidate.get("market_breadth"))
    if market_breadth is None:
        market_breadth = 0.5

    index_return_1d = (
        _to_float(market.get("index_return_1d"))
        if isinstance(market, dict)
        else None
    )
    if index_return_1d is None:
        index_return_1d = _to_float(candidate.get("index_return_1d")) or 0.0

    relative_strength_3d = _to_float(candidate.get("relative_strength_3d")) or 0.0
    relative_strength_5d = _to_float(candidate.get("relative_strength_5d")) or 0.0

    atr_pct = _to_float(candidate.get("atr_pct")) or 0.0
    volatility_10d = _to_float(candidate.get("volatility_10d")) or 0.0

    high = _to_float(intraday.get("high") or candidate.get("high_price"))
    low = _to_float(intraday.get("low") or candidate.get("low_price"))
    close = _to_float(
        candidate.get("current_price")
        or candidate.get("close")
        or candidate.get("last_price")
    )
    open_price = _to_float(intraday.get("open") or candidate.get("open_price"))

    if high is not None and low is not None and close is not None and high > low:
        close_position = (close - low) / (high - low)
    else:
        close_position = _to_float(candidate.get("close_position"))
        if close_position is None:
            close_position = 0.5

    if open_price is not None and open_price > 0 and close is not None:
        intraday_recovery = (close - open_price) / open_price * 100.0
    else:
        intraday_recovery = _to_float(candidate.get("intraday_recovery")) or 0.0

    overheat_score = estimate_overheat_score(candidate)

    return {
        "bias": 1.0,
        "raw_score": _clamp(float(raw_score) / 100.0, 0.0, 1.0),
        "change_rate": _clamp(
            (_to_float(candidate.get("change_rate")) or 0.0) / 10.0,
            -1.0,
            1.5,
        ),
        "volume_ratio": _clamp(volume_ratio / 5.0, 0.0, 2.0),
        "minute_volume_ratio": _clamp(minute_volume_ratio / 5.0, 0.0, 2.0),
        "news_count": _clamp(
            float(_to_int(candidate.get("news_count")) or 0) / 5.0,
            0.0,
            1.0,
        ),
        "disclosure_count": _clamp(
            float(_to_int(candidate.get("disclosure_count")) or 0) / 3.0,
            0.0,
            1.0,
        ),
        "overheated": 1.0 if bool(candidate.get("overheated")) else 0.0,
        "latest_close": 1.0 if candidate.get("price_source") == "latest_close" else 0.0,
        "low_turnover": 1.0 if turnover is not None and turnover < 20_000_000_000 else 0.0,

        "market_regime": market_regime,
        "market_breadth": _clamp(market_breadth, 0.0, 1.0),
        "index_return_1d": _clamp(index_return_1d / 3.0, -1.0, 1.0),
        "relative_strength_3d": _clamp(relative_strength_3d / 5.0, -1.0, 1.0),
        "relative_strength_5d": _clamp(relative_strength_5d / 8.0, -1.0, 1.0),
        "atr_pct": _clamp(atr_pct / 5.0, 0.0, 2.0),
        "volatility_10d": _clamp(volatility_10d / 5.0, 0.0, 2.0),
        "close_position": _clamp(close_position, 0.0, 1.0),
        "intraday_recovery": _clamp(intraday_recovery / 5.0, -1.0, 1.0),
        "overheat_score": overheat_score,
    }


def edge_entry_gate(
    candidates: list[dict[str, Any]] | None = None,
    *,
    calibration_db_path: Path | str | None = None,
    execution_mode: str | None = None,
) -> dict[str, Any]:
    """Return whether calibrated edge quality is strong enough for new entries."""
    if not settings.edge_calibration_enabled:
        return _default_gate(
            status="disabled",
            approved=True,
            message="Edge calibration is disabled",
            execution_mode=execution_mode,
        )

    path = _db_path(calibration_db_path)
    if not path.exists():
        return _apply_broker_paper_calibration_policy(
            _default_gate(
                status="collecting",
                approved=False,
                message="No edge calibration DB has been created yet",
                execution_mode=execution_mode,
            ),
            execution_mode=execution_mode,
        )

    initialize_edge_calibration_db(path)

    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        stored_sample_count = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM edge_training_samples
            WHERE {ELIGIBLE_EDGE_SAMPLE_STATUS_SQL}
            """
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
        return _apply_broker_paper_calibration_policy(
            _default_gate(
                status="collecting",
                approved=False,
                message="No edge calibration run has completed yet",
                sample_count=int(stored_sample_count or 0),
                execution_mode=execution_mode,
            ),
            execution_mode=execution_mode,
        )

    raw = _parse_json(run["raw_json"], {})
    refresh = refresh_top10_performance_if_due(calibration_db_path=path)

    top10_performance = (
        refresh.get("top10_performance")
        or _latest_top10_performance_from_store(path)
        or raw.get("top10_performance")
        or _top10_performance_from_store(calibration_path=path)
    )

    fill_adjustment = raw.get("fill_adjustment") or {
        "multiplier": load_fill_adjustment(calibration_db_path=path),
    }

    ic_metrics = _recent_ic_from_store(
        calibration_path=path,
        limit=300,
    )

    return _apply_broker_paper_calibration_policy(
        _gate_from_metrics(
            sample_count=int(stored_sample_count or run["sample_count"] or 0),
            oos_sample_count=int(raw.get("oos_sample_count") or run["oos_sample_count"] or 0),
            mae_return_bps=_to_float(run["mae_return_bps"]),
            mae_risk_bps=_to_float(run["mae_risk_bps"]),
            top10_performance=top10_performance,
            fill_adjustment=fill_adjustment,
            ic_metrics=ic_metrics,
            candidates=candidates,
            execution_mode=execution_mode,
        ),
        execution_mode=execution_mode,
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
            settings.storage_path(paper_db_path or paper_trading.DEFAULT_DB_PATH),
            limit=limit,
        )
    )
    fill_events.extend(
        _broker_fill_quality_events(
            settings.storage_path(broker_db_path or settings.broker_sync_db_path),
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
            f"""
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


def _label_observation_span_seconds(
    *,
    scan_time: str,
    observed_at: str,
) -> int | None:
    try:
        start = datetime.fromisoformat(scan_time)
        end = datetime.fromisoformat(observed_at)
    except ValueError:
        return None
    return max(0, int((end - start).total_seconds()))


def _label_horizon_key(
    *,
    symbol: str,
    scan_time: str,
    horizon_seconds: int,
) -> str:
    horizon = max(60, int(horizon_seconds or 86400))
    try:
        parsed = datetime.fromisoformat(str(scan_time))
        bucket = int(parsed.timestamp()) // horizon
        return f"{symbol}:{horizon}:{bucket}"
    except (TypeError, ValueError, OSError):
        return f"{symbol}:{horizon}:{str(scan_time)[:10]}"


def _canonical_sample_key(row: sqlite3.Row | dict[str, Any]) -> str:
    symbol = str(_row_value(row, "symbol") or "")
    horizon_key = _row_value(row, "label_horizon_key")
    if horizon_key:
        return f"{symbol}:{horizon_key}"
    return f"{symbol}:{_row_value(row, 'scan_time') or _row_value(row, 'id')}"


def _scan_run_key(row: sqlite3.Row | dict[str, Any]) -> str:
    scan_id = _row_value(row, "scan_id")
    if scan_id:
        return str(scan_id)
    raw = _parse_json(_row_value(row, "raw_json"), {})
    if isinstance(raw, dict):
        raw_scan_id = raw.get("scan_id") or raw.get("scan_run_id")
        candidate = raw.get("candidate")
        if isinstance(candidate, dict):
            raw_scan_id = raw_scan_id or candidate.get("scan_id") or candidate.get("scan_run_id")
        if raw_scan_id:
            return str(raw_scan_id)
    return f"legacy-row:{_row_value(row, 'id')}"


def _metadata_from_candidate_row(
    row: sqlite3.Row,
    raw_payload: dict[str, Any],
) -> dict[str, str | None]:
    market_segment = (
        _row_value(row, "market_segment")
        or raw_payload.get("market_segment")
        or raw_payload.get("market")
    )
    sector = (
        raw_payload.get("sector")
        or raw_payload.get("industry")
        or raw_payload.get("sector_name")
    )
    theme = (
        raw_payload.get("theme")
        or raw_payload.get("strategy_theme")
        or raw_payload.get("universe_profile")
        or _row_value(row, "universe_profile")
    )
    return {
        "market_segment": str(market_segment) if market_segment else None,
        "sector": str(sector) if sector else None,
        "theme": str(theme) if theme else None,
    }


def _candidate_lookback_seconds(horizon_seconds: int) -> int | None:
    configured = settings.edge_calibration_candidate_lookback_seconds
    if configured is not None and int(configured) > 0:
        return int(configured)
    return None


def _candidate_cutoff_time(horizon_seconds: int) -> str:
    lookback_seconds = _candidate_lookback_seconds(horizon_seconds)
    if lookback_seconds is None:
        return "0001-01-01T00:00:00"
    return (
        datetime.now() - timedelta(seconds=lookback_seconds)
    ).isoformat(timespec="seconds")


def _refresh_candidate_examine_limit(candidate_limit: int | None) -> int | None:
    if candidate_limit is not None:
        return max(1, int(candidate_limit))
    return None


def label_policy_summary() -> dict[str, Any]:
    """Expose active label timing rules for admin and API consumers."""
    return {
        "horizon_seconds": int(settings.edge_calibration_horizon_seconds or 86400),
        "min_label_age_seconds": int(settings.edge_calibration_min_label_age_seconds or 0),
        "min_future_snapshots": int(settings.edge_calibration_min_future_snapshots or 1),
        "label_at_horizon_end": bool(settings.edge_calibration_label_at_horizon_end),
        "label_horizon_tolerance_seconds": int(
            settings.edge_calibration_label_horizon_tolerance_seconds or 0
        ),
        "refresh_after_scan": bool(settings.edge_calibration_refresh_after_scan),
        "label_price_rule": (
            "Wait until the horizon has elapsed, then use eligible snapshots "
            "near the horizon timestamp."
            if settings.edge_calibration_label_at_horizon_end
            else "Use the first eligible snapshots after scan time"
        ),
        "realized_net_edge_formula": (
            REALIZED_NET_EDGE_FORMULA
        ),
    }


def _future_price_window(
    scan_time: str,
    horizon_seconds: int,
) -> tuple[str, str] | None:
    try:
        start_dt = datetime.fromisoformat(scan_time)
    except ValueError:
        return None
    if settings.edge_calibration_label_at_horizon_end:
        horizon_dt = start_dt + timedelta(seconds=horizon_seconds)
        tolerance_seconds = max(
            0,
            int(settings.edge_calibration_label_horizon_tolerance_seconds or 0),
        )
        end_dt = horizon_dt + timedelta(seconds=tolerance_seconds)
        return (
            horizon_dt.isoformat(timespec="seconds"),
            end_dt.isoformat(timespec="seconds"),
        )
    min_obs_dt = start_dt + timedelta(
        seconds=max(0, int(settings.edge_calibration_min_label_age_seconds or 0))
    )
    end_dt = start_dt + timedelta(seconds=horizon_seconds)
    return (
        min_obs_dt.isoformat(timespec="seconds"),
        end_dt.isoformat(timespec="seconds"),
    )


def _label_window_contains_now(scan_time: str, horizon_seconds: int) -> bool:
    window = _future_price_window(scan_time, horizon_seconds)
    if window is None:
        return False
    start, end = window
    now = datetime.now()
    try:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
    except ValueError:
        return False
    return start_dt <= now <= end_dt


def _fetch_future_price_rows(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    scan_id: str,
    scan_time: str,
    horizon_seconds: int,
) -> list[sqlite3.Row]:
    """Return later-scan price snapshots used to label a candidate."""
    window = _future_price_window(scan_time, horizon_seconds)
    if window is None:
        return []
    start_time, end_time = window
    min_snapshots = max(1, int(settings.edge_calibration_min_future_snapshots or 1))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT created_at, current_price, raw_json
        FROM universe_price_snapshots
        WHERE symbol = ?
          AND scan_id != ?
          AND created_at >= ?
          AND created_at <= ?
          AND current_price IS NOT NULL
          AND current_price > 0
        ORDER BY created_at ASC
        """,
        (
            symbol,
            scan_id,
            start_time,
            end_time,
        ),
    ).fetchall()
    if len(rows) < min_snapshots:
        return []
    if settings.edge_calibration_label_at_horizon_end:
        return rows
    configured = max(32, int(settings.edge_calibration_future_price_limit or 96))
    estimated = max(32, int(horizon_seconds / 30) + 10)
    row_limit = max(configured, estimated)
    return rows[:row_limit]


def _fetch_excursion_price_rows(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    scan_id: str,
    scan_time: str,
    horizon_seconds: int,
) -> list[sqlite3.Row]:
    try:
        start_dt = datetime.fromisoformat(scan_time)
    except ValueError:
        return []
    tolerance_seconds = max(
        0,
        int(settings.edge_calibration_label_horizon_tolerance_seconds or 0),
    )
    end_dt = start_dt + timedelta(
        seconds=max(60, int(horizon_seconds or 86400)) + tolerance_seconds
    )
    conn.row_factory = sqlite3.Row
    return conn.execute(
        """
        SELECT created_at, current_price, raw_json
        FROM universe_price_snapshots
        WHERE symbol = ?
          AND scan_id != ?
          AND created_at > ?
          AND created_at <= ?
          AND current_price IS NOT NULL
          AND current_price > 0
        ORDER BY created_at ASC
        """,
        (
            symbol,
            scan_id,
            scan_time,
            end_dt.isoformat(timespec="seconds"),
        ),
    ).fetchall()


def _snapshot_price_extremes(row: sqlite3.Row) -> tuple[float | None, float | None]:
    raw = _parse_json(_row_value(row, "raw_json"), {})
    if not isinstance(raw, dict):
        raw = {}
    output = raw.get("output") if isinstance(raw.get("output"), dict) else {}
    intraday = raw.get("intraday") if isinstance(raw.get("intraday"), dict) else {}
    sources = [dict(row), raw, output, intraday]
    lows = [
        _first_float_from_mapping(source, ("low", "low_price", "stck_lwpr", "lwpr"))
        for source in sources
    ]
    highs = [
        _first_float_from_mapping(source, ("high", "high_price", "stck_hgpr", "hgpr"))
        for source in sources
    ]
    current = _to_float(_row_value(row, "current_price"))
    low = min([value for value in lows if value is not None] + ([current] if current else []), default=None)
    high = max([value for value in highs if value is not None] + ([current] if current else []), default=None)
    return low, high


def _adverse_excursion_bps(
    *,
    entry_price: float,
    rows: list[sqlite3.Row],
    side: str | None = None,
) -> float:
    if entry_price <= 0 or not rows:
        return 0.0
    side_text = str(side or "").lower()
    lows: list[float] = []
    highs: list[float] = []
    for row in rows:
        low, high = _snapshot_price_extremes(row)
        if low is not None and low > 0:
            lows.append(low)
        if high is not None and high > 0:
            highs.append(high)
    if side_text in {"sell", "short", "exit"}:
        if not highs:
            return 0.0
        return max(0.0, (max(highs) - entry_price) / entry_price * 10_000)
    if not lows:
        return 0.0
    return max(0.0, (entry_price - min(lows)) / entry_price * 10_000)


def _first_float_from_mapping(
    mapping: dict[str, Any],
    keys: tuple[str, ...],
) -> float | None:
    for key in keys:
        value = _to_float(mapping.get(key))
        if value is not None:
            return value
    return None


def _labeled_candidate_ids(calibration_path: Path) -> set[int]:
    if not calibration_path.exists():
        return set()
    initialize_edge_calibration_db(calibration_path)
    with sqlite3.connect(calibration_path) as conn:
        _purge_invalid_label_samples(conn)
        conn.commit()
        return {
            int(row[0])
            for row in conn.execute(
                "SELECT source_candidate_id FROM edge_training_samples"
            ).fetchall()
            if row[0] is not None
        }


def _iter_label_candidates(
    source_conn: sqlite3.Connection,
    *,
    labeled_ids: set[int],
    cutoff: str,
    batch_size: int,
    max_rows: int | None,
) -> Iterator[sqlite3.Row]:
    """Walk candidate history oldest-first so labels are not dropped by horizon expiry."""
    source_conn.row_factory = sqlite3.Row
    last_id = 0
    yielded = 0
    while True:
        if max_rows is not None and yielded >= max_rows:
            return
        rows = source_conn.execute(
            f"""
            SELECT *
            FROM scanner_candidate_history
            WHERE id > ?
              AND scan_time >= ?
              AND current_price IS NOT NULL
              AND current_price > 0
            ORDER BY id ASC
            LIMIT ?
            """,
            (last_id, cutoff, batch_size),
        ).fetchall()
        if not rows:
            return
        for row in rows:
            last_id = int(row["id"])
            if last_id in labeled_ids:
                continue
            yield row
            yielded += 1
            if max_rows is not None and yielded >= max_rows:
                return


def _append_labeling_price_snapshots(
    *,
    universe_path: Path,
    horizon_seconds: int,
    max_symbols: int | None = None,
) -> dict[str, Any]:
    """Fetch current prices for candidates that still lack a later-scan snapshot."""
    if not settings.edge_calibration_label_snapshots_enabled:
        return {"status": "disabled", "inserted": 0}
    if not universe_path.exists():
        return {"status": "empty", "inserted": 0}

    from app.data_sources.kis import fetch_price_data

    max_symbols = max(
        1,
        int(max_symbols or settings.edge_calibration_label_snapshot_max_symbols or 200),
    )
    calibration_path = _db_path()
    labeled_ids = _labeled_candidate_ids(calibration_path)
    cutoff = _candidate_cutoff_time(horizon_seconds)
    symbols: list[str] = []

    with sqlite3.connect(universe_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT id, symbol, scan_id, scan_time
            FROM scanner_candidate_history
            WHERE scan_time >= ?
              AND symbol IS NOT NULL
              AND symbol != ''
              AND current_price IS NOT NULL
              AND current_price > 0
            ORDER BY scan_time ASC, id ASC
            """,
            (cutoff,),
        ).fetchall()
        seen: set[str] = set()
        for row in rows:
            if int(row["id"]) in labeled_ids:
                continue
            if settings.edge_calibration_label_at_horizon_end and not _label_window_contains_now(
                str(row["scan_time"]),
                horizon_seconds,
            ):
                continue
            symbol = str(row["symbol"])
            if symbol in seen:
                continue
            future_rows = _fetch_future_price_rows(
                conn,
                symbol=symbol,
                scan_id=str(row["scan_id"]),
                scan_time=str(row["scan_time"]),
                horizon_seconds=horizon_seconds,
            )
            if future_rows:
                continue
            seen.add(symbol)
            symbols.append(symbol)
            if len(symbols) >= max_symbols:
                break

        if not symbols:
            return {
                "status": "skipped",
                "inserted": 0,
                "message": "All pending candidates already have later-scan snapshots",
            }

        scan_id = f"label-sync-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        created_at = _now()
        inserted = 0
        errors = 0
        for symbol in symbols:
            try:
                price_data = fetch_price_data(symbol)
                price = _to_float(price_data.get("current_price"))
                if price is None or price <= 0:
                    errors += 1
                    continue
                conn.execute(
                    """
                    INSERT INTO universe_price_snapshots (
                        scan_id, created_at, symbol, name, current_price,
                        change_rate, volume, volume_ratio, turnover_value,
                        market_cap, market_segment, universe_profile,
                        trend, source, raw_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        scan_id,
                        created_at,
                        symbol,
                        price_data.get("name"),
                        price,
                        _to_float(price_data.get("change_rate")),
                        _to_int(price_data.get("volume")),
                        _to_float(price_data.get("volume_ratio")),
                        _to_float(price_data.get("turnover_value")),
                        _to_float(price_data.get("market_cap")),
                        price_data.get("market_segment"),
                        price_data.get("universe_profile"),
                        price_data.get("trend"),
                        price_data.get("source", "label_sync"),
                        _json(price_data),
                    ),
                )
                inserted += 1
            except Exception as exc:
                errors += 1
                logger.debug(
                    "label snapshot fetch failed for %s: %s",
                    symbol,
                    exc,
                )
        conn.commit()

    return {
        "status": "success",
        "inserted": inserted,
        "errors": errors,
        "symbol_count": len(symbols),
        "scan_id": scan_id,
    }


def _training_payload_from_candidate_row(
    *,
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    horizon_seconds: int,
) -> dict[str, Any] | None:
    status = _row_value(row, "status")
    if not _is_edge_training_sample_status_allowed(status):
        return None

    entry_price = _to_float(row["current_price"])
    if entry_price is None or entry_price <= 0:
        return None
    scan_time = str(row["scan_time"])
    scan_id = str(row["scan_id"])
    future_rows = _fetch_future_price_rows(
        conn,
        symbol=str(row["symbol"]),
        scan_id=scan_id,
        scan_time=scan_time,
        horizon_seconds=horizon_seconds,
    )
    if not future_rows:
        return None
    excursion_rows = _fetch_excursion_price_rows(
        conn,
        symbol=str(row["symbol"]),
        scan_id=scan_id,
        scan_time=scan_time,
        horizon_seconds=horizon_seconds,
    ) or future_rows
    final_price = _to_float(future_rows[-1]["current_price"])
    observed_at = str(future_rows[-1]["created_at"])
    prices = [_to_float(item["current_price"]) for item in excursion_rows]
    usable_prices = [price for price in prices if price is not None and price > 0]
    if final_price is None or not usable_prices:
        return None
    realized_return_bps = (final_price - entry_price) / entry_price * 10_000
    raw_payload = _parse_json(row["raw_json"], {})
    side = str(raw_payload.get("side") or raw_payload.get("action") or row["decision"] or "")
    realized_risk_bps = _adverse_excursion_bps(
        entry_price=entry_price,
        rows=excursion_rows,
        side=side,
    )
    label_observation_span = _label_observation_span_seconds(
        scan_time=scan_time,
        observed_at=observed_at,
    )
    raw_score = _to_float(row["raw_score"]) or 0.0
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
    
    expected_return_bps = _to_float(row["expected_return"])
    expected_risk_bps = _to_float(row["expected_risk"])

    trading_cost_bps = _to_float(row["trading_cost"])
    if trading_cost_bps is None:
        trading_cost_bps = DEFAULT_ROUND_TRIP_TRADING_COST_BPS

    slippage_cost_bps = _to_float(row["slippage_cost"])
    if slippage_cost_bps is None:
        slippage_cost_bps = estimate_slippage_cost_bps(candidate)

    liquidity_drag_bps = estimate_liquidity_drag_bps(candidate)

    net_edge_bps = _to_float(row["net_edge"])
    if expected_return_bps is not None and expected_risk_bps is not None:
        net_edge_bps = estimate_expected_net_edge_bps(
            expected_return_bps=expected_return_bps,
            expected_risk_bps=expected_risk_bps,
            trading_cost_bps=trading_cost_bps,
            slippage_cost_bps=slippage_cost_bps,
            liquidity_drag_bps=liquidity_drag_bps,
        )
    metadata = _metadata_from_candidate_row(row, raw_payload)
    label_horizon_key = _label_horizon_key(
        symbol=str(row["symbol"]),
        scan_time=scan_time,
        horizon_seconds=horizon_seconds,
    )
    return {
        "source_candidate_id": int(row["id"]),
        "scan_id": scan_id,
        "label_horizon_key": label_horizon_key,
        "symbol": str(row["symbol"]),
        "scan_time": scan_time,
        "observed_at": observed_at,
        "entry_price": entry_price,
        "observed_price": final_price,
        "features": [feature_map[name] for name in FEATURE_NAMES],
        "realized_return_bps": realized_return_bps,
        "realized_risk_bps": realized_risk_bps,
        "label_observation_span_seconds": label_observation_span,
        "raw_score": raw_score,
        "expected_return_bps": expected_return_bps,
        "expected_risk_bps": expected_risk_bps,
        "trading_cost_bps": trading_cost_bps,
        "slippage_cost_bps": slippage_cost_bps,
        "liquidity_drag_bps": liquidity_drag_bps,
        "net_edge_bps": net_edge_bps,
        "composite_score": _to_float(row["composite_score"]),
        "rank": _to_int(row["rank"]),
        "status": row["status"],
        **metadata,
        "raw_json": {
            "scan_id": scan_id,
            "label_horizon_key": label_horizon_key,
            "candidate": raw_payload,
            "feature_map": feature_map,
            "label_policy": label_policy_summary(),
            "label_observation_span_seconds": label_observation_span,
            "risk_policy": {
                "realized_risk_bps": realized_risk_bps,
                "adverse_excursion_source": "entry_to_horizon_price_extremes",
                "excursion_snapshot_count": len(excursion_rows),
                "side": side or "buy",
            },
            "sample_metrics": {
                "raw_score": raw_score,
                "expected_return_bps": expected_return_bps,
                "expected_risk_bps": expected_risk_bps,
                "trading_cost_bps": trading_cost_bps,
                "slippage_cost_bps": slippage_cost_bps,
                "liquidity_drag_bps": liquidity_drag_bps,
                "net_edge_bps": net_edge_bps,
                "composite_score": _to_float(row["composite_score"]),
            },
            "source": "scanner_candidate_history",
        },
    }


def _refresh_training_samples(
    *,
    calibration_path: Path,
    universe_path: Path,
    horizon_seconds: int,
    candidate_limit: int | None = None,
) -> dict[str, int]:
    inserted_count = 0
    skipped_count = 0
    examined_count = 0
    purged_invalid_count = 0
    if not universe_path.exists():
        return {
            "examined_count": 0,
            "inserted_count": 0,
            "skipped_count": 0,
            "unlabeled_examined_count": 0,
            "purged_invalid_label_count": 0,
        }
    initialize_edge_calibration_db(calibration_path)
    cutoff = _candidate_cutoff_time(horizon_seconds)
    batch_size = max(50, int(settings.edge_calibration_refresh_batch_size or 500))
    max_rows = _refresh_candidate_examine_limit(candidate_limit)
    try:
        with sqlite3.connect(universe_path) as source_conn, sqlite3.connect(calibration_path) as target_conn:
            source_conn.row_factory = sqlite3.Row
            target_conn.executescript(SCHEMA_SQL)
            _ensure_schema_migrations(target_conn)
            purged_invalid_count = _purge_invalid_label_samples(target_conn)
            labeled_ids = {
                int(row[0])
                for row in target_conn.execute(
                    "SELECT source_candidate_id FROM edge_training_samples"
                ).fetchall()
                if row[0] is not None
            }
            for row in _iter_label_candidates(
                source_conn,
                labeled_ids=labeled_ids,
                cutoff=cutoff,
                batch_size=batch_size,
                max_rows=max_rows,
            ):
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
                labeled_ids.add(int(payload["source_candidate_id"]))
            _prune_training_samples(target_conn)
            target_conn.commit()
    except sqlite3.Error as exc:
        logger.warning("edge sample refresh failed: %s", exc)
        return {
            "examined_count": examined_count,
            "inserted_count": inserted_count,
            "skipped_count": skipped_count,
            "purged_invalid_label_count": purged_invalid_count,
            "error": str(exc),
        }
    return {
        "examined_count": examined_count,
        "inserted_count": inserted_count,
        "skipped_count": skipped_count,
        "unlabeled_examined_count": examined_count,
        "purged_invalid_label_count": purged_invalid_count,
    }


def _store_training_sample(
    conn: sqlite3.Connection,
    payload: dict[str, Any],
) -> int:
    if payload.get("label_horizon_key"):
        existing = conn.execute(
            """
            SELECT 1
            FROM edge_training_samples
            WHERE symbol = ?
              AND label_horizon_key = ?
            LIMIT 1
            """,
            (payload["symbol"], payload["label_horizon_key"]),
        ).fetchone()
        if existing:
            return 0
    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO edge_training_samples (
            source_candidate_id, scan_id, label_horizon_key,
            symbol, scan_time, observed_at, entry_price,
            observed_price, features_json, realized_return_bps,
            realized_risk_bps, label_observation_span_seconds, raw_score, expected_return_bps,
            expected_risk_bps, trading_cost_bps, slippage_cost_bps,
            net_edge_bps, composite_score, rank, status,
            market_segment, sector, theme, created_at, raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload["source_candidate_id"],
            payload.get("scan_id"),
            payload.get("label_horizon_key"),
            payload["symbol"],
            payload["scan_time"],
            payload["observed_at"],
            payload["entry_price"],
            payload["observed_price"],
            _json(payload["features"]),
            payload["realized_return_bps"],
            payload["realized_risk_bps"],
            payload.get("label_observation_span_seconds"),
            payload.get("raw_score"),
            payload.get("expected_return_bps"),
            payload.get("expected_risk_bps"),
            payload.get("trading_cost_bps"),
            payload.get("slippage_cost_bps"),
            payload.get("net_edge_bps"),
            payload.get("composite_score"),
            payload.get("rank"),
            payload.get("status"),
            payload.get("market_segment"),
            payload.get("sector"),
            payload.get("theme"),
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
            f"""
            WITH ranked AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol, COALESCE(label_horizon_key, symbol || ':' || scan_time)
                        ORDER BY observed_at ASC, id ASC
                    ) AS symbol_horizon_rank
                FROM edge_training_samples
                WHERE {ELIGIBLE_EDGE_SAMPLE_STATUS_SQL}
            )
            SELECT features_json, realized_return_bps, realized_risk_bps
            FROM (
                SELECT *
                FROM ranked
                WHERE symbol_horizon_rank = 1
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
        realized_return_bps = _clip_bps(
            float(row["realized_return_bps"]),
            -800.0,
            800.0,
        )
        realized_risk_bps = _clip_bps(
            float(row["realized_risk_bps"]),
            0.0,
            800.0,
        )
        return (
            [float(value) for value in features],
            realized_return_bps,
            realized_risk_bps,
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
    limit: int | None = None,
) -> dict[str, Any]:
    initialize_edge_calibration_db(calibration_path)
    _purge_invalid_label_samples_by_path(calibration_path)
    rows = _top10_candidate_rows(calibration_path=calibration_path)
    scan_units = _scan_run_units_from_top10_rows(rows, limit=limit)
    if not scan_units:
        return _empty_top10_performance(limit=limit)
    return _performance_summary_from_rows(
        scan_units,
        top_count=10,
        sample_source=TOP10_SAMPLE_SOURCE_SCAN_RUN,
        sample_limit=limit,
        candidate_sample_count=len(rows),
        scan_run_count=len(scan_units),
        concentration=_concentration_from_rows(rows),
    )


def _top10_candidate_rows(
    *,
    calibration_path: Path,
) -> list[dict[str, Any]]:
    with sqlite3.connect(calibration_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            WITH ranked AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol, COALESCE(label_horizon_key, symbol || ':' || scan_time)
                        ORDER BY observed_at ASC, id ASC
                    ) AS symbol_horizon_rank
                FROM edge_training_samples
                WHERE rank IS NOT NULL
                  AND rank <= 10
                  AND {ELIGIBLE_EDGE_SAMPLE_STATUS_SQL}
            )
            SELECT
                id, source_candidate_id, scan_id, label_horizon_key, symbol,
                scan_time, observed_at, realized_return_bps, realized_risk_bps,
                expected_return_bps, expected_risk_bps, trading_cost_bps,
                slippage_cost_bps, net_edge_bps, raw_score, composite_score,
                rank, status, market_segment, sector, theme, raw_json
            FROM ranked
            WHERE symbol_horizon_rank = 1
            ORDER BY observed_at DESC, id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def _scan_run_units_from_top10_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    max_groups = max(1, int(limit)) if limit is not None else None
    for row in rows:
        scan_key = _scan_run_key(row)
        if scan_key not in grouped:
            if max_groups is not None and len(grouped) >= max_groups:
                continue
            grouped[scan_key] = []
        grouped[scan_key].append(row)
    return [
        _scan_run_unit_from_rows(scan_id=scan_id, rows=items)
        for scan_id, items in grouped.items()
        if items
    ]


def _scan_run_unit_from_rows(
    *,
    scan_id: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    def avg(key: str) -> float | None:
        values = [_to_float(row.get(key)) for row in rows]
        values = [value for value in values if value is not None]
        return sum(values) / len(values) if values else None

    realized_return = avg("realized_return_bps") or 0.0
    realized_risk = avg("realized_risk_bps") or 0.0
    trading_cost = avg("trading_cost_bps") or 0.0
    slippage_cost = avg("slippage_cost_bps") or 0.0
    net_edge = avg("net_edge_bps")
    return {
        "scan_id": scan_id,
        "candidate_count": len(rows),
        "symbol": _dominant_value(row.get("symbol") for row in rows),
        "sector": _dominant_value(row.get("sector") for row in rows),
        "theme": _dominant_value(row.get("theme") for row in rows),
        "market_segment": _dominant_value(row.get("market_segment") for row in rows),
        "realized_return_bps": realized_return,
        "realized_risk_bps": realized_risk,
        "expected_return_bps": avg("expected_return_bps"),
        "expected_risk_bps": avg("expected_risk_bps"),
        "trading_cost_bps": trading_cost,
        "slippage_cost_bps": slippage_cost,
        "net_edge_bps": net_edge,
        "raw_score": avg("raw_score"),
        "composite_score": avg("composite_score"),
        "realized_net_edge_bps": (
            realized_return
            - realized_risk * REALIZED_RISK_WEIGHT
            - trading_cost
            - slippage_cost
        ),
    }


def _performance_summary_from_rows(
    rows: list[dict[str, Any]],
    *,
    top_count: int,
    sample_source: str,
    sample_limit: int | None = None,
    candidate_sample_count: int | None = None,
    scan_run_count: int | None = None,
    concentration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sample_count = len(rows)
    if sample_count <= 0:
        return _empty_top10_performance(limit=sample_limit)

    def values(key: str) -> list[float]:
        parsed = [_to_float(row.get(key)) for row in rows]
        return [value for value in parsed if value is not None]

    def total(key: str) -> float:
        return sum(values(key))

    def avg(key: str) -> float | None:
        items = values(key)
        return sum(items) / len(items) if items else None

    def avg_expr(items: list[float | None]) -> float | None:
        usable = [value for value in items if value is not None]
        return sum(usable) / len(usable) if usable else None

    realized_net_values = [
        _realized_net_edge_from_row(row)
        for row in rows
    ]
    return_errors = [
        (_to_float(row.get("realized_return_bps")) or 0.0)
        - _to_float(row.get("expected_return_bps"))
        if _to_float(row.get("expected_return_bps")) is not None
        else None
        for row in rows
    ]
    risk_errors = [
        (_to_float(row.get("realized_risk_bps")) or 0.0)
        - _to_float(row.get("expected_risk_bps"))
        if _to_float(row.get("expected_risk_bps")) is not None
        else None
        for row in rows
    ]
    net_edge_errors = [
        realized_net - _to_float(row.get("net_edge_bps"))
        if _to_float(row.get("net_edge_bps")) is not None
        else None
        for row, realized_net in zip(rows, realized_net_values, strict=True)
    ]

    returns = values("realized_return_bps")
    wins = [value for value in returns if value > 0]
    losses = [abs(value) for value in returns if value < 0]
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / sample_count
    loss_rate = loss_count / sample_count
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    expectancy = win_rate * avg_win - loss_rate * avg_loss
    total_realized_net = sum(realized_net_values)
    summary = {
        "status": "ready",
        "sample_count": sample_count,
        "top_count": top_count,
        "sample_source": sample_source,
        "sample_limit": sample_limit,
        "scan_run_count": scan_run_count if scan_run_count is not None else sample_count,
        "candidate_sample_count": candidate_sample_count,
        "metric_sample_count": len(values("expected_return_bps")),
        "total_return_bps": round(total("realized_return_bps"), 4),
        "total_risk_bps": round(total("realized_risk_bps"), 4),
        "total_realized_net_edge_bps": round(total_realized_net, 4),
        "total_expected_return_bps": round(total("expected_return_bps"), 4),
        "total_expected_risk_bps": round(total("expected_risk_bps"), 4),
        "total_predicted_net_edge_bps": round(total("net_edge_bps"), 4),
        "total_cost_bps": round(
            sum(
                (_to_float(row.get("trading_cost_bps")) or 0.0)
                + (_to_float(row.get("slippage_cost_bps")) or 0.0)
                for row in rows
            ),
            4,
        ),
        "avg_return_bps": _round_optional(avg("realized_return_bps")),
        "avg_risk_bps": _round_optional(avg("realized_risk_bps")),
        "avg_realized_net_edge_bps": round(total_realized_net / sample_count, 4),
        "avg_expected_return_bps": _round_optional(avg("expected_return_bps")),
        "avg_expected_risk_bps": _round_optional(avg("expected_risk_bps")),
        "avg_predicted_net_edge_bps": _round_optional(avg("net_edge_bps")),
        "avg_raw_score": _round_optional(avg("raw_score")),
        "avg_composite_score": _round_optional(avg("composite_score")),
        "avg_cost_bps": _round_optional(
            sum(
                (_to_float(row.get("trading_cost_bps")) or 0.0)
                + (_to_float(row.get("slippage_cost_bps")) or 0.0)
                for row in rows
            )
            / sample_count
        ),
        "avg_return_error_bps": _round_optional(avg_expr(return_errors)),
        "mae_return_error_bps": _round_optional(
            avg_expr([abs(value) if value is not None else None for value in return_errors])
        ),
        "avg_risk_error_bps": _round_optional(avg_expr(risk_errors)),
        "mae_risk_error_bps": _round_optional(
            avg_expr([abs(value) if value is not None else None for value in risk_errors])
        ),
        "avg_net_edge_error_bps": _round_optional(avg_expr(net_edge_errors)),
        "mae_net_edge_error_bps": _round_optional(
            avg_expr([abs(value) if value is not None else None for value in net_edge_errors])
        ),
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": round(win_rate, 6),
        "loss_rate": round(loss_rate, 6),
        "avg_win_bps": round(avg_win, 4),
        "avg_loss_bps": round(avg_loss, 4),
        "reward_risk_ratio": round(avg_win / avg_loss, 4) if avg_loss else None,
        "expectancy_bps": round(expectancy, 4),
        "expectancy_formula": "E = (win_rate * avg_win_bps) - (loss_rate * avg_loss_bps)",
        "net_edge_formula": "net_edge = (expected_return - expected_risk * EDGE_RISK_WEIGHT - trading_cost - slippage_cost)",
        "realized_net_edge_formula": REALIZED_NET_EDGE_FORMULA,
        "composite_score_formula": COMPOSITE_SCORE_FORMULA,
    }
    if concentration is not None:
        summary["concentration"] = concentration
    return summary


def _realized_net_edge_from_row(row: dict[str, Any]) -> float:
    return (
        (_to_float(row.get("realized_return_bps")) or 0.0)
        - ((_to_float(row.get("realized_risk_bps")) or 0.0) * REALIZED_RISK_WEIGHT)
        - (_to_float(row.get("trading_cost_bps")) or 0.0)
        - (_to_float(row.get("slippage_cost_bps")) or 0.0)
    )


def _dominant_value(values: Iterator[Any]) -> str | None:
    clean = [str(value) for value in values if value not in (None, "")]
    if not clean:
        return None
    return Counter(clean).most_common(1)[0][0]


def _concentration_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sample_count = len(rows)
    if sample_count <= 0:
        return {
            "sample_count": 0,
            "max_symbol_share": None,
            "max_sector_share": None,
            "max_theme_share": None,
        }
    return {
        "sample_count": sample_count,
        **_max_share(rows, "symbol", "symbol"),
        **_max_share(rows, "sector", "sector"),
        **_max_share(rows, "theme", "theme"),
    }


def _max_share(
    rows: list[dict[str, Any]],
    key: str,
    label: str,
) -> dict[str, Any]:
    values = [str(row.get(key)) for row in rows if row.get(key)]
    if not values:
        return {
            f"max_{label}": None,
            f"max_{label}_count": 0,
            f"max_{label}_share": None,
        }
    value, count = Counter(values).most_common(1)[0]
    return {
        f"max_{label}": value,
        f"max_{label}_count": count,
        f"max_{label}_share": round(count / len(rows), 6),
    }


def _unit_performance_summary(
    *,
    calibration_path: Path,
) -> dict[str, Any]:
    return {
        CANDIDATE_LABEL_UNIT: _candidate_label_performance_from_store(calibration_path),
        PAPER_ORDER_UNIT: _paper_order_performance(
            settings.storage_path(paper_trading.DEFAULT_DB_PATH)
        ),
        ACTUAL_BROKER_FILL_UNIT: _actual_broker_fill_performance(
            outcome_path=settings.storage_path(settings.outcome_attribution_db_path),
            broker_path=settings.storage_path(settings.broker_sync_db_path),
        ),
    }


def _empty_unit_performance() -> dict[str, Any]:
    return {
        CANDIDATE_LABEL_UNIT: _empty_win_rate(CANDIDATE_LABEL_UNIT),
        PAPER_ORDER_UNIT: _empty_win_rate(PAPER_ORDER_UNIT),
        ACTUAL_BROKER_FILL_UNIT: _empty_win_rate(ACTUAL_BROKER_FILL_UNIT),
    }


def _candidate_label_performance_from_store(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_win_rate(CANDIDATE_LABEL_UNIT)
    initialize_edge_calibration_db(path)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            WITH ranked AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol, COALESCE(label_horizon_key, symbol || ':' || scan_time)
                        ORDER BY observed_at ASC, id ASC
                    ) AS symbol_horizon_rank
                FROM edge_training_samples
                WHERE {ELIGIBLE_EDGE_SAMPLE_STATUS_SQL}
            )
            SELECT
                id, scan_id, label_horizon_key, symbol, scan_time, observed_at,
                realized_return_bps, realized_risk_bps, expected_return_bps,
                expected_risk_bps, trading_cost_bps, slippage_cost_bps,
                net_edge_bps, raw_score, composite_score, market_segment,
                sector, theme, raw_json
            FROM ranked
            WHERE symbol_horizon_rank = 1
            ORDER BY observed_at DESC, id DESC
            """
        ).fetchall()
    if not rows:
        return _empty_win_rate(CANDIDATE_LABEL_UNIT)
    performance = _performance_summary_from_rows(
        [dict(row) for row in rows],
        top_count=0,
        sample_source="symbol_horizon_deduped_candidate_labels",
        concentration=_concentration_from_rows([dict(row) for row in rows]),
    )
    return {
        "status": "ready",
        "unit": CANDIDATE_LABEL_UNIT,
        "sample_count": performance["sample_count"],
        "win_count": performance["win_count"],
        "loss_count": performance["loss_count"],
        "win_rate": performance["win_rate"],
        "avg_return_bps": performance["avg_return_bps"],
        "avg_realized_net_edge_bps": performance["avg_realized_net_edge_bps"],
        "sample_source": performance["sample_source"],
        "concentration": performance.get("concentration"),
    }


def _paper_order_performance(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_win_rate(PAPER_ORDER_UNIT)
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    symbol, side, status, amount, requested_amount,
                    net_realized_pnl, realized_pnl, gross_realized_pnl
                FROM paper_orders
                WHERE UPPER(COALESCE(side, '')) = 'SELL'
                   OR COALESCE(net_realized_pnl, realized_pnl, gross_realized_pnl, 0) != 0
                ORDER BY id DESC
                """
            ).fetchall()
    except sqlite3.Error as exc:
        return {"status": "error", "unit": PAPER_ORDER_UNIT, "message": str(exc)}
    pnl_values = [
        _to_float(
            row["net_realized_pnl"]
            if row["net_realized_pnl"] is not None
            else row["realized_pnl"]
            if row["realized_pnl"] is not None
            else row["gross_realized_pnl"]
        )
        for row in rows
    ]
    return _win_rate_from_values(
        unit=PAPER_ORDER_UNIT,
        values=[value for value in pnl_values if value is not None],
        value_name="realized_pnl",
    )


def _actual_broker_fill_performance(
    *,
    outcome_path: Path,
    broker_path: Path,
) -> dict[str, Any]:
    outcome = _broker_outcome_performance(outcome_path)
    if outcome["status"] != "empty":
        return outcome
    filled_count = 0
    if broker_path.exists():
        try:
            with sqlite3.connect(broker_path) as conn:
                filled_count = int(
                    conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM broker_order_executions
                        WHERE COALESCE(filled_qty, 0) > 0
                        """
                    ).fetchone()[0]
                    or 0
                )
        except sqlite3.Error:
            filled_count = 0
    return {
        **_empty_win_rate(ACTUAL_BROKER_FILL_UNIT),
        "status": "no_outcome_labels" if filled_count else "empty",
        "filled_execution_count": filled_count,
        "message": (
            "Broker fills exist, but no realized outcome labels are available yet"
            if filled_count
            else "No broker fill outcomes are available yet"
        ),
    }


def _broker_outcome_performance(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_win_rate(ACTUAL_BROKER_FILL_UNIT)
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT realized_net_edge_bps, outcome_label, execution_mode
                FROM outcome_attribution_events
                WHERE LOWER(COALESCE(execution_mode, '')) NOT IN ('', 'paper')
                ORDER BY created_at DESC, id DESC
                """
            ).fetchall()
    except sqlite3.Error as exc:
        return {
            "status": "error",
            "unit": ACTUAL_BROKER_FILL_UNIT,
            "message": str(exc),
        }
    if not rows:
        return _empty_win_rate(ACTUAL_BROKER_FILL_UNIT)
    values = [_to_float(row["realized_net_edge_bps"]) for row in rows]
    return _win_rate_from_values(
        unit=ACTUAL_BROKER_FILL_UNIT,
        values=[value for value in values if value is not None],
        value_name="realized_net_edge_bps",
    )


def _win_rate_from_values(
    *,
    unit: str,
    values: list[float],
    value_name: str,
) -> dict[str, Any]:
    if not values:
        return _empty_win_rate(unit)
    wins = [value for value in values if value > 0]
    losses = [value for value in values if value < 0]
    return {
        "status": "ready",
        "unit": unit,
        "sample_count": len(values),
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate": round(len(wins) / len(values), 6),
        f"avg_{value_name}": round(sum(values) / len(values), 4),
    }


def _empty_win_rate(unit: str) -> dict[str, Any]:
    return {
        "status": "empty",
        "unit": unit,
        "sample_count": 0,
        "win_count": 0,
        "loss_count": 0,
        "win_rate": None,
    }


def _empty_top10_performance(limit: int | None = None) -> dict[str, Any]:
    return {
        "status": "empty",
        "sample_count": 0,
        "top_count": 10,
        "sample_source": TOP10_SAMPLE_SOURCE_SCAN_RUN,
        "sample_limit": limit,
        "scan_run_count": 0,
        "candidate_sample_count": 0,
        "metric_sample_count": 0,
        "total_return_bps": 0.0,
        "total_risk_bps": 0.0,
        "total_realized_net_edge_bps": 0.0,
        "total_expected_return_bps": 0.0,
        "total_expected_risk_bps": 0.0,
        "total_predicted_net_edge_bps": 0.0,
        "total_cost_bps": 0.0,
        "avg_return_bps": None,
        "avg_risk_bps": None,
        "avg_realized_net_edge_bps": None,
        "avg_expected_return_bps": None,
        "avg_expected_risk_bps": None,
        "avg_predicted_net_edge_bps": None,
        "avg_raw_score": None,
        "avg_composite_score": None,
        "avg_cost_bps": None,
        "avg_return_error_bps": None,
        "mae_return_error_bps": None,
        "avg_risk_error_bps": None,
        "mae_risk_error_bps": None,
        "avg_net_edge_error_bps": None,
        "mae_net_edge_error_bps": None,
        "win_count": 0,
        "loss_count": 0,
        "win_rate": None,
        "loss_rate": None,
        "avg_win_bps": None,
        "avg_loss_bps": None,
        "reward_risk_ratio": None,
        "expectancy_bps": None,
        "net_edge_formula": "net_edge = (expected_return - expected_risk * EDGE_RISK_WEIGHT - trading_cost - slippage_cost)",
        "realized_net_edge_formula": REALIZED_NET_EDGE_FORMULA,
        "concentration": {
            "sample_count": 0,
            "max_symbol_share": None,
            "max_sector_share": None,
            "max_theme_share": None,
        },
    }


def _latest_top10_performance_from_store(path: Path) -> dict[str, Any] | None:
    initialize_edge_calibration_db(path)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            f"""
            SELECT raw_json
            FROM top_candidate_performance
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    if not row:
        return None
    parsed = _parse_json(row["raw_json"], None)
    return parsed if isinstance(parsed, dict) else None


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
            conn.execute(f"SELECT COUNT(*) FROM edge_training_samples WHERE {ELIGIBLE_EDGE_SAMPLE_STATUS_SQL}").fetchone()[0]
            or 0
        )


def _empty_sample_message(diagnostics: dict[str, Any]) -> str:
    if not diagnostics.get("universe_db_exists"):
        return "Universe scanner DB does not exist yet; run universe scanning first."
    if not diagnostics.get("universe_scan_count"):
        return "No universe scan runs are stored yet."
    if not diagnostics.get("scanner_candidate_history_count"):
        return "Universe scans exist, but scanner_candidate_history is empty."
    if not diagnostics.get("eligible_candidate_history_count"):
        return "Scanner history exists, but no candidates have a valid current_price."
    if not diagnostics.get("candidate_with_future_snapshot_count"):
        return (
            "Candidates exist, but no later price snapshots are available yet; "
            "edge labels need a future observed price after each scan."
        )
    return "No edge samples are stored yet; run edge sample refresh or wait for the orchestrator cycle."


def _safe_sqlite_scalar(path: Path, query: str) -> Any:
    if not path.exists():
        return 0
    try:
        with connect_sqlite(path) as conn:
            return conn.execute(query).fetchone()[0]
    except (sqlite3.Error, TypeError, IndexError):
        return 0


def _empty_sample_summary() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "metric_sample_count": 0,
        "total_return_bps": 0.0,
        "total_risk_bps": 0.0,
        "net_total_bps": 0.0,
        "total_realized_net_edge_bps": 0.0,
        "total_expected_return_bps": 0.0,
        "total_expected_risk_bps": 0.0,
        "total_predicted_net_edge_bps": 0.0,
        "total_cost_bps": 0.0,
        "avg_return_bps": None,
        "avg_risk_bps": None,
        "avg_net_bps": None,
        "avg_realized_net_edge_bps": None,
        "avg_expected_return_bps": None,
        "avg_expected_risk_bps": None,
        "avg_predicted_net_edge_bps": None,
        "avg_raw_score": None,
        "avg_composite_score": None,
        "avg_cost_bps": None,
        "avg_return_error_bps": None,
        "mae_return_error_bps": None,
        "avg_risk_error_bps": None,
        "mae_risk_error_bps": None,
        "avg_net_edge_error_bps": None,
        "mae_net_edge_error_bps": None,
        "min_return_bps": None,
        "max_return_bps": None,
        "win_count": 0,
        "loss_count": 0,
        "win_rate": None,
        "net_edge_win_count": 0,
        "net_edge_loss_count": 0,
        "net_edge_win_rate": None,
        "net_edge_formula": "net_edge = (expected_return - expected_risk * EDGE_RISK_WEIGHT - trading_cost - slippage_cost)",
        "realized_net_edge_formula": REALIZED_NET_EDGE_FORMULA,
    }


def _sample_summary_from_row(row: dict[str, Any]) -> dict[str, Any]:
    sample_count = int(row.get("sample_count") or 0)
    total_return = float(row.get("total_return_bps") or 0.0)
    total_risk = float(row.get("total_risk_bps") or 0.0)
    total_realized_net = float(row.get("total_realized_net_edge_bps") or 0.0)
    win_count = int(row.get("win_count") or 0)
    loss_count = int(row.get("loss_count") or 0)
    net_edge_win_count = int(row.get("net_edge_win_count") or 0)
    net_edge_loss_count = int(row.get("net_edge_loss_count") or 0)
    if sample_count <= 0:
        return _empty_sample_summary()
    return {
        "sample_count": sample_count,
        "metric_sample_count": int(row.get("metric_sample_count") or 0),
        "total_return_bps": round(total_return, 4),
        "total_risk_bps": round(total_risk, 4),
        "net_total_bps": round(total_return - total_risk, 4),
        "total_realized_net_edge_bps": round(total_realized_net, 4),
        "total_expected_return_bps": _round_optional(row.get("total_expected_return_bps")) or 0.0,
        "total_expected_risk_bps": _round_optional(row.get("total_expected_risk_bps")) or 0.0,
        "total_predicted_net_edge_bps": _round_optional(row.get("total_predicted_net_edge_bps")) or 0.0,
        "total_cost_bps": _round_optional(row.get("total_cost_bps")) or 0.0,
        "avg_return_bps": _round_optional(row.get("avg_return_bps")),
        "avg_risk_bps": _round_optional(row.get("avg_risk_bps")),
        "avg_net_bps": round((total_return - total_risk) / sample_count, 4),
        "avg_realized_net_edge_bps": _round_optional(row.get("avg_realized_net_edge_bps"))
        if row.get("avg_realized_net_edge_bps") is not None
        else round(total_realized_net / sample_count, 4),
        "avg_expected_return_bps": _round_optional(row.get("avg_expected_return_bps")),
        "avg_expected_risk_bps": _round_optional(row.get("avg_expected_risk_bps")),
        "avg_predicted_net_edge_bps": _round_optional(row.get("avg_predicted_net_edge_bps")),
        "avg_raw_score": _round_optional(row.get("avg_raw_score")),
        "avg_composite_score": _round_optional(row.get("avg_composite_score")),
        "avg_cost_bps": _round_optional(row.get("avg_cost_bps")),
        "avg_return_error_bps": _round_optional(row.get("avg_return_error_bps")),
        "mae_return_error_bps": _round_optional(row.get("mae_return_error_bps")),
        "avg_risk_error_bps": _round_optional(row.get("avg_risk_error_bps")),
        "mae_risk_error_bps": _round_optional(row.get("mae_risk_error_bps")),
        "avg_net_edge_error_bps": _round_optional(row.get("avg_net_edge_error_bps")),
        "mae_net_edge_error_bps": _round_optional(row.get("mae_net_edge_error_bps")),
        "min_return_bps": _round_optional(row.get("min_return_bps")),
        "max_return_bps": _round_optional(row.get("max_return_bps")),
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": round(win_count / sample_count, 4),
        "net_edge_win_count": net_edge_win_count,
        "net_edge_loss_count": net_edge_loss_count,
        "net_edge_win_rate": round(net_edge_win_count / sample_count, 4),
        "net_edge_formula": "net_edge = (expected_return - expected_risk * EDGE_RISK_WEIGHT - trading_cost - slippage_cost)",
        "realized_net_edge_formula": REALIZED_NET_EDGE_FORMULA,
        "composite_score_formula": COMPOSITE_SCORE_FORMULA,
    }


def _sample_row(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    for key in (
        "realized_return_bps",
        "realized_risk_bps",
        "raw_score",
        "expected_return_bps",
        "expected_risk_bps",
        "trading_cost_bps",
        "slippage_cost_bps",
        "net_edge_bps",
        "composite_score",
    ):
        item[key] = _round_optional(item.get(key))
    item["realized_net_edge_bps"] = _round_optional(
        (_to_float(item.get("realized_return_bps")) or 0.0)
        - ((_to_float(item.get("realized_risk_bps")) or 0.0) * REALIZED_RISK_WEIGHT)
        - (_to_float(item.get("trading_cost_bps")) or 0.0)
        - (_to_float(item.get("slippage_cost_bps")) or 0.0)
    )
    item["net_edge_error_bps"] = _round_optional(
        (_to_float(item.get("realized_net_edge_bps")) or 0.0)
        - (_to_float(item.get("net_edge_bps")) or 0.0)
        if item.get("net_edge_bps") is not None
        else None
    )
    return item


def _symbol_sample_row(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    sample_count = int(item.get("sample_count") or 0)
    win_count = int(item.get("win_count") or 0)
    total_return = float(item.get("total_return_bps") or 0.0)
    item["sample_count"] = sample_count
    item["win_count"] = win_count
    item["total_return_bps"] = round(total_return, 4)
    item["total_predicted_net_edge_bps"] = _round_optional(
        item.get("total_predicted_net_edge_bps")
    ) or 0.0
    item["avg_return_bps"] = _round_optional(item.get("avg_return_bps"))
    item["avg_risk_bps"] = _round_optional(item.get("avg_risk_bps"))
    item["avg_expected_return_bps"] = _round_optional(item.get("avg_expected_return_bps"))
    item["avg_predicted_net_edge_bps"] = _round_optional(item.get("avg_predicted_net_edge_bps"))
    item["win_rate"] = round(win_count / sample_count, 4) if sample_count else None
    return item


def _round_optional(value: Any, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


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
    execution_mode: str | None = None,
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
            "max_mae_net_edge_error_bps": getattr(
                settings,
                "edge_calibration_gate_max_mae_net_edge_bps",
                180.0,
            ),
            "min_top10_avg_return_bps": _effective_min_top10_avg_return_bps(
                execution_mode=execution_mode,
            ),
            "target_top10_win_rate": settings.edge_calibration_gate_min_top10_win_rate,
            "min_top10_expectancy_bps": settings.edge_calibration_gate_min_top10_expectancy_bps,
            "min_fill_adjusted_edge_bps": settings.edge_calibration_gate_min_fill_adjusted_edge_bps,
        },
    }


def broker_paper_fill_gate_metrics(
    *,
    broker_db_path: Path | str | None = None,
    outcome_db_path: Path | str | None = None,
) -> dict[str, Any]:
    broker_path = settings.storage_path(broker_db_path or settings.broker_sync_db_path)
    outcome_path = settings.storage_path(
        outcome_db_path or settings.outcome_attribution_db_path
    )
    filled_count = _broker_paper_filled_execution_count(broker_path)
    outcome_rows = _broker_paper_outcome_rows(outcome_path)
    values = [
        _to_float(row.get("realized_net_edge_bps"))
        for row in outcome_rows
    ]
    realized_values = [value for value in values if value is not None]
    wins = [value for value in realized_values if value > 0]
    losses = [value for value in realized_values if value < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else None
    )
    errors = []
    for row in outcome_rows:
        realized = _to_float(row.get("realized_net_edge_bps"))
        predicted = _to_float(row.get("predicted_net_edge_bps"))
        if predicted is None:
            predicted = _to_float(row.get("final_entry_edge_bps"))
        if realized is not None and predicted is not None:
            errors.append(abs(realized - predicted))

    return {
        "broker_paper_fill_sample_count": filled_count,
        "broker_paper_oos_fill_sample_count": _broker_paper_oos_count(outcome_rows),
        "broker_paper_fill_outcome_sample_count": len(realized_values),
        "broker_paper_fill_win_rate": (
            round(len(wins) / len(realized_values), 6)
            if realized_values
            else None
        ),
        "broker_paper_fill_profit_factor": (
            round(profit_factor, 6)
            if profit_factor is not None
            else None
        ),
        "broker_paper_fill_avg_realized_net_edge_bps": (
            round(sum(realized_values) / len(realized_values), 4)
            if realized_values
            else None
        ),
        "broker_paper_fill_mae_edge_error_bps": (
            round(sum(errors) / len(errors), 4)
            if errors
            else None
        ),
        "broker_sync_db_path": str(broker_path),
        "outcome_attribution_db_path": str(outcome_path),
    }


def _broker_paper_filled_execution_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with sqlite3.connect(path, timeout=2) as conn:
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM broker_order_executions
                WHERE COALESCE(filled_qty, 0) > 0
                """
            ).fetchone()
            return int((row or [0])[0] or 0)
    except sqlite3.Error:
        return 0


def _broker_paper_outcome_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with sqlite3.connect(path, timeout=2) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT created_at, realized_net_edge_bps,
                       predicted_net_edge_bps, final_entry_edge_bps
                FROM outcome_attribution_events
                WHERE LOWER(COALESCE(execution_mode, '')) = 'broker_paper'
                ORDER BY created_at ASC, id ASC
                """
            ).fetchall()
            return [dict(row) for row in rows]
    except sqlite3.Error:
        return []


def _broker_paper_oos_count(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    split_index = int(len(rows) * 0.75)
    return max(1, len(rows) - split_index)


def _broker_paper_fill_gate_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    min_fill = int(settings.broker_paper_min_fill_samples or 0)
    min_oos = int(settings.broker_paper_min_oos_fill_samples or 0)
    fill_count = int(metrics.get("broker_paper_fill_sample_count") or 0)
    oos_count = int(metrics.get("broker_paper_oos_fill_sample_count") or 0)
    ready_failures: list[str] = []
    if fill_count < min_fill:
        ready_failures.append(
            f"broker_paper_fill_sample_count {fill_count}/{min_fill}"
        )
    if oos_count < min_oos:
        ready_failures.append(
            f"broker_paper_oos_fill_sample_count {oos_count}/{min_oos}"
        )

    ready = not ready_failures
    failures: list[str] = []
    if ready:
        min_win_rate = float(settings.edge_calibration_gate_min_top10_win_rate or 0.50)
        min_profit_factor = float(
            getattr(settings, "edge_calibration_gate_min_profit_factor", 1.10)
            or 1.10
        )
        max_mae = float(
            getattr(settings, "edge_calibration_gate_max_mae_net_edge_bps", 180.0)
            or 180.0
        )
        win_rate = _to_float(metrics.get("broker_paper_fill_win_rate"))
        profit_factor = _to_float(metrics.get("broker_paper_fill_profit_factor"))
        avg_edge = _to_float(
            metrics.get("broker_paper_fill_avg_realized_net_edge_bps")
        )
        mae = _to_float(metrics.get("broker_paper_fill_mae_edge_error_bps"))
        if win_rate is None or win_rate < min_win_rate:
            failures.append(
                f"broker_paper_fill_win_rate {win_rate} < {min_win_rate}"
            )
        if profit_factor is None or profit_factor < min_profit_factor:
            failures.append(
                f"broker_paper_fill_profit_factor {profit_factor} < {min_profit_factor}"
            )
        if avg_edge is None or avg_edge <= 0:
            failures.append(
                f"broker_paper_fill_avg_realized_net_edge_bps {avg_edge} <= 0"
            )
        if mae is not None and mae >= max_mae:
            failures.append(
                f"broker_paper_fill_mae_edge_error_bps {mae} >= {max_mae}"
            )

    approved = ready and not failures
    blocked = ready and not approved
    return {
        "status": "approved" if approved else "collecting" if not ready else "blocked",
        "approved": approved,
        "ready": ready,
        "blocked": blocked,
        "message": (
            "Broker-paper fill calibration gate passed"
            if approved
            else "Broker-paper fill calibration gate collecting: "
            + "; ".join(ready_failures)
            if not ready
            else "Broker-paper fill calibration gate blocked entries: "
            + "; ".join(failures)
        ),
        "required": {
            "min_fill_samples": min_fill,
            "min_oos_fill_samples": min_oos,
            "min_win_rate": float(settings.edge_calibration_gate_min_top10_win_rate or 0.50),
            "min_profit_factor": float(
                getattr(settings, "edge_calibration_gate_min_profit_factor", 1.10)
                or 1.10
            ),
            "max_mae_edge_error_bps": float(
                getattr(settings, "edge_calibration_gate_max_mae_net_edge_bps", 180.0)
                or 180.0
            ),
        },
    }


def _apply_broker_paper_calibration_policy(
    gate: dict[str, Any],
    *,
    execution_mode: str | None,
) -> dict[str, Any]:
    mode = str(execution_mode or "").lower()
    candidate_failed = not bool(gate.get("approved", False))
    overlay: dict[str, Any] = {
        "broker_paper_bootstrap_enabled": bool(
            settings.broker_paper_bootstrap_enabled
        ),
        "broker_paper_calibration_source": str(
            settings.broker_paper_calibration_source or "candidate_labels"
        ).lower(),
        "broker_paper_candidate_label_gate_mode": str(
            settings.broker_paper_candidate_label_gate_mode or "hard_block"
        ).lower(),
        "broker_paper_min_fill_samples": int(
            settings.broker_paper_min_fill_samples or 0
        ),
        "broker_paper_min_oos_fill_samples": int(
            settings.broker_paper_min_oos_fill_samples or 0
        ),
        "candidate_label_gate_failed": candidate_failed,
        "candidate_label_gate_hard_blocking": candidate_failed,
        "broker_paper_bootstrap_allowed": False,
        "broker_paper_fill_gate_ready": False,
        "broker_paper_fill_gate_blocked": False,
        "broker_paper_fill_gate_hard_blocking": False,
        "calibration_gate_mode": (
            "candidate_label_hard_blocking"
            if candidate_failed
            else "candidate_label_approved"
        ),
    }

    if mode != "broker_paper":
        return {**gate, **overlay}

    metrics = broker_paper_fill_gate_metrics()
    fill_gate = _broker_paper_fill_gate_from_metrics(metrics)
    source = overlay["broker_paper_calibration_source"]
    label_mode = overlay["broker_paper_candidate_label_gate_mode"]
    overlay.update(
        {
            **metrics,
            "broker_paper_fill_gate": fill_gate,
            "broker_paper_fill_gate_ready": bool(fill_gate["ready"]),
            "broker_paper_fill_gate_blocked": bool(fill_gate["blocked"]),
            "broker_paper_fill_gate_hard_blocking": bool(fill_gate["blocked"]),
        }
    )

    if source != "broker_fills":
        return {**gate, **overlay}

    overlay["candidate_label_gate_hard_blocking"] = False

    if fill_gate["ready"]:
        overlay["calibration_gate_mode"] = "broker_paper_fill_gate"
        if fill_gate["approved"]:
            message = str(fill_gate["message"])
            if candidate_failed:
                message += (
                    "; candidate label calibration failed but is observe-only "
                    "because broker_paper uses broker fill calibration"
                )
            return {
                **gate,
                **overlay,
                "status": "approved",
                "approved": True,
                "message": message,
            }
        return {
            **gate,
            **overlay,
            "status": "blocked",
            "approved": False,
            "message": str(fill_gate["message"]),
        }

    bootstrap_observe_only = (
        bool(settings.broker_paper_bootstrap_enabled)
        and bool(settings.kis_is_paper)
        and label_mode == "observe_only"
    )
    if not bootstrap_observe_only:
        overlay["candidate_label_gate_hard_blocking"] = candidate_failed
        overlay["calibration_gate_mode"] = (
            "candidate_label_hard_blocking"
            if candidate_failed
            else "candidate_label_approved"
        )
        return {**gate, **overlay}

    overlay["broker_paper_bootstrap_allowed"] = True
    overlay["broker_paper_fill_gate_hard_blocking"] = False
    overlay["calibration_gate_mode"] = (
        "broker_paper_bootstrap_candidate_label_observe_only"
    )
    if not candidate_failed:
        return {**gate, **overlay}

    fill_count = int(metrics.get("broker_paper_fill_sample_count") or 0)
    min_fill = int(settings.broker_paper_min_fill_samples or 0)
    message = (
        "Candidate label calibration gate failed, but "
        "broker_paper bootstrap observe-only mode allowed entry because "
        f"broker_paper_fill_sample_count {fill_count}/{min_fill}. "
        "KIS mock broker order can proceed if all non-calibration guards pass. "
        f"{BROKER_PAPER_OBSERVE_ONLY_REASON}. "
        f"Original candidate label gate: {gate.get('message') or ''}"
    ).strip()
    return {
        **gate,
        **overlay,
        "status": "bootstrap_observe_only",
        "approved": True,
        "message": message,
    }


def _gate_from_metrics(
    *,
    sample_count: int,
    oos_sample_count: int,
    mae_return_bps: float | None,
    mae_risk_bps: float | None,
    top10_performance: dict[str, Any],
    fill_adjustment: dict[str, Any],
    ic_metrics: dict[str, Any] | None = None,
    candidates: list[dict[str, Any]] | None = None,
    execution_mode: str | None = None,
) -> dict[str, Any]:
    failures: list[str] = []

    min_samples = int(settings.edge_calibration_gate_min_samples or 1000)
    min_oos = int(settings.edge_calibration_gate_min_oos_samples or 200)
    max_return_mae = float(settings.edge_calibration_gate_max_mae_return_bps or 180.0)
    max_risk_mae = float(settings.edge_calibration_gate_max_mae_risk_bps or 180.0)

    configured_min_top10_return = float(
        settings.edge_calibration_gate_min_top10_avg_return_bps
        if settings.edge_calibration_gate_min_top10_avg_return_bps is not None
        else 2.0
    )
    min_top10_return = _effective_min_top10_avg_return_bps(
        execution_mode=execution_mode,
    )

    target_top10_win_rate = float(
        settings.edge_calibration_gate_min_top10_win_rate
        if settings.edge_calibration_gate_min_top10_win_rate is not None
        else 0.50
    )

    min_top10_expectancy = float(
        settings.edge_calibration_gate_min_top10_expectancy_bps
        if settings.edge_calibration_gate_min_top10_expectancy_bps is not None
        else 0.0
    )

    min_fill_adjusted_edge = float(
        settings.edge_calibration_gate_min_fill_adjusted_edge_bps or 60.0
    )

    max_net_edge_mae = float(
        getattr(settings, "edge_calibration_gate_max_mae_net_edge_bps", 180.0)
        or 180.0
    )

    min_profit_factor = float(
        getattr(settings, "edge_calibration_gate_min_profit_factor", 1.10) or 1.10
    )

    min_recent_ic = float(
        getattr(settings, "edge_calibration_gate_min_ic", 0.02) or 0.02
    )

    min_cost_coverage = float(
        getattr(settings, "edge_calibration_gate_min_cost_coverage", 2.0) or 2.0
    )

    concentration_min_samples = int(
        getattr(settings, "edge_calibration_gate_min_concentration_samples", 20)
        or 20
    )
    max_symbol_share = float(
        getattr(settings, "edge_calibration_gate_max_symbol_share", 0.20) or 0.20
    )
    max_sector_share = float(
        getattr(settings, "edge_calibration_gate_max_sector_share", 0.35) or 0.35
    )
    max_theme_share = float(
        getattr(settings, "edge_calibration_gate_max_theme_share", 0.35) or 0.35
    )

    multiplier = _to_float(fill_adjustment.get("multiplier")) or 1.0
    best_fill_adjusted_edge: float | None = None
    best_cost_coverage: float | None = None

    if sample_count < min_samples:
        failures.append(f"sample_count {sample_count}/{min_samples}")

    if oos_sample_count < min_oos:
        failures.append(f"oos_sample_count {oos_sample_count}/{min_oos}")

    if mae_return_bps is None or mae_return_bps > max_return_mae:
        failures.append(f"mae_return_bps {mae_return_bps} > {max_return_mae}")

    if mae_risk_bps is None or mae_risk_bps > max_risk_mae:
        failures.append(f"mae_risk_bps {mae_risk_bps} > {max_risk_mae}")

    top10_avg_return = _to_float(top10_performance.get("avg_return_bps"))
    top10_expectancy = _to_float(top10_performance.get("expectancy_bps"))
    top10_win_rate = _to_float(top10_performance.get("win_rate"))
    top10_net_edge_mae = _to_float(top10_performance.get("mae_net_edge_error_bps"))
    top10_profit_factor = estimate_profit_factor_from_top10(top10_performance)

    if top10_avg_return is None or top10_avg_return < min_top10_return:
        failures.append(
            f"top10_avg_return_bps {top10_avg_return} < {min_top10_return}"
        )

    if top10_expectancy is None or top10_expectancy <= min_top10_expectancy:
        failures.append(
            f"top10_expectancy_bps {top10_expectancy} <= {min_top10_expectancy}"
        )

    if top10_win_rate is None or top10_win_rate < target_top10_win_rate:
        failures.append(
            f"top10_win_rate {top10_win_rate} < {target_top10_win_rate}"
        )

    if top10_profit_factor is None or top10_profit_factor < min_profit_factor:
        failures.append(
            f"top10_profit_factor {top10_profit_factor} < {min_profit_factor}"
        )

    if (
        "mae_net_edge_error_bps" in top10_performance
        and (top10_net_edge_mae is None or top10_net_edge_mae >= max_net_edge_mae)
    ):
        failures.append(
            f"mae_net_edge_error_bps {top10_net_edge_mae} >= {max_net_edge_mae}"
        )

    concentration = top10_performance.get("concentration") or {}
    if isinstance(concentration, dict):
        concentration_sample_count = int(
            concentration.get("sample_count")
            or top10_performance.get("candidate_sample_count")
            or 0
        )
        if concentration_sample_count >= concentration_min_samples:
            symbol_share = _to_float(concentration.get("max_symbol_share"))
            sector_share = _to_float(concentration.get("max_sector_share"))
            theme_share = _to_float(concentration.get("max_theme_share"))
            if symbol_share is not None and symbol_share > max_symbol_share:
                failures.append(
                    f"symbol_concentration {symbol_share} > {max_symbol_share}"
                )
            if sector_share is not None and sector_share > max_sector_share:
                failures.append(
                    f"sector_concentration {sector_share} > {max_sector_share}"
                )
            if theme_share is not None and theme_share > max_theme_share:
                failures.append(
                    f"theme_concentration {theme_share} > {max_theme_share}"
                )

    recent_ic = _to_float((ic_metrics or {}).get("ic"))

    if recent_ic is None or recent_ic < min_recent_ic:
        failures.append(f"recent_ic {recent_ic} < {min_recent_ic}")

    if candidates:
        candidate_edges: list[float] = []
        cost_coverages: list[float] = []

        for candidate in candidates:
            if not _is_edge_training_sample_status_allowed(
                candidate.get("status") or candidate.get("decision")
            ):
                continue

            edge_value = _to_float(
                candidate.get("expected_net_edge")
                or candidate.get("net_edge_bps")
                or candidate.get("net_edge")
            )

            if edge_value is not None:
                candidate_edges.append(edge_value * multiplier)

            coverage = candidate_cost_coverage(candidate)
            if coverage is not None:
                cost_coverages.append(coverage)

        if candidate_edges:
            best_fill_adjusted_edge = max(candidate_edges)
            if best_fill_adjusted_edge < min_fill_adjusted_edge:
                failures.append(
                    "best_fill_adjusted_edge_bps "
                    f"{best_fill_adjusted_edge:.2f} < {min_fill_adjusted_edge:.2f}"
                )

        if cost_coverages:
            best_cost_coverage = max(cost_coverages)
            if best_cost_coverage < min_cost_coverage:
                failures.append(
                    f"best_cost_coverage {best_cost_coverage:.2f} < {min_cost_coverage:.2f}"
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
        "top10_performance": {
            **top10_performance,
            "profit_factor": (
                round(top10_profit_factor, 4)
                if top10_profit_factor is not None
                else None
            ),
            "profit_factor_formula": "profit_factor = gross_profit_bps / gross_loss_bps",
        },
        "fill_adjustment": fill_adjustment,
        "ic_metrics": ic_metrics,
        "best_fill_adjusted_edge_bps": (
            round(best_fill_adjusted_edge, 4)
            if best_fill_adjusted_edge is not None
            else None
        ),
        "best_cost_coverage": (
            round(best_cost_coverage, 4)
            if best_cost_coverage is not None
            else None
        ),
        "required": {
            "min_samples": min_samples,
            "min_oos_samples": min_oos,
            "max_mae_return_bps": max_return_mae,
            "max_mae_risk_bps": max_risk_mae,
            "max_mae_net_edge_error_bps": max_net_edge_mae,
            "min_top10_avg_return_bps": min_top10_return,
            "configured_min_top10_avg_return_bps": configured_min_top10_return,
            "target_top10_win_rate": target_top10_win_rate,
            "min_top10_expectancy_bps": min_top10_expectancy,
            "min_fill_adjusted_edge_bps": min_fill_adjusted_edge,
            "min_profit_factor": min_profit_factor,
            "min_recent_ic": min_recent_ic,
            "min_cost_coverage": min_cost_coverage,
            "min_concentration_samples": concentration_min_samples,
            "max_symbol_share": max_symbol_share,
            "max_sector_share": max_sector_share,
            "max_theme_share": max_theme_share,
        },
    }


def _effective_min_top10_avg_return_bps(
    *,
    execution_mode: str | None = None,
) -> float:
    configured = float(
        settings.edge_calibration_gate_min_top10_avg_return_bps
        if settings.edge_calibration_gate_min_top10_avg_return_bps is not None
        else 2.0
    )
    if execution_mode == "paper":
        paper_min = float(
            settings.edge_calibration_paper_min_top10_avg_return_bps
            if settings.edge_calibration_paper_min_top10_avg_return_bps is not None
            else configured
        )
        return min(configured, paper_min)
    return configured


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
            f"""
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
    _ensure_column(
        conn,
        "edge_training_samples",
        "label_observation_span_seconds",
        "INTEGER",
    )
    _ensure_column(conn, "edge_training_samples", "scan_id", "TEXT")
    _ensure_column(conn, "edge_training_samples", "label_horizon_key", "TEXT")
    _ensure_column(conn, "edge_training_samples", "market_segment", "TEXT")
    _ensure_column(conn, "edge_training_samples", "sector", "TEXT")
    _ensure_column(conn, "edge_training_samples", "theme", "TEXT")
    for column in (
        "raw_score",
        "expected_return_bps",
        "expected_risk_bps",
        "trading_cost_bps",
        "slippage_cost_bps",
        "net_edge_bps",
        "composite_score",
    ):
        _ensure_column(conn, "edge_training_samples", column, "REAL")
    _backfill_training_sample_metrics(conn)
    _backfill_training_sample_label_spans(conn)
    _backfill_training_sample_identity(conn)


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


def _backfill_training_sample_metrics(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    last_id = 0
    while True:
        rows = conn.execute(
            f"""
            SELECT id, raw_json
            FROM edge_training_samples
            WHERE id > ?
              AND (
                  raw_score IS NULL
                  OR expected_return_bps IS NULL
                  OR expected_risk_bps IS NULL
                  OR trading_cost_bps IS NULL
                  OR slippage_cost_bps IS NULL
                  OR net_edge_bps IS NULL
                  OR composite_score IS NULL
              )
            ORDER BY id ASC
            LIMIT 200
            """,
            (last_id,),
        ).fetchall()
        if not rows:
            break
        last_id = int(rows[-1]["id"])
        updates = []
        for row in rows:
            metrics = _sample_metrics_from_raw_json(row["raw_json"])
            if not any(value is not None for value in metrics.values()):
                continue
            updates.append(
                (
                    metrics.get("raw_score"),
                    metrics.get("expected_return_bps"),
                    metrics.get("expected_risk_bps"),
                    metrics.get("trading_cost_bps"),
                    metrics.get("slippage_cost_bps"),
                    metrics.get("net_edge_bps"),
                    metrics.get("composite_score"),
                    row["id"],
                )
            )
        if not updates:
            break
        conn.executemany(
            """
            UPDATE edge_training_samples
            SET raw_score = COALESCE(raw_score, ?),
                expected_return_bps = COALESCE(expected_return_bps, ?),
                expected_risk_bps = COALESCE(expected_risk_bps, ?),
                trading_cost_bps = COALESCE(trading_cost_bps, ?),
                slippage_cost_bps = COALESCE(slippage_cost_bps, ?),
                net_edge_bps = COALESCE(net_edge_bps, ?),
                composite_score = COALESCE(composite_score, ?)
            WHERE id = ?
            """,
            updates,
        )


def _backfill_training_sample_label_spans(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    last_id = 0
    while True:
        rows = conn.execute(
            f"""
            SELECT id, raw_json
            FROM edge_training_samples
            WHERE id > ?
              AND label_observation_span_seconds IS NULL
            ORDER BY id ASC
            LIMIT 200
            """,
            (last_id,),
        ).fetchall()
        if not rows:
            break
        last_id = int(rows[-1]["id"])
        updates = []
        for row in rows:
            span = _label_span_from_raw_json(row["raw_json"])
            if span is None:
                continue
            updates.append((span, row["id"]))
        if not updates:
            continue
        conn.executemany(
            """
            UPDATE edge_training_samples
            SET label_observation_span_seconds = ?
            WHERE id = ?
            """,
            updates,
        )


def _backfill_training_sample_identity(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    horizon_seconds = int(settings.edge_calibration_horizon_seconds or 86400)
    last_id = 0
    while True:
        rows = conn.execute(
            """
            SELECT id, symbol, scan_time, raw_json
            FROM edge_training_samples
            WHERE id > ?
              AND (
                  scan_id IS NULL
                  OR label_horizon_key IS NULL
                  OR market_segment IS NULL
                  OR sector IS NULL
                  OR theme IS NULL
              )
            ORDER BY id ASC
            LIMIT 200
            """,
            (last_id,),
        ).fetchall()
        if not rows:
            break
        last_id = int(rows[-1]["id"])
        updates = []
        for row in rows:
            raw = _parse_json(row["raw_json"], {})
            if not isinstance(raw, dict):
                raw = {}
            candidate = raw.get("candidate") if isinstance(raw.get("candidate"), dict) else {}
            scan_id = raw.get("scan_id") or raw.get("scan_run_id") or candidate.get("scan_id")
            market_segment = (
                candidate.get("market_segment")
                or candidate.get("market")
                or raw.get("market_segment")
            )
            sector = (
                candidate.get("sector")
                or candidate.get("industry")
                or candidate.get("sector_name")
                or raw.get("sector")
            )
            theme = (
                candidate.get("theme")
                or candidate.get("strategy_theme")
                or candidate.get("universe_profile")
                or raw.get("theme")
            )
            updates.append(
                (
                    scan_id,
                    _label_horizon_key(
                        symbol=str(row["symbol"]),
                        scan_time=str(row["scan_time"]),
                        horizon_seconds=horizon_seconds,
                    ),
                    market_segment,
                    sector,
                    theme,
                    row["id"],
                )
            )
        conn.executemany(
            """
            UPDATE edge_training_samples
            SET scan_id = COALESCE(scan_id, ?),
                label_horizon_key = COALESCE(label_horizon_key, ?),
                market_segment = COALESCE(market_segment, ?),
                sector = COALESCE(sector, ?),
                theme = COALESCE(theme, ?)
            WHERE id = ?
            """,
            updates,
        )


def _purge_invalid_label_samples_by_path(path: Path) -> int:
    if not path.exists():
        return 0
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
        _ensure_schema_migrations(conn)
        deleted = _purge_invalid_label_samples(conn)
        conn.commit()
        return deleted


def _purge_invalid_label_samples(conn: sqlite3.Connection) -> int:
    if not settings.edge_calibration_label_at_horizon_end:
        return 0
    min_span = max(60, int(settings.edge_calibration_horizon_seconds or 86400))
    cursor = conn.execute(
        """
        DELETE FROM edge_training_samples
        WHERE label_observation_span_seconds IS NOT NULL
          AND label_observation_span_seconds < ?
        """,
        (min_span,),
    )
    return int(cursor.rowcount or 0)


def _label_span_from_raw_json(raw_json: str | None) -> int | None:
    raw = _parse_json(raw_json, {})
    if not isinstance(raw, dict):
        return None
    span = _to_int(raw.get("label_observation_span_seconds"))
    if span is not None:
        return span
    candidate = raw.get("candidate")
    if isinstance(candidate, dict):
        return _to_int(candidate.get("label_observation_span_seconds"))
    return None


def _sample_metrics_from_raw_json(raw_json: str | None) -> dict[str, float | None]:
    raw = _parse_json(raw_json, {})
    if not isinstance(raw, dict):
        raw = {}
    metrics = raw.get("sample_metrics")
    candidate = raw.get("candidate")
    if not isinstance(metrics, dict):
        metrics = {}
    if not isinstance(candidate, dict):
        candidate = raw
    return {
        "raw_score": _to_float(metrics.get("raw_score") or candidate.get("raw_score")),
        "expected_return_bps": _to_float(
            metrics.get("expected_return_bps")
            or candidate.get("expected_return")
            or candidate.get("expected_return_bps")
        ),
        "expected_risk_bps": _to_float(
            metrics.get("expected_risk_bps")
            or candidate.get("expected_risk")
            or candidate.get("expected_risk_bps")
        ),
        "trading_cost_bps": _to_float(
            metrics.get("trading_cost_bps")
            or candidate.get("trading_cost")
            or candidate.get("trading_cost_bps")
        ),
        "slippage_cost_bps": _to_float(
            metrics.get("slippage_cost_bps")
            or candidate.get("slippage_cost")
            or candidate.get("slippage_cost_bps")
        ),
        "net_edge_bps": _to_float(
            metrics.get("net_edge_bps")
            or candidate.get("net_edge")
            or candidate.get("net_edge_bps")
        ),
        "composite_score": _to_float(
            metrics.get("composite_score")
            or candidate.get("composite_score")
            or candidate.get("score")
        ),
    }


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
    return settings.storage_path(db_path or settings.edge_calibration_db_path)
