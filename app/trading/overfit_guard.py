from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
import json
import math
import sqlite3
from typing import Any

from app.config import settings
from app.trading.edge_calibration import ELIGIBLE_EDGE_SAMPLE_STATUS_SQL


REALIZED_RISK_WEIGHT = 0.10


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS overfit_guard_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    approved INTEGER,
    sample_count INTEGER NOT NULL,
    fold_count INTEGER,
    positive_fold_rate REAL,
    median_oos_net_edge_bps REAL,
    worst_fold_net_edge_bps REAL,
    deflated_sharpe_probability REAL,
    raw_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_overfit_guard_reports_time
ON overfit_guard_reports(created_at);
"""


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _db_path(db_path: Path | str | None = None) -> Path:
    return settings.storage_path(db_path or settings.overfit_guard_db_path)


def initialize_overfit_guard_db(db_path: Path | str | None = None) -> None:
    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return (sum((value - mu) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _sharpe(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    sd = _std(values)
    if sd <= 0:
        return None
    return _mean(values) / sd


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _sample_skew(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    mu = _mean(values)
    sd = _std(values)
    if sd <= 0:
        return 0.0
    return sum(((value - mu) / sd) ** 3 for value in values) / len(values)


def _sample_kurtosis(values: list[float]) -> float:
    if len(values) < 4:
        return 3.0
    mu = _mean(values)
    sd = _std(values)
    if sd <= 0:
        return 3.0
    return sum(((value - mu) / sd) ** 4 for value in values) / len(values)


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


def _realized_net_edge(row: sqlite3.Row) -> float:
    realized_return = _to_float(row["realized_return_bps"]) or 0.0
    realized_risk = _to_float(row["realized_risk_bps"]) or 0.0
    trading_cost = _to_float(row["trading_cost_bps"]) or 0.0
    slippage_cost = _to_float(row["slippage_cost_bps"]) or 0.0

    return (
        realized_return
        - realized_risk * REALIZED_RISK_WEIGHT
        - trading_cost
        - slippage_cost
    )


def load_limited_edge_samples(
    *,
    calibration_db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    path = settings.storage_path(calibration_db_path or settings.edge_calibration_db_path)
    if not path.exists():
        return []

    limit = max(100, min(10_000, int(settings.overfit_guard_max_samples or 3000)))

    try:
        conn = sqlite3.connect(path)
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT
                    observed_at,
                    net_edge_bps,
                    realized_return_bps,
                    realized_risk_bps,
                    trading_cost_bps,
                    slippage_cost_bps,
                    rank,
                    composite_score
                FROM edge_training_samples
                WHERE net_edge_bps IS NOT NULL
                  AND {ELIGIBLE_EDGE_SAMPLE_STATUS_SQL}
                ORDER BY observed_at ASC, id ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        finally:
            conn.close()
    except (OSError, sqlite3.Error):
        return []

    samples: list[dict[str, Any]] = []
    for row in rows:
        observed_at = row["observed_at"]
        samples.append(
            {
                "observed_at": observed_at,
                "observed_ts": _parse_time(observed_at),
                "predicted_net_edge_bps": _to_float(row["net_edge_bps"]) or 0.0,
                "realized_net_edge_bps": _realized_net_edge(row),
                "rank": row["rank"],
                "composite_score": row["composite_score"],
            }
        )

    return samples


def _purged_test_block(
    samples: list[dict[str, Any]],
    *,
    train_end: int,
    test_end: int,
    embargo_seconds: int,
) -> list[dict[str, Any]]:
    test = samples[train_end:test_end]
    if not test or embargo_seconds <= 0 or train_end <= 0:
        return test

    train_end_time = samples[train_end - 1].get("observed_ts")
    if not isinstance(train_end_time, datetime):
        return test

    embargo_until = train_end_time + timedelta(seconds=embargo_seconds)
    return [
        item
        for item in test
        if not isinstance(item.get("observed_ts"), datetime)
        or item["observed_ts"] > embargo_until
    ]


def purged_walk_forward_validation(
    *,
    calibration_db_path: Path | str | None = None,
    samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    samples = samples if samples is not None else load_limited_edge_samples(
        calibration_db_path=calibration_db_path
    )
    min_samples = int(settings.overfit_guard_min_samples or 1000)

    if len(samples) < min_samples:
        return {
            "status": "collecting",
            "approved": None,
            "sample_count": len(samples),
            "message": f"Need at least {min_samples} samples for overfit guard validation",
        }

    n_folds = max(3, int(settings.overfit_guard_folds or 5))
    top_n = max(1, int(settings.overfit_guard_top_n or 10))
    embargo_seconds = max(0, int(settings.overfit_guard_embargo_seconds or 0))
    fold_size = len(samples) // (n_folds + 1)

    if fold_size < top_n:
        return {
            "status": "insufficient",
            "approved": None,
            "sample_count": len(samples),
            "message": "Not enough samples per fold for requested top_n",
        }

    folds: list[dict[str, Any]] = []

    for fold_idx in range(n_folds):
        train_end = fold_size * (fold_idx + 1)
        test_end = min(len(samples), train_end + fold_size)
        test = _purged_test_block(
            samples,
            train_end=train_end,
            test_end=test_end,
            embargo_seconds=embargo_seconds,
        )

        if len(test) < top_n:
            continue

        ranked_test = sorted(
            test,
            key=lambda item: float(item.get("predicted_net_edge_bps") or 0.0),
            reverse=True,
        )

        top = ranked_test[:top_n]
        realized = [float(item.get("realized_net_edge_bps") or 0.0) for item in top]
        wins = sum(1 for value in realized if value > 0)
        sharpe = _sharpe(realized)

        folds.append(
            {
                "fold": fold_idx + 1,
                "train_count": train_end,
                "test_count": len(test),
                "top_count": len(top),
                "avg_oos_net_edge_bps": round(_mean(realized), 4),
                "win_rate": round(wins / len(realized), 6) if realized else None,
                "sharpe": round(sharpe, 6) if sharpe is not None else None,
                "embargo_seconds": embargo_seconds,
            }
        )

    if not folds:
        return {
            "status": "insufficient",
            "approved": None,
            "sample_count": len(samples),
            "message": "No valid purged walk-forward folds",
        }

    fold_edges = [float(fold["avg_oos_net_edge_bps"]) for fold in folds]
    positive_fold_rate = sum(1 for value in fold_edges if value > 0) / len(fold_edges)
    median_edge = median(fold_edges)
    worst_edge = min(fold_edges)

    min_positive_rate = float(settings.overfit_guard_min_positive_fold_rate or 0.60)
    min_median_edge = float(settings.overfit_guard_min_median_oos_net_edge_bps or 0.0)

    approved = (
        positive_fold_rate >= min_positive_rate
        and median_edge > min_median_edge
        and worst_edge > -50.0
    )

    return {
        "status": "ready",
        "approved": approved,
        "sample_count": len(samples),
        "fold_count": len(folds),
        "folds": folds,
        "median_oos_net_edge_bps": round(median_edge, 4),
        "worst_fold_net_edge_bps": round(worst_edge, 4),
        "positive_fold_rate": round(positive_fold_rate, 6),
        "approval_thresholds": {
            "min_positive_fold_rate": min_positive_rate,
            "min_median_oos_net_edge_bps": min_median_edge,
            "worst_fold_net_edge_floor_bps": -50.0,
        },
    }


def deflated_sharpe_report(
    *,
    calibration_db_path: Path | str | None = None,
    samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    samples = samples if samples is not None else load_limited_edge_samples(
        calibration_db_path=calibration_db_path
    )
    returns = [float(item.get("realized_net_edge_bps") or 0.0) for item in samples]
    min_samples = int(settings.overfit_guard_min_samples or 1000)

    if len(returns) < min_samples:
        return {
            "status": "collecting",
            "approved": None,
            "sample_count": len(returns),
            "message": f"Need at least {min_samples} samples for deflated Sharpe",
        }

    sd = _std(returns)
    if sd <= 0:
        return {
            "status": "invalid",
            "approved": False,
            "sample_count": len(returns),
            "message": "Zero standard deviation",
        }

    sharpe = _mean(returns) / sd
    skew = _sample_skew(returns)
    kurtosis = _sample_kurtosis(returns)
    sample_count = len(returns)

    var_sharpe = (
        1.0
        - skew * sharpe
        + ((kurtosis - 1.0) / 4.0) * (sharpe ** 2)
    ) / max(1, sample_count - 1)
    var_sharpe = max(var_sharpe, 1e-12)

    num_trials = max(2, int(settings.overfit_guard_folds or 5) * 4)
    sharpe_star = (math.log(num_trials) ** 0.5) * (var_sharpe ** 0.5)
    z_score = (sharpe - sharpe_star) / (var_sharpe ** 0.5)
    probability = _normal_cdf(z_score)

    return {
        "status": "ready",
        "approved": probability >= 0.80,
        "sample_count": sample_count,
        "sharpe": round(sharpe, 6),
        "skew": round(skew, 6),
        "kurtosis": round(kurtosis, 6),
        "num_trials": num_trials,
        "sr_star": round(sharpe_star, 6),
        "deflated_sharpe_probability": round(probability, 6),
    }


def overfit_guard_summary(
    *,
    calibration_db_path: Path | str | None = None,
    db_path: Path | str | None = None,
    execution_mode: str = "paper",
    persist: bool = True,
) -> dict[str, Any]:
    if not bool(settings.overfit_guard_enabled):
        return {
            "status": "disabled",
            "enabled": False,
            "message": "overfit guard disabled",
        }

    samples = load_limited_edge_samples(calibration_db_path=calibration_db_path)
    walk_forward = purged_walk_forward_validation(samples=samples)
    sharpe = deflated_sharpe_report(samples=samples)

    # Walk-forward approval drives guard status. Deflated Sharpe is diagnostic
    # only here, so it cannot block by itself.
    approved = walk_forward.get("approved")

    execution_mode = str(execution_mode or "paper").lower()
    live_block = (
        execution_mode != "paper"
        and bool(settings.overfit_guard_live_block_enabled)
        and approved is False
    )

    result = {
        "status": str(walk_forward.get("status") or "ready"),
        "enabled": True,
        "approved": approved,
        "live_block": live_block,
        "sample_count": len(samples),
        "max_samples": max(100, min(10_000, int(settings.overfit_guard_max_samples or 3000))),
        "execution_mode": execution_mode,
        "walk_forward": walk_forward,
        "deflated_sharpe": sharpe,
        "message": (
            "live entries would be blocked by overfit guard"
            if live_block
            else "diagnostic report only"
        ),
    }

    if persist:
        _record_overfit_guard_report(result, db_path=db_path)

    return result


def _record_overfit_guard_report(
    result: dict[str, Any],
    *,
    db_path: Path | str | None = None,
) -> None:
    try:
        initialize_overfit_guard_db(db_path)
        path = _db_path(db_path)
        walk_forward = result.get("walk_forward") or {}
        sharpe = result.get("deflated_sharpe") or {}

        with sqlite3.connect(path) as conn:
            conn.execute(
                """
                INSERT INTO overfit_guard_reports (
                    created_at, status, approved, sample_count, fold_count,
                    positive_fold_rate, median_oos_net_edge_bps,
                    worst_fold_net_edge_bps, deflated_sharpe_probability,
                    raw_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _now(),
                    str(result.get("status") or ""),
                    None
                    if result.get("approved") is None
                    else int(bool(result.get("approved"))),
                    int(result.get("sample_count") or 0),
                    walk_forward.get("fold_count"),
                    walk_forward.get("positive_fold_rate"),
                    walk_forward.get("median_oos_net_edge_bps"),
                    walk_forward.get("worst_fold_net_edge_bps"),
                    sharpe.get("deflated_sharpe_probability"),
                    json.dumps(result, ensure_ascii=False),
                ),
            )
            _prune_old_reports(conn)
    except (OSError, sqlite3.Error):
        return


def _prune_old_reports(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        DELETE FROM overfit_guard_reports
        WHERE id NOT IN (
            SELECT id
            FROM overfit_guard_reports
            ORDER BY created_at DESC, id DESC
            LIMIT 100
        )
        """
    )
