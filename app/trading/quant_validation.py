from __future__ import annotations

from pathlib import Path
from statistics import median
import math
import sqlite3
from typing import Any

from app.config import settings
from app.trading.edge_calibration import ELIGIBLE_EDGE_SAMPLE_STATUS_SQL


REALIZED_RISK_WEIGHT = 0.10

# TODO: Add PBO after at least 3000 labeled samples and multiple parameter sets are available.


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _sharpe(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    sd = _std(xs)
    if sd <= 0:
        return None
    return _mean(xs) / sd


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _sample_skew(xs: list[float]) -> float:
    if len(xs) < 3:
        return 0.0
    mu = _mean(xs)
    sd = _std(xs)
    if sd <= 0:
        return 0.0
    return sum(((x - mu) / sd) ** 3 for x in xs) / len(xs)


def _sample_kurtosis(xs: list[float]) -> float:
    if len(xs) < 4:
        return 3.0
    mu = _mean(xs)
    sd = _std(xs)
    if sd <= 0:
        return 3.0
    return sum(((x - mu) / sd) ** 4 for x in xs) / len(xs)


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


def _load_samples(calibration_db_path: Path | str | None = None) -> list[dict[str, Any]]:
    path = settings.storage_path(calibration_db_path or settings.edge_calibration_db_path)
    if not path.exists():
        return []

    limit = max(100, min(10_000, int(settings.quant_validation_max_samples or 3000)))

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
        realized_net = _realized_net_edge(row)
        samples.append(
            {
                "observed_at": row["observed_at"],
                "predicted_net_edge_bps": _to_float(row["net_edge_bps"]) or 0.0,
                "realized_net_edge_bps": realized_net,
                "rank": row["rank"],
                "composite_score": row["composite_score"],
            }
        )

    return samples


def walk_forward_validation(
    *,
    calibration_db_path: Path | str | None = None,
) -> dict[str, Any]:
    samples = _load_samples(calibration_db_path)
    min_samples = int(settings.quant_validation_min_samples or 1000)

    if len(samples) < min_samples:
        return {
            "status": "collecting",
            "approved": None,
            "sample_count": len(samples),
            "message": f"Need at least {min_samples} samples for walk-forward validation",
        }

    n_folds = max(3, int(settings.quant_validation_folds or 5))
    top_n = max(1, int(settings.quant_validation_top_n or 10))

    fold_size = len(samples) // (n_folds + 1)
    folds: list[dict[str, Any]] = []

    for fold_idx in range(n_folds):
        train_end = fold_size * (fold_idx + 1)
        test_start = train_end
        test_end = min(len(samples), test_start + fold_size)

        test = samples[test_start:test_end]
        if len(test) < top_n:
            continue

        ranked_test = sorted(
            test,
            key=lambda item: float(item.get("predicted_net_edge_bps") or 0.0),
            reverse=True,
        )

        top = ranked_test[:top_n]
        realized = [float(item["realized_net_edge_bps"]) for item in top]
        wins = sum(1 for x in realized if x > 0)
        sharpe = _sharpe(realized)

        folds.append(
            {
                "fold": fold_idx + 1,
                "train_count": train_end,
                "test_count": len(test),
                "top_count": len(top),
                "avg_realized_net_edge_bps": round(_mean(realized), 4),
                "win_rate": round(wins / len(realized), 6) if realized else None,
                "sharpe": round(sharpe, 6) if sharpe is not None else None,
            }
        )

    if not folds:
        return {
            "status": "insufficient",
            "approved": None,
            "sample_count": len(samples),
            "message": "No valid walk-forward folds",
        }

    fold_edges = [float(fold["avg_realized_net_edge_bps"]) for fold in folds]
    positive_fold_rate = sum(1 for x in fold_edges if x > 0) / len(fold_edges)

    min_positive_rate = float(settings.quant_validation_min_positive_fold_rate or 0.60)
    min_median_edge = float(settings.quant_validation_min_median_net_edge_bps or 0.0)

    approved = (
        median(fold_edges) > min_median_edge
        and positive_fold_rate >= min_positive_rate
        and min(fold_edges) > -50.0
    )

    return {
        "status": "ready",
        "approved": approved,
        "sample_count": len(samples),
        "fold_count": len(folds),
        "folds": folds,
        "median_fold_avg_net_edge_bps": round(median(fold_edges), 4),
        "worst_fold_avg_net_edge_bps": round(min(fold_edges), 4),
        "positive_fold_rate": round(positive_fold_rate, 6),
    }


def deflated_sharpe_report(
    *,
    calibration_db_path: Path | str | None = None,
) -> dict[str, Any]:
    samples = _load_samples(calibration_db_path)
    returns = [float(item["realized_net_edge_bps"]) for item in samples]
    min_samples = int(settings.quant_validation_min_samples or 1000)

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

    sr = _mean(returns) / sd
    skew = _sample_skew(returns)
    kurt = _sample_kurtosis(returns)
    t = len(returns)

    var_sr = (
        1.0
        - skew * sr
        + ((kurt - 1.0) / 4.0) * (sr ** 2)
    ) / max(1, t - 1)

    var_sr = max(var_sr, 1e-12)

    num_trials = max(2, int(settings.quant_validation_num_trials or 20))
    sr_star = (math.log(num_trials) ** 0.5) * (var_sr ** 0.5)

    z = (sr - sr_star) / (var_sr ** 0.5)
    probability = _normal_cdf(z)

    return {
        "status": "ready",
        "approved": probability >= 0.80,
        "sample_count": t,
        "sharpe": round(sr, 6),
        "skew": round(skew, 6),
        "kurtosis": round(kurt, 6),
        "num_trials": num_trials,
        "sr_star": round(sr_star, 6),
        "deflated_sharpe_probability": round(probability, 6),
    }


def quant_validation_summary(
    *,
    calibration_db_path: Path | str | None = None,
) -> dict[str, Any]:
    if not bool(settings.quant_validation_enabled):
        return {
            "status": "disabled",
            "enabled": False,
        }

    walk_forward = walk_forward_validation(calibration_db_path=calibration_db_path)
    dsr = deflated_sharpe_report(calibration_db_path=calibration_db_path)

    approved_values = [
        item.get("approved")
        for item in (walk_forward, dsr)
        if item.get("approved") is not None
    ]

    approved = all(approved_values) if approved_values else None

    return {
        "status": "ready",
        "enabled": True,
        "approved": approved,
        "walk_forward": walk_forward,
        "deflated_sharpe": dsr,
    }
