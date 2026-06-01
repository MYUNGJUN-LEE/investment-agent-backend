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

# TODO: Add PBO after at least 3000 labeled samples and multiple parameter sets are available.


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_json(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value or "")
    except Exception:
        return default


def _parse_dt(value: Any) -> datetime | None:
    try:
        raw = str(value or "").strip()
        if not raw:
            return None
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def _parse_int_list(raw: Any, default: list[int]) -> list[int]:
    values: list[int] = []
    for part in str(raw or "").split(","):
        try:
            value = int(part.strip())
        except Exception:
            continue
        if value > 0:
            values.append(value)
    return values or default


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


def _pearson_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None

    mean_x = _mean(xs)
    mean_y = _mean(ys)

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)

    if var_x <= 0 or var_y <= 0:
        return None

    return cov / ((var_x ** 0.5) * (var_y ** 0.5))


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


def _realized_net_edge(row: sqlite3.Row | dict[str, Any]) -> float:
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


def _sample_group_fields(raw_json: Any) -> dict[str, Any]:
    payload = _parse_json(raw_json, {})
    if not isinstance(payload, dict):
        payload = {}

    candidate = payload.get("candidate") or payload
    if not isinstance(candidate, dict):
        candidate = {}

    market_context = (
        candidate.get("market_context")
        or candidate.get("market")
        or {}
    )
    if not isinstance(market_context, dict):
        market_context = {}

    market_segment = (
        candidate.get("market_segment")
        or candidate.get("market")
        or candidate.get("market_type")
        or "unknown"
    )

    universe_profile = candidate.get("universe_profile") or "unknown"

    sector = (
        candidate.get("sector")
        or candidate.get("industry")
        or candidate.get("theme")
        or candidate.get("market_segment")
        or "unknown"
    )

    market_regime = (
        market_context.get("market_regime")
        or market_context.get("regime")
        or candidate.get("market_regime")
        or candidate.get("regime")
        or "unknown"
    )

    return {
        "market_segment": str(market_segment or "unknown"),
        "universe_profile": str(universe_profile or "unknown"),
        "sector": str(sector or "unknown"),
        "market_regime": str(market_regime or "unknown"),
    }


def _load_samples(calibration_db_path: Path | str | None = None) -> list[dict[str, Any]]:
    path = settings.storage_path(calibration_db_path or settings.edge_calibration_db_path)
    if not path.exists():
        return []

    limit = max(100, min(10_000, int(settings.quant_validation_max_samples or 3000)))

    try:
        conn = sqlite3.connect(path)
        try:
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    f"""
                    SELECT
                        symbol,
                        observed_at,
                        net_edge_bps,
                        realized_return_bps,
                        realized_risk_bps,
                        trading_cost_bps,
                        slippage_cost_bps,
                        rank,
                        composite_score,
                        raw_json,
                        status
                    FROM edge_training_samples
                    WHERE net_edge_bps IS NOT NULL
                      AND {ELIGIBLE_EDGE_SAMPLE_STATUS_SQL}
                    ORDER BY observed_at ASC, id ASC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            except sqlite3.Error:
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
        data = dict(row)
        realized_net = _realized_net_edge(data)
        group_fields = _sample_group_fields(data.get("raw_json"))
        samples.append(
            {
                "symbol": data.get("symbol"),
                "observed_at": data.get("observed_at"),
                "predicted_net_edge_bps": _to_float(data.get("net_edge_bps")) or 0.0,
                "realized_net_edge_bps": realized_net,
                "rank": data.get("rank"),
                "composite_score": data.get("composite_score"),
                "raw_json": data.get("raw_json"),
                "status": data.get("status"),
                **group_fields,
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


def purged_walk_forward_validation(
    *,
    calibration_db_path: Path | str | None = None,
) -> dict[str, Any]:
    try:
        samples = _load_samples(calibration_db_path)
        min_samples = int(settings.quant_validation_min_samples or 1000)

        if len(samples) < min_samples:
            return {
                "status": "collecting",
                "approved": None,
                "sample_count": len(samples),
                "message": f"Need at least {min_samples} samples for purged walk-forward validation",
            }

        samples = sorted(samples, key=lambda item: str(item.get("observed_at") or ""))
        n_folds = max(3, int(settings.quant_validation_folds or 5))
        top_n = max(1, int(settings.quant_validation_top_n or 10))
        embargo_seconds = max(0, int(settings.quant_validation_embargo_seconds or 0))

        fold_size = len(samples) // (n_folds + 1)
        folds: list[dict[str, Any]] = []

        for fold_idx in range(n_folds):
            train_end = fold_size * (fold_idx + 1)
            test_start = train_end
            test_end = min(len(samples), test_start + fold_size)

            test = samples[test_start:test_end]
            if len(test) < top_n:
                continue

            test_start_dt = _parse_dt(test[0].get("observed_at"))
            if test_start_dt is None:
                purged_train = samples[:train_end]
            else:
                cutoff_dt = test_start_dt - timedelta(seconds=embargo_seconds)
                purged_train = [
                    item
                    for item in samples[:train_end]
                    if (_parse_dt(item.get("observed_at")) or test_start_dt) <= cutoff_dt
                ]

            if not purged_train:
                continue

            ranked_test = sorted(
                test,
                key=lambda item: float(item.get("predicted_net_edge_bps") or 0.0),
                reverse=True,
            )

            top = ranked_test[:top_n]
            realized = [float(item.get("realized_net_edge_bps") or 0.0) for item in top]
            predicted = [float(item.get("predicted_net_edge_bps") or 0.0) for item in top]

            wins = sum(1 for x in realized if x > 0)
            sharpe = _sharpe(realized)
            ic = _pearson_corr(predicted, realized)

            folds.append(
                {
                    "fold": fold_idx + 1,
                    "train_count_before_purge": train_end,
                    "train_count_after_purge": len(purged_train),
                    "purged_count": train_end - len(purged_train),
                    "test_count": len(test),
                    "top_count": len(top),
                    "avg_realized_net_edge_bps": round(_mean(realized), 4),
                    "median_realized_net_edge_bps": round(median(realized), 4) if realized else None,
                    "win_rate": round(wins / len(realized), 6) if realized else None,
                    "ic": round(ic, 6) if ic is not None else None,
                    "sharpe": round(sharpe, 6) if sharpe is not None else None,
                }
            )

        if not folds:
            return {
                "status": "insufficient",
                "approved": None,
                "sample_count": len(samples),
                "message": "No valid purged walk-forward folds",
            }

        fold_edges = [float(fold["avg_realized_net_edge_bps"]) for fold in folds]
        fold_ics = [
            float(fold["ic"])
            for fold in folds
            if fold.get("ic") is not None
        ]

        positive_fold_rate = sum(1 for x in fold_edges if x > 0) / len(fold_edges)
        median_ic = median(fold_ics) if fold_ics else None

        approved = (
            median(fold_edges) > float(settings.quant_validation_min_median_net_edge_bps or 0.0)
            and positive_fold_rate >= float(settings.quant_validation_min_positive_fold_rate or 0.60)
            and min(fold_edges) > -50.0
            and (median_ic is None or median_ic >= 0.0)
        )

        return {
            "status": "ready",
            "approved": approved,
            "sample_count": len(samples),
            "fold_count": len(folds),
            "embargo_seconds": embargo_seconds,
            "folds": folds,
            "median_fold_avg_net_edge_bps": round(median(fold_edges), 4),
            "worst_fold_avg_net_edge_bps": round(min(fold_edges), 4),
            "positive_fold_rate": round(positive_fold_rate, 6),
            "median_fold_ic": round(median_ic, 6) if median_ic is not None else None,
        }
    except Exception as exc:
        return {
            "status": "error",
            "approved": None,
            "message": str(exc),
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


def rolling_ic_stability_report(
    *,
    calibration_db_path: Path | str | None = None,
) -> dict[str, Any]:
    try:
        samples = _load_samples(calibration_db_path)
        if not samples:
            return {"status": "empty", "sample_count": 0, "windows": []}

        samples = sorted(
            samples,
            key=lambda item: str(item.get("observed_at") or ""),
            reverse=True,
        )

        windows = _parse_int_list(
            settings.quant_validation_rolling_ic_windows,
            [100, 300, 500],
        )

        results: list[dict[str, Any]] = []

        for window in windows:
            subset = samples[:window]
            predicted = [
                float(item.get("predicted_net_edge_bps") or 0.0)
                for item in subset
            ]
            realized = [
                float(item.get("realized_net_edge_bps") or 0.0)
                for item in subset
            ]
            ic = _pearson_corr(predicted, realized)

            results.append(
                {
                    "window": window,
                    "sample_count": len(subset),
                    "ic": round(ic, 6) if ic is not None else None,
                    "status": (
                        "ready"
                        if len(subset) >= min(window, 30) and ic is not None
                        else "collecting"
                    ),
                }
            )

        ready_ics = [
            float(item["ic"])
            for item in results
            if item.get("ic") is not None
        ]

        ic_100 = next(
            (item.get("ic") for item in results if int(item.get("window") or 0) == 100),
            None,
        )

        largest_ic = None
        for item in reversed(results):
            if item.get("ic") is not None:
                largest_ic = item.get("ic")
                break

        warnings: list[str] = []

        if ic_100 is not None and ic_100 < 0:
            warnings.append("recent_100_ic_negative")

        if (
            ic_100 is not None
            and largest_ic is not None
            and float(ic_100) < float(largest_ic) - 0.05
        ):
            warnings.append("recent_ic_deteriorating")

        approved = None
        if ready_ics:
            approved = all(ic >= -0.02 for ic in ready_ics)

        return {
            "status": "ready",
            "approved": approved,
            "sample_count": len(samples),
            "windows": results,
            "warnings": warnings,
            "message": "Rolling IC is stable" if not warnings else "; ".join(warnings),
        }
    except Exception as exc:
        return {
            "status": "error",
            "approved": None,
            "message": str(exc),
            "windows": [],
        }


def _group_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    realized = [
        float(item.get("realized_net_edge_bps") or 0.0)
        for item in samples
    ]
    predicted = [
        float(item.get("predicted_net_edge_bps") or 0.0)
        for item in samples
    ]

    wins = sum(1 for x in realized if x > 0)
    ic = _pearson_corr(predicted, realized)
    sharpe = _sharpe(realized)

    return {
        "sample_count": len(samples),
        "avg_realized_net_edge_bps": round(_mean(realized), 4),
        "median_realized_net_edge_bps": round(median(realized), 4) if realized else None,
        "win_rate": round(wins / len(realized), 6) if realized else None,
        "ic": round(ic, 6) if ic is not None else None,
        "sharpe": round(sharpe, 6) if sharpe is not None else None,
        "avg_predicted_net_edge_bps": round(_mean(predicted), 4),
    }


def group_oos_breakdown_report(
    *,
    calibration_db_path: Path | str | None = None,
) -> dict[str, Any]:
    try:
        samples = _load_samples(calibration_db_path)
        min_group = int(settings.quant_validation_min_group_samples or 30)
        group_limit = int(settings.quant_validation_group_limit or 50)

        if len(samples) < min_group:
            return {
                "status": "collecting",
                "sample_count": len(samples),
                "message": f"Need at least {min_group} samples for group OOS breakdown",
            }

        dimensions = [
            "market_segment",
            "universe_profile",
            "sector",
            "market_regime",
        ]

        output: dict[str, list[dict[str, Any]]] = {}

        for dimension in dimensions:
            groups: dict[str, list[dict[str, Any]]] = {}

            for sample in samples:
                key = str(sample.get(dimension) or "unknown").strip() or "unknown"
                groups.setdefault(key, []).append(sample)

            rows: list[dict[str, Any]] = []

            for key, group_samples in groups.items():
                if len(group_samples) < min_group:
                    continue

                metrics = _group_metrics(group_samples)
                rows.append(
                    {
                        "group": key,
                        **metrics,
                    }
                )

            rows.sort(
                key=lambda item: (
                    int(item.get("sample_count") or 0),
                    float(item.get("avg_realized_net_edge_bps") or 0.0),
                ),
                reverse=True,
            )

            output[dimension] = rows[:group_limit]

        warnings: list[str] = []

        for dimension, rows in output.items():
            for row in rows:
                if (
                    int(row.get("sample_count") or 0) >= min_group
                    and float(row.get("avg_realized_net_edge_bps") or 0.0) < 0
                ):
                    warnings.append(
                        f"{dimension}:{row.get('group')} avg_realized_net_edge_bps negative"
                    )

        return {
            "status": "ready",
            "sample_count": len(samples),
            "min_group_samples": min_group,
            "dimensions": output,
            "warnings": warnings[:50],
        }
    except Exception as exc:
        return {
            "status": "error",
            "sample_count": 0,
            "message": str(exc),
            "dimensions": {},
            "warnings": [],
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

    try:
        walk_forward = walk_forward_validation(calibration_db_path=calibration_db_path)
    except Exception as exc:
        walk_forward = {"status": "error", "approved": None, "message": str(exc)}

    try:
        dsr = deflated_sharpe_report(calibration_db_path=calibration_db_path)
    except Exception as exc:
        dsr = {"status": "error", "approved": None, "message": str(exc)}

    purged_wf = (
        purged_walk_forward_validation(calibration_db_path=calibration_db_path)
        if bool(settings.quant_validation_include_purged_walk_forward)
        else {"status": "disabled"}
    )

    rolling_ic = (
        rolling_ic_stability_report(calibration_db_path=calibration_db_path)
        if bool(settings.quant_validation_include_rolling_ic)
        else {"status": "disabled"}
    )

    group_oos = (
        group_oos_breakdown_report(calibration_db_path=calibration_db_path)
        if bool(settings.quant_validation_include_group_oos)
        else {"status": "disabled"}
    )

    approved_values = [
        item.get("approved")
        for item in (walk_forward, dsr)
        if isinstance(item, dict) and item.get("approved") is not None
    ]

    approved = all(approved_values) if approved_values else None

    return {
        "status": "ready",
        "enabled": True,
        "approved": approved,
        "walk_forward": walk_forward,
        "purged_walk_forward": purged_wf,
        "deflated_sharpe": dsr,
        "rolling_ic_stability": rolling_ic,
        "group_oos_breakdown": group_oos,
    }
