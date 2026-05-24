from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import json
import sqlite3
from typing import Any, Iterator

from app.config import settings


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
"""


def initialize_edge_calibration_db(db_path: Path | str | None = None) -> None:
    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)


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

    dimension = len(FEATURE_NAMES)
    return_matrix = _ridge_matrix(dimension, ridge_lambda)
    risk_matrix = _ridge_matrix(dimension, ridge_lambda)
    return_vector = [0.0] * dimension
    risk_vector = [0.0] * dimension
    sample_count = 0
    skipped_count = 0

    for sample in _iter_calibration_samples(
        universe_path=universe_path,
        max_samples=max_samples,
        horizon_seconds=horizon_seconds,
    ):
        if sample is None:
            skipped_count += 1
            continue
        features, realized_return_bps, realized_risk_bps = sample
        _accumulate(return_matrix, return_vector, features, realized_return_bps)
        _accumulate(risk_matrix, risk_vector, features, realized_risk_bps)
        sample_count += 1

    if sample_count < min_samples:
        result = {
            "status": "skipped",
            "message": f"Not enough calibration samples: {sample_count}/{min_samples}",
            "sample_count": sample_count,
            "skipped_count": skipped_count,
            "horizon_seconds": horizon_seconds,
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
    metrics = _evaluate_model(
        universe_path=universe_path,
        max_samples=max_samples,
        horizon_seconds=horizon_seconds,
        return_coefficients=blended_return,
        risk_coefficients=blended_risk,
    )
    result = {
        "status": "success",
        "sample_count": sample_count,
        "skipped_count": skipped_count,
        "horizon_seconds": horizon_seconds,
        "mae_return_bps": metrics["mae_return_bps"],
        "mae_risk_bps": metrics["mae_risk_bps"],
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
    return {
        "status": "ready" if coefficient_count else "empty",
        "enabled": settings.edge_calibration_enabled,
        "coefficient_count": int(coefficient_count or 0),
        "last_run": dict(run) if run else None,
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
    expected_return = _predict(return_coefficients, feature_map)
    expected_risk = _predict(risk_coefficients, feature_map)
    return {
        "expected_return": round(max(0.0, min(500.0, expected_return)), 4),
        "expected_risk": round(max(0.0, min(500.0, expected_risk)), 4),
        "edge_model": "calibrated_ridge_v1",
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
    return (
        [feature_map[name] for name in FEATURE_NAMES],
        realized_return_bps,
        realized_risk_bps,
    )


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
        cursor = conn.execute(
            """
            INSERT INTO edge_calibration_runs (
                created_at, status, sample_count, skipped_count, horizon_seconds,
                mae_return_bps, mae_risk_bps, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                result.get("status", "unknown"),
                int(result.get("sample_count") or 0),
                int(result.get("skipped_count") or 0),
                int(result.get("horizon_seconds") or settings.edge_calibration_horizon_seconds),
                result.get("mae_return_bps"),
                result.get("mae_risk_bps"),
                _json(result),
            ),
        )
        return int(cursor.lastrowid)


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
