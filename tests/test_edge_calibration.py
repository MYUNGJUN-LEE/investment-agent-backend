from __future__ import annotations

import sqlite3

from app.trading import edge_calibration
from app.trading import universe_scanner


def test_edge_calibration_fits_coefficients_from_scanner_history(tmp_path, monkeypatch):
    universe_db = tmp_path / "universe.sqlite3"
    calibration_db = tmp_path / "edge.sqlite3"
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_min_samples", 2)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_max_samples", 10)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_horizon_seconds", 3600)

    universe_scanner.initialize_universe_db(universe_db)
    with sqlite3.connect(universe_db) as conn:
        _insert_candidate(
            conn,
            scan_id="scan-a",
            scan_time="2026-05-24T09:00:00",
            symbol="005930",
            raw_score=80,
            current_price=100,
        )
        _insert_candidate(
            conn,
            scan_id="scan-b",
            scan_time="2026-05-24T09:00:00",
            symbol="000660",
            raw_score=35,
            current_price=100,
        )
        _insert_price(conn, scan_id="future-a", created_at="2026-05-24T10:00:00", symbol="005930", price=110)
        _insert_price(conn, scan_id="future-b", created_at="2026-05-24T10:00:00", symbol="000660", price=92)

    result = edge_calibration.calibrate_edge_model(
        universe_db_path=universe_db,
        calibration_db_path=calibration_db,
        min_samples=2,
        max_samples=10,
    )
    status = edge_calibration.get_edge_calibration_status(
        calibration_db_path=calibration_db,
    )
    model = edge_calibration.load_edge_model(
        calibration_db_path=calibration_db,
    )
    estimate = edge_calibration.estimate_expected_edges(
        {"change_rate": 2, "volume_ratio": 2, "turnover_value": 50_000_000_000},
        80,
        model=model,
    )

    assert result["status"] == "success"
    assert result["sample_count"] == 2
    assert status["coefficient_count"] == len(edge_calibration.FEATURE_NAMES) * 2
    assert "expected_return" in model
    assert estimate is not None
    assert estimate["edge_model"] == "calibrated_ridge_v1"


def test_refresh_edge_training_samples_stores_prediction_labels(tmp_path, monkeypatch):
    universe_db = tmp_path / "universe.sqlite3"
    calibration_db = tmp_path / "edge.sqlite3"
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_horizon_seconds", 3600)

    universe_scanner.initialize_universe_db(universe_db)
    with sqlite3.connect(universe_db) as conn:
        _insert_candidate(
            conn,
            scan_id="scan-a",
            scan_time="2026-05-24T09:00:00",
            symbol="005930",
            raw_score=80,
            current_price=100,
        )
        _insert_price(
            conn,
            scan_id="future-a",
            created_at="2026-05-24T10:00:00",
            symbol="005930",
            price=108,
        )

    result = edge_calibration.refresh_edge_training_samples(
        universe_db_path=universe_db,
        calibration_db_path=calibration_db,
        horizon_seconds=3600,
        candidate_limit=10,
    )

    assert result["status"] == "success"
    assert result["inserted_count"] == 1
    assert result["stored_sample_count"] == 1


def test_edge_calibration_if_due_skips_when_recent(tmp_path, monkeypatch):
    universe_db = tmp_path / "universe.sqlite3"
    calibration_db = tmp_path / "edge.sqlite3"
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_interval_seconds", 3600)

    edge_calibration.initialize_edge_calibration_db(calibration_db)
    edge_calibration.calibrate_edge_model_if_due(
        universe_db_path=universe_db,
        calibration_db_path=calibration_db,
    )
    result = edge_calibration.calibrate_edge_model_if_due(
        universe_db_path=universe_db,
        calibration_db_path=calibration_db,
    )

    assert result["status"] == "not_due"


def test_edge_entry_gate_passes_after_oos_and_top10_thresholds(tmp_path, monkeypatch):
    universe_db = tmp_path / "universe.sqlite3"
    calibration_db = tmp_path / "edge.sqlite3"
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_min_samples", 5)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_max_samples", 10)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_target_samples", 5)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_horizon_seconds", 3600)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_samples", 5)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_oos_samples", 1)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_max_mae_return_bps", 10_000)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_max_mae_risk_bps", 10_000)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_top10_avg_return_bps", 0)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_top10_win_rate", 0.0)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_fill_adjusted_edge_bps", 30)

    universe_scanner.initialize_universe_db(universe_db)
    with sqlite3.connect(universe_db) as conn:
        for index in range(5):
            symbol = f"{index + 1:06d}"
            _insert_candidate(
                conn,
                scan_id=f"scan-{index}",
                scan_time=f"2026-05-24T09:0{index}:00",
                symbol=symbol,
                raw_score=70 + index,
                current_price=100,
            )
            _insert_price(
                conn,
                scan_id=f"future-{index}",
                created_at=f"2026-05-24T10:0{index}:00",
                symbol=symbol,
                price=105 + index,
            )

    result = edge_calibration.calibrate_edge_model(
        universe_db_path=universe_db,
        calibration_db_path=calibration_db,
        min_samples=5,
        max_samples=10,
    )
    gate = edge_calibration.edge_entry_gate(
        [{"symbol": "000001", "net_edge": 80}],
        calibration_db_path=calibration_db,
    )

    assert result["status"] == "success"
    assert result["stored_sample_count"] == 5
    assert result["oos_sample_count"] >= 1
    assert gate["approved"] is True
    assert gate["top10_performance"]["sample_count"] == 5


def _insert_candidate(
    conn: sqlite3.Connection,
    *,
    scan_id: str,
    scan_time: str,
    symbol: str,
    raw_score: float,
    current_price: float,
) -> None:
    conn.execute(
        """
        INSERT INTO scanner_candidate_history (
            scan_id, scan_time, symbol, name, raw_score, expected_return,
            expected_risk, trading_cost, slippage_cost, net_edge,
            composite_score, rank, reason, status, decision, current_price,
            change_rate, volume, volume_ratio, turnover_value, news_count,
            disclosure_count, claimed_by_worker, expires_at, raw_json
        )
        VALUES (?, ?, ?, ?, ?, 0, 0, 0, 0, 0, ?, 1, 'test', 'READY',
                'buy_candidate', ?, 2, 1000000, 2, 50000000000, 1, 0,
                NULL, '2026-05-24T10:00:00', ?)
        """,
        (
            scan_id,
            scan_time,
            symbol,
            symbol,
            raw_score,
            raw_score,
            current_price,
            '{"intraday": {"minute_volume_ratio": 2.0}}',
        ),
    )


def _insert_price(
    conn: sqlite3.Connection,
    *,
    scan_id: str,
    created_at: str,
    symbol: str,
    price: float,
) -> None:
    conn.execute(
        """
        INSERT INTO universe_price_snapshots (
            scan_id, created_at, symbol, name, current_price, raw_json
        )
        VALUES (?, ?, ?, ?, ?, '{}')
        """,
        (scan_id, created_at, symbol, symbol, price),
    )
