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
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_min_label_age_seconds", 0)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_min_future_snapshots", 1)

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


def test_label_age_blocks_immediate_future_snapshot(tmp_path, monkeypatch):
    universe_db = tmp_path / "universe.sqlite3"
    calibration_db = tmp_path / "edge.sqlite3"
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_horizon_seconds", 3600)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_min_label_age_seconds", 600)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_min_future_snapshots", 1)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_label_snapshots_enabled", False)

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
            created_at="2026-05-24T09:01:00",
            symbol="005930",
            price=101,
        )

    result = edge_calibration.refresh_edge_training_samples(
        universe_db_path=universe_db,
        calibration_db_path=calibration_db,
        horizon_seconds=3600,
    )

    assert result["inserted_count"] == 0
    assert result["stored_sample_count"] == 0


def test_refresh_edge_training_samples_accumulates_across_multiple_scans(
    tmp_path, monkeypatch
):
    universe_db = tmp_path / "universe.sqlite3"
    calibration_db = tmp_path / "edge.sqlite3"
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_horizon_seconds", 86_400)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_min_label_age_seconds", 0)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_min_future_snapshots", 1)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_label_snapshots_enabled", False)

    universe_scanner.initialize_universe_db(universe_db)
    with sqlite3.connect(universe_db) as conn:
        for index in range(3):
            _insert_candidate(
                conn,
                scan_id=f"scan-{index}",
                scan_time=f"2026-05-2{4 + index}T09:00:00",
                symbol="005930",
                raw_score=80 - index,
                current_price=100 + index,
            )
        for index in range(1, 4):
            _insert_price(
                conn,
                scan_id=f"future-{index}",
                created_at=f"2026-05-2{4 + index}T10:00:00",
                symbol="005930",
                price=108 + index,
            )

    result = edge_calibration.refresh_edge_training_samples(
        universe_db_path=universe_db,
        calibration_db_path=calibration_db,
        horizon_seconds=86_400,
    )

    assert result["status"] == "success"
    assert result["stored_sample_count"] == 3
    assert result["inserted_count"] == 3


def test_refresh_edge_training_samples_stores_prediction_labels(tmp_path, monkeypatch):
    universe_db = tmp_path / "universe.sqlite3"
    calibration_db = tmp_path / "edge.sqlite3"
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_horizon_seconds", 3600)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_min_label_age_seconds", 0)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_min_future_snapshots", 1)

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

    summary = edge_calibration.get_edge_training_sample_summary(
        calibration_db_path=calibration_db,
    )

    assert summary["status"] == "ready"
    assert summary["sample_count"] == 1
    assert summary["summary"]["total_return_bps"] == 800
    assert summary["summary"]["win_count"] == 1
    assert summary["recent_samples"][0]["symbol"] == "005930"


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
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_min_future_snapshots", 1)
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


def test_edge_gate_uses_expectancy_instead_of_direct_win_rate(monkeypatch):
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_samples", 5)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_oos_samples", 1)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_max_mae_return_bps", 10_000)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_max_mae_risk_bps", 10_000)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_top10_avg_return_bps", 10)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_top10_win_rate", 0.50)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_top10_expectancy_bps", 0)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_fill_adjusted_edge_bps", 30)

    gate = edge_calibration._gate_from_metrics(
        sample_count=5,
        oos_sample_count=1,
        mae_return_bps=100,
        mae_risk_bps=100,
        top10_performance={
            "status": "ready",
            "sample_count": 10,
            "top_count": 10,
            "avg_return_bps": 12,
            "win_rate": 0.40,
            "loss_rate": 0.60,
            "avg_win_bps": 60,
            "avg_loss_bps": 10,
            "expectancy_bps": 18,
        },
        fill_adjustment={"multiplier": 1.0},
        candidates=[{"symbol": "005930", "net_edge": 80}],
    )

    assert gate["approved"] is True
    assert gate["required"]["target_top10_win_rate"] == 0.50

    blocked = edge_calibration._gate_from_metrics(
        sample_count=5,
        oos_sample_count=1,
        mae_return_bps=100,
        mae_risk_bps=100,
        top10_performance={
            "status": "ready",
            "sample_count": 10,
            "top_count": 10,
            "avg_return_bps": 12,
            "win_rate": 0.60,
            "loss_rate": 0.40,
            "avg_win_bps": 10,
            "avg_loss_bps": 40,
            "expectancy_bps": -10,
        },
        fill_adjustment={"multiplier": 1.0},
        candidates=[{"symbol": "005930", "net_edge": 80}],
    )

    assert blocked["approved"] is False
    assert "top10_expectancy_bps" in blocked["message"]


def test_edge_gate_uses_configured_top10_return_threshold(monkeypatch):
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_samples", 1)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_oos_samples", 1)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_max_mae_return_bps", 10_000)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_max_mae_risk_bps", 10_000)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_top10_avg_return_bps", 20)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_top10_expectancy_bps", 0)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_fill_adjusted_edge_bps", 30)

    gate = edge_calibration._gate_from_metrics(
        sample_count=10,
        oos_sample_count=10,
        mae_return_bps=50,
        mae_risk_bps=50,
        top10_performance={
            "status": "ready",
            "sample_count": 10,
            "top_count": 10,
            "avg_return_bps": 25,
            "win_rate": 0.40,
            "loss_rate": 0.60,
            "avg_win_bps": 90,
            "avg_loss_bps": 20,
            "expectancy_bps": 24,
        },
        fill_adjustment={"multiplier": 1.0},
        candidates=[{"symbol": "035900", "net_edge": 100}],
    )

    assert gate["approved"] is True
    assert gate["required"]["min_top10_avg_return_bps"] == 20
    assert gate["required"]["configured_min_top10_avg_return_bps"] == 20


def test_top10_performance_uses_all_stored_top10_samples(tmp_path):
    calibration_db = tmp_path / "edge.sqlite3"
    edge_calibration.initialize_edge_calibration_db(calibration_db)
    with sqlite3.connect(calibration_db) as conn:
        for index in range(12):
            _insert_training_sample(
                conn,
                source_candidate_id=index + 1,
                symbol=f"{index + 1:06d}",
                observed_at=f"2026-05-24T10:{index:02d}:00",
                realized_return_bps=float(index + 1),
                rank=1,
                expected_return_bps=5.0,
                expected_risk_bps=1.0,
                trading_cost_bps=0.5,
                slippage_cost_bps=0.5,
                net_edge_bps=3.0,
                raw_score=70.0,
                composite_score=80.0,
            )

    all_performance = edge_calibration._top10_performance_from_store(
        calibration_path=calibration_db,
    )
    limited_performance = edge_calibration._top10_performance_from_store(
        calibration_path=calibration_db,
        limit=10,
    )

    assert all_performance["sample_count"] == 12
    assert all_performance["avg_return_bps"] == 6.5
    assert all_performance["metric_sample_count"] == 12
    assert all_performance["avg_expected_return_bps"] == 5.0
    assert all_performance["avg_predicted_net_edge_bps"] == 3.0
    assert all_performance["avg_net_edge_error_bps"] == 2.5
    assert all_performance["net_edge_formula"]
    assert all_performance["sample_source"] == "all_stored_top10_samples"
    assert limited_performance["sample_count"] == 10


def test_refresh_top10_performance_records_every_interval(tmp_path, monkeypatch):
    calibration_db = tmp_path / "edge.sqlite3"
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_top10_performance_interval_seconds", 600)
    edge_calibration.initialize_edge_calibration_db(calibration_db)
    with sqlite3.connect(calibration_db) as conn:
        for index in range(3):
            _insert_training_sample(
                conn,
                source_candidate_id=index + 1,
                symbol="005930",
                observed_at=f"2026-05-24T10:0{index}:00",
                realized_return_bps=10.0,
                rank=1,
            )

    first = edge_calibration.refresh_top10_performance_if_due(
        calibration_db_path=calibration_db,
        force=True,
    )
    second = edge_calibration.refresh_top10_performance_if_due(
        calibration_db_path=calibration_db,
    )

    assert first["status"] == "success"
    assert first["top10_performance"]["sample_count"] == 3
    assert second["status"] == "not_due"
    assert second["top10_performance"]["sample_count"] == 3


def test_refresh_purges_premature_horizon_labels(tmp_path, monkeypatch):
    calibration_db = tmp_path / "edge.sqlite3"
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_horizon_seconds", 86_400)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_label_at_horizon_end", True)
    edge_calibration.initialize_edge_calibration_db(calibration_db)
    with sqlite3.connect(calibration_db) as conn:
        _insert_training_sample(
            conn,
            source_candidate_id=1,
            symbol="005930",
            observed_at="2026-05-24T10:00:00",
            realized_return_bps=-100.0,
            rank=1,
            label_observation_span_seconds=3600,
        )

    performance = edge_calibration.refresh_top10_performance_if_due(
        calibration_db_path=calibration_db,
        force=True,
    )
    summary = edge_calibration.get_edge_training_sample_summary(
        calibration_db_path=calibration_db,
    )

    assert performance["top10_performance"]["sample_count"] == 0
    assert summary["sample_count"] == 0


def test_top10_performance_includes_archived_ranked_samples(tmp_path):
    calibration_db = tmp_path / "edge.sqlite3"
    edge_calibration.initialize_edge_calibration_db(calibration_db)
    with sqlite3.connect(calibration_db) as conn:
        _insert_training_sample(
            conn,
            source_candidate_id=1,
            symbol="005930",
            observed_at="2026-05-24T10:00:00",
            realized_return_bps=10.0,
            rank=1,
            status="ARCHIVED",
        )
        _insert_training_sample(
            conn,
            source_candidate_id=2,
            symbol="000660",
            observed_at="2026-05-24T10:01:00",
            realized_return_bps=20.0,
            rank=1,
            status="READY",
        )

    performance = edge_calibration._top10_performance_from_store(
        calibration_path=calibration_db,
    )

    assert performance["sample_count"] == 2
    assert performance["avg_return_bps"] == 15.0


def test_paper_and_live_gate_use_two_bps_top10_average(monkeypatch):
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_samples", 1)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_oos_samples", 1)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_max_mae_return_bps", 10_000)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_max_mae_risk_bps", 10_000)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_top10_avg_return_bps", 2)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_paper_min_top10_avg_return_bps", 2)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_top10_expectancy_bps", 0)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_fill_adjusted_edge_bps", 0)

    top10_performance = {
        "status": "ready",
        "sample_count": 1,
        "top_count": 10,
        "avg_return_bps": 2.5,
        "win_rate": 1.0,
        "loss_rate": 0.0,
        "avg_win_bps": 2.5,
        "avg_loss_bps": 0.0,
        "expectancy_bps": 2.5,
    }
    paper_gate = edge_calibration._gate_from_metrics(
        sample_count=1,
        oos_sample_count=1,
        mae_return_bps=0,
        mae_risk_bps=0,
        top10_performance=top10_performance,
        fill_adjustment={"multiplier": 1.0},
        candidates=[{"symbol": "005930", "net_edge": 100}],
        execution_mode="paper",
    )
    live_gate = edge_calibration._gate_from_metrics(
        sample_count=1,
        oos_sample_count=1,
        mae_return_bps=0,
        mae_risk_bps=0,
        top10_performance=top10_performance,
        fill_adjustment={"multiplier": 1.0},
        candidates=[{"symbol": "005930", "net_edge": 100}],
    )

    assert paper_gate["approved"] is True
    assert paper_gate["required"]["min_top10_avg_return_bps"] == 2
    assert live_gate["approved"] is True
    assert live_gate["required"]["min_top10_avg_return_bps"] == 2


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


def _insert_training_sample(
    conn: sqlite3.Connection,
    *,
    source_candidate_id: int,
    symbol: str,
    observed_at: str,
    realized_return_bps: float,
    rank: int,
    expected_return_bps: float | None = None,
    expected_risk_bps: float | None = None,
    trading_cost_bps: float | None = None,
    slippage_cost_bps: float | None = None,
    net_edge_bps: float | None = None,
    raw_score: float | None = None,
    composite_score: float | None = None,
    status: str = "READY",
    label_observation_span_seconds: int | None = None,
) -> None:
    if label_observation_span_seconds is None:
        label_observation_span_seconds = int(
            edge_calibration.settings.edge_calibration_horizon_seconds or 86_400
        )
    conn.execute(
        """
        INSERT INTO edge_training_samples (
            source_candidate_id, symbol, scan_time, observed_at, entry_price,
            observed_price, features_json, realized_return_bps,
            realized_risk_bps, label_observation_span_seconds, raw_score, expected_return_bps,
            expected_risk_bps, trading_cost_bps, slippage_cost_bps,
            net_edge_bps, composite_score, rank, status, created_at, raw_json
        )
        VALUES (?, ?, '2026-05-24T09:00:00', ?, 100, 101, ?,
                ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}')
        """,
        (
            source_candidate_id,
            symbol,
            observed_at,
            "[1,0,0,0,0,0,0,0,0,0]",
            realized_return_bps,
            label_observation_span_seconds,
            raw_score,
            expected_return_bps,
            expected_risk_bps,
            trading_cost_bps,
            slippage_cost_bps,
            net_edge_bps,
            composite_score,
            rank,
            status,
            observed_at,
        ),
    )
