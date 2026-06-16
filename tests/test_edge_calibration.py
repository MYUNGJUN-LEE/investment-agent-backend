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
    assert estimate["edge_model"] == "calibrated_ridge_v2_cost_adjusted"


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
    assert summary["unit_performance"]["candidate_label"]["win_rate"] == 1.0


def test_refresh_edge_training_samples_deduplicates_symbol_within_horizon(
    tmp_path, monkeypatch
):
    universe_db = tmp_path / "universe.sqlite3"
    calibration_db = tmp_path / "edge.sqlite3"
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_horizon_seconds", 3600)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_min_label_age_seconds", 0)
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
        _insert_candidate(
            conn,
            scan_id="scan-b",
            scan_time="2026-05-24T09:10:00",
            symbol="005930",
            raw_score=79,
            current_price=101,
        )
        _insert_price(
            conn,
            scan_id="future-a",
            created_at="2026-05-24T10:00:00",
            symbol="005930",
            price=108,
        )
        _insert_price(
            conn,
            scan_id="future-b",
            created_at="2026-05-24T10:10:00",
            symbol="005930",
            price=109,
        )

    result = edge_calibration.refresh_edge_training_samples(
        universe_db_path=universe_db,
        calibration_db_path=calibration_db,
        horizon_seconds=3600,
    )

    assert result["inserted_count"] == 1
    assert result["stored_sample_count"] == 1


def test_realized_risk_uses_entry_to_horizon_adverse_excursion(
    tmp_path, monkeypatch
):
    universe_db = tmp_path / "universe.sqlite3"
    calibration_db = tmp_path / "edge.sqlite3"
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_horizon_seconds", 3600)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_min_label_age_seconds", 0)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_min_future_snapshots", 1)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_label_snapshots_enabled", False)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_label_at_horizon_end", True)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_label_horizon_tolerance_seconds", 0)

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
            scan_id="future-mid",
            created_at="2026-05-24T09:30:00",
            symbol="005930",
            price=90,
        )
        _insert_price(
            conn,
            scan_id="future-end",
            created_at="2026-05-24T10:00:00",
            symbol="005930",
            price=110,
        )

    result = edge_calibration.refresh_edge_training_samples(
        universe_db_path=universe_db,
        calibration_db_path=calibration_db,
        horizon_seconds=3600,
    )

    assert result["inserted_count"] == 1
    with sqlite3.connect(calibration_db) as conn:
        risk_bps = conn.execute(
            "SELECT realized_risk_bps FROM edge_training_samples"
        ).fetchone()[0]

    assert risk_bps == 1000


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
    monkeypatch.setattr(
        edge_calibration.settings,
        "edge_calibration_gate_max_mae_net_edge_bps",
        10_000,
        raising=False,
    )
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_top10_avg_return_bps", 0)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_top10_win_rate", 0.0)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_fill_adjusted_edge_bps", 60)

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
                net_edge=200 + index * 10,
            )
            _insert_price(
                conn,
                scan_id=f"future-{index}",
                created_at=f"2026-05-24T10:0{index}:00",
                symbol=symbol,
                price=95 if index == 0 else 105 + index,
            )

    result = edge_calibration.calibrate_edge_model(
        universe_db_path=universe_db,
        calibration_db_path=calibration_db,
        min_samples=5,
        max_samples=10,
    )
    with sqlite3.connect(calibration_db) as conn:
        for index in range(5):
            conn.execute(
                """
                UPDATE edge_training_samples
                SET net_edge_bps = ?
                WHERE symbol = ?
                """,
                (100 + index * 50, f"{index + 1:06d}"),
            )
    gate = edge_calibration.edge_entry_gate(
        [
            {
                "symbol": "000001",
                "status": "READY",
                "net_edge": 200,
                "expected_return": 500,
                "trading_cost": 49.4,
                "slippage_cost": 10,
                "liquidity_drag_bps": 0,
            }
        ],
        calibration_db_path=calibration_db,
    )

    assert result["status"] == "success"
    assert result["stored_sample_count"] == 5
    assert result["oos_sample_count"] >= 1
    assert gate["approved"] is True
    assert gate["top10_performance"]["sample_count"] == 5


def test_edge_gate_requires_current_top10_quality_metrics(monkeypatch):
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_samples", 5)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_oos_samples", 1)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_max_mae_return_bps", 10_000)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_max_mae_risk_bps", 10_000)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_top10_avg_return_bps", 10)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_top10_win_rate", 0.50)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_top10_expectancy_bps", 0)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_fill_adjusted_edge_bps", 60)

    top10_performance = {
        "status": "ready",
        "sample_count": 10,
        "top_count": 10,
        "avg_return_bps": 12,
        "win_rate": 0.60,
        "loss_rate": 0.40,
        "avg_win_bps": 60,
        "avg_loss_bps": 10,
        "expectancy_bps": 32,
    }
    candidate = {
        "symbol": "005930",
        "status": "READY",
        "net_edge": 200,
        "expected_return": 500,
        "trading_cost": 49.4,
        "slippage_cost": 10,
        "liquidity_drag_bps": 0,
    }

    gate = edge_calibration._gate_from_metrics(
        sample_count=5,
        oos_sample_count=1,
        mae_return_bps=100,
        mae_risk_bps=100,
        top10_performance=top10_performance,
        fill_adjustment={"multiplier": 1.0},
        ic_metrics={"ic": 0.03},
        candidates=[candidate],
    )

    assert gate["approved"] is True
    assert gate["required"]["target_top10_win_rate"] == 0.50
    assert gate["required"]["min_profit_factor"] == 1.10
    assert gate["required"]["min_recent_ic"] == 0.02
    assert gate["required"]["min_cost_coverage"] == 2.0
    assert gate["top10_performance"]["profit_factor"] == 9.0
    assert gate["best_cost_coverage"] == round(500 / (49.4 + 10), 4)

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
        ic_metrics={"ic": 0.03},
        candidates=[candidate],
    )

    assert blocked["approved"] is False
    assert "top10_expectancy_bps" in blocked["message"]

    low_win_rate = edge_calibration._gate_from_metrics(
        sample_count=5,
        oos_sample_count=1,
        mae_return_bps=100,
        mae_risk_bps=100,
        top10_performance={
            **top10_performance,
            "win_rate": 0.40,
            "loss_rate": 0.60,
            "expectancy_bps": 18,
        },
        fill_adjustment={"multiplier": 1.0},
        ic_metrics={"ic": 0.03},
        candidates=[candidate],
    )

    assert low_win_rate["approved"] is False
    assert "top10_win_rate 0.4 < 0.5" in low_win_rate["message"]


def test_edge_gate_blocks_when_net_edge_mae_is_too_high(monkeypatch):
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_samples", 1)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_oos_samples", 1)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_max_mae_return_bps", 10_000)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_max_mae_risk_bps", 10_000)
    monkeypatch.setattr(
        edge_calibration.settings,
        "edge_calibration_gate_max_mae_net_edge_bps",
        180,
        raising=False,
    )
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_top10_avg_return_bps", 0)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_top10_win_rate", 0.0)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_top10_expectancy_bps", -1)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_fill_adjusted_edge_bps", 0)

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
            "win_rate": 0.60,
            "loss_rate": 0.40,
            "avg_win_bps": 90,
            "avg_loss_bps": 20,
            "expectancy_bps": 24,
            "mae_net_edge_error_bps": 451,
        },
        fill_adjustment={"multiplier": 1.0},
        ic_metrics={"ic": 0.03},
        candidates=[
            {
                "symbol": "035900",
                "status": "READY",
                "net_edge": 200,
                "expected_return": 500,
                "trading_cost": 49.4,
                "slippage_cost": 10,
                "liquidity_drag_bps": 0,
            }
        ],
    )

    assert gate["approved"] is False
    assert "mae_net_edge_error_bps 451.0 >= 180.0" in gate["message"]


def test_edge_gate_uses_configured_top10_return_threshold(monkeypatch):
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_samples", 1)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_oos_samples", 1)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_max_mae_return_bps", 10_000)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_max_mae_risk_bps", 10_000)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_top10_avg_return_bps", 20)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_top10_expectancy_bps", 0)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_fill_adjusted_edge_bps", 60)

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
            "win_rate": 0.60,
            "loss_rate": 0.40,
            "avg_win_bps": 90,
            "avg_loss_bps": 20,
            "expectancy_bps": 24,
        },
        fill_adjustment={"multiplier": 1.0},
        ic_metrics={"ic": 0.03},
        candidates=[
            {
                "symbol": "035900",
                "status": "READY",
                "net_edge": 200,
                "expected_return": 500,
                "trading_cost": 49.4,
                "slippage_cost": 10,
                "liquidity_drag_bps": 0,
            }
        ],
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
    assert all_performance["sample_source"] == edge_calibration.TOP10_SAMPLE_SOURCE_SCAN_RUN
    assert limited_performance["sample_count"] == 10


def test_top10_performance_groups_candidates_by_scan_run_id(tmp_path):
    calibration_db = tmp_path / "edge.sqlite3"
    edge_calibration.initialize_edge_calibration_db(calibration_db)
    with sqlite3.connect(calibration_db) as conn:
        for index, return_bps in enumerate((10.0, 20.0, 30.0), start=1):
            _insert_training_sample(
                conn,
                source_candidate_id=index,
                scan_id="scan-a",
                symbol=f"00593{index}",
                observed_at=f"2026-05-24T10:0{index}:00",
                realized_return_bps=return_bps,
                rank=index,
            )
        _insert_training_sample(
            conn,
            source_candidate_id=10,
            scan_id="scan-b",
            symbol="000660",
            observed_at="2026-05-24T10:10:00",
            realized_return_bps=40.0,
            rank=1,
        )

    performance = edge_calibration._top10_performance_from_store(
        calibration_path=calibration_db,
    )

    assert performance["sample_count"] == 2
    assert performance["scan_run_count"] == 2
    assert performance["candidate_sample_count"] == 4
    assert performance["avg_return_bps"] == 30.0


def test_refresh_top10_performance_records_every_interval(tmp_path, monkeypatch):
    calibration_db = tmp_path / "edge.sqlite3"
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_top10_performance_interval_seconds", 600)
    edge_calibration.initialize_edge_calibration_db(calibration_db)
    with sqlite3.connect(calibration_db) as conn:
        for index in range(3):
            _insert_training_sample(
                conn,
                source_candidate_id=index + 1,
                symbol=f"{index + 1:06d}",
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


def test_top10_performance_uses_only_current_eligible_ranked_samples(tmp_path):
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

    assert performance["sample_count"] == 1
    assert performance["avg_return_bps"] == 20.0


def test_store_training_sample_calculates_realized_net_edge_label_and_penalty(tmp_path):
    calibration_db = tmp_path / "edge.sqlite3"
    edge_calibration.initialize_edge_calibration_db(calibration_db)
    with sqlite3.connect(calibration_db) as conn:
        inserted = edge_calibration._store_training_sample(
            conn,
            {
                "source_candidate_id": 1,
                "scan_id": "scan-a",
                "label_horizon_key": "005930:2026-05-24T09:00:00:86400",
                "symbol": "005930",
                "scan_time": "2026-05-24T09:00:00",
                "observed_at": "2026-05-25T09:00:00",
                "entry_price": 100.0,
                "observed_price": 96.0,
                "features": [0.0] * len(edge_calibration.FEATURE_NAMES),
                "realized_return_bps": -400.0,
                "realized_risk_bps": 500.0,
                "trading_cost_bps": 50.0,
                "slippage_cost_bps": 20.0,
                "tax_bps": 10.0,
                "net_edge_bps": 260.0,
                "rank": 1,
                "status": "READY",
                "raw_json": {},
            },
        )
        row = conn.execute(
            """
            SELECT realized_net_edge_bps, net_edge_label, false_positive_flag,
                   severe_false_positive_flag, sample_weight
            FROM edge_training_samples
            WHERE source_candidate_id = 1
            """
        ).fetchone()

    assert inserted == 1
    assert row[0] == -530.0
    assert row[1] == 0
    assert row[2] == 1
    assert row[3] == 1
    assert row[4] == 5.0


def test_false_positive_metrics_block_calibration_gate(monkeypatch):
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_samples", 20)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_oos_samples", 1)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_max_mae_return_bps", 10_000)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_max_mae_risk_bps", 10_000)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_top10_avg_return_bps", 0)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_top10_win_rate", 0.0)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_top10_expectancy_bps", -1)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_fill_adjusted_edge_bps", 0)

    gate = edge_calibration._gate_from_metrics(
        sample_count=20,
        oos_sample_count=1,
        mae_return_bps=50,
        mae_risk_bps=50,
        top10_performance={
            "status": "ready",
            "sample_count": 20,
            "metric_sample_count": 20,
            "top_count": 10,
            "avg_return_bps": 25,
            "win_rate": 0.60,
            "loss_rate": 0.40,
            "avg_win_bps": 90,
            "avg_loss_bps": 20,
            "expectancy_bps": 24,
            "mae_net_edge_error_bps": 50,
            "false_positive_rate": 0.25,
            "severe_false_positive_count": 2,
        },
        fill_adjustment={"multiplier": 1.0},
        ic_metrics={"ic": 0.03},
        candidates=[
            {
                "symbol": "035900",
                "status": "READY",
                "net_edge": 200,
                "expected_return": 500,
                "trading_cost": 49.4,
                "slippage_cost": 10,
                "liquidity_drag_bps": 0,
            }
        ],
    )

    assert gate["approved"] is False
    assert "false_positive_rate too high" in gate["message"]
    assert "severe false positives detected" in gate["message"]


def test_risk_rejected_samples_are_split_from_executable_aggregate(tmp_path):
    calibration_db = tmp_path / "edge.sqlite3"
    edge_calibration.initialize_edge_calibration_db(calibration_db)
    with sqlite3.connect(calibration_db) as conn:
        _insert_training_sample(
            conn,
            source_candidate_id=1,
            symbol="005930",
            observed_at="2026-05-24T10:00:00",
            realized_return_bps=200.0,
            rank=1,
            net_edge_bps=180.0,
            status="READY",
        )
        _insert_training_sample(
            conn,
            source_candidate_id=2,
            symbol="000660",
            observed_at="2026-05-24T10:01:00",
            realized_return_bps=-800.0,
            rank=1,
            net_edge_bps=180.0,
            status="RISK_REJECTED",
        )

    summary = edge_calibration.get_edge_training_sample_summary(
        calibration_db_path=calibration_db,
    )
    splits = summary["net_edge_aggregate_splits"]

    assert splits["all_observed_candidates"]["sample_count"] == 2
    assert splits["executable_candidates_only"]["sample_count"] == 1
    assert splits["risk_rejected_candidates"]["sample_count"] == 1
    assert splits["executable_candidates_only"]["total_realized_net_edge_bps"] == 200.0
    assert splits["risk_rejected_candidates"]["total_realized_net_edge_bps"] == -800.0
    assert splits["top_rank_executable_only"]["sample_count"] == 1


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
        "win_rate": 0.80,
        "loss_rate": 0.20,
        "avg_win_bps": 3.0,
        "avg_loss_bps": 1.0,
        "expectancy_bps": 2.2,
    }
    candidate = {
        "symbol": "005930",
        "status": "READY",
        "net_edge": 100,
        "expected_return": 500,
        "trading_cost": 49.4,
        "slippage_cost": 10,
        "liquidity_drag_bps": 0,
    }
    paper_gate = edge_calibration._gate_from_metrics(
        sample_count=1,
        oos_sample_count=1,
        mae_return_bps=0,
        mae_risk_bps=0,
        top10_performance=top10_performance,
        fill_adjustment={"multiplier": 1.0},
        ic_metrics={"ic": 0.03},
        candidates=[candidate],
        execution_mode="paper",
    )
    live_gate = edge_calibration._gate_from_metrics(
        sample_count=1,
        oos_sample_count=1,
        mae_return_bps=0,
        mae_risk_bps=0,
        top10_performance=top10_performance,
        fill_adjustment={"multiplier": 1.0},
        ic_metrics={"ic": 0.03},
        candidates=[candidate],
    )

    assert paper_gate["approved"] is True
    assert paper_gate["required"]["min_top10_avg_return_bps"] == 2
    assert live_gate["approved"] is True
    assert live_gate["required"]["min_top10_avg_return_bps"] == 2


def test_broker_paper_bootstrap_makes_candidate_label_gate_observe_only(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(edge_calibration.settings, "kis_is_paper", True)
    monkeypatch.setattr(edge_calibration.settings, "broker_paper_bootstrap_enabled", True)
    monkeypatch.setattr(
        edge_calibration.settings,
        "broker_paper_calibration_source",
        "broker_fills",
    )
    monkeypatch.setattr(
        edge_calibration.settings,
        "broker_paper_candidate_label_gate_mode",
        "observe_only",
    )
    monkeypatch.setattr(edge_calibration.settings, "broker_paper_min_fill_samples", 200)
    monkeypatch.setattr(edge_calibration.settings, "broker_paper_min_oos_fill_samples", 50)
    monkeypatch.setattr(edge_calibration.settings, "broker_sync_db_path", str(tmp_path / "broker.sqlite3"))
    monkeypatch.setattr(
        edge_calibration.settings,
        "outcome_attribution_db_path",
        str(tmp_path / "outcomes.sqlite3"),
    )

    gate = edge_calibration.edge_entry_gate(
        calibration_db_path=tmp_path / "missing_edge.sqlite3",
        execution_mode="broker_paper",
    )

    assert gate["approved"] is True
    assert gate["status"] == "bootstrap_observe_only"
    assert gate["candidate_label_gate_failed"] is True
    assert gate["candidate_label_gate_hard_blocking"] is False
    assert gate["broker_paper_fill_sample_count"] == 0
    assert gate["broker_paper_fill_gate_ready"] is False
    assert gate["broker_paper_fill_gate_hard_blocking"] is False
    assert (
        gate["calibration_gate_mode"]
        == "broker_paper_bootstrap_candidate_label_observe_only"
    )
    assert "broker_paper bootstrap observe-only" in gate["message"]


def test_broker_paper_bootstrap_disabled_keeps_candidate_label_hard_blocking(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(edge_calibration.settings, "kis_is_paper", True)
    monkeypatch.setattr(edge_calibration.settings, "broker_paper_bootstrap_enabled", False)
    monkeypatch.setattr(
        edge_calibration.settings,
        "broker_paper_calibration_source",
        "broker_fills",
    )
    monkeypatch.setattr(
        edge_calibration.settings,
        "broker_paper_candidate_label_gate_mode",
        "observe_only",
    )
    monkeypatch.setattr(edge_calibration.settings, "broker_sync_db_path", str(tmp_path / "broker.sqlite3"))
    monkeypatch.setattr(
        edge_calibration.settings,
        "outcome_attribution_db_path",
        str(tmp_path / "outcomes.sqlite3"),
    )

    gate = edge_calibration.edge_entry_gate(
        calibration_db_path=tmp_path / "missing_edge.sqlite3",
        execution_mode="broker_paper",
    )

    assert gate["approved"] is False
    assert gate["candidate_label_gate_failed"] is True
    assert gate["candidate_label_gate_hard_blocking"] is True
    assert gate["calibration_gate_mode"] == "candidate_label_hard_blocking"


def test_live_mode_keeps_candidate_label_gate_hard_blocking(tmp_path, monkeypatch):
    monkeypatch.setattr(edge_calibration.settings, "kis_is_paper", True)
    monkeypatch.setattr(edge_calibration.settings, "broker_paper_bootstrap_enabled", True)
    monkeypatch.setattr(edge_calibration.settings, "broker_sync_db_path", str(tmp_path / "broker.sqlite3"))
    monkeypatch.setattr(
        edge_calibration.settings,
        "outcome_attribution_db_path",
        str(tmp_path / "outcomes.sqlite3"),
    )

    gate = edge_calibration.edge_entry_gate(
        calibration_db_path=tmp_path / "missing_edge.sqlite3",
        execution_mode="live",
    )

    assert gate["approved"] is False
    assert gate["candidate_label_gate_failed"] is True
    assert gate["candidate_label_gate_hard_blocking"] is True


def test_broker_paper_fill_gate_hard_blocks_after_enough_bad_fills(monkeypatch):
    monkeypatch.setattr(edge_calibration.settings, "kis_is_paper", True)
    monkeypatch.setattr(edge_calibration.settings, "broker_paper_bootstrap_enabled", True)
    monkeypatch.setattr(
        edge_calibration.settings,
        "broker_paper_calibration_source",
        "broker_fills",
    )
    monkeypatch.setattr(
        edge_calibration.settings,
        "broker_paper_candidate_label_gate_mode",
        "observe_only",
    )
    monkeypatch.setattr(edge_calibration.settings, "broker_paper_min_fill_samples", 2)
    monkeypatch.setattr(edge_calibration.settings, "broker_paper_min_oos_fill_samples", 1)
    monkeypatch.setattr(edge_calibration.settings, "edge_calibration_gate_min_top10_win_rate", 0.50)
    monkeypatch.setattr(
        edge_calibration.settings,
        "edge_calibration_gate_max_mae_net_edge_bps",
        180.0,
    )
    monkeypatch.setattr(
        edge_calibration,
        "broker_paper_fill_gate_metrics",
        lambda **kwargs: {
            "broker_paper_fill_sample_count": 2,
            "broker_paper_oos_fill_sample_count": 1,
            "broker_paper_fill_outcome_sample_count": 2,
            "broker_paper_fill_win_rate": 0.0,
            "broker_paper_fill_profit_factor": 0.0,
            "broker_paper_fill_avg_realized_net_edge_bps": -25.0,
            "broker_paper_fill_mae_edge_error_bps": 250.0,
        },
    )

    gate = edge_calibration._apply_broker_paper_calibration_policy(
        {
            "status": "blocked",
            "approved": False,
            "message": "candidate label calibration failed",
            "sample_count": 6,
        },
        execution_mode="broker_paper",
    )

    assert gate["approved"] is False
    assert gate["candidate_label_gate_hard_blocking"] is False
    assert gate["broker_paper_fill_gate_ready"] is True
    assert gate["broker_paper_fill_gate_hard_blocking"] is True
    assert "Broker-paper fill calibration gate blocked entries" in gate["message"]


def _insert_candidate(
    conn: sqlite3.Connection,
    *,
    scan_id: str,
    scan_time: str,
    symbol: str,
    raw_score: float,
    current_price: float,
    expected_return: float = 0.0,
    expected_risk: float = 0.0,
    trading_cost: float = 0.0,
    slippage_cost: float = 0.0,
    net_edge: float = 0.0,
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
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 'test', 'READY',
                'buy_candidate', ?, 2, 1000000, 2, 50000000000, 1, 0,
                NULL, '2026-05-24T10:00:00', ?)
        """,
        (
            scan_id,
            scan_time,
            symbol,
            symbol,
            raw_score,
            expected_return,
            expected_risk,
            trading_cost,
            slippage_cost,
            net_edge,
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
    scan_id: str | None = None,
    symbol: str,
    scan_time: str = "2026-05-24T09:00:00",
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
            source_candidate_id, scan_id, symbol, scan_time, observed_at, entry_price,
            observed_price, features_json, realized_return_bps,
            realized_risk_bps, label_observation_span_seconds, raw_score, expected_return_bps,
            expected_risk_bps, trading_cost_bps, slippage_cost_bps,
            net_edge_bps, composite_score, rank, status, created_at, raw_json
        )
        VALUES (?, ?, ?, ?, ?, 100, 101, ?,
                ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}')
        """,
        (
            source_candidate_id,
            scan_id,
            symbol,
            scan_time,
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
