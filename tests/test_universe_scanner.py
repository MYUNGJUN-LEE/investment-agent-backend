from __future__ import annotations

import sqlite3

import pytest

from app.models import AutoTradeStartRequest, AutoTradeSymbolConfig
from app.trading import auto_trading
from app.trading.atr_exits import atr_exit_levels_from_price_data
from app.trading import universe_scanner


@pytest.fixture(autouse=True)
def stable_market_context(monkeypatch):
    monkeypatch.setattr(
        universe_scanner,
        "get_latest_market_context",
        lambda: {
            "market_regime": "bull",
            "risk_on_score": 70,
            "selected_sector_relative_strength": {"score": 65},
        },
    )


def test_universe_scanner_scores_stores_and_returns_final_symbols(tmp_path, monkeypatch):
    db_path = tmp_path / "universe.sqlite3"
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_seed_symbols", "")
    monkeypatch.setattr(universe_scanner.settings, "monitor_watchlist_symbols", "")
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_max_source_symbols", 3)
    monkeypatch.setattr(
        universe_scanner.settings,
        "universe_scanner_worker_hurdle_rate_bps",
        0,
    )
    monkeypatch.setattr(
        universe_scanner.settings,
        "universe_scanner_symbol_interval_seconds",
        0,
    )
    monkeypatch.setattr(
        universe_scanner,
        "_paper_average_realized_return_bps",
        lambda: None,
    )
    monkeypatch.setattr(
        universe_scanner,
        "edge_entry_gate",
        lambda candidates=None: {
            "status": "approved",
            "approved": True,
            "message": "test gate approved",
        },
    )

    price_rows = {
        "035900": {
            "symbol": "035900",
            "current_price": 70000,
            "change_rate": 2.1,
            "volume": 10_000_000,
            "volume_ratio": 2.4,
            "turnover_value": 700_000_000_000,
            "market_cap": 1_200_000_000_000,
            "market_segment": "KOSDAQ",
            "intraday": {"minute_volume_ratio": 2.0},
            "latest_technical_features": {
                "close": 70000,
                "return_5d": 0.04,
                "return_20d": 0.08,
                "return_60d": 0.18,
                "high_breakout_20d": True,
                "low_breakdown_20d": False,
                "ma20_slope": 350,
                "atr_14": 1000,
                "atr_14_pct": 1000 / 70000,
            },
            "technical_features": [
                {"close": 69000, "atr_14": 1000},
                {"close": 70500, "atr_14": 1000},
            ],
            "overheated": False,
            "source": "test",
        },
        "041510": {
            "symbol": "041510",
            "current_price": 180000,
            "change_rate": 8.5,
            "volume": 3_000_000,
            "volume_ratio": 1.6,
            "turnover_value": 540_000_000_000,
            "market_cap": 900_000_000_000,
            "market_segment": "KOSDAQ",
            "intraday": {"minute_volume_ratio": 1.1},
            "latest_technical_features": {
                "close": 180000,
                "return_5d": 0.01,
                "return_20d": 0.02,
                "return_60d": 0.04,
                "high_breakout_20d": False,
                "low_breakdown_20d": False,
                "ma20_slope": 100,
                "atr_14": 5000,
                "atr_14_pct": 5000 / 180000,
            },
            "overheated": False,
            "source": "test",
        },
        "145020": {
            "symbol": "145020",
            "current_price": 350000,
            "change_rate": -6.0,
            "volume": 100_000,
            "volume_ratio": 0.4,
            "turnover_value": 35_000_000_000,
            "market_cap": 1_500_000_000_000,
            "market_segment": "KOSDAQ",
            "intraday": {},
            "latest_technical_features": {
                "close": 350000,
                "return_5d": -0.04,
                "return_20d": -0.09,
                "return_60d": -0.12,
                "high_breakout_20d": False,
                "low_breakdown_20d": True,
                "ma20_slope": -3000,
                "atr_14": 20000,
                "atr_14_pct": 20000 / 350000,
            },
            "overheated": False,
            "source": "test",
        },
    }

    monkeypatch.setattr(
        universe_scanner,
        "fetch_price_data",
        lambda symbol: dict(price_rows[symbol]),
    )
    monkeypatch.setattr(
        universe_scanner,
        "search_naver_news",
        lambda query, display=5, sort="date": {"items": [{"title": query}]},
    )
    monkeypatch.setattr(
        universe_scanner,
        "fetch_opendart_disclosures",
        lambda symbol, lookback_hours: [],
    )

    req = AutoTradeStartRequest(
        auto_discover_symbols=True,
        universe_seed_symbols=["035900", "041510", "145020"],
        universe_candidate_limit=2,
        universe_final_limit=1,
    )

    result = universe_scanner.scan_universe_for_auto_trade(req, db_path=db_path)
    latest = universe_scanner.get_latest_universe_scan(db_path=db_path)
    expected_levels = atr_exit_levels_from_price_data(
        entry_price=70000,
        price_data=price_rows["035900"],
    )

    assert result["status"] == "success"
    assert result["candidate_count"] == 2
    assert result["final_count"] == 1
    assert result["symbols"][0].symbol == "035900"
    assert result["symbols"][0].account_equity == 10_000_000
    assert result["symbols"][0].risk_per_trade == 0.005
    assert result["symbols"][0].stop_loss == expected_levels["stop_loss"]
    assert result["symbols"][0].take_profit == expected_levels["take_profit"]
    assert result["symbols"][0].expected_loss_bps == 300
    assert result["symbols"][0].expected_win_bps == 500
    assert result["symbols"][0].trailing_stop == expected_levels["trailing_stop"]
    assert result["symbols"][0].strategy_type == "swing"
    assert result["symbols"][0].expected_gross_edge_bps is not None
    assert latest["scan_id"] == result["scan_id"]
    assert latest["candidates"][0]["symbol"] == "035900"
    assert latest["candidates"][0]["net_edge"] > result["worker_hurdle_rate"]


def test_high_quality_swing_candidate_lifts_fill_adjusted_edge(monkeypatch):
    monkeypatch.setattr(
        universe_scanner,
        "estimate_expected_edges",
        lambda candidate, raw_score, model=None: {
            "expected_return": 0.0,
            "expected_risk": 280.0,
            "edge_model": "calibrated_test",
        },
    )
    monkeypatch.setattr(
        universe_scanner,
        "get_latest_market_context",
        lambda: {
            "market_regime": "bull",
            "risk_on_score": 72,
            "selected_sector_relative_strength": {"score": 70},
        },
    )
    candidate = {
        "symbol": "005930",
        "name": "Samsung Electronics",
        "score": 88,
        "decision": "buy_candidate",
        "current_price": 70_000,
        "change_rate": 2.4,
        "volume": 12_000_000,
        "volume_ratio": 2.6,
        "turnover_value": 840_000_000_000,
        "sector": "semiconductors",
        "intraday": {"minute_volume_ratio": 1.6},
        "latest_technical_features": {
            "close": 70_000,
            "return_5d": 0.04,
            "return_20d": 0.11,
            "return_60d": 0.18,
            "high_breakout_20d": True,
            "low_breakdown_20d": False,
            "ma20_slope": 420,
            "atr_14": 1_800,
            "atr_14_pct": 1_800 / 70_000,
        },
        "technical_features": [
            {"close": 69_500, "atr_14": 1_800},
            {"close": 72_000, "atr_14": 1_800},
        ],
        "overheated": False,
    }

    scored = universe_scanner._with_expected_value_scores(
        candidate,
        expires_at="2026-05-25T10:00:00",
        edge_model={"expected_return": {"bias": 0.0}, "expected_risk": {"bias": 280.0}},
    )

    assert scored["edge_quality_score"] >= 75
    assert scored["edge_reward_risk"]["gross_return_floor_bps"] > 300
    assert scored["net_edge"] >= 30
    assert "atr_rr" in scored["edge_model"]


def test_universe_filters_block_illiquid_macro_and_inverse_alignment(monkeypatch):
    base = {
        "symbol": "005930",
        "current_price": 10_000,
        "change_rate": 2.0,
        "volume": 1_000_000,
        "volume_ratio": 2.0,
        "turnover_value": 30_000_000_000,
        "market_context": {"market_regime": "bull", "risk_on_score": 70},
        "latest_technical_features": {
            "close": 10_000,
            "return_5d": 0.03,
            "return_20d": 0.06,
            "return_60d": 0.12,
            "high_breakout_20d": True,
            "low_breakdown_20d": False,
            "ma5": 10_200,
            "ma20": 10_400,
            "ma60": 10_600,
            "ma20_slope": -20,
            "atr_14_pct": 0.03,
        },
    }

    inverse = universe_scanner._score_snapshot(dict(base))
    assert inverse["decision"] == "exclude"
    assert "inverse_alignment" in inverse["reason"]

    illiquid = universe_scanner._score_snapshot(
        {
            **base,
            "volume": 50_000,
            "turnover_value": 1_000_000_000,
            "latest_technical_features": {
                **base["latest_technical_features"],
                "ma5": 10_000,
                "ma20": 9_800,
                "ma60": 9_500,
                "ma20_slope": 20,
            },
        }
    )
    assert illiquid["decision"] == "exclude"
    assert "liquidity" in illiquid["reason"]

    macro = universe_scanner._score_snapshot(
        {
            **base,
            "market_context": {"market_regime": "bear", "risk_on_score": 20},
            "latest_technical_features": {
                **base["latest_technical_features"],
                "ma5": 10_000,
                "ma20": 9_800,
                "ma60": 9_500,
                "ma20_slope": 20,
            },
        }
    )
    assert macro["decision"] == "exclude"
    assert "macro_trend" in macro["reason"]


def test_universe_scanner_adds_weighted_momentum_score():
    scored = universe_scanner._score_snapshot(
        {
            "symbol": "005930",
            "current_price": 10_000,
            "change_rate": 3.0,
            "volume": 2_000_000,
            "volume_ratio": 3.0,
            "turnover_value": 60_000_000_000,
            "relative_strength_score": 78,
            "market_context": {"market_regime": "bull", "risk_on_score": 70},
            "latest_technical_features": {
                "close": 10_000,
                "return_5d": 0.04,
                "return_20d": 0.09,
                "return_60d": 0.16,
                "high_breakout_20d": True,
                "low_breakdown_20d": False,
                "ma5": 10_000,
                "ma20": 9_800,
                "ma60": 9_500,
                "ma20_slope": 30,
                "atr_14_pct": 0.035,
            },
        }
    )

    assert scored["decision"] == "buy_candidate"
    assert scored["momentum_score"] > 75
    assert scored["momentum_components"]["weights"] == {
        "relative_strength": 0.45,
        "volume_ratio": 0.25,
        "volatility_breakout": 0.30,
    }


def test_large_cap_requires_five_percent_expected_return_for_top10(monkeypatch):
    monkeypatch.setattr(
        universe_scanner.settings,
        "universe_scanner_large_cap_min_3d_return_bps",
        500,
    )
    candidate = {
        "symbol": "005930",
        "decision": "buy_candidate",
        "current_price": 70000,
        "expected_return": 420,
        "net_edge": 120,
        "edge_reward_risk": {
            "expected_value_after_cost_bps": 40,
            "reward_risk_ratio": 1.8,
        },
    }

    blocked_status, blocked_reason = universe_scanner._execution_status_for_candidate(
        candidate,
        rank=1,
        execution_limit=10,
        hurdle_rate=0,
        entry_gate={"approved": True},
    )
    allowed_status, _ = universe_scanner._execution_status_for_candidate(
        {**candidate, "expected_return": 520},
        rank=1,
        execution_limit=10,
        hurdle_rate=0,
        entry_gate={"approved": True},
    )

    assert blocked_status == "ARCHIVED"
    assert "large-cap requires" in blocked_reason
    assert allowed_status == "READY"


def test_universe_scanner_uses_fast_price_fetch_and_caps_symbol_sleep(monkeypatch):
    calls: list[tuple[str, bool]] = []
    sleeps: list[float] = []
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_symbol_interval_seconds", 60.0)
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_symbol_interval_cap_seconds", 0.01)
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_intraday_enrichment_enabled", False)
    monkeypatch.setattr(universe_scanner.time_module, "sleep", lambda seconds: sleeps.append(seconds))

    def fake_fetch_price_data(symbol, *, include_intraday=True):
        calls.append((symbol, include_intraday))
        return {
            "symbol": symbol,
            "current_price": 10_000,
            "latest_technical_features": {},
        }

    monkeypatch.setattr(universe_scanner, "fetch_price_data", fake_fetch_price_data)

    snapshots = universe_scanner._collect_price_snapshots(
        {"005930": "Samsung", "000660": "SK hynix"}
    )

    assert [item["symbol"] for item in snapshots] == ["005930", "000660"]
    assert calls == [("005930", False), ("000660", False)]
    assert sleeps == [0.01]


def test_universe_scanner_skips_network_enrichment_when_disabled(monkeypatch):
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_news_enrichment_enabled", False)
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_disclosure_enrichment_enabled", False)
    monkeypatch.setattr(
        universe_scanner,
        "search_naver_news",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("news should be skipped")),
    )
    monkeypatch.setattr(
        universe_scanner,
        "fetch_opendart_disclosures",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("disclosure should be skipped")),
    )

    enriched = universe_scanner._enrich_candidate(
        {
            "symbol": "005930",
            "name": "Samsung",
            "score": 80,
            "decision": "buy_candidate",
        }
    )

    assert enriched["news_count"] == 0
    assert enriched["disclosure_count"] == 0
    assert enriched["enrichment_errors"] == []


def test_auto_trading_empty_symbols_runs_universe_scanner(monkeypatch):
    scanned_symbol = AutoTradeSymbolConfig(
        symbol="005930",
        name="Samsung Electronics",
        price=70000,
        decision_price=70000,
        order_price=70000,
    )

    monkeypatch.setattr(
        auto_trading,
        "scan_universe_for_auto_trade",
        lambda req, **kwargs: {
            "status": "success",
            "scan_id": "scan-test",
            "source_symbol_count": 15,
            "snapshot_count": 15,
            "candidate_count": 2,
            "final_count": 1,
            "executable_count": 1,
            "final_candidates": [{"symbol": "005930", "decision": "buy_candidate"}],
            "ready_candidates": [{"symbol": "005930", "decision": "buy_candidate"}],
            "worker_hurdle_rate": 50,
            "active_candidate_symbols": ["005930"],
            "symbols": [scanned_symbol],
        },
    )
    monkeypatch.setattr(
        auto_trading,
        "_run_symbol",
        lambda req, symbol_cfg, session_id=None: {
            "symbol": symbol_cfg.symbol,
            "status": "blocked",
            "message": "test cycle",
        },
    )

    result = auto_trading.run_auto_trading_once(AutoTradeStartRequest())

    assert result["results"][0]["symbol"] == "__universe__"
    assert result["results"][0]["scan_id"] == "scan-test"
    assert result["results"][1]["symbol"] == "005930"


def test_auto_trading_blocks_when_universe_scan_has_too_few_symbols(monkeypatch):
    scanned_symbol = AutoTradeSymbolConfig(
        symbol="005930",
        name="Samsung Electronics",
        price=70000,
        decision_price=70000,
        order_price=70000,
    )
    run_calls: list[str] = []

    monkeypatch.setattr(
        auto_trading,
        "scan_universe_for_auto_trade",
        lambda req, **kwargs: {
            "status": "success",
            "scan_id": "scan-small",
            "source_symbol_count": 14,
            "snapshot_count": 14,
            "candidate_count": 2,
            "final_count": 1,
            "executable_count": 1,
            "final_candidates": [{"symbol": "005930", "decision": "buy_candidate"}],
            "ready_candidates": [{"symbol": "005930", "decision": "buy_candidate"}],
            "worker_hurdle_rate": 50,
            "active_candidate_symbols": ["005930"],
            "symbols": [scanned_symbol],
        },
    )
    monkeypatch.setattr(
        auto_trading,
        "_run_symbol",
        lambda req, symbol_cfg, session_id=None: run_calls.append(symbol_cfg.symbol),
    )

    result = auto_trading.run_auto_trading_once(AutoTradeStartRequest())

    assert result["results"] == [
        {
            "symbol": "__universe__",
            "status": "blocked",
            "scan_id": "scan-small",
            "source_symbol_count": 14,
            "snapshot_count": 14,
            "candidate_count": 2,
            "final_count": 1,
            "executable_count": 1,
            "final_candidates": [{"symbol": "005930", "decision": "buy_candidate"}],
            "ready_candidates": [{"symbol": "005930", "decision": "buy_candidate"}],
            "worker_hurdle_rate": 50,
            "active_candidate_symbols": ["005930"],
            "entry_gate": None,
            "message": (
                "Universe scanner scanned 14 symbols; "
                "at least 15 symbols are required before trading"
            ),
        }
    ]
    assert run_calls == []


def test_universe_scanner_uses_latest_close_as_watch_when_market_closed(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "universe.sqlite3"
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_seed_symbols", "")
    monkeypatch.setattr(universe_scanner.settings, "monitor_watchlist_symbols", "")
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_max_source_symbols", 1)
    monkeypatch.setattr(
        universe_scanner.settings,
        "universe_scanner_symbol_interval_seconds",
        0,
    )
    monkeypatch.setattr(universe_scanner, "_is_kr_regular_market_open", lambda: False)
    monkeypatch.setattr(
        universe_scanner,
        "_paper_average_realized_return_bps",
        lambda: None,
    )
    monkeypatch.setattr(
        universe_scanner,
        "fetch_price_data",
        lambda symbol: {
            "symbol": symbol,
            "current_price": None,
            "change_rate": None,
            "volume": None,
            "volume_ratio": None,
            "daily_candles": [{"date": "20260522", "close": 70000}],
            "source": "test",
        },
    )
    monkeypatch.setattr(
        universe_scanner,
        "search_naver_news",
        lambda query, display=5, sort="date": {"items": []},
    )
    monkeypatch.setattr(
        universe_scanner,
        "fetch_opendart_disclosures",
        lambda symbol, lookback_hours: [],
    )

    result = universe_scanner.scan_universe_for_auto_trade(
        AutoTradeStartRequest(universe_candidate_limit=1, universe_final_limit=1),
        db_path=db_path,
    )

    assert result["final_count"] == 1
    assert result["final_candidates"][0]["decision"] == "watch"
    assert result["final_candidates"][0]["current_price"] == 70000
    assert "market closed" in result["final_candidates"][0]["reason"]


def test_universe_scanner_keeps_only_top_ten_execution_candidates(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "universe.sqlite3"
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_seed_symbols", "")
    monkeypatch.setattr(universe_scanner.settings, "monitor_watchlist_symbols", "")
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_max_source_symbols", 12)
    monkeypatch.setattr(
        universe_scanner.settings,
        "universe_scanner_symbol_interval_seconds",
        0,
    )
    monkeypatch.setattr(
        universe_scanner,
        "_paper_average_realized_return_bps",
        lambda: None,
    )

    def fake_fetch_price_data(symbol: str) -> dict:
        rank_bonus = int(symbol[-2:]) if symbol[-2:].isdigit() else 1
        return {
            "symbol": symbol,
            "current_price": 10000 + rank_bonus,
            "change_rate": 2.0,
            "volume": 2_000_000 + rank_bonus,
            "volume_ratio": 2.5,
            "turnover_value": 80_000_000_000 + rank_bonus,
            "intraday": {"minute_volume_ratio": 2.0},
            "overheated": False,
            "source": "test",
        }

    monkeypatch.setattr(universe_scanner, "fetch_price_data", fake_fetch_price_data)
    monkeypatch.setattr(
        universe_scanner,
        "search_naver_news",
        lambda query, display=5, sort="date": {"items": [{"title": query}]},
    )
    monkeypatch.setattr(
        universe_scanner,
        "fetch_opendart_disclosures",
        lambda symbol, lookback_hours: [],
    )

    result = universe_scanner.scan_universe_for_auto_trade(
        AutoTradeStartRequest(universe_candidate_limit=12, universe_final_limit=10),
        db_path=db_path,
    )

    with sqlite3.connect(db_path) as conn:
        active_count = conn.execute("SELECT COUNT(*) FROM scanner_candidates").fetchone()[0]
        history_count = conn.execute(
            "SELECT COUNT(*) FROM scanner_candidate_history"
        ).fetchone()[0]
        max_active_rank = conn.execute(
            "SELECT MAX(rank) FROM scanner_candidates"
        ).fetchone()[0]
        archived_count = conn.execute(
            "SELECT COUNT(*) FROM scanner_candidate_history WHERE status = 'ARCHIVED'"
        ).fetchone()[0]

    assert result["candidate_count"] == 12
    assert active_count == 10
    assert history_count == 12
    assert max_active_rank == 10
    assert archived_count == 2


def test_initialize_universe_db_backfills_legacy_candidate_history(tmp_path, monkeypatch):
    db_path = tmp_path / "universe.sqlite3"
    monkeypatch.setattr(
        universe_scanner.settings,
        "universe_scanner_worker_hurdle_rate_bps",
        0,
    )
    universe_scanner.initialize_universe_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO universe_scan_runs (
                scan_id, created_at, source_symbol_count, candidate_limit,
                final_limit, status, error, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-scan",
                "2026-05-24T09:00:00",
                1,
                1,
                1,
                "success",
                None,
                "{}",
            ),
        )
        conn.execute(
            """
            INSERT INTO universe_candidates (
                scan_id, created_at, symbol, name, rank, score, decision, reason,
                current_price, change_rate, volume, volume_ratio, turnover_value,
                market_cap, market_segment, universe_profile, news_count,
                disclosure_count, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-scan",
                "2026-05-24T09:00:00",
                "005930",
                "Samsung Electronics",
                1,
                80.0,
                "buy_candidate",
                "legacy candidate",
                70000.0,
                2.0,
                2_000_000,
                2.0,
                90_000_000_000,
                400_000_000_000_000,
                "KOSPI",
                "large_cap",
                0,
                0,
                "{}",
            ),
        )

    universe_scanner.initialize_universe_db(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT symbol, raw_score, expected_return, expected_risk, net_edge,
                   composite_score, rank, status
            FROM scanner_candidate_history
            WHERE scan_id = 'legacy-scan'
            """
        ).fetchone()
        active_count = conn.execute(
            "SELECT COUNT(*) FROM scanner_candidates"
        ).fetchone()[0]

    assert row is not None
    assert row["symbol"] == "005930"
    assert row["raw_score"] == 80.0
    assert row["expected_return"] is not None
    assert row["expected_risk"] is not None
    assert row["net_edge"] is not None
    assert row["composite_score"] is not None
    assert row["rank"] == 1
    assert row["status"] in {"READY", "SKIPPED", "EXCLUDED", "ARCHIVED"}
    assert active_count == 0


def test_universe_scanner_collects_symbols_sequentially(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        universe_scanner.settings,
        "universe_scanner_symbol_interval_seconds",
        0,
    )

    def fake_fetch_price_data(symbol: str) -> dict:
        calls.append(symbol)
        return {"symbol": symbol, "current_price": 1000, "source": "test"}

    monkeypatch.setattr(universe_scanner, "fetch_price_data", fake_fetch_price_data)

    snapshots = universe_scanner._collect_price_snapshots(
        {"005930": "Samsung Electronics", "000660": "SK hynix"}
    )

    assert calls == ["005930", "000660"]
    assert [item["symbol"] for item in snapshots] == ["005930", "000660"]
