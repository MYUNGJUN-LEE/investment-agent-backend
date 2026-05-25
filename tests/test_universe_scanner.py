from __future__ import annotations

import sqlite3

from app.models import AutoTradeStartRequest, AutoTradeSymbolConfig
from app.trading import auto_trading
from app.trading import universe_scanner


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
        "005930": {
            "symbol": "005930",
            "current_price": 70000,
            "change_rate": 2.1,
            "volume": 10_000_000,
            "volume_ratio": 2.4,
            "turnover_value": 700_000_000_000,
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
        "000660": {
            "symbol": "000660",
            "current_price": 180000,
            "change_rate": 8.5,
            "volume": 3_000_000,
            "volume_ratio": 1.6,
            "turnover_value": 540_000_000_000,
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
        "373220": {
            "symbol": "373220",
            "current_price": 350000,
            "change_rate": -6.0,
            "volume": 100_000,
            "volume_ratio": 0.4,
            "turnover_value": 35_000_000_000,
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
        universe_candidate_limit=2,
        universe_final_limit=1,
    )

    result = universe_scanner.scan_universe_for_auto_trade(req, db_path=db_path)
    latest = universe_scanner.get_latest_universe_scan(db_path=db_path)

    assert result["status"] == "success"
    assert result["candidate_count"] == 2
    assert result["final_count"] == 1
    assert result["symbols"][0].symbol == "005930"
    assert result["symbols"][0].account_equity == 10_000_000
    assert result["symbols"][0].risk_per_trade == 0.005
    assert result["symbols"][0].stop_loss == 70000 - 1.8 * 1000
    assert result["symbols"][0].take_profit == 70000 + 2.5 * (1.8 * 1000)
    assert result["symbols"][0].trailing_stop == 70500 - 2.0 * 1000
    assert result["symbols"][0].strategy_type == "swing"
    assert result["symbols"][0].expected_gross_edge_bps is not None
    assert latest["scan_id"] == result["scan_id"]
    assert latest["candidates"][0]["symbol"] == "005930"
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
