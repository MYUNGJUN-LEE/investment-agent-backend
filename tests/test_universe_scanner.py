from __future__ import annotations

import sqlite3

import pytest

from app.models import AutoTradeStartRequest, AutoTradeSymbolConfig
from app.trading import auto_trading
from app.trading.atr_exits import atr_exit_levels_from_price_data
from app.trading import kospi_universe
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


def test_kospi_csv_sources_keep_priority_and_bypass_normal_cap(tmp_path, monkeypatch):
    csv_path = tmp_path / "kospi_symbols.csv"
    csv_path.write_text(
        "symbol,name,market\n"
        "123456,KOSPI One,KOSPI\n"
        "234567,KOSPI Two,KOSPI\n"
        "KOSPI,Index,KOSPI\n"
        "035900,KOSDAQ Duplicate,KOSDAQ\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(universe_scanner.settings, "universe_include_kospi", True)
    monkeypatch.setattr(universe_scanner.settings, "universe_kospi_symbol_source", "csv")
    monkeypatch.setattr(universe_scanner.settings, "universe_kospi_scan_all", True)
    monkeypatch.setattr(universe_scanner.settings, "universe_full_scan_enabled", True)
    monkeypatch.setattr(universe_scanner.settings, "universe_full_scan_max_symbols", 0)
    monkeypatch.setattr(universe_scanner.settings, "universe_kospi_csv_path", str(csv_path))
    monkeypatch.setattr(
        universe_scanner.settings,
        "universe_kospi_cache_path",
        str(tmp_path / "kospi.sqlite3"),
    )
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_max_source_symbols", 1)
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_seed_symbols", "005930")
    monkeypatch.setattr(universe_scanner.settings, "monitor_watchlist_symbols", "000660")

    req = AutoTradeStartRequest(
        symbols=[AutoTradeSymbolConfig(symbol="035900", name="Explicit JYP")],
        universe_seed_symbols=["000660"],
    )
    source_symbols, diagnostics = universe_scanner._resolve_source_symbols_with_diagnostics(req)

    assert list(source_symbols)[:5] == ["035900", "000660", "005930", "123456", "234567"]
    assert diagnostics["max_source_cap_bypassed_for_full_kospi"] is True
    assert diagnostics["source_symbol_count_after_caps"] > 1
    assert diagnostics["kospi_count"] == 2
    assert "KOSPI" not in source_symbols

    legacy_sources = universe_scanner._resolve_source_symbols(req)
    assert legacy_sources["035900"] == "Explicit JYP"
    assert legacy_sources["123456"] == "KOSPI One"


def test_kospi_loader_falls_back_to_cache_when_configured_csv_missing(tmp_path, monkeypatch):
    cache_path = tmp_path / "kospi.sqlite3"
    missing_csv = tmp_path / "missing_kospi_symbols.csv"
    kospi_universe.upsert_kospi_symbols(
        [
            {"symbol": "005930", "name": "Samsung Electronics", "market": "KOSPI"},
            {"symbol": "000660", "name": "SK hynix", "market": "KOSPI"},
        ],
        source="pytest",
        db_path=cache_path,
    )

    monkeypatch.setattr(kospi_universe.settings, "universe_include_kospi", True)
    monkeypatch.setattr(kospi_universe.settings, "universe_kospi_symbol_source", "csv")
    monkeypatch.setattr(kospi_universe.settings, "universe_kospi_csv_path", str(missing_csv))
    monkeypatch.setattr(kospi_universe.settings, "universe_kospi_cache_path", str(cache_path))

    symbols = kospi_universe.load_kospi_symbols(scan_all=True)

    assert [item["symbol"] for item in symbols] == ["000660", "005930"]
    assert {item["market"] for item in symbols} == {"KOSPI"}
    assert {item["source_detail"] for item in symbols} == {"pytest"}


def test_kospi_loader_uses_builtin_fallback_when_csv_and_cache_missing(
    tmp_path,
    monkeypatch,
):
    missing_csv = tmp_path / "missing_kospi_symbols.csv"
    missing_cache = tmp_path / "missing_kospi.sqlite3"
    monkeypatch.setattr(kospi_universe.settings, "universe_include_kospi", True)
    monkeypatch.setattr(kospi_universe.settings, "universe_kospi_symbol_source", "csv")
    monkeypatch.setattr(kospi_universe.settings, "universe_kospi_csv_path", str(missing_csv))
    monkeypatch.setattr(kospi_universe.settings, "universe_kospi_cache_path", str(missing_cache))
    monkeypatch.setattr(
        kospi_universe.settings,
        "universe_kospi_builtin_fallback_enabled",
        True,
    )

    symbols = kospi_universe.load_kospi_symbols(scan_all=True)
    status = kospi_universe.kospi_universe_cache_status()

    assert len(symbols) >= 20
    assert status["status"] == "builtin_fallback"
    assert status["builtin_fallback_enabled"] is True
    assert symbols[0]["source_detail"] == "builtin"
    assert {"005930", "000660", "005380"}.issubset(
        {item["symbol"] for item in symbols}
    )
    assert {item["market"] for item in symbols} == {"KOSPI"}


def test_default_auto_discover_sources_include_kospi_and_kosdaq_without_files(
    tmp_path,
    monkeypatch,
):
    missing_csv = tmp_path / "missing_kospi_symbols.csv"
    missing_cache = tmp_path / "missing_kospi.sqlite3"
    monkeypatch.setattr(universe_scanner.settings, "universe_include_kospi", True)
    monkeypatch.setattr(universe_scanner.settings, "universe_kospi_symbol_source", "csv")
    monkeypatch.setattr(universe_scanner.settings, "universe_kospi_scan_all", True)
    monkeypatch.setattr(
        universe_scanner.settings,
        "universe_kospi_builtin_fallback_enabled",
        True,
    )
    monkeypatch.setattr(universe_scanner.settings, "universe_full_scan_enabled", True)
    monkeypatch.setattr(universe_scanner.settings, "universe_full_scan_max_symbols", 0)
    monkeypatch.setattr(universe_scanner.settings, "universe_kospi_csv_path", str(missing_csv))
    monkeypatch.setattr(universe_scanner.settings, "universe_kospi_cache_path", str(missing_cache))
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_max_source_symbols", 1)
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_seed_symbols", "")
    monkeypatch.setattr(universe_scanner.settings, "monitor_watchlist_symbols", "")

    source_symbols, diagnostics = universe_scanner._resolve_source_symbols_with_diagnostics(
        AutoTradeStartRequest(auto_discover_symbols=True)
    )

    assert diagnostics["kospi_symbols_loaded"] >= 20
    assert diagnostics["kospi_source_symbol_count"] >= 20
    assert diagnostics["kosdaq_source_symbol_count"] == len(universe_scanner.DEFAULT_UNIVERSE)
    assert diagnostics["full_kospi_scan"] is True
    assert diagnostics["max_source_cap_bypassed_for_full_kospi"] is True
    assert diagnostics["source_detail_counts"]["builtin"] >= 20
    assert "005930" in source_symbols
    assert "035900" in source_symbols


def test_auto_discover_scan_applies_builtin_kospi_fallback_to_trade_plan(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "universe.sqlite3"
    missing_csv = tmp_path / "missing_kospi_symbols.csv"
    missing_cache = tmp_path / "missing_kospi.sqlite3"
    builtin_kospi = {
        "005930": "Samsung Electronics",
        "000660": "SK hynix",
    }
    monkeypatch.setattr(kospi_universe, "BUILTIN_KOSPI_UNIVERSE", builtin_kospi)
    monkeypatch.setattr(universe_scanner.settings, "universe_include_kospi", True)
    monkeypatch.setattr(universe_scanner.settings, "universe_kospi_symbol_source", "csv")
    monkeypatch.setattr(universe_scanner.settings, "universe_kospi_scan_all", True)
    monkeypatch.setattr(
        universe_scanner.settings,
        "universe_kospi_builtin_fallback_enabled",
        True,
    )
    monkeypatch.setattr(universe_scanner.settings, "universe_full_scan_enabled", True)
    monkeypatch.setattr(universe_scanner.settings, "universe_full_scan_max_symbols", 4)
    monkeypatch.setattr(universe_scanner.settings, "universe_kospi_csv_path", str(missing_csv))
    monkeypatch.setattr(universe_scanner.settings, "universe_kospi_cache_path", str(missing_cache))
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_candidate_limit", 4)
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_final_limit", 2)
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_symbol_interval_seconds", 0)
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_worker_hurdle_rate_bps", 0)
    monkeypatch.setattr(universe_scanner, "_is_kr_regular_market_open", lambda: True)
    monkeypatch.setattr(universe_scanner, "_paper_average_realized_return_bps", lambda: None)
    monkeypatch.setattr(
        universe_scanner,
        "edge_entry_gate",
        lambda candidates=None: {
            "status": "approved",
            "approved": True,
            "message": "test gate approved",
        },
    )
    monkeypatch.setattr(
        universe_scanner,
        "estimate_expected_edges",
        lambda candidate, raw_score, model=None: {
            "expected_return": 700.0,
            "expected_risk": 90.0,
            "edge_model": "pytest",
        },
    )

    def fake_fetch_price_data(symbol: str, *, include_intraday=True) -> dict:
        is_kospi = symbol in builtin_kospi
        price = 70_000 if is_kospi else 30_000
        return {
            "symbol": symbol,
            "name": builtin_kospi.get(symbol) or universe_scanner.DEFAULT_UNIVERSE.get(symbol),
            "current_price": price,
            "change_rate": 2.4 if is_kospi else 1.8,
            "volume": 3_000_000,
            "volume_ratio": 2.2,
            "turnover_value": 90_000_000_000,
            "market_cap": 1_000_000_000_000,
            "market_segment": "KOSPI" if is_kospi else "KOSDAQ",
            "intraday": {"minute_volume_ratio": 1.6},
            "latest_technical_features": {
                "close": price,
                "return_5d": 0.04,
                "return_20d": 0.08,
                "return_60d": 0.15,
                "high_breakout_20d": True,
                "low_breakdown_20d": False,
                "ma5": price,
                "ma20": price * 0.98,
                "ma60": price * 0.94,
                "ma20_slope": 120,
                "atr_14": price * 0.025,
                "atr_14_pct": 0.025,
            },
            "technical_features": [
                {"close": price * 0.99, "atr_14": price * 0.025},
                {"close": price, "atr_14": price * 0.025},
            ],
            "overheated": False,
            "source": "pytest",
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
        AutoTradeStartRequest(
            auto_discover_symbols=True,
            universe_candidate_limit=4,
            universe_final_limit=2,
        ),
        db_path=db_path,
    )

    assert result["kospi_source_symbol_count"] == 2
    assert result["kosdaq_source_symbol_count"] == 2
    assert result["kospi_snapshot_count"] == 2
    assert result["kosdaq_snapshot_count"] == 2
    assert result["symbols"]
    assert {item["market_segment"] for item in result["final_candidates"]}.issubset(
        {"KOSPI", "KOSDAQ"}
    )
    assert any(item["market_segment"] == "KOSPI" for item in result["candidates"])


def test_auto_discover_scan_includes_known_kospi_common_stocks(tmp_path, monkeypatch):
    db_path = tmp_path / "universe.sqlite3"
    csv_path = tmp_path / "kospi_symbols.csv"
    known_kospi = {
        "005930": "Samsung Electronics",
        "000660": "SK hynix",
        "005380": "Hyundai Motor",
        "035420": "NAVER",
        "051910": "LG Chem",
        "068270": "Celltrion",
    }
    csv_path.write_text(
        "symbol,name,market\n"
        + "".join(f"{symbol},{name},KOSPI\n" for symbol, name in known_kospi.items()),
        encoding="utf-8",
    )

    monkeypatch.setattr(universe_scanner.settings, "universe_include_kospi", True)
    monkeypatch.setattr(universe_scanner.settings, "universe_kospi_symbol_source", "csv")
    monkeypatch.setattr(universe_scanner.settings, "universe_kospi_scan_all", True)
    monkeypatch.setattr(universe_scanner.settings, "universe_full_scan_enabled", True)
    monkeypatch.setattr(universe_scanner.settings, "universe_full_scan_batch_size", 3)
    monkeypatch.setattr(universe_scanner.settings, "universe_full_scan_batch_pause_seconds", 0)
    monkeypatch.setattr(universe_scanner.settings, "universe_kospi_csv_path", str(csv_path))
    monkeypatch.setattr(universe_scanner.settings, "universe_kospi_cache_path", str(tmp_path / "kospi.sqlite3"))
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_max_source_symbols", 1)
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_candidate_limit", 6)
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_final_limit", 3)
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_symbol_interval_seconds", 0)
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_worker_hurdle_rate_bps", 0)
    monkeypatch.setattr(universe_scanner, "_is_kr_regular_market_open", lambda: True)
    monkeypatch.setattr(universe_scanner, "_paper_average_realized_return_bps", lambda: None)
    monkeypatch.setattr(
        universe_scanner,
        "edge_entry_gate",
        lambda candidates=None: {
            "status": "approved",
            "approved": True,
            "message": "test gate approved",
        },
    )
    monkeypatch.setattr(
        universe_scanner,
        "estimate_expected_edges",
        lambda candidate, raw_score, model=None: {
            "expected_return": 360.0,
            "expected_risk": 80.0,
            "edge_model": "pytest",
        },
    )

    def fake_fetch_price_data(symbol: str, *, include_intraday=True) -> dict:
        if symbol in known_kospi:
            price = 70_000
            return {
                "symbol": symbol,
                "name": known_kospi[symbol],
                "current_price": price,
                "change_rate": 2.0,
                "volume": 5_000_000,
                "volume_ratio": 2.5,
                "turnover_value": 350_000_000_000,
                "market_cap": 300_000_000_000_000,
                "market_segment": "KOSPI",
                "intraday": {"minute_volume_ratio": 1.8},
                "latest_technical_features": {
                    "close": price,
                    "return_5d": 0.04,
                    "return_20d": 0.09,
                    "return_60d": 0.16,
                    "high_breakout_20d": True,
                    "low_breakdown_20d": False,
                    "ma5": price,
                    "ma20": price * 0.98,
                    "ma60": price * 0.94,
                    "ma20_slope": 120,
                    "atr_14": 1_800,
                    "atr_14_pct": 1_800 / price,
                },
                "technical_features": [
                    {"close": price * 0.99, "atr_14": 1_800},
                    {"close": price, "atr_14": 1_800},
                ],
                "overheated": False,
                "source": "pytest",
            }
        return {
            "symbol": symbol,
            "current_price": 10_000,
            "change_rate": 0.1,
            "volume": 1_000,
            "volume_ratio": 0.2,
            "turnover_value": 10_000_000,
            "market_cap": 500_000_000_000,
            "market_segment": "KOSDAQ",
            "latest_technical_features": {},
            "overheated": False,
            "source": "pytest",
        }

    monkeypatch.setattr(universe_scanner, "fetch_price_data", fake_fetch_price_data)

    result = universe_scanner.scan_universe_for_auto_trade(
        AutoTradeStartRequest(
            auto_discover_symbols=True,
            universe_candidate_limit=6,
            universe_final_limit=3,
        ),
        db_path=db_path,
    )

    assert result["kospi_source_symbol_count"] == len(known_kospi)
    assert result["kospi_snapshot_count"] == len(known_kospi)
    assert result["kospi_candidate_count"] == len(known_kospi)
    assert result["kospi_final_candidate_count"] == 3
    assert result["kosdaq_source_symbol_count"] >= 1
    assert result["final_count"] == 3
    assert {
        item["symbol"]
        for item in result["final_candidates"]
    }.issubset(set(known_kospi))

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        snapshot_rows = conn.execute(
            """
            SELECT symbol, market_segment
            FROM universe_price_snapshots
            WHERE symbol IN (?, ?, ?, ?, ?, ?)
            ORDER BY symbol
            """,
            tuple(sorted(known_kospi)),
        ).fetchall()
        candidate_rows = conn.execute(
            """
            SELECT symbol, market_segment
            FROM universe_candidates
            WHERE symbol IN (?, ?, ?, ?, ?, ?)
            ORDER BY symbol
            """,
            tuple(sorted(known_kospi)),
        ).fetchall()

    assert {row["symbol"] for row in snapshot_rows} == set(known_kospi)
    assert {row["market_segment"] for row in snapshot_rows} == {"KOSPI"}
    assert {row["symbol"] for row in candidate_rows} == set(known_kospi)
    assert {row["market_segment"] for row in candidate_rows} == {"KOSPI"}


def test_universe_scanner_scores_stores_and_returns_final_symbols(tmp_path, monkeypatch):
    db_path = tmp_path / "universe.sqlite3"
    monkeypatch.setattr(universe_scanner.settings, "universe_include_kospi", False)
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


def test_scanner_decision_thresholds_use_config(monkeypatch):
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_buy_score_threshold", 70.0)
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_watch_score_threshold", 40.0)

    assert universe_scanner._score_decision(70.0) == "buy_candidate"
    assert universe_scanner._score_decision(43.0) == "watch"
    assert universe_scanner._score_decision(39.99) == "exclude"


def test_paper_bootstrap_soft_pass_collecting_gate_only(monkeypatch):
    monkeypatch.setattr(
        universe_scanner.settings,
        "universe_scanner_paper_bootstrap_soft_pass_enabled",
        True,
    )
    candidate = {
        "symbol": "123456",
        "decision": "buy_candidate",
        "current_price": 10_000,
        "score": 60.0,
        "composite_score": 60.0,
        "net_edge": 25.0,
        "expected_return": 250.0,
        "edge_reward_risk": {
            "expected_value_after_cost_bps": 10.0,
            "reward_risk_ratio": 1.0,
        },
        "large_cap_top10_gate": {"passed": True},
    }
    collecting_gate = {
        "status": "blocked",
        "approved": False,
        "message": "Calibration performance gate blocked entries: sample_count 0/600",
    }
    mature_blocked_gate = {
        "status": "blocked",
        "approved": False,
        "message": "Calibration performance gate blocked entries: recent_ic -0.10 < 0.02",
    }

    paper_status, paper_reason = universe_scanner._execution_status_for_candidate(
        candidate,
        rank=1,
        execution_limit=10,
        hurdle_rate=40.0,
        entry_gate=collecting_gate,
        execution_mode="paper",
    )
    live_status, _ = universe_scanner._execution_status_for_candidate(
        candidate,
        rank=1,
        execution_limit=10,
        hurdle_rate=0.0,
        entry_gate=collecting_gate,
        execution_mode="live",
    )
    mature_status, _ = universe_scanner._execution_status_for_candidate(
        candidate,
        rank=1,
        execution_limit=10,
        hurdle_rate=40.0,
        entry_gate=mature_blocked_gate,
        execution_mode="paper",
    )

    assert paper_status == "READY"
    assert "paper bootstrap soft-pass for collecting calibration gate" in paper_reason
    assert live_status == "SKIPPED"
    assert mature_status == "SKIPPED"


def test_broker_paper_bootstrap_observe_only_candidate_label_gate(monkeypatch):
    candidate = {
        "symbol": "123456",
        "decision": "buy_candidate",
        "current_price": 10_000,
        "score": 70.0,
        "composite_score": 70.0,
        "net_edge": 120.0,
        "expected_return": 250.0,
        "edge_reward_risk": {
            "expected_value_after_cost_bps": 40.0,
            "reward_risk_ratio": 2.0,
        },
        "large_cap_top10_gate": {"passed": True},
    }
    observe_only_gate = {
        "status": "bootstrap_observe_only",
        "approved": True,
        "message": (
            "Candidate label calibration gate failed, but broker_paper "
            "bootstrap observe-only mode allowed entry because "
            "broker_paper_fill_sample_count 0/200."
        ),
        "broker_paper_bootstrap_allowed": True,
        "broker_paper_candidate_label_gate_mode": "observe_only",
        "candidate_label_gate_failed": True,
        "candidate_label_gate_hard_blocking": False,
        "broker_paper_fill_gate_blocked": False,
        "calibration_gate_mode": (
            "broker_paper_bootstrap_candidate_label_observe_only"
        ),
    }
    hard_block_gate = {
        **observe_only_gate,
        "approved": False,
        "candidate_label_gate_hard_blocking": True,
        "broker_paper_bootstrap_allowed": False,
    }

    status, reason = universe_scanner._execution_status_for_candidate(
        candidate,
        rank=1,
        execution_limit=10,
        hurdle_rate=40.0,
        entry_gate=observe_only_gate,
        execution_mode="broker_paper",
    )
    blocked_status, _ = universe_scanner._execution_status_for_candidate(
        candidate,
        rank=1,
        execution_limit=10,
        hurdle_rate=40.0,
        entry_gate=hard_block_gate,
        execution_mode="broker_paper",
    )

    assert status == "READY"
    assert "broker_paper bootstrap observe-only candidate-label calibration" in reason
    assert "broker_paper executable" in reason
    assert blocked_status == "SKIPPED"


def test_broker_paper_observe_only_final_candidate_path_counts_executable(monkeypatch):
    monkeypatch.setattr(
        universe_scanner,
        "market_safety_check",
        lambda symbol, execution_mode="paper": {
            "block": False,
            "penalty_bps": 0.0,
            "message": "ok",
        },
    )
    monkeypatch.setattr(
        universe_scanner,
        "corporate_event_check",
        lambda symbol, execution_mode="paper": {
            "block": False,
            "penalty_bps": 0.0,
            "message": "ok",
        },
    )
    monkeypatch.setattr(universe_scanner, "load_edge_model", lambda: None)
    monkeypatch.setattr(
        universe_scanner,
        "estimate_expected_edges",
        lambda candidate, raw_score, model=None: {
            "expected_return": 360.0,
            "expected_risk": 60.0,
            "edge_model": "pytest",
        },
    )
    monkeypatch.setattr(
        universe_scanner,
        "edge_entry_gate",
        lambda candidates=None, execution_mode=None: {
            "status": "bootstrap_observe_only",
            "approved": True,
            "message": (
                "Candidate label calibration gate failed, but broker_paper "
                "bootstrap observe-only mode allowed entry because "
                "broker_paper_fill_sample_count 0/200."
            ),
            "broker_paper_bootstrap_allowed": True,
            "broker_paper_candidate_label_gate_mode": "observe_only",
            "candidate_label_gate_failed": True,
            "candidate_label_gate_hard_blocking": False,
            "broker_paper_fill_sample_count": 0,
            "broker_paper_min_fill_samples": 200,
            "broker_paper_fill_gate_blocked": False,
            "calibration_gate_mode": (
                "broker_paper_bootstrap_candidate_label_observe_only"
            ),
        },
    )

    ranked = universe_scanner._rank_execution_candidates(
        [
            {
                "symbol": "319660",
                "name": "PSK",
                "score": 92.0,
                "decision": "buy_candidate",
                "current_price": 20_000,
                "change_rate": 3.0,
                "volume_ratio": 3.0,
                "turnover_value": 80_000_000_000,
                "latest_technical_features": {
                    "close": 20_000,
                    "return_5d": 0.05,
                    "return_20d": 0.10,
                    "return_60d": 0.15,
                    "high_breakout_20d": True,
                    "atr_14": 600,
                    "atr_14_pct": 0.03,
                    "ma5": 20_000,
                    "ma20": 19_000,
                    "ma60": 18_500,
                    "ma20_slope": 120,
                },
                "intraday": {"minute_volume_ratio": 2.0},
                "overheated": False,
            }
        ],
        scan_time="2026-06-10T09:00:00",
        final_limit=10,
        hurdle_rate=40.0,
        execution_mode="broker_paper",
    )
    executable_count = len([item for item in ranked if item["status"] == "READY"])

    assert ranked[0]["symbol"] == "319660"
    assert ranked[0]["status"] == "READY"
    assert ranked[0]["broker_paper_executable"] is True
    assert ranked[0]["candidate_label_calibration_gate_hard_blocking"] is False
    assert ranked[0]["broker_paper_fill_sample_count"] == 0
    assert "broker_paper bootstrap observe-only" in ranked[0]["reason"]
    assert executable_count == 1


def test_latest_universe_scan_broker_paper_overlay_marks_stale_psk_ready(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "universe.sqlite3"
    monkeypatch.setattr(
        universe_scanner,
        "edge_entry_gate",
        lambda candidates=None, execution_mode=None: {
            "status": "bootstrap_observe_only",
            "approved": True,
            "message": (
                "Candidate label calibration gate failed, but broker_paper "
                "bootstrap observe-only mode allowed entry because "
                "broker_paper_fill_sample_count 0/200."
            ),
            "broker_paper_bootstrap_allowed": True,
            "broker_paper_candidate_label_gate_mode": "observe_only",
            "candidate_label_gate_failed": True,
            "candidate_label_gate_hard_blocking": False,
            "broker_paper_fill_sample_count": 0,
            "broker_paper_min_fill_samples": 200,
            "broker_paper_fill_gate_blocked": False,
            "calibration_gate_mode": (
                "broker_paper_bootstrap_candidate_label_observe_only"
            ),
        },
    )
    stale_psk = {
        "symbol": "319660",
        "name": "PSK",
        "rank": 1,
        "score": 92.0,
        "raw_score": 92.0,
        "expected_return": 360.0,
        "expected_risk": 60.0,
        "trading_cost": 20.0,
        "slippage_cost": 10.0,
        "net_edge": 330.0,
        "composite_score": 92.0,
        "decision": "buy_candidate",
        "status": "SKIPPED",
        "reason": (
            "composite momentum; Calibration performance gate blocked entries: "
            "sample_count 281/600"
        ),
        "current_price": 20_000,
        "change_rate": 3.0,
        "volume": 1_000_000,
        "volume_ratio": 2.0,
        "turnover_value": 80_000_000_000,
        "expires_at": "2026-06-10T15:00:00",
        "large_cap_top10_gate": {"passed": True},
        "edge_reward_risk": {
            "expected_value_after_cost_bps": 100.0,
            "reward_risk_ratio": 2.0,
        },
    }
    universe_scanner.initialize_universe_db(db_path)
    result = {
        "scan_id": "scan-pytest-stale-psk",
        "created_at": "2026-06-10T09:00:00",
        "source_symbol_count": 1,
        "candidate_limit": 1,
        "final_limit": 1,
        "status": "success",
        "candidate_count": 1,
        "final_count": 1,
        "executable_count": 0,
        "candidates": [stale_psk],
        "final_candidates": [stale_psk],
        "ready_candidates": [],
        "symbols": [],
    }
    universe_scanner._record_scan_run(db_path, result)
    universe_scanner._record_scanner_candidates(
        path=db_path,
        scan_id=result["scan_id"],
        scan_time=result["created_at"],
        candidates=[stale_psk],
        execution_limit=1,
    )

    stored = universe_scanner.get_latest_universe_scan(db_path)
    overlaid = universe_scanner.get_latest_universe_scan(
        db_path,
        execution_mode="broker_paper",
    )

    assert stored["executable_count"] == 0
    assert stored["final_candidates"][0]["status"] == "SKIPPED"
    assert overlaid["stored_executable_count"] == 0
    assert overlaid["executable_count"] == 1
    assert overlaid["ready_candidates"][0]["symbol"] == "319660"
    assert overlaid["final_candidates"][0]["status"] == "READY"
    assert overlaid["final_candidates"][0]["broker_paper_executable"] is True
    assert (
        overlaid["final_candidates"][0]["calibration_gate_mode"]
        == "broker_paper_bootstrap_candidate_label_observe_only"
    )
    assert "broker_paper executable" in overlaid["final_candidates"][0]["reason"]


def test_paper_promotes_safe_exclude_to_watch_only(monkeypatch):
    monkeypatch.setattr(
        universe_scanner.settings,
        "universe_scanner_paper_promote_exclude_to_watch_enabled",
        True,
    )
    candidate = {
        "symbol": "123456",
        "decision": "exclude",
        "current_price": 10_000,
        "score": 43.0,
        "composite_score": 43.0,
        "net_edge": 135.0,
        "large_cap_top10_gate": {"passed": True},
    }

    promoted = universe_scanner._paper_promote_exclude_to_watch_if_eligible(
        candidate,
        hurdle_rate=40.0,
        execution_mode="paper",
    )
    live = universe_scanner._paper_promote_exclude_to_watch_if_eligible(
        candidate,
        hurdle_rate=40.0,
        execution_mode="live",
    )
    safety_blocked = universe_scanner._paper_promote_exclude_to_watch_if_eligible(
        {
            **candidate,
            "market_safety": {"block": True},
            "status": "EXCLUDED",
        },
        hurdle_rate=40.0,
        execution_mode="paper",
    )
    large_cap_blocked = universe_scanner._paper_promote_exclude_to_watch_if_eligible(
        {
            **candidate,
            "large_cap_top10_gate": {"passed": False},
        },
        hurdle_rate=40.0,
        execution_mode="paper",
    )

    assert promoted["decision"] == "watch"
    assert promoted["paper_promoted_to_watch"] is True
    assert live["decision"] == "exclude"
    assert safety_blocked["decision"] == "exclude"
    assert large_cap_blocked["decision"] == "exclude"


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

    snapshots, collection = universe_scanner._collect_price_snapshots(
        {"005930": "Samsung", "000660": "SK hynix"}
    )

    assert [item["symbol"] for item in snapshots] == ["005930", "000660"]
    assert collection["timed_out"] is False
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
            "scanner_is_stale": False,
            "latest_scan_age_seconds": None,
            "scanner_stale_after_seconds": 900,
            "last_scanner_recovery_attempt_at": None,
            "last_scanner_recovery_status": "not_needed",
            "last_scanner_recovery_error": None,
            "message": (
                "Universe scanner scanned 14 symbols; "
                "at least 15 symbols are required before trading"
            ),
        }
    ]
    assert run_calls == []


def test_universe_scanner_excludes_missing_current_price_before_scoring(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "universe.sqlite3"
    monkeypatch.setattr(universe_scanner.settings, "universe_include_kospi", False)
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

    assert result["final_count"] == 0
    assert result["cleaning_excluded_count"] == 1
    assert result["cleaning_excluded"][0]["symbol"] == "035900"
    assert result["cleaning_excluded"][0]["reasons"] == [
        "current_price missing or non-positive"
    ]


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


def test_universe_scanner_caps_fresh_quotes_and_rotates_cached_prefilter(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "universe.sqlite3"
    seed_symbols = [f"10{index:04d}" for index in range(1, 7)]
    calls: list[str] = []

    monkeypatch.setattr(universe_scanner.settings, "universe_include_kospi", False)
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_seed_symbols", "")
    monkeypatch.setattr(universe_scanner.settings, "monitor_watchlist_symbols", "")
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_max_source_symbols", 20)
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_cached_prefilter_enabled", True)
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_prefilter_max_symbols", 6)
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_max_fresh_quote_symbols", 2)
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_cached_only_candidate_limit", 10)
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_rotation_enabled", True)
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_rotation_window_seconds", 1)
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_min_scanned_symbols_for_trading", 0)
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_worker_hurdle_rate_bps", 0)
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_symbol_interval_seconds", 0)
    monkeypatch.setattr(universe_scanner.settings, "universe_cleaning_max_snapshot_age_seconds", 0)
    monkeypatch.setattr(universe_scanner, "_paper_average_realized_return_bps", lambda: None)
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

    universe_scanner.initialize_universe_db(db_path)
    universe_scanner._record_snapshots(
        db_path,
        "cached-scan",
        "2026-06-01T09:00:00",
        [
            {
                "symbol": symbol,
                "name": f"Cached {symbol}",
                "current_price": 10_000 + index,
                "change_rate": 2.0,
                "volume": 2_000_000,
                "volume_ratio": 2.0,
                "turnover_value": 80_000_000_000,
                "market_cap": 500_000_000_000,
                "market_segment": "KOSDAQ",
                "source": "cached_fixture",
            }
            for index, symbol in enumerate(seed_symbols, start=1)
        ],
    )

    def fake_fetch_price_data(symbol: str, *, include_intraday=True) -> dict:
        calls.append(symbol)
        return {
            "symbol": symbol,
            "name": f"Fresh {symbol}",
            "current_price": 11_000 + int(symbol[-2:]),
            "change_rate": 3.0,
            "volume": 3_000_000,
            "volume_ratio": 2.5,
            "turnover_value": 90_000_000_000,
            "market_cap": 600_000_000_000,
            "market_segment": "KOSDAQ",
            "intraday": {"minute_volume_ratio": 2.0},
            "source": "fresh_fixture",
        }

    monkeypatch.setattr(universe_scanner, "fetch_price_data", fake_fetch_price_data)
    req = AutoTradeStartRequest(
        universe_seed_symbols=seed_symbols,
        universe_candidate_limit=6,
        universe_final_limit=6,
    )

    first = universe_scanner.scan_universe_for_auto_trade(req, db_path=db_path)
    first_calls = calls[:]
    universe_scanner._write_scanner_metadata(
        db_path,
        "fresh_quote_rotation",
        {
            "rotation_offset": 0,
            "rotation_epoch": "2000-01-01T00:00:00",
            "rotation_window_seconds": 1,
            "total": 6,
            "cap": 2,
        },
    )
    second = universe_scanner.scan_universe_for_auto_trade(req, db_path=db_path)
    second_calls = calls[len(first_calls):]

    assert first["prefilter_evaluated_count"] == 6
    assert first["fresh_quote_cap"] == 2
    assert first["fresh_quote_requested_count"] == 2
    assert first["fresh_quote_used_count"] == 2
    assert first["cached_only_evaluated_count"] >= 1
    assert first["korea_api_call_count_estimate"] == 2
    assert len(first_calls) == 2
    assert len(second_calls) == 2
    assert set(second_calls) != set(first_calls)
    assert second["rotation_offset"] == 2
    assert any(
        item.get("fresh_quote_used") is False
        and item.get("status") == "SKIPPED"
        and "deferred_due_to_fresh_quote_cap" in str(item.get("reason"))
        for item in first["final_candidates"]
    )


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

    snapshots, collection = universe_scanner._collect_price_snapshots(
        {"005930": "Samsung Electronics", "000660": "SK hynix"}
    )

    assert calls == ["005930", "000660"]
    assert collection["timed_out"] is False
    assert [item["symbol"] for item in snapshots] == ["005930", "000660"]


def test_universe_scanner_stops_collection_at_scan_deadline(monkeypatch):
    clock = {"value": 0.0}
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_max_scan_seconds", 300)
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_symbol_interval_seconds", 0)
    monkeypatch.setattr(universe_scanner.time_module, "monotonic", lambda: clock["value"])

    def fake_fetch_price_data(symbol: str) -> dict:
        clock["value"] += 61
        return {"symbol": symbol, "current_price": 1000, "source": "test"}

    monkeypatch.setattr(universe_scanner, "fetch_price_data", fake_fetch_price_data)

    snapshots, collection = universe_scanner._collect_price_snapshots(
        {
            "000001": "One",
            "000002": "Two",
            "000003": "Three",
            "000004": "Four",
            "000005": "Five",
            "000006": "Six",
        }
    )

    assert [item["symbol"] for item in snapshots] == [
        "000001",
        "000002",
        "000003",
        "000004",
        "000005",
    ]
    assert collection["timed_out"] is True
    assert "before symbol 000006" in collection["message"]
    assert "stored 5/6 snapshots" in collection["message"]
