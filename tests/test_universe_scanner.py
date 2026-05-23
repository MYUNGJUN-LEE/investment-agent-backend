from __future__ import annotations

from app.models import AutoTradeStartRequest, AutoTradeSymbolConfig
from app.trading import auto_trading
from app.trading import universe_scanner


def test_universe_scanner_scores_stores_and_returns_final_symbols(tmp_path, monkeypatch):
    db_path = tmp_path / "universe.sqlite3"
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_seed_symbols", "")
    monkeypatch.setattr(universe_scanner.settings, "monitor_watchlist_symbols", "")
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_max_source_symbols", 3)

    price_rows = {
        "005930": {
            "symbol": "005930",
            "current_price": 70000,
            "change_rate": 2.1,
            "volume": 10_000_000,
            "volume_ratio": 2.4,
            "turnover_value": 700_000_000_000,
            "intraday": {"minute_volume_ratio": 2.0},
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
    assert result["symbols"][0].stop_loss == 70000 * 0.97
    assert latest["scan_id"] == result["scan_id"]
    assert latest["candidates"][0]["symbol"] == "005930"


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
        lambda req: {
            "status": "success",
            "scan_id": "scan-test",
            "source_symbol_count": 3,
            "candidate_count": 2,
            "final_count": 1,
            "final_candidates": [{"symbol": "005930", "decision": "buy_candidate"}],
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


def test_universe_scanner_uses_latest_close_as_watch_when_market_closed(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "universe.sqlite3"
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_seed_symbols", "")
    monkeypatch.setattr(universe_scanner.settings, "monitor_watchlist_symbols", "")
    monkeypatch.setattr(universe_scanner.settings, "universe_scanner_max_source_symbols", 1)
    monkeypatch.setattr(universe_scanner, "_is_kr_regular_market_open", lambda: False)
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
