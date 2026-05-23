from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import sqlite3
from typing import Any


DEFAULT_MARKET_DB_PATH = Path("data/market_data.sqlite3")


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS price_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    current_price REAL,
    change_rate REAL,
    volume INTEGER,
    volume_ratio REAL,
    turnover_value REAL,
    trend TEXT,
    minute_momentum_pct REAL,
    execution_strength REAL,
    orderbook_imbalance REAL,
    spread_pct REAL,
    smart_money_net_buy REAL,
    smart_money_net_buy_5d REAL,
    intraday_score REAL,
    raw_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS news_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    query TEXT NOT NULL,
    source TEXT NOT NULL,
    published_at TEXT,
    title TEXT NOT NULL,
    url TEXT,
    impact_direction TEXT NOT NULL,
    impact_strength REAL NOT NULL,
    raw_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS disclosure_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    source TEXT NOT NULL,
    published_at TEXT,
    title TEXT NOT NULL,
    url TEXT,
    impact_direction TEXT NOT NULL,
    impact_strength REAL NOT NULL,
    raw_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS financial_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    business_year TEXT,
    revenue REAL,
    operating_income REAL,
    net_income REAL,
    total_assets REAL,
    total_liabilities REAL,
    total_equity REAL,
    operating_margin REAL,
    net_margin REAL,
    roe REAL,
    debt_ratio REAL,
    raw_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS market_context_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    source TEXT NOT NULL,
    kospi_close REAL,
    kospi_change_pct REAL,
    kospi_ma20_slope REAL,
    kospi_above_ma20 INTEGER,
    kosdaq_close REAL,
    kosdaq_change_pct REAL,
    kosdaq_ma20_slope REAL,
    kosdaq_above_ma20 INTEGER,
    usdkrw REAL,
    usdkrw_change_pct REAL,
    vix_close REAL,
    vix_change_pct REAL,
    korea_rate REAL,
    us_rate REAL,
    rate_spread REAL,
    market_regime TEXT NOT NULL,
    risk_on_score REAL,
    sector_relative_strength_json TEXT NOT NULL,
    raw_json TEXT NOT NULL
);
"""


def initialize_market_db(db_path: Path | str | None = None) -> None:
    path = Path(db_path) if db_path else DEFAULT_MARKET_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
        _ensure_price_snapshot_columns(conn)
        _ensure_market_context_columns(conn)


def record_price_snapshot(
    price_data: dict[str, Any],
    db_path: Path | str | None = None,
) -> None:
    path = _prepare_db(db_path)
    intraday = price_data.get("intraday") or {}
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO price_snapshots (
                created_at, symbol, current_price, change_rate, volume,
                volume_ratio, turnover_value, trend, minute_momentum_pct,
                execution_strength, orderbook_imbalance, spread_pct,
                smart_money_net_buy, smart_money_net_buy_5d, intraday_score,
                raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now(),
                price_data.get("symbol"),
                price_data.get("current_price"),
                price_data.get("change_rate"),
                price_data.get("volume"),
                price_data.get("volume_ratio"),
                price_data.get("turnover_value"),
                price_data.get("trend"),
                intraday.get("minute_momentum_pct"),
                intraday.get("execution_strength"),
                intraday.get("orderbook_imbalance"),
                intraday.get("spread_pct"),
                intraday.get("smart_money_net_buy"),
                intraday.get("smart_money_net_buy_5d"),
                intraday.get("intraday_score"),
                _json(price_data),
            ),
        )


def record_news_events(
    query: str,
    events: list[dict[str, Any]],
    db_path: Path | str | None = None,
) -> None:
    if not events:
        return
    path = _prepare_db(db_path)
    with sqlite3.connect(path) as conn:
        conn.executemany(
            """
            INSERT INTO news_events (
                created_at, query, source, published_at, title, url,
                impact_direction, impact_strength, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    _now(),
                    query,
                    event.get("source", "Naver Search API"),
                    event.get("date") or event.get("published_at"),
                    event.get("title", ""),
                    event.get("url"),
                    event.get("impact_direction", "uncertain"),
                    event.get("impact_strength", 0),
                    _json(event),
                )
                for event in events
            ],
        )


def record_disclosure_events(
    symbol: str,
    events: list[dict[str, Any]],
    db_path: Path | str | None = None,
) -> None:
    if not events:
        return
    path = _prepare_db(db_path)
    with sqlite3.connect(path) as conn:
        conn.executemany(
            """
            INSERT INTO disclosure_events (
                created_at, symbol, source, published_at, title, url,
                impact_direction, impact_strength, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    _now(),
                    symbol,
                    event.get("source", "OpenDART"),
                    event.get("date"),
                    event.get("title", ""),
                    event.get("url"),
                    event.get("impact_direction", "uncertain"),
                    event.get("impact_strength", 0),
                    _json(event),
                )
                for event in events
            ],
        )


def record_financial_snapshot(
    symbol: str,
    financial_data: dict[str, Any],
    db_path: Path | str | None = None,
) -> None:
    metrics = financial_data.get("metrics") or {}
    if not metrics:
        return
    path = _prepare_db(db_path)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO financial_snapshots (
                created_at, symbol, business_year, revenue, operating_income,
                net_income, total_assets, total_liabilities, total_equity,
                operating_margin, net_margin, roe, debt_ratio, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now(),
                symbol,
                financial_data.get("business_year"),
                metrics.get("revenue"),
                metrics.get("operating_income"),
                metrics.get("net_income"),
                metrics.get("total_assets"),
                metrics.get("total_liabilities"),
                metrics.get("total_equity"),
                metrics.get("operating_margin"),
                metrics.get("net_margin"),
                metrics.get("roe"),
                metrics.get("debt_ratio"),
                _json(financial_data),
            ),
        )


def record_market_context_snapshot(
    context: dict[str, Any],
    db_path: Path | str | None = None,
) -> int:
    path = _prepare_db(db_path)
    kospi = context.get("kospi") or {}
    kosdaq = context.get("kosdaq") or {}
    usdkrw = context.get("usdkrw") or {}
    vix = context.get("vix") or {}
    rates = context.get("rates") or {}
    sector_relative_strength = context.get("sector_relative_strength") or {}

    with sqlite3.connect(path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO market_context_snapshots (
                created_at, trade_date, source,
                kospi_close, kospi_change_pct, kospi_ma20_slope, kospi_above_ma20,
                kosdaq_close, kosdaq_change_pct, kosdaq_ma20_slope, kosdaq_above_ma20,
                usdkrw, usdkrw_change_pct, vix_close, vix_change_pct,
                korea_rate, us_rate, rate_spread, market_regime, risk_on_score,
                sector_relative_strength_json, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now(),
                context.get("trade_date") or _now()[:10],
                context.get("source") or "market_context_provider",
                kospi.get("close"),
                kospi.get("change_pct"),
                kospi.get("ma20_slope"),
                _bool_int(kospi.get("above_ma20")),
                kosdaq.get("close"),
                kosdaq.get("change_pct"),
                kosdaq.get("ma20_slope"),
                _bool_int(kosdaq.get("above_ma20")),
                usdkrw.get("close"),
                usdkrw.get("change_pct"),
                vix.get("close"),
                vix.get("change_pct"),
                rates.get("korea_rate"),
                rates.get("us_rate"),
                rates.get("rate_spread"),
                context.get("market_regime") or "unknown",
                context.get("risk_on_score"),
                _json(sector_relative_strength),
                _json(context),
            ),
        )
    return int(cursor.lastrowid)


def get_latest_market_context(
    db_path: Path | str | None = None,
) -> dict[str, Any] | None:
    path = Path(db_path) if db_path else DEFAULT_MARKET_DB_PATH
    if not path.exists():
        return None

    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_SQL)
        _ensure_market_context_columns(conn)
        row = conn.execute(
            """
            SELECT *
            FROM market_context_snapshots
            ORDER BY trade_date DESC, id DESC
            LIMIT 1
            """
        ).fetchone()

    return _market_context_row_to_dict(row) if row else None


def _prepare_db(db_path: Path | str | None) -> Path:
    path = Path(db_path) if db_path else DEFAULT_MARKET_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
        _ensure_price_snapshot_columns(conn)
        _ensure_market_context_columns(conn)
    return path


def _ensure_price_snapshot_columns(conn: sqlite3.Connection) -> None:
    _ensure_column(conn, "price_snapshots", "minute_momentum_pct", "REAL")
    _ensure_column(conn, "price_snapshots", "execution_strength", "REAL")
    _ensure_column(conn, "price_snapshots", "orderbook_imbalance", "REAL")
    _ensure_column(conn, "price_snapshots", "spread_pct", "REAL")
    _ensure_column(conn, "price_snapshots", "smart_money_net_buy", "REAL")
    _ensure_column(conn, "price_snapshots", "smart_money_net_buy_5d", "REAL")
    _ensure_column(conn, "price_snapshots", "intraday_score", "REAL")


def _ensure_market_context_columns(conn: sqlite3.Connection) -> None:
    _ensure_column(conn, "market_context_snapshots", "trade_date", "TEXT")
    _ensure_column(conn, "market_context_snapshots", "source", "TEXT NOT NULL DEFAULT 'market_context_provider'")
    _ensure_column(conn, "market_context_snapshots", "kospi_close", "REAL")
    _ensure_column(conn, "market_context_snapshots", "kospi_change_pct", "REAL")
    _ensure_column(conn, "market_context_snapshots", "kospi_ma20_slope", "REAL")
    _ensure_column(conn, "market_context_snapshots", "kospi_above_ma20", "INTEGER")
    _ensure_column(conn, "market_context_snapshots", "kosdaq_close", "REAL")
    _ensure_column(conn, "market_context_snapshots", "kosdaq_change_pct", "REAL")
    _ensure_column(conn, "market_context_snapshots", "kosdaq_ma20_slope", "REAL")
    _ensure_column(conn, "market_context_snapshots", "kosdaq_above_ma20", "INTEGER")
    _ensure_column(conn, "market_context_snapshots", "usdkrw", "REAL")
    _ensure_column(conn, "market_context_snapshots", "usdkrw_change_pct", "REAL")
    _ensure_column(conn, "market_context_snapshots", "vix_close", "REAL")
    _ensure_column(conn, "market_context_snapshots", "vix_change_pct", "REAL")
    _ensure_column(conn, "market_context_snapshots", "korea_rate", "REAL")
    _ensure_column(conn, "market_context_snapshots", "us_rate", "REAL")
    _ensure_column(conn, "market_context_snapshots", "rate_spread", "REAL")
    _ensure_column(conn, "market_context_snapshots", "market_regime", "TEXT NOT NULL DEFAULT 'unknown'")
    _ensure_column(conn, "market_context_snapshots", "risk_on_score", "REAL")
    _ensure_column(conn, "market_context_snapshots", "sector_relative_strength_json", "TEXT NOT NULL DEFAULT '{}'")
    _ensure_column(conn, "market_context_snapshots", "raw_json", "TEXT NOT NULL DEFAULT '{}'")


def _ensure_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _market_context_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["kospi"] = {
        "close": data.get("kospi_close"),
        "change_pct": data.get("kospi_change_pct"),
        "ma20_slope": data.get("kospi_ma20_slope"),
        "above_ma20": _bool_from_int(data.get("kospi_above_ma20")),
    }
    data["kosdaq"] = {
        "close": data.get("kosdaq_close"),
        "change_pct": data.get("kosdaq_change_pct"),
        "ma20_slope": data.get("kosdaq_ma20_slope"),
        "above_ma20": _bool_from_int(data.get("kosdaq_above_ma20")),
    }
    data["usdkrw"] = {
        "close": data.get("usdkrw"),
        "change_pct": data.get("usdkrw_change_pct"),
    }
    data["vix"] = {
        "close": data.get("vix_close"),
        "change_pct": data.get("vix_change_pct"),
    }
    data["rates"] = {
        "korea_rate": data.get("korea_rate"),
        "us_rate": data.get("us_rate"),
        "rate_spread": data.get("rate_spread"),
    }
    data["sector_relative_strength"] = _parse_json(
        data.get("sector_relative_strength_json"),
        {},
    )
    data["raw"] = _parse_json(data.get("raw_json"), {})
    return data


def _parse_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _bool_int(value: Any) -> int | None:
    if value is None:
        return None
    return 1 if bool(value) else 0


def _bool_from_int(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
