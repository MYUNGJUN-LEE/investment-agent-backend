from __future__ import annotations

from contextlib import closing
from datetime import datetime
import csv
import json
import logging
import re
from pathlib import Path
import sqlite3
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS kospi_symbols (
    symbol TEXT PRIMARY KEY,
    name TEXT,
    market TEXT NOT NULL DEFAULT 'KOSPI',
    source TEXT,
    updated_at TEXT NOT NULL,
    raw_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kospi_symbols_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

BUILTIN_KOSPI_UNIVERSE: dict[str, str] = {
    "005930": "Samsung Electronics",
    "000660": "SK hynix",
    "373220": "LG Energy Solution",
    "207940": "Samsung Biologics",
    "005380": "Hyundai Motor",
    "000270": "Kia",
    "068270": "Celltrion",
    "105560": "KB Financial Group",
    "055550": "Shinhan Financial Group",
    "035420": "NAVER",
    "035720": "Kakao",
    "012330": "Hyundai Mobis",
    "005490": "POSCO Holdings",
    "028260": "Samsung C&T",
    "051910": "LG Chem",
    "006400": "Samsung SDI",
    "086790": "Hana Financial Group",
    "032830": "Samsung Life Insurance",
    "033780": "KT&G",
    "066570": "LG Electronics",
    "003550": "LG Corp",
    "096770": "SK Innovation",
    "034730": "SK Inc",
    "015760": "Korea Electric Power",
    "017670": "SK Telecom",
    "009150": "Samsung Electro-Mechanics",
    "010130": "Korea Zinc",
    "018260": "Samsung SDS",
    "011200": "HMM",
    "009540": "HD Korea Shipbuilding & Offshore Engineering",
    "010140": "Samsung Heavy Industries",
    "267260": "HD Hyundai Electric",
    "042660": "Hanwha Ocean",
    "047810": "Korea Aerospace Industries",
    "000810": "Samsung Fire & Marine Insurance",
    "024110": "Industrial Bank of Korea",
    "316140": "Woori Financial Group",
    "086280": "Hyundai Glovis",
    "090430": "Amorepacific",
    "251270": "Netmarble",
    "010950": "S-Oil",
    "010060": "OCI Holdings",
    "000720": "Hyundai Engineering & Construction",
    "011070": "LG Innotek",
    "003670": "POSCO Future M",
    "034020": "Doosan Enerbility",
    "005830": "DB Insurance",
    "071050": "Korea Investment Holdings",
    "078930": "GS Holdings",
    "030200": "KT",
    "021240": "Coway",
    "032640": "LG Uplus",
    "011780": "Kumho Petrochemical",
    "006800": "Mirae Asset Securities",
    "138040": "Meritz Financial Group",
    "402340": "SK Square",
    "259960": "Krafton",
    "035250": "Kangwon Land",
    "161390": "Hankook Tire & Technology",
    "271560": "Orion",
    "028050": "Samsung E&A",
    "000100": "Yuhan",
    "097950": "CJ CheilJedang",
    "241560": "Doosan Bobcat",
    "047050": "POSCO International",
    "006260": "LS Corp",
}


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _db_path(db_path: Path | str | None = None) -> Path:
    return settings.storage_path(db_path or settings.universe_kospi_cache_path)


def _csv_path(csv_path: Path | str | None = None) -> Path:
    return settings.storage_path(csv_path or settings.universe_kospi_csv_path)


def _normalize_symbol(value: Any) -> str:
    raw = str(value or "").strip()
    digits = "".join(ch for ch in raw if ch.isdigit())
    if digits:
        return digits.zfill(6)[-6:]
    return raw


def _normalize_item(item: Any, *, source: str) -> dict[str, Any] | None:
    if isinstance(item, dict):
        symbol = _normalize_symbol(
            item.get("symbol")
            or item.get("code")
            or item.get("ticker")
            or item.get("종목코드")
        )
        name = item.get("name") or item.get("종목명") or item.get("company_name")
        market = item.get("market") or item.get("시장구분") or "KOSPI"
        raw_json = item
    else:
        symbol = _normalize_symbol(item)
        name = None
        market = "KOSPI"
        raw_json = {"symbol": item}

    market_text = str(market or "KOSPI").strip().upper()
    if "KOSDAQ" in market_text or (market_text and "KOSPI" not in market_text and market_text != "KS"):
        return None

    if not symbol or not symbol.isdigit() or len(symbol) != 6:
        return None

    if not _looks_like_common_stock(symbol=symbol, name=name, raw_json=raw_json):
        return None

    return {
        "symbol": symbol,
        "name": str(name).strip() if name is not None and str(name).strip() else None,
        "market": "KOSPI",
        "source": source,
        "raw_json": raw_json,
    }


def _looks_like_common_stock(
    *,
    symbol: str,
    name: Any,
    raw_json: dict[str, Any],
) -> bool:
    del symbol
    name_text = str(name or "").strip()
    compact_name = re.sub(r"\s+", "", name_text).upper()
    security_type = str(
        raw_json.get("security_type")
        or raw_json.get("type")
        or raw_json.get("종목종류")
        or raw_json.get("증권종류")
        or ""
    ).upper()
    combined = f"{compact_name} {security_type}"

    excluded_tokens = (
        "ETF",
        "ETN",
        "ELW",
        "SPAC",
        "스팩",
        "인버스",
        "레버리지",
        "선물",
        "채권",
        "워런트",
        "WARRANT",
    )
    if any(token in combined for token in excluded_tokens):
        return False

    # Preferred shares usually carry a trailing 우/우B/1우-style suffix.
    if re.search(r"(\d우|우B|우C|우선주|우)$", compact_name):
        return False

    status_text = str(
        raw_json.get("status")
        or raw_json.get("listing_status")
        or raw_json.get("상태")
        or raw_json.get("거래상태")
        or ""
    )
    if any(token in status_text for token in ("상장폐지", "정리매매", "거래정지")):
        return False

    return True


def _apply_limit(items: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or int(limit or 0) <= 0:
        return items
    return items[: max(0, int(limit))]


def initialize_kospi_universe_cache(db_path: Path | str | None = None) -> None:
    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


def upsert_kospi_symbols(
    symbols: list[dict[str, Any]] | list[str],
    *,
    source: str = "manual",
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    initialize_kospi_universe_cache(db_path)
    path = _db_path(db_path)
    updated_at = _now()
    inserted = 0
    skipped = 0

    with closing(sqlite3.connect(path)) as conn:
        for item in symbols:
            normalized = _normalize_item(item, source=source)
            if normalized is None:
                skipped += 1
                continue

            conn.execute(
                """
                INSERT INTO kospi_symbols (
                    symbol, name, market, source, updated_at, raw_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    name = excluded.name,
                    market = excluded.market,
                    source = excluded.source,
                    updated_at = excluded.updated_at,
                    raw_json = excluded.raw_json
                """,
                (
                    normalized["symbol"],
                    normalized.get("name"),
                    normalized.get("market") or "KOSPI",
                    source,
                    updated_at,
                    json.dumps(normalized.get("raw_json") or normalized, ensure_ascii=False, default=str),
                ),
            )
            inserted += 1

        conn.execute(
            """
            INSERT INTO kospi_symbols_meta (key, value, updated_at)
            VALUES ('last_refresh_at', ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (updated_at, updated_at),
        )
        conn.commit()

    return {
        "status": "success",
        "inserted": inserted,
        "skipped": skipped,
        "source": source,
        "updated_at": updated_at,
    }


def load_cached_kospi_symbols(
    *,
    limit: int | None = None,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    path = _db_path(db_path)
    if not path.exists():
        return []

    sql = """
        SELECT symbol, name, market, source, updated_at, raw_json
        FROM kospi_symbols
        ORDER BY symbol ASC
    """
    params: tuple[Any, ...] = ()
    if limit is not None and int(limit or 0) > 0:
        sql += " LIMIT ?"
        params = (int(limit),)

    try:
        with closing(sqlite3.connect(path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
    except sqlite3.Error:
        return []

    return [
        {
            "symbol": row["symbol"],
            "name": row["name"],
            "market": row["market"] or "KOSPI",
            "source": "kospi",
            "source_detail": row["source"] or "cache",
        }
        for row in rows
        if row["symbol"]
    ]


def load_kospi_symbols_from_csv(
    *,
    limit: int | None = None,
    csv_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    path = _csv_path(csv_path)
    if not path.exists():
        return []

    items: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return []
            for row in reader:
                normalized = _normalize_item(row, source="kospi")
                if normalized is None:
                    continue
                normalized["source_detail"] = "csv"
                items.append(
                    {
                        "symbol": normalized["symbol"],
                        "name": normalized.get("name"),
                        "market": "KOSPI",
                        "source": "kospi",
                        "source_detail": "csv",
                    }
                )
                if limit is not None and int(limit or 0) > 0 and len(items) >= int(limit):
                    break
    except (OSError, csv.Error, UnicodeError):
        return []

    return items


def kospi_universe_cache_status(
    *,
    db_path: Path | str | None = None,
    csv_path: Path | str | None = None,
) -> dict[str, Any]:
    cache_path = _db_path(db_path)
    csv_file = _csv_path(csv_path)
    cache_status: dict[str, Any] = {
        "enabled": bool(settings.universe_include_kospi),
        "source": str(settings.universe_kospi_symbol_source or "csv"),
        "cache_path": str(cache_path),
        "csv_path": str(csv_file),
        "csv_exists": csv_file.exists(),
        "builtin_fallback_enabled": bool(
            settings.universe_kospi_builtin_fallback_enabled
        ),
        "builtin_count": len(BUILTIN_KOSPI_UNIVERSE),
    }

    if not cache_path.exists():
        return {**cache_status, "status": "missing", "cached_count": 0}

    try:
        with closing(sqlite3.connect(cache_path)) as conn:
            conn.row_factory = sqlite3.Row
            count = conn.execute("SELECT COUNT(*) FROM kospi_symbols").fetchone()[0]
            row = conn.execute(
                "SELECT value FROM kospi_symbols_meta WHERE key = 'last_refresh_at'"
            ).fetchone()
    except sqlite3.Error as exc:
        return {**cache_status, "status": "error", "message": str(exc)}

    last_refresh = str(row["value"]) if row else None
    age_seconds = None
    stale = True
    if last_refresh:
        try:
            dt = datetime.fromisoformat(last_refresh.replace("Z", "+00:00")).replace(tzinfo=None)
            age_seconds = (datetime.utcnow() - dt).total_seconds()
            stale = age_seconds > float(settings.universe_kospi_cache_ttl_seconds or 86400)
        except Exception:
            stale = True

    return {
        **cache_status,
        "status": "ready",
        "cached_count": int(count or 0),
        "last_refresh_at": last_refresh,
        "age_seconds": round(age_seconds, 3) if age_seconds is not None else None,
        "stale": stale,
    }


def _load_from_existing_kis_helper(limit: int | None) -> list[dict[str, Any]]:
    try:
        from app.data_sources import kis as kis_source

        fetcher = getattr(kis_source, "fetch_kospi_symbols", None)
        if fetcher is None:
            fetcher = getattr(kis_source, "fetch_kospi_stock_symbols", None)
        if fetcher is None:
            return []
        data = fetcher()
    except Exception:
        return []

    if not isinstance(data, list):
        return []

    items: list[dict[str, Any]] = []
    for item in data:
        normalized = _normalize_item(item, source="kospi")
        if normalized is None:
            continue
        items.append(
            {
                "symbol": normalized["symbol"],
                "name": normalized.get("name"),
                "market": "KOSPI",
                "source": "kospi",
                "source_detail": "kis",
            }
        )
        if limit is not None and int(limit or 0) > 0 and len(items) >= int(limit):
            break
    return items


def load_builtin_kospi_symbols(
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if not bool(settings.universe_kospi_builtin_fallback_enabled):
        return []
    items = [
        {
            "symbol": symbol,
            "name": name,
            "market": "KOSPI",
            "source": "kospi",
            "source_detail": "builtin",
        }
        for symbol, name in BUILTIN_KOSPI_UNIVERSE.items()
    ]
    return _apply_limit(items, limit)


def load_kospi_symbols(
    *,
    limit: int | None = None,
    scan_all: bool | None = None,
) -> list[dict[str, Any]]:
    if not bool(settings.universe_include_kospi):
        return []

    source = str(settings.universe_kospi_symbol_source or "csv").strip().lower()
    if source == "disabled":
        return []

    scan_all = bool(settings.universe_kospi_scan_all) if scan_all is None else bool(scan_all)
    if scan_all and (limit is None or int(limit or 0) <= 0):
        effective_limit = None
    else:
        configured = int(limit if limit is not None else settings.universe_kospi_symbol_limit or 0)
        effective_limit = configured if configured > 0 else None

    source_order = _kospi_source_order(source)
    for source_name in source_order:
        items = _load_kospi_symbols_from_source(source_name, effective_limit)
        if items:
            logger.info(
                "Loaded %s KOSPI symbols from %s",
                len(items),
                source_name,
            )
            return _apply_limit(items, effective_limit)

    logger.warning(
        "No KOSPI symbols loaded from configured/local sources: %s",
        ",".join(source_order),
    )
    return []


def _kospi_source_order(source: str) -> list[str]:
    source = str(source or "").strip().lower()
    if source == "disabled":
        return []
    preferred = source if source in {"cache", "csv", "kis", "builtin"} else "cache"
    ordered = [preferred]
    for fallback in ("cache", "csv", "kis", "builtin"):
        if fallback not in ordered:
            ordered.append(fallback)
    return ordered


def _load_kospi_symbols_from_source(
    source: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    if source == "cache":
        return load_cached_kospi_symbols(limit=limit)
    if source == "csv":
        return load_kospi_symbols_from_csv(limit=limit)
    if source == "kis":
        return _load_from_existing_kis_helper(limit)
    if source == "builtin":
        return load_builtin_kospi_symbols(limit=limit)
    return []
