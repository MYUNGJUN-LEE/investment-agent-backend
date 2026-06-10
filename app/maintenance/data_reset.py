from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from app.config import settings
from app.storage import market_data
from app.trading import paper_trading


RESET_CONFIRMATION = "RESET_TRADING_DATA"


def reset_trading_data(
    *,
    confirm: str,
    include_all_data_files: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Delete generated local/Render disk data so scanners rebuild from scratch."""
    if confirm != RESET_CONFIRMATION:
        return {
            "status": "blocked",
            "message": f"confirm must be exactly {RESET_CONFIRMATION}",
            "deleted_count": 0,
            "deleted_files": [],
        }

    settings.clear_storage_cache()
    files = _data_files(include_all_data_files=include_all_data_files)
    deleted: list[str] = []
    errors: list[dict[str, str]] = []
    for path in files:
        try:
            if not dry_run:
                path.unlink(missing_ok=True)
            deleted.append(str(path))
        except OSError as exc:
            errors.append({"path": str(path), "error": str(exc)})

    return {
        "status": "dry_run" if dry_run else "success",
        "deleted_count": len(deleted),
        "deleted_files": deleted,
        "error_count": len(errors),
        "errors": errors,
        "include_all_data_files": include_all_data_files,
        "data_roots": [str(path) for path in _candidate_data_roots() if path.exists()],
    }


def _data_files(*, include_all_data_files: bool) -> list[Path]:
    candidates: set[Path] = set()
    for path in _known_generated_paths():
        candidates.update(_with_sqlite_sidecars(path))
        if path.name.endswith(".json") or include_all_data_files:
            candidates.add(path)

    for root in _candidate_data_roots():
        if not root.exists() or not root.is_dir():
            continue
        _cleanup_storage_check_files(root)
        patterns = ["*.sqlite3", "*.sqlite3-*", "*.db", "*.db-*", "*.json"]
        if include_all_data_files:
            patterns.extend(["*.csv", "*.log", "*.tmp"])
        for pattern in patterns:
            for path in root.glob(pattern):
                if path.is_file() and not _is_storage_check_file(path):
                    candidates.add(path)

    return sorted(
        (path for path in candidates if path.exists() and path.is_file()),
        key=lambda item: str(item),
    )


def _with_sqlite_sidecars(path: Path) -> set[Path]:
    return {path, Path(f"{path}-wal"), Path(f"{path}-shm"), Path(f"{path}-journal")}


def _cleanup_storage_check_files(root: Path) -> None:
    for pattern in (".storage-write-check-*", ".storage-sqlite-check-*"):
        for path in root.glob(pattern):
            try:
                if path.is_file():
                    path.unlink(missing_ok=True)
            except OSError:
                pass


def _is_storage_check_file(path: Path) -> bool:
    name = path.name
    return name.startswith(".storage-write-check-") or name.startswith(
        ".storage-sqlite-check-"
    )


def _known_generated_paths() -> list[Path]:
    return [
        settings.storage_path(settings.auto_trading_db_path),
        settings.storage_path(settings.edge_calibration_db_path),
        settings.storage_path(settings.broker_sync_db_path),
        settings.storage_path(settings.order_state_db_path),
        settings.storage_path(settings.market_monitor_db_path),
        settings.storage_path(settings.alert_db_path),
        settings.storage_path(settings.universe_scanner_db_path),
        settings.storage_path(settings.kis_token_cache_path),
        settings.storage_path(settings.emergency_stop_file),
        settings.storage_path(market_data.DEFAULT_MARKET_DB_PATH),
        settings.storage_path(paper_trading.DEFAULT_DB_PATH),
    ]


def _candidate_data_roots() -> list[Path]:
    roots: list[Path] = []
    storage_root = settings.storage_root()
    candidates = (
        (settings.storage_path("data"), storage_root)
        if storage_root is not None
        else (Path("data"),)
    )
    for raw in candidates:
        if raw is None:
            continue
        path = Path(raw)
        if path not in roots:
            roots.append(path)
    return roots


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset generated trading data files.")
    parser.add_argument("--confirm", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--sqlite-only",
        action="store_true",
        help="Delete SQLite files and token JSON only; keep CSV/log/tmp files.",
    )
    args = parser.parse_args()
    result = reset_trading_data(
        confirm=args.confirm,
        include_all_data_files=not args.sqlite_only,
        dry_run=args.dry_run,
    )
    import json

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
