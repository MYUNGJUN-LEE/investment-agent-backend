from __future__ import annotations

import argparse
import json
import time
from typing import Any

from app.config import settings
from app.trading.auto_trading import process_due_sessions
from app.trading.auto_trading_store import initialize_auto_trading_db
from app.trading.execution_status import print_startup_log
from app.trading.order_state import initialize_order_state_db


def run_once(worker_id: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
    _initialize_execution_storage()
    return process_due_sessions(worker_id=worker_id, limit=limit)


def run_forever(
    worker_id: str | None = None,
    poll_seconds: float | None = None,
) -> None:
    poll_seconds = settings.auto_trading_worker_poll_seconds if poll_seconds is None else poll_seconds
    _initialize_execution_storage()
    print_startup_log(
        auto_trading_worker_enabled=True,
        scanner_worker_enabled=bool(settings.universe_full_scan_enabled),
    )
    while True:
        try:
            run_once(worker_id=worker_id)
        except Exception as exc:
            print(f"trading_worker cycle failed: {exc}", flush=True)
        time.sleep(float(poll_seconds))


def _initialize_execution_storage() -> None:
    initialize_auto_trading_db()
    initialize_order_state_db()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run auto-trading session worker.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--worker-id", default=None)
    parser.add_argument("--poll-seconds", type=float, default=None)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    if args.once:
        print(json.dumps(run_once(worker_id=args.worker_id, limit=args.limit), ensure_ascii=False, default=str))
        return
    run_forever(worker_id=args.worker_id, poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
