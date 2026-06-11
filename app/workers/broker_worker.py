from __future__ import annotations

import argparse
import json
import time
from typing import Any

from app.config import settings
from app.trading.broker_sync_worker import run_broker_sync_once
from app.trading.order_state import initialize_order_state_db


def run_once(reconcile_order_state: bool = True) -> dict[str, Any]:
    return run_broker_sync_once(reconcile_order_state=reconcile_order_state)


def run_forever(
    poll_seconds: float | None = None,
    reconcile_order_state: bool = True,
) -> None:
    poll_seconds = settings.broker_sync_interval_seconds if poll_seconds is None else poll_seconds
    initialize_order_state_db()
    while True:
        sleep_seconds = float(poll_seconds)
        try:
            result = run_once(reconcile_order_state=reconcile_order_state)
            if result.get("status") in {"config_error", "token_backoff", "account_backoff"}:
                sleep_seconds = max(
                    sleep_seconds,
                    float(settings.broker_sync_config_error_backoff_seconds or 0),
                )
        except Exception as exc:
            print(f"broker_worker cycle failed: {exc}", flush=True)
        time.sleep(sleep_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KIS broker balance/execution sync worker.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=None)
    parser.add_argument("--skip-order-state", action="store_true")
    args = parser.parse_args()
    reconcile = not args.skip_order_state
    if args.once:
        print(json.dumps(run_once(reconcile_order_state=reconcile), ensure_ascii=False, default=str))
        return
    run_forever(poll_seconds=args.poll_seconds, reconcile_order_state=reconcile)


if __name__ == "__main__":
    main()
