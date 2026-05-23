from __future__ import annotations

import argparse
import json
import time
from typing import Any

from app.config import settings
from app.trading import broker_sync, order_state


def run_broker_sync_once(reconcile_order_state: bool = True) -> dict[str, Any]:
    """Synchronize KIS broker state and optionally reconcile position states."""
    sync_result = broker_sync.sync_kis_account()
    reconcile_result: dict[str, Any] | None = None
    if reconcile_order_state:
        reconcile_result = order_state.reconcile_all_after_broker_sync(
            account_no=str(sync_result.get("account_no") or ""),
        )
    return {
        "status": sync_result.get("status", "unknown"),
        "broker_sync": sync_result,
        "order_state_reconcile": reconcile_result,
    }


def run_broker_sync_forever(
    poll_seconds: float | None = None,
    reconcile_order_state: bool = True,
) -> None:
    poll_seconds = (
        settings.broker_sync_interval_seconds
        if poll_seconds is None
        else poll_seconds
    )
    order_state.initialize_order_state_db()
    while True:
        run_broker_sync_once(reconcile_order_state=reconcile_order_state)
        time.sleep(float(poll_seconds))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the separated KIS broker balance/execution sync worker.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one broker sync pass and exit.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=None,
        help="Polling interval for continuous worker mode.",
    )
    parser.add_argument(
        "--skip-order-state",
        action="store_true",
        help="Do not reconcile order-state positions after broker sync.",
    )
    args = parser.parse_args()

    if args.once:
        result = run_broker_sync_once(reconcile_order_state=not args.skip_order_state)
        print(json.dumps(result, ensure_ascii=False, default=str))
        return

    run_broker_sync_forever(
        poll_seconds=args.poll_seconds,
        reconcile_order_state=not args.skip_order_state,
    )


if __name__ == "__main__":
    main()
