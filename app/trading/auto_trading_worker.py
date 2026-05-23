from __future__ import annotations

import argparse
import json

from app.trading.auto_trading import process_due_sessions, run_worker_forever
from app.trading.auto_trading_store import initialize_auto_trading_db


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the persistent auto-trading worker.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process currently due sessions once and exit.",
    )
    parser.add_argument(
        "--worker-id",
        default=None,
        help="Optional stable worker id for DB locks.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=None,
        help="Polling interval for continuous worker mode.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum due sessions to process in --once mode.",
    )
    args = parser.parse_args()

    initialize_auto_trading_db()
    if args.once:
        results = process_due_sessions(worker_id=args.worker_id, limit=args.limit)
        print(json.dumps({"processed": results}, ensure_ascii=False, default=str))
        return

    run_worker_forever(worker_id=args.worker_id, poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
