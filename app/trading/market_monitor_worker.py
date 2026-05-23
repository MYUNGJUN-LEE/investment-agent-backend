from __future__ import annotations

import argparse
import json

from app.trading.market_monitor import (
    initialize_monitor_db,
    process_due_monitor_jobs,
    run_monitor_worker_forever,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the separated periodic market monitor worker.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process currently due monitor jobs once and exit.",
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
        help="Maximum due monitor jobs to process in --once mode.",
    )
    args = parser.parse_args()

    initialize_monitor_db()
    if args.once:
        results = process_due_monitor_jobs(limit=args.limit)
        print(json.dumps({"processed": results}, ensure_ascii=False, default=str))
        return

    run_monitor_worker_forever(poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
