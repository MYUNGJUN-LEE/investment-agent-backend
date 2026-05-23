from __future__ import annotations

import argparse
import json
import time
from typing import Any

from app.config import settings
from app.trading.market_monitor import JOB_KIS_MARKET, initialize_monitor_db, run_monitor_job


def run_once() -> dict[str, Any]:
    initialize_monitor_db()
    return run_monitor_job(JOB_KIS_MARKET)


def run_forever(poll_seconds: float | None = None) -> None:
    poll_seconds = settings.monitor_price_interval_seconds if poll_seconds is None else poll_seconds
    initialize_monitor_db()
    while True:
        try:
            run_once()
        except Exception as exc:
            print(f"market_worker cycle failed: {exc}", flush=True)
        time.sleep(float(poll_seconds))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 1-minute KIS market scanner worker.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=None)
    args = parser.parse_args()
    if args.once:
        print(json.dumps(run_once(), ensure_ascii=False, default=str))
        return
    run_forever(poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
