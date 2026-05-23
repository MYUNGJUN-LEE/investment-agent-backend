from __future__ import annotations

import argparse
import json
import time
from typing import Any

from app.config import settings
from app.trading.trade_orchestrator import run_trade_orchestrator_once


def run_once(worker_id: str | None = None) -> dict[str, Any]:
    return run_trade_orchestrator_once(worker_id=worker_id)


def run_forever(
    worker_id: str | None = None,
    poll_seconds: float | None = None,
) -> None:
    poll_seconds = (
        settings.trade_orchestrator_interval_seconds
        if poll_seconds is None
        else poll_seconds
    )
    while True:
        try:
            run_once(worker_id=worker_id)
        except Exception as exc:
            print(f"orchestrator_worker cycle failed: {exc}", flush=True)
        time.sleep(float(poll_seconds))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trade orchestration worker.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--worker-id", default=None)
    parser.add_argument("--poll-seconds", type=float, default=None)
    args = parser.parse_args()
    if args.once:
        print(json.dumps(run_once(worker_id=args.worker_id), ensure_ascii=False, default=str))
        return
    run_forever(worker_id=args.worker_id, poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
