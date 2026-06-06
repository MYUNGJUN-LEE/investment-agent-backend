from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Callable

from app.config import settings
from app.trading.execution_status import print_startup_log
from app.workers import (
    broker_worker,
    dart_worker,
    market_worker,
    news_worker,
    orchestrator_worker,
    trading_worker,
)


@dataclass(frozen=True)
class WorkerSpec:
    name: str
    target: Callable[[], None]


_lock = threading.RLock()
_threads: dict[str, threading.Thread] = {}


def ensure_embedded_workers_started() -> dict[str, object]:
    """Start embedded workers once inside the web process."""
    with _lock:
        for spec in _worker_specs():
            thread = _threads.get(spec.name)
            if thread and thread.is_alive():
                continue
            thread = threading.Thread(
                target=_guarded_target(spec.name, spec.target),
                name=f"embedded-{spec.name}",
                daemon=True,
            )
            thread.start()
            _threads[spec.name] = thread
        return embedded_worker_status()


def embedded_worker_status() -> dict[str, object]:
    with _lock:
        workers = [
            {
                "name": name,
                "alive": thread.is_alive(),
                "thread_name": thread.name,
            }
            for name, thread in sorted(_threads.items())
        ]
    return {
        "enabled": settings.embedded_workers_enabled,
        "workers": workers,
        "count": len(workers),
    }


def start_on_app_startup_if_enabled() -> dict[str, object]:
    print_startup_log(
        auto_trading_worker_enabled=bool(settings.embedded_workers_enabled),
        scanner_worker_enabled=bool(
            settings.embedded_workers_enabled and settings.universe_full_scan_enabled
        ),
    )
    if not settings.embedded_workers_enabled:
        return embedded_worker_status()
    return ensure_embedded_workers_started()


def _worker_specs() -> list[WorkerSpec]:
    specs = [
        WorkerSpec("trading_worker", trading_worker.run_forever),
        WorkerSpec("orchestrator_worker", orchestrator_worker.run_forever),
        WorkerSpec("market_worker", market_worker.run_forever),
        WorkerSpec("news_worker", news_worker.run_forever),
        WorkerSpec("dart_worker", dart_worker.run_forever),
    ]
    if not settings.trade_orchestrator_enabled:
        specs = [spec for spec in specs if spec.name != "orchestrator_worker"]
    if settings.embedded_worker_broker_sync_enabled:
        specs.append(WorkerSpec("broker_worker", broker_worker.run_forever))
    return specs


def _guarded_target(name: str, target: Callable[[], None]) -> Callable[[], None]:
    def run() -> None:
        try:
            target()
        except Exception as exc:
            # Avoid crashing the web server; status will show the thread stopped.
            print(f"Embedded worker {name} stopped: {exc}", flush=True)

    return run
