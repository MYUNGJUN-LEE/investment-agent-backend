from __future__ import annotations

from datetime import datetime
import argparse
import json
import time
from pathlib import Path
from typing import Any

from app.brokers.kis_client import KisApiError, KisClient, is_invalid_account_error
from app.trading import broker_sync


class KisPaperE2EError(RuntimeError):
    """Raised when the KIS paper E2E flow cannot prove the requested state."""


def preflight_kis_paper_e2e(
    *,
    symbol: str = "005930",
    client: KisClient | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Validate KIS paper connectivity without placing an order."""
    client = client or KisClient(is_paper=True)
    if getattr(client, "is_paper", True) is False:
        raise KisPaperE2EError("Preflight must use a KIS paper client")

    quote = client.get_current_price(symbol)
    resolved_price = _extract_current_price(quote)
    if resolved_price <= 0:
        raise KisPaperE2EError("KIS paper preflight could not resolve current price")

    balance, executions = _load_account_snapshot_with_token_retry(
        client=client,
        symbol=symbol,
    )
    sync_result = broker_sync.record_kis_sync(
        balance=balance,
        executions=executions,
        account_no=getattr(client, "account_no", "") or "",
        db_path=db_path,
    )
    return {
        "status": "ready",
        "symbol": symbol,
        "current_price": resolved_price,
        "quote_connected": True,
        "balance_connected": True,
        "executions_connected": True,
        "broker_sync": sync_result,
        "message": "KIS paper preflight completed without placing an order",
    }


def run_kis_paper_order_e2e(
    *,
    symbol: str = "005930",
    side: str = "buy",
    quantity: int = 1,
    price: float | None = None,
    poll_seconds: float = 5.0,
    timeout_seconds: float = 60.0,
    require_fill: bool = False,
    client: KisClient | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    Run a real KIS paper-trading order -> execution lookup -> balance sync flow.

    This function is intentionally not called by the normal unit test suite. Use
    it only with a KIS paper account and a tiny quantity.
    """
    if side not in ("buy", "sell"):
        raise KisPaperE2EError("side must be buy or sell")
    if quantity <= 0:
        raise KisPaperE2EError("quantity must be positive")

    client = client or KisClient(is_paper=True)
    quote = client.get_current_price(symbol)
    resolved_price = price or _extract_current_price(quote)
    if resolved_price <= 0:
        raise KisPaperE2EError("Unable to resolve a positive order price")

    before_sync = broker_sync.sync_kis_account(client=client, db_path=db_path)
    order = client.place_domestic_limit_order(
        symbol=symbol,
        side=side,
        price=resolved_price,
        quantity=quantity,
    )
    order_no = _extract_order_no(order)
    if not order_no:
        raise KisPaperE2EError("KIS order response did not include order number")

    started = time.monotonic()
    deadline = started + max(0, timeout_seconds)
    last_sync: dict[str, Any] | None = None
    matched_execution: dict[str, Any] | None = None

    while True:
        balance = client.get_balance()
        executions = client.get_daily_order_executions(
            start_date=datetime.now().strftime("%Y%m%d"),
            end_date=datetime.now().strftime("%Y%m%d"),
            symbol=symbol,
        )
        last_sync = broker_sync.record_kis_sync(
            balance=balance,
            executions=executions,
            account_no=client.account_no or "",
            db_path=db_path,
        )
        matched_execution = _find_execution(
            executions=executions,
            order_no=order_no,
            symbol=symbol,
        )
        if matched_execution and _is_filled(matched_execution, quantity):
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(max(0.0, poll_seconds))

    filled = bool(matched_execution and _is_filled(matched_execution, quantity))
    if require_fill and not filled:
        raise KisPaperE2EError(
            f"KIS paper order {order_no} was submitted but not fully filled before timeout"
        )

    return {
        "status": "filled" if filled else "submitted_not_filled",
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "price": resolved_price,
        "order_no": order_no,
        "before_sync": before_sync,
        "last_sync": last_sync,
        "matched_execution": matched_execution,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def _extract_current_price(quote: dict[str, Any]) -> float:
    output = quote.get("output") if isinstance(quote, dict) else {}
    if not isinstance(output, dict):
        output = {}
    for key in ("stck_prpr", "prpr", "last", "price"):
        value = _to_float(output.get(key) or quote.get(key))
        if value and value > 0:
            return value
    return 0.0


def _load_account_snapshot_with_token_retry(
    *,
    client: KisClient,
    symbol: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        return _load_account_snapshot(client=client, symbol=symbol)
    except KisApiError as exc:
        if not is_invalid_account_error(exc):
            raise
        refresh = getattr(client, "issue_access_token", None)
        if not callable(refresh):
            raise
        refresh(force_refresh=True)
        return _load_account_snapshot(client=client, symbol=symbol)


def _load_account_snapshot(
    *,
    client: KisClient,
    symbol: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    today = datetime.now().strftime("%Y%m%d")
    balance = client.get_balance()
    executions = client.get_daily_order_executions(
        start_date=today,
        end_date=today,
        symbol=symbol,
    )
    return balance, executions


def _extract_order_no(order: dict[str, Any]) -> str:
    output = order.get("output") if isinstance(order, dict) else {}
    if isinstance(output, dict):
        for key in ("ODNO", "odno", "order_no"):
            if output.get(key):
                return str(output[key])
    for key in ("ODNO", "odno", "order_no"):
        if order.get(key):
            return str(order[key])
    return ""


def _find_execution(
    executions: dict[str, Any],
    order_no: str,
    symbol: str,
) -> dict[str, Any] | None:
    for row in _rows(executions, "output1", "output"):
        row_order_no = str(row.get("odno") or row.get("ODNO") or "").strip()
        row_symbol = str(row.get("pdno") or row.get("PDNO") or "").strip()
        if row_order_no == order_no and (not row_symbol or row_symbol == symbol):
            return row
    return None


def _is_filled(row: dict[str, Any], expected_quantity: int) -> bool:
    remaining_qty = _to_int(row.get("rmn_qty") or row.get("RMN_QTY"))
    filled_qty = _to_int(
        row.get("tot_ccld_qty")
        or row.get("ccld_qty")
        or row.get("TOT_CCLD_QTY")
        or row.get("CCLD_QTY")
    )
    if remaining_qty is not None and remaining_qty <= 0 and filled_qty:
        return True
    return bool(filled_qty is not None and filled_qty >= expected_quantity)


def _rows(data: dict[str, Any], *keys: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, dict))
        elif isinstance(value, dict):
            rows.append(value)
    return rows


def _to_float(value: Any) -> float | None:
    if value in (None, "", "-"):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    return int(number) if number is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KIS paper order E2E flow")
    parser.add_argument("--symbol", default="005930")
    parser.add_argument("--side", choices=["buy", "sell"], default="buy")
    parser.add_argument("--quantity", type=int, default=1)
    parser.add_argument("--price", type=float)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--require-fill", action="store_true")
    parser.add_argument(
        "--place-order",
        action="store_true",
        help="Actually place a KIS paper order. Without this flag, only preflight runs.",
    )
    args = parser.parse_args()

    if args.place_order:
        result = run_kis_paper_order_e2e(
            symbol=args.symbol,
            side=args.side,
            quantity=args.quantity,
            price=args.price,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
            require_fill=args.require_fill,
        )
    else:
        result = preflight_kis_paper_e2e(symbol=args.symbol)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
