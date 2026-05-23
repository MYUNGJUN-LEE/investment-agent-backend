from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.55
    validation_ratio: float = 0.225
    test_ratio: float = 0.225
    paper_trading_min_days: int = 14
    paper_trading_max_days: int = 31

    def validate(self) -> None:
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if not 0.99 <= total <= 1.01:
            raise ValueError("Train, validation, and test ratios must sum to 1")
        if not 0.50 <= self.train_ratio <= 0.60:
            raise ValueError("Train ratio must be between 50% and 60%")
        if not 0.20 <= self.validation_ratio <= 0.25:
            raise ValueError("Validation ratio must be between 20% and 25%")
        if not 0.20 <= self.test_ratio <= 0.25:
            raise ValueError("Test ratio must be between 20% and 25%")
        if self.paper_trading_min_days < 14:
            raise ValueError("Paper trading should run at least 2 weeks")


def split_time_series(
    rows: list[dict[str, Any]],
    date_key: str = "date",
    config: SplitConfig | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Split sorted time-series data without shuffling or look-ahead leakage."""
    config = config or SplitConfig()
    config.validate()
    ordered = sorted(rows, key=lambda row: str(row.get(date_key) or ""))
    train_end = int(len(ordered) * config.train_ratio)
    validation_end = train_end + int(len(ordered) * config.validation_ratio)
    return {
        "train": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "test": ordered[validation_end:],
    }


def walk_forward_windows(
    start_year: int = 2018,
    end_year: int = 2024,
    train_years: int = 3,
) -> list[dict[str, int]]:
    """
    Build rolling walk-forward windows.

    Example: 2018~2020 train -> 2021 validation, then rolls forward one year.
    """
    windows: list[dict[str, int]] = []
    train_start = start_year
    while train_start + train_years <= end_year:
        train_end = train_start + train_years - 1
        validation_year = train_end + 1
        if validation_year > end_year:
            break
        windows.append(
            {
                "train_start_year": train_start,
                "train_end_year": train_end,
                "validation_year": validation_year,
            }
        )
        train_start += 1
    return windows


def evaluate_overfit_risk(metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Flag common over-optimization symptoms from train/validation/test results.
    """
    flags: list[str] = []
    in_sample_sharpe = metrics.get("in_sample_sharpe")
    out_sample_sharpe = metrics.get("out_of_sample_sharpe")
    trade_count = int(metrics.get("trade_count") or 0)
    profitable_symbol_count = int(metrics.get("profitable_symbol_count") or 0)
    profitable_year_count = int(metrics.get("profitable_year_count") or 0)
    net_return_before_cost = metrics.get("net_return_before_cost")
    net_return_after_cost = metrics.get("net_return_after_cost")

    if (
        isinstance(in_sample_sharpe, (int, float))
        and isinstance(out_sample_sharpe, (int, float))
        and in_sample_sharpe >= 3.0
        and out_sample_sharpe <= 0.3
    ):
        flags.append("in_sample_out_of_sample_sharpe_collapse")

    if trade_count <= 30:
        flags.append("trade_count_too_low")
    if profitable_symbol_count <= 2:
        flags.append("profit_concentrated_in_few_symbols")
    if profitable_year_count <= 1:
        flags.append("profit_concentrated_in_one_year")

    if (
        isinstance(net_return_before_cost, (int, float))
        and isinstance(net_return_after_cost, (int, float))
        and net_return_before_cost > 0
        and net_return_after_cost <= net_return_before_cost * 0.5
    ):
        flags.append("cost_sensitivity_too_high")

    severity = "low"
    if len(flags) >= 3:
        severity = "high"
    elif flags:
        severity = "medium"

    return {
        "overfit_risk": severity,
        "flags": flags,
        "approved_for_paper": severity != "high",
        "approved_for_live": severity == "low" and trade_count > 30,
    }
