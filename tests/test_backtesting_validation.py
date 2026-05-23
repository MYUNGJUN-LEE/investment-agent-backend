from __future__ import annotations

from app.backtesting.validation import (
    evaluate_overfit_risk,
    split_time_series,
    walk_forward_windows,
)


def test_time_series_split_uses_expected_ratios_without_shuffle():
    rows = [{"date": f"{day:03d}", "value": day} for day in range(1, 101)]

    split = split_time_series(rows)

    assert len(split["train"]) == 55
    assert len(split["validation"]) == 22
    assert len(split["test"]) == 23
    assert split["train"][0]["value"] == 1
    assert split["test"][-1]["value"] == 100


def test_walk_forward_windows_match_requested_years():
    assert walk_forward_windows(2018, 2024, 3) == [
        {"train_start_year": 2018, "train_end_year": 2020, "validation_year": 2021},
        {"train_start_year": 2019, "train_end_year": 2021, "validation_year": 2022},
        {"train_start_year": 2020, "train_end_year": 2022, "validation_year": 2023},
        {"train_start_year": 2021, "train_end_year": 2023, "validation_year": 2024},
    ]


def test_overfit_risk_flags_sharpe_collapse_and_low_sample():
    result = evaluate_overfit_risk(
        {
            "in_sample_sharpe": 3.2,
            "out_of_sample_sharpe": 0.2,
            "trade_count": 20,
            "profitable_symbol_count": 1,
            "profitable_year_count": 1,
            "net_return_before_cost": 0.2,
            "net_return_after_cost": 0.05,
        }
    )

    assert result["overfit_risk"] == "high"
    assert result["approved_for_live"] is False
    assert "in_sample_out_of_sample_sharpe_collapse" in result["flags"]
