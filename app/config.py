import os
import logging
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

from pydantic_settings import BaseSettings, SettingsConfigDict


logger = logging.getLogger(__name__)

_RENDER_NON_PERSISTENT_WARNING = (
    "WARNING: using non-persistent fallback storage. Attach a Render persistent "
    "disk and set DATA_DIR to its mount path."
)
_RENDER_VAR_DATA_WARNING = (
    "WARNING: /var/data is not writable. Check that the persistent disk is "
    "attached to this Render service and mounted at /var/data."
)
_LOCAL_NON_PERSISTENT_WARNING = (
    "WARNING: using non-persistent fallback storage; data may be lost on restart."
)
_STORAGE_STATUS_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_STORAGE_WARNING_KEYS: set[tuple[Any, ...]] = set()


class Settings(BaseSettings):
    app_name: str = "Investment Agent Backend"

    # If this is set, requests must include X-API-Key with the same value.
    backend_api_key: str | None = None
    action_schema_public_url: str | None = "https://api.autoinvestmentkorea.online"

    # Comma-separated browser/GPT origins allowed to call this API.
    cors_allow_origins: str = (
        "http://localhost:3000,http://127.0.0.1:3000,"
        "https://chat.openai.com,https://chatgpt.com,https://www.chatgpt.com"
    )

    # Persistent storage. On Render, mount the disk and set either DATA_DIR or
    # RENDER_DISK_MOUNT_PATH. If unset, common Render disk paths are detected.
    data_dir: str | None = None
    render_disk_mount_path: str | None = None
    execution_mode: str | None = None
    broker_provider: str | None = None

    # External APIs
    opendart_api_key: str | None = None

    naver_client_id: str | None = None
    naver_client_secret: str | None = None

    kis_app_key: str | None = None
    kis_app_secret: str | None = None
    kis_account_no: str | None = None
    kis_account_product_code: str | None = None
    kis_is_paper: bool = True
    kis_token_cache_path: str = "data/kis_token_cache.json"
    kis_token_issue_cooldown_seconds: int = 70
    kis_token_refresh_buffer_seconds: int = 600
    kis_account_cache_path: str = "data/kis_account_cache.json"
    kis_account_cache_ttl_seconds: int = 20
    kis_account_rate_limit_backoff_seconds: int = 70
    kis_account_min_probe_interval_seconds: float = 2.0
    kis_request_min_interval_seconds: float = 1.5

    # Optional JSON feed used by app.data_sources.market_context.
    market_context_url: str | None = None
    market_context_timeout: float = 10.0
    regime_gate_enabled: bool = True
    regime_gate_cache_seconds: int = 60
    regime_gate_live_block_stress: bool = True
    regime_gate_live_block_bear: bool = False
    regime_gate_missing_context_live_hurdle_add_bps: float = 15.0
    regime_gate_bull_hurdle_add_bps: float = -10.0
    regime_gate_neutral_hurdle_add_bps: float = 0.0
    regime_gate_bear_hurdle_add_bps: float = 30.0
    regime_gate_stress_hurdle_add_bps: float = 80.0
    regime_gate_bull_position_multiplier: float = 1.00
    regime_gate_neutral_position_multiplier: float = 0.85
    regime_gate_bear_position_multiplier: float = 0.50
    regime_gate_stress_position_multiplier: float = 0.00
    regime_gate_min_risk_on_score_bull: float = 65.0
    regime_gate_min_risk_on_score_neutral: float = 45.0
    regime_gate_stress_risk_on_score: float = 25.0
    regime_gate_negative_index_return_1d_stress_pct: float = -2.0

    # Live trading is disabled unless both flags are explicitly set.
    enable_live_trading: bool = False
    live_trading_confirm_token: str | None = None

    # Trading cost assumptions. Override these per broker/account in .env.
    commission_rate: float = 0.00147
    kr_stock_sell_tax_rate: float = 0.002
    us_fx_spread_bps: float = 5.0
    default_fill_probability: float = 0.95
    min_fill_probability: float = 0.7
    margin_interest_annual_rate: float = 0.08
    short_borrow_annual_rate: float = 0.03
    performance_min_trade_count: int = 10
    emergency_stop: bool = False
    emergency_stop_file: str = "data/EMERGENCY_STOP"
    max_order_api_retries: int = 1
    max_order_price_deviation_bps: float = 300.0
    fill_quality_feedback_enabled: bool = True
    fill_quality_db_path: str = "data/fill_quality.sqlite3"
    fill_quality_ewma_alpha: float = 0.15
    fill_quality_min_samples: int = 5
    fill_quality_max_recent_orders: int = 200
    fill_quality_default_probability: float = 0.95
    fill_quality_min_probability: float = 0.70
    fill_quality_bad_slippage_bps: float = 35.0
    fill_quality_bad_fill_ratio: float = 0.70
    fill_quality_live_block_bad_fills: bool = True
    fill_quality_live_max_slippage_penalty_bps: float = 50.0
    fill_quality_paper_max_slippage_penalty_bps: float = 20.0
    fill_quality_delay_penalty_threshold_seconds: float = 30.0
    fill_quality_delay_penalty_bps: float = 5.0
    auto_trading_db_path: str = "data/auto_trading.sqlite3"
    auto_trading_worker_poll_seconds: float = 30
    auto_trading_worker_lock_seconds: int = 600
    auto_trading_stale_lock_recover_seconds: int = 1200
    auto_trading_symbol_workers: int = 1
    auto_trading_max_open_positions: int = 5
    portfolio_penalty_enabled: bool = True
    portfolio_corr_lookback_prices: int = 30
    portfolio_corr_min_returns: int = 5
    portfolio_corr_threshold: float = 0.65
    portfolio_corr_penalty_cap_bps: float = 35.0
    portfolio_sector_penalty_bps: float = 10.0
    portfolio_sector_penalty_cap_bps: float = 30.0
    pretrade_orderbook_check_enabled: bool = False
    pretrade_orderbook_top_n: int = 3
    pretrade_orderbook_max_spread_bps: float = 25.0
    pretrade_orderbook_min_depth_coverage: float = 2.0
    pretrade_orderbook_min_imbalance: float = -0.20
    position_sizing_edge_enabled: bool = True
    position_sizing_edge_floor_bps: float = 0.0
    position_sizing_edge_cap_bps: float = 150.0
    position_sizing_min_multiplier: float = 0.35
    position_sizing_max_multiplier: float = 1.0
    position_sizing_max_symbol_weight: float = 0.10
    position_sizing_default_stop_bps: float = 250.0
    signal_decay_enabled: bool = True
    signal_decay_half_life_seconds: int = 1800
    signal_decay_min_multiplier: float = 0.35
    signal_decay_live_max_candidate_age_seconds: int = 3600
    signal_decay_paper_max_candidate_age_seconds: int = 7200
    signal_decay_missing_age_live_penalty_bps: float = 10.0
    outcome_attribution_enabled: bool = True
    outcome_attribution_db_path: str = "data/outcome_attribution.sqlite3"
    outcome_attribution_risk_weight: float = 0.10
    outcome_attribution_max_records: int = 1500
    outcome_attribution_min_hold_seconds: int = 60
    outcome_attribution_recent_limit: int = 100
    auto_tuning_enabled: bool = True
    auto_tuning_mode: str = "recommend"
    auto_tuning_db_path: str = "data/auto_tuning.sqlite3"
    auto_tuning_recent_limit: int = 100
    auto_tuning_min_samples: int = 30
    auto_tuning_min_loss_samples: int = 10
    auto_tuning_max_recommendations: int = 20
    auto_tuning_bad_avg_net_edge_bps: float = -20.0
    auto_tuning_bad_execution_component_bps: float = -25.0
    auto_tuning_bad_regime_component_bps: float = -20.0
    auto_tuning_bad_time_decay_component_bps: float = -15.0
    auto_tuning_bad_unexplained_component_bps: float = -35.0
    auto_tuning_slippage_step_bps: float = 5.0
    auto_tuning_hurdle_step_bps: float = 10.0
    auto_tuning_position_scale_step: float = 0.10
    auto_tuning_signal_decay_scale_step: float = 0.20
    tuning_review_enabled: bool = True
    tuning_review_db_path: str = "data/tuning_review.sqlite3"
    tuning_review_max_records: int = 300
    tuning_review_allowed_keys: str = (
        "UNIVERSE_SCANNER_DEFAULT_SLIPPAGE_BPS,"
        "SIGNAL_DECAY_HALF_LIFE_SECONDS,"
        "SIGNAL_DECAY_LIVE_MAX_CANDIDATE_AGE_SECONDS,"
        "REGIME_GATE_BEAR_HURDLE_ADD_BPS,"
        "POSITION_SIZING_MAX_MULTIPLIER,"
        "POSITION_SIZING_MAX_SYMBOL_WEIGHT,"
        "UNIVERSE_SCANNER_WORKER_HURDLE_RATE_BPS"
    )
    auto_trading_one_session_per_account: bool = True
    trade_orchestrator_enabled: bool = True
    trade_orchestrator_interval_seconds: int = 600
    trade_orchestrator_execute_entries: bool = True
    trade_orchestrator_base_position_weight: float = 0.10
    trade_orchestrator_min_position_weight: float = 0.05
    trade_orchestrator_max_position_weight: float = 0.15
    trade_orchestrator_edge_weight_cap_bps: float = 300.0
    live_exit_confirm_before_entry: bool = True
    position_time_stop_trading_days: int = 2
    strategy_circuit_breaker_enabled: bool = True
    strategy_circuit_breaker_win_rate_floor: float = 0.375
    strategy_circuit_breaker_min_trades: int = 8
    strategy_circuit_breaker_lookback_trades: int = 8
    strategy_circuit_breaker_scale: float = 0.5
    edge_calibration_enabled: bool = True
    edge_calibration_db_path: str = "data/edge_calibration.sqlite3"
    edge_calibration_interval_seconds: int = 3600
    edge_calibration_top10_performance_interval_seconds: int = 600
    edge_calibration_horizon_seconds: int = 86400
    edge_calibration_max_samples: int = 2000
    edge_calibration_min_samples: int = 30
    edge_calibration_future_price_limit: int = 300
    edge_calibration_label_snapshots_enabled: bool = True
    edge_calibration_label_snapshot_max_symbols: int = 50
    edge_calibration_refresh_batch_size: int = 100
    edge_calibration_refresh_after_scan: bool = False
    edge_calibration_min_label_age_seconds: int = 300
    edge_calibration_min_future_snapshots: int = 2
    edge_calibration_label_at_horizon_end: bool = True
    edge_calibration_label_horizon_tolerance_seconds: int = 3600
    edge_calibration_candidate_lookback_seconds: int | None = None
    edge_calibration_ridge_lambda: float = 10.0
    edge_calibration_blend: float = 0.35
    edge_calibration_target_samples: int = 1500
    edge_calibration_sample_retention_limit: int = 3000
    edge_calibration_gate_min_samples: int = 100
    edge_calibration_gate_min_oos_samples: int = 20
    edge_calibration_gate_max_mae_return_bps: float = 180.0
    edge_calibration_gate_max_mae_risk_bps: float = 180.0
    edge_calibration_gate_max_mae_net_edge_bps: float = 180.0
    edge_calibration_gate_min_top10_avg_return_bps: float = 5.0
    edge_calibration_paper_min_top10_avg_return_bps: float = 5.0
    edge_calibration_gate_min_top10_win_rate: float = 0.50
    edge_calibration_gate_min_top10_expectancy_bps: float = 0.0
    edge_calibration_gate_min_fill_adjusted_edge_bps: float = 60.0
    edge_calibration_gate_min_concentration_samples: int = 20
    edge_calibration_gate_max_symbol_share: float = 0.20
    edge_calibration_gate_max_sector_share: float = 0.35
    edge_calibration_gate_max_theme_share: float = 0.35
    quant_validation_enabled: bool = False
    quant_validation_min_samples: int = 1000
    quant_validation_max_samples: int = 3000
    quant_validation_folds: int = 5
    quant_validation_top_n: int = 10
    quant_validation_min_positive_fold_rate: float = 0.60
    quant_validation_min_median_net_edge_bps: float = 0.0
    quant_validation_num_trials: int = 20
    quant_validation_embargo_seconds: int = 86400
    quant_validation_rolling_ic_windows: str = "100,300,500"
    quant_validation_min_group_samples: int = 30
    quant_validation_group_limit: int = 50
    quant_validation_include_group_oos: bool = True
    quant_validation_include_rolling_ic: bool = True
    quant_validation_include_purged_walk_forward: bool = True
    overfit_guard_enabled: bool = False
    overfit_guard_db_path: str = "data/overfit_guard.sqlite3"
    overfit_guard_min_samples: int = 1000
    overfit_guard_max_samples: int = 3000
    overfit_guard_folds: int = 5
    overfit_guard_top_n: int = 10
    overfit_guard_embargo_seconds: int = 86400
    overfit_guard_min_positive_fold_rate: float = 0.60
    overfit_guard_min_median_oos_net_edge_bps: float = 0.0
    overfit_guard_live_block_enabled: bool = False
    broker_sync_db_path: str = "data/broker_sync.sqlite3"
    order_state_db_path: str = "data/order_state.sqlite3"
    order_dedupe_window_seconds: int = 120
    allow_position_additions: bool = False
    broker_paper_max_order_krw: float = 0.0
    broker_paper_max_daily_orders: int = 0
    broker_paper_max_daily_orders_per_symbol: int = 0
    broker_paper_symbol_cooldown_days: int = 0
    broker_paper_max_daily_notional_krw: float = 0.0
    broker_paper_bootstrap_enabled: bool = True
    broker_paper_calibration_source: str = "broker_fills"
    broker_paper_candidate_label_gate_mode: str = "observe_only"
    broker_paper_min_fill_samples: int = 200
    broker_paper_min_oos_fill_samples: int = 50
    broker_paper_bootstrap_relaxed_enabled: bool = True
    broker_paper_bootstrap_min_score: float = 40.0
    broker_paper_bootstrap_min_net_edge_bps: float = -20.0
    broker_paper_bootstrap_allow_watch: bool = True
    broker_paper_bootstrap_allow_promoted_watch: bool = True
    broker_paper_bootstrap_order_size_multiplier: float = 0.20
    broker_paper_bootstrap_max_order_amount_krw: float = 100_000.0
    broker_paper_bootstrap_min_expected_return_bps: float = 60.0
    broker_paper_bootstrap_min_predicted_edge_bps: float = 40.0
    broker_paper_bootstrap_max_symbols_per_cycle: int = 1
    dynamic_risk_limits_enabled: bool = True
    dynamic_risk_min_multiplier: float = 0.35
    dynamic_risk_max_multiplier: float = 1.15
    dynamic_risk_high_atr_pct: float = 0.06
    dynamic_risk_low_atr_pct: float = 0.025
    dynamic_risk_high_correlation: float = 0.75
    dynamic_risk_bear_multiplier: float = 0.6
    dynamic_risk_bull_multiplier: float = 1.05
    dynamic_risk_unknown_multiplier: float = 1.0
    market_monitor_enabled: bool = True
    market_monitor_db_path: str = "data/market_monitor.sqlite3"
    monitor_watchlist_symbols: str = ""
    monitor_market_keywords: str = "코스피,코스닥,환율,금리,반도체,AI"
    monitor_price_interval_seconds: int = 300
    monitor_disclosure_interval_seconds: int = 3600
    monitor_news_interval_seconds: int = 3600
    monitor_surge_change_pct: float = 5.0
    monitor_drop_change_pct: float = -5.0
    monitor_volume_spike_ratio: float = 3.0
    monitor_news_display: int = 20
    broker_sync_interval_seconds: int = 60
    broker_sync_config_error_backoff_seconds: int = 900
    alert_db_path: str = "data/alerts.sqlite3"
    alert_webhook_url: str | None = None
    alert_webhook_timeout: float = 5.0
    alert_min_severity: str = "high"
    alert_min_impact_strength: float = 70.0
    universe_scanner_db_path: str = "data/universe_scanner.sqlite3"
    universe_scanner_seed_symbols: str = ""
    universe_scanner_candidate_limit: int = 20
    universe_scanner_final_limit: int = 10
    universe_scanner_max_source_symbols: int = 100
    universe_scanner_symbol_interval_seconds: float = 0.0
    universe_scanner_symbol_interval_cap_seconds: float = 2.0
    universe_scanner_intraday_enrichment_enabled: bool = False
    universe_scanner_news_enrichment_enabled: bool = False
    universe_scanner_disclosure_enrichment_enabled: bool = False
    universe_scanner_min_scanned_symbols_for_trading: int = 15
    universe_scanner_max_scan_seconds: int = 300
    universe_scanner_candidate_ttl_seconds: int = 7200
    universe_scanner_min_interval_seconds: int = 900
    universe_scanner_cached_prefilter_enabled: bool = True
    universe_scanner_prefilter_max_symbols: int = 500
    universe_scanner_max_fresh_quote_symbols: int = 42
    universe_scanner_cached_snapshot_max_age_seconds: int = 86400
    universe_scanner_rotation_enabled: bool = True
    universe_scanner_rotation_window_seconds: int = 1800
    universe_scanner_cached_only_candidate_limit: int = 300
    universe_scanner_worker_hurdle_rate_bps: float = 40.0
    universe_scanner_paper_bootstrap_soft_pass_enabled: bool = True
    universe_scanner_paper_bootstrap_min_net_edge_bps: float = -20.0
    universe_scanner_paper_bootstrap_min_score: float = 40.0
    universe_scanner_buy_score_threshold: float = 70.0
    universe_scanner_watch_score_threshold: float = 40.0
    universe_scanner_paper_promote_exclude_to_watch_enabled: bool = True
    universe_scanner_paper_promote_exclude_min_score: float = 40.0
    universe_scanner_paper_promote_exclude_min_net_edge_bps: float = 0.0
    universe_scanner_default_spread_bps: float = 5.0
    universe_scanner_default_slippage_bps: float = 10.0
    universe_scanner_min_turnover_value: float = 20_000_000_000.0
    universe_scanner_min_volume: int = 300_000
    universe_include_kospi: bool = True
    universe_kospi_symbol_source: str = "csv"
    universe_kospi_scan_all: bool = True
    universe_kospi_symbol_limit: int = 0
    universe_kospi_builtin_fallback_enabled: bool = True
    universe_kospi_cache_path: str = "data/kospi_symbols.sqlite3"
    universe_kospi_csv_path: str = "data/kospi_symbols.csv"
    universe_kospi_cache_ttl_seconds: int = 86400
    universe_full_scan_enabled: bool = True
    universe_full_scan_batch_size: int = 50
    universe_full_scan_batch_pause_seconds: float = 0.2
    universe_full_scan_max_symbols: int = 0
    universe_prioritize_kospi: bool = True
    universe_cleaning_enabled: bool = True
    universe_cleaning_min_price: float = 1000.0
    universe_cleaning_min_turnover_value: float = 20_000_000_000.0
    universe_cleaning_min_volume: int = 300_000
    universe_cleaning_max_abs_change_rate: float = 18.0
    universe_cleaning_upper_limit_guard_pct: float = 27.0
    universe_cleaning_lower_limit_guard_pct: float = -27.0
    universe_cleaning_max_snapshot_age_seconds: int = 900
    universe_cleaning_exclude_warning_symbols: bool = True
    market_safety_cache_enabled: bool = True
    market_safety_db_path: str = "data/market_safety.sqlite3"
    market_safety_cache_ttl_seconds: int = 86400
    market_safety_live_block_halt: bool = True
    market_safety_live_block_managed: bool = True
    market_safety_live_block_delisting: bool = True
    market_safety_live_block_liquidation: bool = True
    market_safety_live_block_investment_risk: bool = True
    market_safety_live_block_investment_warning: bool = True
    market_safety_live_block_investment_caution: bool = False
    market_safety_paper_block_halt: bool = True
    market_safety_paper_block_delisting: bool = True
    market_safety_warning_penalty_bps: float = 40.0
    market_safety_caution_penalty_bps: float = 15.0
    market_safety_missing_cache_live_penalty_bps: float = 5.0
    corporate_event_cache_enabled: bool = True
    corporate_event_db_path: str = "data/corporate_events.sqlite3"
    corporate_event_cache_ttl_seconds: int = 86400
    corporate_event_block_live_severe: bool = True
    corporate_event_block_paper_severe: bool = False
    corporate_event_pre_event_window_days: int = 3
    corporate_event_post_event_window_days: int = 2
    corporate_event_earnings_window_days: int = 1
    corporate_event_severe_penalty_bps: float = 60.0
    corporate_event_warning_penalty_bps: float = 30.0
    corporate_event_info_penalty_bps: float = 10.0
    corporate_event_missing_cache_live_penalty_bps: float = 0.0
    universe_scanner_macro_min_risk_on_score: float = 35.0
    universe_scanner_min_market_cap: float = 300_000_000_000.0
    universe_scanner_max_market_cap: float = 2_000_000_000_000.0
    universe_scanner_large_cap_min_3d_return_bps: float = 200.0
    entry_time_filter_enabled: bool = True
    entry_time_windows: str = "09:00-10:30,14:30-15:20"
    embedded_workers_enabled: bool = os.getenv("RENDER_SERVICE_TYPE") == "web"
    embedded_worker_broker_sync_enabled: bool = True

    def storage_status(self) -> dict[str, Any]:
        """Resolve and verify the writable storage root used for relative paths."""
        key = self._storage_cache_key()
        cached = _STORAGE_STATUS_CACHE.get(key)
        if cached is not None:
            return dict(cached)

        status = self._resolve_storage_status()
        _STORAGE_STATUS_CACHE[key] = dict(status)
        self._log_storage_warnings(status, key)
        return dict(status)

    def storage_root(self) -> Path | None:
        """Return the verified writable storage root for relative DB/cache paths."""
        resolved = self.storage_status().get("resolved_data_dir")
        return Path(str(resolved)) if resolved else None

    def storage_path(self, value: str | Path) -> Path:
        """Resolve relative DB/cache paths onto the verified storage root."""
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        status = self.storage_status()
        root = Path(str(status["resolved_data_dir"])) if status.get("resolved_data_dir") else None
        if (
            root is not None
            and status.get("storage_root_source") == "local_data"
            and path.parts
            and path.parts[0] == "data"
        ):
            path = Path(*path.parts[1:]) if len(path.parts) > 1 else Path(".")
        return root / path if root else path

    def execution_mode_status(self, explicit: str | None = None) -> dict[str, Any]:
        raw: str | None = None
        source = "default"
        if explicit:
            raw = explicit
            source = "request"
        else:
            for key in (
                "EXECUTION_MODE",
                "TRADING_EXECUTION_MODE",
                "AUTO_TRADING_EXECUTION_MODE",
                "DEFAULT_EXECUTION_MODE",
            ):
                value = os.getenv(key)
                if value:
                    raw = value
                    source = key
                    break
            if raw is None and self.execution_mode:
                raw = self.execution_mode
                source = "settings.execution_mode"

        mode = str(raw or "paper").strip().lower()
        if mode not in {"paper", "broker_paper", "live"}:
            logger.warning("Unknown EXECUTION_MODE=%s; falling back to paper", raw)
            mode = "paper"
            source = f"{source}:invalid"

        return {
            "configured_execution_mode": str(raw).strip().lower() if raw else None,
            "resolved_execution_mode": mode,
            "execution_mode_source": source,
        }

    def resolved_execution_mode(self, explicit: str | None = None) -> str:
        return str(self.execution_mode_status(explicit).get("resolved_execution_mode"))

    def broker_provider_status(self, explicit: str | None = None) -> dict[str, Any]:
        raw: str | None = None
        source = "default"
        if explicit:
            raw = explicit
            source = "request"
        else:
            value = os.getenv("BROKER_PROVIDER")
            if value:
                raw = value
                source = "BROKER_PROVIDER"
            elif self.broker_provider:
                raw = self.broker_provider
                source = "settings.broker_provider"

        provider = str(raw or "kis").strip().lower()
        if provider != "kis":
            logger.warning("Unknown BROKER_PROVIDER=%s; falling back to kis", raw)
            provider = "kis"
            source = f"{source}:invalid"
        return {
            "configured_broker_provider": str(raw).strip().lower() if raw else None,
            "broker_provider": provider,
            "broker_provider_source": source,
        }

    def resolved_broker_provider(self, explicit: str | None = None) -> str:
        return str(self.broker_provider_status(explicit).get("broker_provider"))

    def broker_paper_risk_limits(self) -> dict[str, Any]:
        return {
            "max_order_krw": 0.0,
            "max_daily_orders": 0,
            "max_daily_orders_per_symbol": 0,
            "symbol_cooldown_days": 0,
            "max_daily_notional_krw": 0.0,
            "allow_position_additions": bool(self.allow_position_additions),
        }

    def clear_storage_cache(self) -> None:
        """Clear resolved storage diagnostics. Intended for tests and env reloads."""
        _STORAGE_STATUS_CACHE.clear()
        _STORAGE_WARNING_KEYS.clear()

    def _resolve_storage_status(self) -> dict[str, Any]:
        configured_data_dir = os.getenv("DATA_DIR") or self.data_dir
        configured_render_mount = (
            os.getenv("RENDER_DISK_MOUNT_PATH") or self.render_disk_mount_path
        )
        is_render = self._is_render_runtime()
        candidates = self._storage_candidates(
            configured_data_dir=configured_data_dir,
            configured_render_mount=configured_render_mount,
            is_render=is_render,
        )
        attempts: list[dict[str, Any]] = []
        selected: dict[str, Any] | None = None
        failed_before_selected = False

        for candidate in candidates:
            check = self._check_storage_candidate(
                candidate["path"],
                create_dir=candidate["create_dir"],
                require_exists=candidate["require_exists"],
            )
            attempt = {
                "source": candidate["source"],
                "path": str(candidate["path"]),
                "writable": check["writable"],
                "reason": check.get("reason"),
            }
            attempts.append(attempt)
            if check["writable"]:
                selected = {**candidate, **check}
                break
            if candidate["counts_as_failure"]:
                failed_before_selected = True

        if selected is None:
            fallback = self._tmp_fallback_data_dir()
            check = self._check_storage_candidate(
                fallback,
                create_dir=True,
                require_exists=False,
            )
            selected = {
                "source": "tmp_fallback",
                "path": fallback,
                "persistent": False,
                "create_dir": True,
                "require_exists": False,
                "counts_as_failure": True,
                **check,
            }
            attempts.append(
                {
                    "source": "tmp_fallback",
                    "path": str(fallback),
                    "writable": check["writable"],
                    "reason": check.get("reason"),
                }
            )
            failed_before_selected = True

        selected_path = Path(selected["path"]).expanduser().resolve()
        persistent = bool(selected.get("persistent")) and not (
            is_render and self._is_tmp_path(selected_path)
        )
        fallback_used = self._storage_fallback_used(
            selected_source=str(selected.get("source")),
            failed_before_selected=failed_before_selected,
            configured_data_dir=configured_data_dir,
            configured_render_mount=configured_render_mount,
            is_render=is_render,
        )
        warnings = self._storage_warnings(
            attempts=attempts,
            selected_path=selected_path,
            selected_source=str(selected.get("source")),
            persistent=persistent,
            fallback_used=fallback_used,
            configured_data_dir=configured_data_dir,
            is_render=is_render,
        )

        configured = configured_data_dir or configured_render_mount
        return {
            "configured_data_dir": str(Path(configured).expanduser()) if configured else None,
            "resolved_data_dir": str(selected_path),
            "data_dir_writable": bool(selected.get("writable")),
            "data_dir_is_persistent": persistent,
            "data_dir_warning": "; ".join(warnings) if warnings else None,
            "storage_root_fallback_used": fallback_used,
            "storage_root_source": str(selected.get("source")),
            "storage_attempts": attempts,
        }

    def _storage_candidates(
        self,
        *,
        configured_data_dir: str | None,
        configured_render_mount: str | None,
        is_render: bool,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add(
            source: str,
            raw_path: str | Path,
            *,
            persistent: bool,
            create_dir: bool,
            require_exists: bool,
            counts_as_failure: bool,
        ) -> None:
            path = Path(raw_path).expanduser()
            key = str(path)
            if key in seen:
                return
            seen.add(key)
            candidates.append(
                {
                    "source": source,
                    "path": path,
                    "persistent": persistent,
                    "create_dir": create_dir,
                    "require_exists": require_exists,
                    "counts_as_failure": counts_as_failure,
                }
            )

        if configured_data_dir:
            add(
                "DATA_DIR",
                configured_data_dir,
                persistent=True,
                create_dir=True,
                require_exists=False,
                counts_as_failure=True,
            )
        if configured_render_mount:
            add(
                "RENDER_DISK_MOUNT_PATH",
                configured_render_mount,
                persistent=True,
                create_dir=True,
                require_exists=False,
                counts_as_failure=True,
            )
        add(
            "default_var_data",
            self._default_render_data_dir(),
            persistent=True,
            create_dir=False,
            require_exists=True,
            counts_as_failure=is_render,
        )
        add(
            "local_data",
            self._local_data_dir(),
            persistent=not is_render,
            create_dir=True,
            require_exists=False,
            counts_as_failure=is_render,
        )
        add(
            "tmp_fallback",
            self._tmp_fallback_data_dir(),
            persistent=False,
            create_dir=True,
            require_exists=False,
            counts_as_failure=True,
        )
        return candidates

    def _check_storage_candidate(
        self,
        path: Path,
        *,
        create_dir: bool,
        require_exists: bool,
    ) -> dict[str, Any]:
        path = path.expanduser()
        sqlite_path: Path | None = None
        try:
            if require_exists and not path.exists():
                return {"writable": False, "reason": "path does not exist"}
            if path.exists() and not path.is_dir():
                return {"writable": False, "reason": "path is not a directory"}
            if create_dir:
                path.mkdir(parents=True, exist_ok=True)
            elif not path.exists():
                return {"writable": False, "reason": "path does not exist"}

            write_path = path / f".storage-write-check-{uuid4().hex}.tmp"
            write_path.write_text("ok", encoding="utf-8")
            write_path.unlink(missing_ok=True)

            sqlite_path = path / f".storage-sqlite-check-{uuid4().hex}.sqlite3"
            with sqlite3.connect(sqlite_path) as conn:
                conn.execute("CREATE TABLE IF NOT EXISTS storage_check (id INTEGER)")
                conn.execute("INSERT INTO storage_check (id) VALUES (1)")
                conn.commit()
            self._remove_sqlite_check_files(sqlite_path)
            return {"writable": True, "reason": None}
        except Exception as exc:
            if sqlite_path is not None:
                self._remove_sqlite_check_files(sqlite_path)
            return {"writable": False, "reason": str(exc)}

    def _remove_sqlite_check_files(self, sqlite_path: Path) -> None:
        for suffix in ("", "-wal", "-shm", "-journal"):
            try:
                Path(f"{sqlite_path}{suffix}").unlink(missing_ok=True)
            except OSError:
                pass

    def _storage_fallback_used(
        self,
        *,
        selected_source: str,
        failed_before_selected: bool,
        configured_data_dir: str | None,
        configured_render_mount: str | None,
        is_render: bool,
    ) -> bool:
        if failed_before_selected:
            return True
        if configured_data_dir and selected_source != "DATA_DIR":
            return True
        if (
            not configured_data_dir
            and configured_render_mount
            and selected_source != "RENDER_DISK_MOUNT_PATH"
        ):
            return True
        if is_render and selected_source in {"local_data", "tmp_fallback"}:
            return True
        return False

    def _storage_warnings(
        self,
        *,
        attempts: list[dict[str, Any]],
        selected_path: Path,
        selected_source: str,
        persistent: bool,
        fallback_used: bool,
        configured_data_dir: str | None,
        is_render: bool,
    ) -> list[str]:
        warnings: list[str] = []
        var_path = self._default_render_data_dir().expanduser()
        configured_path = (
            Path(configured_data_dir).expanduser() if configured_data_dir else None
        )
        failed_attempts = [attempt for attempt in attempts if not attempt["writable"]]
        if any(
            attempt["source"] in {"DATA_DIR", "RENDER_DISK_MOUNT_PATH"}
            for attempt in failed_attempts
        ):
            warnings.append(
                "WARNING: configured storage path is not writable; using fallback storage."
            )
        if is_render and any(
            attempt["source"] == "default_var_data" for attempt in failed_attempts
        ):
            warnings.append(_RENDER_VAR_DATA_WARNING)
        if (
            is_render
            and configured_path is not None
            and self._same_path(configured_path, var_path)
            and any(
                attempt["source"] == "DATA_DIR" and not attempt["writable"]
                for attempt in attempts
            )
        ):
            warnings.append(_RENDER_VAR_DATA_WARNING)

        if fallback_used and not persistent:
            warnings.append(
                _RENDER_NON_PERSISTENT_WARNING
                if is_render
                else _LOCAL_NON_PERSISTENT_WARNING
            )
        elif is_render and self._is_tmp_path(selected_path):
            warnings.append(_RENDER_NON_PERSISTENT_WARNING)
        return list(dict.fromkeys(warnings))

    def _log_storage_warnings(
        self,
        status: dict[str, Any],
        key: tuple[Any, ...],
    ) -> None:
        if key in _STORAGE_WARNING_KEYS:
            return
        _STORAGE_WARNING_KEYS.add(key)
        warning = status.get("data_dir_warning")
        if not warning:
            return
        for item in str(warning).split("; "):
            if item:
                logger.warning(item)

    def _storage_cache_key(self) -> tuple[Any, ...]:
        return (
            os.getenv("DATA_DIR"),
            self.data_dir,
            os.getenv("RENDER_DISK_MOUNT_PATH"),
            self.render_disk_mount_path,
            os.getenv("RENDER"),
            os.getenv("RENDER_SERVICE_ID"),
            os.getenv("RENDER_SERVICE_TYPE"),
            str(self._default_render_data_dir()),
            str(self._local_data_dir()),
            str(self._tmp_fallback_data_dir()),
            os.getcwd(),
        )

    def _is_render_runtime(self) -> bool:
        return bool(
            os.getenv("RENDER")
            or os.getenv("RENDER_SERVICE_ID")
            or os.getenv("RENDER_SERVICE_TYPE")
        )

    def _default_render_data_dir(self) -> Path:
        return Path("/var/data")

    def _local_data_dir(self) -> Path:
        return Path("data")

    def _tmp_fallback_data_dir(self) -> Path:
        return Path("/tmp/investment_orchestrator_data")

    def _is_tmp_path(self, path: Path) -> bool:
        return str(path.expanduser()).replace("\\", "/").startswith("/tmp/")

    def _same_path(self, left: Path, right: Path) -> bool:
        return str(left.expanduser()).rstrip("\\/") == str(right.expanduser()).rstrip("\\/")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
