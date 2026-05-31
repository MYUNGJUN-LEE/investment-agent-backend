import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


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
    outcome_attribution_max_records: int = 5000
    outcome_attribution_min_hold_seconds: int = 60
    outcome_attribution_recent_limit: int = 200
    auto_tuning_enabled: bool = True
    auto_tuning_mode: str = "recommend"
    auto_tuning_db_path: str = "data/auto_tuning.sqlite3"
    auto_tuning_recent_limit: int = 200
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
    edge_calibration_future_price_limit: int = 1200
    edge_calibration_label_snapshots_enabled: bool = True
    edge_calibration_label_snapshot_max_symbols: int = 200
    edge_calibration_refresh_batch_size: int = 500
    edge_calibration_refresh_after_scan: bool = False
    edge_calibration_min_label_age_seconds: int = 300
    edge_calibration_min_future_snapshots: int = 2
    edge_calibration_label_at_horizon_end: bool = True
    edge_calibration_label_horizon_tolerance_seconds: int = 3600
    edge_calibration_candidate_lookback_seconds: int | None = None
    edge_calibration_ridge_lambda: float = 10.0
    edge_calibration_blend: float = 0.35
    edge_calibration_target_samples: int = 3000
    edge_calibration_sample_retention_limit: int = 10_000
    edge_calibration_gate_min_samples: int = 600
    edge_calibration_gate_min_oos_samples: int = 200
    edge_calibration_gate_max_mae_return_bps: float = 180.0
    edge_calibration_gate_max_mae_risk_bps: float = 180.0
    edge_calibration_gate_min_top10_avg_return_bps: float = 5.0
    edge_calibration_paper_min_top10_avg_return_bps: float = 5.0
    edge_calibration_gate_min_top10_win_rate: float = 0.50
    edge_calibration_gate_min_top10_expectancy_bps: float = 0.0
    edge_calibration_gate_min_fill_adjusted_edge_bps: float = 60.0
    quant_validation_enabled: bool = False
    quant_validation_min_samples: int = 1000
    quant_validation_max_samples: int = 3000
    quant_validation_folds: int = 5
    quant_validation_top_n: int = 10
    quant_validation_min_positive_fold_rate: float = 0.60
    quant_validation_min_median_net_edge_bps: float = 0.0
    quant_validation_num_trials: int = 20
    broker_sync_db_path: str = "data/broker_sync.sqlite3"
    order_state_db_path: str = "data/order_state.sqlite3"
    order_dedupe_window_seconds: int = 120
    allow_position_additions: bool = False
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
    universe_scanner_worker_hurdle_rate_bps: float = 40.0
    universe_scanner_default_spread_bps: float = 5.0
    universe_scanner_default_slippage_bps: float = 10.0
    universe_scanner_min_turnover_value: float = 20_000_000_000.0
    universe_scanner_min_volume: int = 300_000
    universe_scanner_macro_min_risk_on_score: float = 35.0
    universe_scanner_min_market_cap: float = 300_000_000_000.0
    universe_scanner_max_market_cap: float = 2_000_000_000_000.0
    universe_scanner_large_cap_min_3d_return_bps: float = 500.0
    entry_time_filter_enabled: bool = True
    entry_time_windows: str = "09:00-10:30,14:30-15:20"
    embedded_workers_enabled: bool = os.getenv("RENDER_SERVICE_TYPE") == "web"
    embedded_worker_broker_sync_enabled: bool = True

    def storage_root(self) -> Path | None:
        """Return the persistent storage root when one is configured/attached."""
        for raw in (
            self.data_dir,
            self.render_disk_mount_path,
            os.getenv("DATA_DIR"),
            os.getenv("APP_DATA_DIR"),
            os.getenv("RENDER_DISK_MOUNT_PATH"),
            os.getenv("RENDER_PERSISTENT_DISK_PATH"),
            os.getenv("PERSISTENT_DISK_PATH"),
        ):
            if raw:
                return Path(raw).expanduser()

        if (
            os.getenv("RENDER")
            or os.getenv("RENDER_SERVICE_ID")
            or os.getenv("RENDER_SERVICE_TYPE")
        ):
            for raw in ("/var/data", "/data"):
                candidate = Path(raw)
                if candidate.exists():
                    return candidate
        return None

    def storage_path(self, value: str | Path) -> Path:
        """Resolve relative DB/cache paths onto persistent storage when present."""
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        root = self.storage_root()
        return root / path if root else path

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
