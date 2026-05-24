import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Investment Agent Backend"

    # If this is set, requests must include X-API-Key with the same value.
    backend_api_key: str | None = None

    # Comma-separated browser origins allowed to call this API.
    cors_allow_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

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

    # Live trading is disabled unless both flags are explicitly set.
    enable_live_trading: bool = False
    live_trading_confirm_token: str | None = None

    # Trading cost assumptions. Override these per broker/account in .env.
    commission_rate: float = 0.00015
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
    auto_trading_db_path: str = "data/auto_trading.sqlite3"
    auto_trading_worker_poll_seconds: float = 2.0
    auto_trading_worker_lock_seconds: int = 7200
    auto_trading_symbol_workers: int = 1
    auto_trading_max_open_positions: int = 5
    trade_orchestrator_enabled: bool = True
    trade_orchestrator_interval_seconds: int = 60
    trade_orchestrator_execute_entries: bool = True
    edge_calibration_enabled: bool = True
    edge_calibration_db_path: str = "data/edge_calibration.sqlite3"
    edge_calibration_interval_seconds: int = 3600
    edge_calibration_horizon_seconds: int = 86400
    edge_calibration_max_samples: int = 1000
    edge_calibration_min_samples: int = 30
    edge_calibration_future_price_limit: int = 96
    edge_calibration_ridge_lambda: float = 10.0
    edge_calibration_blend: float = 0.35
    edge_calibration_target_samples: int = 1000
    edge_calibration_sample_retention_limit: int = 10_000
    edge_calibration_gate_min_samples: int = 1000
    edge_calibration_gate_min_oos_samples: int = 200
    edge_calibration_gate_max_mae_return_bps: float = 180.0
    edge_calibration_gate_max_mae_risk_bps: float = 180.0
    edge_calibration_gate_min_top10_avg_return_bps: float = 20.0
    edge_calibration_gate_min_top10_win_rate: float = 0.52
    edge_calibration_gate_min_fill_adjusted_edge_bps: float = 30.0
    broker_sync_db_path: str = "data/broker_sync.sqlite3"
    order_state_db_path: str = "data/order_state.sqlite3"
    order_dedupe_window_seconds: int = 120
    allow_position_additions: bool = False
    market_monitor_enabled: bool = True
    market_monitor_db_path: str = "data/market_monitor.sqlite3"
    monitor_watchlist_symbols: str = ""
    monitor_market_keywords: str = "코스피,코스닥,환율,금리,반도체,AI"
    monitor_price_interval_seconds: int = 60
    monitor_disclosure_interval_seconds: int = 300
    monitor_news_interval_seconds: int = 600
    monitor_surge_change_pct: float = 5.0
    monitor_drop_change_pct: float = -5.0
    monitor_volume_spike_ratio: float = 3.0
    monitor_default_stop_loss_pct: float = 3.0
    monitor_default_take_profit_pct: float = 5.0
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
    universe_scanner_max_source_symbols: int = 15
    universe_scanner_symbol_interval_seconds: float = 60.0
    universe_scanner_min_scanned_symbols_for_trading: int = 15
    universe_scanner_candidate_ttl_seconds: int = 3600
    universe_scanner_worker_hurdle_rate_bps: float = 50.0
    universe_scanner_default_spread_bps: float = 5.0
    universe_scanner_default_slippage_bps: float = 10.0
    embedded_workers_enabled: bool = os.getenv("RENDER_SERVICE_TYPE") == "web"
    embedded_worker_broker_sync_enabled: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
