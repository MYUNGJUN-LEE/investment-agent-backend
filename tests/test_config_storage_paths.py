import logging
import sqlite3
from pathlib import Path

from app.config import settings
from app.trading import edge_calibration, execution_status, universe_scanner
from app.trading import auto_trading_store, order_state


def test_storage_path_routes_relative_paths_to_configured_data_dir(tmp_path, monkeypatch):
    _clear_storage_env(monkeypatch)
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(settings, "render_disk_mount_path", None)

    assert settings.storage_path("data/auto.sqlite3") == tmp_path / "data" / "auto.sqlite3"


def test_storage_path_preserves_absolute_paths(tmp_path, monkeypatch):
    _clear_storage_env(monkeypatch)
    monkeypatch.setattr(settings, "data_dir", str(tmp_path / "disk"))
    absolute = tmp_path / "explicit.sqlite3"

    assert settings.storage_path(absolute) == absolute


def test_storage_path_uses_render_disk_env_when_settings_are_empty(tmp_path, monkeypatch):
    _clear_storage_env(monkeypatch)
    monkeypatch.setattr(settings, "data_dir", None)
    monkeypatch.setattr(settings, "render_disk_mount_path", None)
    monkeypatch.setenv("RENDER_DISK_MOUNT_PATH", str(tmp_path))

    assert settings.storage_path(Path("data/edge.sqlite3")) == tmp_path / "data" / "edge.sqlite3"


def test_storage_path_defaults_to_var_data_only_when_writable(tmp_path, monkeypatch):
    _clear_storage_env(monkeypatch)
    var_data = tmp_path / "var_data"
    var_data.mkdir()
    monkeypatch.setattr(settings, "data_dir", None)
    monkeypatch.setattr(settings, "render_disk_mount_path", None)
    monkeypatch.setattr(
        type(settings),
        "_default_render_data_dir",
        lambda self: var_data,
    )
    monkeypatch.setenv("RENDER_SERVICE_TYPE", "web")

    assert settings.storage_path(Path("data/edge.sqlite3")) == var_data / "data" / "edge.sqlite3"


def test_var_data_not_writable_falls_back_and_logs_warning(
    tmp_path,
    monkeypatch,
    caplog,
):
    _clear_storage_env(monkeypatch)
    var_data = tmp_path / "var_data"
    var_data.write_text("not a directory", encoding="utf-8")
    local_data = tmp_path / "local_data"
    tmp_fallback = tmp_path / "tmp_fallback"
    monkeypatch.setattr(settings, "data_dir", None)
    monkeypatch.setattr(settings, "render_disk_mount_path", None)
    monkeypatch.setattr(
        type(settings),
        "_default_render_data_dir",
        lambda self: var_data,
    )
    monkeypatch.setattr(type(settings), "_local_data_dir", lambda self: local_data)
    monkeypatch.setattr(
        type(settings),
        "_tmp_fallback_data_dir",
        lambda self: tmp_fallback,
    )
    monkeypatch.setenv("RENDER", "true")
    caplog.set_level(logging.WARNING)

    status = settings.storage_status()

    assert Path(status["resolved_data_dir"]) == local_data
    assert status["data_dir_writable"] is True
    assert status["storage_root_fallback_used"] is True
    assert "WARNING: /var/data is not writable" in caplog.text


def test_broker_paper_blocks_submit_on_non_persistent_fallback(
    tmp_path,
    monkeypatch,
):
    _clear_storage_env(monkeypatch)
    var_data = tmp_path / "var_data"
    local_data = tmp_path / "local_data"
    var_data.write_text("not a directory", encoding="utf-8")
    local_data.write_text("not a directory", encoding="utf-8")
    tmp_fallback = tmp_path / "tmp_fallback"
    monkeypatch.setattr(settings, "data_dir", None)
    monkeypatch.setattr(settings, "render_disk_mount_path", None)
    monkeypatch.setattr(settings, "kis_is_paper", True)
    monkeypatch.setattr(
        type(settings),
        "_default_render_data_dir",
        lambda self: var_data,
    )
    monkeypatch.setattr(type(settings), "_local_data_dir", lambda self: local_data)
    monkeypatch.setattr(
        type(settings),
        "_tmp_fallback_data_dir",
        lambda self: tmp_fallback,
    )
    monkeypatch.setenv("RENDER", "true")

    status = execution_status.trading_status_snapshot(execution_mode="broker_paper")

    assert Path(status["resolved_data_dir"]) == tmp_fallback
    assert status["data_dir_writable"] is True
    assert status["data_dir_is_persistent"] is False
    assert status["storage_root_fallback_used"] is True
    assert status["broker_submit_blocked"] is True
    assert status["broker_submit_block_reason"] == "persistent_order_storage_unavailable"
    assert status["submits_to_broker"] is False


def test_valid_data_dir_places_sqlite_dbs_under_that_directory(
    tmp_path,
    monkeypatch,
):
    _clear_storage_env(monkeypatch)
    data_dir = tmp_path / "persistent"
    monkeypatch.setattr(settings, "data_dir", None)
    monkeypatch.setattr(settings, "render_disk_mount_path", None)
    monkeypatch.setenv("DATA_DIR", str(data_dir))

    auto_trading_store.initialize_auto_trading_db()
    order_state.initialize_order_state_db()
    universe_scanner.initialize_universe_db()
    edge_calibration.initialize_edge_calibration_db()

    db_paths = [
        settings.storage_path(settings.auto_trading_db_path),
        settings.storage_path(settings.order_state_db_path),
        settings.storage_path(settings.universe_scanner_db_path),
        settings.storage_path(settings.edge_calibration_db_path),
    ]

    for path in db_paths:
        assert str(path).startswith(str(data_dir))
        assert path.exists()
        with sqlite3.connect(path) as conn:
            assert conn.execute("SELECT 1").fetchone()[0] == 1


def _clear_storage_env(monkeypatch) -> None:
    settings.clear_storage_cache()
    for key in (
        "DATA_DIR",
        "APP_DATA_DIR",
        "RENDER_DISK_MOUNT_PATH",
        "RENDER_PERSISTENT_DISK_PATH",
        "PERSISTENT_DISK_PATH",
        "RENDER",
        "RENDER_SERVICE_ID",
        "RENDER_SERVICE_TYPE",
        "EXECUTION_MODE",
        "TRADING_EXECUTION_MODE",
        "AUTO_TRADING_EXECUTION_MODE",
        "DEFAULT_EXECUTION_MODE",
        "BROKER_PROVIDER",
    ):
        monkeypatch.delenv(key, raising=False)
