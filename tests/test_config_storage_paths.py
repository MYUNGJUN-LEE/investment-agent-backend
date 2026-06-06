from pathlib import Path

from app.config import settings


def test_storage_path_routes_relative_paths_to_configured_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(settings, "render_disk_mount_path", None)

    assert settings.storage_path("data/auto.sqlite3") == tmp_path / "data" / "auto.sqlite3"


def test_storage_path_preserves_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path / "disk"))
    absolute = tmp_path / "explicit.sqlite3"

    assert settings.storage_path(absolute) == absolute


def test_storage_path_uses_render_disk_env_when_settings_are_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", None)
    monkeypatch.setattr(settings, "render_disk_mount_path", None)
    monkeypatch.setenv("RENDER_DISK_MOUNT_PATH", str(tmp_path))

    assert settings.storage_path(Path("data/edge.sqlite3")) == tmp_path / "data" / "edge.sqlite3"


def test_storage_path_defaults_to_var_data_on_render(monkeypatch):
    monkeypatch.setattr(settings, "data_dir", None)
    monkeypatch.setattr(settings, "render_disk_mount_path", None)
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.delenv("APP_DATA_DIR", raising=False)
    monkeypatch.delenv("RENDER_DISK_MOUNT_PATH", raising=False)
    monkeypatch.delenv("RENDER_PERSISTENT_DISK_PATH", raising=False)
    monkeypatch.delenv("PERSISTENT_DISK_PATH", raising=False)
    monkeypatch.setenv("RENDER_SERVICE_TYPE", "web")

    assert settings.storage_path(Path("data/edge.sqlite3")) == Path("/var/data") / "data" / "edge.sqlite3"
