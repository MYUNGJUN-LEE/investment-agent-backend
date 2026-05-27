from app.config import settings
from app.maintenance.data_reset import RESET_CONFIRMATION, reset_trading_data


def test_reset_trading_data_requires_exact_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    path = tmp_path / "data" / "edge_calibration.sqlite3"
    path.parent.mkdir(parents=True)
    path.write_text("db", encoding="utf-8")

    result = reset_trading_data(confirm="wrong")

    assert result["status"] == "blocked"
    assert path.exists()


def test_reset_trading_data_deletes_generated_files_from_configured_data_dir(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    keep = data_dir / "notes.txt"
    targets = [
        data_dir / "edge_calibration.sqlite3",
        data_dir / "edge_calibration.sqlite3-wal",
        data_dir / "kis_token_cache.json",
        data_dir / "corp_map.csv",
    ]
    for path in targets:
        path.write_text("generated", encoding="utf-8")
    keep.write_text("keep", encoding="utf-8")

    result = reset_trading_data(confirm=RESET_CONFIRMATION)

    assert result["status"] == "success"
    assert result["deleted_count"] == len(targets)
    assert all(not path.exists() for path in targets)
    assert keep.exists()
