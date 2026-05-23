import pytest

from app.config import settings


@pytest.fixture(autouse=True)
def disable_backend_api_key_by_default(monkeypatch):
    monkeypatch.setattr(settings, "backend_api_key", None)
