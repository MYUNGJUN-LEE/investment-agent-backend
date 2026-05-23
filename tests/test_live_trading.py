from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.trading import live_trading, paper_trading, risk_manager


client = TestClient(app)


def _auth_headers() -> dict[str, str]:
    if settings.backend_api_key:
        return {"X-API-Key": settings.backend_api_key}
    return {}


def _payload(**overrides):
    payload = {
        "symbol": "005930",
        "market": "KR",
        "side": "buy",
        "order_type": "limit",
        "price": 75000,
        "quantity": 3,
        "decision_price": 75000,
        "confirm_token": "confirm-live",
        "reason": "manual approval",
    }
    payload.update(overrides)
    return payload


def test_live_order_endpoint_is_disabled_by_default(monkeypatch):
    monkeypatch.setattr(settings, "enable_live_trading", False)
    monkeypatch.setattr(settings, "kis_is_paper", False)
    monkeypatch.setattr(settings, "live_trading_confirm_token", "confirm-live")

    response = client.post(
        "/live/orders",
        headers=_auth_headers(),
        json=_payload(),
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Live trading is disabled"


def test_live_order_endpoint_rejects_paper_mode(monkeypatch):
    monkeypatch.setattr(settings, "enable_live_trading", True)
    monkeypatch.setattr(settings, "kis_is_paper", True)
    monkeypatch.setattr(settings, "live_trading_confirm_token", "confirm-live")

    response = client.post(
        "/live/orders",
        headers=_auth_headers(),
        json=_payload(),
    )

    assert response.status_code == 403
    assert "KIS_IS_PAPER=false" in response.json()["detail"]


def test_live_order_endpoint_rejects_bad_confirm_token(monkeypatch):
    monkeypatch.setattr(settings, "enable_live_trading", True)
    monkeypatch.setattr(settings, "kis_is_paper", False)
    monkeypatch.setattr(settings, "live_trading_confirm_token", "confirm-live")

    response = client.post(
        "/live/orders",
        headers=_auth_headers(),
        json=_payload(confirm_token="wrong"),
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid live trading confirm_token"


def test_live_order_endpoint_rejects_market_order(monkeypatch):
    monkeypatch.setattr(settings, "enable_live_trading", True)
    monkeypatch.setattr(settings, "kis_is_paper", False)
    monkeypatch.setattr(settings, "live_trading_confirm_token", "confirm-live")

    response = client.post(
        "/live/orders",
        headers=_auth_headers(),
        json=_payload(order_type="market"),
    )

    assert response.status_code == 422


def test_live_order_requires_decision_price(monkeypatch):
    monkeypatch.setattr(settings, "enable_live_trading", True)
    monkeypatch.setattr(settings, "kis_is_paper", False)
    monkeypatch.setattr(settings, "live_trading_confirm_token", "confirm-live")
    payload = _payload()
    payload.pop("decision_price")

    response = client.post(
        "/live/orders",
        headers=_auth_headers(),
        json=payload,
    )

    assert response.status_code == 422


def test_live_order_must_pass_risk_manager(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "enable_live_trading", True)
    monkeypatch.setattr(settings, "kis_is_paper", False)
    monkeypatch.setattr(settings, "live_trading_confirm_token", "confirm-live")
    monkeypatch.setattr(paper_trading, "DEFAULT_DB_PATH", tmp_path / "paper.sqlite3")

    def reject_order(*args, **kwargs):
        return risk_manager.RiskDecision(
            approved=False,
            code="forced_risk_rejection",
            message="forced risk rejection",
            checks={},
        )

    monkeypatch.setattr(live_trading.risk_manager, "approve_order", reject_order)

    response = client.post(
        "/live/orders",
        headers=_auth_headers(),
        json=_payload(),
    )

    assert response.status_code == 403
    assert "forced_risk_rejection" in response.json()["detail"]


def test_live_order_submits_limit_order_after_all_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "enable_live_trading", True)
    monkeypatch.setattr(settings, "kis_is_paper", False)
    monkeypatch.setattr(settings, "live_trading_confirm_token", "confirm-live")
    monkeypatch.setattr(paper_trading, "DEFAULT_DB_PATH", tmp_path / "paper.sqlite3")
    monkeypatch.setattr(settings, "order_state_db_path", str(tmp_path / "state.sqlite3"))

    def approve_order(*args, **kwargs):
        return risk_manager.RiskDecision(
            approved=True,
            code="approved",
            message="ok",
            checks={},
        )

    calls = []

    class FakeKisClient:
        def __init__(self, is_paper: bool):
            self.is_paper = is_paper

        def place_domestic_limit_order(self, symbol, side, price, quantity):
            calls.append(
                {
                    "is_paper": self.is_paper,
                    "symbol": symbol,
                    "side": side,
                    "price": price,
                    "quantity": quantity,
                }
            )
            return {
                "rt_cd": "0",
                "msg_cd": "APBK0013",
                "msg1": "주문 전송 완료",
                "output": {"ODNO": "12345"},
            }

    monkeypatch.setattr(live_trading.risk_manager, "approve_order", approve_order)
    monkeypatch.setattr(live_trading, "KisClient", FakeKisClient)

    response = client.post(
        "/live/orders",
        headers=_auth_headers(),
        json=_payload(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "submitted"
    assert body["order_type"] == "limit"
    assert body["kis_result"]["rt_cd"] == "0"
    assert body["order_state"]["state"] == "ENTRY_PENDING"
    assert calls == [
        {
            "is_paper": False,
            "symbol": "005930",
            "side": "buy",
            "price": 75000.0,
            "quantity": 3,
        }
    ]
