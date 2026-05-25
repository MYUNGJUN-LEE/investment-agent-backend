from __future__ import annotations

from app.data_sources import kis


def test_fetch_price_data_returns_placeholder_without_credentials(monkeypatch):
    monkeypatch.setattr(kis.settings, "kis_app_key", None)
    monkeypatch.setattr(kis.settings, "kis_app_secret", None)

    result = kis.fetch_price_data("005930")

    assert result["status"] == "not_connected_yet"
    assert result["symbol"] == "005930"
    assert result["current_price"] is None


def test_fetch_price_data_uses_kis_client(monkeypatch):
    monkeypatch.setattr(kis.settings, "kis_app_key", "app-key")
    monkeypatch.setattr(kis.settings, "kis_app_secret", "app-secret")
    monkeypatch.setattr(kis, "record_price_snapshot", lambda result: None)

    class FakeKisClient:
        def get_current_price(self, symbol: str):
            assert symbol == "005930"
            return {
                "rt_cd": "0",
                "output": {
                    "stck_prpr": "75000",
                    "prdy_ctrt": "1.25",
                    "acml_vol": "1234567",
                    "acml_tr_pbmn": "92500000000",
                },
            }

        def get_daily_prices(self, symbol: str, start_date: str, end_date: str):
            return {
                "rt_cd": "0",
                "output2": [
                    {
                        "stck_bsop_date": "20260520",
                        "stck_clpr": "74000",
                        "stck_oprc": "73000",
                        "stck_hgpr": "76000",
                        "stck_lwpr": "72000",
                        "acml_vol": "800000",
                    },
                    *[
                        {
                            "stck_bsop_date": f"202605{day:02d}",
                            "stck_clpr": "70000",
                            "stck_oprc": "69500",
                            "stck_hgpr": "71000",
                            "stck_lwpr": "69000",
                            "acml_vol": "500000",
                        }
                        for day in range(1, 21)
                    ],
                ],
            }

        def get_minute_prices(self, symbol: str):
            assert symbol == "005930"
            return {
                "rt_cd": "0",
                "output2": [
                    {
                        "stck_bsop_date": "20260521",
                        "stck_cntg_hour": "090300",
                        "stck_prpr": "75200",
                        "stck_oprc": "75000",
                        "stck_hgpr": "75200",
                        "stck_lwpr": "74900",
                        "cntg_vol": "3500",
                    },
                    {
                        "stck_bsop_date": "20260521",
                        "stck_cntg_hour": "090100",
                        "stck_prpr": "74500",
                        "stck_oprc": "74400",
                        "stck_hgpr": "74600",
                        "stck_lwpr": "74300",
                        "cntg_vol": "1000",
                    },
                    {
                        "stck_bsop_date": "20260521",
                        "stck_cntg_hour": "090200",
                        "stck_prpr": "74800",
                        "stck_oprc": "74500",
                        "stck_hgpr": "74900",
                        "stck_lwpr": "74400",
                        "cntg_vol": "1800",
                    },
                ],
            }

        def get_orderbook(self, symbol: str):
            assert symbol == "005930"
            return {
                "rt_cd": "0",
                "output1": {
                    "askp1": "75010",
                    "bidp1": "75000",
                    "askp_rsqn1": "8000",
                    "bidp_rsqn1": "12000",
                    "total_askp_rsqn": "8000",
                    "total_bidp_rsqn": "12000",
                },
                "output2": {
                    "antc_cnpr": "75000",
                    "antc_cntg_vol": "3000",
                },
            }

        def get_executions(self, symbol: str):
            assert symbol == "005930"
            return {
                "rt_cd": "0",
                "output": [
                    {
                        "stck_cntg_hour": "090301",
                        "stck_prpr": "75000",
                        "cntg_vol": "500",
                        "tday_rltv": "135",
                        "prdy_vrss_sign": "2",
                    },
                    {
                        "stck_cntg_hour": "090300",
                        "stck_prpr": "74900",
                        "cntg_vol": "400",
                        "tday_rltv": "128",
                        "prdy_vrss_sign": "2",
                    },
                ],
            }

        def get_investor_flow(self, symbol: str):
            assert symbol == "005930"
            return {
                "rt_cd": "0",
                "output": [
                    {
                        "stck_bsop_date": "20260521",
                        "prsn_ntby_qty": "-1500",
                        "frgn_ntby_qty": "1000",
                        "orgn_ntby_qty": "500",
                    }
                ],
            }

        def get_investor_daily(self, symbol: str, start_date: str):
            assert symbol == "005930"
            return {
                "rt_cd": "0",
                "output": [
                    {
                        "stck_bsop_date": f"202605{day:02d}",
                        "prsn_ntby_qty": "-1500",
                        "frgn_ntby_qty": "1000",
                        "orgn_ntby_qty": "500",
                    }
                    for day in range(17, 22)
                ],
            }

    monkeypatch.setattr(kis, "KisClient", FakeKisClient)

    result = kis.fetch_price_data("005930")

    assert result["status"] == "connected"
    assert result["current_price"] == 75000
    assert result["change_rate"] == 1.25
    assert result["volume"] == 1234567
    assert result["volume_ratio"] > 1
    assert result["trend"] == "uptrend"
    assert result["moving_averages"]["ma5"] is not None
    assert result["intraday"]["intraday_score"] >= 70
    assert result["intraday"]["execution_strength"] == 135
    assert result["orderbook"]["orderbook_imbalance"] == 0.2
    assert result["investor_flow"]["smart_money_net_buy_5d"] == 7500
    assert result["optional_errors"] == []
    assert result["source"] == "KIS Open API"


def test_fetch_price_data_can_skip_intraday_optional_calls(monkeypatch):
    monkeypatch.setattr(kis.settings, "kis_app_key", "app-key")
    monkeypatch.setattr(kis.settings, "kis_app_secret", "app-secret")
    monkeypatch.setattr(kis, "record_price_snapshot", lambda result: None)

    class FastKisClient:
        def get_current_price(self, symbol: str):
            return {
                "rt_cd": "0",
                "output": {
                    "stck_prpr": "75000",
                    "prdy_ctrt": "1.25",
                    "acml_vol": "1234567",
                    "acml_tr_pbmn": "92500000000",
                },
            }

        def get_daily_prices(self, symbol: str, start_date: str, end_date: str):
            return {
                "rt_cd": "0",
                "output2": [
                    {
                        "stck_bsop_date": f"202605{day:02d}",
                        "stck_clpr": str(70000 + day * 100),
                        "stck_oprc": "70000",
                        "stck_hgpr": "76000",
                        "stck_lwpr": "69000",
                        "acml_vol": "800000",
                    }
                    for day in range(1, 25)
                ],
            }

        def get_minute_prices(self, symbol: str):
            raise AssertionError("intraday API should be skipped")

    monkeypatch.setattr(kis, "KisClient", FastKisClient)

    result = kis.fetch_price_data("005930", include_intraday=False)

    assert result["status"] == "connected"
    assert result["intraday_enriched"] is False
    assert result["current_price"] == 75000
    assert result["minute_candles"] == []
    assert result["orderbook"] == {}
    assert result["executions"] == {}
    assert result["investor_flow"] == {}
    assert result["optional_errors"] == []


def test_fetch_price_data_returns_error_when_kis_client_fails(monkeypatch):
    monkeypatch.setattr(kis.settings, "kis_app_key", "app-key")
    monkeypatch.setattr(kis.settings, "kis_app_secret", "app-secret")

    class FailingKisClient:
        def get_current_price(self, symbol: str):
            raise RuntimeError("KIS failed")

    monkeypatch.setattr(kis, "KisClient", FailingKisClient)

    result = kis.fetch_price_data("005930")

    assert result["status"] == "error"
    assert result["message"] == "KIS failed"
