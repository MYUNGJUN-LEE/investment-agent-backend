from __future__ import annotations

from datetime import datetime, timedelta
import json

import httpx
import pytest

from app.brokers import kis_client
from app.brokers.kis_client import (
    KisApiError,
    KisClient,
    KisConfigError,
    is_invalid_account_error,
)


def test_issue_access_token_uses_kis_token_endpoint():
    calls: list[httpx.Request] = []
    expires_at = (datetime.now() + timedelta(hours=23)).strftime("%Y-%m-%d %H:%M:%S")

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.method == "POST"
        assert request.url.path == "/oauth2/tokenP"
        assert request.url.host == "openapivts.koreainvestment.com"
        return httpx.Response(
            200,
            json={
                "access_token": "mock-token",
                "access_token_token_expired": expires_at,
                "token_type": "Bearer",
            },
        )

    client = KisClient(
        app_key="test-app-key",
        app_secret="test-app-secret",
        is_paper=True,
        transport=httpx.MockTransport(handler),
    )

    token = client.issue_access_token()
    cached_token = client.issue_access_token()

    assert token == "mock-token"
    assert cached_token == "mock-token"
    assert len(calls) == 1


def test_issue_access_token_reuses_shared_file_cache_across_clients(tmp_path):
    calls: list[httpx.Request] = []
    cache_path = tmp_path / "kis_token_cache.json"
    expires_at = (datetime.now() + timedelta(hours=23)).strftime("%Y-%m-%d %H:%M:%S")

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "access_token": "shared-token",
                "access_token_token_expired": expires_at,
                "token_type": "Bearer",
            },
        )

    first_client = KisClient(
        app_key="shared-app-key",
        app_secret="shared-app-secret",
        is_paper=True,
        transport=httpx.MockTransport(handler),
        token_cache_path=cache_path,
    )
    second_client = KisClient(
        app_key="shared-app-key",
        app_secret="shared-app-secret",
        is_paper=True,
        transport=httpx.MockTransport(handler),
        token_cache_path=cache_path,
    )

    assert first_client.issue_access_token() == "shared-token"
    assert second_client.issue_access_token() == "shared-token"
    assert [request.url.path for request in calls] == ["/oauth2/tokenP"]


def test_http_error_exposes_kis_error_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            request=request,
            json={
                "error_code": "EGW00133",
                "error_description": "접근토큰 발급 잠시 후 다시 시도하세요(1분당 1회)",
            },
        )

    client = KisClient(
        app_key="test-app-key",
        app_secret="test-app-secret",
        is_paper=True,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(KisApiError) as exc_info:
        client.issue_access_token()

    assert exc_info.value.status_code == 403
    assert exc_info.value.error_code == "EGW00133"
    assert "1분당 1회" in str(exc_info.value)


def test_token_rate_limit_sets_shared_cooldown(tmp_path):
    calls: list[httpx.Request] = []
    cache_path = tmp_path / "kis_token_cache.json"

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            403,
            request=request,
            json={
                "error_code": "EGW00133",
                "error_description": "접근토큰 발급 잠시 후 다시 시도하세요(1분당 1회)",
            },
        )

    first_client = KisClient(
        app_key="cooldown-app-key",
        app_secret="cooldown-app-secret",
        is_paper=True,
        transport=httpx.MockTransport(handler),
        token_cache_path=cache_path,
    )
    second_client = KisClient(
        app_key="cooldown-app-key",
        app_secret="cooldown-app-secret",
        is_paper=True,
        transport=httpx.MockTransport(handler),
        token_cache_path=cache_path,
    )

    with pytest.raises(KisApiError):
        first_client.issue_access_token()
    with pytest.raises(KisApiError) as exc_info:
        second_client.issue_access_token()

    assert exc_info.value.error_code == "EGW00133"
    assert "cooling down" in str(exc_info.value)
    assert [request.url.path for request in calls] == ["/oauth2/tokenP"]


def test_get_current_price_requests_domestic_stock_quote():
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(
                200,
                json={
                    "access_token": "mock-token",
                    "expires_in": 86400,
                },
            )

        assert request.method == "GET"
        assert request.url.path == "/uapi/domestic-stock/v1/quotations/inquire-price"
        assert request.headers["authorization"] == "Bearer mock-token"
        assert request.headers["appkey"] == "test-app-key"
        assert request.headers["appsecret"] == "test-app-secret"
        assert request.headers["tr_id"] == "FHKST01010100"
        assert request.url.params["FID_COND_MRKT_DIV_CODE"] == "J"
        assert request.url.params["FID_INPUT_ISCD"] == "005930"
        return httpx.Response(
            200,
            json={
                "rt_cd": "0",
                "msg_cd": "MCA00000",
                "output": {
                    "stck_prpr": "75000",
                    "prdy_ctrt": "1.25",
                },
            },
        )

    client = KisClient(
        app_key="test-app-key",
        app_secret="test-app-secret",
        is_paper=True,
        transport=httpx.MockTransport(handler),
    )

    result = client.get_current_price("005930")

    assert result["rt_cd"] == "0"
    assert result["output"]["stck_prpr"] == "75000"
    assert paths == [
        "/oauth2/tokenP",
        "/uapi/domestic-stock/v1/quotations/inquire-price",
    ]


def test_send_with_throttle_waits_between_real_kis_requests(monkeypatch):
    sleeps: list[float] = []
    moments = iter([10.25, 12.0])
    sent: list[str] = []
    client = KisClient(
        app_key="test-app-key",
        app_secret="test-app-secret",
        is_paper=True,
    )

    monkeypatch.setattr(kis_client, "_LAST_REQUEST_AT", 10.0)
    monkeypatch.setattr(kis_client.settings, "kis_request_min_interval_seconds", 1.5)
    monkeypatch.setattr(kis_client.time, "monotonic", lambda: next(moments))
    monkeypatch.setattr(kis_client.time, "sleep", lambda seconds: sleeps.append(seconds))

    response = client._send_with_throttle(
        lambda: sent.append("request") or httpx.Response(200, json={"rt_cd": "0"})
    )

    assert response.status_code == 200
    assert sent == ["request"]
    assert sleeps == [1.25]


def test_get_daily_prices_requests_domestic_stock_chart():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(
                200,
                json={
                    "access_token": "mock-token",
                    "expires_in": 86400,
                },
            )

        assert request.method == "GET"
        assert request.url.path == "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        assert request.headers["tr_id"] == "FHKST03010100"
        assert request.url.params["FID_INPUT_ISCD"] == "005930"
        assert request.url.params["FID_INPUT_DATE_1"] == "20260501"
        assert request.url.params["FID_INPUT_DATE_2"] == "20260520"
        assert request.url.params["FID_PERIOD_DIV_CODE"] == "D"
        return httpx.Response(
            200,
            json={
                "rt_cd": "0",
                "output2": [
                    {
                        "stck_bsop_date": "20260520",
                        "stck_clpr": "75000",
                    }
                ],
            },
        )

    client = KisClient(
        app_key="test-app-key",
        app_secret="test-app-secret",
        is_paper=True,
        transport=httpx.MockTransport(handler),
    )

    result = client.get_daily_prices("005930", "20260501", "20260520")

    assert result["rt_cd"] == "0"
    assert result["output2"][0]["stck_clpr"] == "75000"


def test_intraday_market_data_methods_use_official_paths_and_tr_ids():
    expected_calls = [
        (
            "/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice",
            "FHKST03010200",
        ),
        (
            "/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn",
            "FHKST01010200",
        ),
        (
            "/uapi/domestic-stock/v1/quotations/inquire-ccnl",
            "FHKST01010300",
        ),
        (
            "/uapi/domestic-stock/v1/quotations/inquire-investor",
            "FHKST01010900",
        ),
        (
            "/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily",
            "FHPTJ04160001",
        ),
    ]
    seen_calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(
                200,
                json={
                    "access_token": "mock-token",
                    "expires_in": 86400,
                },
            )

        path = request.url.path
        tr_id = request.headers["tr_id"]
        seen_calls.append((path, tr_id))
        assert request.method == "GET"
        assert request.headers["authorization"] == "Bearer mock-token"
        assert request.url.params["FID_COND_MRKT_DIV_CODE"] == "J"
        assert request.url.params["FID_INPUT_ISCD"] == "005930"

        if path.endswith("inquire-time-itemchartprice"):
            assert request.url.params["FID_INPUT_HOUR_1"] == "153000"
            assert request.url.params["FID_PW_DATA_INCU_YN"] == "Y"
        if path.endswith("investor-trade-by-stock-daily"):
            assert request.url.params["FID_INPUT_DATE_1"] == "20260501"

        return httpx.Response(200, json={"rt_cd": "0", "output": []})

    client = KisClient(
        app_key="test-app-key",
        app_secret="test-app-secret",
        is_paper=True,
        transport=httpx.MockTransport(handler),
    )

    client.get_minute_prices("005930", input_hour="153000", include_prev=True)
    client.get_orderbook("005930")
    client.get_executions("005930")
    client.get_investor_flow("005930")
    client.get_investor_daily("005930", start_date="20260501")

    assert seen_calls == expected_calls


def test_get_balance_requests_paper_balance_endpoint():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(
                200,
                json={
                    "access_token": "mock-token",
                    "expires_in": 86400,
                },
            )

        assert request.method == "GET"
        assert request.url.path == "/uapi/domestic-stock/v1/trading/inquire-balance"
        assert request.headers["authorization"] == "Bearer mock-token"
        assert request.headers["tr_id"] == "VTTC8434R"
        assert request.url.params["CANO"] == "12345678"
        assert request.url.params["ACNT_PRDT_CD"] == "01"
        assert request.url.params["INQR_DVSN"] == "01"
        return httpx.Response(
            200,
            json={
                "rt_cd": "0",
                "output1": [
                    {
                        "pdno": "005930",
                        "prdt_name": "삼성전자",
                    }
                ],
                "output2": [
                    {
                        "dnca_tot_amt": "1000000",
                    }
                ],
            },
        )

    client = KisClient(
        app_key="test-app-key",
        app_secret="test-app-secret",
        account_no="12345678",
        account_product_code="01",
        is_paper=True,
        transport=httpx.MockTransport(handler),
    )

    result = client.get_balance()

    assert result["rt_cd"] == "0"
    assert result["output1"][0]["pdno"] == "005930"


def test_account_credentials_are_normalized_from_combined_account_value():
    client = KisClient(
        app_key="test-app-key",
        app_secret="test-app-secret",
        account_no=" 50189471-01 ",
        is_paper=True,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
    )

    assert client.account_no == "50189471"
    assert client.account_product_code == "01"


def test_token_cache_key_is_isolated_by_secret_and_account():
    first = KisClient(
        app_key="same-app-key",
        app_secret="first-secret",
        account_no="12345678",
        account_product_code="01",
        is_paper=True,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
    )
    second = KisClient(
        app_key="same-app-key",
        app_secret="second-secret",
        account_no="87654321",
        account_product_code="01",
        is_paper=True,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
    )

    assert first._token_cache_key() != second._token_cache_key()


def test_get_balance_rejects_invalid_account_format_before_http_call():
    calls: list[httpx.Request] = []

    client = KisClient(
        app_key="test-app-key",
        app_secret="test-app-secret",
        account_no="501",
        account_product_code="01",
        is_paper=True,
        transport=httpx.MockTransport(
            lambda request: calls.append(request) or httpx.Response(200, json={})
        ),
    )

    with pytest.raises(KisConfigError, match="exactly 8 digits"):
        client.get_balance()

    assert calls == []


def test_invalid_account_error_detector_matches_kis_check_account_error():
    exc = KisApiError(
        "KIS API error OPSQ2000: ERROR : INPUT INVALID_CHECK_ACNO",
        error_code="OPSQ2000",
        error_description="ERROR : INPUT INVALID_CHECK_ACNO",
    )

    assert is_invalid_account_error(exc) is True


def test_get_balance_requires_account_info():
    client = KisClient(
        app_key="test-app-key",
        app_secret="test-app-secret",
        is_paper=True,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
    )
    client.account_no = None
    client.account_product_code = None

    with pytest.raises(KisConfigError):
        client.get_balance()


def test_get_balance_merges_paginated_balance_pages():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(
                200,
                json={
                    "access_token": "mock-token",
                    "expires_in": 86400,
                },
            )

        if len([r for r in requests if r.url.path.endswith("inquire-balance")]) == 1:
            assert request.headers["tr_cont"] == ""
            assert request.url.params["CTX_AREA_FK100"] == ""
            return httpx.Response(
                200,
                headers={"tr_cont": "M"},
                json={
                    "rt_cd": "0",
                    "output1": [{"pdno": "005930"}],
                    "output2": [
                        {
                            "ctx_area_fk100": "next-fk",
                            "ctx_area_nk100": "next-nk",
                        }
                    ],
                },
            )

        assert request.headers["tr_cont"] == "N"
        assert request.url.params["CTX_AREA_FK100"] == "next-fk"
        assert request.url.params["CTX_AREA_NK100"] == "next-nk"
        return httpx.Response(
            200,
            headers={"tr_cont": "D"},
            json={
                "rt_cd": "0",
                "output1": [{"pdno": "000660"}],
                "output2": [{}],
            },
        )

    client = KisClient(
        app_key="test-app-key",
        app_secret="test-app-secret",
        account_no="12345678",
        account_product_code="01",
        is_paper=True,
        transport=httpx.MockTransport(handler),
    )

    result = client.get_balance()

    assert result["__complete_snapshot"] is True
    assert result["__page_count"] == 2
    assert [row["pdno"] for row in result["output1"]] == ["005930", "000660"]


def test_get_daily_order_executions_requests_domestic_execution_history():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(
                200,
                json={
                    "access_token": "mock-token",
                    "expires_in": 86400,
                },
            )

        assert request.method == "GET"
        assert request.url.path == "/uapi/domestic-stock/v1/trading/inquire-daily-ccld"
        assert request.headers["authorization"] == "Bearer mock-token"
        assert request.headers["tr_id"] == "VTTC8001R"
        assert request.url.params["CANO"] == "12345678"
        assert request.url.params["ACNT_PRDT_CD"] == "01"
        assert request.url.params["INQR_STRT_DT"] == "20260520"
        assert request.url.params["INQR_END_DT"] == "20260520"
        return httpx.Response(
            200,
            json={
                "rt_cd": "0",
                "output1": [
                    {
                        "odno": "12345",
                        "pdno": "005930",
                    }
                ],
            },
        )

    client = KisClient(
        app_key="test-app-key",
        app_secret="test-app-secret",
        account_no="12345678",
        account_product_code="01",
        is_paper=True,
        transport=httpx.MockTransport(handler),
    )

    result = client.get_daily_order_executions("20260520", "20260520")

    assert result["rt_cd"] == "0"
    assert result["output1"][0]["odno"] == "12345"


def test_place_domestic_limit_order_posts_real_cash_order_with_hashkey():
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(
                200,
                json={
                    "access_token": "mock-token",
                    "expires_in": 86400,
                },
            )
        if request.url.path == "/uapi/hashkey":
            return httpx.Response(200, json={"HASH": "mock-hash"})

        assert request.method == "POST"
        assert request.url.host == "openapi.koreainvestment.com"
        assert request.url.path == "/uapi/domestic-stock/v1/trading/order-cash"
        assert request.headers["authorization"] == "Bearer mock-token"
        assert request.headers["tr_id"] == "TTTC0802U"
        assert request.headers["hashkey"] == "mock-hash"
        body = json.loads(request.content)
        assert body["CANO"] == "12345678"
        assert body["ACNT_PRDT_CD"] == "01"
        assert body["PDNO"] == "005930"
        assert body["ORD_DVSN"] == "00"
        assert body["ORD_QTY"] == "3"
        assert body["ORD_UNPR"] == "75000"
        return httpx.Response(
            200,
            json={
                "rt_cd": "0",
                "msg_cd": "APBK0013",
                "msg1": "주문 전송 완료",
                "output": {"ODNO": "12345"},
            },
        )

    client = KisClient(
        app_key="test-app-key",
        app_secret="test-app-secret",
        account_no="12345678",
        account_product_code="01",
        is_paper=False,
        transport=httpx.MockTransport(handler),
    )

    result = client.place_domestic_limit_order(
        symbol="005930",
        side="buy",
        price=75000,
        quantity=3,
    )

    assert result["rt_cd"] == "0"
    assert paths == [
        "/oauth2/tokenP",
        "/uapi/hashkey",
        "/uapi/domestic-stock/v1/trading/order-cash",
    ]


def test_place_domestic_limit_order_posts_paper_cash_order_with_virtual_tr_id():
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(
                200,
                json={
                    "access_token": "mock-token",
                    "expires_in": 86400,
                },
            )
        if request.url.path == "/uapi/hashkey":
            return httpx.Response(200, json={"HASH": "mock-hash"})

        assert request.method == "POST"
        assert request.url.host == "openapivts.koreainvestment.com"
        assert request.url.path == "/uapi/domestic-stock/v1/trading/order-cash"
        assert request.headers["authorization"] == "Bearer mock-token"
        assert request.headers["tr_id"] == "VTTC0802U"
        assert request.headers["hashkey"] == "mock-hash"
        body = json.loads(request.content)
        assert body["CANO"] == "12345678"
        assert body["ACNT_PRDT_CD"] == "01"
        assert body["PDNO"] == "005930"
        assert body["ORD_QTY"] == "1"
        return httpx.Response(
            200,
            json={
                "rt_cd": "0",
                "msg_cd": "APBK0013",
                "msg1": "paper order accepted",
                "output": {"ODNO": "99999"},
            },
        )

    client = KisClient(
        app_key="test-app-key",
        app_secret="test-app-secret",
        account_no="12345678",
        account_product_code="01",
        is_paper=True,
        transport=httpx.MockTransport(handler),
    )

    result = client.place_domestic_limit_order(
        symbol="005930",
        side="buy",
        price=75000,
        quantity=1,
    )

    assert result["rt_cd"] == "0"
    assert paths == [
        "/oauth2/tokenP",
        "/uapi/hashkey",
        "/uapi/domestic-stock/v1/trading/order-cash",
    ]
