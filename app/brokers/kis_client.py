from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import httpx

from app.config import settings


KIS_PROD_BASE_URL = "https://openapi.koreainvestment.com:9443"
KIS_PAPER_BASE_URL = "https://openapivts.koreainvestment.com:29443"


class KisConfigError(ValueError):
    """Raised when required KIS configuration is missing."""


class KisApiError(RuntimeError):
    """Raised when KIS returns an HTTP or business-level API error."""


class KisClient:
    """
    Minimal KIS Open API client.

    Order support is intentionally limited to domestic stock limit orders and
    must be called from an execution layer with separate safety checks.
    """

    def __init__(
        self,
        app_key: str | None = None,
        app_secret: str | None = None,
        account_no: str | None = None,
        account_product_code: str | None = None,
        is_paper: bool | None = None,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.app_key = app_key or settings.kis_app_key
        self.app_secret = app_secret or settings.kis_app_secret
        self.account_no = account_no or settings.kis_account_no
        self.account_product_code = (
            account_product_code or settings.kis_account_product_code
        )
        self.is_paper = settings.kis_is_paper if is_paper is None else is_paper
        self.timeout = timeout
        self.transport = transport

        self._access_token: str | None = None
        self._access_token_expires_at: datetime | None = None

    @property
    def base_url(self) -> str:
        return KIS_PAPER_BASE_URL if self.is_paper else KIS_PROD_BASE_URL

    def issue_access_token(self, force_refresh: bool = False) -> str:
        """
        Issue or reuse an access token.

        KIS token issuance uses /oauth2/tokenP with client_credentials.
        """
        self._ensure_app_credentials()

        if not force_refresh and self._has_valid_token():
            return self._access_token or ""

        payload = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }

        data = self._post(
            "/oauth2/tokenP",
            json=payload,
            headers=self._base_headers(),
            check_rt_cd=False,
        )
        token = data.get("access_token")
        if not token:
            raise KisApiError("KIS token response did not include access_token")

        self._access_token = str(token)
        self._access_token_expires_at = self._parse_token_expiry(data)
        return self._access_token

    def get_current_price(
        self,
        symbol: str,
        market_div_code: str = "J",
    ) -> dict[str, Any]:
        """
        Fetch domestic stock current price.

        market_div_code examples: J=KRX, NX=NXT, UN=integrated market.
        """
        token = self.issue_access_token()
        data = self._get(
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            params={
                "FID_COND_MRKT_DIV_CODE": market_div_code,
                "FID_INPUT_ISCD": symbol,
            },
            headers=self._auth_headers(token=token, tr_id="FHKST01010100"),
        )
        return data

    def get_daily_prices(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        period_div_code: str = "D",
        market_div_code: str = "J",
    ) -> dict[str, Any]:
        """Fetch domestic stock daily chart prices."""
        token = self.issue_access_token()
        data = self._get(
            "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
            params={
                "FID_COND_MRKT_DIV_CODE": market_div_code,
                "FID_INPUT_ISCD": symbol,
                "FID_INPUT_DATE_1": start_date,
                "FID_INPUT_DATE_2": end_date,
                "FID_PERIOD_DIV_CODE": period_div_code,
                "FID_ORG_ADJ_PRC": "0",
            },
            headers=self._auth_headers(token=token, tr_id="FHKST03010100"),
        )
        return data

    def get_minute_prices(
        self,
        symbol: str,
        input_hour: str = "",
        market_div_code: str = "J",
        include_prev: bool = False,
    ) -> dict[str, Any]:
        """
        Fetch domestic stock intraday minute candles.

        KIS returns same-day minute candles. If input_hour is omitted, the
        current local HHMMSS is used.
        """
        input_hour = input_hour or datetime.now().strftime("%H%M%S")
        token = self.issue_access_token()
        data = self._get(
            "/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice",
            params={
                "FID_COND_MRKT_DIV_CODE": market_div_code,
                "FID_INPUT_ISCD": symbol,
                "FID_INPUT_HOUR_1": input_hour,
                "FID_PW_DATA_INCU_YN": "Y" if include_prev else "N",
                "FID_ETC_CLS_CODE": "",
            },
            headers=self._auth_headers(token=token, tr_id="FHKST03010200"),
        )
        return data

    def get_orderbook(
        self,
        symbol: str,
        market_div_code: str = "J",
    ) -> dict[str, Any]:
        """Fetch domestic stock orderbook and expected execution data."""
        token = self.issue_access_token()
        data = self._get(
            "/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn",
            params={
                "FID_COND_MRKT_DIV_CODE": market_div_code,
                "FID_INPUT_ISCD": symbol,
            },
            headers=self._auth_headers(token=token, tr_id="FHKST01010200"),
        )
        return data

    def get_executions(
        self,
        symbol: str,
        market_div_code: str = "J",
    ) -> dict[str, Any]:
        """Fetch domestic stock recent executions."""
        token = self.issue_access_token()
        data = self._get(
            "/uapi/domestic-stock/v1/quotations/inquire-ccnl",
            params={
                "FID_COND_MRKT_DIV_CODE": market_div_code,
                "FID_INPUT_ISCD": symbol,
            },
            headers=self._auth_headers(token=token, tr_id="FHKST01010300"),
        )
        return data

    def get_investor_flow(
        self,
        symbol: str,
        market_div_code: str = "J",
    ) -> dict[str, Any]:
        """Fetch current investor flow by investor type for a stock."""
        token = self.issue_access_token()
        data = self._get(
            "/uapi/domestic-stock/v1/quotations/inquire-investor",
            params={
                "FID_COND_MRKT_DIV_CODE": market_div_code,
                "FID_INPUT_ISCD": symbol,
            },
            headers=self._auth_headers(token=token, tr_id="FHKST01010900"),
        )
        return data

    def get_investor_daily(
        self,
        symbol: str,
        start_date: str,
        market_div_code: str = "J",
    ) -> dict[str, Any]:
        """Fetch daily investor trading trend by stock."""
        token = self.issue_access_token()
        data = self._get(
            "/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily",
            params={
                "FID_COND_MRKT_DIV_CODE": market_div_code,
                "FID_INPUT_ISCD": symbol,
                "FID_INPUT_DATE_1": start_date,
                "FID_ORG_ADJ_PRC": "",
                "FID_ETC_CLS_CODE": "",
            },
            headers=self._auth_headers(token=token, tr_id="FHPTJ04160001"),
        )
        return data

    def place_domestic_limit_order(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: int,
        account_no: str | None = None,
        account_product_code: str | None = None,
    ) -> dict[str, Any]:
        """Place a domestic cash limit order. Market orders are not supported."""
        if side not in ("buy", "sell"):
            raise KisConfigError("side must be buy or sell")
        if price <= 0 or quantity <= 0:
            raise KisConfigError("price and quantity must be positive")

        cano = account_no or self.account_no
        product_code = account_product_code or self.account_product_code
        if not cano or not product_code:
            raise KisConfigError(
                "KIS account_no and account_product_code are required for live orders"
            )

        token = self.issue_access_token()
        body = {
            "CANO": cano,
            "ACNT_PRDT_CD": product_code,
            "PDNO": symbol,
            "ORD_DVSN": "00",
            "ORD_QTY": str(quantity),
            "ORD_UNPR": str(int(price)),
        }
        if self.is_paper:
            tr_id = "VTTC0802U" if side == "buy" else "VTTC0801U"
        else:
            tr_id = "TTTC0802U" if side == "buy" else "TTTC0801U"
        headers = self._auth_headers(token=token, tr_id=tr_id)
        headers["hashkey"] = self.create_hashkey(body)
        return self._post(
            "/uapi/domestic-stock/v1/trading/order-cash",
            json=body,
            headers=headers,
            check_rt_cd=True,
        )

    def create_hashkey(self, body: dict[str, Any]) -> str:
        headers = self._base_headers()
        headers.update(
            {
                "appkey": self.app_key or "",
                "appsecret": self.app_secret or "",
            }
        )
        data = self._post(
            "/uapi/hashkey",
            json=body,
            headers=headers,
            check_rt_cd=False,
        )
        hashkey = data.get("HASH") or data.get("hashkey")
        if not hashkey:
            raise KisApiError("KIS hashkey response did not include HASH")
        return str(hashkey)

    def get_balance(
        self,
        account_no: str | None = None,
        account_product_code: str | None = None,
    ) -> dict[str, Any]:
        """Fetch domestic stock account balance as a complete snapshot when possible."""
        cano = account_no or self.account_no
        product_code = account_product_code or self.account_product_code
        if not cano or not product_code:
            raise KisConfigError(
                "KIS account_no and account_product_code are required for balance lookup"
            )

        token = self.issue_access_token()
        tr_id = "VTTC8434R" if self.is_paper else "TTTC8434R"
        params = {
            "CANO": cano,
            "ACNT_PRDT_CD": product_code,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "01",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }
        pages: list[dict[str, Any]] = []
        complete_snapshot = True

        for page_index in range(20):
            headers = self._auth_headers(
                token=token,
                tr_id=tr_id,
                tr_cont="N" if page_index else "",
            )
            data, response_headers = self._get_with_headers(
                "/uapi/domestic-stock/v1/trading/inquire-balance",
                params=params,
                headers=headers,
            )
            pages.append(data)

            if not _has_more_pages(data, response_headers):
                return _merge_balance_pages(
                    pages,
                    complete_snapshot=complete_snapshot,
                )

            next_fk, next_nk = _next_context(data)
            if not next_fk and not next_nk:
                complete_snapshot = False
                break
            params["CTX_AREA_FK100"] = next_fk
            params["CTX_AREA_NK100"] = next_nk
        else:
            complete_snapshot = False

        return _merge_balance_pages(pages, complete_snapshot=complete_snapshot)

    def get_daily_order_executions(
        self,
        start_date: str,
        end_date: str,
        symbol: str = "",
        side_code: str = "00",
        execution_filter: str = "00",
        account_no: str | None = None,
        account_product_code: str | None = None,
    ) -> dict[str, Any]:
        """Fetch domestic daily order/execution history."""
        cano = account_no or self.account_no
        product_code = account_product_code or self.account_product_code
        if not cano or not product_code:
            raise KisConfigError(
                "KIS account_no and account_product_code are required for order execution lookup"
            )

        token = self.issue_access_token()
        tr_id = "VTTC8001R" if self.is_paper else "TTTC8001R"
        data = self._get(
            "/uapi/domestic-stock/v1/trading/inquire-daily-ccld",
            params={
                "CANO": cano,
                "ACNT_PRDT_CD": product_code,
                "INQR_STRT_DT": start_date,
                "INQR_END_DT": end_date,
                "SLL_BUY_DVSN_CD": side_code,
                "INQR_DVSN": "00",
                "PDNO": symbol,
                "CCLD_DVSN": execution_filter,
                "ORD_GNO_BRNO": "",
                "ODNO": "",
                "INQR_DVSN_3": "00",
                "INQR_DVSN_1": "",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": "",
            },
            headers=self._auth_headers(token=token, tr_id=tr_id),
        )
        return data

    def _ensure_app_credentials(self) -> None:
        if not self.app_key or not self.app_secret:
            raise KisConfigError("KIS app_key and app_secret are required")

    def _has_valid_token(self) -> bool:
        if not self._access_token or not self._access_token_expires_at:
            return False
        return datetime.now() < self._access_token_expires_at - timedelta(minutes=1)

    def _base_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "charset": "UTF-8",
        }

    def _auth_headers(self, token: str, tr_id: str, tr_cont: str = "") -> dict[str, str]:
        headers = self._base_headers()
        headers.update(
            {
                "authorization": f"Bearer {token}",
                "appkey": self.app_key or "",
                "appsecret": self.app_secret or "",
                "tr_id": tr_id,
                "tr_cont": tr_cont,
                "custtype": "P",
            }
        )
        return headers

    def _get(
        self,
        path: str,
        params: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        with httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            transport=self.transport,
        ) as client:
            response = client.get(path, params=params, headers=headers)
        return self._parse_response(response)

    def _get_with_headers(
        self,
        path: str,
        params: dict[str, Any],
        headers: dict[str, str],
    ) -> tuple[dict[str, Any], httpx.Headers]:
        with httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            transport=self.transport,
        ) as client:
            response = client.get(path, params=params, headers=headers)
        return self._parse_response(response), response.headers

    def _post(
        self,
        path: str,
        json: dict[str, Any],
        headers: dict[str, str],
        check_rt_cd: bool = True,
    ) -> dict[str, Any]:
        with httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            transport=self.transport,
        ) as client:
            response = client.post(path, json=json, headers=headers)
        return self._parse_response(response, check_rt_cd=check_rt_cd)

    def _parse_response(
        self,
        response: httpx.Response,
        check_rt_cd: bool = True,
    ) -> dict[str, Any]:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise KisApiError(f"KIS HTTP error: {exc.response.status_code}") from exc

        data = response.json()
        if check_rt_cd and data.get("rt_cd") not in (None, "0"):
            msg_cd = data.get("msg_cd", "unknown")
            msg1 = data.get("msg1", "Unknown KIS API error")
            raise KisApiError(f"KIS API error {msg_cd}: {msg1}")
        return data

    def _parse_token_expiry(self, data: dict[str, Any]) -> datetime:
        raw_expiry = data.get("access_token_token_expired")
        if isinstance(raw_expiry, str):
            try:
                return datetime.strptime(raw_expiry, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass

        expires_in = data.get("expires_in")
        if isinstance(expires_in, (int, float)):
            return datetime.now() + timedelta(seconds=float(expires_in))

        return datetime.now() + timedelta(hours=23)


def _merge_balance_pages(
    pages: list[dict[str, Any]],
    complete_snapshot: bool,
) -> dict[str, Any]:
    if not pages:
        return {"__complete_snapshot": False}

    merged = dict(pages[0])
    merged["output1"] = [
        row
        for page in pages
        for row in _rows(page.get("output1"))
    ]
    merged["output2"] = [
        row
        for page in pages
        for row in _rows(page.get("output2"))
    ]
    merged["__complete_snapshot"] = complete_snapshot
    merged["__page_count"] = len(pages)
    return merged


def _has_more_pages(data: dict[str, Any], headers: httpx.Headers) -> bool:
    tr_cont = str(headers.get("tr_cont") or headers.get("TR_CONT") or "").strip().upper()
    if tr_cont in {"M", "F"}:
        return True
    if tr_cont in {"D", "E"}:
        return False
    next_fk, next_nk = _next_context(data)
    return bool(next_fk or next_nk)


def _next_context(data: dict[str, Any]) -> tuple[str, str]:
    output2 = _first_dict(data.get("output2"))
    return (
        str(output2.get("ctx_area_fk100") or output2.get("CTX_AREA_FK100") or "").strip(),
        str(output2.get("ctx_area_nk100") or output2.get("CTX_AREA_NK100") or "").strip(),
    )


def _first_dict(value: Any) -> dict[str, Any]:
    rows = _rows(value)
    return rows[0] if rows else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        return [value]
    return []
