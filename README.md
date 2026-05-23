# Investment Agent Backend

Custom GPT Actions에 연결할 수 있는 **투자 분석 백엔드 FastAPI 서버**입니다.

이 프로젝트는 다음 파이프라인을 실행합니다.

1. Research Agent
2. Financial Analysis Agent
3. Chart & Flow Agent
4. Devil's Advocate Agent
5. Final Check Agent

> 공개 데이터 기반으로 "진입 후보 조건 / 진입 금지 조건 / 손절 후보 / 익절 후보"를 생성하는 분석 서버입니다.

---

## 1. VSCode에서 실행하는 방법

### 1) 폴더 열기

VSCode에서 이 폴더를 엽니다.

```bash
investment_agent_backend
```

### 2) 가상환경 만들기

Windows PowerShell:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) 패키지 설치

```bash
pip install -r requirements.txt
```

### 4) 환경변수 파일 만들기

`.env.example` 파일을 복사해서 `.env` 파일을 만듭니다.

Windows PowerShell:

```bash
copy .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

처음에는 API 키가 없어도 서버는 실행됩니다.  
OpenDART 키를 발급받은 뒤 `.env`에 넣으면 공시 조회가 작동합니다.

```env
OPENDART_API_KEY=여기에_발급받은_키
```

### 5) 서버 실행

```bash
uvicorn app.main:app --reload
```

정상 실행되면 아래 주소를 브라우저에서 확인합니다.

```text
http://127.0.0.1:8000/health
```

API 문서는 여기에서 볼 수 있습니다.

```text
http://127.0.0.1:8000/docs
```

---

## 2. 테스트 호출 예시

Windows PowerShell:

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/run-full-pipeline" `
  -ContentType "application/json" `
  -Body '{"symbol":"000660","name":"SK하이닉스","market":"KR","strategy_type":"daytrade","lookback_hours":24,"risk_level":"medium"}'
```

macOS/Linux:

```bash
curl -X POST "http://127.0.0.1:8000/run-full-pipeline" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"000660","name":"SK하이닉스","market":"KR","strategy_type":"daytrade","lookback_hours":24,"risk_level":"medium"}'
```

---

## 3. Custom GPT Actions 연결 순서

1. 서버를 Render/Railway/Fly.io 같은 곳에 배포합니다.
2. 배포된 주소 예시:

```text
https://your-invest-api.onrender.com
```

3. `action_schema.yaml` 파일 안의 서버 주소를 배포 주소로 바꿉니다.

```yaml
servers:
  - url: https://your-invest-api.onrender.com
```

4. ChatGPT → Explore GPTs → Create → Configure → Actions
5. `action_schema.yaml` 내용을 붙여넣습니다.
6. Test 버튼으로 `/health` 또는 `/run-full-pipeline`을 테스트합니다.

---

## 4. 파일 구조

```text
investment_agent_backend/
  app/
    main.py
    config.py
    models.py
    scoring.py
    agents.py
    services/
      pipeline.py
    data_sources/
      opendart.py
      news.py
      kis.py
      financials.py
  data/
    corp_map.csv
  action_schema.yaml
  requirements.txt
  .env.example
  README.md
```

---

## 5. 현재 구현 상태

현재 버전은 다음을 포함합니다.

- FastAPI 서버
- `/health`
- `/run-full-pipeline`
- OpenDART 공시 조회 기본 연결
- NAVER 뉴스 검색 연동
- KIS 시세 데이터 연동
- OpenDART 재무 데이터 조회
- 5개 Agent rule-based 분석
- Custom GPT Action schema
- 비용 반영 주문 승인: 수수료, 매도세, 스프레드, 슬리피지, 환전비용, 레버리지 이자, 대차비용, 체결확률
- 성과지표 저장: CAGR, MDD, Sharpe, Sortino, Calmar, Profit Factor, Win Rate, 손익비, Expectancy, Turnover, Exposure, Beta, Tail Loss
- 과최적화 방지: train/validation/test 시계열 분리, walk-forward window, in/out-sample 붕괴 탐지
- 주문 안전장치: 중복 주문 방지, 부분 체결 기록, 비정상 가격 차단, 잔고 초과 차단, 주문 수 제한, 긴급정지 플래그
- 포지션 사이징: `account_equity * risk_per_trade / abs(entry_price - stop_price)`
- 과거 데이터 기반 피처: 수익률, 돌파, 갭, 거래량 비율, ATR, realized volatility, Bollinger width, MA slope, MACD, ADX, 5/20/60일 모멘텀

추후 확장 순서:

1. OpenDART 키 연결
2. 뉴스 API 연결
3. KIS 시세 API 연결
4. OpenAI API를 이용한 Agent별 LLM 분석 추가
5. 모의매매 로그 DB 추가
6. 사용자 승인형 주문 미리보기 추가
7. 가장 마지막에 실주문 API 연결

---

## 6. 안전 원칙

이 서버는 다음을 하지 않습니다.

- 확정적 매수/매도 지시
- 수익 보장
- 미공개정보 사용
- 허수주문/시세조종성 주문
- 자동 실주문

이 서버는 다음만 합니다.

- 공개 정보 수집
- 이벤트 분류
- 리스크 체크
- 진입 후보 조건 제시
- 진입 금지 조건 제시
- 손절/익절 후보 제시

---

## Persistent Auto-Trading Worker

The `/auto-trading/start` endpoint saves an auto-trading session to SQLite
instead of running only inside the FastAPI process. Start a separate worker
process to execute due sessions:

```bash
python -m app.trading.auto_trading_worker
```

For a one-shot worker pass:

```bash
python -m app.trading.auto_trading_worker --once
```

Sessions are stored in `AUTO_TRADING_DB_PATH` and can be checked or stopped via:

- `GET /auto-trading/sessions`
- `GET /auto-trading/status/{session_id}`
- `GET /auto-trading/events/{session_id}`
- `POST /auto-trading/stop/{session_id}`
- `POST /auto-trading/restart/{session_id}`

Default execution mode is `paper`. Live auto-trading still requires
`ENABLE_LIVE_TRADING=true`, `KIS_IS_PAPER=false`, a matching live confirm token,
and all existing risk checks.

The default auto-trading cycle is 60 seconds. If `/auto-trading/start` is called
with an empty `symbols` list, the universe scanner runs first, stores KIS price
snapshots in SQLite, ranks candidates, checks NAVER/OpenDART only for candidates,
and passes the final 5-10 symbols into the normal analyzer. The scanner calls
one symbol at a time and waits `UNIVERSE_SCANNER_SYMBOL_INTERVAL_SECONDS`
between symbols. KIS HTTP requests are also serialized by
`KIS_REQUEST_MIN_INTERVAL_SECONDS`, so each symbol's price, daily, minute,
orderbook, execution, and investor-flow requests are not fired at once.
Auto-trading, market monitoring, and broker sync now run as separate workers:

```bash
python -m app.trading.auto_trading_worker
python -m app.trading.market_monitor_worker
python -m app.trading.broker_sync_worker
```

Broker state is stored in `BROKER_SYNC_DB_PATH`; it can also be synchronized
manually with `POST /broker/kis/sync`.

Universe scanner endpoints:

- `POST /universe/scan`
- `GET /universe/latest`
- `POST /gpt/auto-trading/control`

Minimal "start only" body:

```json
{
  "execution_mode": "paper",
  "interval_seconds": 60,
  "auto_discover_symbols": true,
  "symbols": []
}
```

Custom GPT can use the single control endpoint instead of managing session ids:

```json
{ "command": "start", "execution_mode": "paper" }
```

```json
{ "command": "stop" }
```

```json
{ "command": "status" }
```

When the Korean market is closed or it is a weekend, the universe scanner does
not turn missing current prices into immediate buy candidates. If recent daily
candles include a close price, it stores the latest close and marks the symbol as
`watch`; otherwise it records `exclude` with a no-price reason.

The built-in monitoring dashboard is available at:

```text
http://127.0.0.1:8000/dashboard/auto-trading
```

Enter `X-API-Key` in the dashboard when `BACKEND_API_KEY` is configured. The
dashboard can list sessions, inspect recent events, stop active sessions, and
restart stopped or errored sessions.

## Render Single-Service Mode

For the lowest-cost hosted setup, run the FastAPI server and embedded workers in
one Render Web Service. Set this in Render environment variables:

```env
EMBEDDED_WORKERS_ENABLED=true
EMBEDDED_WORKER_BROKER_SYNC_ENABLED=true
```

Use this Render start command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

When enabled, the API process starts these worker threads:

- `app.workers.trading_worker`: claims and processes auto-trading sessions.
- `app.workers.market_worker`: 1-minute KIS price/volume/change watch.
- `app.workers.dart_worker`: 5-minute OpenDART disclosure watch.
- `app.workers.news_worker`: 10-minute NAVER news watch.
- `app.workers.broker_worker`: KIS balance/execution sync and order-state reconcile.

Custom GPT can still use only:

```json
{ "command": "start", "execution_mode": "paper" }
```

The `start` command also verifies that embedded workers are running. Later, each
`app.workers.*` module can be moved to a separate Render Background Worker
without changing the trading logic.

## KIS Paper E2E Check

The KIS paper command is safe by default: without `--place-order`, it only
checks quote, balance, execution-history, and SQLite sync connectivity.

```bash
python -m app.trading.kis_paper_e2e --symbol 005930
```

KIS limits access-token issuance. The backend reuses tokens through
`KIS_TOKEN_CACHE_PATH`; if KIS returns `EGW00133`, wait for the configured
`KIS_TOKEN_ISSUE_COOLDOWN_SECONDS` before retrying. Use
`GET /broker/kis/config-status` to inspect non-secret KIS runtime settings and
token-cache status.

If KIS returns `OPSQ2000: INPUT INVALID_CHECK_ACNO`, the account number does not
match the configured KIS app key/environment. Use `KIS_ACCOUNT_NO` for the
8-digit CANO only, for example `50189471`, and
`KIS_ACCOUNT_PRODUCT_CODE=01`. Do not include a hyphen or the product code in
`KIS_ACCOUNT_NO`. When `KIS_IS_PAPER=true`, both the app key and account must be
from the KIS paper-trading environment.

KIS also rejects excessive per-second API traffic. In the default safe
configuration, `AUTO_TRADING_SYMBOL_WORKERS=1`,
`KIS_REQUEST_MIN_INTERVAL_SECONDS=1.5`, and
`UNIVERSE_SCANNER_SYMBOL_INTERVAL_SECONDS=60`. A 20-symbol universe scan will
therefore take at least about 20 minutes plus network time, but it keeps the
single Render process below roughly one KIS request every 1.5 seconds.
Auto-trading will not start order analysis unless the universe scanner has
scanned at least `UNIVERSE_SCANNER_MIN_SCANNED_SYMBOLS_FOR_TRADING` symbols,
which defaults to 15.

Actual KIS paper order placement remains opt-in. Use only a KIS paper account
and a tiny quantity:

```bash
python -m app.trading.kis_paper_e2e --place-order --symbol 005930 --side buy --quantity 1 --timeout-seconds 60
```

The same flow is covered by an opt-in pytest test. It is skipped by default:

```bash
RUN_KIS_PAPER_E2E=1 pytest tests/test_kis_paper_e2e.py -q
```

Order-state safety is stored in `ORDER_STATE_DB_PATH`. It blocks duplicate live
orders with the same idempotency key or an already pending symbol transition,
tracks previous/target/current quantities, records broker order numbers, and
keeps partially filled orders in `PARTIAL` until broker sync confirms completion.

## Periodic Market Monitor

The separated market monitor worker runs periodic data checks:

- Every 60 seconds: checks watchlist KIS price, volume, change rate, surge/drop,
  volume spikes, and held-position stop/take-profit conditions.
- Every 300 seconds: checks recent OpenDART disclosures for watchlist symbols.
- Every 600 seconds: searches NAVER news for watchlist names and market keywords,
  then stores only new deduped news items.

The monitor uses `MARKET_MONITOR_DB_PATH` and currently stores data in local
SQLite. Supabase is not used by this codebase yet.

Useful endpoints:

- `GET /monitor/status`
- `POST /monitor/run-due`
- `POST /monitor/run/kis_market_watch`
- `POST /monitor/run/opendart_disclosures`
- `POST /monitor/run/naver_news`

Configure the monitor in `.env`:

```env
MARKET_MONITOR_ENABLED=true
MARKET_MONITOR_DB_PATH=data/market_monitor.sqlite3
MONITOR_WATCHLIST_SYMBOLS=005930,000660
MONITOR_MARKET_KEYWORDS=코스피,코스닥,환율,금리,반도체,AI
MONITOR_PRICE_INTERVAL_SECONDS=60
MONITOR_DISCLOSURE_INTERVAL_SECONDS=300
MONITOR_NEWS_INTERVAL_SECONDS=600
MONITOR_SURGE_CHANGE_PCT=5
MONITOR_DROP_CHANGE_PCT=-5
MONITOR_VOLUME_SPIKE_RATIO=3
MONITOR_DEFAULT_STOP_LOSS_PCT=3
MONITOR_DEFAULT_TAKE_PROFIT_PCT=5
BROKER_SYNC_INTERVAL_SECONDS=60
ALERT_DB_PATH=data/alerts.sqlite3
ALERT_WEBHOOK_URL=
ALERT_MIN_SEVERITY=high
ALERT_MIN_IMPACT_STRENGTH=70
```

If `ALERT_WEBHOOK_URL` is empty, alerts are still stored in `ALERT_DB_PATH` and
are not sent externally. Setting the webhook enables real-time alert delivery.

## Cloudflare Tunnel For Custom GPT

The lowest-cost setup keeps this backend on your PC and exposes it through a
Cloudflare Tunnel HTTPS hostname.

```text
Custom GPT -> https://api.your-domain.com -> Cloudflare Tunnel -> 127.0.0.1:8010
```

Create the tunnel with the guide in `deploy/cloudflare/README.md`, then run:

```powershell
.\scripts\start_backend.ps1 -Port 8010
.\scripts\start_cloudflare_tunnel.ps1 -ConfigPath "$env:USERPROFILE\.cloudflared\investment-agent-backend.yml"
```

Generate a Custom GPT schema that points to the tunnel URL:

```powershell
.\scripts\set_action_schema_server.ps1 -PublicUrl "https://api.your-domain.com"
```

Paste `action_schema.local-tunnel.yaml` into the Custom GPT Actions schema
window. Set Custom GPT authentication to send `X-API-Key` with the same value as
`BACKEND_API_KEY` in `.env`.
