# Investment Agent Backend

Custom GPT Actions에 연결할 수 있는 **투자 분석 백엔드 FastAPI 서버**입니다.

이 프로젝트는 다음 파이프라인을 실행합니다.

1. Research Agent
2. Financial Analysis Agent
3. Chart & Flow Agent
4. Devil's Advocate Agent
5. Final Check Agent

> 주의: 이 코드는 자동 매수/매도 실행 코드가 아닙니다.  
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
- 뉴스 API stub
- KIS 시세 API stub
- 재무 데이터 stub
- 5개 Agent rule-based 분석
- Custom GPT Action schema

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
