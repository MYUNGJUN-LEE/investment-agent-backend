from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_naver_news_endpoint(monkeypatch):
    def fake_search_naver_news(query: str, display: int = 10):
        return {
            "connected": True,
            "source": "Naver Search API",
            "query": query,
            "total": 1,
            "items": [
                {
                    "title": "SK하이닉스 테스트 뉴스",
                    "description": "테스트 설명",
                    "originallink": "https://example.com/original",
                    "link": "https://example.com/news",
                    "pubDate": "Wed, 20 May 2026 09:00:00 +0900",
                }
            ],
            "display": display,
        }

    monkeypatch.setattr("app.main.search_naver_news", fake_search_naver_news)

    response = client.get("/naver/news", params={"query": "SK하이닉스", "display": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["connected"] is True
    assert body["query"] == "SK하이닉스"
    assert body["display"] == 3
    assert body["items"][0]["title"] == "SK하이닉스 테스트 뉴스"
