import os
import re
from html import unescape

import httpx


def clean_html(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"<.*?>", "", text)
    return unescape(text).strip()


def search_naver_news(query: str, display: int = 10, sort: str = "date") -> dict:
    client_id = os.getenv("NAVER_CLIENT_ID")
    client_secret = os.getenv("NAVER_CLIENT_SECRET")

    if not client_id or not client_secret:
        return {
            "connected": False,
            "source": "Naver Search API",
            "error": "NAVER_CLIENT_ID or NAVER_CLIENT_SECRET is missing",
            "items": [],
        }

    url = "https://openapi.naver.com/v1/search/news.json"

    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }

    params = {
        "query": query,
        "display": display,
        "start": 1,
        "sort": sort,
    }

    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(url, headers=headers, params=params)

        if response.status_code != 200:
            return {
                "connected": False,
                "source": "Naver Search API",
                "status_code": response.status_code,
                "error": response.text,
                "items": [],
            }

        data = response.json()

        items = []
        for item in data.get("items", []):
            items.append(
                {
                    "title": clean_html(item.get("title")),
                    "description": clean_html(item.get("description")),
                    "originallink": item.get("originallink"),
                    "link": item.get("link"),
                    "pubDate": item.get("pubDate"),
                }
            )

        return {
            "connected": True,
            "source": "Naver Search API",
            "query": query,
            "total": data.get("total"),
            "items": items,
        }

    except Exception as e:
        return {
            "connected": False,
            "source": "Naver Search API",
            "error": str(e),
            "items": [],
        }