# news_sentiment.py
"""
news_sentiment.py
Funciones para descargar noticias y calcular sentimiento diario.
- fetch_news_agg(query, start, end, api_key=None)
Devuelve DataFrame diario indexado por fecha con columnas: news_count, avg_sent
Usa NewsAPI si NEWSAPI_KEY en env o pasado como parámetro, else fallback heurístico.
"""

from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np

try:
    import requests
except Exception:
    requests = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:
    SentimentIntensityAnalyzer = None

def _date_range_daily(start: str, end: str):
    idx = pd.date_range(start=start, end=end, freq="D")
    return idx


def fetch_news_agg(query: str, start: str, end: str, api_key: str = None) -> pd.DataFrame:
    """
    query: símbolo o término (ej. 'bitcoin' o 'BTC')
    start,end: 'YYYY-MM-DD'
    api_key: NewsAPI key (opcional)
    """
    idx = _date_range_daily(start, end)
    zeros = pd.DataFrame(index=idx, data={"news_count": 0, "avg_sent": 0.0})
    if requests is None:
        return zeros
    key = api_key or os.environ.get("NEWSAPI_KEY") or None
    if not key:
        # no key -> return zeros (cheap fallback)
        return zeros

    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "from": start, "to": end, "language": "en", "pageSize": 100, "apiKey": key}
    try:
        r = requests.get(url, params=params, timeout=30)
        data = r.json()
        articles = data.get("articles", [])
        if not articles:
            return zeros
        analyzer = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None
        rows = []
        for a in articles:
            title = a.get("title") or ""
            desc = a.get("description") or ""
            txt = (title + ". " + desc)[:2000]
            dt = a.get("publishedAt") or a.get("published")
            try:
                dt_date = pd.to_datetime(dt).date()
            except Exception:
                dt_date = None
            if dt_date is None:
                continue
            if analyzer:
                s = analyzer.polarity_scores(txt)["compound"]
            else:
                s = 0.0
                ltxt = txt.lower()
                if any(w in ltxt for w in ["gain", "surge", "rise", "bull"]):
                    s += 0.5
                if any(w in ltxt for w in ["drop", "crash", "dump", "bear", "sell-off"]):
                    s -= 0.5
            rows.append((pd.to_datetime(dt_date), s))
        if not rows:
            return zeros
        ndf = pd.DataFrame(rows, columns=["date", "sent"])
        agg = ndf.groupby("date").agg(news_count=("sent", "count"), avg_sent=("sent", "mean"))
        full_idx = pd.date_range(start=start, end=end, freq="D")
        agg = agg.reindex(full_idx, fill_value=0)
        agg.index = pd.to_datetime(agg.index)
        return agg
    except Exception:
        return zeros
