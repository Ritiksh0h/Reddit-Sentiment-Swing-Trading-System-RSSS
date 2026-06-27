"""
Finnhub news fetcher.
Primary news source for RSSS — replaces yfinance news in daily_run_live.py.

API key: FINNHUB_API_KEY from .env
Rate limit: 60 calls/minute free tier → sleep 1.1s between ticker calls
Backfill history: ~12 months (free tier limit)

Live usage:
    from data.finnhub_news_fetcher import fetch_live_news_sentiment
    news_data = fetch_live_news_sentiment(tickers, api_key)

Output format matches fetch_yfinance_news() — drop-in replacement.
"""

import logging
import os
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import requests

logger = logging.getLogger(__name__)

FINNHUB_BASE    = "https://finnhub.io/api/v1"
RATE_LIMIT_SLEEP = 1.1   # free tier: 60 calls/min
RETRY_SLEEP_429  = 60    # sleep on 429 before one retry

SKIP_FINBERT = os.getenv('SKIP_FINBERT', '0') == '1'


# ─────────────────────────────────────────────────────────────────────────────
# Step 1A — Single ticker fetch
# ─────────────────────────────────────────────────────────────────────────────

def fetch_ticker_news(
    ticker: str,
    from_date: str,
    to_date: str,
    api_key: str,
) -> list[dict]:
    """
    Fetch company news from Finnhub for one ticker.

    from_date / to_date: YYYY-MM-DD
    Returns list of article dicts with fields:
        category, datetime (Unix), headline, id, image,
        related, source, summary, url
    Returns empty list on any failure — never raises.
    """
    if not api_key:
        logger.warning("finnhub_no_api_key — skipping")
        return []

    url = (
        f"{FINNHUB_BASE}/company-news"
        f"?symbol={ticker}"
        f"&from={from_date}"
        f"&to={to_date}"
        f"&token={api_key}"
    )

    try:
        resp = requests.get(url, timeout=15)

        if resp.status_code == 429:
            logger.warning(
                f"finnhub_429 ticker={ticker} — sleeping {RETRY_SLEEP_429}s"
            )
            time.sleep(RETRY_SLEEP_429)
            resp = requests.get(url, timeout=15)

        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list):
                return data
            logger.warning(
                f"finnhub_unexpected_shape ticker={ticker} type={type(data)}"
            )
            return []

        logger.warning(
            f"finnhub_error status={resp.status_code} ticker={ticker}"
        )
        return []

    except Exception as e:
        logger.warning(f"finnhub_failed ticker={ticker}: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Step 1B — Finnhub native sentiment (no model loading)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_native_sentiment(ticker: str, api_key: str) -> float | None:
    """
    Fetch Finnhub pre-computed news sentiment via /news-sentiment endpoint.
    Returns score in [-1, +1], or None if endpoint unavailable or no data.

    Prefers buzz.sentiment if present; falls back to bullishPercent - bearishPercent.
    Completely avoids model loading — primary path when SKIP_FINBERT=1.
    """
    if not api_key:
        return None
    url = f"{FINNHUB_BASE}/news-sentiment?symbol={ticker}&token={api_key}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        buzz_sentiment = data.get("buzz", {}).get("sentiment", None)
        if buzz_sentiment is not None:
            return float(buzz_sentiment)
        bull = data.get("sentiment", {}).get("bullishPercent", None)
        bear = data.get("sentiment", {}).get("bearishPercent", None)
        if bull is not None and bear is not None:
            return float(bull) - float(bear)
        return None
    except Exception as e:
        logger.debug(f"finnhub_native_sentiment_failed ticker={ticker}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 1C — FinBERT scoring with VADER fallback
# ─────────────────────────────────────────────────────────────────────────────

def score_articles(articles: list[dict]) -> list[float]:
    """
    Score each article using headline + ". " + summary.
    Primary:   FinBERT (ProsusAI/finbert)
    Fallback:  VADER (vaderSentiment)
    Last resort: 0.0 for each article
    Returns list of floats in [-1.0, +1.0].
    """
    if not articles:
        return []

    texts = []
    for a in articles:
        headline = (a.get("headline") or "").strip()
        summary  = (a.get("summary")  or "").strip()
        combined = (headline + ". " + summary).strip()
        texts.append(combined[:512])

    # ── Try FinBERT (skip in CI) ─────────────────────────────────────────────
    try:
        if SKIP_FINBERT:
            raise RuntimeError('SKIP_FINBERT=1')
        from transformers import pipeline as hf_pipeline
        import torch

        if not hasattr(score_articles, "_model"):
            device = 0 if torch.cuda.is_available() else -1
            score_articles._model = hf_pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                device=device,
                truncation=True,
                max_length=128,
            )

        results = score_articles._model([t[:128] for t in texts])
        scores  = []
        for r in results:
            if r["label"] == "positive":
                scores.append(float(r["score"]))
            elif r["label"] == "negative":
                scores.append(-float(r["score"]))
            else:
                scores.append(0.0)
        return scores

    except Exception as e:
        logger.warning(f"finbert_failed: {e} — trying VADER")

    # ── Try VADER ────────────────────────────────────────────────────────────
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        if not hasattr(score_articles, "_vader"):
            score_articles._vader = SentimentIntensityAnalyzer()

        return [
            float(score_articles._vader.polarity_scores(t)["compound"])
            for t in texts
        ]

    except Exception as e:
        logger.warning(f"vader_failed: {e} — returning neutral scores")

    return [0.0] * len(articles)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1D — Daily aggregation
# ─────────────────────────────────────────────────────────────────────────────

def compute_daily_sentiment(
    articles: list[dict],
    ticker: str,
    date_str: str,
) -> dict:
    """
    Aggregate a list of articles into one daily row.

    Returns dict with: ticker, date, news_count_1d,
        news_sentiment_1d, news_sentiment_max,
        news_sentiment_min, news_source_count
    All floats = 0.0 when no articles.
    """
    if not articles:
        return {
            "ticker":            ticker,
            "date":              date_str,
            "news_count_1d":     0,
            "news_sentiment_1d": 0.0,
            "news_sentiment_max": 0.0,
            "news_sentiment_min": 0.0,
            "news_source_count": 0,
        }

    scores  = score_articles(articles)
    sources = {
        a.get("source", "")
        for a in articles
        if a.get("source")
    }

    return {
        "ticker":             ticker,
        "date":               date_str,
        "news_count_1d":      len(articles),
        "news_sentiment_1d":  float(np.mean(scores))  if scores else 0.0,
        "news_sentiment_max": float(max(scores))       if scores else 0.0,
        "news_sentiment_min": float(min(scores))       if scores else 0.0,
        "news_source_count":  len(sources),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 1E — Live daily fetch (called by daily_run_live.py)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_live_news_sentiment(
    tickers: list[str],
    api_key: str,
) -> dict[str, dict]:
    """
    Fetch today's news for all tickers via Finnhub.

    Returns {ticker: {'news_count_1d': int,
                       'news_sentiment_1d': float,
                       'news_titles': list[str]}}
    — same format as fetch_yfinance_news() for drop-in use.

    Missing tickers default to neutral (not included in output).
    """
    # NOTE: Finnhub /news-sentiment endpoint
    # requires paid tier (returns 403 on free).
    # This function uses /company-news articles
    # + VADER sentiment scoring instead.
    # Do NOT add /news-sentiment calls here.

    today     = date.today().strftime("%Y-%m-%d")
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    results: dict[str, dict] = {}

    for ticker in tickers:
        # Primary path: Finnhub native sentiment — no model loading
        native_score = fetch_native_sentiment(ticker, api_key)
        if native_score is not None:
            results[ticker] = {
                "news_count_1d":     1,
                "news_sentiment_1d": round(native_score, 4),
                "news_titles":       [],
            }
            time.sleep(RATE_LIMIT_SLEEP)
            continue

        # Fallback: score article headlines with FinBERT/VADER
        articles = fetch_ticker_news(ticker, yesterday, today, api_key)
        if articles:
            scores = score_articles(articles)
            titles = [
                (a.get("headline") or "").strip()
                for a in articles[:3]
                if a.get("headline")
            ]
            results[ticker] = {
                "news_count_1d":     len(articles),
                "news_sentiment_1d": round(
                    float(np.mean(scores)) if scores else 0.0, 4
                ),
                "news_titles": titles,
            }

        time.sleep(RATE_LIMIT_SLEEP)

    logger.info(f"finnhub_live_fetched tickers={len(results)}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Helper for backfill: fetch full history in monthly batches
# ─────────────────────────────────────────────────────────────────────────────

def fetch_ticker_history(
    ticker: str,
    start: str,
    end: str,
    api_key: str,
    batch_days: int = 30,
) -> list[dict]:
    """
    Fetch all articles for a ticker over a date range in monthly batches.
    Returns deduplicated list of article dicts (dedup by article id).
    """
    all_articles: list[dict] = []
    seen_ids: set             = set()

    current = datetime.strptime(start, "%Y-%m-%d")
    end_dt  = datetime.strptime(end,   "%Y-%m-%d")

    while current < end_dt:
        batch_end = min(current + timedelta(days=batch_days), end_dt)
        batch     = fetch_ticker_news(
            ticker,
            current.strftime("%Y-%m-%d"),
            batch_end.strftime("%Y-%m-%d"),
            api_key,
        )
        for article in batch:
            a_id = article.get("id")
            if a_id and a_id not in seen_ids:
                seen_ids.add(a_id)
                all_articles.append(article)
            elif not a_id:
                all_articles.append(article)

        current = batch_end
        time.sleep(RATE_LIMIT_SLEEP)

    return all_articles


def articles_to_daily_rows(
    articles: list[dict],
    ticker: str,
) -> list[dict]:
    """
    Group articles by date and produce one daily sentiment row per date.
    Uses article['datetime'] (Unix timestamp) for date assignment.
    """
    from collections import defaultdict

    by_date: dict[str, list[dict]] = defaultdict(list)

    for a in articles:
        ts = a.get("datetime", 0)
        if ts:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            by_date[dt.strftime("%Y-%m-%d")].append(a)

    rows = []
    for date_str, day_articles in sorted(by_date.items()):
        rows.append(compute_daily_sentiment(day_articles, ticker, date_str))

    return rows
