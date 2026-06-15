"""
Tiingo News API fetcher.
Fetches financial news headlines for tracked tickers from last 24 hours.

Free tier limits:
  - 500 requests/day
  - News endpoint: GET /tiingo/news

No complex NLP extraction needed — Tiingo already tags articles by ticker.

Output format matches reddit_live_fetcher.py:
  {ticker: {
      'news_count_1d':      int,
      'news_sentiment_1d':  float,  # FinBERT on headlines
      'news_titles':        list,   # for logging
  }}
"""
import os
import logging
import requests
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

TIINGO_BASE = 'https://api.tiingo.com/tiingo/news'


def _get_api_key() -> str:
    return os.getenv('TIINGO_API_KEY', '')

TRACKED_TICKERS = [
    'NVDA', 'TSLA', 'AMD', 'AAPL', 'GME', 'AMC', 'PLTR', 'MARA', 'COIN',
    'META', 'MSFT', 'AMZN', 'GOOG', 'NFLX', 'SOFI', 'HOOD',
    'ROKU', 'SNAP', 'UBER', 'NIO', 'BABA', 'SHOP', 'PYPL',
    'DKNG', 'DIS', 'RKLB', 'HIMS', 'RDDT', 'SOUN', 'IONQ', 'F',
    'BA', 'BB', 'GS', 'JPM', 'BAC', 'SQ', 'NOK', 'SPCE',
]


def _score_headlines_finbert(headlines: list[str]) -> list[float]:
    """
    Run FinBERT on a list of headlines.
    Returns list of sentiment scores (-1.0 to +1.0).
    Loads model once and caches it.
    """
    if not headlines:
        return []

    try:
        from transformers import pipeline as hf_pipeline
        import torch

        # Cache model across calls
        if not hasattr(_score_headlines_finbert, '_model'):
            device = 0 if torch.cuda.is_available() else -1
            _score_headlines_finbert._model = hf_pipeline(
                'text-classification',
                model='ProsusAI/finbert',
                device=device,
                truncation=True,
                max_length=128,
            )

        model   = _score_headlines_finbert._model
        results = model([h[:128] for h in headlines])

        scores = []
        for r in results:
            label = r['label']
            score = r['score']
            if label == 'positive':
                scores.append(score)
            elif label == 'negative':
                scores.append(-score)
            else:
                scores.append(0.0)
        return scores

    except Exception as e:
        logger.warning(f'finbert_failed error={e} — returning neutral scores')
        return [0.0] * len(headlines)


def fetch_tiingo_news(
    hours_back: int = 24,
    max_articles_per_ticker: int = 20,
) -> dict:
    """
    Fetch news headlines from Tiingo for all tracked tickers.

    Returns:
        dict of ticker → {
            news_count_1d:     int,
            news_sentiment_1d: float,   # mean FinBERT score
            news_titles:       list,    # for execution logging
        }

    On API failure: returns empty dict (triggers api_anomaly handler).
    On missing API key: logs warning and returns empty dict.
    """
    api_key = _get_api_key()
    if not api_key:
        logger.warning(
            'TIINGO_API_KEY not set. Add to .env file. '
            'Get free key at https://api.tiingo.com'
        )
        return {}

    now        = datetime.now(timezone.utc)
    start_time = (now - timedelta(hours=hours_back)).strftime('%Y-%m-%dT%H:%M:%SZ')

    headers = {'Content-Type': 'application/json'}

    ticker_news = defaultdict(list)
    total_fetched = 0

    # Fetch in batches to avoid rate limits
    # Tiingo allows filtering by tickers directly
    batch_size = 10
    for i in range(0, len(TRACKED_TICKERS), batch_size):
        batch   = TRACKED_TICKERS[i:i + batch_size]
        tickers = ','.join(batch)

        try:
            params = {
                'token':     api_key,
                'tickers':   tickers,
                'startDate': start_time,
                'limit':     100,
                'sortBy':    'publishedDate',
            }
            resp = requests.get(
                TIINGO_BASE, headers=headers,
                params=params, timeout=15,
            )

            if resp.status_code == 401:
                logger.error(
                    'tiingo_auth_failed status=401 — '
                    'token is invalid. Check TIINGO_API_KEY in .env.'
                )
                return {}

            if resp.status_code == 403:
                # News API requires a paid Tiingo plan add-on.
                # Free tier covers EOD prices only.
                # Bail immediately instead of logging one warning per batch.
                logger.warning(
                    'tiingo_news_forbidden status=403 — '
                    '"You do not have permission to access the News API". '
                    'Upgrade your plan at https://api.tiingo.com/about/pricing. '
                    'Returning empty result — news_sentiment_1d defaults to 0.0.'
                )
                return {}

            resp.raise_for_status()
            articles = resp.json()
            total_fetched += len(articles)

            for article in articles:
                title   = article.get('title', '') or ''
                desc    = article.get('description', '') or ''
                text    = (title + ' ' + desc).strip()
                tickers_in_article = article.get('tickers', [])

                for ticker in tickers_in_article:
                    ticker = ticker.upper()
                    if ticker in set(TRACKED_TICKERS):
                        ticker_news[ticker].append({
                            'title':     title,
                            'text':      text,
                            'published': article.get('publishedDate', ''),
                        })

        except requests.RequestException as e:
            logger.warning(f'tiingo_fetch_failed batch={batch[:3]} error={e}')
            continue

    logger.info(f'tiingo_articles_fetched total={total_fetched} '
                f'tickers_with_news={len(ticker_news)}')

    if total_fetched == 0:
        logger.warning('tiingo_zero_articles — API may be down or key invalid')
        return {}

    # Score sentiment with FinBERT
    result = {}
    for ticker, articles in ticker_news.items():
        headlines = [a['text'][:128] for a in articles[:max_articles_per_ticker]]
        scores    = _score_headlines_finbert(headlines)

        result[ticker] = {
            'news_count_1d':     len(articles),
            'news_sentiment_1d': float(sum(scores) / len(scores)) if scores else 0.0,
            'news_titles':       [a['title'] for a in articles[:5]],  # log first 5
        }

    logger.info(f'tiingo_sentiment_scored tickers={len(result)}')
    return result
