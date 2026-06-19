"""
News fetcher using yfinance.
Replaces data/tiingo_fetcher.py — drop-in replacement, same output format.

No API key required. yfinance is already installed.
Coverage: 20-35 tickers with recent news on a typical trading day.
Latency: ~15-25 seconds for full ticker list.
"""
import logging
import time

import yfinance as yf

logger = logging.getLogger(__name__)

from config.settings import load_tickers, TICKERS_TRADE_PATH, TICKERS_WATCH_PATH

TRACKED_TICKERS = sorted(
    set(load_tickers(TICKERS_TRADE_PATH)) | set(load_tickers(TICKERS_WATCH_PATH))
)


def _score_headlines_finbert(headlines: list) -> list:
    """Run FinBERT on headlines. Returns scores -1.0 to +1.0."""
    if not headlines:
        return []
    try:
        from transformers import pipeline as hf_pipeline
        import torch
        if not hasattr(_score_headlines_finbert, '_model'):
            device = 0 if torch.cuda.is_available() else -1
            _score_headlines_finbert._model = hf_pipeline(
                'text-classification',
                model='ProsusAI/finbert',
                device=device,
                truncation=True,
                max_length=128,
            )
        results = _score_headlines_finbert._model([h[:128] for h in headlines])
        scores = []
        for r in results:
            if r['label'] == 'positive':   scores.append(r['score'])
            elif r['label'] == 'negative': scores.append(-r['score'])
            else:                          scores.append(0.0)
        return scores
    except Exception as e:
        logger.warning(f'finbert_failed: {e} — returning neutral')
        return [0.0] * len(headlines)


def fetch_yfinance_news(max_articles_per_ticker: int = 10) -> dict:
    """
    Fetch recent news headlines via yfinance for all tracked tickers.
    Same output format as tiingo_fetcher.fetch_tiingo_news() — drop-in replacement.

    Returns:
        dict of ticker → {
            news_count_1d:     int,
            news_sentiment_1d: float,   # mean FinBERT score -1.0 to +1.0
            news_titles:       list,    # first 3 headlines for logging
        }
    """
    result = {}

    for ticker in TRACKED_TICKERS:
        try:
            t    = yf.Ticker(ticker)
            news = t.news or []
            if not news:
                continue

            # Handle both old and new yfinance response formats
            titles = []
            for n in news[:max_articles_per_ticker]:
                title = (
                    n.get('content', {}).get('title') or
                    n.get('title') or
                    ''
                )
                if title:
                    titles.append(title.strip())

            if not titles:
                continue

            scores = _score_headlines_finbert(titles)
            result[ticker] = {
                'news_count_1d':     len(titles),
                'news_sentiment_1d': round(
                    sum(scores) / len(scores), 4
                ) if scores else 0.0,
                'news_titles': titles[:3],
            }
            time.sleep(0.1)

        except Exception as e:
            logger.warning(f'yfinance_news_failed ticker={ticker}: {e}')
            continue

    logger.info(f'yfinance_news_fetched tickers={len(result)}')
    return result
