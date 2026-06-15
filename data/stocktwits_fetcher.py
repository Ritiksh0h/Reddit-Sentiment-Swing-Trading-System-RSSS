"""
StockTwits API fetcher.
Fetches recent posts for tracked tickers. No API key required.

StockTwits endpoint:
    GET https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json

StockTwits posts include a native sentiment field (bullish/bearish)
set by the user when posting. No NLP needed for basic sentiment.

Output format:
  {ticker: {
      'st_count_1d':     int,
      'st_sentiment_1d': float,   # bullish ratio: +1.0 = all bullish
      'st_bull_pct':     float,   # % of posts tagged bullish
  }}
"""
import logging
import time
import requests

logger = logging.getLogger(__name__)

STOCKTWITS_BASE = 'https://api.stocktwits.com/api/2/streams/symbol'

TRACKED_TICKERS = [
    'NVDA', 'TSLA', 'AMD', 'AAPL', 'GME', 'AMC', 'PLTR', 'MARA', 'COIN',
    'META', 'MSFT', 'AMZN', 'GOOG', 'NFLX', 'SOFI', 'HOOD',
    'ROKU', 'SNAP', 'UBER', 'NIO', 'BABA', 'SHOP', 'PYPL',
    'DKNG', 'DIS', 'RKLB', 'HIMS', 'RDDT', 'SOUN', 'IONQ', 'F',
    'BA', 'BB', 'GS', 'JPM', 'BAC', 'SQ', 'NOK', 'SPCE',
]


def fetch_stocktwits(
    max_messages_per_ticker: int = 30,
    min_messages: int = 3,
) -> dict:
    """
    Fetch recent StockTwits messages for all tracked tickers.

    Uses the native bullish/bearish sentiment field — no NLP required.
    Rate limit: ~200 requests/hour unauthenticated. Sleeps 0.5s between calls.

    Returns:
        dict of ticker → {
            st_count_1d:     int,
            st_sentiment_1d: float,   # net sentiment: (bull-bear)/(bull+bear)
            st_bull_pct:     float,   # fraction of tagged posts that are bullish
        }

    On fetch failure: skips that ticker and continues.
    Returns empty dict only if ALL tickers fail.
    """
    # Browser User-Agent required — Cloudflare blocks the default python-requests UA.
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        ),
        'Accept': 'application/json',
    }

    result   = {}
    failures = 0

    for ticker in TRACKED_TICKERS:
        try:
            url  = f'{STOCKTWITS_BASE}/{ticker}.json'
            resp = requests.get(
                url, headers=headers, timeout=10,
                params={'limit': max_messages_per_ticker},
            )

            if resp.status_code == 403:
                logger.warning(
                    f'stocktwits_forbidden ticker={ticker} status=403 — '
                    'skipping ticker'
                )
                failures += 1
                continue

            if resp.status_code == 429:
                logger.warning(f'stocktwits_rate_limit ticker={ticker} — sleeping 60s')
                time.sleep(60)
                continue

            if resp.status_code == 404:
                # Ticker not on StockTwits — skip silently
                continue

            resp.raise_for_status()
            data     = resp.json()
            messages = data.get('messages', [])

            if len(messages) < min_messages:
                time.sleep(0.5)
                continue

            # Count native sentiment tags.
            # entities.sentiment can be null in JSON → use `or {}` not default arg
            # so a None value is also replaced with an empty dict.
            def _sentiment_basic(m):
                entities  = m.get('entities') or {}
                sentiment = entities.get('sentiment') or {}
                return sentiment.get('basic')

            bull = sum(1 for m in messages if _sentiment_basic(m) == 'Bullish')
            bear = sum(1 for m in messages if _sentiment_basic(m) == 'Bearish')
            total_tagged = bull + bear

            # Net sentiment: +1.0 = all bullish, -1.0 = all bearish, 0 = neutral/mixed
            if total_tagged > 0:
                net_sentiment = (bull - bear) / total_tagged
                bull_pct      = bull / total_tagged
            else:
                net_sentiment = 0.0
                bull_pct      = 0.5   # neutral default

            result[ticker] = {
                'st_count_1d':     len(messages),
                'st_sentiment_1d': round(net_sentiment, 4),
                'st_bull_pct':     round(bull_pct, 4),
            }

            time.sleep(0.5)   # polite rate limiting

        except requests.RequestException as e:
            logger.warning(f'stocktwits_fetch_failed ticker={ticker} error={e}')
            failures += 1
            continue
        except Exception as e:
            logger.warning(f'stocktwits_parse_failed ticker={ticker} error={e}')
            continue

    logger.info(f'stocktwits_fetched tickers={len(result)} failures={failures}')
    return result
