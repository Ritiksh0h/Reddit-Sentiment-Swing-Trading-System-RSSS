"""
Live Reddit post fetcher using Arctic Shift API.
Uses a 24-72h delay window to stay within Arctic Shift's indexing lag.

Arctic Shift endpoint:
    https://arctic-shift.photon-reddit.com/api/posts/search

No authentication required.
Rate limit: ~60 requests per minute.
Fallback: if API fails, returns empty dict and triggers api_anomaly handler.
"""
import json
import re
import time
import logging
from datetime import datetime, date, timezone, timedelta
from collections import defaultdict
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

ARCTIC_SHIFT_BASE  = 'https://arctic-shift.photon-reddit.com/api/posts/search'
TRACKED_SUBREDDITS = ['wallstreetbets', 'stocks', 'investing', 'options', 'SecurityAnalysis']

# Delay window: fetch posts from 72h ago up to 24h ago.
# Arctic Shift indexes posts with a ~24-48h lag — fetching too recent
# risks missing posts that haven't been indexed yet.
FETCH_HOURS_START = 72  # window open (older boundary)
FETCH_HOURS_END   = 24  # window close (newer boundary)

USER_AGENT = 'rsss-swing-trader/1.0'

from config.settings import load_tickers, TICKERS_TRADE_PATH, TICKERS_WATCH_PATH

# Fetch Reddit data for all tickers: proven signal generators (trade) +
# new additions building history (watch). Signal generation uses trade only.
TRACKED_TICKERS = sorted(
    set(load_tickers(TICKERS_TRADE_PATH)) | set(load_tickers(TICKERS_WATCH_PATH))
)

DOLLAR_PATTERN = re.compile(r'\$([A-Z]{1,5})\b')
TICKER_SET     = set(TRACKED_TICKERS)

# False positives — never treat these as tickers
FALSE_POSITIVES = {
    'I', 'A', 'THE', 'FOR', 'ARE', 'BE', 'DO', 'OR', 'AN', 'IT',
    'IF', 'GO', 'SO', 'ON', 'IN', 'AT', 'US', 'DD', 'IMO', 'EPS',
    'YOY', 'ATH', 'ETF', 'IPO', 'CEO', 'CFO', 'SEC', 'IRS',
}


def extract_tickers_from_text(text: str) -> list:
    """Extract ticker mentions from post title/body."""
    found = set()
    # Dollar-sign format: $NVDA — highest confidence
    for m in DOLLAR_PATTERN.finditer(text or ''):
        sym = m.group(1)
        if sym in TICKER_SET and sym not in FALSE_POSITIVES:
            found.add(sym)
    # Direct match for known tickers (word boundary)
    for ticker in TRACKED_TICKERS:
        if re.search(rf'\b{re.escape(ticker)}\b', text or ''):
            found.add(ticker)
    return list(found)


API_PAGE_LIMIT = 100   # Arctic Shift hard limit per request


def _fetch_subreddit_page(subreddit: str, after: int, before: int) -> list:
    """Fetch one page (≤100 posts) from Arctic Shift."""
    params = {
        'subreddit': subreddit,
        'after':     after,
        'before':    before,
        'limit':     API_PAGE_LIMIT,
    }
    headers = {'User-Agent': USER_AGENT}
    resp = requests.get(ARCTIC_SHIFT_BASE, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json().get('data', [])


def fetch_recent_posts(
    hours_start: int = FETCH_HOURS_START,
    hours_end: int   = FETCH_HOURS_END,
    max_pages_per_subreddit: int = 5,
) -> dict:
    """
    Fetch posts from the 24–72h delay window across all tracked subreddits.
    Uses a delayed window (default: 72h → 24h ago) to stay within Arctic
    Shift's indexing lag — fetching real-time risks missing un-indexed posts.

    Returns:
        dict of ticker → {
            post_count_1d: int,
            mention_growth_1d: float,   # 1.0 placeholder until history accumulates
            mention_growth_7d: float,   # 1.0 placeholder until history accumulates
            posts: list
        }

    Returns empty dict on total API failure → triggers api_anomaly handler.
    """
    now    = datetime.now(timezone.utc)
    after  = int((now - timedelta(hours=hours_start)).timestamp())
    before = int((now - timedelta(hours=hours_end)).timestamp())

    ticker_posts  = defaultdict(list)
    total_fetched = 0

    for subreddit in TRACKED_SUBREDDITS:
        subreddit_count = 0
        page_before     = before

        try:
            for _ in range(max_pages_per_subreddit):
                posts = _fetch_subreddit_page(subreddit, after, page_before)
                if not posts:
                    break

                for post in posts:
                    title    = post.get('title', '')
                    score    = post.get('score', 0) or 0
                    comments = post.get('num_comments', 0) or 0
                    created  = post.get('created_utc', 0)

                    for ticker in extract_tickers_from_text(title):
                        ticker_posts[ticker].append({
                            'title':        title,
                            'score':        score,
                            'num_comments': comments,
                            'created_utc':  created,
                            'subreddit':    subreddit,
                        })

                subreddit_count += len(posts)
                total_fetched   += len(posts)

                if len(posts) < API_PAGE_LIMIT:
                    break  # no more pages

                # Advance window: next page fetches posts older than this batch
                oldest_ts = min(p.get('created_utc', before) for p in posts)
                page_before = oldest_ts - 1
                if page_before <= after:
                    break

                time.sleep(0.5)  # polite rate limiting between pages

            logger.info(f'subreddit={subreddit} fetched={subreddit_count}')
            time.sleep(1.0)  # polite between subreddits

        except requests.RequestException as e:
            logger.warning(f'arctic_shift_fetch_failed subreddit={subreddit} error={e}')
            continue
        except Exception as e:
            logger.error(f'unexpected_fetch_error subreddit={subreddit} error={e}')
            continue

    logger.info(f'posts_fetched total={total_fetched} tickers_found={len(ticker_posts)}')

    if total_fetched == 0:
        logger.warning('zero_posts_fetched — possible API outage')
        return {}

    reddit_counts = {}
    for ticker, posts in ticker_posts.items():
        reddit_counts[ticker] = {
            'post_count_1d':     len(posts),
            'mention_growth_1d': 1.0,  # replaced once history accumulates (week 3+)
            'mention_growth_7d': 1.0,
            'posts':             posts,
        }

    return reddit_counts


def compute_mention_growth(
    ticker: str,
    current_count: int,
    history_db_path: str = 'data/mention_history.json',
) -> dict:
    """
    Compute mention_growth_1d and mention_growth_7d from accumulated history.

    Replaces the 1.0 placeholders after 7+ days of live data.
    History stored as a rolling 30-day buffer in data/mention_history.json —
    30 calendar days ≈ 21 weekday entries, enough for the 20-day rolling
    baseline behind abnormal_attention_1d (14 days could never hold 20 entries).
    """
    history_path = Path(history_db_path)
    today_str    = date.today().isoformat()

    history = {}
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

    ticker_history = history.get(ticker, {})

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    week_ago  = (date.today() - timedelta(days=7)).isoformat()

    count_yesterday = ticker_history.get(yesterday, current_count)
    count_week_ago  = ticker_history.get(week_ago, current_count)

    growth_1d = current_count / max(count_yesterday, 1)
    growth_7d = current_count / max(count_week_ago, 1)

    # Update and trim history
    ticker_history[today_str] = current_count
    cutoff = (date.today() - timedelta(days=30)).isoformat()
    ticker_history = {k: v for k, v in ticker_history.items() if k >= cutoff}
    history[ticker] = ticker_history

    Path(history_db_path).parent.mkdir(exist_ok=True)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    return {
        'mention_growth_1d': round(growth_1d, 4),
        'mention_growth_7d': round(growth_7d, 4),
    }
