#!/usr/bin/env python3
"""
Nightly ticker discovery scanner.

Scans raw Reddit post titles for capitalized 1-5 letter tokens that are NOT
already in TRACKED_TICKERS (config/tickers_trade.txt + tickers_watch.txt).
Validates each candidate via yfinance (must be a real, liquid US equity).
Tracks daily mention counts in data/discovery_candidates.json. A candidate
that clears MIN_DISCOVERY_MENTIONS for MIN_DISCOVERY_DAYS consecutive days
is appended to config/tickers_watch.txt.

Never touches tickers_trade.txt — promotion to live trading stays manual.

Run nightly via launchd, after the main daily_run completes (e.g. 16:00 ET),
so it doesn't compete with the live pipeline for Arctic Shift rate limits.
"""
import sys, os, re, json, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import Counter

import yfinance as yf

from config.settings import load_tickers, TICKERS_TRADE_PATH, TICKERS_WATCH_PATH, TICKERS_DROP_PATH
from data.reddit_live_fetcher import TRACKED_SUBREDDITS, _fetch_subreddit_page, FALSE_POSITIVES

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

CANDIDATES_PATH        = Path('data/discovery_candidates.json')
MIN_DISCOVERY_MENTIONS = 8      # min posts/day to count as a candidate day
MIN_DISCOVERY_DAYS     = 3      # consecutive qualifying days before promotion
MAX_CANDIDATE_AGE_DAYS = 14     # drop candidates that haven't requalified in this long

# Broader token pattern than extract_tickers_from_text — this is discovery,
# not extraction, so it intentionally over-matches and relies on the
# yfinance validation step + mention-count gate to filter noise.
TOKEN_PATTERN = re.compile(r'\$?\b([A-Z]{1,5})\b')

# Common English words / forum jargon that survive the FALSE_POSITIVES set
# because that set was built for known-ticker context, not raw discovery.
DISCOVERY_NOISE = FALSE_POSITIVES | {
    'YOLO', 'FOMO', 'WSB', 'TLDR', 'TBH', 'IMO', 'AKA', 'ASAP',
    'USA', 'UK', 'EU', 'GDP', 'CPI', 'FED', 'PE', 'PT', 'AH',
    'OTM', 'ITM', 'YTD', 'QOQ', 'IRA', 'LOL', 'WTF', 'NSFW',
}


def load_candidates() -> dict:
    """Supabase kv_store first (survives fresh GH Actions runners), file fallback."""
    try:
        from api.db import kv_get
        remote = kv_get('discovery_candidates')
        if remote is not None:
            return remote if isinstance(remote, dict) else {}
    except Exception:
        pass
    if CANDIDATES_PATH.exists():
        return json.loads(CANDIDATES_PATH.read_text())
    return {}


def save_candidates(data: dict) -> None:
    CANDIDATES_PATH.parent.mkdir(exist_ok=True)
    CANDIDATES_PATH.write_text(json.dumps(data, indent=2, sort_keys=True))
    try:
        from api.db import kv_set
        kv_set('discovery_candidates', data)
    except Exception:
        pass


def _load_watch_list() -> set:
    """Supabase kv_store first, config/tickers_watch.txt fallback."""
    try:
        from api.db import kv_get
        remote = kv_get('tickers_watch')
        if remote and isinstance(remote, list):
            return set(remote)
    except Exception:
        pass
    return set(load_tickers(TICKERS_WATCH_PATH))


def _save_watch_list(watch_set: set) -> None:
    with open(Path(TICKERS_WATCH_PATH), 'w') as f:
        for t in sorted(watch_set):
            f.write(f'{t}\n')
    try:
        from api.db import kv_set
        kv_set('tickers_watch', sorted(watch_set))
    except Exception:
        pass


def is_valid_liquid_equity(symbol: str) -> bool:
    """
    Validate a candidate is a real, liquid, tradeable US equity.
    Mirrors the yf.Ticker validation pattern already used in
    data/news_fetcher.py and data/earnings_fetcher.py.
    Returns False on any lookup failure (fail closed — never auto-add
    on an ambiguous or errored yfinance response).
    """
    try:
        t = yf.Ticker(symbol)
        info = t.fast_info
        price       = getattr(info, 'last_price', None)
        avg_volume  = getattr(info, 'three_month_average_volume', None)
        if price is None or price <= 0:
            return False
        if avg_volume is None or avg_volume < 500_000:
            logger.debug(f'discovery_reject {symbol}: avg_volume={avg_volume} (too illiquid)')
            return False
        return True
    except Exception as e:
        logger.debug(f'discovery_reject {symbol}: yfinance lookup failed: {e}')
        return False


def scan_today() -> Counter:
    """
    Fetch the last 24h of raw posts from all tracked subreddits and count
    mentions of any token NOT already in TRACKED_TICKERS.
    """
    from data.reddit_live_fetcher import TRACKED_TICKERS

    now    = datetime.now(timezone.utc)
    after  = int((now - timedelta(hours=48)).timestamp())
    before = int((now - timedelta(hours=24)).timestamp())

    known = set(TRACKED_TICKERS)
    counts: Counter = Counter()

    for subreddit in TRACKED_SUBREDDITS:
        try:
            page_before = before
            for _ in range(5):
                posts = _fetch_subreddit_page(subreddit, after, page_before)
                if not posts:
                    break
                for post in posts:
                    title = post.get('title', '') or ''
                    for m in TOKEN_PATTERN.finditer(title):
                        sym = m.group(1)
                        if sym in known or sym in DISCOVERY_NOISE:
                            continue
                        counts[sym] += 1
                oldest = min(p.get('created_utc', before) for p in posts)
                page_before = oldest - 1
                if page_before <= after:
                    break
        except Exception as e:
            logger.warning(f'discovery_fetch_failed subreddit={subreddit}: {e}')

    return counts


def run_discovery():
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    candidates = load_candidates()
    today_counts = scan_today()

    logger.info(f'discovery_scan date={today} raw_unknown_tokens={len(today_counts)}')

    promoted = []
    watch_list = _load_watch_list()
    trade_list = set(load_tickers(TICKERS_TRADE_PATH))
    drop_list  = set(load_tickers(TICKERS_DROP_PATH))

    for symbol, count in today_counts.items():
        if symbol in trade_list or symbol in watch_list or symbol in drop_list:
            continue  # already known somewhere — discovery only cares about unknowns

        record = candidates.get(symbol, {'days': {}, 'first_seen': today})

        if count >= MIN_DISCOVERY_MENTIONS:
            record['days'][today] = count
        candidates[symbol] = record

        # Prune stale day-entries older than MAX_CANDIDATE_AGE_DAYS
        cutoff = (datetime.now(timezone.utc) - timedelta(days=MAX_CANDIDATE_AGE_DAYS)).strftime('%Y-%m-%d')
        record['days'] = {d: c for d, c in record['days'].items() if d >= cutoff}

        qualifying_days = sorted(record['days'].keys())[-MIN_DISCOVERY_DAYS:]
        if len(qualifying_days) >= MIN_DISCOVERY_DAYS:
            consecutive = True
            for i in range(len(qualifying_days) - 1):
                d1 = datetime.fromisoformat(qualifying_days[i])
                d2 = datetime.fromisoformat(qualifying_days[i + 1])
                if (d2 - d1).days > 3:  # allow weekend gaps
                    consecutive = False
                    break
            if consecutive and is_valid_liquid_equity(symbol):
                promoted.append(symbol)
                logger.info(f'discovery_promote symbol={symbol} '
                            f'days={qualifying_days} counts={[record["days"][d] for d in qualifying_days]}')

    if promoted:
        for symbol in promoted:
            watch_list.add(symbol)
            candidates.pop(symbol, None)  # remove from candidate tracking once promoted
        _save_watch_list(watch_list)
        logger.info(f'discovery_complete promoted={promoted} appended_to=supabase+local')
    else:
        logger.info('discovery_complete promoted=none')

    save_candidates(candidates)

    return {
        'date': today,
        'raw_unknown_tokens': len(today_counts),
        'top_unknown': dict(today_counts.most_common(10)),
        'promoted': promoted,
        'active_candidates': len(candidates),
    }


if __name__ == '__main__':
    result = run_discovery()
    print(json.dumps(result, indent=2))
