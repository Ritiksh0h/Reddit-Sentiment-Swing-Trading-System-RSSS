"""
Retroactive V2 signal run for missing paper trading dates.

Fetches real historical Reddit data from Arctic Shift (arbitrary Unix timestamps),
computes features from yfinance OHLCV ending on each target date, runs V2 model,
and logs BULLISH/BEARISH signals to paper_trades.jsonl.

Does NOT call daily_run.run() — this avoids corrupting paper_portfolio.json and
ensures market features are computed from the correct historical date, not today.

Usage:
    python scripts/retroactive_run.py --dry-run          # preview, no writes
    python scripts/retroactive_run.py                    # all missing Jun 15 – today
    python scripts/retroactive_run.py --start 2026-06-22 --end 2026-06-25
"""
import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xgboost as xgb
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sys.path.insert(0, '.')

Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/retroactive_run.log'),
    ],
)
log = logging.getLogger('retroactive_run')

ARCTIC_SHIFT_URL = 'https://arctic-shift.photon-reddit.com/api/posts/search'
SUBREDDITS       = ['wallstreetbets', 'stocks', 'investing', 'options']
PAGE_LIMIT       = 100

from config.settings import load_tickers, TICKERS_TRADE_PATH, TICKERS_DROP_PATH
from data.reddit_live_fetcher import extract_tickers_from_text
from portfolio.signal_generator import (
    compute_features_live, load_models, FEATURES,
    DENSITY_GATE, BULLISH_THRESHOLD, BEARISH_THRESHOLD, MIN_PRED_RET,
    DROP_TICKERS, TRADE_UNIVERSE,
)
from portfolio.execution_logger import log_signal

_vader = SentimentIntensityAnalyzer()


def _open_set() -> set:
    """Return {(date, ticker)} for all OPEN records already in paper_trades.jsonl."""
    seen = set()
    pt   = Path('logs/paper_trades.jsonl')
    if not pt.exists():
        return seen
    for line in pt.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            if r.get('action') == 'OPEN':
                seen.add((r['date'], r['ticker']))
        except Exception:
            pass
    return seen


def get_missing_dates(start: date, end: date) -> list:
    """Return weekdays in [start, end] that have no OPEN records logged."""
    logged_dates = {d for d, _ in _open_set()}
    days, cur = [], start
    while cur <= end:
        if cur.weekday() < 5 and cur.isoformat() not in logged_dates:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def fetch_reddit_for_date(target: date, max_pages: int = 5) -> dict:
    """
    Fetch Reddit posts from Arctic Shift for target date (UTC midnight to midnight).
    Returns {ticker: {post_count_1d, total_comments_1d, vader_sentiment_1d, ...}}
    """
    after  = int(datetime(target.year, target.month, target.day,
                          0, 0, 0, tzinfo=timezone.utc).timestamp())
    before = int(datetime(target.year, target.month, target.day,
                          23, 59, 59, tzinfo=timezone.utc).timestamp())

    ticker_posts: dict = defaultdict(list)

    for sub in SUBREDDITS:
        page_before = before
        sub_total   = 0
        for _ in range(max_pages):
            try:
                resp = requests.get(
                    ARCTIC_SHIFT_URL,
                    params={'subreddit': sub, 'after': after,
                            'before': page_before, 'limit': PAGE_LIMIT},
                    headers={'User-Agent': 'rsss-retroactive/1.0'},
                    timeout=20,
                )
                resp.raise_for_status()
                posts = resp.json().get('data', [])
            except Exception as e:
                log.warning(f'  arctic_shift_fail sub={sub}: {e}')
                break

            if not posts:
                break

            for post in posts:
                title = post.get('title', '') or ''
                for ticker in extract_tickers_from_text(title):
                    ticker_posts[ticker].append(post)

            sub_total += len(posts)
            if len(posts) < PAGE_LIMIT:
                break
            page_before = min(p.get('created_utc', before) for p in posts) - 1
            if page_before <= after:
                break
            time.sleep(0.3)

        log.info(f'  sub={sub} posts={sub_total}')
        time.sleep(0.5)

    result = {}
    for ticker, posts in ticker_posts.items():
        scores = [
            _vader.polarity_scores(p.get('title', ''))['compound']
            for p in posts if (p.get('title') or '').strip()
        ]
        result[ticker] = {
            'post_count_1d':     len(posts),
            'total_comments_1d': int(sum(p.get('num_comments', 0) for p in posts)),
            'vader_sentiment_1d': round(sum(scores) / len(scores), 4) if scores else 0.0,
            'mention_growth_1d': 1.0,
            'mention_growth_7d': 1.0,
        }

    log.info(f'Reddit {target}: {len(result)} tickers with mentions')
    return result


def get_spy_vix(target: date) -> tuple:
    """Return (spy_above_200ma, vix_percentile) using historical data ending at target."""
    start     = (target - timedelta(days=365)).isoformat()
    end       = (target + timedelta(days=1)).isoformat()
    spy_above = 1.0
    vix_pct   = 0.5

    try:
        spy = yf.download('SPY', start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        sc    = spy['Close'].dropna()
        ma200 = float(sc.rolling(200, min_periods=100).mean().iloc[-1])
        spy_above = 1.0 if float(sc.iloc[-1]) > ma200 else 0.0
    except Exception as e:
        log.warning(f'SPY fetch failed: {e}')

    try:
        vix = yf.download('^VIX', start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        vals   = vix['Close'].dropna().values
        window = vals[-252:] if len(vals) >= 252 else vals
        vix_pct = float(np.mean(window < vals[-1])) if len(window) >= 20 else 0.5
    except Exception as e:
        log.warning(f'VIX fetch failed: {e}')

    log.info(f'spy_above_200ma={spy_above:.0f} vix_pct={vix_pct:.3f}')
    return spy_above, vix_pct


def run_date(
    target:       date,
    models:       dict,
    mention_hist: dict,
    dry_run:      bool,
    already_open: set,
) -> int:
    """Fetch, compute, score, and log signals for one historical date."""
    date_str  = target.isoformat()
    end_str   = (target + timedelta(days=1)).isoformat()
    start_str = (target - timedelta(days=120)).isoformat()

    log.info(f'=== {date_str} ===')

    reddit = fetch_reddit_for_date(target)
    if not reddit:
        log.info(f'  no Reddit data — skipping')
        return 0

    spy_above, vix_pct = get_spy_vix(target)

    # Filter mention history to dates before target to avoid lookahead
    hist_slice = {
        t: {d: c for d, c in dates.items() if d < date_str}
        for t, dates in mention_hist.items()
    }

    ts     = datetime.now(timezone.utc).isoformat()
    logged = 0

    for ticker, rdata in sorted(reddit.items(), key=lambda x: -x[1]['post_count_1d']):
        if TRADE_UNIVERSE and ticker not in TRADE_UNIVERSE:
            continue
        if ticker in DROP_TICKERS:
            continue
        if (date_str, ticker) in already_open:
            log.info(f'  {ticker} already logged')
            continue

        post_count = rdata['post_count_1d']
        if post_count < DENSITY_GATE:
            continue

        try:
            # Explicit historical date range — correct features, not today's prices
            mkt = yf.download(ticker, start=start_str, end=end_str,
                               auto_adjust=True, progress=False)
            if isinstance(mkt.columns, pd.MultiIndex):
                mkt.columns = mkt.columns.get_level_values(0)
            if len(mkt) < 55:
                log.warning(f'  {ticker} only {len(mkt)} rows')
                continue
        except Exception as e:
            log.warning(f'  {ticker} OHLCV fail: {e}')
            continue

        # MA filter: same logic as live pipeline
        price = float(mkt['Close'].iloc[-1])
        ma20  = float(mkt['Close'].tail(20).mean())
        if price < ma20:
            log.info(f'  {ticker} MA-filter price={price:.2f} ma20={ma20:.2f}')
            continue

        feats = compute_features_live(
            ticker=ticker,
            market_data=mkt,
            post_count_1d=post_count,
            news_sentiment_1d=0.0,           # no historical FinBERT corpus
            total_comments_1d=rdata['total_comments_1d'],
            vader_sentiment_1d=rdata['vader_sentiment_1d'],
            mention_growth_7d=1.0,
            vix_percentile=vix_pct,
            spy_above_200ma=spy_above,
            mention_history=hist_slice,
        )
        if feats is None:
            continue

        avail = [f for f in FEATURES if f in feats]
        X     = pd.DataFrame([feats])[avail].fillna(0)
        dm    = xgb.DMatrix(X)

        pred_1d = float(models['1d'].predict(dm)[0]) if '1d' in models else 0.0
        pred_3d = float(models['3d'].predict(dm)[0]) if '3d' in models else 0.0
        pred_5d = float(models['5d'].predict(dm)[0]) if '5d' in models else 0.0

        if max(abs(pred_1d), abs(pred_3d), abs(pred_5d)) < MIN_PRED_RET:
            continue

        if pred_5d >= BULLISH_THRESHOLD:
            signal, hold, horiz, best = 'BULLISH', 5, '5D', pred_5d
        elif pred_3d >= BULLISH_THRESHOLD:
            signal, hold, horiz, best = 'BULLISH', 3, '3D', pred_3d
        elif pred_1d >= BULLISH_THRESHOLD:
            signal, hold, horiz, best = 'BULLISH', 1, '1D', pred_1d
        elif pred_5d <= BEARISH_THRESHOLD:
            signal, hold, horiz, best = 'BEARISH', 5, '5D', pred_5d
        elif pred_3d <= BEARISH_THRESHOLD:
            signal, hold, horiz, best = 'BEARISH', 3, '3D', pred_3d
        elif pred_1d <= BEARISH_THRESHOLD:
            signal, hold, horiz, best = 'BEARISH', 1, '1D', pred_1d
        else:
            continue  # NEUTRAL — no trade

        conf = round(min(abs(best) / (BULLISH_THRESHOLD * 2), 1.0), 4)

        log.info(
            f'  {"DRY" if dry_run else "LOG"} {ticker} {signal} '
            f'1D={pred_1d*100:+.2f}% 3D={pred_3d*100:+.2f}% 5D={pred_5d*100:+.2f}% '
            f'conf={conf:.2f} posts={post_count}'
        )

        if not dry_run:
            log_signal(
                ticker=ticker,
                date=date_str,
                feature_vector={k: round(v, 6) if isinstance(v, float) else v
                                 for k, v in feats.items()},
                regime_state='unknown',
                regime_multiplier=1.0,
                predicted_return_5d=round(pred_5d, 6),
                atr_14=feats['atr_14'],
                position_size_dollars=0.0,   # retroactive — portfolio not modified
                slippage_applied=0.0,
                fill_price=round(price, 4),
                signal_timestamp=ts,
                action='OPEN',
                hold_days=hold,
                horizon=horiz,
                predicted_1d=round(pred_1d, 6),
                predicted_3d=round(pred_3d, 6),
                signal=signal,
                price_target_1d=round(price * (1 + pred_1d), 4),
                price_target_3d=round(price * (1 + pred_3d), 4),
                price_target_5d=round(price * (1 + pred_5d), 4),
                confidence=conf,
                news_count_1d=0,
                st_count_1d=0,
                notes='retroactive_v2',
            )
            already_open.add((date_str, ticker))
            logged += 1

    return logged


def main():
    parser = argparse.ArgumentParser(description='Retroactive V2 signal run')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview signals without writing')
    parser.add_argument('--start', default='2026-06-15',
                        help='Start date YYYY-MM-DD (default: 2026-06-15)')
    parser.add_argument('--end', default=date.today().isoformat(),
                        help='End date YYYY-MM-DD (default: today)')
    args = parser.parse_args()

    start   = date.fromisoformat(args.start)
    end     = date.fromisoformat(args.end)
    missing = get_missing_dates(start, end)

    if not missing:
        log.info('No missing dates — nothing to do.')
        return

    log.info(f'{"DRY RUN — " if args.dry_run else ""}Processing {len(missing)} dates: '
             f'{[d.isoformat() for d in missing]}')

    models       = load_models()
    mention_hist = {}
    hist_path    = Path('data/mention_history.json')
    if hist_path.exists():
        mention_hist = json.loads(hist_path.read_text())

    already_open = _open_set()
    total        = 0

    for d in missing:
        n      = run_date(d, models, mention_hist, dry_run=args.dry_run,
                          already_open=already_open)
        total += n
        time.sleep(2)

    action = 'would log' if args.dry_run else 'logged'
    log.info(f'Done — {total} signals {action} across {len(missing)} dates')


if __name__ == '__main__':
    main()
