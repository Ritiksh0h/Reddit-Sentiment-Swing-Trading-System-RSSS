"""
Alpha Vantage news sentiment collector.

Usage:
  python scripts/collect_av_news.py                        # daily (yesterday)
  python scripts/collect_av_news.py --dry-run              # print plan only
  python scripts/collect_av_news.py --test-one             # fetch NVDA only
  python scripts/collect_av_news.py --backfill \
      --start 2023-01-01 --end 2025-06-01                  # historical backfill
  python scripts/collect_av_news.py --merge                # merge all sources

Output files:
  data/processed/news_features_live.parquet        (daily append target)
  data/processed/news_features_2023_2025.parquet   (backfill output)
  data/processed/news_features_merged.parquet      (final merged)
  data/processed/av_backfill_progress.json         (resume state)

Rate limit: 25 calls/day on free tier.
  Daily mode:   8 calls (39 tickers / 5 per batch)
  Backfill:     8 calls/month — spread across days with progress file
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
log = logging.getLogger('av_news')

# ── Config ────────────────────────────────────────────────────────────────────

TICKERS_FILE        = Path('config/tickers.txt')
LIVE_OUT            = Path('data/processed/news_features_live.parquet')
BACKFILL_OUT        = Path('data/processed/news_features_2023_2025.parquet')
MERGED_OUT          = Path('data/processed/news_features_merged.parquet')
PROGRESS_FILE       = Path('data/processed/av_backfill_progress.json')
FNSPID_PATH         = Path('data/processed/news_features_2019_2023.parquet')

AV_URL              = 'https://www.alphavantage.co/query'
BATCH_SIZE          = 5      # tickers per API call
RELEVANCE_THRESHOLD = 0.3    # minimum relevance_score to include
DAILY_SLEEP_SEC     = 1.0    # pause between calls in daily mode
BACKFILL_SLEEP_SEC  = 12.0   # pause between calls in backfill mode
MAX_DAILY_CALLS     = 24     # stop before hitting AV 25/day hard limit


def _load_tickers() -> list[str]:
    """Load tracked ticker universe from config/tickers.txt."""
    if not TICKERS_FILE.exists():
        raise FileNotFoundError(f'{TICKERS_FILE} not found')
    tickers = []
    for line in TICKERS_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            tickers.append(line.upper())
    return tickers


def _batches(tickers: list[str], size: int = BATCH_SIZE) -> list[list[str]]:
    return [tickers[i:i + size] for i in range(0, len(tickers), size)]


def _fmt_dt(d: date) -> str:
    """Format date as YYYYMMDDTHHMM for Alpha Vantage."""
    return d.strftime('%Y%m%dT0000')


def _fetch_av(
    tickers_batch: list[str],
    time_from: str,
    time_to: str,
    sort: str = 'LATEST',
    dry_run: bool = False,
) -> list[dict]:
    """
    Call AV NEWS_SENTIMENT for up to 5 tickers over a time window.
    Returns raw article list.  Handles rate-limit responses with one retry.
    """
    api_key = os.environ.get('ALPHAVANTAGE_API_KEY', '')
    if not api_key:
        raise EnvironmentError(
            'ALPHAVANTAGE_API_KEY not set. Add it to .env or export it.'
        )

    ticker_str = ','.join(tickers_batch)
    params = {
        'function':   'NEWS_SENTIMENT',
        'tickers':    ticker_str,
        'time_from':  time_from,
        'time_to':    time_to,
        'sort':       sort,
        'limit':      '1000',
        'apikey':     api_key,
    }
    log.info(f'AV call | tickers={ticker_str} | {time_from} → {time_to}')

    if dry_run:
        return []

    for attempt in (1, 2):
        try:
            resp = requests.get(AV_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.error(f'Request failed (attempt {attempt}): {e}')
            if attempt == 1:
                time.sleep(5)
                continue
            return []

        # Rate-limit / info messages come back as top-level keys, not HTTP 429
        if 'Note' in data or 'Information' in data:
            msg = data.get('Note') or data.get('Information', '')
            log.warning(f'AV rate-limit response: {msg[:120]}')
            if attempt == 1:
                log.info('Sleeping 60 s before retry...')
                time.sleep(60)
                continue
            return []

        articles = data.get('feed', [])
        log.info(f'  → {len(articles)} articles')
        return articles

    return []


def _parse_articles(
    articles: list[dict],
    tracked: set[str],
) -> list[tuple[str, str, float]]:
    """
    Extract (ticker, date_str, sentiment_score) from raw article list.
    Filters: relevance_score > RELEVANCE_THRESHOLD, ticker in tracked set.
    """
    records = []
    for art in articles:
        raw_ts = art.get('time_published', '')
        if len(raw_ts) < 8:
            continue
        date_str = f'{raw_ts[:4]}-{raw_ts[4:6]}-{raw_ts[6:8]}'

        for ts in art.get('ticker_sentiment', []):
            ticker = ts.get('ticker', '').upper()
            if ticker not in tracked:
                continue
            try:
                rel   = float(ts.get('relevance_score', 0))
                score = float(ts.get('ticker_sentiment_score', 0))
            except (TypeError, ValueError):
                continue
            if rel > RELEVANCE_THRESHOLD:
                records.append((ticker, date_str, score))

    return records


def _aggregate(records: list[tuple[str, str, float]]) -> pd.DataFrame:
    """Group (ticker, date, score) records → daily aggregates."""
    if not records:
        return pd.DataFrame(columns=[
            'ticker', 'date', 'news_count_1d',
            'news_sentiment_1d', 'news_sentiment_std',
        ])
    df = pd.DataFrame(records, columns=['ticker', 'date', 'score'])
    agg = (
        df.groupby(['ticker', 'date'])['score']
        .agg(
            news_count_1d='count',
            news_sentiment_1d='mean',
            news_sentiment_std='std',
        )
        .reset_index()
    )
    agg['news_sentiment_std'] = agg['news_sentiment_std'].fillna(0.0)
    agg['news_count_1d']      = agg['news_count_1d'].astype(int)
    return agg[['ticker', 'date', 'news_count_1d',
                'news_sentiment_1d', 'news_sentiment_std']]


def _append_parquet(new_df: pd.DataFrame, out_path: Path) -> int:
    """Append new rows to parquet, deduplicating by (ticker, date). Returns rows written."""
    if new_df.empty:
        return 0
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df.copy()
    before = len(combined)
    combined = combined.drop_duplicates(subset=['ticker', 'date'], keep='last')
    combined = combined.sort_values(['ticker', 'date']).reset_index(drop=True)
    combined.to_parquet(out_path, index=False)
    written = len(combined) - (before - len(new_df))
    log.info(f'Saved {len(combined)} rows to {out_path} (+{len(new_df)} new)')
    return len(new_df)


# ── Modes ─────────────────────────────────────────────────────────────────────

def daily_mode(dry_run: bool = False, test_one: bool = False) -> None:
    """
    Fetch yesterday's news for all tracked tickers.
    39 tickers / 5 per batch = 8 API calls.
    """
    tickers  = _load_tickers()
    tracked  = set(tickers)
    yesterday = date.today() - timedelta(days=1)
    today     = date.today()
    time_from = _fmt_dt(yesterday)
    time_to   = _fmt_dt(today)

    log.info(f'Daily mode: {yesterday} | {len(tickers)} tickers | dry_run={dry_run}')

    if test_one:
        tickers = ['NVDA']
        log.info('--test-one: restricting to NVDA only')

    if dry_run:
        batches = _batches(tickers)
        log.info(f'DRY RUN — would make {len(batches)} API calls:')
        for i, batch in enumerate(batches, 1):
            log.info(f'  Call {i}: tickers={",".join(batch)} '
                     f'time_from={time_from} time_to={time_to}')
        return

    all_records = []
    for i, batch in enumerate(_batches(tickers), 1):
        articles = _fetch_av(batch, time_from, time_to, sort='LATEST')
        all_records.extend(_parse_articles(articles, tracked))
        if i < len(_batches(tickers)):
            time.sleep(DAILY_SLEEP_SEC)

    result = _aggregate(all_records)
    log.info(f'Parsed {len(all_records)} sentiment data points → '
             f'{len(result)} (ticker, date) rows')

    if test_one:
        print('\n── Test result for NVDA ──')
        print(result.to_string() if not result.empty else '(no data found)')
        return

    _append_parquet(result, LIVE_OUT)


def backfill_mode(start: str, end: str, dry_run: bool = False) -> None:
    """
    Backfill month by month from start to end.
    Progress saved to av_backfill_progress.json for resuming.
    """
    tickers = _load_tickers()
    tracked = set(tickers)

    start_d = date.fromisoformat(start)
    end_d   = date.fromisoformat(end)

    # Build ordered list of (year, month) pairs to process
    months = []
    cur = date(start_d.year, start_d.month, 1)
    while cur < end_d:
        months.append((cur.year, cur.month))
        # Advance to next month
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)

    # Load progress
    progress: dict = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            progress = json.load(f)
    completed_keys = set(progress.get('completed', []))

    log.info(f'Backfill: {start} → {end} | {len(months)} months | '
             f'{len(tickers)} tickers | {len(_batches(tickers))} batches/month')
    log.info(f'Already completed: {len(completed_keys)} months')

    total_calls  = sum(
        len(_batches(tickers))
        for (y, m) in months
        if f'{y}-{m:02d}' not in completed_keys
    )
    log.info(f'Remaining API calls: {total_calls} '
             f'(~{total_calls // 25 + 1} day(s) at 25/day)')

    if dry_run:
        log.info('DRY RUN — no API calls made')
        return

    all_records = []
    call_count  = 0

    for year, month in months:
        month_key  = f'{year}-{month:02d}'
        if month_key in completed_keys:
            log.info(f'Skipping {month_key} (already done)')
            continue

        month_start = date(year, month, 1)
        if month == 12:
            month_end = date(year + 1, 1, 1)
        else:
            month_end = date(year, month + 1, 1)

        time_from = _fmt_dt(month_start)
        time_to   = _fmt_dt(month_end)
        log.info(f'Processing {month_key} ({time_from} → {time_to})')

        month_records = []
        hit_limit = False
        for batch in _batches(tickers):
            if call_count >= MAX_DAILY_CALLS:
                log.info(
                    f'Daily call limit reached ({call_count}/{MAX_DAILY_CALLS}). '
                    f'Save progress and stop — resume tomorrow.'
                )
                hit_limit = True
                break
            articles = _fetch_av(batch, time_from, time_to, sort='EARLIEST')
            month_records.extend(_parse_articles(articles, tracked))
            call_count += 1
            log.info(f'  Calls so far this run: {call_count}')
            time.sleep(BACKFILL_SLEEP_SEC)

        if hit_limit:
            # Flush whatever was fetched for the partial month (not marked complete)
            if month_records:
                all_records.extend(month_records)
            if all_records:
                partial = _aggregate(all_records)
                _append_parquet(partial, BACKFILL_OUT)
                all_records = []
            total_rows = len(pd.read_parquet(BACKFILL_OUT)) if BACKFILL_OUT.exists() else 0
            progress['total_records_so_far'] = total_rows
            PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(progress, f, indent=2)
            return

        all_records.extend(month_records)

        # Flush to parquet and mark complete ONLY when rows were actually fetched
        if all_records:
            partial = _aggregate(all_records)
            _append_parquet(partial, BACKFILL_OUT)
            all_records = []  # reset after flush

            completed_keys.add(month_key)
            total_rows = len(pd.read_parquet(BACKFILL_OUT))
            progress['completed']            = sorted(completed_keys)
            progress['last_completed']       = month_key
            progress['total_records_so_far'] = total_rows
            PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(progress, f, indent=2)
            log.info(f'Progress saved: {month_key} done, {total_rows} rows in parquet')
        else:
            log.warning(
                f'{month_key}: 0 articles fetched — NOT marking complete, will retry. '
                f'Check ALPHAVANTAGE_API_KEY and AV response logs above.'
            )

    log.info(f'Backfill complete. Output: {BACKFILL_OUT}')


def merge_mode() -> None:
    """
    Merge FNSPID (2019-2023), AV backfill (2023-2025), and daily live parquets.
    Deduplicates by (ticker, date), keeps row with highest news_count_1d.
    """
    sources = []
    labels  = []

    if FNSPID_PATH.exists():
        df = pd.read_parquet(FNSPID_PATH)
        sources.append(df)
        labels.append(f'FNSPID ({len(df):,} rows)')
    else:
        log.warning(f'{FNSPID_PATH} not found — skipping')

    if BACKFILL_OUT.exists():
        df = pd.read_parquet(BACKFILL_OUT)
        sources.append(df)
        labels.append(f'AV backfill ({len(df):,} rows)')
    else:
        log.warning(f'{BACKFILL_OUT} not found — run --backfill first')

    if LIVE_OUT.exists():
        df = pd.read_parquet(LIVE_OUT)
        sources.append(df)
        labels.append(f'AV live ({len(df):,} rows)')
    else:
        log.info(f'{LIVE_OUT} not found — skipping (no daily data yet)')

    if not sources:
        log.error('No source files found. Nothing to merge.')
        return

    log.info('Merging: ' + ', '.join(labels))
    combined = pd.concat(sources, ignore_index=True)

    # Keep the row with the highest news_count_1d when duplicates exist
    combined = (
        combined
        .sort_values('news_count_1d', ascending=False)
        .drop_duplicates(subset=['ticker', 'date'], keep='first')
        .sort_values(['ticker', 'date'])
        .reset_index(drop=True)
    )

    MERGED_OUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(MERGED_OUT, index=False)
    log.info(f'Merged → {MERGED_OUT} | {len(combined):,} rows')

    # Coverage stats by year
    combined['year'] = combined['date'].str[:4]
    print('\n── Coverage by year ──')
    for year, grp in combined.groupby('year'):
        n_rows    = len(grp)
        n_tickers = grp['ticker'].nunique()
        mean_sent = grp['news_sentiment_1d'].mean()
        print(f'  {year}: {n_rows:>5} rows  {n_tickers:>2} tickers  '
              f'mean_sentiment={mean_sent:+.4f}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Alpha Vantage news sentiment collector for RSSS'
    )
    parser.add_argument('--dry-run',  action='store_true',
                        help='Print plan without making API calls')
    parser.add_argument('--test-one', action='store_true',
                        help='Fetch only NVDA for yesterday and print result')
    parser.add_argument('--backfill', action='store_true',
                        help='Run historical backfill (requires --start and --end)')
    parser.add_argument('--start',    default='2023-01-01',
                        help='Backfill start date YYYY-MM-DD (default: 2023-01-01)')
    parser.add_argument('--end',
                        default=date.today().isoformat(),
                        help='Backfill end date YYYY-MM-DD (default: today)')
    parser.add_argument('--merge',    action='store_true',
                        help='Merge all news sources into news_features_merged.parquet')
    args = parser.parse_args()

    if args.merge:
        merge_mode()
    elif args.backfill:
        backfill_mode(args.start, args.end, dry_run=args.dry_run)
    else:
        daily_mode(dry_run=args.dry_run, test_one=args.test_one)


if __name__ == '__main__':
    main()
