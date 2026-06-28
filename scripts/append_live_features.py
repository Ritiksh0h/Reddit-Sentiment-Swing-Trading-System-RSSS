"""
Append today's live feature vectors to the live feature store.

Called after each daily_run to persist computed features for
future retraining. Saves (ticker, date, 16 v2 features, close_price)
so that target returns can be computed later when t+5 prices are available.

Usage:
    python scripts/append_live_features.py
    python scripts/append_live_features.py --date 2026-06-18
    python scripts/append_live_features.py --fill-targets-only

Output:
    data/features/features_live_v2.parquet (appended daily)
    data/features/features_target_pending.json (awaiting t+5 prices)
"""
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, '.')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

LIVE_FEATURES_PATH   = Path('data/features/features_live_v2.parquet')
PENDING_TARGETS_PATH = Path('data/processed/features_target_pending.json')
TARGET_HORIZON       = 5   # trading days

FEATURE_COLS = [
    'post_count_1d',       'abnormal_attention_1d',
    'total_comments_1d',   'vader_sentiment_1d',
    'sentiment_extremity', 'sentiment_accel',
    'volume',              'relative_volume',
    'returns_1d',          'returns_20d',
    'rsi_14',              'news_sentiment_1d',
    'vix_percentile',      'vix_x_volume',
    'spy_above_200ma',     'regime_score',
]


def load_today_feature_vectors(today: str) -> list[dict]:
    """
    Load today's feature vectors for all density-gate-passing tickers.

    Primary source:  logs/all_signals.jsonl  (action == 'SIGNAL')
      Written by daily_run.py for every ticker scored by XGBoost,
      regardless of whether a trade fired. Captures BULLISH + NEUTRAL
      + BEARISH rows — all valid training examples.

    Fallback source: logs/paper_trades.jsonl (action == 'OPEN')
      Used when all_signals.jsonl doesn't exist (backward compat with
      runs before this fix was deployed).

    Args:
        today: YYYY-MM-DD — only records matching this date are returned

    Returns:
        list of dicts with ticker, date, close, 16 feature values,
        predicted_return_5d, signal, confidence.
    """
    all_signals_path  = Path('logs/all_signals.jsonl')
    paper_trades_path = Path('logs/paper_trades.jsonl')

    if all_signals_path.exists():
        log_path      = all_signals_path
        action_filter = 'SIGNAL'
    elif paper_trades_path.exists():
        log_path      = paper_trades_path
        action_filter = 'OPEN'
        logger.info('all_signals.jsonl not found — falling back to paper_trades.jsonl OPEN records')
    else:
        return []

    rows = []
    with open(log_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get('date') != today:
                continue
            if record.get('action') != action_filter:
                continue

            ticker = record.get('ticker')
            if not ticker:
                continue

            # all_signals.jsonl stores feature_vector as a nested dict;
            # paper_trades.jsonl uses the same key (feature_vector).
            fv = (record.get('feature_vector')
                  or record.get('feature_vector_14')
                  or record.get('feature_vector_11')
                  or {})

            if not fv:
                logger.warning(f'no_feature_vector ticker={ticker} date={today}')
                continue

            # close_price for SIGNAL records; fill_price for OPEN records
            close = record.get('close_price') or record.get('fill_price') or 0.0

            row = {
                'ticker':              ticker,
                'date':                today,
                'close':               float(close),
                'predicted_return_5d': float(record.get('predicted_return_5d', 0.0)),
                'signal':              record.get('signal', 'NEUTRAL'),
                'confidence':          float(record.get('confidence', 0.0)),
            }

            for col in FEATURE_COLS:
                row[col] = float(fv.get(col, 0.0))

            rows.append(row)
            logger.info(
                f'feature_row ticker={ticker} signal={row["signal"]} '
                f'post_count={row["post_count_1d"]:.0f} '
                f'news={row["news_sentiment_1d"]:.3f} '
                f'vix_pct={row["vix_percentile"]:.3f} '
                f'regime={row["regime_score"]:.3f}'
            )

    return rows


def fill_pending_targets() -> int:
    """
    Fill target_return_5d for pending rows where t+5 price is now available.

    Reads data/processed/features_target_pending.json. A row becomes eligible
    after 7+ calendar days (ensures 5 trading days have passed). Fetches the
    t+5 close price via yfinance and computes (close_t5 - close_t0) / close_t0.
    Filled rows are appended to data/features/features_live_v2.parquet; rows
    still awaiting prices are written back to the pending file.

    Returns:
        number of rows filled in this run
    """
    if not PENDING_TARGETS_PATH.exists():
        return 0

    with open(PENDING_TARGETS_PATH) as f:
        pending = json.load(f)

    still_pending = []
    filled        = 0
    new_rows      = []
    today         = date.today()

    for entry in pending:
        entry_date = date.fromisoformat(entry['date'])
        # Need 7+ calendar days for 5 trading days to pass
        if (today - entry_date).days < 7:
            still_pending.append(entry)
            continue

        ticker = entry['ticker']
        try:
            mkt = yf.download(
                ticker,
                start=str(entry_date),
                auto_adjust=True,
                progress=False,
            )
            if isinstance(mkt.columns, pd.MultiIndex):
                mkt.columns = mkt.columns.get_level_values(0)

            if len(mkt) < 5:
                still_pending.append(entry)
                continue

            close_t0 = float(mkt['Close'].iloc[0])
            close_t5 = float(mkt['Close'].iloc[4])
            target   = (close_t5 - close_t0) / close_t0

            entry['target_return_5d'] = round(target, 6)
            entry['close_t5']         = round(close_t5, 4)
            new_rows.append(entry)
            filled += 1
            logger.info(
                f'target_filled ticker={ticker} date={entry["date"]} '
                f'target_5d={target*100:+.2f}%'
            )
        except Exception as e:
            logger.warning(f'target_fetch_failed ticker={ticker}: {e}')
            still_pending.append(entry)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if LIVE_FEATURES_PATH.exists():
            existing = pd.read_parquet(LIVE_FEATURES_PATH)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=['ticker', 'date'], keep='last'
            )
        else:
            combined = new_df
        LIVE_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(LIVE_FEATURES_PATH, index=False)
        logger.info(
            f'live_features_updated rows={len(combined)} '
            f'new={len(new_rows)} path={LIVE_FEATURES_PATH}'
        )

    with open(PENDING_TARGETS_PATH, 'w') as f:
        json.dump(still_pending, f, indent=2)

    return filled


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default=None,
                        help='Override date YYYY-MM-DD')
    parser.add_argument('--fill-targets-only', action='store_true',
                        help='Only fill pending targets, skip feature saving')
    args = parser.parse_args()

    today = args.date or datetime.now(timezone.utc).strftime('%Y-%m-%d')
    logger.info(f'=== Append Live Features date={today} ===')

    filled = fill_pending_targets()
    logger.info(f'Targets filled: {filled}')

    if args.fill_targets_only:
        return

    rows = load_today_feature_vectors(today)
    if not rows:
        logger.info(f'No qualifying signals found for {today} — nothing to save')
        return

    logger.info(f'Feature rows to save: {len(rows)}')

    pending = []
    if PENDING_TARGETS_PATH.exists():
        with open(PENDING_TARGETS_PATH) as f:
            pending = json.load(f)

    existing_keys = {(p['ticker'], p['date']) for p in pending}
    added = 0
    for row in rows:
        key = (row['ticker'], row['date'])
        if key not in existing_keys:
            pending.append(row)
            existing_keys.add(key)
            added += 1

    PENDING_TARGETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PENDING_TARGETS_PATH, 'w') as f:
        json.dump(pending, f, indent=2)

    logger.info(
        f'Pending targets: {len(pending)} total, {added} new added today'
    )

    if LIVE_FEATURES_PATH.exists():
        df       = pd.read_parquet(LIVE_FEATURES_PATH)
        news_ok  = (df['news_sentiment_1d'] != 0.0).mean() * 100
        vix_ok   = (df['vix_percentile']    != 0.0).mean() * 100
        logger.info(
            f'Live feature store: {len(df)} rows, '
            f'news_ok={news_ok:.1f}%, '
            f'vix_ok={vix_ok:.1f}%'
        )


if __name__ == '__main__':
    main()
