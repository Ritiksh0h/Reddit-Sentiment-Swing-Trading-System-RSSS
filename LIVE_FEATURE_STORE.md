# Claude Code — Live Feature Store + Wire News/ST into Signal Generator
# Reddit Sentiment Swing Trading System (RSSS)

---

## Context

Two problems to fix in one session:

Problem 1: News and StockTwits data are fetched daily but never
used in signal generation. signal_generator.py only uses Reddit
features — news_sentiment_1d, st_sentiment_1d, st_bull_pct are
always 0.0 in live signals even though real data is available.

Problem 2: Live feature vectors are never saved to disk.
Every day the system computes 14 features per qualifying ticker
but throws them away after generating signals. There is no way
to retrain on 2026 live data because it's not stored anywhere.

Both fixes together mean: live data is real, complete, and saved.

---

## Session Start

```bash
git pull origin main
source .venv/bin/activate

# Confirm current state
grep -n "news_sentiment\|st_sentiment\|st_bull" \
    portfolio/signal_generator.py | head -10
# Expected: nothing — these features are not wired in yet

# Confirm news and ST are fetched but not passed to signals
grep -n "news_data\|stocktwits_data\|st_data\|news_sentiment" \
    scripts/daily_run_live.py | head -10
# Expected: fetched but not passed to generate_signals

# Check feature store
python3 -c "
import pandas as pd
df = pd.read_parquet('data/features/features_complete.parquet')
print('Columns:', df.columns.tolist())
print('Shape:', df.shape)
"
```

---

## Task 1 — Wire News + ST into Signal Generator

### Step 1a — Read these files completely before touching anything

```bash
cat portfolio/signal_generator.py
cat scripts/daily_run_live.py
cat scripts/daily_run.py
```

### Step 1b — Update compute_features_live() in signal_generator.py

Find the `compute_features_live()` function. Currently it only
accepts Reddit features. Add news and ST parameters:

```python
def compute_features_live(
    ticker: str,
    market_data: pd.DataFrame,
    post_count_1d: float,
    mention_growth_1d: float,
    mention_growth_7d: float,
    news_sentiment_1d: float = 0.0,   # ← ADD
    st_sentiment_1d:   float = 0.0,   # ← ADD
    st_bull_pct:       float = 0.5,   # ← ADD
) -> dict | None:
```

In the return dict at the end of the function, add the three
new fields:

```python
    return {
        'returns_1d':        returns_1d,
        'returns_5d':        returns_5d,
        'returns_20d':       returns_20d,
        'rsi_14':            rsi,
        'atr_14':            atr_14,
        'relative_volume':   relative_vol,
        'dist_from_20ma':    dist_from_20ma,
        'dist_from_50ma':    dist_from_50ma,
        'post_count_1d':     float(post_count_1d),
        'mention_growth_1d': float(mention_growth_1d),
        'mention_growth_7d': float(mention_growth_7d),
        'news_sentiment_1d': float(news_sentiment_1d),   # ← ADD
        'st_sentiment_1d':   float(st_sentiment_1d),     # ← ADD
        'st_bull_pct':       float(st_bull_pct),         # ← ADD
    }
```

### Step 1c — Update generate_signals() to accept news/ST data

The function signature currently is:
```python
def generate_signals(
    reddit_counts: dict,
    model: xgb.XGBRegressor,
    today: str = None,
) -> list:
```

Change to:
```python
def generate_signals(
    reddit_counts:   dict,
    model:           xgb.XGBRegressor,
    today:           str  = None,
    news_data:       dict = None,    # ← ADD {ticker: {news_sentiment_1d}}
    stocktwits_data: dict = None,    # ← ADD {ticker: {st_sentiment_1d, st_bull_pct}}
) -> list:
```

Inside generate_signals(), find the call to compute_features_live()
and update it to pass the news/ST values:

```python
        # Get news sentiment for this ticker
        news_sent = 0.0
        if news_data and ticker in news_data:
            news_sent = float(news_data[ticker].get('news_sentiment_1d', 0.0))

        # Get StockTwits sentiment for this ticker
        st_sent  = 0.0
        st_bull  = 0.5
        if stocktwits_data and ticker in stocktwits_data:
            st_sent = float(stocktwits_data[ticker].get('st_sentiment_1d', 0.0))
            st_bull = float(stocktwits_data[ticker].get('st_bull_pct', 0.5))

        features = compute_features_live(
            ticker=ticker,
            market_data=mkt,
            post_count_1d=post_count,
            mention_growth_1d=reddit_data.get('mention_growth_1d', 0.0),
            mention_growth_7d=reddit_data.get('mention_growth_7d', 0.0),
            news_sentiment_1d=news_sent,    # ← ADD
            st_sentiment_1d=st_sent,        # ← ADD
            st_bull_pct=st_bull,            # ← ADD
        )
```

Also add logging to show which features are real vs default:

```python
        logger.debug(
            f'features ticker={ticker} '
            f'news={news_sent:.3f} st={st_sent:.3f} '
            f'st_bull={st_bull:.3f}'
        )
```

### Step 1d — Update daily_run_live.py to pass news/ST to run()

Find where `run()` is called in daily_run_live.py:

```python
    summary = run(reddit_counts=reddit_counts, today=today)
```

Change to:

```python
    summary = run(
        reddit_counts=reddit_counts,
        today=today,
        news_data=news_data,           # ← ADD
        stocktwits_data=stocktwits_data,  # ← ADD
    )
```

### Step 1e — Update daily_run.py run() signature

Find the `run()` function signature in scripts/daily_run.py:

```python
def run(reddit_counts: dict, today: str = None) -> dict:
```

Change to:

```python
def run(
    reddit_counts:   dict,
    today:           str  = None,
    news_data:       dict = None,
    stocktwits_data: dict = None,
) -> dict:
```

Find where `generate_signals()` is called inside run():

```python
        signals = generate_signals(reddit_counts, model, today)
```

Change to:

```python
        signals = generate_signals(
            reddit_counts=reddit_counts,
            model=model,
            today=today,
            news_data=news_data,
            stocktwits_data=stocktwits_data,
        )
```

### Step 1f — Also wire news/ST in dry-run mode

In daily_run_live.py, find the dry-run block:

```python
        signals = generate_signals(reddit_counts, model, today)
```

Change to:

```python
        signals = generate_signals(
            reddit_counts=reddit_counts,
            model=model,
            today=today,
            news_data=news_data,
            stocktwits_data=stocktwits_data,
        )
```

### Step 1g — Verify news_data and stocktwits_data exist in daily_run_live.py

Check that `news_data` and `stocktwits_data` are defined before
the `run()` call. They should already be fetched. If the variable
names are different, use the actual variable names from the file.

Look for lines like:
```python
    news_data       = ...  # dict of {ticker: {news_sentiment_1d, ...}}
    stocktwits_data = ...  # dict of {ticker: {st_sentiment_1d, st_bull_pct, ...}}
```

If they are named differently (e.g., `av_news`, `st_scores`),
use those names when passing to run().

---

## Task 2 — Save Live Features to Parquet

Create `scripts/append_live_features.py`

```python
"""
Append today's live feature vectors to the live feature store.

Called after each daily_run to persist computed features for
future retraining. Saves (ticker, date, 14 features, close_price)
so that target returns can be computed later when t+5 prices are available.

Usage:
    python scripts/append_live_features.py
    python scripts/append_live_features.py --date 2026-06-18

Output:
    data/features/features_live_2026.parquet (appended daily)
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

LIVE_FEATURES_PATH   = Path('data/features/features_live_2026.parquet')
PENDING_TARGETS_PATH = Path('data/processed/features_target_pending.json')
TARGET_HORIZON       = 5   # trading days
INITIAL_CAPITAL      = 10000.0

# ── Feature columns (must match ARCH features) ─────────────────────────
FEATURE_COLS = [
    'returns_1d', 'returns_5d', 'returns_20d', 'rsi_14', 'atr_14',
    'relative_volume', 'dist_from_20ma', 'dist_from_50ma',
    'post_count_1d', 'mention_growth_1d', 'mention_growth_7d',
    'news_sentiment_1d', 'st_sentiment_1d', 'st_bull_pct',
]


def load_today_feature_vectors(today: str) -> list[dict]:
    """
    Load today's feature vectors from paper_trades.jsonl (OPEN records)
    and from execution log if available.

    Falls back to signal generator output if needed.
    """
    rows = []
    log_path = Path('logs/paper_trades.jsonl')
    if not log_path.exists():
        return rows

    with open(log_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Only process today's OPEN signals
            if record.get('date') != today:
                continue
            if record.get('action') != 'OPEN':
                continue

            ticker = record.get('ticker')
            if not ticker:
                continue

            # Extract feature vector
            fv = (record.get('feature_vector_11')
                  or record.get('feature_vector_14')
                  or record.get('feature_vector')
                  or {})

            if not fv:
                logger.warning(f'no_feature_vector ticker={ticker} date={today}')
                continue

            row = {
                'ticker':             ticker,
                'date':               today,
                'close':              record.get('entry_price', 0.0),
                'predicted_return_5d': record.get('predicted_return', 0.0),
                'signal':             record.get('signal', 'NEUTRAL'),
                'confidence':         record.get('confidence', 0.0),
            }

            # Add all feature columns (default 0.0 if missing)
            for col in FEATURE_COLS:
                row[col] = float(fv.get(col, 0.0))

            rows.append(row)
            logger.info(
                f'feature_row ticker={ticker} '
                f'news={row["news_sentiment_1d"]:.3f} '
                f'st={row["st_sentiment_1d"]:.3f} '
                f'post_count={row["post_count_1d"]:.0f}'
            )

    return rows


def fill_pending_targets() -> int:
    """
    For each pending feature row, check if t+5 price is now available.
    If yes: compute target_return_5d and move to live features parquet.

    Returns number of targets filled.
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

        # Fetch price at entry + 5 trading days
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

    # Append filled rows to live features parquet
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if LIVE_FEATURES_PATH.exists():
            existing = pd.read_parquet(LIVE_FEATURES_PATH)
            # Deduplicate by (ticker, date)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=['ticker', 'date'], keep='last'
            )
        else:
            combined = new_df
        combined.to_parquet(LIVE_FEATURES_PATH, index=False)
        logger.info(
            f'live_features_updated rows={len(combined)} '
            f'new={len(new_rows)} path={LIVE_FEATURES_PATH}'
        )

    # Save remaining pending entries
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

    # Step 1: Fill any pending targets from 5+ days ago
    filled = fill_pending_targets()
    logger.info(f'Targets filled: {filled}')

    if args.fill_targets_only:
        return

    # Step 2: Load today's feature vectors from trade log
    rows = load_today_feature_vectors(today)
    if not rows:
        logger.info(f'No OPEN signals found for {today} — nothing to save')
        return

    logger.info(f'Feature rows to save: {len(rows)}')

    # Step 3: Save to pending targets (need t+5 prices later)
    pending = []
    if PENDING_TARGETS_PATH.exists():
        with open(PENDING_TARGETS_PATH) as f:
            pending = json.load(f)

    # Add new rows (avoid duplicates by ticker+date)
    existing_keys = {(p['ticker'], p['date']) for p in pending}
    added = 0
    for row in rows:
        key = (row['ticker'], row['date'])
        if key not in existing_keys:
            pending.append(row)
            existing_keys.add(key)
            added += 1

    with open(PENDING_TARGETS_PATH, 'w') as f:
        json.dump(pending, f, indent=2)

    logger.info(
        f'Pending targets: {len(pending)} total, {added} new added today'
    )

    # Step 4: Print coverage stats
    if LIVE_FEATURES_PATH.exists():
        df = pd.read_parquet(LIVE_FEATURES_PATH)
        news_real = (df['news_sentiment_1d'] != 0.0).mean() * 100
        st_real   = (df['st_sentiment_1d']   != 0.0).mean() * 100
        logger.info(
            f'Live feature store: {len(df)} rows, '
            f'news_real={news_real:.1f}%, '
            f'st_real={st_real:.1f}%'
        )


if __name__ == '__main__':
    main()
```

---

## Task 3 — Wire append_live_features.py into daily_run_live.py

At the end of `scripts/daily_run_live.py`, after the `run()` call
and before `print(json.dumps(summary, indent=2))`, add:

```python
    # ── Save live feature vectors for future retraining ───────────────────
    if not summary.get('skipped') and summary.get('actions'):
        try:
            from scripts.append_live_features import main as append_features
            # Run feature append in same process
            import subprocess
            result = subprocess.run(
                [sys.executable, 'scripts/append_live_features.py',
                 '--date', today],
                capture_output=True, text=True, cwd='.'
            )
            if result.returncode == 0:
                logger.info('live_features_appended')
            else:
                logger.warning(
                    f'live_features_append_failed: {result.stderr[:200]}'
                )
        except Exception as e:
            logger.warning(f'live_features_append_error: {e}')
```

Also add target-filling to the daily run (runs even on HOLD_CASH days):

```python
    # ── Fill pending targets regardless of signal outcome ─────────────────
    try:
        result = subprocess.run(
            [sys.executable, 'scripts/append_live_features.py',
             '--fill-targets-only'],
            capture_output=True, text=True, cwd='.'
        )
        if result.returncode != 0:
            logger.warning(
                f'target_fill_failed: {result.stderr[:200]}'
            )
    except Exception as e:
        logger.warning(f'target_fill_error: {e}')
```

---

## Task 4 — Add merge script to rebuild features_complete.parquet

Add a new function to `scripts/merge_external_features.py` (or
create `scripts/rebuild_feature_store.py`) that merges:

```python
def merge_live_features():
    """
    Merge live 2026 features into features_complete.parquet.
    Run manually every 30 days or when retraining.
    """
    import pandas as pd
    from pathlib import Path

    base_path = Path('data/features/features_complete.parquet')
    live_path = Path('data/features/features_live_2026.parquet')

    if not live_path.exists():
        print('No live features yet — run daily_run_live.py first')
        return

    live = pd.read_parquet(live_path)
    # Only rows with target_return_5d filled
    live_complete = live[live['target_return_5d'].notna()].copy()

    if len(live_complete) == 0:
        print('No completed live features yet — need 5+ trading days')
        return

    base = pd.read_parquet(base_path)
    combined = pd.concat([base, live_complete], ignore_index=True)
    combined = combined.drop_duplicates(subset=['ticker', 'date'], keep='last')
    combined = combined.sort_values(['ticker', 'date'])
    combined.to_parquet(base_path, index=False)

    print(f'Merged {len(live_complete)} live rows into feature store')
    print(f'Total: {len(combined)} rows')
    print(f'Years: {sorted(pd.to_datetime(combined["date"]).dt.year.unique())}')
```

---

## Task 5 — Run Tests and Dry Run

```bash
# Run all tests
pytest tests/ -v --tb=short

# Dry run with full output
python scripts/daily_run_live.py --dry-run 2>&1 | \
    grep -v "httpx\|HF_TOKEN\|Loading\|huggingface\|Redirect"
```

Look for these lines in dry run output confirming news/ST are wired:
```
INFO  features ticker=NVDA news=0.432 st=0.612 st_bull=0.71
```

If news/ST are 0.0 for all tickers — the wiring worked but today
has no qualifying tickers (expected at low-activity hours).

---

## Build Order

```bash
# 1. Read all three files completely (required)
cat portfolio/signal_generator.py
cat scripts/daily_run_live.py
cat scripts/daily_run.py

# 2. Update signal_generator.py (Task 1b, 1c)
# 3. Update daily_run.py (Task 1e)
# 4. Update daily_run_live.py (Task 1d, 1f, Task 3)
# 5. Create scripts/append_live_features.py (Task 2)
# 6. Add merge function (Task 4)
# 7. Run tests
pytest tests/ -v --tb=short
# 8. Dry run
python scripts/daily_run_live.py --dry-run 2>&1 | grep -v "httpx\|HF_TOKEN\|Loading\|huggingface"
# 9. Push
bash push.sh "[data] wire news+ST into signals + live feature store"
```

---

## Expected Final State

```
After today:
  News and ST sentiment used in live signal predictions
  feature_vector in paper_trades.jsonl has real news/ST values
  append_live_features.py saves feature rows daily
  features_target_pending.json accumulates rows awaiting t+5 prices

After 7+ days:
  First completed rows appear in features_live_2026.parquet
  IC monitoring can use real feature values

After 60-90 days:
  ~2,000-3,000 live rows with real news/ST sentiment
  Merge into features_complete.parquet
  Retrain with genuinely complete features for 2026

Every Monday:
  append_live_features.py --fill-targets-only fills completed rows
  automatically via daily_run_live.py
```

---

## Hard Rules

- NEVER change the model or thresholds — only the data pipeline
- NEVER break backward compatibility of daily_run.py run() —
  news_data and stocktwits_data must default to None
- ALWAYS default news_sentiment_1d=0.0 and st_sentiment_1d=0.0
  when data is unavailable — never crash on missing source
- NEVER save feature rows without a ticker and date
- The append script runs AFTER trading — never before
- features_target_pending.json is append-only — never delete entries
  until target_return_5d is confirmed filled
- 20+ tests must pass before pushing

---

*Live Feature Store — June 2026*
*Wire news+ST into signals | Save daily features | Fill t+5 targets*
*After 60 days: real 2026 retraining data available*
