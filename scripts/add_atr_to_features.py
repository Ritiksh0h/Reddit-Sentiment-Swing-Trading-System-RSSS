#!/usr/bin/env python3
"""
add_atr_to_features.py — Append atr_14 and atr_pct to features_v2.parquet.

Fetches OHLC from yfinance (2018-01-01 → 2026-07-01), computes Wilder 14-day ATR
for every (ticker, date) row, merges into the source parquet, and saves to
features_v2_with_atr.parquet.

Usage:
    python scripts/add_atr_to_features.py
    python scripts/add_atr_to_features.py \
        --in  data/features/features_v2.parquet \
        --out data/features/features_v2_with_atr.parquet
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, '.')

INPUT_DEFAULT  = 'data/features/features_v2.parquet'
OUTPUT_DEFAULT = 'data/features/features_v2_with_atr.parquet'
OHLC_START     = '2018-01-01'
OHLC_END       = '2026-07-01'
ATR_PERIOD     = 14


def wilder_atr(
    high: pd.Series,
    low:  pd.Series,
    close: pd.Series,
    period: int = ATR_PERIOD,
) -> pd.Series:
    """Wilder 14-day ATR using EWM (alpha=1/period, adjust=False)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def fetch_atr_for_tickers(tickers: list) -> pd.DataFrame:
    """Download OHLC and compute ATR for all tickers.

    Returns a DataFrame with columns: ticker, date, atr_14, atr_pct.
    """
    print(f'Downloading OHLC for {len(tickers)} tickers '
          f'({OHLC_START} → {OHLC_END})...')
    raw = yf.download(
        tickers, start=OHLC_START, end=OHLC_END,
        auto_adjust=True, progress=True, threads=True,
    )

    results = []
    failed  = []

    is_multi = isinstance(raw.columns, pd.MultiIndex)

    for ticker in tickers:
        try:
            if is_multi:
                h = raw['High'][ticker].dropna()
                l = raw['Low'][ticker].dropna()
                c = raw['Close'][ticker].dropna()
            else:
                h = raw['High'].dropna()
                l = raw['Low'].dropna()
                c = raw['Close'].dropna()

            if len(c) < ATR_PERIOD + 5:
                failed.append(ticker)
                continue

            idx   = c.index.intersection(h.index).intersection(l.index)
            h, l, c = h.loc[idx], l.loc[idx], c.loc[idx]

            atr = wilder_atr(h, l, c, ATR_PERIOD)

            df_t = pd.DataFrame({
                'ticker':   ticker,
                'date':     atr.index.strftime('%Y-%m-%d'),
                'atr_14':   atr.values,
                '_close':   c.values,
            })
            df_t['atr_pct'] = df_t['atr_14'] / df_t['_close']
            df_t = df_t.drop(columns='_close')
            df_t = df_t.dropna(subset=['atr_14'])
            results.append(df_t)

        except Exception as e:
            print(f'  WARNING: ATR failed for {ticker}: {e}')
            failed.append(ticker)

    if failed:
        print(f'ATR unavailable for {len(failed)} tickers: {sorted(failed)}')

    if not results:
        raise RuntimeError('No ATR data computed — check yfinance connection.')

    return pd.concat(results, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description='Append atr_14 and atr_pct to features_v2.parquet')
    parser.add_argument('--in',  dest='input',  default=INPUT_DEFAULT,
                        help='Source parquet path')
    parser.add_argument('--out', dest='output', default=OUTPUT_DEFAULT,
                        help='Output parquet path')
    args = parser.parse_args()

    # ── Load feature store ────────────────────────────────────────────────────
    print(f'Loading {args.input}...')
    df = pd.read_parquet(args.input)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    print(f'  {len(df):,} rows  {df["ticker"].nunique()} tickers  '
          f'{df.shape[1]} cols')

    tickers = sorted(df['ticker'].unique().tolist())

    # ── Fetch + compute ATR ────────────────────────────────────────────────────
    atr_df = fetch_atr_for_tickers(tickers)
    atr_df['date'] = atr_df['date'].astype(str)
    print(f'ATR frame: {len(atr_df):,} rows  {atr_df["ticker"].nunique()} tickers')

    # ── Merge ──────────────────────────────────────────────────────────────────
    df = df.drop(columns=[c for c in ('atr_14', 'atr_pct') if c in df.columns])
    merged = df.merge(
        atr_df[['ticker', 'date', 'atr_14', 'atr_pct']],
        on=['ticker', 'date'],
        how='left',
    )

    # ── Coverage report ────────────────────────────────────────────────────────
    n_total  = len(merged)
    n_filled = merged['atr_14'].notna().sum()
    coverage = n_filled / n_total * 100
    print(f'\nATR coverage: {n_filled:,}/{n_total:,}  ({coverage:.1f}%)')

    if coverage < 99.0:
        missing = merged[merged['atr_14'].isna()].groupby('ticker').size()
        print('Tickers with missing ATR rows:')
        print(missing.to_string())

    # ── Sanity: atr_pct distribution ──────────────────────────────────────────
    pct_stats = merged['atr_pct'].describe(percentiles=[0.01, 0.05, 0.95, 0.99])
    print('\natr_pct distribution:')
    print(pct_stats.to_string())

    outliers = merged[merged['atr_pct'] > 0.20]
    if not outliers.empty:
        print(f'\nHigh-ATR rows (atr_pct > 20%): {len(outliers)}')
        print(outliers.groupby('ticker')['atr_pct']
              .max().sort_values(ascending=False).head(10).to_string())

    print('\nPer-ticker atr_pct summary (reference tickers):')
    for t in ['MU', 'NVDA', 'AAPL', 'TSLA', 'GME']:
        subset = merged[merged['ticker'] == t]['atr_pct']
        if not subset.empty:
            print(f'  {t:<6}: mean={subset.mean():.4f}  '
                  f'max={subset.max():.4f}  '
                  f'p95={subset.quantile(0.95):.4f}')

    # ── Save ───────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    print(f'\nSaved → {out_path}  ({len(merged):,} rows, {merged.shape[1]} cols)')

    # ── Auto-add to .gitignore ─────────────────────────────────────────────────
    gitignore = Path('.gitignore')
    if gitignore.exists():
        content = gitignore.read_text()
        entry   = 'data/features/features_v2_with_atr.parquet'
        if entry not in content:
            with gitignore.open('a') as gf:
                gf.write('\n# ATR-augmented feature store (generated — do not commit)\n')
                gf.write(entry + '\n')
            print(f'Added to .gitignore: {entry}')


if __name__ == '__main__':
    main()
