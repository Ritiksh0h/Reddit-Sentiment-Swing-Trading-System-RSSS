"""
scripts/merge_external_features.py

Merges news (FNSPID) and StockTwits features into the feature store.
Run this after both Colab notebooks complete.

Input files:
  data/features/features_full.parquet            ← rebuilt feature store (Reddit + market)
  data/processed/news_features_2019_2023.parquet ← FNSPID FinBERT output
  data/processed/stocktwits_features_2019_2022.parquet ← StockTwits archive

Output:
  data/features/features_complete.parquet        ← all 14 features, ready to train

Usage:
  python scripts/merge_external_features.py
  python scripts/merge_external_features.py --verify-only
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
FEATURE_STORE  = 'data/features/features_full.parquet'
NEWS_FEATURES  = 'data/processed/news_features_2019_2023.parquet'
ST_FEATURES    = 'data/processed/stocktwits_features_2019_2022.parquet'
OUTPUT_PATH    = 'data/features/features_complete.parquet'


def check_inputs():
    """Verify all input files exist and have correct columns."""
    errors = []

    if not Path(FEATURE_STORE).exists():
        errors.append(f'Missing: {FEATURE_STORE} — run FEATURE_STORE_REBUILD_RETRAIN.md first')

    if not Path(NEWS_FEATURES).exists():
        errors.append(f'Missing: {NEWS_FEATURES} — run fnspid_news_processing.ipynb first')
    else:
        df = pd.read_parquet(NEWS_FEATURES, columns=['ticker','date'])
        logger.info(f'News features: {len(df):,} rows, '
                    f'{df["ticker"].nunique()} tickers, '
                    f'dates {df["date"].min()} to {df["date"].max()}')

    if not Path(ST_FEATURES).exists():
        errors.append(f'Missing: {ST_FEATURES} — run stocktwits_archive_processing.ipynb first')
    else:
        df = pd.read_parquet(ST_FEATURES, columns=['ticker','date'])
        if len(df) > 0:
            logger.info(f'StockTwits features: {len(df):,} rows, '
                        f'{df["ticker"].nunique()} tickers, '
                        f'dates {df["date"].min()} to {df["date"].max()}')
        else:
            logger.warning('StockTwits features file is empty (placeholder mode)')

    if errors:
        for e in errors:
            logger.error(e)
        raise FileNotFoundError('\n'.join(errors))


def merge_features():
    """Merge all three sources into one complete feature store."""

    # ── 1. Load feature store ──────────────────────────────────────────────
    logger.info(f'Loading feature store: {FEATURE_STORE}')
    df = pd.read_parquet(FEATURE_STORE)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    logger.info(f'Feature store: {len(df):,} rows, {df["ticker"].nunique()} tickers')

    # Check current columns
    has_news = 'news_sentiment_1d' in df.columns
    has_st   = 'st_sentiment_1d' in df.columns
    logger.info(f'Current news features: {"present" if has_news else "missing (will add)"}')
    logger.info(f'Current ST features:   {"present" if has_st else "missing (will add)"}')

    # ── 2. Merge news features ─────────────────────────────────────────────
    logger.info(f'Loading news features: {NEWS_FEATURES}')
    df_news = pd.read_parquet(NEWS_FEATURES)
    df_news['date'] = pd.to_datetime(df_news['date']).dt.strftime('%Y-%m-%d')

    # Rename columns to match architecture spec
    rename_news = {}
    if 'news_count_1d' not in df_news.columns and 'count' in df_news.columns:
        rename_news['count'] = 'news_count_1d'
    if 'news_sentiment_1d' not in df_news.columns and 'sentiment' in df_news.columns:
        rename_news['sentiment'] = 'news_sentiment_1d'
    if rename_news:
        df_news = df_news.rename(columns=rename_news)

    news_cols = [c for c in ['news_count_1d','news_sentiment_1d','news_sentiment_std']
                 if c in df_news.columns]
    df_news   = df_news[['ticker','date'] + news_cols]

    before = len(df)
    df = df.merge(df_news, on=['ticker','date'], how='left', suffixes=('','_news'))
    logger.info(f'After news merge: {len(df):,} rows (was {before:,})')

    # Fill missing news values (days with no news coverage)
    for col in news_cols:
        if col in df.columns:
            null_count = df[col].isna().sum()
            df[col] = df[col].fillna(0.0)
            logger.info(f'  {col}: {null_count:,} NaN rows filled with 0.0')

    # ── 3. Merge StockTwits features ───────────────────────────────────────
    logger.info(f'Loading StockTwits features: {ST_FEATURES}')
    df_st = pd.read_parquet(ST_FEATURES)

    if len(df_st) == 0:
        logger.warning('StockTwits file is empty (placeholder mode)')
        logger.warning('Adding zero-padded st_sentiment_1d and st_bull_pct')
        df['st_count_1d']     = 0
        df['st_sentiment_1d'] = 0.0
        df['st_bull_pct']     = 0.5
    else:
        df_st['date'] = pd.to_datetime(df_st['date']).dt.strftime('%Y-%m-%d')

        st_cols = [c for c in ['st_count_1d','st_bull_count','st_bear_count',
                                'st_bull_pct','st_sentiment_1d']
                   if c in df_st.columns]
        df_st = df_st[['ticker','date'] + st_cols]

        df = df.merge(df_st, on=['ticker','date'], how='left', suffixes=('','_st'))
        logger.info(f'After StockTwits merge: {len(df):,} rows')

        # Fill missing ST values
        st_defaults = {
            'st_count_1d':     0,
            'st_bull_count':   0,
            'st_bear_count':   0,
            'st_bull_pct':     0.5,
            'st_sentiment_1d': 0.0,
        }
        for col, default in st_defaults.items():
            if col in df.columns:
                null_count = df[col].isna().sum()
                df[col] = df[col].fillna(default)
                if null_count > 0:
                    logger.info(f'  {col}: {null_count:,} NaN rows filled with {default}')

    # ── 4. Verify all 14 features present ─────────────────────────────────
    REQUIRED_FEATURES = [
        'returns_1d','returns_5d','returns_20d','rsi_14','atr_14',
        'relative_volume','dist_from_20ma','dist_from_50ma',
        'post_count_1d','mention_growth_1d','mention_growth_7d',
        'news_sentiment_1d','st_sentiment_1d','st_bull_pct',
    ]
    missing = [f for f in REQUIRED_FEATURES if f not in df.columns]
    if missing:
        logger.error(f'Missing features after merge: {missing}')
        raise ValueError(f'Missing features: {missing}')

    logger.info(f'All 14 features present ✓')

    # ── 5. Coverage report ─────────────────────────────────────────────────
    df['year'] = pd.to_datetime(df['date']).dt.year

    logger.info('\n=== FEATURE COVERAGE BY YEAR ===')
    for year in sorted(df['year'].unique()):
        yr = df[df['year'] == year]
        news_cov = (yr['news_sentiment_1d'] != 0.0).mean() * 100
        st_cov   = (yr['st_sentiment_1d'] != 0.0).mean() * 100
        logger.info(f'  {year}: {len(yr):>5,} rows | '
                    f'news coverage {news_cov:>5.1f}% | '
                    f'ST coverage {st_cov:>5.1f}%')

    df = df.drop(columns=['year'])

    return df


def verify_no_leakage(df):
    """Confirm no future data in features."""
    target_cols = [c for c in df.columns if 'target' in c]
    feature_cols = [
        'news_sentiment_1d','st_sentiment_1d','st_bull_pct',
        'post_count_1d','mention_growth_1d','mention_growth_7d',
    ]

    logger.info('\n=== LEAKAGE CHECK ===')
    for fc in feature_cols:
        if fc not in df.columns:
            continue
        for tc in target_cols[:1]:  # check against first target only
            corr = df[[fc, tc]].dropna()
            if len(corr) < 10:
                continue
            r = stats.spearmanr(corr[fc], corr[tc]).correlation
            logger.info(f'  {fc} vs {tc}: IC={r:.4f} (expected small, not > 0.9)')
            if abs(r) > 0.9:
                logger.error(f'  POSSIBLE LEAKAGE: IC={r:.4f} is suspiciously high')

    logger.info('Leakage check complete')


def merge_live_features():
    """
    Merge live 2026 features into features_complete.parquet.
    Run manually every 30 days or when retraining.
    Only rows with target_return_5d filled are merged.
    """
    base_path = Path(OUTPUT_PATH)
    live_path = Path('data/features/features_live_2026.parquet')

    if not live_path.exists():
        logger.info('No live features yet — run daily_run_live.py first')
        return

    live = pd.read_parquet(live_path)
    live_complete = live[live['target_return_5d'].notna()].copy()

    if len(live_complete) == 0:
        logger.info('No completed live features yet — need 5+ trading days')
        return

    if not base_path.exists():
        logger.error(f'Base feature store not found: {base_path}')
        return

    base     = pd.read_parquet(base_path)
    combined = pd.concat([base, live_complete], ignore_index=True)
    combined = combined.drop_duplicates(subset=['ticker', 'date'], keep='last')
    combined = combined.sort_values(['ticker', 'date']).reset_index(drop=True)
    combined.to_parquet(base_path, index=False)

    years = sorted(pd.to_datetime(combined['date']).dt.year.unique().tolist())
    logger.info(f'Merged {len(live_complete)} live rows into {base_path}')
    logger.info(f'Total: {len(combined)} rows | Years: {years}')


def main(verify_only=False):
    # Verify inputs exist
    check_inputs()

    if verify_only:
        logger.info('Verify-only mode — not merging')
        return

    # Merge
    df = merge_features()

    # Leakage check
    verify_no_leakage(df)

    # Save
    Path('data/features').mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    size_mb = Path(OUTPUT_PATH).stat().st_size / 1024**2
    logger.info(f'\n=== SAVED ===')
    logger.info(f'Path:     {OUTPUT_PATH}')
    logger.info(f'Size:     {size_mb:.1f} MB')
    logger.info(f'Rows:     {len(df):,}')
    logger.info(f'Features: {len([c for c in df.columns if c not in ["ticker","date","split","year"] and "target" not in c])}')
    logger.info(f'Targets:  {[c for c in df.columns if "target" in c]}')
    logger.info(f'\nNext: python scripts/train_phase3_model.py \\')
    logger.info(f'        --feature-path {OUTPUT_PATH} \\')
    logger.info(f'        --train-years 2022,2023 \\')
    logger.info(f'        --test-years 2024,2025')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true',
                        help='Only check input files exist, do not merge')
    parser.add_argument('--merge-live', action='store_true',
                        help='Merge features_live_2026.parquet into features_complete.parquet')
    args = parser.parse_args()
    if args.merge_live:
        merge_live_features()
    else:
        main(verify_only=args.verify_only)
