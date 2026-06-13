"""
For each feature, show:
- Distribution shape (normal? skewed? bimodal?)
- Mean and std per ticker
- % of rows that are zero or NaN
- Correlation with target_return_5d
"""
import pandas as pd
import numpy as np
from scipy import stats
import json
from pathlib import Path

Path('reports/eda').mkdir(parents=True, exist_ok=True)

df = pd.read_parquet('data/features/features_expanded.parquet')
df10 = df[df['post_count_1d'] >= 10].copy()

REDDIT_FEATURES = [
    'post_count_1d', 'post_count_3d', 'post_count_7d',
    'avg_sentiment_1d', 'avg_sentiment_3d', 'weighted_sentiment',
    'sentiment_std', 'sentiment_accel', 'bullish_ratio',
    'mention_growth_1d', 'mention_growth_7d',
    'total_upvotes_1d', 'total_comments_1d', 'unique_authors_1d',
]
REDDIT_FEATURES = [f for f in REDDIT_FEATURES if f in df10.columns]

MARKET_FEATURES = [
    'returns_1d', 'returns_5d', 'returns_20d',
    'rsi_14', 'atr_14', 'relative_volume',
    'dist_from_20ma', 'dist_from_50ma'
]

print(f'Density-filtered rows: {len(df10)} (from {len(df)} total)')
print()
print(f'{"Feature":<25} {"Mean":>8} {"Std":>8} {"Skew":>8} '
      f'{"% Zero":>8} {"% NaN":>7} {"IC_5d":>8}')
print('-' * 80)

eda_results = {}
for feat in REDDIT_FEATURES + MARKET_FEATURES:
    if feat not in df10.columns:
        continue
    col = df10[feat]
    ic = stats.spearmanr(col.fillna(0), df10['target_return_5d']).correlation
    eda_results[feat] = {
        'mean':    float(col.mean()),
        'std':     float(col.std()),
        'skew':    float(col.skew()),
        'pct_zero': float((col == 0).mean() * 100),
        'pct_nan':  float(col.isna().mean() * 100),
        'ic_5d':   float(ic),
        'type':    'reddit' if feat in REDDIT_FEATURES else 'market'
    }
    print(f'{feat:<25} {col.mean():>8.3f} {col.std():>8.3f} '
          f'{col.skew():>8.3f} {(col==0).mean()*100:>7.1f}% '
          f'{col.isna().mean()*100:>6.1f}% {ic:>8.4f}')

with open('reports/eda/feature_stats.json', 'w') as f:
    json.dump(eda_results, f, indent=2)
print('\nSaved: reports/eda/feature_stats.json')
