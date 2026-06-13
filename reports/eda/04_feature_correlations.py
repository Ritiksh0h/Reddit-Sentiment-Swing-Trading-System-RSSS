"""
Show correlations between features.
High correlations (>0.7) between features = redundancy = remove one.
"""
import pandas as pd
import numpy as np

df = pd.read_parquet('data/features/features_expanded.parquet')
df10 = df[df['post_count_1d'] >= 10].copy()

ALL_FEATURES = [
    'post_count_1d', 'post_count_3d', 'avg_sentiment_1d',
    'avg_sentiment_3d', 'weighted_sentiment', 'sentiment_accel',
    'bullish_ratio', 'mention_growth_1d', 'mention_growth_7d',
    'returns_1d', 'returns_5d', 'returns_20d',
    'rsi_14', 'relative_volume', 'dist_from_20ma', 'dist_from_50ma'
]
ALL_FEATURES = [f for f in ALL_FEATURES if f in df10.columns]

corr = df10[ALL_FEATURES].corr(method='spearman')

print('High correlation pairs (|r| > 0.6):')
print(f'{"Feature A":<25} {"Feature B":<25} {"Corr":>8}')
print('-' * 60)
found = []
for i in range(len(ALL_FEATURES)):
    for j in range(i+1, len(ALL_FEATURES)):
        r = corr.iloc[i, j]
        if abs(r) > 0.6:
            found.append((ALL_FEATURES[i], ALL_FEATURES[j], float(r)))
            print(f'{ALL_FEATURES[i]:<25} {ALL_FEATURES[j]:<25} {r:>8.3f}')

if not found:
    print('No high correlation pairs found.')

print(f'\nTotal high-correlation pairs: {len(found)}')

print('\nTop IC features vs target (individual Spearman):')
from scipy import stats
ic_pairs = []
for f in ALL_FEATURES:
    ic = stats.spearmanr(df10[f].fillna(0), df10['target_return_5d']).correlation
    ic_pairs.append((f, ic))
ic_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
for f, ic in ic_pairs:
    print(f'  {f:<25} IC={ic:>8.4f}')

corr.to_csv('reports/eda/feature_correlation_matrix.csv')
print('\nSaved: reports/eda/feature_correlation_matrix.csv')
