"""
Replace fixed post_count >= 10 with rolling Z-score > threshold.
Hypothesis: Relative attention spike is better signal than absolute count.
"""
import pandas as pd
import numpy as np
from scipy import stats
import xgboost as xgb
import json
from pathlib import Path

Path('experiments/ic_improvements').mkdir(parents=True, exist_ok=True)

df = pd.read_parquet('data/features/features_expanded.parquet')
df = df.sort_values(['ticker', 'date'])

df['post_zscore'] = df.groupby('ticker')['post_count_1d'].transform(
    lambda x: (x - x.rolling(30, min_periods=5).mean()) /
              (x.rolling(30, min_periods=5).std().replace(0, 1) + 1e-8)
)

FEATURES = [
    'returns_1d','returns_5d','returns_20d','rsi_14','atr_14',
    'relative_volume','dist_from_20ma','dist_from_50ma',
    'avg_sentiment_1d','avg_sentiment_3d','weighted_sentiment',
    'sentiment_std','sentiment_accel','bullish_ratio',
    'post_count_1d','mention_growth_1d','mention_growth_7d'
]
FEATURES = [f for f in FEATURES if f in df.columns]

params = dict(n_estimators=500, max_depth=4, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
              reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1)

configs = [
    ('Fixed >= 10 (baseline)',  df['post_count_1d'] >= 10),
    ('Z-score > 0.5',           df['post_zscore'] > 0.5),
    ('Z-score > 1.0',           df['post_zscore'] > 1.0),
    ('Z-score > 1.5',           df['post_zscore'] > 1.5),
    ('Z-score > 2.0',           df['post_zscore'] > 2.0),
    ('Fixed >= 5',              df['post_count_1d'] >= 5),
    ('Fixed >= 15',             df['post_count_1d'] >= 15),
]

results = {}
baseline_ic = None
print(f'{"Config":<30} {"IC":>8} {"n_test":>8} {"n_train":>8} {"improvement":>12}')
print('-' * 70)

for name, mask in configs:
    filtered = df[mask].copy()
    train = filtered[filtered['split']=='train']
    test  = filtered[filtered['split']=='test']
    if len(train) < 100 or len(test) < 30:
        print(f'{name:<30}  insufficient data (train={len(train)}, test={len(test)})')
        continue
    avail = [f for f in FEATURES if f in train.columns]
    m = xgb.XGBRegressor(**params)
    m.fit(train[avail].fillna(0), train['target_return_5d'])
    pred = m.predict(test[avail].fillna(0))
    ic = stats.spearmanr(pred, test['target_return_5d']).correlation
    if baseline_ic is None:
        baseline_ic = ic
    improvement = ic - baseline_ic
    results[name] = {'ic': float(ic), 'n_train': len(train),
                     'n_test': len(test), 'improvement': float(improvement)}
    print(f'{name:<30} {ic:>8.4f} {len(test):>8} {len(train):>8} {improvement:>+12.4f}')

best = max(results, key=lambda k: results[k]['ic'])
print(f'\nBest gate: {best} (IC={results[best]["ic"]:.4f})')
verdict = 'ADOPT' if results[best]['improvement'] > 0.005 else 'MARGINAL'
print(f'Verdict: {verdict}')

with open('experiments/ic_improvements/exp3_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved: experiments/ic_improvements/exp3_results.json')
