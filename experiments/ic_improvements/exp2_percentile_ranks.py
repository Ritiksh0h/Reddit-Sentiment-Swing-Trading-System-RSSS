"""
Convert raw feature values to rolling 30-day percentile ranks per ticker.
Hypothesis: Normalizing relative to each ticker's history improves IC.
"""
import pandas as pd
import numpy as np
from scipy import stats
import xgboost as xgb
import json
from pathlib import Path

Path('experiments/ic_improvements').mkdir(parents=True, exist_ok=True)

df = pd.read_parquet('data/features/features_expanded.parquet')
df = df[df['post_count_1d'] >= 10].copy()
df = df.sort_values(['ticker', 'date'])

RANK_COLS = [
    'avg_sentiment_1d', 'avg_sentiment_3d', 'weighted_sentiment',
    'sentiment_accel', 'mention_growth_1d', 'mention_growth_7d',
    'post_count_1d', 'total_upvotes_1d', 'relative_volume',
    'rsi_14', 'returns_5d'
]
RANK_COLS = [c for c in RANK_COLS if c in df.columns]

print('Computing rolling percentile ranks (30-day window per ticker)...')
for col in RANK_COLS:
    df[f'{col}_rank'] = df.groupby('ticker')[col].transform(
        lambda x: x.rolling(30, min_periods=5).rank(pct=True)
    )
print(f'Added {len(RANK_COLS)} rank features.')

MARKET_BASE   = ['returns_1d','returns_5d','returns_20d','rsi_14','atr_14',
                  'relative_volume','dist_from_20ma','dist_from_50ma']
REDDIT_BASE   = ['avg_sentiment_1d','weighted_sentiment','sentiment_accel',
                  'mention_growth_1d','mention_growth_7d','post_count_1d']
REDDIT_RANKED = [f'{c}_rank' for c in REDDIT_BASE if f'{c}_rank' in df.columns]
MARKET_RANKED = [f'{c}_rank' for c in ['rsi_14','relative_volume','returns_5d']
                 if f'{c}_rank' in df.columns]

params = dict(n_estimators=500, max_depth=4, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
              reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1)

train = df[df['split']=='train']
test  = df[df['split']=='test']

configs = [
    ('Baseline (raw features)',        MARKET_BASE + REDDIT_BASE),
    ('Reddit ranked only',             MARKET_BASE + REDDIT_RANKED),
    ('Market+Reddit ranked',           MARKET_RANKED + MARKET_BASE + REDDIT_RANKED),
    ('All ranked',                     MARKET_RANKED + REDDIT_RANKED),
]

results = {}
print(f'\n{"Config":<35} {"IC":>8} {"n_test":>8} {"improvement":>12}')
print('-' * 65)

baseline_ic = None
for name, feats in configs:
    avail = [f for f in feats if f in train.columns]
    if len(avail) < 5:
        continue
    m = xgb.XGBRegressor(**params)
    m.fit(train[avail].fillna(0), train['target_return_5d'])
    pred = m.predict(test[avail].fillna(0))
    ic = stats.spearmanr(pred, test['target_return_5d']).correlation
    if baseline_ic is None:
        baseline_ic = ic
    improvement = ic - baseline_ic
    results[name] = {'ic': float(ic), 'improvement': float(improvement),
                     'n_features': len(avail)}
    print(f'{name:<35} {ic:>8.4f} {len(test):>8} {improvement:>+12.4f}')

best = max(results, key=lambda k: results[k]['ic'])
print(f'\nBest config: {best} (IC={results[best]["ic"]:.4f})')
verdict = 'ADOPT' if results[best]['improvement'] > 0.005 else 'SKIP'
print(f'Verdict: {verdict}')

with open('experiments/ic_improvements/exp2_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved: experiments/ic_improvements/exp2_results.json')
