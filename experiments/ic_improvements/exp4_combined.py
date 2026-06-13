"""
Combine the best results from experiments 1-3.
Only run this after reviewing exp1-3 results.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import json
from scipy import stats
import xgboost as xgb
from pathlib import Path

exp1 = json.load(open('experiments/ic_improvements/exp1_results.json'))
exp2 = json.load(open('experiments/ic_improvements/exp2_results.json'))
exp3 = json.load(open('experiments/ic_improvements/exp3_results.json'))

use_neutral_target = exp1.get('verdict') == 'ADOPT'
best_rank_config   = max(
    (k for k, v in exp2.items() if isinstance(v, dict) and 'ic' in v),
    key=lambda k: exp2[k]['ic']
)
best_gate = max(
    (k for k, v in exp3.items() if isinstance(v, dict) and 'ic' in v),
    key=lambda k: exp3[k]['ic']
)

print('Building combined config:')
print(f'  Neutral target: {use_neutral_target}')
print(f'  Rank config:    {best_rank_config}')
print(f'  Gate:           {best_gate}')

df = pd.read_parquet('data/features/features_expanded.parquet')
df = df.sort_values(['ticker', 'date'])

df['post_zscore'] = df.groupby('ticker')['post_count_1d'].transform(
    lambda x: (x - x.rolling(30, min_periods=5).mean()) /
              (x.rolling(30, min_periods=5).std().replace(0, 1) + 1e-8)
)

if 'Z-score' in best_gate:
    threshold = float(best_gate.split('> ')[1])
    df = df[df['post_zscore'] > threshold].copy()
else:
    threshold = int(best_gate.replace('Fixed >= ', '').split(' ')[0])
    df = df[df['post_count_1d'] >= threshold].copy()

RANK_COLS = ['avg_sentiment_1d','weighted_sentiment','mention_growth_1d',
             'post_count_1d','rsi_14','relative_volume','returns_5d']
RANK_COLS = [c for c in RANK_COLS if c in df.columns]
for col in RANK_COLS:
    df[f'{col}_rank'] = df.groupby('ticker')[col].transform(
        lambda x: x.rolling(30, min_periods=5).rank(pct=True)
    )

if use_neutral_target:
    spy = yf.download('SPY', start='2019-01-01', end='2025-01-01',
                      auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.columns = [c.lower() for c in spy.columns]
    spy['spy_ret_5d'] = spy['close'].shift(-5) / spy['close'] - 1
    spy.index = pd.to_datetime(spy.index).date
    df['date_key'] = pd.to_datetime(df['date']).dt.date
    df['spy_5d']   = df['date_key'].map(spy['spy_ret_5d'].to_dict())
    df['target_use'] = df['target_return_5d'] - df['spy_5d'].fillna(0)
else:
    df['target_use'] = df['target_return_5d']

df = df.dropna(subset=['target_use'])

MARKET = ['returns_1d','returns_5d','returns_20d','rsi_14','atr_14',
          'relative_volume','dist_from_20ma','dist_from_50ma']
REDDIT = ['avg_sentiment_1d','weighted_sentiment','sentiment_accel',
          'mention_growth_1d','mention_growth_7d','post_count_1d']
RANKED = [f'{c}_rank' for c in RANK_COLS if f'{c}_rank' in df.columns]
ALL_FEATURES = list(dict.fromkeys(MARKET + REDDIT + RANKED))
ALL_FEATURES = [f for f in ALL_FEATURES if f in df.columns]

params = dict(n_estimators=500, max_depth=4, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
              reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1)

train = df[df['split']=='train']
test  = df[df['split']=='test']

# Baseline
df_base = pd.read_parquet('data/features/features_expanded.parquet')
df_base = df_base[df_base['post_count_1d'] >= 10]
base_feats = [f for f in MARKET + REDDIT if f in df_base.columns]
m_base = xgb.XGBRegressor(**params)
m_base.fit(df_base[df_base['split']=='train'][base_feats].fillna(0),
           df_base[df_base['split']=='train']['target_return_5d'])
pred_base = m_base.predict(df_base[df_base['split']=='test'][base_feats].fillna(0))
ic_base = stats.spearmanr(pred_base, df_base[df_base['split']=='test']['target_return_5d']).correlation

# Combined model
m = xgb.XGBRegressor(**params)
m.fit(train[ALL_FEATURES].fillna(0), train['target_use'])
pred = m.predict(test[ALL_FEATURES].fillna(0))
ic_combined = stats.spearmanr(pred, test['target_return_5d']).correlation

print(f'\n=== COMBINED EXPERIMENT RESULTS ===')
print(f'Baseline IC (current):     {ic_base:.4f}')
print(f'Combined IC (improved):    {ic_combined:.4f}')
print(f'Improvement:               {ic_combined - ic_base:+.4f}')
print(f'Test rows:                 {len(test)}')
print()
if ic_combined >= 0.10:
    verdict = 'STRONG — adopt for Phase 3'
elif ic_combined > ic_base + 0.005:
    verdict = 'IMPROVED — adopt for Phase 3'
elif abs(ic_combined - ic_base) <= 0.005:
    verdict = 'NEUTRAL — keep current architecture'
else:
    verdict = 'WORSE — keep current architecture'
print(f'VERDICT: {verdict}')

results = {
    'baseline_ic': float(ic_base),
    'combined_ic': float(ic_combined),
    'improvement': float(ic_combined - ic_base),
    'verdict': verdict,
    'n_test': len(test),
    'config': {
        'neutral_target': use_neutral_target,
        'density_gate': best_gate,
        'rank_features': RANKED,
    }
}
with open('experiments/ic_improvements/exp4_combined_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved: experiments/ic_improvements/exp4_combined_results.json')
