"""
Replace raw 5-day return target with SPY-residual return.
Hypothesis: Stripping market noise improves IC.
"""
import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
import xgboost as xgb
import json
from pathlib import Path

Path('experiments/ic_improvements').mkdir(parents=True, exist_ok=True)

df = pd.read_parquet('data/features/features_expanded.parquet')
df = df[df['post_count_1d'] >= 10].copy()

spy = yf.download('SPY', start='2019-01-01', end='2025-01-01',
                  auto_adjust=True, progress=False)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)
spy.columns = [c.lower() for c in spy.columns]
spy['spy_ret_5d'] = spy['close'].shift(-5) / spy['close'] - 1
spy.index = pd.to_datetime(spy.index).date
spy_dict = spy['spy_ret_5d'].to_dict()

df['date_key'] = pd.to_datetime(df['date']).dt.date
df['spy_5d']   = df['date_key'].map(spy_dict)
df['target_neutral_5d'] = df['target_return_5d'] - df['spy_5d']
df = df.dropna(subset=['target_neutral_5d'])

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

train = df[df['split']=='train']
test  = df[df['split']=='test']

results = {}
for target_name, target_col in [
    ('raw_return_5d',     'target_return_5d'),
    ('neutral_return_5d', 'target_neutral_5d'),
]:
    m = xgb.XGBRegressor(**params)
    m.fit(train[FEATURES].fillna(0), train[target_col])
    pred = m.predict(test[FEATURES].fillna(0))
    ic_raw     = stats.spearmanr(pred, test['target_return_5d']).correlation
    ic_neutral = stats.spearmanr(pred, test['target_neutral_5d']).correlation
    results[target_name] = {
        'ic_vs_raw_return':     float(ic_raw),
        'ic_vs_neutral_return': float(ic_neutral),
        'n_test': len(test)
    }
    print(f'{target_name:<22}  IC_raw={ic_raw:.4f}  IC_neutral={ic_neutral:.4f}')

baseline_ic = results['raw_return_5d']['ic_vs_raw_return']
improved_ic = results['neutral_return_5d']['ic_vs_raw_return']
improvement = improved_ic - baseline_ic
print(f'\nIC improvement from neutral target: {improvement:+.4f}')
verdict = 'ADOPT' if improvement > 0.005 else 'SKIP'
print(f'Verdict: {verdict} (threshold: +0.005)')

with open('experiments/ic_improvements/exp1_results.json', 'w') as f:
    json.dump({**results, 'improvement': float(improvement), 'verdict': verdict}, f, indent=2)
print('Saved: experiments/ic_improvements/exp1_results.json')
