"""
Test XGBoost rank:ndcg vs reg:squarederror.
Hypothesis: Training to rank (not predict exact return) improves IC.
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
df['date_str'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

FEATURES = [
    'returns_1d','returns_5d','returns_20d','rsi_14','atr_14',
    'relative_volume','dist_from_20ma','dist_from_50ma',
    'avg_sentiment_1d','weighted_sentiment','sentiment_accel',
    'mention_growth_1d','mention_growth_7d','post_count_1d'
]
FEATURES = [f for f in FEATURES if f in df.columns]

train = df[df['split']=='train'].copy()
test  = df[df['split']=='test'].copy()

# Regression baseline
m_reg = xgb.XGBRegressor(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    objective='reg:squarederror', random_state=42, n_jobs=-1
)
m_reg.fit(train[FEATURES].fillna(0), train['target_return_5d'])
pred_reg = m_reg.predict(test[FEATURES].fillna(0))
ic_reg = stats.spearmanr(pred_reg, test['target_return_5d']).correlation
print(f'Regression objective IC:  {ic_reg:.4f}')

# Ranking model
train_sorted = train.sort_values('date_str')
test_sorted  = test.sort_values('date_str')
train_groups = train_sorted.groupby('date_str').size().values
test_groups  = test_sorted.groupby('date_str').size().values

train_sorted = train_sorted.copy()
test_sorted  = test_sorted.copy()
train_sorted['rank_label'] = train_sorted.groupby('date_str')['target_return_5d'].rank(pct=True)
test_sorted['rank_label']  = test_sorted.groupby('date_str')['target_return_5d'].rank(pct=True)

ic_rank = float('nan')
try:
    m_rank = xgb.XGBRanker(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        objective='rank:ndcg', random_state=42, n_jobs=-1
    )
    m_rank.fit(
        train_sorted[FEATURES].fillna(0),
        train_sorted['rank_label'],
        group=train_groups
    )
    pred_rank = m_rank.predict(test_sorted[FEATURES].fillna(0))
    ic_rank = stats.spearmanr(pred_rank, test_sorted['target_return_5d']).correlation
    print(f'Ranking objective IC:     {ic_rank:.4f}  (rank:ndcg)')
except Exception as e:
    print(f'Ranking model failed: {e}')
    print(f'Ranking objective IC:     N/A')

improvement = ic_rank - ic_reg if not np.isnan(ic_rank) else 0.0
print(f'Improvement:              {improvement:+.4f}')
verdict = 'ADOPT rank:ndcg' if improvement > 0.005 else 'KEEP reg:squarederror'
print(f'Verdict: {verdict}')

with open('experiments/ic_improvements/exp5_results.json', 'w') as f:
    json.dump({'ic_regression': float(ic_reg), 'ic_ranking': float(ic_rank),
               'improvement': float(improvement), 'verdict': verdict}, f, indent=2)
print('Saved: experiments/ic_improvements/exp5_results.json')
