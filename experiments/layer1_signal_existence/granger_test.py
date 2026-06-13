"""
Layer 1: Signal existence test.
Granger causality per year + sign consistency check.

Limitation: Granger detects predictive lag structure, not true causality.
Treat significant results as "predictive information exists", not causal proof.
"""
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
import sys, json
from pathlib import Path

sys.path.insert(0, '.')
from experiments.shared.validation_utils import DENSITY_GATE

Path('experiments/layer1_signal_existence').mkdir(parents=True, exist_ok=True)

DROP_TICKERS = ['ASTS', 'LCID', 'MSTR', 'RIOT', 'RIVN', 'SMCI', 'WMT']

df = pd.read_parquet('data/features/features_expanded.parquet')
df = df[df['post_count_1d'] >= DENSITY_GATE].copy()
df = df[~df['ticker'].isin(DROP_TICKERS)].copy()
df['year'] = pd.to_datetime(df['date']).dt.year

SENTIMENT_FAMILY = ['avg_sentiment_1d', 'sentiment_accel']
ATTENTION_FAMILY = ['post_count_1d', 'mention_growth_7d']

print('=== LAYER 1: SIGNAL EXISTENCE ===')
print()
print('IMPORTANT: Granger tests predictive lag structure, not true causality.')
print('Significant p-value means: past feature predicts future returns')
print('beyond what past returns alone predict.')
print()

results = {}

for family_name, features in [
    ('SENTIMENT', SENTIMENT_FAMILY),
    ('ATTENTION', ATTENTION_FAMILY),
]:
    print(f'--- {family_name} FAMILY ---')
    features = [f for f in features if f in df.columns]
    family_results = {}

    for year in sorted(df['year'].unique()):
        yr = df[df['year'] == year].sort_values('date')
        year_results = {}

        for feat in features:
            series = yr[['target_return_5d', feat]].dropna()
            if len(series) < 30:
                year_results[feat] = {'p_value': None, 'significant': False,
                                      'n': len(series), 'note': 'too_few_rows'}
                continue
            try:
                gc = grangercausalitytests(series, maxlag=3, verbose=False)
                p_val = float(gc[1][0]['ssr_ftest'][1])
                sig   = p_val < 0.05
            except Exception as e:
                p_val = None
                sig   = False

            year_results[feat] = {
                'p_value':     round(p_val, 4) if p_val is not None else None,
                'significant': bool(sig),
                'n':           int(len(series)),
            }

            p_str = f'{p_val:.4f}' if p_val is not None else 'N/A'
            flag  = ' ***' if sig else ''
            print(f'  {year}  {feat:<22}  p={p_str}{flag}  n={len(series)}')

        family_results[str(year)] = year_results

    results[family_name] = family_results

    sig_years = [
        yr for yr, yr_data in family_results.items()
        if any(v.get('significant', False) for v in yr_data.values())
    ]
    print(f'  → {family_name} significant in {len(sig_years)}/6 years: {sig_years}')
    print()

sentiment_sig = sum(
    1 for yr in results.get('SENTIMENT', {}).values()
    if any(v.get('significant') for v in yr.values())
)
attention_sig = sum(
    1 for yr in results.get('ATTENTION', {}).values()
    if any(v.get('significant') for v in yr.values())
)

print('=== LAYER 1 DECISION ===')
print(f'Sentiment family significant in {sentiment_sig}/6 years')
print(f'Attention family significant in {attention_sig}/6 years')
print()

any_signal = sentiment_sig >= 2 or attention_sig >= 2
if any_signal:
    print('VERDICT: PROCEED to Layer 2')
    print('  Causal-predictive signal detected in at least one family.')
else:
    print('VERDICT: WEAK SIGNAL')
    print('  Signal < 2 years. Phase 3 uses market features + Reddit as')
    print('  attention filter only. Sentiment family dropped from model.')

output = {
    'results':               results,
    'sentiment_sig_years':   int(sentiment_sig),
    'attention_sig_years':   int(attention_sig),
    'any_signal':            bool(any_signal),
    'proceed_to_layer2':     bool(any_signal),
    'limitation_note':       'Granger detects predictive lag only, not true causality.',
}
with open('experiments/layer1_signal_existence/granger_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print('\nSaved: experiments/layer1_signal_existence/granger_results.json')
