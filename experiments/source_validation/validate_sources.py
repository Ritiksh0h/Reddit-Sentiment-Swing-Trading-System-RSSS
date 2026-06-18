"""
Multi-source signal validation.
Answers: which data source (Reddit/news/StockTwits) adds predictive value?

Tests each source independently then in combination against market baseline.
Granger causality tests run per-year due to cross-sectional panel structure.
Walk-forward IC uses expanding window (2022→2023→2024→2025 test years).

Output: experiments/source_validation/results.json

Run: python experiments/source_validation/validate_sources.py
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

Path('experiments/source_validation').mkdir(parents=True, exist_ok=True)

# ── Load feature store ────────────────────────────────────────────────────
logger.info('Loading feature store...')
df = pd.read_parquet('data/features/features_complete.parquet')
df = df[df['post_count_1d'] >= 10].copy()
df = df[~df['ticker'].isin(['ASTS', 'LCID', 'MSTR', 'RIOT', 'RIVN', 'SMCI', 'WMT'])].copy()
df['year'] = pd.to_datetime(df['date']).dt.year
# Exclude 2026 — incomplete forward returns
df = df[df['year'] < 2026].copy()
logger.info(f'Rows: {len(df):,}  Tickers: {df["ticker"].nunique()}  '
            f'Years: {sorted(df["year"].unique())}')

TARGET = 'target_return_5d'

# ── Feature families ──────────────────────────────────────────────────────
MARKET_FEATURES = [
    'returns_1d', 'returns_5d', 'returns_20d', 'rsi_14', 'atr_14',
    'relative_volume', 'dist_from_20ma', 'dist_from_50ma',
]
REDDIT_FEATURES = ['post_count_1d', 'mention_growth_1d', 'mention_growth_7d']
NEWS_FEATURES   = ['news_sentiment_1d']
ST_FEATURES     = ['st_sentiment_1d', 'st_bull_pct']
ALL_FEATURES    = MARKET_FEATURES + REDDIT_FEATURES + NEWS_FEATURES + ST_FEATURES

XGB_PARAMS = dict(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.6, colsample_bytree=0.6, min_child_weight=20,
    reg_alpha=0.5, reg_lambda=2.0, random_state=42, n_jobs=-1,
    objective='reg:squarederror',
)

WALK_FORWARD_WINDOWS = [
    {'train': [2019, 2020, 2021],             'test': [2022]},
    {'train': [2019, 2020, 2021, 2022],       'test': [2023]},
    {'train': [2019, 2020, 2021, 2022, 2023], 'test': [2024]},
    {'train': [2019, 2020, 2021, 2022, 2023, 2024], 'test': [2025]},
]

REGIME_LABELS = {
    2019: 'BULL (pre-COVID, low vol)',
    2020: 'CRASH + RECOVERY (retail surge)',
    2021: 'RETAIL BULL (meme peak, zero rates)',
    2022: 'BEAR (rate hikes, crypto crash)',
    2023: 'RECOVERY (mixed, AI emerges)',
    2024: 'AI BULL (institutional momentum)',
    2025: 'MIXED (regime uncertain)',
}


def annual_ic(df, feature):
    """Compute raw Spearman IC for a single feature vs TARGET, per year."""
    ics = {}
    for year in sorted(df['year'].unique()):
        yr = df[df['year'] == year][[feature, TARGET]].dropna()
        if len(yr) < 20:
            ics[int(year)] = None
            continue
        result = stats.spearmanr(yr[feature], yr[TARGET])
        ics[int(year)] = round(float(result.statistic), 4)
    return ics


def walk_forward_ic(df, features):
    """Run walk-forward IC across all windows. Returns list of IC per window."""
    ics = []
    for window in WALK_FORWARD_WINDOWS:
        train = df[df['year'].isin(window['train'])]
        test  = df[df['year'].isin(window['test'])]
        avail = [f for f in features if f in train.columns]
        if len(train) < 50 or len(test) < 10 or not avail:
            ics.append(float('nan'))
            continue
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(train[avail].fillna(0), train[TARGET])
        pred = model.predict(test[avail].fillna(0))
        ic   = float(stats.spearmanr(pred, test[TARGET]).statistic)
        ics.append(ic)
    return ics


def granger_test(df, feature, max_lag=1):
    """Run Granger causality test per year on panel data sorted by date."""
    results = {}
    for year in sorted(df['year'].unique()):
        yr = df[df['year'] == year].sort_values('date')[[TARGET, feature]].dropna()
        if len(yr) < 30:
            results[int(year)] = {'p_value': None, 'significant': False, 'n': int(len(yr))}
            continue
        try:
            gc    = grangercausalitytests(yr, maxlag=max_lag, verbose=False)
            p_val = float(gc[1][0]['ssr_ftest'][1])
            results[int(year)] = {
                'p_value':     round(p_val, 4),
                'significant': bool(p_val < 0.05),
                'n':           int(len(yr)),
            }
        except Exception as e:
            results[int(year)] = {'p_value': None, 'significant': False,
                                  'n': int(len(yr)), 'error': str(e)}
    return results


# ════════════════════════════════════════════════════════════════════════════
# LAYER 1 — Annual raw IC per source feature (no model, no walk-forward)
# ════════════════════════════════════════════════════════════════════════════
print()
print('=' * 70)
print('LAYER 1 — ANNUAL RAW IC PER FEATURE')
print('=' * 70)
print('IC > 0.05 in 2+ years = meaningful signal threshold')
print()

test_features = {
    'Reddit — post_count_1d':       'post_count_1d',
    'Reddit — mention_growth_7d':   'mention_growth_7d',
    'Reddit — avg_sentiment_1d':    'avg_sentiment_1d',
    'News — news_sentiment_1d':     'news_sentiment_1d',
    'ST — st_sentiment_1d':         'st_sentiment_1d',
    'ST — st_bull_pct':             'st_bull_pct',
    'Market — returns_5d':          'returns_5d',
    'Market — relative_volume':     'relative_volume',
    'Market — dist_from_20ma':      'dist_from_20ma',
}

years = sorted(df['year'].unique())
header = f'{"Feature":<35}' + ''.join(f'{y:>8}' for y in years)
print(header)
print('-' * (35 + 8 * len(years)))

layer1_results = {}
for label, feature in test_features.items():
    if feature not in df.columns:
        print(f'{label:<35} {"N/A (missing)":>8}')
        continue
    ics = annual_ic(df, feature)
    row = f'{label:<35}'
    for year in years:
        val = ics.get(int(year))
        if val is None:
            row += f'{"—":>8}'
        else:
            flag = '*' if abs(val) > 0.05 else ' '
            row += f'{val:>7.4f}{flag}'
    print(row)
    layer1_results[feature] = {
        'ic_by_year': ics,
        'regime_labels': {str(y): REGIME_LABELS.get(int(y), 'UNKNOWN') for y in years},
    }

print()
print('* = |IC| > 0.05 (meaningful signal threshold)')
print()
print('Macro regime context:')
print('─' * 55)
for y in years:
    print(f'  {y}: {REGIME_LABELS.get(int(y), "UNKNOWN")}')
print()
print('Interpretation: HIGH IC in 2020-2021 (retail mania) + LOW in 2022 (bear)')
print('→ regime-dependent signal, unreliable. Consistent IC across all regimes')
print('→ structural signal worth keeping.')

# ════════════════════════════════════════════════════════════════════════════
# LAYER 2 — Granger causality per source
# ════════════════════════════════════════════════════════════════════════════
print()
print('=' * 70)
print('LAYER 2 — GRANGER CAUSALITY TEST PER SOURCE')
print('=' * 70)
print('Does past X predict future returns beyond what past returns predict?')
print('p < 0.05 = significant causal-predictive structure (max_lag=1)')
print()

granger_features = {
    'post_count_1d':     'Reddit attention',
    'avg_sentiment_1d':  'Reddit sentiment',
    'news_sentiment_1d': 'News sentiment (FinBERT)',
    'st_sentiment_1d':   'StockTwits sentiment',
    'st_bull_pct':       'StockTwits bull %',
}

layer2_results = {}
for feature, label in granger_features.items():
    if feature not in df.columns:
        continue
    logger.info(f'Granger test: {label}...')
    print(f'--- {label} ---')
    result = granger_test(df, feature)
    sig_years = [y for y, r in result.items() if r.get('significant')]
    for year, r in sorted(result.items()):
        p     = r.get('p_value')
        p_str = f'{p:.4f}' if p is not None else 'N/A '
        flag  = ' ***' if r.get('significant') else '    '
        n     = r.get('n', 0)
        print(f'  {year}  p={p_str}{flag}  n={n:,}')
    print(f'  → Significant in {len(sig_years)}/{len(result)} years: {sig_years}')
    print()
    layer2_results[feature] = {
        'label':     label,
        'results':   result,
        'sig_years': sig_years,
        'sig_count': len(sig_years),
        'verdict':   'HAS_SIGNAL' if len(sig_years) >= 2 else 'WEAK_SIGNAL',
    }

# ════════════════════════════════════════════════════════════════════════════
# LAYER 3 — Walk-forward IC by feature combination
# ════════════════════════════════════════════════════════════════════════════
print('=' * 70)
print('LAYER 3 — WALK-FORWARD IC BY FEATURE COMBINATION')
print('=' * 70)
print('Which combination of sources produces the best out-of-sample IC?')
print('Expanding window: train grows, test always = 1 year held out')
print()

combinations = {
    'Market only':             MARKET_FEATURES,
    'Market + Reddit':         MARKET_FEATURES + REDDIT_FEATURES,
    'Market + News':           MARKET_FEATURES + NEWS_FEATURES,
    'Market + StockTwits':     MARKET_FEATURES + ST_FEATURES,
    'Market + Reddit + News':  MARKET_FEATURES + REDDIT_FEATURES + NEWS_FEATURES,
    'Market + Reddit + ST':    MARKET_FEATURES + REDDIT_FEATURES + ST_FEATURES,
    'Market + News + ST':      MARKET_FEATURES + NEWS_FEATURES + ST_FEATURES,
    'All sources':             ALL_FEATURES,
}

window_labels = [f'→{w["test"][0]}' for w in WALK_FORWARD_WINDOWS]
print(f'{"Combination":<28}' + ''.join(f'{w:>8}' for w in window_labels) + f'{"Mean":>8}')
print('-' * (28 + 8 * (len(window_labels) + 1)))

layer3_results = {}
for label, features in combinations.items():
    avail = [f for f in features if f in df.columns]
    if not avail:
        continue
    logger.info(f'Walk-forward: {label} ({len(avail)} features)...')
    wf_ics = walk_forward_ic(df, avail)
    valid  = [ic for ic in wf_ics if not np.isnan(ic)]
    mean   = float(np.mean(valid)) if valid else float('nan')
    row    = f'{label:<28}'
    for ic in wf_ics:
        row += f'{"—":>8}' if np.isnan(ic) else f'{ic:>8.4f}'
    row += f'{"—":>8}' if np.isnan(mean) else f'{mean:>8.4f}'
    print(row)
    layer3_results[label] = {
        'features':   avail,
        'n_features': len(avail),
        'wf_ics':     [round(ic, 4) if not np.isnan(ic) else None for ic in wf_ics],
        'mean_ic':    round(mean, 4) if not np.isnan(mean) else None,
        'windows':    window_labels,
    }

# ════════════════════════════════════════════════════════════════════════════
# CONCLUSION
# ════════════════════════════════════════════════════════════════════════════
print()
print('=' * 70)
print('CONCLUSION')
print('=' * 70)

# Load current production model IC for retrain threshold comparison
_meta_path = Path('models/registry/phase3_model_baseline.json')
CURRENT_MODEL_IC = 0.0796  # fallback
if _meta_path.exists():
    with open(_meta_path) as _f:
        _meta = json.load(_f)
    CURRENT_MODEL_IC = _meta.get('horizons', {}).get('5d', {}).get('ic_test', 0.0796)

valid_combos = {k: v for k, v in layer3_results.items() if v['mean_ic'] is not None}
if valid_combos:
    best_combo  = max(valid_combos.items(), key=lambda x: x[1]['mean_ic'])
    best_label  = best_combo[0]
    best_mean   = best_combo[1]['mean_ic']
    market_mean = layer3_results.get('Market only', {}).get('mean_ic') or 0
    improvement = round(best_mean - market_mean, 4) if best_mean and market_mean else 0
else:
    best_label, best_mean, market_mean, improvement = 'N/A', 0, 0, 0

# Retrain threshold compares walk-forward best IC vs current production IC_test
# RULE: only retrain if improvement > 0.005 over current 0.0796 (from CLAUDE.md)
improvement_vs_current = round(best_mean - CURRENT_MODEL_IC, 4) if best_mean else 0

print(f'\nBest combination:  {best_label}')
print(f'Best mean IC:      {best_mean:.4f}')
print(f'Market-only IC:    {market_mean:.4f}')
print(f'Improvement:       {improvement:+.4f}')
print()

print('Source verdicts (Granger):')
for feature, data in layer2_results.items():
    verdict = data['verdict']
    sig     = data['sig_count']
    total   = len(data['results'])
    print(f'  {data["label"]:<35} {verdict} ({sig}/{total} years significant)')

print()
print(f'Improvement vs market-only:        {improvement:+.4f}')
print(f'Current production IC_test (5D):   {CURRENT_MODEL_IC:.4f}')
print(f'Improvement vs current model:      {improvement_vs_current:+.4f}')
print(f'Retrain threshold (> 0.005):       {improvement_vs_current > 0.005}')
print()

if improvement > 0.005:
    recommendation = 'use_external_sources'
    print(f'External sources add value vs market baseline ({improvement:+.4f}).')
else:
    recommendation = 'market_attention_only'
    print('External sources do not consistently improve over market features.')

if improvement_vs_current > 0.005:
    print(f'RETRAIN RECOMMENDED: Best combo ({best_label}) beats current model by {improvement_vs_current:+.4f}.')
else:
    print(f'NO RETRAIN: Best combo IC ({best_mean:.4f}) does not exceed current model IC ({CURRENT_MODEL_IC:.4f}) + 0.005 threshold.')

# ── Save results ─────────────────────────────────────────────────────────
output = {
    'meta': {
        'feature_store':  'data/features/features_complete.parquet',
        'rows_after_gate': int(len(df)),
        'tickers':         int(df['ticker'].nunique()),
        'years':           [int(y) for y in sorted(df['year'].unique())],
        'target':          TARGET,
    },
    'layer1_annual_ic':   layer1_results,
    'layer2_granger':     layer2_results,
    'layer3_walkforward': layer3_results,
    'conclusion': {
        'best_combination':        best_label,
        'best_mean_ic':            best_mean,
        'market_only_ic':          market_mean,
        'current_model_ic':        CURRENT_MODEL_IC,
        'improvement_vs_market':   improvement,
        'improvement_vs_current':  improvement_vs_current,
        'recommendation':          recommendation,
        'retrain_needed':          improvement_vs_current > 0.005,
    },
}

out_path = Path('experiments/source_validation/results.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f'\nSaved: {out_path}')
print('Source validation complete.')
