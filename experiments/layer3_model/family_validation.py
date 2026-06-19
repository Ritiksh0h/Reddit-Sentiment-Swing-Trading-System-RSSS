"""
Layer 3: Signal family validation.
Validates CLEAN_FEATURES (10) vs ORIGINAL_FEATURES (17)
using regime-sliced walk-forward + two-stage permutation test + effective IC.

Adoption gates (ALL must pass for ADOPT):
  R1: Regime gate — statistical (mean positive, 60% sign consistency,
      worst > -0.02, std < 0.08)
  R2: Permutation — Stage A p < 0.10 (required), Stage B p < 0.05 (preferred)
  R3: Effective IC > 0 (coverage × stability × raw_ic − cost)
  R4: Pruned walk-forward mean IC >= original walk-forward mean IC − 0.005
"""
import pandas as pd
import numpy as np
import json, sys
from pathlib import Path

sys.path.insert(0, '.')
from experiments.shared.validation_utils import (
    CLEAN_FEATURES, DROP_TICKERS, XGB_PARAMS, WALK_FORWARD_WINDOWS,
    compute_ic, compute_effective_ic, evaluate_regime_gate,
    run_permutation_test, run_walk_forward,
)

# Config/thresholds override (same values, but allows Phase 3 to import cleanly)
try:
    from config.thresholds import PHASE3_FEATURES, DROP_TICKERS
    CLEAN_FEATURES = PHASE3_FEATURES
except ImportError:
    pass

Path('experiments/layer3_model').mkdir(parents=True, exist_ok=True)

df = pd.read_parquet('data/features/features_expanded.parquet')
df = df[df['post_count_1d'] >= 10].copy()
df = df[~df['ticker'].isin(DROP_TICKERS)].copy()
df['year'] = pd.to_datetime(df['date']).dt.year
print(f'Dataset: {len(df)} rows, {df["ticker"].nunique()} tickers')

ORIGINAL_FEATURES = [
    'returns_1d', 'returns_5d', 'returns_20d', 'rsi_14', 'atr_14',
    'relative_volume', 'dist_from_20ma', 'dist_from_50ma',
    'avg_sentiment_1d', 'avg_sentiment_3d', 'weighted_sentiment',
    'sentiment_std', 'sentiment_accel', 'bullish_ratio',
    'post_count_1d', 'mention_growth_1d', 'mention_growth_7d'
]
ORIGINAL_FEATURES = [f for f in ORIGINAL_FEATURES if f in df.columns]
CLEAN_FEATURES    = [f for f in CLEAN_FEATURES if f in df.columns]

SENTIMENT_FAMILY = [f for f in CLEAN_FEATURES if f in
    ['avg_sentiment_1d', 'sentiment_accel', 'bullish_ratio']]
ATTENTION_FAMILY = [f for f in CLEAN_FEATURES if f in
    ['post_count_1d', 'mention_growth_7d']]

print(f'Original features: {len(ORIGINAL_FEATURES)}')
print(f'Clean features:    {len(CLEAN_FEATURES)}  {CLEAN_FEATURES}')

# ── Walk-forward ───────────────────────────────────────────────────────────
print('\n=== WALK-FORWARD COMPARISON ===')
orig_ics  = run_walk_forward(df, ORIGINAL_FEATURES)
clean_ics = run_walk_forward(df, CLEAN_FEATURES)

print(f'{"Window":<20} {"Original":>10} {"Clean":>10} {"Delta":>8}')
print('-' * 50)
window_names = ['→2022', '→2023', '→2024']
for i, name in enumerate(window_names):
    if i < len(orig_ics) and i < len(clean_ics):
        o = orig_ics[i]
        c = clean_ics[i]
        delta = (c - o) if not (np.isnan(c) or np.isnan(o)) else float('nan')
        o_s = f'{o:.4f}' if not np.isnan(o) else 'N/A'
        c_s = f'{c:.4f}' if not np.isnan(c) else 'N/A'
        d_s = f'{delta:+.4f}' if not np.isnan(delta) else 'N/A'
        print(f'{name:<20} {o_s:>10} {c_s:>10} {d_s:>8}')

# ── R1: Regime gate ────────────────────────────────────────────────────────
print('\n=== R1: REGIME GATE ===')
regime_eval = evaluate_regime_gate(clean_ics)
print(f'Mean IC:          {regime_eval["mean_ic"]:.4f}')
print(f'Sign consistency: {regime_eval["pct_positive"]*100:.0f}%')
print(f'Worst window:     {regime_eval["worst_ic"]:.4f}')
print(f'IC std:           {regime_eval["ic_std"]:.4f}')
for gate, passed in regime_eval['gates'].items():
    print(f'  {"PASS" if passed else "FAIL"}  {gate}')
print(f'R1 VERDICT: {regime_eval["verdict"]}')

# ── R2: Permutation test ───────────────────────────────────────────────────
print('\n=== R2: PERMUTATION TEST (shuffling sentiment + attention families) ===')
train_df = df[df['split'] == 'train']
test_df  = df[df['split'] == 'test']

shuffle_features = SENTIMENT_FAMILY + ATTENTION_FAMILY
perm_result = run_permutation_test(
    train_df, test_df,
    all_features=CLEAN_FEATURES,
    shuffle_features=shuffle_features,
    n_permutations=100,
)
print(f'Real IC:          {perm_result["real_ic"]:.4f}')
print(f'Permuted IC mean: {perm_result["perm_ic_mean"]:.4f} ± {perm_result["perm_ic_std"]:.4f}')
print(f'p-value:          {perm_result["p_value"]:.3f}')
print(f'Stage A (p<0.10): {"PASS" if perm_result["stage_a_pass"] else "FAIL"}  ← required')
print(f'Stage B (p<0.05): {"PASS" if perm_result["stage_b_pass"] else "FAIL"}  ← preferred')

# ── R3: Effective IC ───────────────────────────────────────────────────────
print('\n=== R3: EFFECTIVE IC ===')
eff_ic_result = compute_effective_ic(
    raw_ic=perm_result['real_ic'],
    window_ics=clean_ics,
    n_test=len(test_df),
    n_total=len(df),
)
print(f'Raw IC:            {perm_result["real_ic"]:.4f}')
print(f'Coverage:          {eff_ic_result["coverage"]:.3f}')
print(f'Stability:         {eff_ic_result["stability"]:.3f}')
print(f'Turnover penalty:  {eff_ic_result["turnover_penalty"]:.6f}')
print(f'Effective IC:      {eff_ic_result["effective_ic"]:.4f}')
r3_pass = eff_ic_result['effective_ic'] > 0
print(f'R3 PASS: {r3_pass}  (effective IC > 0)')

# ── R4: Walk-forward mean comparison ──────────────────────────────────────
valid_orig  = [ic for ic in orig_ics  if not np.isnan(ic)]
valid_clean = [ic for ic in clean_ics if not np.isnan(ic)]
orig_mean   = float(np.mean(valid_orig))  if valid_orig  else float('nan')
clean_mean  = float(np.mean(valid_clean)) if valid_clean else float('nan')
r4_pass     = (not np.isnan(clean_mean)) and (not np.isnan(orig_mean)) and \
              (clean_mean >= orig_mean - 0.005)

print(f'\n=== R4: WALK-FORWARD MEAN COMPARISON ===')
print(f'Original mean IC: {orig_mean:.4f}')
print(f'Clean mean IC:    {clean_mean:.4f}')
print(f'Delta:            {clean_mean - orig_mean:+.4f}')
print(f'R4 PASS: {r4_pass}  (clean mean >= original mean − 0.005)')

# ── Final verdict ──────────────────────────────────────────────────────────
r1_pass  = regime_eval['verdict'] in ('ADOPT', 'MARGINAL')
r2a_pass = perm_result['stage_a_pass']
r2b_pass = perm_result['stage_b_pass']

print('\n=== FINAL ADOPTION VERDICT ===')
print(f'R1 Regime gate:       {"PASS" if r1_pass else "FAIL"}  ({regime_eval["verdict"]})')
print(f'R2 Permutation StgA:  {"PASS" if r2a_pass else "FAIL"}  (p={perm_result["p_value"]:.3f})')
print(f'R2 Permutation StgB:  {"PASS" if r2b_pass else "FAIL"}')
print(f'R3 Effective IC:      {"PASS" if r3_pass else "FAIL"}  ({eff_ic_result["effective_ic"]:.4f})')
print(f'R4 WF comparison:     {"PASS" if r4_pass else "FAIL"}')
print()

if r1_pass and r2a_pass and r3_pass and r4_pass:
    if r2b_pass:
        verdict = 'ADOPT'
        note    = 'All gates pass including Stage B permutation. Adopt pruned features.'
    else:
        verdict = 'ADOPT_MARGINAL'
        note    = 'All gates pass but Stage B marginal (p<0.10 only). Adopt with paper trading validation required.'
else:
    verdict = 'REJECT'
    failed  = []
    if not r1_pass:  failed.append('R1_regime')
    if not r2a_pass: failed.append('R2_permutation')
    if not r3_pass:  failed.append('R3_effective_ic')
    if not r4_pass:  failed.append('R4_walkforward')
    note = f'Gates failed: {failed}. Keep original 17 features unchanged.'

print(f'VERDICT: {verdict}')
print(f'NOTE:    {note}')

output = {
    'original_walk_forward_ics': orig_ics,
    'clean_walk_forward_ics':    clean_ics,
    'regime_gate':               regime_eval,
    'permutation_test':          perm_result,
    'effective_ic':              eff_ic_result,
    'r4_walkforward_pass':       bool(r4_pass),
    'orig_mean_ic':              round(orig_mean, 4),
    'clean_mean_ic':             round(clean_mean, 4),
    'verdict':                   verdict,
    'note':                      note,
    'adopted_features':          CLEAN_FEATURES if verdict != 'REJECT' else ORIGINAL_FEATURES,
    'dropped_tickers':           DROP_TICKERS,
}
with open('experiments/layer3_model/family_validation_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print('\nSaved: experiments/layer3_model/family_validation_results.json')
