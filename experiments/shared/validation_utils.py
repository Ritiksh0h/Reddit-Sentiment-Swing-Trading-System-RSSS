"""
Shared validation utilities for the signal validation sprint.
All three layers import from here.
"""
import numpy as np
import pandas as pd
from scipy import stats
import xgboost as xgb

# ── Constants ──────────────────────────────────────────────────────────────
ANNUAL_TRADES       = 60          # observed from Experiment C
COST_PER_ROUND_TRIP = 0.002       # 0.1% per leg × 2
DENSITY_GATE        = 10          # post_count_1d minimum

XGB_PARAMS = dict(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1
)

WALK_FORWARD_WINDOWS = [
    ('2019-2021', [2019, 2020, 2021], '2022', [2022]),
    ('2020-2022', [2020, 2021, 2022], '2023', [2023]),
    ('2021-2023', [2021, 2022, 2023], '2024', [2024]),
]

# Clean feature set — imported by layer3.  Source of truth is config/thresholds.py;
# defined here as fallback so layers work without a full project install.
try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from config.thresholds import PHASE3_FEATURES, DROP_TICKERS
    CLEAN_FEATURES = PHASE3_FEATURES
except Exception:
    CLEAN_FEATURES = [
        'returns_1d', 'returns_5d', 'relative_volume', 'dist_from_20ma', 'dist_from_50ma',
        'avg_sentiment_1d', 'sentiment_accel', 'bullish_ratio',
        'post_count_1d', 'mention_growth_7d',
    ]
    DROP_TICKERS = ['ASTS', 'LCID', 'MSTR', 'RIOT', 'RIVN', 'SMCI', 'WMT']


# ── IC computation ─────────────────────────────────────────────────────────
def compute_ic(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Spearman rank correlation. Returns NaN if insufficient data."""
    if len(y_pred) < 20:
        return float('nan')
    ic, _ = stats.spearmanr(y_pred, y_true)
    return float(ic)


# ── Effective IC ───────────────────────────────────────────────────────────
def compute_effective_ic(
    raw_ic: float,
    window_ics: list,
    n_test: int,
    n_total: int,
) -> dict:
    """
    Compute Effective IC — a regime-robust, cost-adjusted signal quality score.

    Formula:
        effective_ic = (raw_ic × coverage × stability) − turnover_penalty

    Components:
        coverage:         fraction of test rows that pass density gate
        stability:        cross-window IC consistency (0=unstable, 1=stable)
        turnover_penalty: estimated annual cost drag mapped to IC scale
    """
    if not window_ics or len(window_ics) < 2:
        return {'effective_ic': float('nan'), 'coverage': float('nan'),
                'stability': float('nan'), 'turnover_penalty': float('nan')}

    coverage = min(n_test / max(n_total, 1), 1.0)

    ic_array = np.array([ic for ic in window_ics if not np.isnan(ic)])
    if len(ic_array) == 0:
        stability = 0.0
    else:
        ic_std  = float(np.std(ic_array))
        ic_mean = float(np.mean(np.abs(ic_array)))
        if ic_mean < 1e-6:
            stability = 0.0
        else:
            stability = float(np.clip(1.0 - (ic_std / ic_mean), 0.0, 1.0))

    annual_cost      = ANNUAL_TRADES * COST_PER_ROUND_TRIP
    turnover_penalty = annual_cost / 252 * 5
    turnover_penalty = float(np.clip(turnover_penalty, 0, 0.05))

    effective_ic = float((raw_ic * coverage * stability) - turnover_penalty)

    return {
        'effective_ic':    effective_ic,
        'coverage':        round(coverage, 4),
        'stability':       round(stability, 4),
        'turnover_penalty': round(turnover_penalty, 6),
    }


# ── Regime gate ────────────────────────────────────────────────────────────
def evaluate_regime_gate(window_ics: list) -> dict:
    """
    Statistical regime gate — replaces boolean 'N of M windows' check.

    Gates:
        mean_positive:      mean IC across windows > 0
        sign_consistency:   >= 60% of windows have positive IC
        worst_bounded:      worst window IC > -0.02
        variance_stable:    std of window ICs < 0.08
    """
    valid = [ic for ic in window_ics if not np.isnan(ic)]
    if len(valid) < 2:
        return {'verdict': 'INSUFFICIENT_DATA', 'gates': {}}

    mean_ic      = float(np.mean(valid))
    pct_positive = sum(1 for ic in valid if ic > 0) / len(valid)
    worst_ic     = float(np.min(valid))
    ic_std       = float(np.std(valid))

    gates = {
        'mean_ic_positive':     mean_ic > 0,
        'sign_consistency_60':  pct_positive >= 0.60,
        'worst_window_gt_m002': worst_ic > -0.02,
        'variance_below_008':   ic_std < 0.08,
    }

    n_pass = sum(gates.values())
    if n_pass == 4:
        verdict = 'ADOPT'
    elif n_pass >= 3:
        verdict = 'MARGINAL'
    else:
        verdict = 'REJECT'

    return {
        'verdict':      verdict,
        'mean_ic':      round(mean_ic, 4),
        'pct_positive': round(pct_positive, 3),
        'worst_ic':     round(worst_ic, 4),
        'ic_std':       round(ic_std, 4),
        'gates':        {k: bool(v) for k, v in gates.items()},
    }


# ── Permutation test ───────────────────────────────────────────────────────
def run_permutation_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    all_features: list,
    shuffle_features: list,
    target_col: str = 'target_return_5d',
    n_permutations: int = 100,
    seed: int = 42,
) -> dict:
    """
    Permutation test: shuffle shuffle_features in training set.
    Measures whether those features add real signal or are noise.
    """
    avail = [f for f in all_features if f in train_df.columns]

    m_real = xgb.XGBRegressor(**XGB_PARAMS)
    m_real.fit(train_df[avail].fillna(0), train_df[target_col])
    pred_real = m_real.predict(test_df[avail].fillna(0))
    real_ic = compute_ic(pred_real, test_df[target_col].values)

    rng = np.random.RandomState(seed)
    perm_ics = []
    for _ in range(n_permutations):
        t_perm = train_df.copy()
        for col in shuffle_features:
            if col in t_perm.columns:
                t_perm[col] = rng.permutation(t_perm[col].values)
        m_p = xgb.XGBRegressor(**XGB_PARAMS)
        m_p.fit(t_perm[avail].fillna(0), t_perm[target_col])
        p_pred = m_p.predict(test_df[avail].fillna(0))
        perm_ics.append(compute_ic(p_pred, test_df[target_col].values))

    perm_arr = np.array(perm_ics)
    p_value  = float((perm_arr >= real_ic).mean())

    return {
        'real_ic':        round(real_ic, 4),
        'perm_ic_mean':   round(float(perm_arr.mean()), 4),
        'perm_ic_std':    round(float(perm_arr.std()), 4),
        'p_value':        round(p_value, 3),
        'stage_a_pass':   bool(p_value < 0.10),
        'stage_b_pass':   bool(p_value < 0.05),
        'n_permutations': n_permutations,
    }


# ── Walk-forward runner ────────────────────────────────────────────────────
def run_walk_forward(
    df: pd.DataFrame,
    features: list,
    target_col: str = 'target_return_5d',
    year_col: str = 'year',
) -> list:
    """
    Run walk-forward IC on WALK_FORWARD_WINDOWS.
    Returns list of ICs, one per window.
    """
    ics = []
    for train_label, train_years, test_label, test_years in WALK_FORWARD_WINDOWS:
        train = df[df[year_col].isin(train_years)]
        test  = df[df[year_col].isin(test_years)]
        avail = [f for f in features if f in train.columns]
        if len(train) < 100 or len(test) < 30:
            ics.append(float('nan'))
            continue
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(train[avail].fillna(0), train[target_col])
        pred = m.predict(test[avail].fillna(0))
        ics.append(compute_ic(pred, test[target_col].values))
    return ics
