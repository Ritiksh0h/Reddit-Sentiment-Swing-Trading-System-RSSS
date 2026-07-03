#!/usr/bin/env python3
"""
scripts/debug_report.py

Full system diagnostic: overfit, underfit, signal quality,
walk-forward variance, live signal health.

Usage:  python scripts/debug_report.py
Output: stdout  +  logs/debug_report.txt
Runtime target: < 5 minutes
"""
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from scipy.stats import skew as _skew, kurtosis as _kurtosis

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

BASE   = Path(__file__).resolve().parent.parent
FEAT   = BASE / 'data' / 'features'
MODELS = BASE / 'models'
LOGS   = BASE / 'logs'
LOGS.mkdir(exist_ok=True)

FEATURE_COLS = [
    'post_count_1d',        'abnormal_attention_1d', 'total_comments_1d',
    'vader_sentiment_1d',   'sentiment_extremity',   'sentiment_accel',
    'volume',               'relative_volume',       'returns_1d',
    'returns_20d',          'rsi_14',                'news_sentiment_1d',
    'vix_percentile',       'vix_x_volume',          'spy_above_200ma',
    'regime_score',         'dist_from_20ma_pct',    'pead_proxy',
]

GKX_PARAMS = dict(
    max_depth=1, learning_rate=0.02, objective='reg:pseudohubererror',
    subsample=0.7, colsample_bytree=0.8, reg_alpha=0.1,
    random_state=42, n_jobs=-1,
)
HORIZON_PARAMS = {
    '1d': {'gamma': 0.0,  'reg_lambda': 3.0, 'min_child_weight': 15},
    '3d': {'gamma': 0.05, 'reg_lambda': 1.0, 'min_child_weight': 10},
    '5d': {'gamma': 0.5,  'reg_lambda': 5.0, 'min_child_weight': 20},
}
DENSITY_GATE = 5
TARGETS = {'1d': 'target_return_1d', '3d': 'target_return_3d', '5d': 'target_return_5d'}


# ── helpers ───────────────────────────────────────────────────────────────────
class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self._streams: s.flush()
    def fileno(self): return sys.__stdout__.fileno()


def _ic(pred, actual) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r, _ = spearmanr(pred, actual)
    return float(r) if not np.isnan(r) else 0.0


def _load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    lines = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return lines


def _gated(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_gated, test_gated) after density gate + NaN fill."""
    train = df[df['split'] == 'train']
    test  = df[df['split'] == 'test']
    tg = train[train['post_count_1d'] >= DENSITY_GATE].copy()
    te = test[test['post_count_1d']  >= DENSITY_GATE].copy()
    for d in (tg, te):
        for c in FEATURE_COLS:
            if c not in d.columns:
                d[c] = 0.0
        d[FEATURE_COLS] = d[FEATURE_COLS].fillna(0.0)
    return tg, te


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA HEALTH
# ─────────────────────────────────────────────────────────────────────────────
def section1(df: pd.DataFrame) -> dict:
    print('\n══════════════════════════════════════════════════════')
    print('SECTION 1 — DATA HEALTH')
    print('══════════════════════════════════════════════════════')

    flags: list[str] = []

    # 1.1 ── basic stats
    print('\n── 1.1  Basic stats ─────────────────────────────────')
    train = df[df['split'] == 'train']
    test  = df[df['split'] == 'test']
    live  = df[df['split'] == 'live']
    tg = train[train['post_count_1d'] >= DENSITY_GATE]
    te = test[test['post_count_1d']  >= DENSITY_GATE]
    print(f'  Total rows:    {len(df):,}')
    print(f'  Date range:    {df["date"].min().date()} → {df["date"].max().date()}')
    print(f'  Ticker count:  {df["ticker"].nunique()}')
    print(f'  Train rows:    {len(train):,}  ({train["date"].min().date()} → {train["date"].max().date()})')
    print(f'  Test rows:     {len(test):,}  ({test["date"].min().date()} → {test["date"].max().date()})')
    print(f'  After density gate (post_count_1d >= {DENSITY_GATE}):')
    print(f'    Train gated: {len(tg):,}')
    print(f'    Test gated:  {len(te):,}')
    print(f'  Live rows:     {len(live):,}  ({live["date"].min().date()} → {live["date"].max().date()})')

    # 1.2 ── target distribution
    print('\n── 1.2  Target distribution ─────────────────────────')
    bull_bias = high_vol = False
    for col in ['target_return_1d', 'target_return_3d', 'target_return_5d']:
        s = df[col].dropna()
        pct_pos = (s > 0).mean() * 100
        pct_out = (s.abs() > 0.20).mean() * 100
        print(f'  {col}:')
        print(f'    mean={s.mean():+.4f}  std={s.std():.4f}  '
              f'skew={_skew(s):+.2f}  kurt={_kurtosis(s):.1f}')
        print(f'    % positive={pct_pos:.1f}%  % |ret|>20%={pct_out:.1f}%')
        if pct_pos > 60:
            print('    WARNING: bull bias in training data')
            bull_bias = True
        if s.std() > 0.10:
            print('    WARNING: high return volatility')
            high_vol = True

    # 1.3 ── feature null rates
    print('\n── 1.3  Feature null rates ──────────────────────────')
    null_warn = 0
    for feat in FEATURE_COLS:
        if feat not in df.columns:
            print(f'  WARNING: {feat} missing from parquet')
            null_warn += 1
            continue
        pct = df[feat].isnull().mean() * 100
        if pct > 5:
            print(f'  WARNING: {feat:<28} {pct:.1f}% null')
            null_warn += 1
            flags.append(f'{feat} {pct:.1f}% null')
    if null_warn == 0:
        print('  All features: null rate < 5%  ✓')

    # 1.4 ── feature drift by year
    print('\n── 1.4  Feature drift by year ───────────────────────')
    drift_flags: list[str] = []
    drift_feats = ['post_count_1d', 'vader_sentiment_1d', 'relative_volume']
    for feat in drift_feats:
        yearly = df.groupby('year')[feat].mean()
        mu, sg = float(yearly.mean()), float(yearly.std())
        if sg == 0: sg = 1.0
        print(f'  {feat}:')
        for yr, val in yearly.items():
            flag = ''
            if abs(float(val) - mu) > 2 * sg:
                flag = f'  ← DRIFT'
                drift_flags.append(f'DRIFT: {feat} {int(yr)}')
            print(f'    {int(yr)}: {float(val):+.4f}{flag}')
    if not drift_flags:
        print('  No significant drift detected')
    else:
        for f in drift_flags:
            flags.append(f)

    # 1.5 ── regime class balance
    print('\n── 1.5  Class balance by regime ─────────────────────')
    bull = int((df['spy_above_200ma'] == 1.0).sum())
    bear = int((df['spy_above_200ma'] == 0.0).sum())
    tot  = bull + bear
    print(f'  Bull rows (spy_above_200ma=1): {bull:,} ({bull/tot*100:.1f}%)')
    print(f'  Bear rows (spy_above_200ma=0): {bear:,} ({bear/tot*100:.1f}%)')
    if bear / tot * 100 < 20:
        print('  WARNING: insufficient bear data')
        flags.append('insufficient bear data')

    status = ('FAIL' if null_warn > 2
              else 'WARN' if (bull_bias or high_vol or drift_flags or null_warn > 0)
              else 'PASS')
    return {'status': status, 'flags': flags, 'bull_bias': bull_bias,
            'high_vol': high_vol, 'drift': drift_flags}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — MODEL OVERFIT / UNDERFIT DIAGNOSIS
# ─────────────────────────────────────────────────────────────────────────────
def section2(df: pd.DataFrame) -> dict:
    print('\n══════════════════════════════════════════════════════')
    print('SECTION 2 — MODEL OVERFIT / UNDERFIT DIAGNOSIS')
    print('══════════════════════════════════════════════════════')

    boosters: dict[str, xgb.Booster] = {}
    for hz in ['1d', '3d', '5d']:
        b = xgb.Booster()
        b.load_model(str(MODELS / f'model_{hz}_v2.json'))
        boosters[hz] = b

    tg, te = _gated(df)
    # cache predictions per horizon
    cache: dict[str, dict] = {}
    for hz, target_col in TARGETS.items():
        tr = tg.dropna(subset=[target_col])
        ts = te.dropna(subset=[target_col])
        dtr = xgb.DMatrix(tr[FEATURE_COLS].astype(np.float32))
        dts = xgb.DMatrix(ts[FEATURE_COLS].astype(np.float32))
        p_tr = boosters[hz].predict(dtr)
        p_ts = boosters[hz].predict(dts)
        cache[hz] = dict(
            p_tr=p_tr, p_ts=p_ts,
            y_tr=tr[target_col].values,
            y_ts=ts[target_col].values,
            X_tr=tr, X_ts=ts,
        )

    fit_status   = 'OK'
    overfit_hzs  = []
    underfit_hzs = []
    test_ic_5d   = 0.0

    # 2.1 ── IC gap
    print('\n── 2.1  IC gap (primary overfit signal) ─────────────')
    for hz in ['1d', '3d', '5d']:
        c = cache[hz]
        tr_ic = _ic(c['p_tr'], c['y_tr'])
        ts_ic = _ic(c['p_ts'], c['y_ts'])
        gap   = tr_ic - ts_ic
        if hz == '5d':
            test_ic_5d = ts_ic
        print(f'  Model_{hz.upper()}:  Train IC={tr_ic:+.4f}  '
              f'Test IC={ts_ic:+.4f}  Gap={gap:+.4f}')
        if tr_ic < 0.02 and ts_ic < 0.02:
            print(f'    → UNDERFIT: model not learning signal from features')
            underfit_hzs.append(hz); fit_status = 'UNDERFIT'
        elif tr_ic > 0.10 and gap > 0.05:
            print(f'    → OVERFIT: large gap between train and test IC')
            overfit_hzs.append(hz); fit_status = 'OVERFIT'
        elif gap < 0.02:
            print(f'    → WELL-CALIBRATED: train/test IC close')
        if ts_ic < 0.02:
            print(f'    → WARNING: test IC near zero — model may not generalize')
        elif ts_ic < 0.04:
            print(f'    → MARGINAL: test IC below target threshold of 0.040')
        else:
            print(f'    → PASS: test IC above threshold')

    # 2.2 ── directional accuracy
    print('\n── 2.2  Directional accuracy ────────────────────────')
    for hz in ['1d', '3d', '5d']:
        c = cache[hz]
        tr_dir = float((np.sign(c['p_tr']) == np.sign(c['y_tr'])).mean()) * 100
        ts_dir = float((np.sign(c['p_ts']) == np.sign(c['y_ts'])).mean()) * 100
        print(f'  Model_{hz.upper()}:  Train dir={tr_dir:.1f}%  '
              f'Test dir={ts_dir:.1f}%  (baseline=50.0%)')
        if ts_dir < 52.0:
            print(f'    → WARNING: barely above random')
        if tr_dir - ts_dir > 5.0:
            print(f'    → OVERFIT: direction accuracy drops {tr_dir-ts_dir:.1f}pp OOS')

    # 2.3 ── prediction distribution
    print('\n── 2.3  Prediction distribution ─────────────────────')
    collapsed = biased = False
    for hz in ['1d', '3d', '5d']:
        c = cache[hz]
        p, y = c['p_ts'], c['y_ts']
        print(f'  Model_{hz.upper()} (test set):')
        print(f'    Predictions: mean={p.mean():+.5f}  std={p.std():.5f}  '
              f'min={p.min():+.4f}  max={p.max():+.4f}  '
              f'%>0={float((p>0).mean())*100:.1f}%')
        print(f'    Actuals:     mean={y.mean():+.5f}  std={y.std():.5f}  '
              f'min={y.min():+.4f}  max={y.max():+.4f}  '
              f'%>0={float((y>0).mean())*100:.1f}%')
        if p.std() < 0.005:
            print('    → WARNING: model predicting near-constant values (collapsed)')
            collapsed = True
        if abs(p.mean()) > 0.01:
            print('    → WARNING: systematic bias in predictions')
            biased = True

    # 2.4 ── learning curve (5D only for speed)
    print('\n── 2.4  Learning curve (Model_5D) ───────────────────')
    target_col = TARGETS['5d']
    tr5 = tg.dropna(subset=[target_col]).sort_values('date')
    te5 = te.dropna(subset=[target_col])
    X_te5 = te5[FEATURE_COLS].astype(np.float32)
    y_te5 = te5[target_col].values
    h_params = {**GKX_PARAMS, **HORIZON_PARAMS['5d']}
    n_total  = len(tr5)
    print(f'  {"Size":<6}  {"Train IC":>10}  {"Test IC":>10}')
    print(f'  {"─"*6}  {"─"*10}  {"─"*10}')
    lc_te_ics = []
    for frac in [0.25, 0.50, 0.75, 1.00]:
        n = max(30, int(n_total * frac))
        sub = tr5.iloc[:n]
        X_s = sub[FEATURE_COLS].astype(np.float32)
        y_s = sub[target_col].values.astype(np.float32)
        w_s = (sub['sample_weight'].values.astype(np.float32)
               if 'sample_weight' in sub.columns
               else np.ones(n, dtype=np.float32))
        m = xgb.XGBRegressor(n_estimators=100, **h_params)
        m.fit(X_s, y_s, sample_weight=w_s, verbose=False)
        tr_ic = _ic(m.predict(X_s), y_s)
        ts_ic = _ic(m.predict(X_te5), y_te5)
        lc_te_ics.append(ts_ic)
        print(f'  {frac:.0%}     {tr_ic:>+10.4f}  {ts_ic:>+10.4f}')
    if len(lc_te_ics) >= 3:
        if lc_te_ics[-1] > lc_te_ics[-2]:
            print('  → MORE DATA NEEDED: model not saturated')
        elif abs(lc_te_ics[-1] - lc_te_ics[len(lc_te_ics)//2]) < 0.005:
            print('  → DATA SUFFICIENT: model saturated around 50% of training data')
        elif lc_te_ics[-1] < lc_te_ics[0]:
            print('  → WARNING: model degrades with more data — possible noise contamination')

    # 2.5 ── feature importance stability
    print('\n── 2.5  Feature importance stability ────────────────')
    for hz in ['1d', '3d', '5d']:
        scores = boosters[hz].get_score(importance_type='gain')
        if not scores:
            print(f'  Model_{hz.upper()}: no importance available')
            continue
        imp = pd.Series(scores).sort_values(ascending=False)
        print(f'  Model_{hz.upper()} top 5 (gain):')
        for feat, val in imp.head(5).items():
            print(f'    {feat:<30} {val:>10.1f}')
        if len(imp) >= 2 and imp.iloc[1] > 0:
            ratio = imp.iloc[0] / imp.iloc[1]
            if ratio > 3.0:
                print(f'    → WARNING: single feature dominance '
                      f'(ratio={ratio:.1f}x) — model may be fragile')
                print(f'    → Dominant feature: {imp.index[0]}')

    return {
        'status':      fit_status,
        'overfit':     overfit_hzs,
        'underfit':    underfit_hzs,
        'collapsed':   collapsed,
        'biased':      biased,
        'test_ic_5d':  test_ic_5d,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — WALK-FORWARD VARIANCE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def section3() -> dict:
    print('\n══════════════════════════════════════════════════════')
    print('SECTION 3 — WALK-FORWARD VARIANCE ANALYSIS')
    print('══════════════════════════════════════════════════════')

    wf_path = BASE / 'experiments' / 'walk_forward' / 'results.json'
    sl_path = BASE / 'experiments' / 'walk_forward_sliding' / 'results.json'

    if not wf_path.exists():
        print('  experiments/walk_forward/results.json not found — skipping')
        return {'status': 'MISSING', 'aggregate': {}}

    with open(wf_path) as f:
        wf = json.load(f)
    folds = wf.get('folds', [])
    agg   = wf.get('aggregate', {})

    def _wfe_str(oos_s: float, is_s: float) -> str:
        if abs(is_s) < 1e-6: return 'N/A'
        return f'{oos_s / is_s:+.2f}'

    # 3.1 ── fold-by-fold expanding window
    print('\n── 3.1  Fold-by-fold expanding window ───────────────')
    sharpes: list[float] = []
    for fold in folds:
        oos      = fold.get('oos', {})
        regime   = fold.get('regime', 'unknown')
        oos_ret  = float(oos.get('total_return_pct', 0.0))
        oos_shp  = float(oos.get('sharpe', 0.0))
        is_shp   = float(fold.get('is_sharpe', 0.0))
        wfe      = _wfe_str(oos_shp, is_shp)
        print(f'  Fold {fold["fold"]:>2} [{regime:<22}]:  '
              f'OOS ret={oos_ret:+6.1f}%  OOS Sharpe={oos_shp:+.3f}  '
              f'IS Sharpe={is_shp:+.3f}  WFE={wfe}')
        sharpes.append(oos_shp)

    sharpe_std = float(np.std(sharpes)) if sharpes else 0.0
    wi = int(np.argmin(sharpes)) if sharpes else 0
    bi = int(np.argmax(sharpes)) if sharpes else 0
    print()
    print(f'  Worst fold: Fold {folds[wi]["fold"]} [{folds[wi].get("regime","?")}]  '
          f'Sharpe={sharpes[wi]:+.3f}')
    print(f'  Best fold:  Fold {folds[bi]["fold"]} [{folds[bi].get("regime","?")}]  '
          f'Sharpe={sharpes[bi]:+.3f}')
    print(f'  Sharpe std across folds: {sharpe_std:.3f}')

    wf_status = 'STABLE'
    if sharpe_std > 0.5:
        print('  → HIGH VARIANCE: fold results are unstable')
        wf_status = 'HIGH_VARIANCE'
    if sharpes and min(sharpes) < -0.5:
        print('  → FRAGILE: system loses badly in at least one regime')
        if wf_status == 'STABLE': wf_status = 'FRAGILE'
    if sharpes and all(s > 0.0 for s in sharpes):
        print('  → ROBUST: profitable in every tested regime')

    # 3.2 ── regime breakdown
    print('\n── 3.2  Regime breakdown ────────────────────────────')
    regime_data: dict[str, list] = {}
    for fold in folds:
        r  = fold.get('regime', 'unknown')
        ss = float(fold.get('oos', {}).get('sharpe', 0.0))
        rr = float(fold.get('oos', {}).get('total_return_pct', 0.0))
        regime_data.setdefault(r, {'sharpes': [], 'returns': []})
        regime_data[r]['sharpes'].append(ss)
        regime_data[r]['returns'].append(rr)
    regime_avg = {r: float(np.mean(v['sharpes'])) for r, v in regime_data.items()}
    for r, avg_s in sorted(regime_avg.items(), key=lambda x: x[1]):
        avg_ret = float(np.mean(regime_data[r]['returns']))
        print(f'  {r:<25}: avg Sharpe={avg_s:+.3f}  avg return={avg_ret:+.1f}%')
    weakest   = min(regime_avg, key=regime_avg.get) if regime_avg else 'N/A'
    strongest = max(regime_avg, key=regime_avg.get) if regime_avg else 'N/A'
    print(f'\n  Weakest regime:   {weakest} avg Sharpe={regime_avg.get(weakest, 0):+.3f}')
    print(f'  Strongest regime: {strongest} avg Sharpe={regime_avg.get(strongest, 0):+.3f}')

    # 3.3 ── sliding window variance
    print('\n── 3.3  Sliding window variance (4 folds) ───────────')
    if sl_path.exists():
        with open(sl_path) as f:
            sl = json.load(f)
        sl_folds  = sl.get('folds', [])
        sl_sharpes: list[float] = []
        for fold in sl_folds:
            oos     = fold.get('oos', {})
            oos_ret = float(oos.get('total_return_pct', 0.0))
            oos_shp = float(oos.get('sharpe', 0.0))
            is_shp  = float(fold.get('is_sharpe', 0.0))
            regime  = fold.get('regime', 'unknown')
            print(f'  Fold {fold["fold"]:>2} [{regime:<22}]:  '
                  f'OOS ret={oos_ret:+6.1f}%  OOS Sharpe={oos_shp:+.3f}  '
                  f'IS Sharpe={is_shp:+.3f}  WFE={_wfe_str(oos_shp, is_shp)}')
            sl_sharpes.append(oos_shp)
        if sl_sharpes:
            print(f'  Sharpe range: {min(sl_sharpes):+.3f} to {max(sl_sharpes):+.3f}')
    else:
        print('  walk_forward_sliding/results.json not found')
    print()
    print('  NOTE: 4 folds insufficient for statistical conclusions.')
    print('        DSR requires ~20+ independent folds for p<0.05.')

    # 3.4 ── WFE trend
    print('\n── 3.4  WFE trend (expanding window) ────────────────')
    print('  WFE over time (should be stable > 1.0):')
    fold_map = {f['fold']: f for f in folds}
    for fn in [1, 5, 10, 15, 23]:
        if fn not in fold_map: continue
        fold  = fold_map[fn]
        is_s  = float(fold.get('is_sharpe', 0.0))
        oos_s = float(fold.get('oos', {}).get('sharpe', 0.0))
        print(f'    Fold {fn:>2}:  WFE={_wfe_str(oos_s, is_s)}')
    # Trend diagnosis using valid WFEs
    wfe_vals = []
    for fold in folds:
        is_s  = float(fold.get('is_sharpe', 0.0))
        oos_s = float(fold.get('oos', {}).get('sharpe', 0.0))
        if abs(is_s) > 1e-6:
            wfe_vals.append(oos_s / is_s)
    if len(wfe_vals) >= 4:
        half      = len(wfe_vals) // 2
        early_avg = float(np.mean(wfe_vals[:half]))
        late_avg  = float(np.mean(wfe_vals[half:]))
        if late_avg < early_avg - 0.3:
            print('  → WARNING: model losing edge over time — possible regime shift')
        elif 0.5 <= float(np.mean(wfe_vals)) <= 2.5 and float(np.std(wfe_vals)) < 1.5:
            print('  → STABLE: consistent IS/OOS transfer')

    return {
        'status':        wf_status,
        'sharpe_std':    sharpe_std,
        'weakest':       weakest,
        'strongest':     strongest,
        'aggregate':     agg,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — SIGNAL QUALITY (LIVE SIGNALS)
# ─────────────────────────────────────────────────────────────────────────────
def section4(df: pd.DataFrame) -> dict:
    print('\n══════════════════════════════════════════════════════')
    print('SECTION 4 — SIGNAL QUALITY (LIVE SIGNALS)')
    print('══════════════════════════════════════════════════════')

    signals = _load_jsonl(LOGS / 'all_signals.jsonl')
    trades  = _load_jsonl(LOGS / 'paper_trades.jsonl')

    # 4.1 ── signal count and distribution
    print('\n── 4.1  Signal count and distribution ───────────────')
    if not signals:
        print('  No signals in logs/all_signals.jsonl')
        print('  (Paper trading started June 15 2026 — populates on next live run)')
        if not trades:
            print('  No trades in logs/paper_trades.jsonl either')
        return {'status': 'NO_DATA', 'drift_features': []}

    print(f'  Total signals logged: {len(signals)}')
    sig_types: dict[str, int]   = {}
    ticker_counts: dict[str, int] = {}
    for s in signals:
        st = s.get('signal', 'UNKNOWN')
        sig_types[st] = sig_types.get(st, 0) + 1
        t = s.get('ticker', '?')
        ticker_counts[t] = ticker_counts.get(t, 0) + 1
    for st, n in sorted(sig_types.items()):
        print(f'  {st:<10}: {n} ({n/len(signals)*100:.1f}%)')
    top5 = sorted(ticker_counts.items(), key=lambda x: -x[1])[:5]
    most_freq = top5[0][0] if top5 else 'N/A'
    print(f'  Most frequent ticker: {most_freq} ({ticker_counts.get(most_freq, 0)} times)')
    print(f'  Top tickers: {", ".join(f"{t}({n})" for t, n in top5)}')

    # 4.2 ── prediction distribution (live)
    print('\n── 4.2  Prediction distribution (live signals) ──────')
    preds_5d: list[float] = []
    confs:    list[float] = []
    for s in signals:
        p5   = s.get('predicted_5d', s.get('predicted_return'))
        conf = s.get('confidence')
        if p5   is not None: preds_5d.append(float(p5))
        if conf is not None: confs.append(float(conf))
    if preds_5d:
        p = np.array(preds_5d)
        print(f'  predicted_5d: mean={p.mean():+.5f}  std={p.std():.5f}  '
              f'min={p.min():+.5f}  max={p.max():+.5f}')
        if p.std() < 0.001:
            print('  → WARNING: model predicting near-constant value')
            print('             — check if feature vector is varying day to day')
    if confs:
        c = np.array(confs)
        print(f'  confidence:   mean={c.mean():+.4f}  std={c.std():.4f}  '
              f'min={c.min():+.4f}  max={c.max():+.4f}')
        if c.mean() < 0.30:
            print('  → WARNING: consistently low confidence — model uncertainty high')

    # 4.3 ── feature vector stability (live vs training)
    print('\n── 4.3  Feature vector stability (live vs training) ─')
    train_data = df[df['split'] == 'train'].copy()
    for c in FEATURE_COLS:
        if c not in train_data.columns:
            train_data[c] = 0.0
    train_data[FEATURE_COLS] = train_data[FEATURE_COLS].fillna(0.0)
    t_means = {f: float(train_data[f].mean()) for f in FEATURE_COLS}
    t_stds  = {f: max(float(train_data[f].std()), 1e-6) for f in FEATURE_COLS}

    feat_live: dict[str, list[float]] = {f: [] for f in FEATURE_COLS}
    for s in signals:
        fv = s.get('feature_vector', {})
        for feat in FEATURE_COLS:
            v = fv.get(feat)
            if v is not None:
                feat_live[feat].append(float(v))

    drift_feats: list[str] = []
    for feat in FEATURE_COLS:
        vals = feat_live[feat]
        if not vals: continue
        live_mean = float(np.mean(vals))
        if abs(live_mean - t_means[feat]) > 3 * t_stds[feat]:
            print(f'  DRIFT: {feat:<28} live={live_mean:.4f}  '
                  f'train_mean={t_means[feat]:.4f}')
            drift_feats.append(feat)
    if not drift_feats:
        print('  All features within ±3σ of training distribution  ✓')

    # 4.4 ── MU constant signal analysis
    print('\n── 4.4  MU constant signal analysis ─────────────────')
    mu_sigs = [s for s in signals if s.get('ticker') == 'MU']
    if len(mu_sigs) < 2:
        print(f'  MU appearances: {len(mu_sigs)} — need ≥2 consecutive runs to check')
    else:
        watch = ['vader_sentiment_1d', 'post_count_1d', 'relative_volume', 'atr_14']
        frozen = 0
        for feat in watch:
            vals = [s.get('feature_vector', {}).get(feat) for s in mu_sigs]
            vals = [v for v in vals if v is not None]
            if not vals:
                # try top-level field (atr_14 is stored top-level)
                vals = [s.get(feat) for s in mu_sigs if s.get(feat) is not None]
            if not vals:
                print(f'  {feat}: no data')
                continue
            unique = len(set(round(v, 4) for v in vals))
            if unique == 1:
                print(f'  {feat}: CONSTANT = {vals[0]}')
                frozen += 1
            else:
                print(f'  {feat}: varies {min(vals):.4f} – {max(vals):.4f}')
        if frozen > 3:
            print('  → WARNING: feature vector is frozen — check live data pipeline')
        else:
            print('  → OK: features updating correctly, prediction stable due to model')

    # 4.5 ── MA filter impact
    print('\n── 4.5  MA filter impact ─────────────────────────────')
    if trades:
        blocked  = [t for t in trades if t.get('event') == 'ma_filter_skip'
                    or 'ma_filter' in str(t.get('type', ''))]
        print(f'  Signals blocked by MA filter: {len(blocked)}/{len(trades)} '
              f'({len(blocked)/max(len(trades),1)*100:.1f}%)')
        if blocked:
            blocked_tickers = {}
            for t in blocked:
                tk = t.get('ticker', '?')
                blocked_tickers[tk] = blocked_tickers.get(tk, 0) + 1
            top_blocked = sorted(blocked_tickers.items(), key=lambda x: -x[1])[:3]
            print(f'  Most blocked: {", ".join(f"{t}({n})" for t, n in top_blocked)}')
    else:
        print('  No trade log data available for MA filter analysis')

    sq_status = 'DEGRADED' if drift_feats else 'HEALTHY'
    return {'status': sq_status, 'drift_features': drift_feats}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — ICEARLYSTOPPING BUG CHECK
# ─────────────────────────────────────────────────────────────────────────────
class _ICEarlyStopping(xgb.callback.TrainingCallback):
    def __init__(self, rounds: int, X_eval, y_eval) -> None:
        super().__init__()
        self.rounds          = rounds
        self.X_eval          = X_eval.values if hasattr(X_eval, 'values') else X_eval
        self.y_eval          = np.asarray(y_eval)
        self.best_ic         = -np.inf
        self.best_iteration  = 0
        self._no_improve     = 0

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        pred = model.inplace_predict(self.X_eval)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ic_raw, _ = spearmanr(pred, self.y_eval)
        ic = float(ic_raw) if not np.isnan(ic_raw) else -1.0
        if ic > self.best_ic:
            self.best_ic, self.best_iteration, self._no_improve = ic, epoch, 0
        else:
            self._no_improve += 1
        return self._no_improve >= self.rounds


def section5(df: pd.DataFrame) -> dict:
    print('\n══════════════════════════════════════════════════════')
    print('SECTION 5 — ICEARLYSTOPPING BUG CHECK')
    print('══════════════════════════════════════════════════════')
    print()
    print('  Context: walk_forward_validation.py uses X_eval=X_train (in-sample).')
    print('  train_models_v2.py uses X_eval=X_test (correct for production).')
    print('  This section quantifies the bias in the walk-forward scout phase.')

    tg, te = _gated(df)
    tr5    = tg.dropna(subset=['target_return_5d']).sort_values('date').copy()
    te5    = te.dropna(subset=['target_return_5d']).copy()
    n      = len(tr5)
    n_val  = int(n * 0.20)

    X_full = tr5[FEATURE_COLS].astype(np.float32)
    y_full = tr5['target_return_5d'].values.astype(np.float32)
    w_full = (tr5['sample_weight'].values.astype(np.float32)
              if 'sample_weight' in tr5.columns
              else np.ones(n, dtype=np.float32))

    X_80, y_80, w_80 = X_full.iloc[:-n_val], y_full[:-n_val], w_full[:-n_val]
    X_val, y_val      = X_full.iloc[-n_val:], y_full[-n_val:]
    X_te5 = te5[FEATURE_COLS].astype(np.float32)
    y_te5 = te5['target_return_5d'].values.astype(np.float32)

    h_params = {**GKX_PARAMS, **HORIZON_PARAMS['5d']}

    # Scout A — in-sample early stopping (walk_forward_validation.py behavior)
    st_a = _ICEarlyStopping(rounds=20, X_eval=X_80, y_eval=y_80)
    sc_a = xgb.XGBRegressor(n_estimators=200, **h_params, callbacks=[st_a])
    sc_a.fit(X_80, y_80, sample_weight=w_80,
             eval_set=[(X_80, y_80)], verbose=False)
    best_n_a = max(1, st_a.best_iteration + 1)
    m_a = xgb.XGBRegressor(n_estimators=best_n_a, **h_params)
    m_a.fit(X_80, y_80, sample_weight=w_80, verbose=False)
    ic_a = _ic(m_a.predict(X_te5), y_te5)

    # Scout B — validation early stopping (correct behavior)
    st_b = _ICEarlyStopping(rounds=20, X_eval=X_val, y_eval=y_val)
    sc_b = xgb.XGBRegressor(n_estimators=200, **h_params, callbacks=[st_b])
    sc_b.fit(X_80, y_80, sample_weight=w_80,
             eval_set=[(X_val, y_val)], verbose=False)
    best_n_b = max(1, st_b.best_iteration + 1)
    m_b = xgb.XGBRegressor(n_estimators=best_n_b, **h_params)
    m_b.fit(X_80, y_80, sample_weight=w_80, verbose=False)
    ic_b = _ic(m_b.predict(X_te5), y_te5)

    print('\n── 5.1  Quantify in-sample early stopping bias ──────')
    print(f'  Scout A (train data — walk_fwd behavior):  '
          f'best_n={best_n_a:>3}  test IC={ic_a:+.4f}')
    print(f'  Scout B (validation data — correct):       '
          f'best_n={best_n_b:>3}  test IC={ic_b:+.4f}')

    bias = ic_a - ic_b
    biased = bias > 0.002
    if biased:
        print(f'  → CONFIRMED: in-sample early stopping inflates IC by {bias:.4f}')
    else:
        print('  → OK: in-sample early stopping has minimal impact')
        print('        (likely because max_depth=1 stumps prevent overfitting anyway)')

    print('\n── 5.2  Recommended fix (print only) ────────────────')
    print('  To fix ICEarlyStopping, pass a held-out validation set:')
    print('    stopper = ICEarlyStopping(rounds=20,')
    print('                              X_eval=X_val,')
    print('                              y_eval=y_val)')
    print('    where X_val = last 20% of training data (chronologically)')
    print('    This gives honest early stopping without future leakage.')

    return {'status': 'BIASED' if biased else 'MINIMAL_IMPACT',
            'ic_a': ic_a, 'ic_b': ic_b}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────
def section6(s1: dict, s2: dict, s3: dict, s4: dict, s5: dict) -> None:
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print()
    print('════════════════════════════════════════════════════════')
    print(f'RSSS DIAGNOSTIC REPORT — {ts}')
    print('════════════════════════════════════════════════════════')
    print()
    print(f'  DATA HEALTH:       {s1["status"]}')
    print(f'  MODEL FIT:         {s2["status"]}')
    print(f'  WALK-FORWARD:      {s3["status"]}')
    print(f'  SIGNAL QUALITY:    {s4["status"]}')
    print(f'  ICEARLYSTOPPING:   {s5["status"]}')
    print()

    findings: list[str] = []
    if s2.get('overfit'):
        findings.append(f'Model overfit on horizons {", ".join(s2["overfit"])} '
                        f'(train/test IC gap > 0.05)')
    if s2.get('underfit'):
        findings.append(f'Model underfit on horizons {", ".join(s2["underfit"])} '
                        f'(train IC and test IC both < 0.02)')
    if s2.get('collapsed'):
        findings.append('Prediction collapse detected — model outputs near-constant values')
    if s3['sharpe_std'] > 0.5:
        findings.append(f'Walk-forward high variance (Sharpe σ={s3["sharpe_std"]:.2f}) '
                        f'— unstable across regimes')
    agg = s3.get('aggregate', {})
    if agg.get('bear_2022_return', 0) < -20:
        findings.append(f'Bear 2022 return={agg["bear_2022_return"]:.1f}% — below -20% gate')
    if s5['status'] == 'BIASED':
        delta = s5['ic_a'] - s5['ic_b']
        findings.append(f'walk_forward_validation.py ICEarlyStopping evaluates on training '
                        f'data — inflates scout IC by {delta:.4f}')
    if s4.get('drift_features'):
        findings.append(f'Live feature drift: {", ".join(s4["drift_features"][:3])}')
    if s4['status'] == 'NO_DATA':
        findings.append('No live signals logged yet — pipeline not yet producing output')
    if s1.get('drift'):
        findings.append(f'Historical feature drift detected: {", ".join(s1["drift"][:2])}')
    if not findings:
        findings.append('No major issues detected — system appears healthy')

    print('  KEY FINDINGS:')
    for i, f in enumerate(findings[:5], 1):
        print(f'    {i}. {f}')
    print()

    actions: list[str] = []
    if s5['status'] == 'BIASED':
        actions.append(
            'Fix walk_forward_validation.py ICEarlyStopping: '
            'pass X_val (last 20% of train) instead of X (full train)')
    if s2.get('overfit'):
        actions.append(
            f'Increase reg_lambda or min_child_weight for overfit horizons '
            f'({", ".join(s2["overfit"])})')
    if s2.get('underfit'):
        actions.append(
            f'Reduce regularisation or add features for underfit horizons '
            f'({", ".join(s2["underfit"])})')
    if s3['status'] in ('HIGH_VARIANCE', 'FRAGILE'):
        actions.append(
            'Run Part B source validation to identify regime-stable features '
            '(experiments/source_validation/validate_sources.py)')
    if s4['status'] == 'NO_DATA':
        actions.append(
            'Verify live pipeline: '
            'launchctl list | grep rsss  and  tail -20 logs/daily_runs.log')
    if s1.get('drift'):
        actions.append(
            'Investigate Reddit API drift in post_count_1d — '
            'check Arctic Shift API consistency across years')
    if not actions:
        actions.append('No fixes required — continue monitoring live IC')

    print('  RECOMMENDED ACTIONS:')
    for i, a in enumerate(actions[:3], 1):
        print(f'    {i}. {a}')
    print()

    test_ic_5d = s2.get('test_ic_5d', 0.0)
    pooled_shp = agg.get('pooled_sharpe', 0.0)
    print('  NEXT RETRAIN RECOMMENDATION:')
    if s4['status'] == 'NO_DATA':
        print('    Timing:  WAIT UNTIL SEP 2026 — need 30+ days of live IC first')
    elif test_ic_5d < 0.02 or pooled_shp < 0.3:
        print('    Timing:  NOW — IC or Sharpe below minimum threshold')
    else:
        print('    Timing:  WAIT UNTIL SEP 2026 — model above IC threshold')
    print('    Method:  EXPANDING window + re-run full walk-forward validation')
    if s5['status'] == 'BIASED':
        print('    Changes: Fix ICEarlyStopping eval set before retraining; '
              'then re-evaluate IC gate (must beat 0.0796 + 0.005)')
    else:
        print('    Changes: Confirm IC improvement > 0.005 over 0.0796 before deploying')
    print()
    print('════════════════════════════════════════════════════════')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    start    = datetime.now()
    out_path = LOGS / 'debug_report.txt'

    log_file = open(out_path, 'w', encoding='utf-8')
    orig_stdout = sys.stdout
    sys.stdout  = _Tee(orig_stdout, log_file)

    try:
        print('╔══════════════════════════════════════════════════════╗')
        print('║   RSSS DEBUG REPORT — starting                       ║')
        print('╚══════════════════════════════════════════════════════╝')
        print(f'  Generated: {start.strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'  Output:    {out_path}')

        df = pd.read_parquet(FEAT / 'features_v2.parquet')
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0

        s1 = section1(df)
        s2 = section2(df)
        s3 = section3()
        s4 = section4(df)
        s5 = section5(df)
        section6(s1, s2, s3, s4, s5)

        elapsed = (datetime.now() - start).total_seconds()
        print(f'\n  Runtime: {elapsed:.1f}s')
        print(f'  Report:  {out_path}')

    finally:
        sys.stdout = orig_stdout
        log_file.close()


if __name__ == '__main__':
    main()
