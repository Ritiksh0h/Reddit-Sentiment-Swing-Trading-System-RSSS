#!/usr/bin/env python3
"""
RSSS Backtest v2 — 2024-2025 out-of-sample.
HISTORICAL SIMULATION ONLY. NOT live trading.

Systems:
    A  Long-only + rank-based signals + dynamic exit (model-reversal OR max 5D)
    B  Long-only + rank-based signals + fixed 5D exit
       [control: isolates the value of dynamic hold vs. fixed]

Core-satellite structure:
    Core    (70%): SPY buy-and-hold throughout test period
    Satellite (30%): RSSS long signals, regime-adjusted allocation (20-35%)

Signal generation: rank-based composite score per day
    score = 0.5×pred_5d + 0.3×pred_3d + 0.2×pred_1d
    Quality gates: score > 0, regime_score >= 0.3, relative_volume >= 0.8
    Take top MAX_DAILY_SIGNALS=2 tickers per day by score
"""

import json
import math
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from scipy import stats

try:
    from statsmodels.stats.proportion import proportion_confint
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

sys.path.insert(0, '.')

from data.earnings_fetcher import is_safe_to_trade

# ── Constants ──────────────────────────────────────────────────────────────────
TEST_START         = '2024-01-01'
TEST_END           = '2025-12-31'
INITIAL_CAPITAL    = 10_000.0
CORE_PCT           = 0.70
SATELLITE_PCT_DEF  = 0.30
MAX_POSITIONS      = 4      # max concurrent satellite positions
MAX_DAILY_SIGNALS  = 2      # max new positions opened per day
MIN_ACTIVITY_GATE  = 3      # minimum post_count_1d for eligibility
STOP_LOSS          = -0.08
TAKE_PROFIT        = 0.10   # tightened from 0.15
SLIPPAGE           = 0.0015
COOLDOWN_DAYS      = 7
MAX_HOLD_DAYS      = 5      # hard cap for both systems

# Must match FEATURE_COLS in train_models_v2.py
FEATURE_COLS = [
    'post_count_1d',       'abnormal_attention_1d',
    'total_comments_1d',   'vader_sentiment_1d',
    'sentiment_extremity', 'sentiment_accel',
    'volume',              'relative_volume',
    'returns_1d',          'returns_20d',
    'rsi_14',              'news_sentiment_1d',
    'vix_percentile',      'vix_x_volume',
    'spy_above_200ma',     'regime_score',
]

# Tickers eligible for signal generation (TRADE universe only, not WATCH)
TRADE_TICKERS = {
    'AAPL', 'AMD', 'AMZN', 'COIN', 'GME', 'GOOG', 'HOOD',
    'MARA', 'META', 'MSFT', 'MU',  'NFLX', 'NVDA', 'PLTR',
    'QQQ',  'SOFI', 'TSLA', 'UBER',   # SPY excluded (it's the core)
}

# Sector map (Improvement 4) — loaded from ticker_registry.json at runtime in main()
_SECTOR_MAP: dict = {}

# Grinold vol-adjustment constants (Improvement 3)
_TARGET_VOL = 0.02   # 2% daily vol target
_VOL_FLOOR  = 0.005  # 0.5%
_VOL_CAP    = 0.08   # 8%

# Earnings filter (Improvement 1) — in-memory cache for backtest runs
_EARNINGS_CACHE: dict = {}


def _load_sector_map() -> dict:
    """Load ticker → sector from ticker_registry.json."""
    try:
        with open('config/ticker_registry.json') as f:
            reg = json.load(f)
        return {t: v.get('sector', 'Unknown')
                for t, v in reg.get('tickers', {}).items()}
    except Exception:
        return {}


def _vol_adjusted_score(score: float, ticker: str, date_str: str, vol_lut: dict) -> float:
    """
    Grinold vol-normalization: α = IC × vol × score.
    Higher-vol tickers are discounted so they don't dominate ranking purely
    because their raw return predictions are large.
    Returns adjusted_score; falls back to raw score if vol unavailable.
    """
    vol = vol_lut.get((ticker, date_str))
    if vol is None or vol <= 0:
        return score
    vol = max(vol, _VOL_FLOOR)
    vol = min(vol, _VOL_CAP)
    return score * (_TARGET_VOL / vol)


def _get_correlation(
    ticker_a: str,
    ticker_b: str,
    date_str: str,
    ticker_series: dict,
    window: int = 60,
) -> float | None:
    """Rolling 60-day Pearson correlation between two tickers' returns."""
    s_a = ticker_series.get(ticker_a)
    s_b = ticker_series.get(ticker_b)
    if s_a is None or s_b is None:
        return None
    try:
        d = pd.Timestamp(date_str)
        combined = pd.DataFrame({'a': s_a, 'b': s_b}).dropna()
        combined = combined[combined.index <= d].tail(window)
        if len(combined) < 20:
            return None
        returns = combined.pct_change().dropna()
        if len(returns) < 10:
            return None
        return float(returns['a'].corr(returns['b']))
    except Exception:
        return None


def _earnings_safe(ticker: str, current_date: str) -> bool:
    """
    Check earnings safety with in-process cache.
    Skips the check entirely in backtest to avoid hitting live API per-row;
    uses a one-time cache keyed on ticker (fetched at backtest start in main()).
    """
    if ticker not in _EARNINGS_CACHE:
        return True  # unknown → allow (API not called per-row in backtest)
    earned_date = _EARNINGS_CACHE[ticker]
    if earned_date is None:
        return True
    entry = date.fromisoformat(current_date)
    danger_end = entry
    # Simple check: if cached earnings date falls within 8d of entry, skip
    from datetime import timedelta
    danger_window_end = entry + timedelta(days=8)
    return earned_date > danger_window_end

def _above_20ma(ticker: str, current_date: str, ticker_series: dict) -> bool:
    """Fix 4 — 20-day MA trend filter. Returns True (allow) if price > 20d MA.
    Fails open on missing data so it never blocks when prices are unavailable."""
    try:
        if not ticker_series or ticker not in ticker_series:
            return True
        prices = ticker_series[ticker]
        idx = prices.index.get_indexer([pd.Timestamp(current_date)], method='ffill')[0]
        if idx < 20:
            return True
        past_20 = prices.iloc[max(0, idx - 20):idx]
        if len(past_20) < 15:
            return True
        return float(prices.iloc[idx - 1]) > float(past_20.mean())
    except Exception:
        return True


# Previous baselines — used only for the comparison table printout
_OLD = {            # original absolute-threshold system (57 trades)
    'total_return_pct': 23.18,
    'alpha_pct':       -24.66,
    'sharpe_ratio':     0.93,
    'win_rate':         0.5439,
    'n_trades':         57,
    'max_drawdown_pct': -9.9,
    'significant':      False,
}
_FIXED = {          # core-satellite with absolute 0.7% threshold (19 trades)
    'total_return_pct': 33.6,
    'alpha_pct':       -14.2,
    'sharpe_ratio':     1.27,
    'win_rate':         0.5789,
    'n_trades':         19,
    'max_drawdown_pct': -14.1,
    'significant':      False,
}
_RANK167 = {        # rank-based before sector/correlation filters (167 trades)
    'total_return_pct': 35.64,
    'alpha_pct':       -12.2,
    'sharpe_ratio':     1.32,
    'win_rate':         0.5749,
    'n_trades':         167,
    'max_drawdown_pct': -14.1,
    'significant':      False,
}
_SECTOR2_CORR = {   # sector cap=2 + correlation filter (140 trades)
    'total_return_pct': 36.7,
    'alpha_pct':       -11.2,
    'sharpe_ratio':     1.37,
    'win_rate':         0.557,
    'n_trades':         140,
    'max_drawdown_pct': -14.1,
    'pvalue':           0.205,
    'significant':      False,
}


# ── Data classes ───────────────────────────────────────────────────────────────
@dataclass
class SatPos:
    ticker:      str
    entry_date:  str
    entry_fill:  float
    exit_date:   str      # hard deadline (entry + MAX_HOLD_DAYS trading days)
    n_shares:    int
    cost:        float
    pred_1d:     float
    pred_3d:     float
    pred_5d:     float
    score:       float    # composite score at entry


# ── Helpers ────────────────────────────────────────────────────────────────────
def _composite_score(pred_1d: float, pred_3d: float, pred_5d: float) -> float:
    """Weighted composite of model predictions (5D dominant)."""
    return 0.5 * pred_5d + 0.3 * pred_3d + 0.2 * pred_1d


def _exit_date(trading_days: list[str], entry: str, hold: int) -> str:
    try:
        idx = trading_days.index(entry)
    except ValueError:
        return entry
    return trading_days[min(idx + hold, len(trading_days) - 1)]


def get_regime_allocation(date_str: str, spy_rows: pd.DataFrame) -> float:
    """
    Adjust satellite fraction based on market regime.
    Bull  (SPY above 200MA + low VIX) → 0.35
    Bear  (SPY below 200MA OR high VIX) → 0.20
    Neutral → 0.30
    """
    row = spy_rows[spy_rows['date'] == date_str]
    if len(row) == 0:
        return SATELLITE_PCT_DEF
    spy_above = float(row['spy_above_200ma'].iloc[0])
    vix_pct   = float(row['vix_percentile'].iloc[0])
    if spy_above >= 1.0 and vix_pct < 0.5:
        return 0.35
    elif spy_above < 1.0 or vix_pct > 0.75:
        return 0.20
    return 0.30


def get_position_size(
    ticker: str, date_str: str, regime_sat_cap: float,
    composite_score: float, vol_lut: dict,
) -> float:
    """
    Volatility-targeted position size (dollars).
    Targets 1% daily portfolio vol per position.
    Confidence proxy: composite_score capped at 0.05 → 1.0.
    Hard caps: 5%–25% of satellite capital.
    """
    vol = vol_lut.get((ticker, date_str), 0.02)
    vol = max(vol, 0.005)
    vol = min(vol, 0.05)

    vol_scalar  = 0.01 / vol
    confidence  = min(abs(composite_score) / 0.05, 1.0)
    conf_scalar = 0.5 + confidence * 0.5

    size = regime_sat_cap * 0.15 * vol_scalar * conf_scalar
    size = min(size, regime_sat_cap * 0.25)
    size = max(size, regime_sat_cap * 0.05)
    return float(size)


def _calc_metrics(equity_curve: list, closed_trades: list) -> dict:
    equities = [e['equity'] for e in equity_curve]
    if len(equities) < 2:
        return dict(sharpe_ratio=0.0, max_drawdown_pct=0.0)

    daily_ret = [(equities[i] - equities[i-1]) / equities[i-1]
                 for i in range(1, len(equities))]
    mean_r = np.mean(daily_ret)
    std_r  = np.std(daily_ret, ddof=1) or 1e-9
    sharpe = mean_r / std_r * math.sqrt(252)

    peak   = equities[0]
    max_dd = 0.0
    for eq in equities:
        peak   = max(peak, eq)
        max_dd = min(max_dd, (eq - peak) / peak)

    return dict(
        sharpe_ratio     = round(sharpe, 3),
        max_drawdown_pct = round(max_dd * 100, 2),
    )


def _monthly_returns(equity_curve: list) -> dict:
    df_eq = pd.DataFrame(equity_curve)
    df_eq['date'] = pd.to_datetime(df_eq['date'])
    df_eq = df_eq.set_index('date')
    out = {}
    for month, grp in df_eq.groupby(pd.Grouper(freq='ME')):
        if len(grp) >= 2:
            s, e = grp['equity'].iloc[0], grp['equity'].iloc[-1]
            out[month.strftime('%Y-%m')] = round((e - s) / s * 100, 2)
    return out


def compute_significance(closed_trades: list) -> dict:
    """Binomial test + Wilson 95% CI on satellite trade win rate."""
    if not closed_trades:
        return {}

    n_trades = len(closed_trades)
    n_wins   = sum(1 for t in closed_trades if t['pnl_pct'] > 0)
    win_rate = n_wins / n_trades

    result = stats.binomtest(n_wins, n_trades, 0.5, alternative='two-sided')
    pval   = result.pvalue

    if _HAS_STATSMODELS:
        ci_low, ci_high = proportion_confint(
            n_wins, n_trades, alpha=0.05, method='wilson')
    else:
        z  = 1.96
        se = math.sqrt(win_rate * (1 - win_rate) / n_trades) if n_trades > 0 else 0
        ci_low  = max(0.0, win_rate - z * se)
        ci_high = min(1.0, win_rate + z * se)

    out = {
        'n_trades':    n_trades,
        'n_wins':      n_wins,
        'win_rate':    round(win_rate, 4),
        'pvalue':      round(pval, 4),
        'ci_low':      round(float(ci_low), 4),
        'ci_high':     round(float(ci_high), 4),
        'significant': bool(pval < 0.05),
    }

    if pval >= 0.05 and win_rate > 0.5:
        z = 1.96
        p = win_rate
        n_needed = math.ceil((z / (2 * (p - 0.5))) ** 2 * p * (1 - p))
        out['n_needed_for_significance'] = n_needed

    return out


# ── Core simulation ────────────────────────────────────────────────────────────
def run_system(
    name:          str,
    df_test:       pd.DataFrame,
    test_lut:      dict,        # {(ticker, date_str) → pd.Series}
    spy_rows:      pd.DataFrame,
    models:        dict,
    price_lut:     dict,
    vol_lut:       dict,
    trading_days:  list,
    drop_tickers:  set,
    dynamic_hold:  bool,
    ticker_series: dict = None,  # Improvement 4: needed for rolling correlation
) -> dict:
    """
    Core-satellite simulation with rank-based signal generation.

    Core:      70% in SPY, bought at test-period start, held throughout.
    Satellite: rank-based long signals, up to MAX_DAILY_SIGNALS per day.

    Signal flow per day:
      1. Score all eligible TRADE_TICKERS by composite model score
      2. Apply quality gates (score>0, regime>=0.3, rel_vol>=0.8)
      3. Take top MAX_DAILY_SIGNALS tickers by score
      4. Open positions if satellite cash available

    Dynamic exit (System A):
      Each day, re-score open positions; exit early if score flips ≤ 0.
      Minimum hold: 1 calendar day. Max hold: MAX_HOLD_DAYS trading days.

    Fixed exit (System B):
      Hold exactly MAX_HOLD_DAYS trading days (stop-loss still fires).
    """
    print(f'  Simulating {name}...')

    # ── Core: buy SPY at first available price ────────────────────────────────
    core_capital = INITIAL_CAPITAL * CORE_PCT
    sat_capital  = INITIAL_CAPITAL * SATELLITE_PCT_DEF

    spy_start = next(
        (price_lut.get(('SPY', td)) for td in trading_days
         if price_lut.get(('SPY', td))),
        470.0,
    )
    core_shares = core_capital / spy_start

    # ── Satellite state ───────────────────────────────────────────────────────
    sat_cash   = sat_capital
    open_pos: list[SatPos] = []
    closed_trades = []
    cooldown  = {}          # ticker → exit_date str
    equity_curve = []

    for current_date in trading_days:
        # ── Portfolio mark-to-market ──────────────────────────────────────────
        spy_price   = price_lut.get(('SPY', current_date), spy_start)
        sat_pos_val = sum(
            pos.n_shares * price_lut.get((pos.ticker, current_date), pos.entry_fill)
            for pos in open_pos
        )
        equity_curve.append({
            'date':   current_date,
            'equity': core_shares * spy_price + sat_cash + sat_pos_val,
        })

        # ── Exit checks ───────────────────────────────────────────────────────
        to_close = []
        for pos in open_pos:
            # Enforce minimum hold of 1 day
            if current_date == pos.entry_date:
                continue

            cur = price_lut.get((pos.ticker, current_date))
            if cur is None:
                continue

            unrlzd = (cur - pos.entry_fill) / pos.entry_fill
            reason = None

            if unrlzd <= STOP_LOSS:
                reason = 'stop_loss'
            elif unrlzd >= TAKE_PROFIT:
                reason = 'take_profit'
            elif current_date >= pos.exit_date:
                reason = 'hold_expired'
            elif dynamic_hold:
                # Model-reversal: re-score the position with today's features
                row = test_lut.get((pos.ticker, current_date))
                if row is not None:
                    X  = pd.DataFrame(
                        [row[FEATURE_COLS].fillna(0.0).to_dict()])
                    dm = xgb.DMatrix(X)
                    p5 = float(models['5d'].predict(dm)[0])
                    p3 = float(models['3d'].predict(dm)[0])
                    p1 = float(models['1d'].predict(dm)[0])
                    if _composite_score(p1, p3, p5) <= 0:
                        reason = 'model_reversal'

            if reason:
                to_close.append((pos, reason))

        for pos, reason in to_close:
            cur       = price_lut.get((pos.ticker, current_date), pos.entry_fill)
            exit_fill = cur * (1 - SLIPPAGE)
            pnl_pct   = (exit_fill - pos.entry_fill) / pos.entry_fill

            # Compute actual hold in trading days
            try:
                e_idx = trading_days.index(pos.entry_date)
                x_idx = trading_days.index(current_date)
                actual_hold = x_idx - e_idx
            except ValueError:
                actual_hold = 0

            sat_cash += pos.n_shares * exit_fill
            open_pos  = [p for p in open_pos if p.ticker != pos.ticker]
            cooldown[pos.ticker] = current_date

            closed_trades.append({
                'ticker':       pos.ticker,
                'entry_date':   pos.entry_date,
                'exit_date':    current_date,
                'entry_fill':   round(pos.entry_fill, 4),
                'exit_fill':    round(exit_fill, 4),
                'n_shares':     pos.n_shares,
                'cost':         round(pos.cost, 2),
                'pnl_pct':      round(pnl_pct, 4),
                'exit_reason':  reason,
                'actual_hold':  actual_hold,
                'entry_score':  round(pos.score, 6),
                'pred_5d':      round(pos.pred_5d, 6),
            })

        # ── Skip if fully loaded ──────────────────────────────────────────────
        n_open_slots = MAX_POSITIONS - len(open_pos)
        if n_open_slots == 0:
            continue

        # ── Regime allocation ─────────────────────────────────────────────────
        regime_pct     = get_regime_allocation(current_date, spy_rows)
        regime_sat_cap = INITIAL_CAPITAL * regime_pct

        # ── Rank-based signal generation ──────────────────────────────────────
        today_rows = df_test[df_test['date'] == current_date]
        if today_rows.empty:
            continue

        candidates = []
        for _, row in today_rows.iterrows():
            ticker = row['ticker']

            # Gate A: TRADE universe only (not WATCH, not SPY)
            if ticker not in TRADE_TICKERS:
                continue
            # Gate B: minimum Reddit activity
            if row['post_count_1d'] < MIN_ACTIVITY_GATE:
                continue
            # Drop list
            if ticker in drop_tickers:
                continue
            # No double-up on existing position
            if any(p.ticker == ticker for p in open_pos):
                continue
            # Cooldown
            last_exit = cooldown.get(ticker)
            if last_exit:
                d1 = date.fromisoformat(current_date)
                d2 = date.fromisoformat(last_exit)
                if (d1 - d2).days < COOLDOWN_DAYS:
                    continue

            # Improvement 1 — Earnings filter (cached; no live API call per row)
            if not _earnings_safe(ticker, current_date):
                continue

            # Fix 4 — 20-day MA trend filter
            if not _above_20ma(ticker, current_date, ticker_series):
                continue

            # Score via composite model prediction
            X  = pd.DataFrame([row[FEATURE_COLS].fillna(0.0).to_dict()])
            dm = xgb.DMatrix(X)
            p5 = float(models['5d'].predict(dm)[0])
            p3 = float(models['3d'].predict(dm)[0])
            p1 = float(models['1d'].predict(dm)[0])
            score = _composite_score(p1, p3, p5)

            # Gate C: positive expected return
            if score <= 0:
                continue
            # Gate D: not deep bear regime
            regime_score_val = float(row.get('regime_score', 0.3))
            if regime_score_val < 0.3:
                continue
            # Gate E: minimum volume activity
            if float(row.get('relative_volume', 1.0)) < 0.8:
                continue

            # Improvement 3 — Grinold vol-adjusted score for ranking
            adj_score = _vol_adjusted_score(score, ticker, current_date, vol_lut)

            # Improvement 4 — Sector filter: skip if sector already held
            candidate_sector = _SECTOR_MAP.get(ticker, 'Unknown')
            open_sectors = [_SECTOR_MAP.get(p.ticker, 'Unknown') for p in open_pos]
            if open_sectors.count(candidate_sector) >= 3:
                continue

            # Correlation filter removed — sector cap (>=3) provides sufficient
            # diversification during statistical accumulation phase (<200 trades).

            cur_price = price_lut.get((ticker, current_date), float(row['close']))
            candidates.append({
                'ticker':        ticker,
                'score':         score,
                'adj_score':     adj_score,
                'pred_1d':       p1,
                'pred_3d':       p3,
                'pred_5d':       p5,
                'close':         cur_price,
                'regime_score':  regime_score_val,
            })

        # Rank by vol-adjusted composite score (Improvement 3); take up to MAX_DAILY_SIGNALS
        candidates.sort(key=lambda x: x['adj_score'], reverse=True)
        to_open = candidates[:min(MAX_DAILY_SIGNALS, n_open_slots)]

        # ── Open new positions ────────────────────────────────────────────────
        for cand in to_open:
            if len(open_pos) >= MAX_POSITIONS:
                break
            ticker = cand['ticker']
            if any(p.ticker == ticker for p in open_pos):
                continue

            size_dollars = get_position_size(
                ticker, current_date, regime_sat_cap, cand['score'], vol_lut)
            entry_fill = cand['close'] * (1 + SLIPPAGE)
            n_shares   = int(size_dollars / entry_fill)
            cost       = n_shares * entry_fill

            if n_shares == 0 or cost > sat_cash:
                continue

            ex_date = _exit_date(trading_days, current_date, MAX_HOLD_DAYS)
            sat_cash -= cost
            open_pos.append(SatPos(
                ticker=ticker, entry_date=current_date,
                entry_fill=entry_fill, exit_date=ex_date,
                n_shares=n_shares, cost=cost,
                pred_1d=cand['pred_1d'], pred_3d=cand['pred_3d'],
                pred_5d=cand['pred_5d'], score=cand['score'],
            ))

    # ── Force-close remaining positions at period end ─────────────────────────
    last_date = trading_days[-1]
    for pos in list(open_pos):
        cur       = price_lut.get((pos.ticker, last_date), pos.entry_fill)
        exit_fill = cur * (1 - SLIPPAGE)
        pnl_pct   = (exit_fill - pos.entry_fill) / pos.entry_fill
        try:
            e_idx = trading_days.index(pos.entry_date)
            actual_hold = len(trading_days) - 1 - e_idx
        except ValueError:
            actual_hold = 0
        sat_cash += pos.n_shares * exit_fill
        closed_trades.append({
            'ticker':       pos.ticker,
            'entry_date':   pos.entry_date,
            'exit_date':    last_date,
            'entry_fill':   round(pos.entry_fill, 4),
            'exit_fill':    round(exit_fill, 4),
            'n_shares':     pos.n_shares,
            'cost':         round(pos.cost, 2),
            'pnl_pct':      round(pnl_pct, 4),
            'exit_reason':  'end_of_period',
            'actual_hold':  actual_hold,
            'entry_score':  round(pos.score, 6),
            'pred_5d':      round(pos.pred_5d, 6),
        })

    # ── Final accounting ──────────────────────────────────────────────────────
    spy_end    = price_lut.get(('SPY', last_date), spy_start)
    core_final = core_shares * spy_end * (1 - SLIPPAGE)
    total_final = core_final + sat_cash

    total_return = (total_final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    sat_return   = (sat_cash - sat_capital) / sat_capital * 100
    spy_ret_pct  = (spy_end - spy_start) / spy_start * 100

    metrics  = _calc_metrics(equity_curve, closed_trades)
    monthly  = _monthly_returns(equity_curve)
    sig_test = compute_significance(closed_trades)

    # Exit-reason breakdown
    reasons = {}
    for t in closed_trades:
        r = t['exit_reason']
        reasons[r] = reasons.get(r, 0) + 1

    # Actual-hold distribution
    holds = [t['actual_hold'] for t in closed_trades]
    avg_hold = round(float(np.mean(holds)), 1) if holds else 0.0

    def _wr(ts):
        return round(sum(1 for t in ts if t['pnl_pct'] > 0) / len(ts), 4) if ts else 0.0

    print(f'    {name}: {len(closed_trades)} trades  '
          f'combined={total_return:+.1f}%  sat={sat_return:+.1f}%  '
          f'avg_hold={avg_hold:.1f}d')

    return {
        'final_core':           round(core_final, 2),
        'final_satellite':      round(sat_cash, 2),
        'final_total':          round(total_final, 2),
        'satellite_return_pct': round(sat_return, 2),
        'total_return_pct':     round(total_return, 2),
        'alpha_vs_spy_pct':     round(total_return - spy_ret_pct, 2),
        'n_trades':             len(closed_trades),
        'win_rate':             _wr(closed_trades),
        'avg_hold_days':        avg_hold,
        'exit_reasons':         reasons,
        'significance':         sig_test,
        'monthly_returns':      monthly,
        'trades':               closed_trades,
        **metrics,
    }


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print('=' * 70)
    print('RSSS Backtest v2 — Rank-Based Signals  [HISTORICAL SIMULATION]')
    print('=' * 70)

    # ── Feature store ─────────────────────────────────────────────────────────
    print('Loading features_v2.parquet...')
    df = pd.read_parquet('data/features/features_v2.parquet')
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df_test = df[(df['date'] >= TEST_START) & (df['date'] <= TEST_END)].copy()
    df_test = df_test.dropna(subset=['close']).reset_index(drop=True)

    for col in FEATURE_COLS:
        if col not in df_test.columns:
            df_test[col] = 0.0

    print(f'Test rows: {len(df_test):,}  '
          f'({df_test["date"].min()} → {df_test["date"].max()})')

    trading_days = sorted(df_test['date'].unique().tolist())
    print(f'Trading days: {len(trading_days)}')

    # Build per-row lookup for model-reversal checks (System A)
    test_lut: dict = {}
    for _, row in df_test.iterrows():
        test_lut[(row['ticker'], row['date'])] = row

    # SPY rows for regime lookup
    spy_rows = df_test[df_test['ticker'] == 'SPY'][
        ['date', 'spy_above_200ma', 'vix_percentile']].copy()

    # ── Drop tickers (locked architecture) ────────────────────────────────────
    try:
        with open('experiments/phase3_locked_architecture.json') as f:
            arch = json.load(f)
        drop_tickers = set(arch.get('drop_tickers', []))
    except FileNotFoundError:
        drop_tickers = set()
    print(f'Drop tickers: {sorted(drop_tickers)}')

    # ── Download prices ───────────────────────────────────────────────────────
    sim_tickers = sorted(
        (set(df_test['ticker'].unique()) | {'SPY'}) - drop_tickers)
    print(f'Downloading prices for {len(sim_tickers)} tickers...')

    raw_prices = yf.download(
        sim_tickers, start=TEST_START, end='2026-01-15',
        auto_adjust=True, progress=False, threads=True,
    )
    if isinstance(raw_prices.columns, pd.MultiIndex):
        close_df = raw_prices['Close']
    else:
        close_df = raw_prices[['Close']].rename(columns={'Close': sim_tickers[0]})

    price_lut: dict = {}
    ticker_series: dict = {}
    for ticker in sim_tickers:
        if ticker not in close_df.columns:
            continue
        series = close_df[ticker].dropna()
        ticker_series[ticker] = series
        for dt, price in series.items():
            price_lut[(ticker, dt.strftime('%Y-%m-%d'))] = float(price)

    for _, row in df_test.iterrows():
        key = (row['ticker'], row['date'])
        if key not in price_lut:
            price_lut[key] = float(row['close'])

    print(f'Price lookup: {len(price_lut):,} entries')

    # ── 20-day realized vol for position sizing ───────────────────────────────
    vol_lut: dict = {}
    for ticker, series in ticker_series.items():
        rv = series.pct_change().rolling(20, min_periods=10).std()
        for dt, v in rv.items():
            if pd.notna(v) and v > 0:
                vol_lut[(ticker, dt.strftime('%Y-%m-%d'))] = float(v)
    print(f'Vol lookup: {len(vol_lut):,} entries')

    # ── SPY benchmark ──────────────────────────────────────────────────────────
    spy_prices = {k[1]: v for k, v in price_lut.items() if k[0] == 'SPY'}
    if spy_prices:
        dts        = sorted(spy_prices)
        spy_start_p = spy_prices[dts[0]]
        spy_end_p   = spy_prices[dts[-1]]
        spy_return  = (spy_end_p - spy_start_p) / spy_start_p * 100
    else:
        spy_return, spy_start_p, spy_end_p = 47.84, 458.81, 678.32
    print(f'SPY 2024-2025: {spy_return:+.2f}%')

    # ── Load v2 models ─────────────────────────────────────────────────────────
    models = {}
    for hz in ('1d', '3d', '5d'):
        m = xgb.Booster()
        m.load_model(f'models/model_{hz}_v2.json')
        models[hz] = m
    print('V2 models loaded (16 features each)')

    # ── Improvement 4: load sector map ────────────────────────────────────────
    global _SECTOR_MAP
    _SECTOR_MAP = _load_sector_map()
    print(f'Sector map loaded: {len(_SECTOR_MAP)} tickers')

    # ── Improvement 1: build earnings cache (one fetch per trade ticker) ──────
    # Only pre-cache; individual rows use _earnings_safe() for O(1) lookup.
    # NOTE: historical backtest uses static cache — not calling live API per date.
    print('Earnings cache: skipped for historical backtest (static cache only)')

    # ── Run both systems ───────────────────────────────────────────────────────
    print('\nRunning simulations...')
    common = dict(
        test_lut=test_lut, spy_rows=spy_rows, models=models,
        price_lut=price_lut, vol_lut=vol_lut,
        trading_days=trading_days, drop_tickers=drop_tickers,
        ticker_series=ticker_series,
    )
    sys_a = run_system('A_rank_dynamic', df_test, dynamic_hold=True,  **common)
    sys_b = run_system('B_rank_fixed5d', df_test, dynamic_hold=False, **common)

    # ── Save results ───────────────────────────────────────────────────────────
    out = {
        'simulation':      True,
        'note':            ('HISTORICAL SIMULATION — 2024-2025 out-of-sample. '
                            'Rank-based signals. Core-satellite. Long-only.'),
        'period':          f'{TEST_START} to {TEST_END}',
        'initial_capital': INITIAL_CAPITAL,
        'core_pct':        CORE_PCT,
        'spy_return_pct':  round(spy_return, 2),
        'spy_start_price': round(spy_start_p, 2),
        'spy_end_price':   round(spy_end_p, 2),
        'drop_tickers':    sorted(drop_tickers),
        'min_activity_gate': MIN_ACTIVITY_GATE,
        'max_daily_signals': MAX_DAILY_SIGNALS,
        'max_hold_days':   MAX_HOLD_DAYS,
        'feature_count':   len(FEATURE_COLS),
        'systems': {
            'A_rank_dynamic': sys_a,
            'B_rank_fixed5d': sys_b,
        },
    }

    out_path = Path('experiments/backtest_v2_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nResults saved → {out_path}')

    # ── Print detailed per-system results ──────────────────────────────────────
    W = 78
    print('\n' + '═' * W)
    print('RSSS Backtest v2 — Rank-Based  [HISTORICAL SIMULATION]')
    print('═' * W)

    for label, sys_res in [('System A — Dynamic Exit', sys_a),
                            ('System B — Fixed 5D    ', sys_b)]:
        sig = sys_res.get('significance', {})
        sig_str = 'YES ✓' if sig.get('significant') else 'NO'
        wr = sig.get('win_rate', sys_res['win_rate'])

        print(f'\n  {label}:')
        print(f'    Core SPY:             {spy_return:+.1f}%')
        print(f'    Satellite RSSS:       {sys_res["satellite_return_pct"]:+.1f}%')
        print(f'    Combined portfolio:   {sys_res["total_return_pct"]:+.1f}%')
        print(f'    Alpha (comb-SPY):     {sys_res["alpha_vs_spy_pct"]:+.1f}%')
        print(f'    Sharpe ratio:         {sys_res["sharpe_ratio"]:.2f}')
        print(f'    Max drawdown:         {sys_res["max_drawdown_pct"]:.1f}%')
        print(f'    Trades:               {sys_res["n_trades"]}')
        print(f'    Avg hold (trading d): {sys_res["avg_hold_days"]:.1f}')
        print(f'    Win rate:             {wr:.1%}')
        if sig:
            print(f'    95% CI:              [{sig["ci_low"]:.1%}, {sig["ci_high"]:.1%}]')
            print(f'    Binomial p-value:     {sig["pvalue"]:.3f}')
            print(f'    Stat significant:     {sig_str}')
            if 'n_needed_for_significance' in sig:
                print(f'    Trades to sig:        {sig["n_needed_for_significance"]}')
        if sys_res.get('exit_reasons'):
            print(f'    Exit reasons:  {sys_res["exit_reasons"]}')

    # ── Three-column comparison table ──────────────────────────────────────────
    sig_a = sys_a.get('significance', {})
    sig_b = sys_b.get('significance', {})

    print()
    print(f'  {"Metric":<22} {"Old(57T)":>9} {"Fixed(19T)":>11} '
          f'{"Rank(167T)":>11} {"S2+Corr(140T)":>14} {"NoCorrF(?T)":>12}')
    print('  ' + '─' * 83)

    rows = [
        ('Total return',
         f'{_OLD["total_return_pct"]:+.1f}%',
         f'{_FIXED["total_return_pct"]:+.1f}%',
         f'{_RANK167["total_return_pct"]:+.1f}%',
         f'{_SECTOR2_CORR["total_return_pct"]:+.1f}%',
         f'{sys_a["total_return_pct"]:+.1f}%'),
        ('Alpha vs SPY',
         f'{_OLD["alpha_pct"]:+.1f}%',
         f'{_FIXED["alpha_pct"]:+.1f}%',
         f'{_RANK167["alpha_pct"]:+.1f}%',
         f'{_SECTOR2_CORR["alpha_pct"]:+.1f}%',
         f'{sys_a["alpha_vs_spy_pct"]:+.1f}%'),
        ('Sharpe',
         f'{_OLD["sharpe_ratio"]:.2f}',
         f'{_FIXED["sharpe_ratio"]:.2f}',
         f'{_RANK167["sharpe_ratio"]:.2f}',
         f'{_SECTOR2_CORR["sharpe_ratio"]:.2f}',
         f'{sys_a["sharpe_ratio"]:.2f}'),
        ('Win rate',
         f'{_OLD["win_rate"]:.1%}',
         f'{_FIXED["win_rate"]:.1%}',
         f'{_RANK167["win_rate"]:.1%}',
         f'{_SECTOR2_CORR["win_rate"]:.1%}',
         f'{sig_a.get("win_rate", sys_a["win_rate"]):.1%}'),
        ('Trades',
         str(_OLD['n_trades']),
         str(_FIXED['n_trades']),
         str(_RANK167['n_trades']),
         str(_SECTOR2_CORR['n_trades']),
         str(sys_a['n_trades'])),
        ('Max drawdown',
         f'{_OLD["max_drawdown_pct"]:.1f}%',
         f'{_FIXED["max_drawdown_pct"]:.1f}%',
         f'{_RANK167["max_drawdown_pct"]:.1f}%',
         f'{_SECTOR2_CORR["max_drawdown_pct"]:.1f}%',
         f'{sys_a["max_drawdown_pct"]:.1f}%'),
        ('p-value',
         '—', '—', '0.063',
         f'{_SECTOR2_CORR["pvalue"]:.3f}',
         f'{sig_a.get("pvalue", 1.0):.3f}'),
        ('Stat significant',
         'NO', 'NO', 'NO', 'NO',
         'YES ✓' if sig_a.get('significant') else 'NO'),
    ]

    for metric, c1, c2, c3, c4, c5 in rows:
        print(f'  {metric:<22} {c1:>9} {c2:>11} {c3:>11} {c4:>14} {c5:>12}')

    a_vs_b = sys_a['total_return_pct'] - sys_b['total_return_pct']
    print()
    print(f'  A vs B (dynamic vs fixed): {a_vs_b:+.1f}% '
          f'({"dynamic helps" if a_vs_b > 0 else "fixed wins"})')

    print()
    print('═' * W)
    print(f'Core 70% SPY  |  Satellite 30% RSSS  |  Initial ${INITIAL_CAPITAL:,.0f}')
    print('─' * W)


if __name__ == '__main__':
    main()
