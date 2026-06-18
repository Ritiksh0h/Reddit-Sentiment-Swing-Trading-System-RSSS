"""
Historical backtest — 2024-2025 out-of-sample test period.

Walk-forward simulation: each day uses only data available up to that day.
Entry and exit prices fetched from yfinance (batch download at startup).
Uses the same thresholds, gates, and sizing as the live system.

CRITICAL rules enforced here:
  - Only 2024-2025 rows used (split='test' in feature store)
  - target_return_5d is NEVER passed to model.predict()
  - BULLISH >= 1.5%, stop-loss -8%, take-profit 15%
  - Marked SIMULATION throughout — this is NOT live trading

Output: experiments/backtest_results.json
"""
import json
import logging
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

sys.path.insert(0, '.')
from portfolio.position_sizer import compute_position_size, compute_slippage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
log = logging.getLogger('backtest')

INITIAL_CAPITAL   = 10_000.0
MAX_POSITIONS     = 3
DENSITY_GATE      = 10
MIN_PRED_RET      = 0.005
BULLISH_THRESHOLD = 0.015
BEARISH_THRESHOLD = -0.015
STOP_LOSS_PCT     = -0.08
TAKE_PROFIT_PCT   = 0.15
HOLD_DAYS         = 5    # trading days
TICKER_COOLDOWN   = 7    # calendar days
RISK_FREE_DAILY   = 0.05 / 252


@dataclass
class BtPos:
    ticker: str
    entry_date: str
    entry_price: float
    n_shares: int
    predicted_return: float
    signal: str
    entry_idx: int


def _load():
    df = pd.read_parquet('data/features/features_complete.parquet')
    df['date'] = df['date'].astype(str).str[:10]
    df = df[df['date'] >= '2024-01-01'].copy()
    df = df[df['date'] <= '2025-12-31'].copy()

    with open('experiments/phase3_locked_architecture.json') as f:
        arch = json.load(f)
    features    = arch['features']
    drop_tickers = set(arch['drop_tickers'])

    df = df[~df['ticker'].isin(drop_tickers)].copy()

    with open('models/registry/model_5d.pkl', 'rb') as f:
        model = pickle.load(f)
    model.set_params(n_jobs=1)

    return df, model, features, drop_tickers


def _download_prices(tickers):
    log.info(f'Downloading prices for {len(tickers) + 1} tickers (2024-2025)...')
    all_tickers = sorted(set(tickers) | {'SPY'})
    raw = yf.download(
        all_tickers, start='2023-12-15', end='2026-03-01',
        auto_adjust=True, progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw[['Close']]
        close.columns = [all_tickers[0]]

    close.index = close.index.strftime('%Y-%m-%d')
    return close


def _get_px(close_df, ticker, date_str):
    if ticker not in close_df.columns:
        return None
    try:
        v = close_df.loc[date_str, ticker]
        return float(v) if pd.notna(v) else None
    except KeyError:
        return None


def _compute_metrics(equity_curve, closed_trades):
    if len(equity_curve) < 2:
        return {}
    vals = np.array([e['value'] for e in equity_curve], dtype=float)
    rets = np.diff(vals) / vals[:-1]

    mean_r = float(rets.mean())
    std_r  = float(rets.std())
    down   = rets[rets < RISK_FREE_DAILY]
    dstd   = float(down.std()) if len(down) > 1 else std_r

    sharpe  = (mean_r - RISK_FREE_DAILY) / std_r  * np.sqrt(252) if std_r > 0 else 0.0
    sortino = (mean_r - RISK_FREE_DAILY) / dstd   * np.sqrt(252) if dstd > 0 else 0.0

    peak = vals[0]; max_dd = 0.0
    for v in vals:
        peak = max(peak, v)
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd

    pnls = [t['pnl_dollars'] for t in closed_trades]
    wins  = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    wr = len(wins) / len(pnls) if pnls else 0.0
    pf = round(sum(wins) / abs(sum(losses)), 4) if losses else None

    return {
        'sharpe_ratio':     round(float(sharpe), 4),
        'sortino_ratio':    round(float(sortino), 4),
        'max_drawdown_pct': round(float(max_dd * 100), 2),
        'win_rate':         round(wr, 4),
        'profit_factor':    pf,
    }


def _monthly_returns(equity_curve):
    df = pd.DataFrame(equity_curve)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    monthly = {}
    for _, grp in df.resample('ME'):
        if len(grp) < 1:
            continue
        start = grp['value'].iloc[0]
        end   = grp['value'].iloc[-1]
        key   = grp.index[0].strftime('%Y-%m')
        monthly[key] = round((end - start) / start * 100, 2)
    return monthly


def run():
    log.info('=== RSSS Historical Backtest 2024-2025 ===')
    log.info('SIMULATION — not live trading results')

    df, model, FEATURES, DROP_TICKERS = _load()
    tickers_in_test = list(df['ticker'].unique())
    log.info(f'Test tickers: {len(tickers_in_test)}  Features: {len(FEATURES)}')

    close_df = _download_prices(tickers_in_test)

    # Trading dates: all market days in 2024-2025 from the price data
    trading_dates = [d for d in close_df.index
                     if '2024-01-01' <= d <= '2025-12-31']
    date_to_idx = {d: i for i, d in enumerate(trading_dates)}
    log.info(f'Trading days in simulation: {len(trading_dates)} '
             f'({trading_dates[0]} → {trading_dates[-1]})')

    # Feature store grouped by date for fast lookup
    feature_by_date = {d: grp for d, grp in df.groupby('date')}

    # SPY reference for comparison
    spy_start = _get_px(close_df, 'SPY', trading_dates[0])

    # ── Walk-forward simulation ──────────────────────────────────────────────
    cash            = INITIAL_CAPITAL
    open_positions  = []        # list[BtPos]
    closed_trades   = []
    equity_curve    = []
    cooldown        = {}        # ticker → last trade date str

    for date_str in trading_dates:
        d_idx = date_to_idx[date_str]

        # ── 1. Check exits (stop-loss first, then take-profit, then hold) ────
        to_close = []
        for pos in open_positions:
            price = _get_px(close_df, pos.ticker, date_str)
            if price is None:
                continue
            unreal    = (price - pos.entry_price) / pos.entry_price
            days_held = d_idx - pos.entry_idx

            if unreal <= STOP_LOSS_PCT:
                to_close.append((pos, price, 'stop_loss'))
            elif unreal >= TAKE_PROFIT_PCT:
                to_close.append((pos, price, 'take_profit'))
            elif days_held >= HOLD_DAYS:
                to_close.append((pos, price, 'hold_period_expired'))

        tickers_closing = {pos.ticker for pos, _, _ in to_close}
        for pos, exit_price, reason in to_close:
            proceeds   = pos.n_shares * exit_price
            cost_basis = pos.n_shares * pos.entry_price
            pnl_pct    = (exit_price - pos.entry_price) / pos.entry_price
            pnl_dollars = proceeds - cost_basis
            cash += proceeds
            cooldown[pos.ticker] = date_str
            closed_trades.append({
                'ticker':           pos.ticker,
                'entry_date':       pos.entry_date,
                'exit_date':        date_str,
                'entry_price':      round(pos.entry_price, 4),
                'exit_price':       round(exit_price, 4),
                'n_shares':         pos.n_shares,
                'pnl_pct':          round(pnl_pct * 100, 2),
                'pnl_dollars':      round(pnl_dollars, 2),
                'exit_reason':      reason,
                'signal':           pos.signal,
                'predicted_return': round(pos.predicted_return * 100, 2),
            })

        open_positions = [p for p in open_positions
                          if p.ticker not in tickers_closing]

        # ── 2. Equity snapshot ───────────────────────────────────────────────
        pos_value = sum(
            p.n_shares * (_get_px(close_df, p.ticker, date_str) or p.entry_price)
            for p in open_positions
        )
        equity = cash + pos_value

        spy_now = _get_px(close_df, 'SPY', date_str)
        spy_val = round(INITIAL_CAPITAL * spy_now / spy_start, 2) \
                  if (spy_now and spy_start) else None

        equity_curve.append({
            'date':      date_str,
            'value':     round(equity, 2),
            'spy_value': spy_val,
        })

        # ── 3. Generate signals from feature store ───────────────────────────
        if date_str not in feature_by_date:
            continue

        today_rows = feature_by_date[date_str]
        qualifying = today_rows[today_rows['post_count_1d'] >= DENSITY_GATE].copy()
        if qualifying.empty:
            continue

        avail = [f for f in FEATURES if f in qualifying.columns]
        X     = qualifying[avail].fillna(0.0)
        preds = model.predict(X)

        raw_signals = []
        for i, (_, row) in enumerate(qualifying.iterrows()):
            pred = float(preds[i])
            if abs(pred) < MIN_PRED_RET:
                continue
            if pred >= BULLISH_THRESHOLD:
                sig = 'BULLISH'
            elif pred <= BEARISH_THRESHOLD:
                sig = 'BEARISH'
            else:
                sig = 'NEUTRAL'
            raw_signals.append((abs(pred), sig, pred, row))

        # Sort descending by |prediction| — strongest conviction first
        raw_signals.sort(reverse=True)

        # ── 4. Open new positions ────────────────────────────────────────────
        for _, sig, pred, row in raw_signals:
            if len(open_positions) >= MAX_POSITIONS:
                break

            ticker = row['ticker']
            if any(p.ticker == ticker for p in open_positions):
                continue

            last_trade = cooldown.get(ticker)
            if last_trade:
                gap = (date.fromisoformat(date_str) - date.fromisoformat(last_trade)).days
                if gap < TICKER_COOLDOWN:
                    continue

            entry_price = _get_px(close_df, ticker, date_str)
            if entry_price is None:
                continue

            atr_14 = float(row['atr_14'])
            pv = cash + sum(
                p.n_shares * (_get_px(close_df, p.ticker, date_str) or p.entry_price)
                for p in open_positions
            )
            sizing = compute_position_size(pv, entry_price, atr_14)
            if sizing['n_shares'] == 0:
                continue

            slip      = compute_slippage(entry_price, float(row.get('mention_growth_7d', 0.0)))
            fill_price = entry_price * (1.0 + slip)
            cost       = sizing['n_shares'] * fill_price
            if cost > cash:
                continue

            cash -= cost
            open_positions.append(BtPos(
                ticker=ticker,
                entry_date=date_str,
                entry_price=round(fill_price, 4),
                n_shares=sizing['n_shares'],
                predicted_return=pred,
                signal=sig,
                entry_idx=d_idx,
            ))

    # Close any positions still open at simulation end
    last_date = trading_dates[-1]
    for pos in open_positions:
        price = _get_px(close_df, pos.ticker, last_date) or pos.entry_price
        proceeds    = pos.n_shares * price
        pnl_pct     = (price - pos.entry_price) / pos.entry_price
        pnl_dollars = proceeds - pos.n_shares * pos.entry_price
        cash += proceeds
        closed_trades.append({
            'ticker':           pos.ticker,
            'entry_date':       pos.entry_date,
            'exit_date':        last_date,
            'entry_price':      round(pos.entry_price, 4),
            'exit_price':       round(price, 4),
            'n_shares':         pos.n_shares,
            'pnl_pct':          round(pnl_pct * 100, 2),
            'pnl_dollars':      round(pnl_dollars, 2),
            'exit_reason':      'simulation_end',
            'signal':           pos.signal,
            'predicted_return': round(pos.predicted_return * 100, 2),
        })

    # ── Compute final stats ──────────────────────────────────────────────────
    final_val    = equity_curve[-1]['value'] if equity_curve else INITIAL_CAPITAL
    total_return = (final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    spy_end = _get_px(close_df, 'SPY', trading_dates[-1])
    spy_ret = ((spy_end / spy_start) - 1) * 100 if (spy_end and spy_start) else None
    alpha   = round(total_return - spy_ret, 2) if spy_ret is not None else None

    metrics        = _compute_metrics(equity_curve, closed_trades)
    monthly_rets   = _monthly_returns(equity_curve)

    # Annual breakdown
    def _year_stats(year):
        yr_trades = [t for t in closed_trades if t['entry_date'].startswith(year)]
        yr_curve  = [e for e in equity_curve if e['date'].startswith(year)]
        yr_start  = yr_curve[0]['value']  if yr_curve else INITIAL_CAPITAL
        yr_end    = yr_curve[-1]['value'] if yr_curve else INITIAL_CAPITAL
        yr_ret    = (yr_end - yr_start) / yr_start * 100 if yr_start else 0.0
        pnls      = [t['pnl_dollars'] for t in yr_trades]
        wins      = [p for p in pnls if p > 0]
        return {
            'n_trades':    len(yr_trades),
            'return_pct':  round(yr_ret, 2),
            'start_value': round(yr_start, 2),
            'end_value':   round(yr_end, 2),
            'win_rate':    round(len(wins) / len(pnls), 4) if pnls else 0.0,
        }

    annual = {'2024': _year_stats('2024'), '2025': _year_stats('2025')}

    # Exit-reason breakdown
    exit_counts = {}
    for t in closed_trades:
        r = t['exit_reason']
        exit_counts[r] = exit_counts.get(r, 0) + 1

    results = {
        'simulation':         True,
        'note':               ('SIMULATION: 2024-2025 out-of-sample test period. '
                               'NOT live trading. Past simulation results do not '
                               'guarantee future performance.'),
        'period':             f'{trading_dates[0]} to {trading_dates[-1]}',
        'initial_capital':    INITIAL_CAPITAL,
        'final_value':        round(final_val, 2),
        'total_return_pct':   round(total_return, 2),
        'spy_return_pct':     round(spy_ret, 2) if spy_ret is not None else None,
        'alpha':              alpha,
        'n_trades':           len(closed_trades),
        'exit_breakdown':     exit_counts,
        **metrics,
        'annual_breakdown':   annual,
        'monthly_returns':    monthly_rets,
        'equity_curve':       equity_curve,
        'trades':             closed_trades,
    }

    out = Path('experiments/backtest_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)

    log.info('─' * 60)
    log.info(f'2024 : {annual["2024"]["n_trades"]:>3} trades  '
             f'return={annual["2024"]["return_pct"]:+.1f}%  '
             f'win={annual["2024"]["win_rate"]*100:.0f}%')
    log.info(f'2025 : {annual["2025"]["n_trades"]:>3} trades  '
             f'return={annual["2025"]["return_pct"]:+.1f}%  '
             f'win={annual["2025"]["win_rate"]*100:.0f}%')
    log.info(f'Full : Sharpe={metrics.get("sharpe_ratio", 0):.2f}  '
             f'Sortino={metrics.get("sortino_ratio", 0):.2f}  '
             f'MaxDD={metrics.get("max_drawdown_pct", 0):.1f}%  '
             f'WinRate={metrics.get("win_rate", 0)*100:.0f}%')
    log.info(f'Total return: {total_return:+.1f}%  vs SPY: '
             f'{spy_ret:+.1f}%' if spy_ret else f'Total return: {total_return:+.1f}%')
    log.info(f'Saved → {out}')
    return results


if __name__ == '__main__':
    run()
