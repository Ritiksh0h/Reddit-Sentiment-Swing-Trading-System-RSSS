"""
RSSS Backtest v2 — System A / B / C comparison, 2024-2025 out-of-sample.

NOTE: ALL RESULTS ARE HISTORICAL SIMULATION.
      This script uses only rows where date is in [2024-01-01, 2025-12-31],
      which is the model's two-year out-of-sample test period.
      No 2019-2023 training data is used for simulation.

Systems:
    A  Long-only, dynamic hold (1D/3D/5D priority cascade)
    B  Long+Short,  dynamic hold
    C  Long+Short,  fixed 5D model only  [control — no 1D/3D models]

Shared rules (all systems):
    Capital:       $10,000 starting
    Max positions: 3 concurrent
    Sizing:        ATR-based, target_risk_pct=0.02, hard cap 25% of portfolio
    Stop-loss:     -8%   long   (price falls 8% below entry)
                   +8%   short  (price rises 8% above entry)
    Take-profit:   +15%  long   (price rises 15% above entry)
                   -15%  short  (price falls 15% below entry)
    Slippage:      0.15% per side (0.30% round-trip)
    Density gate:  post_count_1d >= 10
    Drop tickers:  from experiments/phase3_locked_architecture.json
    Cooldown:      7 calendar days after exit
    Signal floor:  |best_pred| >= 1.5%

Usage:
    python scripts/run_backtest_v2.py
"""

import json
import math
import pickle
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, '.')

# ── Constants ──────────────────────────────────────────────────────────────────
TEST_START      = '2024-01-01'
TEST_END        = '2025-12-31'
INITIAL_CAPITAL = 10_000.0
MAX_POSITIONS   = 3
STOP_LOSS       = -0.08
TAKE_PROFIT     = 0.15
SLIPPAGE        = 0.0015      # 0.15% per side
DENSITY_GATE    = 10
COOLDOWN_DAYS   = 7
BULLISH_T       = 0.015
BEARISH_T       = -0.015
FEATURES = [
    'returns_1d', 'returns_5d', 'returns_20d', 'rsi_14', 'atr_14',
    'relative_volume', 'dist_from_20ma', 'dist_from_50ma',
    'post_count_1d', 'mention_growth_1d', 'mention_growth_7d',
    'news_sentiment_1d', 'st_sentiment_1d', 'st_bull_pct',
]


@dataclass
class BtPos:
    ticker:     str
    direction:  str    # 'LONG' or 'SHORT'
    entry_date: str
    entry_fill: float  # price after slippage
    exit_date:  str    # target exit date (trading-day aligned)
    n_shares:   int
    hold_days:  int    # 1, 3, or 5 trading days
    horizon:    str    # '1D', '3D', or '5D'
    pred_1d:    float
    pred_3d:    float
    pred_5d:    float


def _atr_size(portfolio_value: float, price: float, atr_14: float):
    """ATR-based sizing matching compute_position_size() (target_risk_pct=0.02)."""
    if price <= 0 or atr_14 <= 0:
        return 0, 0.0
    n       = int(0.02 * portfolio_value / atr_14)
    dollars = n * price
    max_d   = 0.25 * portfolio_value
    if dollars > max_d:
        n       = int(max_d / price)
        dollars = n * price
    return n, dollars


def _exit_date(trading_days: list, entry: str, hold: int) -> str:
    """Return the date hold trading days after entry, clamped to last day."""
    try:
        idx = trading_days.index(entry)
    except ValueError:
        return entry
    return trading_days[min(idx + hold, len(trading_days) - 1)]


def _calc_metrics(equity_curve: list, closed_trades: list) -> dict:
    equities = [e['equity'] for e in equity_curve]
    if len(equities) < 2:
        return dict(sharpe_ratio=0.0, sortino_ratio=0.0,
                    max_drawdown_pct=0.0, win_rate=0.0, profit_factor=0.0)

    daily_ret  = [(equities[i] - equities[i - 1]) / equities[i - 1]
                  for i in range(1, len(equities))]
    mean_r = np.mean(daily_ret)
    std_r  = np.std(daily_ret, ddof=1) or 1e-9
    neg_r  = [r for r in daily_ret if r < 0]
    sort_d = (np.std(neg_r, ddof=1) if len(neg_r) > 1 else 1e-9) or 1e-9

    sharpe  = mean_r / std_r  * math.sqrt(252)
    sortino = mean_r / sort_d * math.sqrt(252)

    peak   = equities[0]
    max_dd = 0.0
    for eq in equities:
        peak   = max(peak, eq)
        max_dd = min(max_dd, (eq - peak) / peak)

    pnls    = [t['pnl_pct'] for t in closed_trades]
    wins    = [p for p in pnls if p > 0]
    losses  = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls) if pnls else 0.0
    pf       = sum(wins) / (abs(sum(losses)) or 1e-9) if wins else 0.0

    return dict(
        sharpe_ratio     = round(sharpe, 3),
        sortino_ratio    = round(sortino, 3),
        max_drawdown_pct = round(max_dd * 100, 2),
        win_rate         = round(win_rate, 4),
        profit_factor    = round(pf, 3),
    )


def _monthly_returns(equity_curve: list) -> dict:
    df_eq = pd.DataFrame(equity_curve)
    df_eq['date'] = pd.to_datetime(df_eq['date'])
    df_eq = df_eq.set_index('date')
    out   = {}
    for month, group in df_eq.groupby(pd.Grouper(freq='ME')):
        if len(group) >= 2:
            start, end = group['equity'].iloc[0], group['equity'].iloc[-1]
            out[month.strftime('%Y-%m')] = round((end - start) / start * 100, 2)
    return out


def _pos_value(pos: BtPos, cur_price: float) -> float:
    """Mark-to-market value of one open position."""
    if pos.direction == 'LONG':
        return pos.n_shares * cur_price
    # SHORT: collateral + unrealised gain/loss
    return pos.n_shares * (2 * pos.entry_fill - cur_price)


def run_system(
    name:         str,
    df_test:      pd.DataFrame,
    models:       dict,
    price_lut:    dict,
    trading_days: list,
    drop_tickers: set,
    long_only:    bool,
    dynamic_hold: bool,
    use_5d_only:  bool,
) -> dict:
    """
    Simulate one system over the 2024-2025 test period.

    price_lut: {(ticker, 'YYYY-MM-DD') → float}
    trading_days: sorted list of date strings present in df_test
    """
    print(f'  Simulating {name}...')
    cash           = INITIAL_CAPITAL
    open_pos: list[BtPos] = []
    closed_trades  = []
    cooldown       = {}   # ticker → exit date str
    equity_curve   = []

    for current_date in trading_days:
        # ── Portfolio mark-to-market ─────────────────────────────────────────
        pos_val = sum(
            _pos_value(p, price_lut.get((p.ticker, current_date), p.entry_fill))
            for p in open_pos
        )
        equity_curve.append({'date': current_date, 'equity': cash + pos_val})

        # ── Exit checks ──────────────────────────────────────────────────────
        to_close = []
        seen     = set()
        for pos in open_pos:
            if pos.ticker in seen:
                continue
            cur = price_lut.get((pos.ticker, current_date))
            if cur is None:
                continue

            if pos.direction == 'LONG':
                unrlzd = (cur - pos.entry_fill) / pos.entry_fill
            else:
                unrlzd = (pos.entry_fill - cur) / pos.entry_fill

            reason = None
            if unrlzd <= STOP_LOSS:
                reason = 'stop_loss'
            elif unrlzd >= TAKE_PROFIT:
                reason = 'take_profit'
            elif current_date >= pos.exit_date:
                reason = 'hold_expired'

            if reason:
                seen.add(pos.ticker)
                to_close.append((pos, reason))

        for pos, reason in to_close:
            cur = price_lut.get((pos.ticker, current_date), pos.entry_fill)
            if pos.direction == 'LONG':
                exit_fill = cur * (1 - SLIPPAGE)
                pnl_pct   = (exit_fill - pos.entry_fill) / pos.entry_fill
                proceeds  = pos.n_shares * exit_fill
            else:
                exit_fill = cur * (1 + SLIPPAGE)
                pnl_pct   = (pos.entry_fill - exit_fill) / pos.entry_fill
                proceeds  = pos.n_shares * (2 * pos.entry_fill - exit_fill)

            cash    += proceeds
            open_pos = [p for p in open_pos if p.ticker != pos.ticker]
            cooldown[pos.ticker] = current_date

            closed_trades.append({
                'ticker':      pos.ticker,
                'direction':   pos.direction,
                'entry_date':  pos.entry_date,
                'exit_date':   current_date,
                'entry_fill':  round(pos.entry_fill, 4),
                'exit_fill':   round(exit_fill, 4),
                'n_shares':    pos.n_shares,
                'pnl_pct':     round(pnl_pct, 4),
                'exit_reason': reason,
                'hold_days':   pos.hold_days,
                'horizon':     pos.horizon,
                'pred_1d':     round(pos.pred_1d, 6),
                'pred_3d':     round(pos.pred_3d, 6),
                'pred_5d':     round(pos.pred_5d, 6),
            })

        if len(open_pos) >= MAX_POSITIONS:
            continue

        # ── Generate signals for today ───────────────────────────────────────
        today_rows = df_test[df_test['date'] == current_date]
        if today_rows.empty:
            continue

        signals = []
        for _, row in today_rows.iterrows():
            ticker = row['ticker']
            if row['post_count_1d'] < DENSITY_GATE:
                continue
            if ticker in drop_tickers:
                continue
            if any(p.ticker == ticker for p in open_pos):
                continue

            last_exit = cooldown.get(ticker)
            if last_exit:
                gap = (date.fromisoformat(current_date) - date.fromisoformat(last_exit)).days
                if gap < COOLDOWN_DAYS:
                    continue

            X = pd.DataFrame([row[FEATURES].fillna(0.0).to_dict()])

            if use_5d_only:
                pred_5d = float(models['5d'].predict(X)[0])
                pred_1d = pred_3d = pred_5d
            else:
                pred_1d = float(models['1d'].predict(X)[0])
                pred_3d = float(models['3d'].predict(X)[0])
                pred_5d = float(models['5d'].predict(X)[0])

            # Classify signal and determine hold horizon
            sig = hz = None
            hold = 0
            if dynamic_hold:
                # BULLISH: 5D > 3D > 1D priority
                if pred_5d >= BULLISH_T:
                    sig, hold, hz = 'BULLISH', 5, '5D'
                elif pred_3d >= BULLISH_T:
                    sig, hold, hz = 'BULLISH', 3, '3D'
                elif pred_1d >= BULLISH_T:
                    sig, hold, hz = 'BULLISH', 1, '1D'
                # BEARISH: same priority
                elif pred_5d <= BEARISH_T:
                    sig, hold, hz = 'BEARISH', 5, '5D'
                elif pred_3d <= BEARISH_T:
                    sig, hold, hz = 'BEARISH', 3, '3D'
                elif pred_1d <= BEARISH_T:
                    sig, hold, hz = 'BEARISH', 1, '1D'
                else:
                    continue
            else:
                # Fixed 5D only (System C control)
                if pred_5d >= BULLISH_T:
                    sig, hold, hz = 'BULLISH', 5, '5D'
                elif pred_5d <= BEARISH_T:
                    sig, hold, hz = 'BEARISH', 5, '5D'
                else:
                    continue

            if long_only and sig == 'BEARISH':
                continue

            signals.append({
                'ticker':  ticker,
                'signal':  sig,
                'hold':    hold,
                'hz':      hz,
                'pred_1d': pred_1d,
                'pred_3d': pred_3d,
                'pred_5d': pred_5d,
                'atr_14':  float(row['atr_14']),
                'close':   float(row['close']),
            })

        # Sort: best BULLISH (by pred_5d desc) first, then best BEARISH (pred_5d asc)
        bull = sorted([s for s in signals if s['signal'] == 'BULLISH'],
                      key=lambda x: x['pred_5d'], reverse=True)
        bear = sorted([s for s in signals if s['signal'] == 'BEARISH'],
                      key=lambda x: x['pred_5d'])
        sorted_sigs = bull + bear

        # ── Open positions ───────────────────────────────────────────────────
        port_val = cash + sum(
            _pos_value(p, price_lut.get((p.ticker, current_date), p.entry_fill))
            for p in open_pos
        )

        for sig in sorted_sigs:
            if len(open_pos) >= MAX_POSITIONS:
                break
            ticker = sig['ticker']
            if any(p.ticker == ticker for p in open_pos):
                continue

            n, _ = _atr_size(port_val, sig['close'], sig['atr_14'])
            if n == 0:
                continue

            if sig['signal'] == 'BULLISH':
                entry_fill = sig['close'] * (1 + SLIPPAGE)
                direction  = 'LONG'
            else:
                entry_fill = sig['close'] * (1 - SLIPPAGE)
                direction  = 'SHORT'

            cost = n * entry_fill
            if cost > cash:
                continue

            ex_date = _exit_date(trading_days, current_date, sig['hold'])
            cash -= cost
            port_val -= cost   # update for subsequent signals this day
            open_pos.append(BtPos(
                ticker=ticker, direction=direction,
                entry_date=current_date, entry_fill=entry_fill,
                exit_date=ex_date, n_shares=n,
                hold_days=sig['hold'], horizon=sig['hz'],
                pred_1d=sig['pred_1d'], pred_3d=sig['pred_3d'], pred_5d=sig['pred_5d'],
            ))

    # ── Force-close all remaining positions at final date ─────────────────────
    last_date = trading_days[-1]
    for pos in list(open_pos):
        cur = price_lut.get((pos.ticker, last_date), pos.entry_fill)
        if pos.direction == 'LONG':
            exit_fill = cur * (1 - SLIPPAGE)
            pnl_pct   = (exit_fill - pos.entry_fill) / pos.entry_fill
            proceeds  = pos.n_shares * exit_fill
        else:
            exit_fill = cur * (1 + SLIPPAGE)
            pnl_pct   = (pos.entry_fill - exit_fill) / pos.entry_fill
            proceeds  = pos.n_shares * (2 * pos.entry_fill - exit_fill)
        cash += proceeds
        closed_trades.append({
            'ticker':      pos.ticker,
            'direction':   pos.direction,
            'entry_date':  pos.entry_date,
            'exit_date':   last_date,
            'entry_fill':  round(pos.entry_fill, 4),
            'exit_fill':   round(exit_fill, 4),
            'n_shares':    pos.n_shares,
            'pnl_pct':     round(pnl_pct, 4),
            'exit_reason': 'end_of_period',
            'hold_days':   pos.hold_days,
            'horizon':     pos.horizon,
            'pred_1d':     round(pos.pred_1d, 6),
            'pred_3d':     round(pos.pred_3d, 6),
            'pred_5d':     round(pos.pred_5d, 6),
        })

    final_value  = cash
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    metrics  = _calc_metrics(equity_curve, closed_trades)
    monthly  = _monthly_returns(equity_curve)

    longs  = [t for t in closed_trades if t['direction'] == 'LONG']
    shorts = [t for t in closed_trades if t['direction'] == 'SHORT']
    t1d    = [t for t in closed_trades if t['hold_days'] == 1]
    t3d    = [t for t in closed_trades if t['hold_days'] == 3]
    t5d    = [t for t in closed_trades if t['hold_days'] == 5]

    def wr(ts):
        return round(sum(1 for t in ts if t['pnl_pct'] > 0) / len(ts), 4) if ts else 0.0

    def avg_ret(ts):
        return round(sum(t['pnl_pct'] for t in ts) / len(ts) * 100, 2) if ts else 0.0

    print(f'    {name}: {len(closed_trades)} trades, '
          f'final=${final_value:.2f}, return={total_return:+.1f}%')

    return {
        'final_value':        round(final_value, 2),
        'total_return_pct':   round(total_return, 2),
        'alpha_pct':          0.0,           # filled in after SPY computed
        'n_trades':           len(closed_trades),
        'n_long':             len(longs),
        'n_short':            len(shorts),
        'n_1d_trades':        len(t1d),
        'n_3d_trades':        len(t3d),
        'n_5d_trades':        len(t5d),
        'win_rate':           wr(closed_trades),
        'win_rate_1d':        wr(t1d),
        'win_rate_3d':        wr(t3d),
        'win_rate_5d':        wr(t5d),
        'long_win_rate':      wr(longs),
        'short_win_rate':     wr(shorts),
        'long_avg_return_pct':  avg_ret(longs),
        'short_avg_return_pct': avg_ret(shorts),
        'monthly_returns':    monthly,
        'trades':             closed_trades,
        **metrics,
    }


def main():
    print('=' * 70)
    print('RSSS Backtest v2 — HISTORICAL SIMULATION 2024-2025')
    print('=' * 70)

    # ── Load locked architecture drop list ────────────────────────────────────
    with open('experiments/phase3_locked_architecture.json') as f:
        arch = json.load(f)
    drop_tickers = set(arch.get('drop_tickers', []))
    print(f'Drop tickers: {sorted(drop_tickers)}')

    # ── Feature store — test period only ──────────────────────────────────────
    print('Loading feature store...')
    df = pd.read_parquet('data/features/features_complete.parquet')
    df_test = df[(df['date'] >= TEST_START) & (df['date'] <= TEST_END)].copy()
    df_test = df_test.dropna(subset=['close', 'atr_14']).reset_index(drop=True)
    print(f'Test rows: {len(df_test)} ({df_test["date"].min()} to {df_test["date"].max()})')

    trading_days = sorted(df_test['date'].unique().tolist())
    print(f'Trading days: {len(trading_days)}')

    # ── Pre-download prices for all tickers in test universe ─────────────────
    sim_tickers = sorted(set(df_test['ticker'].unique()) - drop_tickers)
    print(f'Downloading prices for {len(sim_tickers)} tickers: {sim_tickers}')
    raw = yf.download(
        sim_tickers, start=TEST_START, end='2026-01-15',
        auto_adjust=True, progress=False, threads=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        close_df = raw['Close']
    else:
        col_name = sim_tickers[0] if len(sim_tickers) == 1 else 'Close'
        close_df = raw[['Close']].rename(columns={'Close': col_name})

    price_lut: dict = {}
    for ticker in sim_tickers:
        if ticker not in close_df.columns:
            continue
        series = close_df[ticker].dropna()
        for dt, price in series.items():
            price_lut[(ticker, dt.strftime('%Y-%m-%d'))] = float(price)

    # Backfill from feature store for any missing entries
    for _, row in df_test.iterrows():
        key = (row['ticker'], row['date'])
        if key not in price_lut and row['ticker'] not in drop_tickers:
            price_lut[key] = float(row['close'])

    print(f'Price lookup: {len(price_lut):,} entries')

    # ── Load models ───────────────────────────────────────────────────────────
    models = {}
    for hz in ('1d', '3d', '5d'):
        with open(f'models/registry/model_{hz}.pkl', 'rb') as f:
            m = pickle.load(f)
        try:
            m.set_params(n_jobs=1)
        except Exception:
            pass
        models[hz] = m
    print('Models loaded: model_1d, model_3d, model_5d')

    # ── SPY benchmark ─────────────────────────────────────────────────────────
    spy_raw = yf.download('SPY', start=TEST_START, end='2026-01-01',
                          auto_adjust=True, progress=False)
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = spy_raw.columns.get_level_values(0)
    spy_start  = float(spy_raw['Close'].dropna().iloc[0])
    spy_end    = float(spy_raw['Close'].dropna().iloc[-1])
    spy_return = round((spy_end - spy_start) / spy_start * 100, 2)
    print(f'SPY 2024-2025: start=${spy_start:.2f} → end=${spy_end:.2f} = {spy_return:+.2f}%')

    # ── Run three systems ─────────────────────────────────────────────────────
    print('\nRunning simulations...')

    sys_a = run_system(
        'A_long_dynamic', df_test, models, price_lut, trading_days, drop_tickers,
        long_only=True,  dynamic_hold=True,  use_5d_only=False,
    )
    sys_b = run_system(
        'B_long_short_dynamic', df_test, models, price_lut, trading_days, drop_tickers,
        long_only=False, dynamic_hold=True,  use_5d_only=False,
    )
    sys_c = run_system(
        'C_long_short_fixed5d', df_test, models, price_lut, trading_days, drop_tickers,
        long_only=False, dynamic_hold=False, use_5d_only=True,
    )

    for s in (sys_a, sys_b, sys_c):
        s['alpha_pct'] = round(s['total_return_pct'] - spy_return, 2)

    # ── Save results ──────────────────────────────────────────────────────────
    out = {
        'simulation':      True,
        'note':            ('HISTORICAL SIMULATION — 2024-2025 out-of-sample test period. '
                            'NOT live trading. Past performance does not indicate future results.'),
        'period':          f'{TEST_START} to {TEST_END}',
        'initial_capital': INITIAL_CAPITAL,
        'spy_return_pct':  spy_return,
        'spy_start_price': round(spy_start, 2),
        'spy_end_price':   round(spy_end, 2),
        'drop_tickers':    sorted(drop_tickers),
        'density_gate':    f'post_count_1d >= {DENSITY_GATE}',
        'systems': {
            'A_long_dynamic':       sys_a,
            'B_long_short_dynamic': sys_b,
            'C_long_short_fixed5d': sys_c,
        },
    }

    out_path = Path('experiments/backtest_v2_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nResults saved → {out_path}')

    # ── Print comparison table ────────────────────────────────────────────────
    W = 74
    print('\n' + '═' * W)
    print('RSSS BACKTEST v2 — 2024-2025 OUT-OF-SAMPLE   [HISTORICAL SIMULATION]')
    print('═' * W)
    print()

    header = f'  {"System":<26} {"Return":>8} {"Alpha":>7} {"Sharpe":>7} '
    header += f'{"Sortino":>8} {"MaxDD":>7} {"WinRate":>8} {"Trades":>7}'
    print(header)
    print('  ' + '─' * 70)

    rows = [
        ('SPY Buy & Hold',      spy_return, None,  None,  None, None, None, None),
        ('A  Long+Dynamic',     sys_a['total_return_pct'], sys_a['alpha_pct'],
         sys_a['sharpe_ratio'], sys_a['sortino_ratio'], sys_a['max_drawdown_pct'],
         sys_a['win_rate'], sys_a['n_trades']),
        ('B  Long+Short+Dyn',   sys_b['total_return_pct'], sys_b['alpha_pct'],
         sys_b['sharpe_ratio'], sys_b['sortino_ratio'], sys_b['max_drawdown_pct'],
         sys_b['win_rate'], sys_b['n_trades']),
        ('C  Long+Short+5D',    sys_c['total_return_pct'], sys_c['alpha_pct'],
         sys_c['sharpe_ratio'], sys_c['sortino_ratio'], sys_c['max_drawdown_pct'],
         sys_c['win_rate'], sys_c['n_trades']),
    ]
    for r in rows:
        name, ret, alpha, sharpe, sortino, maxdd, wr, n = r
        print(f'  {name:<26} {ret:>+7.1f}% '
              f'{alpha if alpha is not None else "    ─":>7}  '
              f'{f"{sharpe:.2f}" if sharpe is not None else "─":>7}  '
              f'{f"{sortino:.2f}" if sortino is not None else "─":>8}  '
              f'{f"{maxdd:.1f}%" if maxdd is not None else "─":>7}  '
              f'{f"{wr*100:.1f}%" if wr is not None else "─":>7}  '
              f'{str(n) if n is not None else "─":>6}')

    print()
    print('  System A — Dynamic Hold Breakdown:')
    print(f'    1D: {sys_a["n_1d_trades"]:>3} trades  win={sys_a["win_rate_1d"]*100:.0f}%')
    print(f'    3D: {sys_a["n_3d_trades"]:>3} trades  win={sys_a["win_rate_3d"]*100:.0f}%')
    print(f'    5D: {sys_a["n_5d_trades"]:>3} trades  win={sys_a["win_rate_5d"]*100:.0f}%')

    print()
    print('  System B — Long vs Short:')
    print(f'    Long : {sys_b["n_long"]:>3} trades  '
          f'win={sys_b["long_win_rate"]*100:.0f}%  '
          f'avg={sys_b["long_avg_return_pct"]:+.2f}% per trade')
    print(f'    Short: {sys_b["n_short"]:>3} trades  '
          f'win={sys_b["short_win_rate"]*100:.0f}%  '
          f'avg={sys_b["short_avg_return_pct"]:+.2f}% per trade')

    print()
    print('  System C — Control Breakdown:')
    print(f'    Long : {sys_c["n_long"]:>3} trades  '
          f'win={sys_c["long_win_rate"]*100:.0f}%  '
          f'avg={sys_c["long_avg_return_pct"]:+.2f}% per trade')
    print(f'    Short: {sys_c["n_short"]:>3} trades  '
          f'win={sys_c["short_win_rate"]*100:.0f}%  '
          f'avg={sys_c["short_avg_return_pct"]:+.2f}% per trade')

    # Recommendation
    ranked_ret = sorted(
        [('A', sys_a), ('B', sys_b), ('C', sys_c)],
        key=lambda x: x[1]['total_return_pct'], reverse=True,
    )
    ranked_sh = sorted(
        [('A', sys_a), ('B', sys_b), ('C', sys_c)],
        key=lambda x: x[1]['sharpe_ratio'], reverse=True,
    )
    best_ret = ranked_ret[0]
    best_sh  = ranked_sh[0]
    live_pick = best_sh[0]

    print()
    print('  RECOMMENDATION (HISTORICAL SIMULATION only):')
    print(f'    Best raw return : System {best_ret[0]}  {best_ret[1]["total_return_pct"]:+.1f}%')
    print(f'    Best Sharpe     : System {best_sh[0]}  {best_sh[1]["sharpe_ratio"]:.2f}')
    print(f'    Best for live   : System {live_pick}  '
          f'(Sharpe is more stable than raw return)')

    # A vs C: does dynamic hold help on long-only?
    a_vs_c = sys_a['total_return_pct'] - sys_c['total_return_pct']
    # B vs C: does dynamic hold help on long+short?
    b_vs_c = sys_b['total_return_pct'] - sys_c['total_return_pct']
    # B vs A: does short-selling add value?
    b_vs_a = sys_b['total_return_pct'] - sys_a['total_return_pct']

    print()
    print('  CAUSAL DECOMPOSITION:')
    print(f'    A vs C (+dyn, -shorts)  : {a_vs_c:+.1f}%  '
          f'({"dynamic hold helps" if a_vs_c > 0 else "dynamic hold hurts"} vs fixed 5D)')
    print(f'    B vs C (+dyn, +shorts)  : {b_vs_c:+.1f}%  '
          f'({"dynamic+shorts helps" if b_vs_c > 0 else "dynamic+shorts hurts"} vs fixed 5D)')
    print(f'    B vs A (+shorts only)   : {b_vs_a:+.1f}%  '
          f'({"shorts add value" if b_vs_a > 0 else "shorts destroy value"})')

    print()
    print('═' * W)
    print(f'All results: HISTORICAL SIMULATION. Initial capital: ${INITIAL_CAPITAL:,.0f}')
    print('─' * W)


if __name__ == '__main__':
    main()
