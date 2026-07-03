"""
Paper trading tracker.
Records simulated PnL and computes performance vs SPY benchmark.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)

PERF_FILE      = 'data/live/paper_performance.json'
PERF_JSONL     = 'data/live/paper_performance.jsonl'


def record_daily_snapshot(
    portfolio_value: float,
    starting_capital: float,
    n_trades_today: int,
    actions: list,
    date: str = None,
) -> dict:
    """
    Record end-of-day portfolio snapshot vs SPY benchmark.
    Appends to data/paper_performance.jsonl (append-only, one record per day).
    """
    if date is None:
        date = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    try:
        spy = yf.download('SPY', period='2d', auto_adjust=True, progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        spy_return_today = float(
            (spy['Close'].iloc[-1] - spy['Close'].iloc[0]) / spy['Close'].iloc[0]
        )
    except Exception as e:
        logger.warning(f'spy_fetch_failed error={e}')
        spy_return_today = None

    portfolio_return = (portfolio_value - starting_capital) / starting_capital

    snapshot = {
        'date':              date,
        'portfolio_value':   round(portfolio_value, 2),
        'starting_capital':  round(starting_capital, 2),
        'portfolio_return':  round(portfolio_return, 4),
        'spy_return_today':  round(spy_return_today, 4) if spy_return_today is not None else None,
        'alpha':             round(portfolio_return - spy_return_today, 4)
                             if spy_return_today is not None else None,
        'n_trades_today':   n_trades_today,
        'actions':          actions,
        'timestamp':        datetime.now(timezone.utc).isoformat(),
    }

    Path('data/live').mkdir(parents=True, exist_ok=True)
    with open(PERF_JSONL, 'a') as f:
        f.write(json.dumps(snapshot) + '\n')

    # Persist to PostgreSQL portfolio_snapshots (non-blocking)
    try:
        from api.db import insert_portfolio_snapshot  # noqa: PLC0415
        insert_portfolio_snapshot({
            'snapshot_date':    date,
            'equity':           round(portfolio_value, 2),
            'cash':             None,
            'position_value':   None,
            'total_return_pct': round(portfolio_return * 100, 4),
            'spy_return_today': round(spy_return_today, 4) if spy_return_today is not None else None,
            'alpha':            snapshot.get('alpha'),
            'n_positions':      n_trades_today,
            'regime_label':     None,
        })
    except Exception as _db_exc:
        logger.debug(f'db_snapshot_skipped: {_db_exc}')

    logger.info(f'daily_snapshot_recorded portfolio_return={portfolio_return:.4f} '
                f'spy_return={spy_return_today}')
    return snapshot


def record_daily_pnl(
    portfolio_value: float,
    prev_portfolio_value: float,
    date: str,
) -> float:
    """Record daily PnL fraction. Returns daily return."""
    if prev_portfolio_value <= 0:
        return 0.0
    daily_return = (portfolio_value - prev_portfolio_value) / prev_portfolio_value

    Path('data/live').mkdir(parents=True, exist_ok=True)
    perf = {}
    if Path(PERF_FILE).exists():
        with open(PERF_FILE) as f:
            perf = json.load(f)

    perf[date] = {
        'portfolio_value': round(portfolio_value, 2),
        'daily_return':    round(daily_return, 6),
        'recorded_at':     datetime.now(timezone.utc).isoformat(),
    }

    with open(PERF_FILE, 'w') as f:
        json.dump(perf, f, indent=2)

    return daily_return


def compute_summary(initial_capital: float = 100000.0) -> dict:
    """Compute paper trading summary vs SPY."""
    if not Path(PERF_FILE).exists():
        return {'error': 'No performance data yet'}

    with open(PERF_FILE) as f:
        perf = json.load(f)

    if not perf:
        return {'error': 'No performance data yet'}

    dates   = sorted(perf.keys())
    returns = [perf[d]['daily_return'] for d in dates]

    final_value  = perf[dates[-1]]['portfolio_value']
    total_return = (final_value - initial_capital) / initial_capital

    import numpy as np
    arr = np.array(returns)
    sharpe = (arr.mean() / arr.std() * (252 ** 0.5)) if arr.std() > 0 else 0.0

    # Fetch SPY return over same period
    try:
        spy = yf.download('SPY', start=dates[0], end=dates[-1],
                          auto_adjust=True, progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        spy_return = float(spy['Close'].iloc[-1] / spy['Close'].iloc[0] - 1)
    except Exception:
        spy_return = None

    return {
        'n_days':        len(dates),
        'start_date':    dates[0],
        'end_date':      dates[-1],
        'initial_cap':   initial_capital,
        'final_value':   round(final_value, 2),
        'total_return':  round(total_return, 4),
        'sharpe_ratio':  round(sharpe, 3),
        'spy_return':    round(spy_return, 4) if spy_return is not None else None,
        'alpha':         round(total_return - spy_return, 4) if spy_return is not None else None,
    }
