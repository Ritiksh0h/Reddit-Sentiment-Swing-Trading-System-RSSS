"""
RSSS API — research, backtest, and source validation routes.
"""
import json
import math
from pathlib import Path

from fastapi import APIRouter

from api._helpers import _sanitize

router = APIRouter()


@router.get('/backtest')
def get_backtest(ticker: str = None, year: int = None):
    """
    Return backtest results from Experiment C.
    Optionally filter by ticker and/or year.
    Source: experiments/experiment_c/results.json
    Historical simulation (2024 data) — not live trading.
    """
    results_path = Path('experiments/experiment_c/results.json')
    if not results_path.exists():
        return {'error': 'Backtest results not found', 'path': str(results_path)}

    with open(results_path) as f:
        results = json.load(f)

    trades = list(results.get('trade_log', []))

    if ticker:
        ticker = ticker.upper()
        trades = [t for t in trades if t.get('ticker') == ticker]
    if year:
        trades = [t for t in trades if t.get('entry_date', '').startswith(str(year))]

    if not trades:
        return {
            'n_trades': 0,
            'filter':   {'ticker': ticker, 'year': year},
            'message':  'No trades match the filter',
            'trades':   [],
        }

    pnls   = [t.get('pnl', t.get('gross_pnl', 0)) for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    total  = sum(pnls)

    equity = [10000.0]
    for p in pnls:
        equity.append(round(equity[-1] + p, 2))

    max_dd = 0.0
    peak   = equity[0]
    for v in equity:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd

    formatted = sorted([{
        'ticker':      t.get('ticker'),
        'entry_date':  t.get('entry_date'),
        'exit_date':   t.get('exit_date'),
        'entry_price': t.get('entry_price'),
        'exit_price':  t.get('exit_price'),
        'pred_return': round(t.get('pred_return', 0) * 100, 2),
        'pnl':         round(t.get('pnl', t.get('gross_pnl', 0)), 2),
        'exit_reason': t.get('exit_reason', 'hold_days'),
        'result':      'WIN' if t.get('pnl', t.get('gross_pnl', 0)) > 0 else 'LOSS',
    } for t in trades], key=lambda x: x['entry_date'])

    def _safe(v):
        return None if (v is None or (isinstance(v, float) and math.isnan(v))) else v

    return _sanitize({
        'source':        'Experiment C — Historical Simulation 2024',
        'filter':        {'ticker': ticker, 'year': year},
        'n_trades':      len(trades),
        'win_rate':      round(len(wins) / len(pnls), 3) if pnls else 0,
        'total_pnl':     round(total, 2),
        'mean_pnl':      round(total / len(pnls), 2) if pnls else 0,
        'max_drawdown':  round(max_dd * 100, 2),
        'profit_factor': _safe(
            round(sum(wins) / abs(sum(losses)), 3) if losses else None
        ),
        'equity_curve':  equity,
        'trades':        formatted,
        'full_stats': {
            'ic_test':        results.get('ic_test'),
            'sharpe_ratio':   results.get('sharpe_ratio'),
            'total_return':   results.get('total_return'),
            'spy_return':     results.get('spy_return'),
            'alpha':          results.get('alpha'),
            'n_trades_total': len(results.get('trade_log', [])),
        } if not ticker and not year else None,
    })


@router.get('/backtest-full')
def get_backtest_full(year: int = None, month: int = None):
    """
    Walk-forward backtest results for 2024-2025 out-of-sample period.
    Optionally filter by year (2024/2025) and/or month (1-12).
    """
    path = Path('experiments/backtest_results.json')
    if not path.exists():
        return {
            'error':   'Backtest results not found',
            'message': 'Run: python scripts/run_backtest.py',
        }

    with open(path) as f:
        data = json.load(f)

    trades = list(data.get('trades', []))

    if year:
        trades = [t for t in trades if t.get('entry_date', '').startswith(str(year))]
    if month and year:
        prefix = f'{year}-{month:02d}'
        trades = [t for t in trades if t.get('entry_date', '').startswith(prefix)]

    if not trades:
        return {
            'n_trades': 0,
            'filter':   {'year': year, 'month': month},
            'message':  'No trades match the filter',
            'trades':   [],
        }

    pnls   = [t.get('pnl_dollars', 0) for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    total  = sum(pnls)
    wr     = round(len(wins) / len(pnls), 3) if pnls else 0

    month_key = (f'{year}-{month:02d}' if (year and month) else
                 str(year) if year else None)
    monthly_return = None
    if month_key and len(month_key) == 7:
        monthly_return = data.get('monthly_returns', {}).get(month_key)
    elif year:
        yr_data = data.get('annual_breakdown', {}).get(str(year), {})
        monthly_return = yr_data.get('return_pct')

    pf = round(sum(wins) / abs(sum(losses)), 3) if losses else None

    return _sanitize({
        'source':        'Walk-Forward Backtest 2024-2025 (SIMULATION)',
        'filter':        {'year': year, 'month': month},
        'n_trades':      len(trades),
        'win_rate':      wr,
        'total_pnl':     round(total, 2),
        'mean_pnl':      round(total / len(pnls), 2) if pnls else 0,
        'period_return': monthly_return,
        'profit_factor': pf,
        'trades':        trades,
        'full_stats': {
            'total_return_pct': data.get('total_return_pct'),
            'spy_return_pct':   data.get('spy_return_pct'),
            'alpha':            data.get('alpha'),
            'sharpe_ratio':     data.get('sharpe_ratio'),
            'sortino_ratio':    data.get('sortino_ratio'),
            'max_drawdown_pct': data.get('max_drawdown_pct'),
            'n_trades_total':   data.get('n_trades'),
            'annual_2024':      data.get('annual_breakdown', {}).get('2024'),
            'annual_2025':      data.get('annual_breakdown', {}).get('2025'),
        } if not year and not month else None,
    })


@router.get('/research-findings')
def get_research_findings():
    """Return source validation results. Run validate_sources.py first."""
    path = Path('experiments/source_validation/results.json')
    if not path.exists():
        return {
            'status':  'not_run',
            'message': 'Run: python experiments/source_validation/validate_sources.py',
        }
    with open(path) as f:
        data = json.load(f)
    return _sanitize(data)
