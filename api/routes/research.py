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
def get_backtest_full(system: str = 'A'):
    """
    Backtest v2 results for 2024-2025 out-of-sample period (HISTORICAL SIMULATION).
    system: A | B | C
        A = Long-only, dynamic hold (1D/3D/5D)   [default — best Sharpe]
        B = Long+Short, dynamic hold
        C = Long+Short, fixed 5D only             [control]
    Returns selected system's trades + metrics, plus comparison summary for all three.
    Source: experiments/backtest_v2_results.json
    """
    path = Path('experiments/backtest_v2_results.json')
    if not path.exists():
        return {
            'error':   'Backtest v2 results not found',
            'message': 'Run: python scripts/run_backtest_v2.py',
        }

    with open(path) as f:
        data = json.load(f)

    sys_map = {
        'A': 'A_long_dynamic',
        'B': 'B_long_short_dynamic',
        'C': 'C_long_short_fixed5d',
    }
    sys_key = sys_map.get((system or 'A').upper(), 'A_long_dynamic')

    if sys_key not in data.get('systems', {}):
        return {
            'error':     f'System {system!r} not found',
            'available': list(data.get('systems', {}).keys()),
        }

    selected = data['systems'][sys_key]

    # Compact comparison summary for all three systems
    comparison = {}
    labels = {
        'A_long_dynamic':       'A',
        'B_long_short_dynamic': 'B',
        'C_long_short_fixed5d': 'C',
    }
    descriptions = {
        'A': 'Long+Dynamic',
        'B': 'Long+Short+Dynamic',
        'C': 'Long+Short+Fixed5D',
    }
    for k, v in data['systems'].items():
        lbl = labels.get(k, k)
        comparison[lbl] = {
            'description':      descriptions.get(lbl, lbl),
            'total_return_pct': v['total_return_pct'],
            'alpha_pct':        v['alpha_pct'],
            'sharpe_ratio':     v['sharpe_ratio'],
            'sortino_ratio':    v['sortino_ratio'],
            'max_drawdown_pct': v['max_drawdown_pct'],
            'win_rate':         v['win_rate'],
            'n_trades':         v['n_trades'],
            'profit_factor':    v['profit_factor'],
        }

    # Best system by Sharpe ratio
    best_lbl  = max(comparison, key=lambda k: comparison[k]['sharpe_ratio'])
    best_s    = comparison[best_lbl]
    sys_names = {'A': 'Long+Dynamic', 'B': 'Long+Short+Dynamic', 'C': 'Long+Short+Fixed5D'}
    recommendation = (
        f'System {best_lbl} — {sys_names.get(best_lbl, best_lbl)} '
        f'(best Sharpe {best_s["sharpe_ratio"]:.2f})'
    )

    sel_lbl = (system or 'A').upper()
    selected_data = {
        'n_trades':         selected['n_trades'],
        'n_long':           selected['n_long'],
        'n_short':          selected['n_short'],
        'n_1d_trades':      selected.get('n_1d_trades', 0),
        'n_3d_trades':      selected.get('n_3d_trades', 0),
        'n_5d_trades':      selected.get('n_5d_trades', 0),
        'win_rate':         selected['win_rate'],
        'win_rate_1d':      selected['win_rate_1d'],
        'win_rate_3d':      selected['win_rate_3d'],
        'win_rate_5d':      selected['win_rate_5d'],
        'long_win_rate':    selected['long_win_rate'],
        'short_win_rate':   selected['short_win_rate'],
        'long_avg_return_pct':  selected.get('long_avg_return_pct', 0),
        'short_avg_return_pct': selected.get('short_avg_return_pct', 0),
        'total_return_pct': selected['total_return_pct'],
        'alpha_pct':        selected['alpha_pct'],
        'sharpe_ratio':     selected['sharpe_ratio'],
        'sortino_ratio':    selected['sortino_ratio'],
        'max_drawdown_pct': selected['max_drawdown_pct'],
        'profit_factor':    selected['profit_factor'],
        'monthly_returns':  selected.get('monthly_returns', {}),
    }

    return _sanitize({
        'simulation':       True,
        'note':             data.get('note', ''),
        'period':           data['period'],
        'spy_return_pct':   data['spy_return_pct'],
        'recommendation':   recommendation,
        'selected_system':  sel_lbl,
        'system':           sel_lbl,           # kept for backwards compat
        'selected_data':    selected_data,
        'comparison':       comparison,
        'trades':           selected.get('trades', []),
        # flat fields kept for backwards compat
        **{k: v for k, v in selected_data.items() if k != 'monthly_returns'},
        'monthly_returns':  selected_data['monthly_returns'],
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
