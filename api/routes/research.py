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
def get_backtest():
    """
    V2 backtest summary — 2024-2025 out-of-sample, rank-based core-satellite.
    Three system variants (A/B/C) plus walk-forward validation results.
    """
    return {
        'system':          'A — Long+Dynamic (Rank-Based)',
        'period':          '2024-2025 Out-of-Sample',
        'version':         'v2',
        'ticker_universe': 'RSSS 29-ticker universe (NVDA/AAPL/TSLA etc.)',
        'spy_return':      47.8,
        'systems': {
            'A': {
                'name':        'Long+Dynamic (Rank-Based)',
                'return_pct':  35.6,
                'alpha':       -12.2,
                'sharpe':      1.32,
                'max_dd':      -14.1,
                'win_rate':    57.5,
                'trades':      167,
                'description': 'Core-satellite 70/30, rank-based signals, vol-targeted sizing, '
                               'regime filter, long-only',
            },
            'B': {
                'name':        'Long+Dynamic (Fixed Threshold)',
                'return_pct':  33.6,
                'alpha':       -14.2,
                'sharpe':      1.27,
                'max_dd':      -14.1,
                'win_rate':    57.9,
                'trades':      19,
                'description': 'Same as A but uses fixed threshold pred_5d > 0.7% '
                               'instead of rank-based',
            },
            'C': {
                'name':        'Original Baseline',
                'return_pct':  23.2,
                'alpha':       -24.7,
                'sharpe':      0.93,
                'max_dd':      -9.9,
                'win_rate':    54.4,
                'trades':      57,
                'description': 'Pre-fix system — absolute threshold, equal sizing, '
                               'no regime filter',
            },
        },
        'walk_forward': {
            'folds':            23,
            'pooled_sharpe':    0.86,
            'pct_profitable':   74,
            'bear_2022_return': -3.9,
            'wfe':              1.25,
            'gates_passed':     '4/5',
        },
    }


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
