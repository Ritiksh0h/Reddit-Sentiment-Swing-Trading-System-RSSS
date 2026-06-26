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
        'spy_return':      49.7,
        'systems': {
            'A': {
                'name':        'Long+Dynamic (Rank-Based, Phase 5)',
                'return_pct':  36.5,
                'alpha':       -11.3,
                'sharpe':      1.36,
                'max_dd':      -14.1,
                'win_rate':    57.5,
                'trades':      141,
                'description': 'Core-satellite 70/30, rank-based signals, vol-targeted sizing, '
                               '20-day MA filter, earnings filter, regime filter, long-only. '
                               '18 features (Phase 5): dist_from_20ma_pct + pead_proxy added.',
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
            'pooled_sharpe':    0.84,
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
    Source priority: MongoDB backtest_results → JSON file → hardcoded fallback.
    """
    # ── Try MongoDB first; populate `_mongo_data` for use below ─────────────
    _mongo_data = None
    try:
        from api.db import get_mongo_db
        from pymongo import DESCENDING
        mdb = get_mongo_db()
        if mdb is not None:
            doc = mdb['backtest_results'].find_one(
                {'version': 'v2'}, {'_id': 0},
                sort=[('created_at', DESCENDING)],
            )
            if doc and doc.get('systems'):
                doc.pop('created_at', None)
                doc.pop('version', None)
                _mongo_data = doc
    except Exception:
        pass

    path = Path('experiments/backtest_v2_results.json')
    if not path.exists() and _mongo_data is None:
        # Fall back to hardcoded numbers from /backtest
        bt = get_backtest()
        sys_key = (system or 'A').upper()
        sys_data = bt['systems'].get(sys_key, bt['systems']['A'])
        return _sanitize({
            'simulation':      True,
            'note':            'Fallback to hardcoded backtest summary (backtest_v2_results.json not found)',
            'period':          bt['period'],
            'spy_return_pct':  bt['spy_return'],
            'recommendation':  f'System A — {sys_data["name"]} (best Sharpe {sys_data["sharpe"]:.2f})',
            'selected_system': sys_key,
            'system':          sys_key,
            'selected_data': {
                'n_trades':         sys_data['trades'],
                'win_rate':         sys_data['win_rate'] / 100,
                'total_return_pct': sys_data['return_pct'],
                'alpha_pct':        sys_data['alpha'],
                'sharpe_ratio':     sys_data['sharpe'],
                'max_drawdown_pct': sys_data['max_dd'],
                'monthly_returns':  {},
            },
            'comparison': {
                k: {
                    'description':      v['name'],
                    'total_return_pct': v['return_pct'],
                    'alpha_pct':        v['alpha'],
                    'sharpe_ratio':     v['sharpe'],
                    'max_drawdown_pct': v['max_dd'],
                    'win_rate':         v['win_rate'] / 100,
                    'n_trades':         v['trades'],
                }
                for k, v in bt['systems'].items()
            },
            'trades':           [],
            'n_trades':         sys_data['trades'],
            'win_rate':         sys_data['win_rate'] / 100,
            'total_return_pct': sys_data['return_pct'],
            'alpha_pct':        sys_data['alpha'],
            'sharpe_ratio':     sys_data['sharpe'],
            'max_drawdown_pct': sys_data['max_dd'],
            'monthly_returns':  {},
        })

    if _mongo_data is not None:
        data = _mongo_data
    else:
        with open(path) as f:
            data = json.load(f)

    sys_map = {
        'A': 'A_rank_dynamic',
        'B': 'B_rank_fixed5d',
    }
    sys_key = sys_map.get((system or 'A').upper(), 'A_rank_dynamic')

    if sys_key not in data.get('systems', {}):
        return {
            'error':     f'System {system!r} not found',
            'available': list(data.get('systems', {}).keys()),
        }

    selected = data['systems'][sys_key]

    # Compact comparison summary for available systems
    comparison = {}
    labels = {
        'A_rank_dynamic':  'A',
        'B_rank_fixed5d':  'B',
    }
    descriptions = {
        'A': 'Long+Dynamic (Rank-Based)',
        'B': 'Long+Fixed Threshold',
    }
    for k, v in data['systems'].items():
        lbl = labels.get(k, k)
        comparison[lbl] = {
            'description':      descriptions.get(lbl, lbl),
            'total_return_pct': v['total_return_pct'],
            'alpha_pct':        v.get('alpha_vs_spy_pct', v.get('alpha_pct', 0)),
            'sharpe_ratio':     v['sharpe_ratio'],
            'sortino_ratio':    v.get('sortino_ratio', 0),
            'max_drawdown_pct': v['max_drawdown_pct'],
            'win_rate':         v['win_rate'],
            'n_trades':         v['n_trades'],
            'profit_factor':    v.get('profit_factor', 0),
        }

    # Best system by Sharpe ratio
    best_lbl  = max(comparison, key=lambda k: comparison[k]['sharpe_ratio'])
    best_s    = comparison[best_lbl]
    sys_names = {'A': 'Long+Dynamic (Rank-Based)', 'B': 'Long+Fixed Threshold'}
    recommendation = (
        f'System {best_lbl} — {sys_names.get(best_lbl, best_lbl)} '
        f'(best Sharpe {best_s["sharpe_ratio"]:.2f})'
    )

    sel_lbl = (system or 'A').upper()
    selected_data = {
        'n_trades':         selected['n_trades'],
        'n_long':           selected.get('n_long', selected['n_trades']),
        'n_short':          selected.get('n_short', 0),
        'n_1d_trades':      selected.get('n_1d_trades', 0),
        'n_3d_trades':      selected.get('n_3d_trades', 0),
        'n_5d_trades':      selected.get('n_5d_trades', selected['n_trades']),
        'win_rate':         selected['win_rate'],
        'win_rate_1d':      selected.get('win_rate_1d', 0),
        'win_rate_3d':      selected.get('win_rate_3d', 0),
        'win_rate_5d':      selected.get('win_rate_5d', selected['win_rate']),
        'long_win_rate':    selected.get('long_win_rate', selected['win_rate']),
        'short_win_rate':   selected.get('short_win_rate', 0),
        'long_avg_return_pct':  selected.get('long_avg_return_pct', 0),
        'short_avg_return_pct': selected.get('short_avg_return_pct', 0),
        'total_return_pct': selected['total_return_pct'],
        'alpha_pct':        selected.get('alpha_vs_spy_pct', selected.get('alpha_pct', 0)),
        'sharpe_ratio':     selected['sharpe_ratio'],
        'sortino_ratio':    selected.get('sortino_ratio', 0),
        'max_drawdown_pct': selected['max_drawdown_pct'],
        'profit_factor':    selected.get('profit_factor', 0),
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
