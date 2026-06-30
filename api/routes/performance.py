"""
RSSS API — performance, accuracy, and monitoring routes.
"""
import json
import logging
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, Query

from api._helpers import _load_portfolio, _sanitize

router = APIRouter()

_log = logging.getLogger(__name__)


@router.get('/performance')
def get_performance():
    """Paper trading performance summary."""
    state  = _load_portfolio()
    closed = state.get('closed_trades', [])
    if not closed:
        return {'message': 'No closed trades yet', 'n_trades': 0}

    pnls     = [t.get('pnl_pct', 0) for t in closed]
    wins     = [p for p in pnls if p > 0]
    loss_sum = sum(p for p in pnls if p < 0)

    return {
        'n_trades':      len(pnls),
        'win_rate':      round(len(wins) / len(pnls), 3) if pnls else 0,
        'mean_pnl':      round(sum(pnls) / len(pnls), 4) if pnls else 0,
        'total_pnl':     round(sum(pnls), 4),
        'profit_factor': round(sum(wins) / abs(loss_sum), 3) if loss_sum != 0 else None,
    }


@router.get('/signal-accuracy')
def get_signal_accuracy():
    """
    Per-horizon directional accuracy (1D, 3D, 5D) from closed paper trades.
    1D/3D actual prices fetched via yfinance; 5D uses recorded pnl_pct.
    Signal lookup falls back to logs/paper_trades.jsonl OPEN records.
    Results cached in data/processed/signal_accuracy_cache.json.
    """
    import yfinance as yf
    import pandas as pd
    from datetime import datetime, timedelta, date as _date

    CACHE_PATH = Path('data/processed/signal_accuracy_cache.json')

    state  = _load_portfolio()
    closed = state.get('closed_trades', [])
    real   = [t for t in closed if abs(t.get('pnl_pct', 0)) > 0.0001]

    if not real:
        return {
            'n_evaluated': len(closed), 'n_real': 0,
            'message': 'No trades with real PnL yet',
            '1D': '—', '3D': '—', '5D': '—',
            'interpretation': 'Accumulating signals — need 10+ closed trades',
        }

    signal_lookup: dict = {}
    log_path = Path('logs/paper_trades.jsonl')
    if log_path.exists():
        for line in log_path.read_text().strip().split('\n'):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if r.get('action') == 'OPEN':
                    signal_lookup[f"{r['ticker']}_{r['date']}"] = r.get('signal', 'BULLISH')
            except Exception:
                pass

    cache: dict = {}
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH) as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    today      = _date.today()
    needs_save = False

    for trade in real:
        ticker     = trade.get('ticker', '')
        entry_date = trade.get('entry_date', '')
        if not ticker or not entry_date:
            continue
        cache_key = f'{ticker}_{entry_date}'
        if cache_key in cache:
            continue
        if (today - _date.fromisoformat(entry_date)).days < 7:
            continue
        try:
            hist = yf.download(
                ticker,
                start=entry_date,
                end=(datetime.fromisoformat(entry_date) + timedelta(days=20)).strftime('%Y-%m-%d'),
                auto_adjust=True,
                progress=False,
            )
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            if len(hist) < 2:
                continue
            c0  = float(hist['Close'].iloc[0])
            r1d = (float(hist['Close'].iloc[1]) - c0) / c0 if len(hist) > 1 else None
            r3d = (float(hist['Close'].iloc[3]) - c0) / c0 if len(hist) > 3 else None
            sig = signal_lookup.get(cache_key, 'BULLISH')
            cache[cache_key] = {
                'ticker':     ticker,
                'entry_date': entry_date,
                'signal':     sig,
                'r1d':        round(r1d, 6) if r1d is not None else None,
                'r3d':        round(r3d, 6) if r3d is not None else None,
            }
            needs_save = True
        except Exception as e:
            _log.warning(f'signal_accuracy_fetch {ticker} {entry_date}: {e}')

    if needs_save:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, 'w') as f:
            json.dump(cache, f, indent=2)

    def _accuracy(pairs):
        evaluated = [(r, s) for r, s in pairs if r is not None]
        if not evaluated:
            return None, 0
        correct = sum(
            1 for r, s in evaluated
            if (s == 'BEARISH' and r < 0) or (s != 'BEARISH' and r > 0)
        )
        return round(correct / len(evaluated), 3), len(evaluated)

    pairs_1d, pairs_3d, pairs_5d = [], [], []
    for trade in real:
        key   = f'{trade["ticker"]}_{trade["entry_date"]}'
        entry = cache.get(key, {})
        sig   = entry.get('signal') or signal_lookup.get(key, 'BULLISH')
        pairs_1d.append((entry.get('r1d'), sig))
        pairs_3d.append((entry.get('r3d'), sig))
        pairs_5d.append((trade.get('pnl_pct'), sig))

    acc_1d, n_1d = _accuracy(pairs_1d)
    acc_3d, n_3d = _accuracy(pairs_3d)
    acc_5d, n_5d = _accuracy(pairs_5d)

    def _fmt(acc):
        return '—' if acc is None else f'{round(acc * 100, 1)}%'

    pnls     = [t.get('pnl_pct', 0) for t in real]
    mean_pnl = round(sum(pnls) / len(pnls) * 100, 2) if pnls else 0
    wins     = sum(1 for p in pnls if p > 0)

    lag = (acc_1d is not None and acc_5d is not None and acc_5d > acc_1d + 0.05)
    interpretation = (
        f'{wins}/{len(real)} trades profitable. Mean PnL={mean_pnl:+.2f}%. '
        + ('1D low + 5D high = multi-day lag in signal (expected).'
           if lag else 'Too few trades for conclusions — need 30+.')
    )

    return _sanitize({
        'n_evaluated':    len(real),
        'n_zero_pnl':     len(closed) - len(real),
        'win_rate':       acc_5d or 0.0,
        'mean_pnl_pct':   mean_pnl,
        '1D':             _fmt(acc_1d),
        '3D':             _fmt(acc_3d),
        '5D':             _fmt(acc_5d),
        'n_evaluated_1d': n_1d,
        'n_evaluated_3d': n_3d,
        'n_evaluated_5d': n_5d,
        'interpretation': interpretation,
        'trades':         real,
    })


@router.get('/ic-monitor')
def get_ic_monitor():
    """Return IC monitor history from logs/ic_monitor.jsonl."""
    path = Path('logs/ic_monitor.jsonl')
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


@router.get('/backfill-log')
def get_backfill_log():
    """Return last 100 lines from the backfill test log."""
    path = Path('logs/backfill_test.log')
    if not path.exists():
        return {'lines': []}
    lines = path.read_text().strip().split('\n')
    return {'lines': lines[-100:]}


@router.post('/backfill')
def run_backfill_endpoint(
    start: str = Query(..., description='Start date YYYY-MM-DD'),
    end:   str = Query(..., description='End date YYYY-MM-DD'),
):
    """Trigger a backfill test run (async — returns immediately)."""
    project_root = str(Path(__file__).parent.parent.parent)
    try:
        subprocess.Popen(
            [sys.executable, 'scripts/test_historical_run.py',
             '--start', start, '--end', end, '--no-restore'],
            cwd=project_root,
        )
        return {'status': 'started', 'start': start, 'end': end}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


@router.get('/dashboard-stats')
def get_dashboard_stats():
    """
    Computed dashboard statistics from experiment_c backtest + live paper performance.
    Returns equity curve, drawdown, monthly returns heatmap, return distribution,
    rolling win rate, and summary stats (Sharpe, Sortino, Calmar, Profit Factor).
    Uses precomputed stats from results.json — no recomputation of Sharpe/Sortino.
    """
    ec_path = Path('experiments/experiment_c/results.json')
    if not ec_path.exists():
        return {'error': 'backtest data not found'}

    with open(ec_path) as f:
        ec = json.load(f)

    trades    = sorted(ec.get('trade_log', []), key=lambda x: x['exit_date'])
    raw_curve = ec.get('equity_curve', [])
    if not trades or not raw_curve:
        return {'error': 'no trade log found'}

    # Normalize equity curve to 10000 base, paired with exit dates
    init      = raw_curve[0] if raw_curve[0] != 0 else 1.0
    spy_total = ec.get('spy_return', 0.261)
    spy_end   = 10000 * (1 + spy_total)
    n_pts     = len(raw_curve)
    dates     = [trades[0]['entry_date']] + [t['exit_date'] for t in trades]

    curve = []
    for i, (val, date) in enumerate(zip(raw_curve, dates)):
        curve.append({
            'date':      date,
            'portfolio': round(10000 * val / init, 2),
            'spy':       round(10000 + (spy_end - 10000) * i / max(n_pts - 1, 1), 2),
        })

    # Drawdown from normalized curve
    peak = 10000.0
    drawdown = []
    for pt in curve:
        peak = max(peak, pt['portfolio'])
        drawdown.append({'date': pt['date'], 'dd': round((pt['portfolio'] - peak) / peak * 100, 3)})

    # Per-trade pnl_pct from cost basis: pnl / (entry_price × shares)
    pnl_pcts = []
    for t in trades:
        cost = t.get('entry_price', 0) * t.get('shares', 0)
        pnl_pcts.append(round(t.get('pnl', 0) / cost * 100 if cost > 0 else 0, 2))

    # Monthly returns — read directly from normalized equity curve to avoid cost-basis inflation
    # curve[0] = initial equity; curve[i+1] = equity after trades[i] exits
    month_names  = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    eq_list      = [pt['portfolio'] for pt in curve]
    month_bounds: dict = {}
    for i, t in enumerate(trades):
        ym = t['exit_date'][:7]
        if ym not in month_bounds:
            month_bounds[ym] = {'start_idx': i, 'end_idx': i}
        month_bounds[ym]['end_idx'] = i

    monthly_returns: dict = {}
    for ym, idx in month_bounds.items():
        start_val = eq_list[idx['start_idx']]      # equity before first exit this month
        end_val   = eq_list[idx['end_idx'] + 1]    # equity after last exit this month
        ret = (end_val - start_val) / start_val * 100 if start_val > 0 else 0
        year, month = ym.split('-')
        monthly_returns.setdefault(year, {})[month_names[int(month) - 1]] = round(ret, 1)

    # 2026 live data — portfolio_return is cumulative so use first/last pv per month
    perf_path = Path('data/live/paper_performance.jsonl')
    if perf_path.exists():
        live_month_pv: dict = {}  # ym -> {'first': pv, 'last': pv}
        for line in perf_path.read_text().strip().split('\n'):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                d = r.get('date', '')
                pv = r.get('portfolio_value', 0)
                if len(d) >= 7 and pv > 0:
                    ym = d[:7]
                    if ym not in live_month_pv:
                        live_month_pv[ym] = {'first': pv, 'last': pv}
                    live_month_pv[ym]['last'] = pv
            except Exception:
                pass
        for ym, pvs in live_month_pv.items():
            year, month = ym.split('-')
            ret = (pvs['last'] - pvs['first']) / pvs['first'] * 100 if pvs['first'] > 0 else 0
            monthly_returns.setdefault(year, {})[month_names[int(month) - 1]] = round(ret, 1)

    # Rolling 20-trade win rate
    rolling_wr = []
    for i in range(len(pnl_pcts)):
        subset = pnl_pcts[max(0, i - 19):i + 1]
        wr = sum(1 for p in subset if p > 0) / len(subset) * 100 if subset else 0
        rolling_wr.append({'date': trades[i]['exit_date'], 'win_rate': round(wr, 1)})

    # Avg win/loss/expectancy from cost-basis pnl_pcts
    wins   = [p for p in pnl_pcts if p > 0]
    losses = [p for p in pnl_pcts if p < 0]
    wr     = ec.get('win_rate', len(wins) / len(pnl_pcts) if pnl_pcts else 0)
    avg_win  = round(sum(wins)   / len(wins),   2) if wins   else 0
    avg_loss = round(sum(losses) / len(losses), 2) if losses else 0
    expect   = round(wr * avg_win + (1 - wr) * avg_loss, 2)

    total_return_pct = ec.get('total_return', 0) * 100
    max_dd_pct       = ec.get('max_drawdown',  0) * 100  # negative value
    calmar = round(abs(total_return_pct / max_dd_pct), 2) if max_dd_pct != 0 else 0

    return _sanitize({
        'equity_curve':        curve,
        'drawdown':            drawdown,
        'monthly_returns':     monthly_returns,
        'return_distribution': pnl_pcts,
        'rolling_win_rate':    rolling_wr,
        'stats': {
            'sharpe':        round(ec.get('sharpe_ratio',  0), 2),
            'sortino':       round(ec.get('sortino_ratio', 0), 2),
            'calmar':        calmar,
            'profit_factor': round(ec.get('profit_factor', 0), 2),
            'win_rate':      round(wr * 100, 1),
            'avg_win':       avg_win,
            'avg_loss':      avg_loss,
            'expectancy':    expect,
            'n_trades':      ec.get('n_trades', len(trades)),
            'total_return':  round(total_return_pct, 2),
            'max_drawdown':  round(max_dd_pct, 2),
        },
        'period': {'start': trades[0]['entry_date'], 'end': trades[-1]['exit_date']},
    })


@router.get('/model-metadata')
def get_model_metadata():
    """Return model training metadata. Reads v2 metadata first, falls back to phase3 baseline."""
    v2_path = Path('models/training_metadata_v2.json')
    if v2_path.exists():
        return json.loads(v2_path.read_text())

    registry_path = Path('models/registry/phase3_model_baseline.json')
    if registry_path.exists():
        return json.loads(registry_path.read_text())

    return {
        'status':        'models_not_found',
        'message':       'Run train_models_v2.py to generate models',
        'expected_path': 'models/training_metadata_v2.json',
    }
