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
