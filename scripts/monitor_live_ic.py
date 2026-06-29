"""
Monitor live IC from paper trading execution log.
Run weekly: python scripts/monitor_live_ic.py

Compares predicted_return_5d vs actual 5-day return for closed trades.
Uses Phase 4 monitoring gates:
    Green:  30-day IC > 0.03 — model working, continue
    Amber:  30-day IC 0.01-0.03 — watch closely, do not intervene yet
    Red:    30-day IC < 0.01 — Fix 3 triggered (switch to 17 features)

Appends results to logs/ic_monitor.jsonl.
"""
import json
import logging
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
import yfinance as yf

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

GREEN_GATE       = 0.03
AMBER_GATE       = 0.01
BASELINE_TEST_IC = 0.0562   # V2 model_5d test IC (retrained 2026-06-29)


def load_open_signals() -> list:
    """Load all OPEN signals from execution log."""
    path = Path('logs/paper_trades.jsonl')
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get('action') == 'OPEN':
                records.append(r)
    return records


def compute_live_ic(lookback_days: int = 30) -> dict:
    """
    Compute IC on paper trades from the last `lookback_days` days.

    For each OPEN signal, fetches actual 5-day return from yfinance
    and compares to predicted_return_5d.

    Needs 5+ completed trades for a reliable estimate.
    """
    signals = load_open_signals()
    if not signals:
        result = {'error': 'No OPEN signals in execution log yet. '
                           'System needs to run for 30+ days first.'}
        print(json.dumps(result, indent=2))
        return result

    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
    recent = [s for s in signals if s.get('date', '') >= cutoff]

    if len(recent) < 5:
        result = {
            'warning':   f'Only {len(recent)} signals in last {lookback_days} days. '
                         'Need 5+ for reliable IC estimate.',
            'n_signals': len(recent),
            'note':      'Check back after more trading days accumulate.',
        }
        print(json.dumps(result, indent=2))
        return result

    predictions = []
    actuals     = []
    skipped     = 0

    for signal in recent:
        ticker   = signal['ticker']
        sig_date = signal['date']
        pred_ret = signal.get('predicted_return_5d', 0)

        try:
            mkt = yf.download(ticker, start=sig_date,
                              auto_adjust=True, progress=False)
            if isinstance(mkt.columns, pd.MultiIndex):
                mkt.columns = mkt.columns.get_level_values(0)
            if len(mkt) < 6:
                skipped += 1
                continue  # trade not yet completed (< 5 trading days elapsed)
            actual_ret = float(mkt['Close'].iloc[5] / mkt['Close'].iloc[0] - 1)
            predictions.append(pred_ret)
            actuals.append(actual_ret)
        except Exception as e:
            logger.debug(f'skip {ticker} {sig_date}: {e}')
            skipped += 1
            continue

    if len(predictions) < 5:
        result = {
            'warning':      'Insufficient completed trades for IC computation.',
            'n_signals':    len(recent),
            'n_completed':  len(predictions),
            'n_skipped':    skipped,
            'note':         f'{skipped} trades not yet completed (< 5 trading days).',
        }
        print(json.dumps(result, indent=2))
        return result

    ic, pval = stats.spearmanr(predictions, actuals)
    ic   = float(ic)
    pval = float(pval)

    # Gate classification
    if ic >= GREEN_GATE:
        gate   = 'GREEN'
        action = 'Model working — continue paper trading'
    elif ic >= AMBER_GATE:
        gate   = 'AMBER'
        action = 'Weak signal — watch closely, do not intervene yet'
    else:
        gate   = 'RED'
        action = ('Fix 3 triggered — if RED for 2 consecutive weeks, '
                  'run scripts/fix3_switch_to_17_features.py')

    result = {
        'date':            date.today().isoformat(),
        'lookback_days':   lookback_days,
        'n_signals':       len(recent),
        'n_completed':     len(predictions),
        'n_skipped':       skipped,
        'live_ic':         round(ic, 4),
        'p_value':         round(pval, 3),
        'gate':            gate,
        'action':          action,
        'baseline_ic':     BASELINE_TEST_IC,
        'ic_vs_baseline':  round(ic - BASELINE_TEST_IC, 4),
        'mean_pred_ret':   round(float(np.mean(predictions)), 4),
        'mean_actual_ret': round(float(np.mean(actuals)), 4),
    }

    print(f'\n{"="*50}')
    print(f'LIVE IC MONITOR — {result["date"]}')
    print(f'{"="*50}')
    print(f'Lookback:      {lookback_days} days')
    print(f'Signals used:  {result["n_completed"]} of {result["n_signals"]}')
    print(f'Live IC:       {result["live_ic"]:.4f}')
    print(f'p-value:       {result["p_value"]:.3f}')
    print(f'Gate:          {gate}')
    print(f'Action:        {action}')
    print(f'Baseline IC:   {BASELINE_TEST_IC:.4f} (phase3 test set)')
    print(f'vs Baseline:   {result["ic_vs_baseline"]:+.4f}')
    print(f'{"="*50}\n')

    Path('logs').mkdir(exist_ok=True)
    with open('logs/ic_monitor.jsonl', 'a') as f:
        f.write(json.dumps(result) + '\n')

    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=30,
                        help='Lookback window in days (default: 30)')
    args = parser.parse_args()
    compute_live_ic(lookback_days=args.days)
