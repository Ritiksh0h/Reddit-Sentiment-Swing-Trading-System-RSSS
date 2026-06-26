"""
Rolling IC + CUSUM signal decay monitor.
Runs weekly to detect if Reddit signal has decayed.

Output:
  GREEN:  rolling IC > 0.02, CUSUM normal
  AMBER:  rolling IC 0.00-0.02 or CUSUM warning
  RED:    rolling IC < 0.00 or CUSUM breach

RED = stop trading Reddit-driven signals
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

WINDOW = 60        # rolling trade window
CUSUM_K = 0.5      # allowance (half sigma)
CUSUM_H = 4.0      # decision threshold


def compute_rolling_ic(trades: list) -> dict:
    """
    Compute rolling Spearman IC on last N trades.
    trades: list of dicts with pred_5d, actual_return_5d
    """
    if len(trades) < 20:
        return {
            'status': 'INSUFFICIENT_DATA',
            'n_trades': len(trades),
            'rolling_ic': None,
            'ic_tstat': None,
        }

    recent = trades[-WINDOW:]
    preds   = [t.get('pred_5d', 0) for t in recent]
    actuals = [t.get('actual_return_5d', 0) for t in recent]

    # Remove missing actuals
    valid = [
        (p, a) for p, a in zip(preds, actuals)
        if a is not None and a != 0
    ]

    if len(valid) < 10:
        return {
            'status': 'INSUFFICIENT_ACTUALS',
            'n_trades': len(valid),
            'rolling_ic': None,
        }

    p_arr, a_arr = zip(*valid)
    ic, pval = spearmanr(p_arr, a_arr)
    tstat = ic * np.sqrt(len(valid) - 2) / np.sqrt(1 - ic ** 2 + 1e-10)

    return {
        'rolling_ic': round(float(ic), 4),
        'ic_pval':    round(float(pval), 4),
        'ic_tstat':   round(float(tstat), 4),
        'n_valid':    len(valid),
    }


def cusum_test(trades: list) -> dict:
    """
    CUSUM control chart on standardized returns.
    Detects persistent shift from expected IC.
    """
    if len(trades) < 20:
        return {'cusum_breach': False, 'cusum_s_neg': 0}

    # Signed IC contribution per trade
    signed = []
    for t in trades[-WINDOW:]:
        pred   = t.get('pred_5d', 0)
        actual = t.get('actual_return_5d')
        if actual is not None:
            signed.append(1.0 if (pred * actual > 0) else -1.0)

    if not signed:
        return {'cusum_breach': False, 'cusum_s_neg': 0}

    mu    = np.mean(signed)
    sigma = max(np.std(signed), 0.1)
    std_signed = [(x - mu) / sigma for x in signed]

    # CUSUM (one-sided negative — detect decay)
    s_neg     = 0.0
    s_neg_max = 0.0
    for x in std_signed:
        s_neg     = max(0, s_neg - x + CUSUM_K)
        s_neg_max = max(s_neg_max, s_neg)

    breach = s_neg > CUSUM_H

    return {
        'cusum_s_neg':     round(s_neg, 3),
        'cusum_s_neg_max': round(s_neg_max, 3),
        'cusum_breach':    breach,
        'cusum_threshold': CUSUM_H,
    }


def _load_trades_from_jsonl(path: Path) -> list:
    trades = []
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    t = json.loads(line)
                    if t.get('status') == 'CLOSED':
                        trades.append(t)
                except Exception:
                    pass
    return trades


def run_decay_monitor() -> dict:
    """
    Main function — run full decay check.
    Returns status dict with GREEN/AMBER/RED.
    """
    trades = _load_trades_from_jsonl(Path('logs/paper_trades_pre_v2.jsonl'))
    trades += _load_trades_from_jsonl(Path('logs/paper_trades.jsonl'))

    ic_result    = compute_rolling_ic(trades)
    cusum_result = cusum_test(trades)

    rolling_ic   = ic_result.get('rolling_ic')
    cusum_breach = cusum_result.get('cusum_breach', False)

    if rolling_ic is None:
        status = 'INSUFFICIENT_DATA'
        color  = 'GREY'
    elif rolling_ic < 0.0 or cusum_breach:
        status = 'SIGNAL_DEAD'
        color  = 'RED'
    elif rolling_ic < 0.02:
        status = 'SIGNAL_WEAK'
        color  = 'AMBER'
    else:
        status = 'SIGNAL_ALIVE'
        color  = 'GREEN'

    action_map = {
        'GREEN': 'Continue trading normally',
        'AMBER': 'Reduce position sizes by 50%',
        'RED':   'STOP Reddit-driven signals',
        'GREY':  'Accumulate more trades',
    }

    result = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'status':    status,
        'color':     color,
        'n_trades':  len(trades),
        **ic_result,
        **cusum_result,
        'action':    action_map.get(color, 'Unknown'),
    }

    log_path = Path('logs/signal_decay_monitor.jsonl')
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, 'a') as f:
        f.write(json.dumps(result) + '\n')

    print(f'\n{"=" * 50}')
    print(f'SIGNAL DECAY MONITOR — {color}')
    print(f'{"=" * 50}')
    print(f'Status:      {status}')
    print(f'Rolling IC:  {rolling_ic}')
    print(f'CUSUM:       {cusum_result["cusum_s_neg"]}')
    print(f'Trades:      {len(trades)}')
    print(f'Action:      {result["action"]}')
    print(f'{"=" * 50}\n')

    return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_decay_monitor()
