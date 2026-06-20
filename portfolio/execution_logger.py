"""
Execution logger.
Appends structured JSON records to logs/paper_trades.jsonl.

Every signal (not just executed trades) is logged.
This lets you reconstruct exactly what the system saw on any given day.

Fields logged per signal (from phase3_locked_architecture.json):
    ticker, date, feature_vector_11, regime_state, regime_multiplier,
    predicted_return_5d, atr_14, position_size_dollars, slippage_applied,
    fill_price, signal_timestamp, action
    + multi-horizon: predicted_1d/3d, signal, price_target_1d/3d/5d,
      confidence, news_count_1d, st_count_1d
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

LOG_FILE = 'logs/paper_trades.jsonl'
logger   = logging.getLogger(__name__)


def log_signal(
    ticker:                str,
    date:                  str,
    feature_vector:        dict,
    regime_state:          str,
    regime_multiplier:     float,
    predicted_return_5d:   float,
    atr_14:                float,
    position_size_dollars: float,
    slippage_applied:      float,
    fill_price:            float,
    signal_timestamp:      str,
    action:                str,
    raw_finbert_scores:    Optional[dict] = None,
    notes:                 str = '',
    hold_days:             int = 5,
    horizon:               str = '5D',
    predicted_1d:          float = 0.0,
    predicted_3d:          float = 0.0,
    signal:                str = 'NEUTRAL',
    price_target_1d:       float = 0.0,
    price_target_3d:       float = 0.0,
    price_target_5d:       float = 0.0,
    confidence:            float = 0.0,
    news_count_1d:         int = 0,
    st_count_1d:           int = 0,
    pcr:                   Optional[float] = None,
    pcr_confirmation:      str = 'UNKNOWN',
    pcr_size_mult:         float = 1.0,
    pcr_reason:            str = '',
) -> None:
    """Append one signal record to the execution log."""
    Path('logs').mkdir(exist_ok=True)

    record = {
        'ticker':                ticker,
        'date':                  date,
        'action':                action,
        'feature_vector_11':     feature_vector,
        'raw_finbert_scores':    raw_finbert_scores,
        'regime_state':          regime_state,
        'regime_multiplier':     regime_multiplier,
        'predicted_return_5d':   predicted_return_5d,
        'hold_days':             hold_days,
        'horizon':               horizon,
        'predicted_1d':          predicted_1d,
        'predicted_3d':          predicted_3d,
        'signal':                signal,
        'price_target_1d':       price_target_1d,
        'price_target_3d':       price_target_3d,
        'price_target_5d':       price_target_5d,
        'confidence':            confidence,
        'news_count_1d':         news_count_1d,
        'st_count_1d':           st_count_1d,
        'pcr':                   pcr,
        'pcr_confirmation':      pcr_confirmation,
        'pcr_size_mult':         pcr_size_mult,
        'pcr_reason':            pcr_reason,
        'atr_14':                atr_14,
        'position_size_dollars': position_size_dollars,
        'slippage_applied':      slippage_applied,
        'fill_price':            fill_price,
        'signal_timestamp':      signal_timestamp,
        'log_timestamp':         datetime.now(timezone.utc).isoformat(),
        'notes':                 notes,
    }

    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(record) + '\n')

    logger.info(f'signal_logged ticker={ticker} action={action} '
                f'predicted_return={predicted_return_5d:.4f}')
