"""
PCR (Put/Call Ratio) fetcher using yfinance options chain.
Uses the nearest expiration date to get a fresh, liquid ratio.

Confirmation thresholds (NEVER blocks a signal — only modulates size):
  < 0.7  → CONFIRM   (size mult = 1.0)
  0.7-1.0 → NEUTRAL   (size mult = 1.0)
  > 1.0  → CAUTION   (size mult = 0.5)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

PCR_CONFIRM_THRESHOLD = 0.7
PCR_CAUTION_THRESHOLD = 1.0
PCR_CAUTION_SIZE_MULT = 0.5


def fetch_pcr(ticker: str) -> Optional[float]:
    """
    Fetch Put/Call Ratio for ticker from the nearest options expiration.
    Returns None on any error (caller must handle gracefully).
    """
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            logger.warning('pcr_fetch ticker=%s no_expirations', ticker)
            return None

        # Use nearest expiration for freshest liquidity signal
        nearest_exp = expirations[0]
        chain = tk.option_chain(nearest_exp)
        puts  = chain.puts
        calls = chain.calls

        put_volume  = float(puts['volume'].fillna(0).sum())
        call_volume = float(calls['volume'].fillna(0).sum())

        if call_volume == 0:
            logger.warning('pcr_fetch ticker=%s call_volume=0', ticker)
            return None

        pcr = round(put_volume / call_volume, 3)
        logger.info('pcr_fetch ticker=%s pcr=%.3f exp=%s', ticker, pcr, nearest_exp)
        return pcr

    except Exception as exc:
        logger.warning('pcr_fetch ticker=%s error=%s', ticker, exc)
        return None


def interpret_pcr(pcr: Optional[float]) -> dict:
    """
    Map a raw PCR to confirmation label and size multiplier.
    Returns a dict with confirmation, size_multiplier, and reason.
    Never raises — falls back to UNKNOWN on None input.
    """
    if pcr is None:
        return {
            'confirmation':    'UNKNOWN',
            'size_multiplier': 1.0,
            'reason':          'pcr_fetch_failed',
        }

    if pcr < PCR_CONFIRM_THRESHOLD:
        return {
            'confirmation':    'CONFIRM',
            'size_multiplier': 1.0,
            'reason':          f'pcr={pcr:.3f} < {PCR_CONFIRM_THRESHOLD} (calls dominate)',
        }
    elif pcr <= PCR_CAUTION_THRESHOLD:
        return {
            'confirmation':    'NEUTRAL',
            'size_multiplier': 1.0,
            'reason':          f'pcr={pcr:.3f} in neutral band',
        }
    else:
        return {
            'confirmation':    'CAUTION',
            'size_multiplier': PCR_CAUTION_SIZE_MULT,
            'reason':          f'pcr={pcr:.3f} > {PCR_CAUTION_THRESHOLD} (puts dominate, size halved)',
        }


def batch_fetch_pcr(tickers: list[str]) -> dict[str, dict]:
    """
    Fetch PCR for a list of tickers. Returns {ticker: interpret_pcr result}.
    Failures for individual tickers don't stop the batch.
    """
    results = {}
    for ticker in tickers:
        pcr = fetch_pcr(ticker)
        results[ticker] = interpret_pcr(pcr)
    return results
