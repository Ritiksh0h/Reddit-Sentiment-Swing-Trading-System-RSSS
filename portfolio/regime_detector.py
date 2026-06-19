"""
Rule-based regime detector.
Classifies current market regime using SPY indicators + rolling sentiment IC.

POSITIVE regime: full position size (100%)
NEUTRAL regime:  reduced position size (75%)
NEGATIVE regime: minimum position size (50%)

Production uses rule-based thresholds derived from known yearly regimes:
  2019 (IC=+0.086): SPY uptrend → POSITIVE
  2022 (IC=-0.083): SPY downtrend → NEGATIVE
  2023 (IC=-0.103): SPY uptrend but IC negative → use rolling_30d_IC
  → Use rolling_30d_IC as the primary live signal to disambiguate 2021/2023
"""
import logging
from dataclasses import dataclass
from typing import Literal, Optional
import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)
RegimeLabel = Literal['positive', 'neutral', 'negative']

POSITION_SIZING = {
    'positive': 1.00,
    'neutral':  0.75,
    'negative': 0.50,
}


@dataclass
class RegimeState:
    label:           RegimeLabel
    multiplier:      float
    spy_above_200ma: Optional[bool]
    spy_ret_60d:     Optional[float]
    rolling_30d_ic:  Optional[float]
    reason:          str


def classify_regime(
    rolling_30d_ic: Optional[float] = None,
    spy_ticker: str = 'SPY',
) -> RegimeState:
    """
    Classify current market regime using SPY price trend and rolling IC signal.

    Args:
        rolling_30d_ic: 30-day rolling Spearman IC from monitor_live_ic.py;
                        pass None during the first month before enough live trades exist
        spy_ticker:     ticker symbol for the market benchmark (override in tests)

    Returns:
        RegimeState with label ('positive'|'neutral'|'negative'), position-size
        multiplier (1.0|0.75|0.50), and diagnostic fields (spy_above_200ma,
        spy_ret_60d, rolling_30d_ic, reason)

    Decision logic:
        NEGATIVE if: SPY below 200MA OR spy_ret_60d < -0.10
        POSITIVE if: SPY above 200MA AND spy_ret_60d > 0
                     AND (rolling_30d_ic is None OR rolling_30d_ic > 0.03)
        NEUTRAL:     everything else (including SPY uptrend but live IC <= 0.03)
    """
    spy = pd.DataFrame()
    for period in ['1y', '6mo', '3mo']:
        try:
            spy = yf.download(spy_ticker, period=period,
                              auto_adjust=True, progress=False)
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)
            if not spy.empty and len(spy) >= 60:
                logger.debug(f'SPY data loaded period={period} rows={len(spy)}')
                break
        except Exception as e:
            logger.warning(f'SPY download failed period={period}: {e}')
        spy = pd.DataFrame()

    if spy.empty or len(spy) < 60:
        logger.warning('SPY data unavailable after all retries — defaulting to NEUTRAL')
        return RegimeState(
            label='neutral',
            multiplier=POSITION_SIZING['neutral'],
            spy_above_200ma=None,
            spy_ret_60d=None,
            rolling_30d_ic=rolling_30d_ic,
            reason='SPY data unavailable — default neutral',
        )

    close     = spy['Close']
    ma_200    = close.rolling(200).mean().iloc[-1]
    above_200 = bool(close.iloc[-1] > ma_200)
    ret_60d   = float(close.pct_change(60).iloc[-1])

    if not above_200 or ret_60d < -0.10:
        return RegimeState(
            label='negative',
            multiplier=POSITION_SIZING['negative'],
            spy_above_200ma=above_200,
            spy_ret_60d=round(ret_60d, 4),
            rolling_30d_ic=rolling_30d_ic,
            reason='SPY below 200MA or 60d return < -10%',
        )

    if above_200 and ret_60d > 0:
        if rolling_30d_ic is None or rolling_30d_ic > 0.03:
            return RegimeState(
                label='positive',
                multiplier=POSITION_SIZING['positive'],
                spy_above_200ma=above_200,
                spy_ret_60d=round(ret_60d, 4),
                rolling_30d_ic=rolling_30d_ic,
                reason='SPY above 200MA, 60d positive, IC positive or unknown',
            )
        else:
            return RegimeState(
                label='neutral',
                multiplier=POSITION_SIZING['neutral'],
                spy_above_200ma=above_200,
                spy_ret_60d=round(ret_60d, 4),
                rolling_30d_ic=rolling_30d_ic,
                reason='SPY uptrend but rolling_30d_IC <= 0.03 — reduce sizing',
            )

    return RegimeState(
        label='neutral',
        multiplier=POSITION_SIZING['neutral'],
        spy_above_200ma=above_200,
        spy_ret_60d=round(ret_60d, 4),
        rolling_30d_ic=rolling_30d_ic,
        reason='mixed signals — default neutral',
    )
