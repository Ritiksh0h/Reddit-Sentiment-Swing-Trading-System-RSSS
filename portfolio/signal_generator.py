"""
Daily signal generator.
Loads trained XGBoost model, computes features for qualifying tickers,
returns ranked predictions.

Run time: before market open (08:00-09:00 ET)
Input:    live market data + Reddit post counts from prior 24h
Output:   list of SignalRecord objects, ranked by predicted return
"""
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf

logger = logging.getLogger(__name__)

with open('experiments/phase3_locked_architecture.json') as f:
    ARCH = json.load(f)

FEATURES     = ARCH['features']          # 11 features
DROP_TICKERS = set(ARCH['drop_tickers'])
DENSITY_GATE = 10
MIN_PRED_RET = 0.01


@dataclass
class SignalRecord:
    ticker:            str
    date:              str
    predicted_return:  float
    feature_vector:    dict
    post_count_1d:     int
    atr_14:            float
    price:             float
    signal_timestamp:  str


def load_model(model_path: str = 'models/registry/phase3_model.pkl') -> xgb.XGBRegressor:
    import pickle
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f'Phase 3 model not found at {model_path}. '
            'Run scripts/train_phase3_model.py first.'
        )
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def compute_features_live(
    ticker: str,
    market_data: pd.DataFrame,
    post_count_1d: int,
    mention_growth_1d: float,
    mention_growth_7d: float,
) -> Optional[dict]:
    """
    Compute the 11-feature vector for a ticker using live data.
    Returns None if insufficient market data.
    """
    if len(market_data) < 55:
        logger.warning(f'insufficient_market_data ticker={ticker} n_rows={len(market_data)}')
        return None

    close = market_data['Close']
    high  = market_data['High']
    low   = market_data['Low']
    vol   = market_data['Volume']

    returns_1d  = float(close.pct_change(1).iloc[-1])
    returns_5d  = float(close.pct_change(5).iloc[-1])
    returns_20d = float(close.pct_change(20).iloc[-1])

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = float((100 - 100 / (1 + rs)).iloc[-1])

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_14 = float(tr.rolling(14).mean().iloc[-1])

    avg_vol_20d  = float(vol.rolling(20).mean().iloc[-1])
    relative_vol = float(vol.iloc[-1] / avg_vol_20d) if avg_vol_20d > 0 else 1.0

    ma_20 = float(close.rolling(20).mean().iloc[-1])
    ma_50 = float(close.rolling(50).mean().iloc[-1])
    price = float(close.iloc[-1])
    dist_from_20ma = (price - ma_20) / ma_20 if ma_20 > 0 else 0.0
    dist_from_50ma = (price - ma_50) / ma_50 if ma_50 > 0 else 0.0

    return {
        'returns_1d':        returns_1d,
        'returns_5d':        returns_5d,
        'returns_20d':       returns_20d,
        'rsi_14':            rsi,
        'atr_14':            atr_14,
        'relative_volume':   relative_vol,
        'dist_from_20ma':    dist_from_20ma,
        'dist_from_50ma':    dist_from_50ma,
        'post_count_1d':     float(post_count_1d),
        'mention_growth_1d': float(mention_growth_1d),
        'mention_growth_7d': float(mention_growth_7d),
    }


def generate_signals(
    reddit_counts: dict,
    model: xgb.XGBRegressor,
    today: str = None,
) -> list:
    """
    Generate ranked signals for all qualifying tickers.

    Args:
        reddit_counts: {ticker: {post_count_1d, mention_growth_1d, mention_growth_7d}}
        model:         trained XGBoost regressor
        today:         date string YYYY-MM-DD (defaults to today)

    Returns:
        list of SignalRecord, sorted by predicted_return descending
    """
    if today is None:
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    ts = datetime.now(timezone.utc).isoformat()
    signals = []

    for ticker, reddit_data in reddit_counts.items():
        if ticker in DROP_TICKERS:
            continue

        post_count = reddit_data.get('post_count_1d', 0)
        if post_count < DENSITY_GATE:
            logger.debug(f'density_gate_fail ticker={ticker} post_count={post_count}')
            continue

        try:
            mkt = yf.download(ticker, period='90d',
                              auto_adjust=True, progress=False)
            if isinstance(mkt.columns, pd.MultiIndex):
                mkt.columns = mkt.columns.get_level_values(0)
            if len(mkt) == 0:
                logger.warning(f'empty_market_data ticker={ticker}')
                continue
        except Exception as e:
            logger.warning(f'market_data_fail ticker={ticker} error={e}')
            continue

        features = compute_features_live(
            ticker=ticker,
            market_data=mkt,
            post_count_1d=post_count,
            mention_growth_1d=reddit_data.get('mention_growth_1d', 0.0),
            mention_growth_7d=reddit_data.get('mention_growth_7d', 0.0),
        )
        if features is None:
            continue

        X = pd.DataFrame([features])[FEATURES].fillna(0)
        try:
            pred = float(model.predict(X)[0])
        except Exception as e:
            logger.error(f'model_predict_fail ticker={ticker} error={e}')
            continue

        if pred < MIN_PRED_RET:
            continue

        price = float(mkt['Close'].iloc[-1])
        atr   = features['atr_14']

        signals.append(SignalRecord(
            ticker=ticker,
            date=today,
            predicted_return=round(pred, 6),
            feature_vector=features,
            post_count_1d=post_count,
            atr_14=round(atr, 6),
            price=round(price, 4),
            signal_timestamp=ts,
        ))

    signals.sort(key=lambda s: s.predicted_return, reverse=True)
    logger.info(f'signals_generated count={len(signals)} date={today}')
    return signals
