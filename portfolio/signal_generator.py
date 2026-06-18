"""
Daily signal generator.
Loads trained XGBoost models, computes features for qualifying tickers,
returns ranked multi-horizon predictions.

Run time: before market open (08:00-09:00 ET)
Input:    live market data + Reddit/news/StockTwits data from prior 24h
Output:   list of SignalRecord objects, sorted bullish-first then bearish
"""
import os
# Must be set before xgboost import — prevents OpenMP/OMP thread conflicts
# between XGBoost workers and uvicorn on Python 3.13 + macOS (exit code 139).
os.environ['OMP_NUM_THREADS']     = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS']     = '1'

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

FEATURES     = ARCH['features']          # 14 features
DROP_TICKERS = set(ARCH['drop_tickers'])
DENSITY_GATE = 10
# Signal thresholds — calibrated to model prediction distribution
# Model mean ≈ 0.45%, median ≈ 0.65%, std ≈ 3%; 27% rows > 1.5%, 12% < -1.5%
MIN_PRED_RET      = 0.005   # minimum |pred| to consider (noise filter)
BULLISH_THRESHOLD = 0.015   # pred >= 1.5% → BULLISH
BEARISH_THRESHOLD = -0.015  # pred <= -1.5% → BEARISH


@dataclass
class SignalRecord:
    ticker:            str
    date:              str
    predicted_return:  float       # 5D prediction (primary, backward compat)
    predicted_1d:      float       # 1D prediction
    predicted_3d:      float       # 3D prediction
    predicted_5d:      float       # 5D prediction (same as predicted_return)
    confidence:        float       # 0.0-1.0 confidence score
    signal:            str         # 'BULLISH' | 'BEARISH' | 'NEUTRAL'
    price_target_1d:   float       # current_price * (1 + predicted_1d)
    price_target_3d:   float       # current_price * (1 + predicted_3d)
    price_target_5d:   float       # current_price * (1 + predicted_5d)
    feature_vector:    dict
    post_count_1d:     int
    news_count_1d:     int         # from news_fetcher
    st_count_1d:       int         # from stocktwits_fetcher
    atr_14:            float
    price:             float
    signal_timestamp:  str


def load_models(model_dir: str = 'models/registry') -> dict:
    """
    Load all three horizon models.
    Returns dict: {'1d': model, '3d': model, '5d': model}
    Falls back to phase3_model.pkl for 5d if individual models missing.
    """
    import pickle
    models   = {}
    dir_path = Path(model_dir)

    for horizon in ['1d', '3d', '5d']:
        model_path = dir_path / f'model_{horizon}.pkl'
        if model_path.exists():
            with open(model_path, 'rb') as f:
                models[horizon] = pickle.load(f)
            models[horizon].set_params(n_jobs=1)
            logger.info(f'Loaded model_{horizon}')
        else:
            if horizon == '5d':
                fallback = dir_path / 'phase3_model.pkl'
                if fallback.exists():
                    with open(fallback, 'rb') as f:
                        models['5d'] = pickle.load(f)
                    models['5d'].set_params(n_jobs=1)
                    logger.warning('Using legacy phase3_model.pkl for 5d')
                else:
                    raise FileNotFoundError(
                        f'No model found for horizon {horizon}. '
                        'Run scripts/train_phase3_model.py first.'
                    )
            else:
                logger.warning(f'model_{horizon}.pkl not found — '
                               f'will use 5d model as proxy')
                models[horizon] = models.get('5d')

    return models


def load_model(model_path: str = 'models/registry/phase3_model.pkl'):
    """Backward-compatible single model loader for daily_run.py."""
    import pickle
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f'Model not found at {model_path}. '
            'Run scripts/train_phase3_model.py first.'
        )
    with open(model_path, 'rb') as f:
        m = pickle.load(f)
    m.set_params(n_jobs=1)
    return m


def compute_features_live(
    ticker: str,
    market_data: pd.DataFrame,
    post_count_1d: int,
    mention_growth_1d: float,
    mention_growth_7d: float,
    news_sentiment_1d: float = 0.0,
    st_sentiment_1d:   float = 0.0,
    st_bull_pct:       float = 0.5,
) -> Optional[dict]:
    """
    Compute the 14-feature vector for a ticker using live data.
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
        'news_sentiment_1d': float(news_sentiment_1d),
        'st_sentiment_1d':   float(st_sentiment_1d),
        'st_bull_pct':       float(st_bull_pct),
    }


def generate_signals(
    reddit_counts: dict,
    model=None,           # backward compat — ignored if models provided
    today: str = None,
    models: dict = None,  # {'1d': model, '3d': model, '5d': model}
) -> list:
    """
    Generate multi-horizon ranked signals for all qualifying tickers.

    Args:
        reddit_counts: {ticker: {post_count_1d, mention_growth_1d, ...}}
        model:         single model (backward compat, used as 5d if models=None)
        today:         date string YYYY-MM-DD
        models:        dict of horizon → model (preferred)

    Returns:
        list of SignalRecord sorted bullish-first by pred_5d, then bearish
    """
    if today is None:
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    if models is None:
        try:
            models = load_models()
        except Exception:
            if model is not None:
                models = {'5d': model, '3d': model, '1d': model}
            else:
                raise

    ts      = datetime.now(timezone.utc).isoformat()
    signals = []

    for ticker, reddit_data in reddit_counts.items():
        if ticker in DROP_TICKERS:
            continue

        post_count = reddit_data.get('post_count_1d', 0)
        if post_count < DENSITY_GATE:
            logger.debug(f'density_gate_fail ticker={ticker} posts={post_count}')
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
            logger.warning(f'market_data_fail ticker={ticker}: {e}')
            continue

        features = compute_features_live(
            ticker=ticker,
            market_data=mkt,
            post_count_1d=post_count,
            mention_growth_1d=reddit_data.get('mention_growth_1d', 0.0),
            mention_growth_7d=reddit_data.get('mention_growth_7d', 0.0),
            news_sentiment_1d=reddit_data.get('news_sentiment_1d', 0.0),
            st_sentiment_1d=reddit_data.get('st_sentiment_1d', 0.0),
            st_bull_pct=reddit_data.get('st_bull_pct', 0.5),
        )
        if features is None:
            continue

        price = float(mkt['Close'].iloc[-1])
        atr   = features['atr_14']
        avail = [f for f in FEATURES if f in features]
        X     = pd.DataFrame([features])[avail].fillna(0)

        preds = {}
        for horizon, m in models.items():
            if m is None:
                preds[horizon] = 0.0
                continue
            try:
                preds[horizon] = float(m.predict(X)[0])
            except Exception as e:
                logger.error(f'predict_fail ticker={ticker} horizon={horizon}: {e}')
                preds[horizon] = 0.0

        pred_1d = preds.get('1d', 0.0)
        pred_3d = preds.get('3d', 0.0)
        pred_5d = preds.get('5d', 0.0)

        if abs(pred_5d) < MIN_PRED_RET:
            continue

        # Confidence: 0.0 at BULLISH_THRESHOLD, 1.0 at 2× threshold
        confidence = min(abs(pred_5d) / (BULLISH_THRESHOLD * 2), 1.0)

        if pred_5d >= BULLISH_THRESHOLD:
            signal = 'BULLISH'
        elif pred_5d <= BEARISH_THRESHOLD:
            signal = 'BEARISH'
        else:
            signal = 'NEUTRAL'

        signals.append(SignalRecord(
            ticker=ticker,
            date=today,
            predicted_return=round(pred_5d, 6),   # backward compat
            predicted_1d=round(pred_1d, 6),
            predicted_3d=round(pred_3d, 6),
            predicted_5d=round(pred_5d, 6),
            confidence=round(confidence, 4),
            signal=signal,
            price_target_1d=round(price * (1 + pred_1d), 2),
            price_target_3d=round(price * (1 + pred_3d), 2),
            price_target_5d=round(price * (1 + pred_5d), 2),
            feature_vector=features,
            post_count_1d=post_count,
            news_count_1d=int(reddit_data.get('news_count_1d', 0)),
            st_count_1d=int(reddit_data.get('st_count_1d', 0)),
            atr_14=round(atr, 6),
            price=round(price, 4),
            signal_timestamp=ts,
        ))

    bullish = sorted([s for s in signals if s.signal == 'BULLISH'],
                     key=lambda x: x.predicted_5d, reverse=True)
    neutral = sorted([s for s in signals if s.signal == 'NEUTRAL'],
                     key=lambda x: abs(x.predicted_5d), reverse=True)
    bearish = sorted([s for s in signals if s.signal == 'BEARISH'],
                     key=lambda x: x.predicted_5d)

    result = bullish + neutral + bearish
    logger.info(f'signals_generated count={len(result)} '
                f'bullish={len(bullish)} neutral={len(neutral)} '
                f'bearish={len(bearish)} date={today}')
    return result
