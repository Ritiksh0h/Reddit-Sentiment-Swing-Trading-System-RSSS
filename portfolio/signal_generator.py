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
DENSITY_GATE = 10

from config.settings import load_tickers, TICKERS_TRADE_PATH, TICKERS_DROP_PATH
from data.options_fetcher import fetch_pcr, interpret_pcr

TRADE_UNIVERSE = set(load_tickers(TICKERS_TRADE_PATH)) or None  # None = unrestricted fallback
DROP_TICKERS   = set(load_tickers(TICKERS_DROP_PATH)) or set(ARCH['drop_tickers'])
# Signal thresholds — calibrated to model prediction distribution
# Model mean ≈ 0.45%, median ≈ 0.65%, std ≈ 3%; 27% rows > 1.5%, 12% < -1.5%
MIN_PRED_RET      = 0.005   # minimum |pred| to consider (noise filter)
BULLISH_THRESHOLD = 0.015   # pred >= 1.5% → BULLISH
BEARISH_THRESHOLD = -0.015  # pred <= -1.5% → BEARISH


@dataclass
class SignalRecord:
    ticker:            str
    date:              str
    predicted_return:  float       # best horizon pred (backward compat)
    predicted_1d:      float
    predicted_3d:      float
    predicted_5d:      float
    confidence:        float       # 0.0-1.0, based on best horizon pred
    signal:            str         # 'BULLISH' | 'BEARISH' | 'NEUTRAL'
    hold_days:         int         # 1, 3, or 5 — from winning horizon
    horizon:           str         # '1D', '3D', or '5D'
    price_target_1d:   float
    price_target_3d:   float
    price_target_5d:   float
    feature_vector:    dict
    post_count_1d:     int
    news_count_1d:     int
    st_count_1d:       int
    atr_14:            float
    price:             float
    signal_timestamp:  str
    pcr:               Optional[float] = None
    pcr_confirmation:  str = 'UNKNOWN'
    pcr_size_multiplier: float = 1.0
    pcr_reason:        str = ''


def _load_booster_from_pkl(path: Path) -> xgb.Booster:
    """Load an XGBRegressor pickle and return its underlying native Booster.

    Avoids the sklearn wrapper entirely — `set_params` / `get_params` are not
    called, so this survives XGBoost major-version mismatches between the
    training environment (local 3.x) and CI (2.x).
    """
    import pickle
    with open(path, 'rb') as f:
        sklearn_model = pickle.load(f)
    return sklearn_model.get_booster()


def load_models(model_dir: str = 'models/registry') -> dict:
    """
    Load all three horizon models.
    Returns dict: {'1d': booster, '3d': booster, '5d': booster}
    Falls back to phase3_model.pkl for 5d if individual models missing.
    """
    models   = {}
    dir_path = Path(model_dir)

    for horizon in ['1d', '3d', '5d']:
        model_path = dir_path / f'model_{horizon}.pkl'
        if model_path.exists():
            models[horizon] = _load_booster_from_pkl(model_path)
            logger.info(f'Loaded model_{horizon}')
        else:
            if horizon == '5d':
                fallback = dir_path / 'phase3_model.pkl'
                if fallback.exists():
                    models['5d'] = _load_booster_from_pkl(fallback)
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
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f'Model not found at {model_path}. '
            'Run scripts/train_phase3_model.py first.'
        )
    return _load_booster_from_pkl(Path(model_path))


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
    Compute the 14-feature vector for a single ticker using live OHLCV + sentiment data.

    Args:
        ticker:            ticker symbol (used for log messages only)
        market_data:       90-day OHLCV DataFrame from yfinance (needs >= 55 rows)
        post_count_1d:     Reddit post count in last 24h (attention gate feature)
        mention_growth_1d: today's count / yesterday's count ratio
        mention_growth_7d: today's count / 7-day average ratio
        news_sentiment_1d: FinBERT news sentiment [-1, +1]; 0.0 if unavailable
        st_sentiment_1d:   StockTwits sentiment [-1, +1]; 0.0 if unavailable
        st_bull_pct:       fraction of StockTwits messages tagged bullish; 0.5 if unavailable

    Returns:
        dict of 14 feature values matching ARCH['features'], or None if < 55 rows of market data
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
    reddit_counts:   dict,
    model=None,           # backward compat — ignored if models provided
    today:           str  = None,
    models:          dict = None,  # {'1d': model, '3d': model, '5d': model}
    news_data:       dict = None,  # {ticker: {news_sentiment_1d}}
    stocktwits_data: dict = None,  # {ticker: {st_sentiment_1d, st_bull_pct}}
) -> list:
    """
    Generate multi-horizon ranked signals for all tickers that pass the density gate.

    Density gate: post_count_1d >= 10 — tickers below are skipped silently.
    Tickers in ARCH['drop_tickers'] are always excluded regardless of post count.

    Args:
        reddit_counts:   {ticker: {post_count_1d, mention_growth_1d, mention_growth_7d, ...}}
        model:           single model (backward compat, ignored if models is provided)
        today:           date string YYYY-MM-DD; defaults to UTC today
        models:          preferred — {'1d': model, '3d': model, '5d': model}
        news_data:       {ticker: {news_sentiment_1d, news_count_1d}} from news_fetcher
        stocktwits_data: {ticker: {st_sentiment_1d, st_bull_pct, st_count_1d}} from stocktwits_fetcher

    Returns:
        list of SignalRecord ordered: BULLISH (desc pred_5d), NEUTRAL, BEARISH (asc pred_5d)
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

    logger.info(
        f'Trade universe: {len(TRADE_UNIVERSE) if TRADE_UNIVERSE else "all"} tickers  '
        f'drop_list: {len(DROP_TICKERS)}'
    )

    ts      = datetime.now(timezone.utc).isoformat()
    signals = []

    for ticker, reddit_data in reddit_counts.items():
        if TRADE_UNIVERSE is not None and ticker not in TRADE_UNIVERSE:
            logger.debug(f'trade_universe_skip ticker={ticker}')
            continue
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

        if news_data and ticker in news_data:
            news_sent = float(news_data[ticker].get('news_sentiment_1d', 0.0))
        else:
            news_sent = float(reddit_data.get('news_sentiment_1d', 0.0))

        if stocktwits_data and ticker in stocktwits_data:
            st_sent = float(stocktwits_data[ticker].get('st_sentiment_1d', 0.0))
            st_bull = float(stocktwits_data[ticker].get('st_bull_pct', 0.5))
        else:
            st_sent = float(reddit_data.get('st_sentiment_1d', 0.0))
            st_bull = float(reddit_data.get('st_bull_pct', 0.5))

        logger.debug(
            f'features ticker={ticker} '
            f'news={news_sent:.3f} st={st_sent:.3f} '
            f'st_bull={st_bull:.3f}'
        )

        features = compute_features_live(
            ticker=ticker,
            market_data=mkt,
            post_count_1d=post_count,
            mention_growth_1d=reddit_data.get('mention_growth_1d', 0.0),
            mention_growth_7d=reddit_data.get('mention_growth_7d', 0.0),
            news_sentiment_1d=news_sent,
            st_sentiment_1d=st_sent,
            st_bull_pct=st_bull,
        )
        if features is None:
            continue

        price = float(mkt['Close'].iloc[-1])
        atr   = features['atr_14']
        avail = [f for f in FEATURES if f in features]
        X     = pd.DataFrame([features])[avail].fillna(0)

        preds = {}
        dmatrix = xgb.DMatrix(X)
        for horizon, m in models.items():
            if m is None:
                preds[horizon] = 0.0
                continue
            try:
                preds[horizon] = float(m.predict(dmatrix)[0])
            except Exception as e:
                logger.error(f'predict_fail ticker={ticker} horizon={horizon}: {e}')
                preds[horizon] = 0.0

        pred_1d = preds.get('1d', 0.0)
        pred_3d = preds.get('3d', 0.0)
        pred_5d = preds.get('5d', 0.0)

        # Noise filter: skip if no horizon has a meaningful prediction
        if max(abs(pred_1d), abs(pred_3d), abs(pred_5d)) < MIN_PRED_RET:
            continue

        # Dynamic hold: 5D > 3D > 1D priority for both directions
        if pred_5d >= BULLISH_THRESHOLD:
            signal    = 'BULLISH'
            hold_days = 5
            horizon   = '5D'
            best_pred = pred_5d
        elif pred_3d >= BULLISH_THRESHOLD:
            signal    = 'BULLISH'
            hold_days = 3
            horizon   = '3D'
            best_pred = pred_3d
        elif pred_1d >= BULLISH_THRESHOLD:
            signal    = 'BULLISH'
            hold_days = 1
            horizon   = '1D'
            best_pred = pred_1d
        elif pred_5d <= BEARISH_THRESHOLD:
            signal    = 'BEARISH'
            hold_days = 5
            horizon   = '5D'
            best_pred = pred_5d
        elif pred_3d <= BEARISH_THRESHOLD:
            signal    = 'BEARISH'
            hold_days = 3
            horizon   = '3D'
            best_pred = pred_3d
        elif pred_1d <= BEARISH_THRESHOLD:
            signal    = 'BEARISH'
            hold_days = 1
            horizon   = '1D'
            best_pred = pred_1d
        else:
            signal    = 'NEUTRAL'
            hold_days = 5
            horizon   = '5D'
            best_pred = pred_5d

        # Confidence: based on winning horizon pred
        confidence = min(abs(best_pred) / (BULLISH_THRESHOLD * 2), 1.0)

        # PCR confirmation for BULLISH signals only — never blocks, only modulates size
        pcr_val  = None
        pcr_info = {'confirmation': 'UNKNOWN', 'size_multiplier': 1.0, 'reason': 'not_bullish'}
        if signal == 'BULLISH':
            pcr_val  = fetch_pcr(ticker)
            pcr_info = interpret_pcr(pcr_val)

        logger.info(
            f'  {signal:<8} {ticker:<6} '
            f'1D={pred_1d*100:+.2f}% '
            f'3D={pred_3d*100:+.2f}% '
            f'5D={pred_5d*100:+.2f}% '
            f'hold={hold_days}d horizon={horizon} '
            f'conf={confidence*100:.0f}% '
            f'posts={post_count} '
            f'news={int(news_sent*100)} '
            f'st={int(st_bull*100)} '
            f'pcr={pcr_val} pcr_conf={pcr_info["confirmation"]}'
        )

        signals.append(SignalRecord(
            ticker=ticker,
            date=today,
            predicted_return=round(best_pred, 6),
            predicted_1d=round(pred_1d, 6),
            predicted_3d=round(pred_3d, 6),
            predicted_5d=round(pred_5d, 6),
            confidence=round(confidence, 4),
            signal=signal,
            hold_days=hold_days,
            horizon=horizon,
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
            pcr=pcr_val,
            pcr_confirmation=pcr_info['confirmation'],
            pcr_size_multiplier=pcr_info['size_multiplier'],
            pcr_reason=pcr_info['reason'],
        ))

    bullish = sorted([s for s in signals if s.signal == 'BULLISH'],
                     key=lambda x: x.predicted_5d, reverse=True)
    neutral = sorted([s for s in signals if s.signal == 'NEUTRAL'],
                     key=lambda x: abs(x.predicted_5d), reverse=True)
    bearish = sorted([s for s in signals if s.signal == 'BEARISH'],
                     key=lambda x: x.predicted_5d)

    result = bullish + neutral + bearish
    logger.info(f'signals_generated count={len(result)} '
                f'bullish={len(bullish)} '
                f'bearish_logged={len(bearish)} '
                f'neutral={len(neutral)} '
                f'date={today}')
    return result
