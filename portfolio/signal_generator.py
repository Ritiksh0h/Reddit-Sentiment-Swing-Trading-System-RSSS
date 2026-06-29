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

# Grinold vol-adjustment constants (Improvement 3)
_TARGET_VOL = 0.02
_VOL_FLOOR  = 0.005
_VOL_CAP    = 0.08

with open('experiments/phase3_locked_architecture.json') as f:
    ARCH = json.load(f)

# V2 feature set — loaded from training_metadata_v2.json (tracked).
# Falls back to phase3 ARCH features only if file is absent.
try:
    with open('models/training_metadata_v2.json') as _f:
        FEATURES = json.load(_f).get('feature_cols', ARCH['features'])
except Exception:
    FEATURES = ARCH['features']

DENSITY_GATE = 5

from config.settings import load_tickers, TICKERS_TRADE_PATH, TICKERS_DROP_PATH
from data.options_fetcher import fetch_pcr, interpret_pcr
from data.earnings_fetcher import is_safe_to_trade


def _load_sector_map() -> dict:
    """Load ticker → sector from ticker_registry.json."""
    try:
        with open('config/ticker_registry.json') as f:
            reg = json.load(f)
        return {t: v.get('sector', 'Unknown')
                for t, v in reg.get('tickers', {}).items()}
    except Exception:
        return {}

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


def load_models(model_dir: str = 'models') -> dict:
    """
    Load multi-horizon XGBoost models.

    Priority: v2 JSON models in models/ → phase3 PKL fallback in models/registry/.
    Returns dict: {'1d': booster, '3d': booster, '5d': booster}
    """
    models   = {}
    dir_path = Path(model_dir)

    for horizon in ['1d', '3d', '5d']:
        v2_path = dir_path / f'model_{horizon}_v2.json'
        if v2_path.exists():
            booster = xgb.Booster()
            booster.load_model(str(v2_path))
            models[horizon] = booster
            logger.info(f'Loaded v2 model_{horizon}_v2.json')
        else:
            # Fallback to phase3 PKL — feature set mismatch risk, log warning
            pkl_path = Path('models/registry') / f'model_{horizon}.pkl'
            if pkl_path.exists():
                models[horizon] = _load_booster_from_pkl(pkl_path)
                logger.warning(
                    f'v2 model_{horizon}_v2.json not found — '
                    f'using phase3 fallback (feature mismatch risk)'
                )
            elif horizon == '5d':
                fallback = Path('models/registry/phase3_model.pkl')
                if fallback.exists():
                    models['5d'] = _load_booster_from_pkl(fallback)
                    logger.warning('Using legacy phase3_model.pkl for 5d')
                else:
                    raise FileNotFoundError(
                        f'No model found for horizon {horizon}. '
                        'Run scripts/train_models_v2.py first.'
                    )
            else:
                logger.warning(f'model_{horizon} not found — '
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
    news_sentiment_1d: float = 0.0,
    total_comments_1d: int = 0,
    vader_sentiment_1d: float = 0.0,
    mention_growth_7d: float = 1.0,
    vix_percentile: float = 0.5,
    spy_above_200ma: float = 1.0,
    mention_history: dict = None,
) -> Optional[dict]:
    """
    Compute 18-feature v2 vector for a single ticker from live OHLCV + sentiment data.

    Args:
        ticker:             ticker symbol (used for log messages only)
        market_data:        90-day OHLCV DataFrame from yfinance (needs >= 55 rows)
        post_count_1d:      Reddit post count in last 24h
        news_sentiment_1d:  FinBERT news sentiment [-1, +1]; 0.0 if unavailable
        total_comments_1d:  sum of num_comments on Reddit posts; 0 if unavailable
        vader_sentiment_1d: VADER compound score on post titles [-1, +1]; 0.0 if unavailable
        mention_growth_7d:  today / 7d avg ratio; extra field for dynamic slippage (not a model feature)
        vix_percentile:     fraction of trailing 252d VIX below current VIX; pre-fetched once per run
        spy_above_200ma:    1.0 if SPY close > SPY 200-day MA, else 0.0; pre-fetched once per run
        mention_history:    {ticker: {date_str: count}} from data/mention_history.json

    Returns:
        dict of 18 v2 feature values + atr_14 and mention_growth_7d as extra fields
        (extras are used for position sizing / slippage, not for model inference).
        Returns None if < 55 rows of market data.
    """
    if len(market_data) < 55:
        logger.warning(f'insufficient_market_data ticker={ticker} n_rows={len(market_data)}')
        return None

    close = market_data['Close']
    high  = market_data['High']
    low   = market_data['Low']
    vol   = market_data['Volume']

    returns_1d  = float(close.pct_change(1).iloc[-1])
    returns_20d = float(close.pct_change(20).iloc[-1])

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = float((100 - 100 / (1 + rs)).iloc[-1])

    # atr_14: position sizing only (NOT a v2 model feature)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_14 = float(tr.rolling(14).mean().iloc[-1])

    volume_today = float(vol.iloc[-1])
    avg_vol_20d  = float(vol.rolling(20).mean().iloc[-1])
    relative_vol = float(vol.iloc[-1] / avg_vol_20d) if avg_vol_20d > 0 else 1.0

    # abnormal_attention_1d: post_count / (20d rolling avg + 1), clipped at 10
    if mention_history is not None:
        ticker_hist  = mention_history.get(ticker, {})
        sorted_dates = sorted(ticker_hist.keys())[-20:]
        hist_counts  = [ticker_hist[d] for d in sorted_dates]
        rolling_avg  = float(sum(hist_counts) / len(hist_counts)) if hist_counts else 1.0
    else:
        rolling_avg = 1.0
    abnormal_attention = min(float(post_count_1d) / (rolling_avg + 1.0), 10.0)

    sentiment_extremity = abs(float(vader_sentiment_1d))
    # sentiment_accel = vader_1d - vader_3d; no 3d history in live feed — fallback 0.0
    sentiment_accel = 0.0

    regime_score = 0.6 * float(spy_above_200ma) + 0.4 * (1.0 - float(vix_percentile))
    vix_x_volume = float(vix_percentile) * relative_vol

    # dist_from_20ma_pct: (price - 20d MA) / 20d MA — positive = above MA (uptrend)
    ma_20 = close.rolling(20, min_periods=10).mean()
    dist_from_20ma_pct = float(
        ((close - ma_20) / ma_20.replace(0, np.nan)).iloc[-1]
    ) if not ma_20.isna().iloc[-1] else 0.0

    # pead_proxy: decaying signal from earnings-like jumps (|ret| > 5%) in past 20 days
    ret_series = close.pct_change()
    pead_val = 0.0
    for lookback in range(1, 21):
        idx = -1 - lookback
        if abs(len(close)) <= abs(idx):
            break
        jump = float(ret_series.iloc[idx])
        if abs(jump) > 0.05:
            decay = 1.0 - lookback / 20.0
            pead_val += jump * decay
    pead_val = max(-0.1, min(0.1, pead_val))

    return {
        # ── 18 v2 model features (must match training_metadata_v2.json feature_cols) ──
        'post_count_1d':         float(post_count_1d),
        'abnormal_attention_1d': round(abnormal_attention, 4),
        'total_comments_1d':     float(total_comments_1d),
        'vader_sentiment_1d':    round(float(vader_sentiment_1d), 4),
        'sentiment_extremity':   round(sentiment_extremity, 4),
        'sentiment_accel':       round(sentiment_accel, 4),
        'volume':                volume_today,
        'relative_volume':       round(relative_vol, 4),
        'returns_1d':            round(returns_1d, 6),
        'returns_20d':           round(returns_20d, 6),
        'rsi_14':                round(rsi, 4),
        'news_sentiment_1d':     float(news_sentiment_1d),
        'vix_percentile':        round(float(vix_percentile), 4),
        'vix_x_volume':          round(vix_x_volume, 4),
        'spy_above_200ma':       float(spy_above_200ma),
        'regime_score':          round(regime_score, 4),
        'dist_from_20ma_pct':    round(dist_from_20ma_pct, 6),
        'pead_proxy':            round(pead_val, 6),
        # ── Extra fields: used for position sizing / slippage (not model features) ──
        'atr_14':                round(atr_14, 6),
        'mention_growth_7d':     float(mention_growth_7d),
    }


def generate_signals(
    reddit_counts:   dict,
    model=None,           # backward compat — ignored if models provided
    today:           str  = None,
    models:          dict = None,  # {'1d': model, '3d': model, '5d': model}
    news_data:       dict = None,  # {ticker: {news_sentiment_1d}}
    stocktwits_data: dict = None,  # kept for backward compat; not used in v2 features
) -> list:
    """
    Generate multi-horizon ranked signals for all tickers that pass the density gate.

    Density gate: post_count_1d >= 5  — tickers below are skipped silently.
    Tickers in ARCH['drop_tickers'] are always excluded regardless of post count.

    Args:
        reddit_counts:   {ticker: {post_count_1d, mention_growth_7d, total_comments_1d,
                                   vader_sentiment_1d, news_sentiment_1d, ...}}
        model:           single model (backward compat, ignored if models is provided)
        today:           date string YYYY-MM-DD; defaults to UTC today
        models:          preferred — {'1d': model, '3d': model, '5d': model}
        news_data:       {ticker: {news_sentiment_1d, news_count_1d}} from news_fetcher
        stocktwits_data: kept for API compat; v2 models do not use ST features

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

    # ── Pre-fetch SPY 200MA + VIX percentile once per run (shared regime features) ──
    spy_above_200ma: float = 1.0
    vix_percentile:  float = 0.5
    try:
        spy_raw = yf.download('SPY', period='350d', auto_adjust=True, progress=False)
        if isinstance(spy_raw.columns, pd.MultiIndex):
            spy_raw.columns = spy_raw.columns.get_level_values(0)
        spy_close  = spy_raw['Close'].dropna()
        spy_ma200  = float(spy_close.rolling(200, min_periods=100).mean().iloc[-1])
        spy_above_200ma = 1.0 if float(spy_close.iloc[-1]) > spy_ma200 else 0.0
        logger.info(f'regime spy_above_200ma={spy_above_200ma:.0f} '
                    f'spy_close={float(spy_close.iloc[-1]):.2f}')
    except Exception as e:
        logger.warning(f'SPY regime fetch failed: {e} — spy_above_200ma default 1.0')

    try:
        vix_raw = yf.download('^VIX', period='350d', auto_adjust=True, progress=False)
        if isinstance(vix_raw.columns, pd.MultiIndex):
            vix_raw.columns = vix_raw.columns.get_level_values(0)
        vix_vals   = vix_raw['Close'].dropna().values
        window     = vix_vals[-252:] if len(vix_vals) >= 252 else vix_vals
        vix_percentile = float(np.mean(window < vix_vals[-1])) if len(window) >= 20 else 0.5
        logger.info(f'regime vix_percentile={vix_percentile:.3f} '
                    f'vix={float(vix_vals[-1]):.1f}')
    except Exception as e:
        logger.warning(f'VIX percentile fetch failed: {e} — vix_percentile default 0.5')

    # Load mention history for abnormal_attention_1d computation
    mention_history: dict = {}
    try:
        hist_path = Path('data/mention_history.json')
        if hist_path.exists():
            mention_history = json.loads(hist_path.read_text())
    except Exception as e:
        logger.warning(f'mention_history load failed: {e}')

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

        # Improvement 1 — Earnings filter
        try:
            from datetime import date as _date
            _today_date = _date.fromisoformat(today)
            _safe, _earnings_dt = is_safe_to_trade(ticker, _today_date)
            if not _safe:
                logger.info(
                    f'earnings_skip ticker={ticker} '
                    f'next_earnings={_earnings_dt}'
                )
                continue
        except Exception as _e:
            logger.debug(f'earnings_check_error ticker={ticker}: {_e}')

        # Fix 4 — 20-day MA trend filter: only trade tickers above their 20d MA
        try:
            import yfinance as _yf_ma
            _hist_ma = _yf_ma.Ticker(ticker).history(period='30d')
            if len(_hist_ma) >= 20:
                _ma20 = _hist_ma['Close'].tail(20).mean()
                _price_now = _hist_ma['Close'].iloc[-1]
                if _price_now < _ma20:
                    logger.info(
                        f'ma_filter_skip ticker={ticker} '
                        f'price={_price_now:.2f} ma20={_ma20:.2f}')
                    continue
        except Exception as _e:
            logger.warning(f'ma_filter_error ticker={ticker}: {_e}')
            # fail open — do not skip on error

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

        features = compute_features_live(
            ticker=ticker,
            market_data=mkt,
            post_count_1d=post_count,
            news_sentiment_1d=news_sent,
            total_comments_1d=int(reddit_data.get('total_comments_1d', 0)),
            vader_sentiment_1d=float(reddit_data.get('vader_sentiment_1d', 0.0)),
            mention_growth_7d=float(reddit_data.get('mention_growth_7d', 1.0)),
            vix_percentile=vix_percentile,
            spy_above_200ma=spy_above_200ma,
            mention_history=mention_history,
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
            f'vix_pct={vix_percentile:.2f} '
            f'regime={features.get("regime_score", 0):.2f} '
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

    # Improvement 4 — Sector dedup: keep only the top-ranked signal per sector
    sector_map = _load_sector_map()
    sector_counts: dict = {}
    deduped = []
    for sig in result:
        sector = sector_map.get(sig.ticker, 'Unknown')
        count = sector_counts.get(sector, 0)
        if count >= 3 and sector not in ('Index', 'Unknown'):
            logger.debug(
                f'sector_dedup_skip ticker={sig.ticker} sector={sector}'
            )
            continue
        sector_counts[sector] = count + 1
        deduped.append(sig)

    logger.info(f'signals_generated count={len(deduped)} '
                f'(pre_dedup={len(result)}) '
                f'bullish={sum(1 for s in deduped if s.signal == "BULLISH")} '
                f'bearish_logged={sum(1 for s in deduped if s.signal == "BEARISH")} '
                f'neutral={sum(1 for s in deduped if s.signal == "NEUTRAL")} '
                f'date={today}')
    return deduped
