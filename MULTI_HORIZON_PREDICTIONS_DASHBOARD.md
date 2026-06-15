# Claude Code — yfinance News + Multi-Horizon Predictions + Dashboard Update
# Reddit Sentiment Swing Trading System (RSSS)
# GitHub: https://github.com/Ritiksh0h/Reddit-Sentiment-Swing-Trading-System-RSSS

---

## What This Session Builds

```
Task 1: data/news_fetcher.py          ← yfinance news (replaces Tiingo 403)
Task 2: Multi-horizon models           ← train Model_1D, Model_3D, Model_5D
Task 3: Update SignalRecord            ← add pred_1d, pred_3d, confidence, signal
Task 4: New API endpoint               ← GET /predictions (the full signal table)
Task 5: Dashboard update               ← Bullish/Bearish signal panels
```

**End state:** Dashboard shows exactly this:

```
BULLISH SIGNALS
TICKER  PRICE   1D      3D      5D      TARGET(5D)  CONF  POSTS  NEWS  ST
PLTR    $27.43  +0.8%  +2.1%  +3.8%   $28.47       74%   45     3     12
COIN    $165.2  +1.2%  +2.9%  +4.2%   $172.14      71%   38     5     28
NVDA    $205.4  +0.6%  +1.8%  +3.1%   $211.77      68%   89     7     41

BEARISH SIGNALS
TICKER  PRICE   1D      3D      5D      TARGET(5D)  CONF  POSTS  NEWS  ST
AMC     $2.31   -1.1%  -2.4%  -4.1%   $2.22        72%   22     1     8
SNAP    $14.2   -0.8%  -1.9%  -3.3%   $13.73       68%   15     2     5
```

---

## Session Start — Read Current State First

```bash
git pull origin main

# Confirm model exists
ls models/registry/phase3_model.pkl
ls models/registry/phase3_model_baseline.json

# Confirm feature store
python3 -c "
import pandas as pd
df = pd.read_parquet('data/features/features_expanded.parquet')
print(f'Feature store: {len(df)} rows, {len(df.columns)} cols')
print(f'Columns: {[c for c in df.columns if \"target\" in c]}')
"
```

**Confirm target columns exist:** The feature store must have
`target_return_1d`, `target_return_3d`, `target_return_5d`.
If only `target_return_5d` exists, Task 2 adds the missing ones.

---

## Task 1 — data/news_fetcher.py

Replace the broken Tiingo integration with yfinance news.
Zero setup, no API key, already installed.

```python
"""
News fetcher using yfinance.
Replaces data/tiingo_fetcher.py — drop-in replacement, same output format.

No API key required. yfinance is already installed.
Coverage: 20-35 tickers with recent news on a typical trading day.
Latency: ~15-25 seconds for full ticker list.
"""
import logging
import time
from collections import defaultdict

import yfinance as yf

logger = logging.getLogger(__name__)

TRACKED_TICKERS = [
    'NVDA', 'TSLA', 'AMD', 'AAPL', 'GME', 'AMC', 'PLTR', 'MARA', 'COIN',
    'META', 'MSFT', 'AMZN', 'GOOG', 'NFLX', 'SOFI', 'HOOD',
    'ROKU', 'SNAP', 'UBER', 'NIO', 'BABA', 'SHOP', 'PYPL',
    'DKNG', 'DIS', 'RKLB', 'HIMS', 'RDDT', 'SOUN', 'IONQ', 'F',
    'BA', 'BB', 'GS', 'JPM', 'BAC', 'SQ', 'NOK', 'SPCE',
]


def _score_headlines_finbert(headlines: list) -> list:
    """Run FinBERT on headlines. Returns scores -1.0 to +1.0."""
    if not headlines:
        return []
    try:
        from transformers import pipeline as hf_pipeline
        import torch
        if not hasattr(_score_headlines_finbert, '_model'):
            device = 0 if torch.cuda.is_available() else -1
            _score_headlines_finbert._model = hf_pipeline(
                'text-classification',
                model='ProsusAI/finbert',
                device=device,
                truncation=True,
                max_length=128,
            )
        results = _score_headlines_finbert._model([h[:128] for h in headlines])
        scores = []
        for r in results:
            if r['label'] == 'positive':   scores.append(r['score'])
            elif r['label'] == 'negative': scores.append(-r['score'])
            else:                          scores.append(0.0)
        return scores
    except Exception as e:
        logger.warning(f'finbert_failed: {e} — returning neutral')
        return [0.0] * len(headlines)


def fetch_yfinance_news(max_articles_per_ticker: int = 10) -> dict:
    """
    Fetch recent news headlines via yfinance for all tracked tickers.
    Same output format as tiingo_fetcher.fetch_tiingo_news() — drop-in replacement.

    Returns:
        dict of ticker → {
            news_count_1d:     int,
            news_sentiment_1d: float,   # mean FinBERT score -1.0 to +1.0
            news_titles:       list,    # first 3 headlines for logging
        }
    """
    result = {}

    for ticker in TRACKED_TICKERS:
        try:
            t    = yf.Ticker(ticker)
            news = t.news or []
            if not news:
                continue

            # Handle both old and new yfinance response formats
            titles = []
            for n in news[:max_articles_per_ticker]:
                title = (
                    n.get('content', {}).get('title') or
                    n.get('title') or
                    ''
                )
                if title:
                    titles.append(title.strip())

            if not titles:
                continue

            scores = _score_headlines_finbert(titles)
            result[ticker] = {
                'news_count_1d':     len(titles),
                'news_sentiment_1d': round(
                    sum(scores) / len(scores), 4
                ) if scores else 0.0,
                'news_titles': titles[:3],
            }
            time.sleep(0.1)   # gentle rate limiting

        except Exception as e:
            logger.warning(f'yfinance_news_failed ticker={ticker}: {e}')
            continue

    logger.info(f'yfinance_news_fetched tickers={len(result)}')
    return result
```

### Update daily_run_live.py Step 1b

Find:
```python
from data.tiingo_fetcher import fetch_tiingo_news
news_data = fetch_tiingo_news(hours_back=24)
```

Replace with:
```python
from data.news_fetcher import fetch_yfinance_news
news_data = fetch_yfinance_news()
```

### Test

```bash
python3 -c "
from data.news_fetcher import fetch_yfinance_news
result = fetch_yfinance_news()
print(f'News tickers: {len(result)}')
for t, d in list(result.items())[:5]:
    print(f'  {t}: {d[\"news_count_1d\"]} articles  sentiment={d[\"news_sentiment_1d\"]:.3f}')
    print(f'     → {d[\"news_titles\"][0][:70]}')
"
```

Expected: 20-35 tickers with news, sentiment values between -0.5 and +0.5.

---

## Task 2 — Multi-Horizon Models (1D, 3D, 5D)

### Step 2a — Add missing target columns to feature store

Check if `target_return_1d` and `target_return_3d` exist:

```python
# Run this check first
import pandas as pd
df = pd.read_parquet('data/features/features_expanded.parquet')
print([c for c in df.columns if 'target' in c])
```

If `target_return_1d` or `target_return_3d` are missing, add them to
`pipeline/01_feature_builder.py`. Find where `target_return_5d` is computed:

```python
# Find this line in pipeline/01_feature_builder.py:
raw['target_return_5d'] = raw['close'].shift(-5) / raw['close'] - 1

# Add these two lines immediately after:
raw['target_return_1d'] = raw['close'].shift(-1) / raw['close'] - 1
raw['target_return_3d'] = raw['close'].shift(-3) / raw['close'] - 1
```

Then rebuild the feature store:
```bash
python pipeline/01_feature_builder.py --force-recompute
# Also rebuild expanded features if they exist separately:
# python pipeline/01_feature_builder.py --input-file data/raw/merged_with_sentiment_expanded.parquet --output-file data/features/features_expanded.parquet --force-recompute
```

### Step 2b — Update train_phase3_model.py for multi-horizon

Replace the entire `scripts/train_phase3_model.py` with:

```python
"""
Train Phase 3 multi-horizon XGBoost models.
Trains three separate regressors: Model_1D, Model_3D, Model_5D.

Each model predicts forward return for its horizon.
All three use the same 14-feature set.
Saves to models/registry/model_{1d,3d,5d}.pkl

Run: python scripts/train_phase3_model.py
"""
import json
import pickle
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import xgboost as xgb

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Load architecture contract
with open('experiments/phase3_locked_architecture.json') as f:
    ARCH = json.load(f)

FEATURES     = ARCH['features']
DROP_TICKERS = set(ARCH['drop_tickers'])

XGB_PARAMS = dict(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.6, colsample_bytree=0.6, min_child_weight=20,
    reg_alpha=0.5, reg_lambda=2.0,
    random_state=42, n_jobs=-1,
    objective='reg:squarederror',
)

HORIZONS = {
    '1d': 'target_return_1d',
    '3d': 'target_return_3d',
    '5d': 'target_return_5d',
}


def train(
    feature_path: str = 'data/features/features_expanded.parquet',
    output_dir:   str = 'models/registry',
) -> dict:
    """Train all three horizon models. Returns metrics dict."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(feature_path)
    df = df[df['post_count_1d'] >= 10].copy()
    df = df[~df['ticker'].isin(DROP_TICKERS)].copy()

    # Add 0.0 defaults for new features not in historical data
    NEW_FEATURES = ['news_sentiment_1d', 'st_sentiment_1d', 'st_bull_pct']
    for feat in NEW_FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0

    train_df = df[df['split'] == 'train']
    test_df  = df[df['split'] == 'test']

    avail = [f for f in FEATURES if f in train_df.columns]
    logger.info(f'Training on {len(train_df)} rows, testing on {len(test_df)} rows')
    logger.info(f'Features: {len(avail)} — {avail}')

    metrics  = {}
    models   = {}

    for horizon, target_col in HORIZONS.items():
        if target_col not in train_df.columns:
            logger.warning(f'Missing target {target_col} — skipping {horizon}')
            continue

        logger.info(f'Training Model_{horizon.upper()}...')

        X_tr = train_df[avail].fillna(0)
        y_tr = train_df[target_col]
        X_te = test_df[avail].fillna(0)
        y_te = test_df[target_col]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr, y_tr)

        pred_te  = model.predict(X_te)
        pred_tr  = model.predict(X_tr)
        ic_te    = float(stats.spearmanr(pred_te, y_te).correlation)
        ic_tr    = float(stats.spearmanr(pred_tr, y_tr).correlation)
        dir_acc  = float(np.mean(np.sign(pred_te) == np.sign(y_te)))

        logger.info(f'Model_{horizon.upper()}: IC_test={ic_te:.4f}  '
                    f'IC_train={ic_tr:.4f}  dir_acc={dir_acc:.3f}')

        # Save model
        model_path = Path(output_dir) / f'model_{horizon}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        models[horizon]  = model
        metrics[horizon] = {
            'ic_test':    round(ic_te, 4),
            'ic_train':   round(ic_tr, 4),
            'dir_acc':    round(dir_acc, 3),
            'model_path': str(model_path),
            'n_train':    len(train_df),
            'n_test':     len(test_df),
        }

    # Save backward-compatible model_5d as phase3_model.pkl
    if '5d' in models:
        compat_path = Path(output_dir) / 'phase3_model.pkl'
        with open(compat_path, 'wb') as f:
            pickle.dump(models['5d'], f)
        logger.info(f'Saved backward-compatible: {compat_path}')

    # Save metrics
    metadata = {
        'model_version':  'phase3_v3_multihorizon',
        'trained_at':     datetime.now(timezone.utc).isoformat(),
        'features':       avail,
        'feature_count':  len(avail),
        'horizons':       metrics,
        'fix3_trigger':   'live 30-day IC < 0.01 → switch to 17 features',
    }
    meta_path = Path(output_dir) / 'phase3_model_baseline.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info('All models saved.')
    for h, m in metrics.items():
        logger.info(f'  Model_{h.upper()}: IC={m["ic_test"]:.4f}  '
                    f'DirAcc={m["dir_acc"]:.1%}')

    return metrics


if __name__ == '__main__':
    results = train()
    import json as _json
    print(_json.dumps(results, indent=2))
```

Run training:
```bash
python scripts/train_phase3_model.py
```

Expected output:
```
Model_1D: IC_test=0.04-0.07  dir_acc=51-54%
Model_3D: IC_test=0.06-0.09  dir_acc=52-55%
Model_5D: IC_test=0.09-0.11  dir_acc=52-55%
```

1D IC will be lower than 5D — short-term returns are harder to predict.
That is expected and correct.

---

## Task 3 — Update SignalRecord and signal_generator.py

### Update the SignalRecord dataclass

Find the `@dataclass class SignalRecord:` block in
`portfolio/signal_generator.py` and replace it:

```python
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
```

### Update load_model() to load all three models

Replace the existing `load_model()` function:

```python
def load_models(model_dir: str = 'models/registry') -> dict:
    """
    Load all three horizon models.
    Returns dict: {'1d': model, '3d': model, '5d': model}
    Falls back to phase3_model.pkl for 5d if individual models missing.
    """
    import pickle
    models = {}
    dir_path = Path(model_dir)

    for horizon in ['1d', '3d', '5d']:
        model_path = dir_path / f'model_{horizon}.pkl'
        if model_path.exists():
            with open(model_path, 'rb') as f:
                models[horizon] = pickle.load(f)
            logger.info(f'Loaded model_{horizon}')
        else:
            # Fall back to legacy model for 5d
            if horizon == '5d':
                fallback = dir_path / 'phase3_model.pkl'
                if fallback.exists():
                    with open(fallback, 'rb') as f:
                        models['5d'] = pickle.load(f)
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
        return pickle.load(f)
```

### Update generate_signals() for multi-horizon

Replace the entire `generate_signals()` function:

```python
def generate_signals(
    reddit_counts: dict,
    model=None,           # kept for backward compat — ignored if models provided
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
        list of SignalRecord sorted by predicted_5d descending (bullish first)
    """
    if today is None:
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    # Load models if not provided
    if models is None:
        try:
            models = load_models()
        except Exception:
            if model is not None:
                models = {'5d': model, '3d': model, '1d': model}
            else:
                raise

    ts = datetime.now(timezone.utc).isoformat()
    signals = []

    for ticker, reddit_data in reddit_counts.items():
        if ticker in DROP_TICKERS:
            continue

        post_count = reddit_data.get('post_count_1d', 0)
        if post_count < DENSITY_GATE:
            logger.debug(f'density_gate_fail ticker={ticker} posts={post_count}')
            continue

        # Download market data
        try:
            mkt = yf.download(ticker, period='90d',
                              auto_adjust=True, progress=False)
            if isinstance(mkt.columns, pd.MultiIndex):
                mkt.columns = mkt.columns.get_level_values(0)
        except Exception as e:
            logger.warning(f'market_data_fail ticker={ticker}: {e}')
            continue

        # Compute features
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

        # Predict all three horizons
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

        # Filter by minimum predicted return (5d primary signal)
        if abs(pred_5d) < MIN_PRED_RET:
            continue

        # Confidence score: how far above/below the min threshold is the prediction?
        # Scale: 0.0 = exactly at threshold, 1.0 = 3x threshold or beyond
        confidence = min(abs(pred_5d) / (MIN_PRED_RET * 3), 1.0)

        # Signal classification
        if pred_5d >= 0.03 and confidence >= 0.5:
            signal = 'BULLISH'
        elif pred_5d <= -0.03 and confidence >= 0.5:
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

    # Sort: BULLISH first (by pred_5d desc), BEARISH last (by pred_5d asc)
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
```

### Update generate_signals call in daily_run_live.py

Find the dry-run signal generation call and update it:

```python
    if args.dry_run:
        logger.info('DRY RUN MODE — signals logged, no trades executed')
        try:
            from portfolio.signal_generator import generate_signals, load_models
            models  = load_models()
            signals = generate_signals(reddit_counts, models=models, today=today)
            logger.info(f'Dry run: {len(signals)} qualifying signals')
            for s in signals[:5]:
                logger.info(
                    f'  {s.signal:<8} {s.ticker:<6} '
                    f'1D={s.predicted_1d:+.2%}  '
                    f'3D={s.predicted_3d:+.2%}  '
                    f'5D={s.predicted_5d:+.2%}  '
                    f'target={s.price_target_5d:.2f}  '
                    f'conf={s.confidence:.0%}  '
                    f'posts={s.post_count_1d}  '
                    f'news={s.news_count_1d}  '
                    f'st={s.st_count_1d}'
                )
        except Exception as e:
            logger.error(f'Dry run failed: {e}')
        return
```

---

## Task 4 — New API Endpoint: GET /predictions

Add to `api/main.py`:

```python
@app.get('/predictions')
def get_predictions():
    """
    Return today's multi-horizon predictions with signal classification.
    Reads from the most recent execution log entries (today only).

    Returns bullish signals first, bearish last.
    Includes 1D, 3D, 5D predictions, price targets, and confidence.
    """
    from datetime import date as _date
    today = _date.today().isoformat()

    log_path = Path('logs/paper_trades.jsonl')
    if not log_path.exists():
        return {'bullish': [], 'bearish': [], 'neutral': [], 'date': today}

    # Load today's OPEN signals
    todays_signals = []
    with open(log_path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if (record.get('action') == 'OPEN' and
                    record.get('date') == today):
                todays_signals.append(record)

    # If no trades today, load from last run date
    if not todays_signals:
        with open(log_path) as f:
            lines = [l for l in f if l.strip()]
        if lines:
            last = json.loads(lines[-1])
            last_date = last.get('date')
            with open(log_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if (record.get('action') == 'OPEN' and
                            record.get('date') == last_date):
                        todays_signals.append(record)

    def _format_signal(r: dict) -> dict:
        fv = r.get('feature_vector_11') or r.get('feature_vector') or {}
        return {
            'ticker':         r.get('ticker'),
            'price':          r.get('fill_price', 0),
            'signal':         r.get('signal', 'NEUTRAL'),
            'predicted_1d':   r.get('predicted_1d', 0),
            'predicted_3d':   r.get('predicted_3d', 0),
            'predicted_5d':   r.get('predicted_return_5d', 0),
            'price_target_1d': r.get('price_target_1d'),
            'price_target_3d': r.get('price_target_3d'),
            'price_target_5d': r.get('price_target_5d'),
            'confidence':     r.get('confidence', 0),
            'post_count_1d':  fv.get('post_count_1d', r.get('post_count_1d', 0)),
            'news_count_1d':  r.get('news_count_1d', 0),
            'st_count_1d':    r.get('st_count_1d', 0),
            'regime':         r.get('regime_state', 'neutral'),
            'date':           r.get('date'),
        }

    formatted = [_format_signal(s) for s in todays_signals]
    bullish   = [s for s in formatted if s['signal'] == 'BULLISH']
    bearish   = [s for s in formatted if s['signal'] == 'BEARISH']
    neutral   = [s for s in formatted if s['signal'] == 'NEUTRAL']

    # Sort: bullish by predicted_5d desc, bearish by predicted_5d asc
    bullish.sort(key=lambda x: x['predicted_5d'], reverse=True)
    bearish.sort(key=lambda x: x['predicted_5d'])

    return {
        'date':    today,
        'bullish': bullish,
        'bearish': bearish,
        'neutral': neutral,
        'total':   len(formatted),
    }
```

Also update the execution logger to save the new fields.
In `portfolio/execution_logger.py`, add to the `log_signal()` function signature
and record dict:

```python
def log_signal(
    ...,
    predicted_1d:    float = 0.0,     # ← new
    predicted_3d:    float = 0.0,     # ← new
    signal:          str   = 'NEUTRAL', # ← new
    price_target_1d: float = None,    # ← new
    price_target_3d: float = None,    # ← new
    price_target_5d: float = None,    # ← new
    confidence:      float = 0.0,     # ← new
    news_count_1d:   int   = 0,       # ← new
    st_count_1d:     int   = 0,       # ← new
    ...,
) -> None:
```

Add the new fields to the `record` dict inside the function:
```python
    record = {
        ...,
        'signal':          signal,
        'predicted_1d':    predicted_1d,
        'predicted_3d':    predicted_3d,
        'price_target_1d': price_target_1d,
        'price_target_3d': price_target_3d,
        'price_target_5d': price_target_5d,
        'confidence':      confidence,
        'news_count_1d':   news_count_1d,
        'st_count_1d':     st_count_1d,
        ...,
    }
```

---

## Task 5 — Dashboard Update

Add a new Predictions section to `dashboard/index.html`.

Find the existing KPI grid section and add this block IMMEDIATELY AFTER the
closing `</div>` of the KPI grid, BEFORE the `.grid-2 section`:

```html
<!-- ── PREDICTIONS PANEL ──────────────────────────────────────────── -->
<div class="panel section" id="predictions-panel">
  <div class="panel-header" onclick="togglePanel(this)">
    <span class="panel-title">Today's Predictions</span>
    <span id="pred-summary" style="font-family:var(--mono);font-size:11px;color:var(--dim);"></span>
    <span class="collapse-icon">▾</span>
  </div>
  <div class="panel-body">

    <!-- Bullish signals -->
    <div style="margin-bottom:20px;">
      <div style="font-family:var(--mono);font-size:10px;color:var(--green);
                  letter-spacing:.15em;text-transform:uppercase;
                  margin-bottom:10px;padding-bottom:8px;
                  border-bottom:1px solid rgba(0,214,143,0.2);">
        ▲ BULLISH SIGNALS
      </div>
      <div class="tbl-wrap">
        <table>
          <thead>
            <tr>
              <th>Ticker</th><th>Price</th>
              <th>1D</th><th>3D</th><th>5D</th>
              <th>Target (5D)</th><th>Conf</th>
              <th>Posts</th><th>News</th><th>ST</th>
              <th>Regime</th>
            </tr>
          </thead>
          <tbody id="bullish-tbody">
            <tr><td colspan="11" class="empty">No bullish signals today</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Bearish signals -->
    <div>
      <div style="font-family:var(--mono);font-size:10px;color:var(--red);
                  letter-spacing:.15em;text-transform:uppercase;
                  margin-bottom:10px;padding-bottom:8px;
                  border-bottom:1px solid rgba(255,69,96,0.2);">
        ▼ BEARISH SIGNALS
      </div>
      <div class="tbl-wrap">
        <table>
          <thead>
            <tr>
              <th>Ticker</th><th>Price</th>
              <th>1D</th><th>3D</th><th>5D</th>
              <th>Target (5D)</th><th>Conf</th>
              <th>Posts</th><th>News</th><th>ST</th>
              <th>Regime</th>
            </tr>
          </thead>
          <tbody id="bearish-tbody">
            <tr><td colspan="11" class="empty">No bearish signals today</td></tr>
          </tbody>
        </table>
      </div>
    </div>

  </div>
</div>
```

Add this CSS to the `<style>` block (before the closing `</style>`):

```css
/* ── Prediction table rows ─────────────────────────────────────────── */
.pred-row-bull { border-left: 2px solid var(--green); }
.pred-row-bear { border-left: 2px solid var(--red); }
.conf-bar {
  display: inline-block; height: 6px; border-radius: 3px;
  background: var(--green); vertical-align: middle; margin-right: 4px;
}
```

Add this JavaScript function to the `<script>` block,
before the closing `</script>`:

```javascript
// ── Predictions panel ─────────────────────────────────────────────────
async function updatePredictions() {
  const data = await apiFetch('/predictions');
  if (!data) return;

  const bullish = data.bullish || [];
  const bearish = data.bearish || [];
  const total   = data.total   || 0;

  // Update summary badge
  document.getElementById('pred-summary').textContent =
    total > 0
      ? `${bullish.length} bullish  ${bearish.length} bearish`
      : 'No signals today';

  function renderRows(signals, tbodyId, isBull) {
    const tbody = document.getElementById(tbodyId);
    if (!signals.length) {
      tbody.innerHTML = `<tr><td colspan="11" class="empty">No ${isBull ? 'bullish' : 'bearish'} signals today</td></tr>`;
      return;
    }

    tbody.innerHTML = signals.map(s => {
      const p1    = ((s.predicted_1d || 0) * 100).toFixed(1);
      const p3    = ((s.predicted_3d || 0) * 100).toFixed(1);
      const p5    = ((s.predicted_5d || 0) * 100).toFixed(1);
      const conf  = Math.round((s.confidence || 0) * 100);
      const price = (s.price || 0).toFixed(2);
      const tgt5  = s.price_target_5d ? s.price_target_5d.toFixed(2) : '—';
      const regime = s.regime || 'neutral';
      const rowCls = isBull ? 'pred-row-bull' : 'pred-row-bear';
      const pCls   = isBull ? 'pos' : 'neg';
      const sign   = isBull ? '+' : '';
      const confBarW = Math.round(conf * 0.8);  // max 80px

      return `<tr class="${rowCls}">
        <td><span class="ticker-badge">${s.ticker}</span></td>
        <td class="dim-text">$${price}</td>
        <td class="${pCls}">${sign}${p1}%</td>
        <td class="${pCls}">${sign}${p3}%</td>
        <td class="${pCls}" style="font-weight:600;">${sign}${p5}%</td>
        <td class="${pCls}" style="font-weight:600;">$${tgt5}</td>
        <td>
          <div style="display:flex;align-items:center;gap:6px;">
            <div class="conf-bar" style="width:${confBarW}px;background:${isBull ? 'var(--green)' : 'var(--red)'}"></div>
            <span class="dim-text" style="font-family:var(--mono);font-size:11px;">${conf}%</span>
          </div>
        </td>
        <td class="dim-text">${s.post_count_1d || 0}</td>
        <td class="dim-text">${s.news_count_1d || 0}</td>
        <td class="dim-text">${s.st_count_1d || 0}</td>
        <td><span class="regime ${regime}">${regime.slice(0,3).toUpperCase()}</span></td>
      </tr>`;
    }).join('');
  }

  renderRows(bullish, 'bullish-tbody', true);
  renderRows(bearish, 'bearish-tbody', false);
}
```

Add `updatePredictions()` to the `refreshAll()` function:

```javascript
async function refreshAll() {
  // ... existing code ...
  updatePredictions();   // ← add this line
  // ... existing code ...
}
```

---

## Build Order

```bash
# Step 1: Create data/news_fetcher.py
# Step 2: Update daily_run_live.py Step 1b (tiingo → yfinance)

# Step 3: Add target_return_1d and target_return_3d to feature builder
# (only if missing from feature store)
python3 -c "
import pandas as pd
df = pd.read_parquet('data/features/features_expanded.parquet')
print([c for c in df.columns if 'target' in c])
"

# Step 4: Rebuild feature store if targets missing
# python pipeline/01_feature_builder.py --force-recompute

# Step 5: Replace train_phase3_model.py with multi-horizon version
# Step 6: Train all three models
python scripts/train_phase3_model.py

# Step 7: Update SignalRecord dataclass (add 10 new fields)
# Step 8: Replace generate_signals() with multi-horizon version
# Step 9: Update load_model() → load_models()

# Step 10: Add /predictions endpoint to api/main.py
# Step 11: Update execution_logger.py to save new fields

# Step 12: Add predictions panel HTML to dashboard/index.html
# Step 13: Add CSS and updatePredictions() JS

# Step 14: Test dry run
python scripts/daily_run_live.py --dry-run
# Expected: signals show 1D/3D/5D predictions and signal classification

# Step 15: Test API endpoint
curl http://localhost:8000/predictions | python3 -m json.tool

# Step 16: Open dashboard and verify predictions panel
open http://localhost:8000/dashboard

# Step 17: Push
bash push.sh "[feature] multi-horizon models + predictions dashboard panel"
```

---

## Expected Dry Run Output After All Changes

```
INFO  Fetching live Reddit data...
INFO  posts_fetched total=379 tickers_found=5
INFO  Fetching yfinance news...
INFO  yfinance_news_fetched tickers=28
INFO  Fetching StockTwits...
INFO  stocktwits_fetched tickers=38 failures=0
INFO  Combined data: 38 tickers across all sources
INFO  DRY RUN MODE
INFO  BULLISH  PLTR   1D=+0.8%  3D=+2.1%  5D=+3.8%  target=28.47  conf=74%  posts=45  news=3  st=12
INFO  NEUTRAL  TSLA   1D=+0.4%  3D=+1.1%  5D=+2.1%  target=415.3  conf=51%  posts=89  news=7  st=41
INFO  BEARISH  AMC    1D=-1.1%  3D=-2.4%  5D=-4.1%  target=2.22   conf=72%  posts=22  news=1  st=8
INFO  signals_generated count=X bullish=X neutral=X bearish=X
```

---

## Hard Rules for Claude Code

- NEVER remove the existing `predicted_return` field from SignalRecord
  (backward compat with daily_run.py position sizing)
- NEVER change the density gate — post_count_1d >= 10 stays
- ALWAYS load all three models in load_models() with graceful fallback
- NEVER assume target_return_1d exists — check first, add if missing
- ALWAYS maintain backward compat: phase3_model.pkl = copy of model_5d.pkl
- NEVER hardcode signal thresholds — use BULLISH >= 3%, BEARISH <= -3%
- The predictions panel is ADDITIVE — existing dashboard panels unchanged
- ALWAYS test /predictions endpoint returns valid JSON before pushing

---

## Files to Create

```
CREATE:   data/news_fetcher.py
```

## Files to Modify

```
MODIFY:   scripts/daily_run_live.py          ← Step 1b: tiingo → yfinance
MODIFY:   pipeline/01_feature_builder.py     ← add target_return_1d, target_return_3d
MODIFY:   scripts/train_phase3_model.py      ← multi-horizon training
MODIFY:   portfolio/signal_generator.py      ← new SignalRecord + generate_signals
MODIFY:   portfolio/execution_logger.py      ← new fields in log_signal()
MODIFY:   api/main.py                        ← add /predictions endpoint
MODIFY:   dashboard/index.html              ← predictions panel
```

## Files NOT to touch

```
portfolio/portfolio_engine.py   ← unchanged
scripts/daily_run.py            ← unchanged
portfolio/drift_monitor.py      ← unchanged (add news to HISTORICAL_MEANS after 30 days)
data/reddit_live_fetcher.py     ← unchanged
data/stocktwits_fetcher.py      ← unchanged
experiments/                    ← locked
data/features/                  ← only rebuild if targets missing
```

---

*Multi-Horizon Predictions + Dashboard — June 2026*
*Three models (1D, 3D, 5D) + yfinance news + predictions panel*
*End state: dashboard shows bullish/bearish table with price targets*
