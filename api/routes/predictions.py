"""
RSSS API — signal prediction and SHAP attribution routes.
"""
import json
from pathlib import Path

from fastapi import APIRouter

from api._helpers import _load_trade_log, _sanitize

router = APIRouter()


@router.get('/top-bullish')
def get_top_bullish(n: int = 5):
    """
    Top n tickers by composite score from the latest daily run.
    Composite = 0.5 * pred5d + 0.3 * pred3d + 0.2 * pred1d
    Only OPEN signals are considered; sorted descending by composite.
    """
    trades = _load_trade_log(300)
    opens  = [t for t in trades if t.get('action') == 'OPEN']
    if not opens:
        return {'date': None, 'tickers': [], 'total': 0}

    latest_date = max(t.get('date', '') for t in opens)
    day_opens   = [t for t in opens if t.get('date') == latest_date]

    def _composite(t: dict) -> float:
        p5 = float(t.get('predicted_5d') or t.get('predicted_return_5d') or 0)
        p3 = float(t.get('predicted_3d') or 0)
        p1 = float(t.get('predicted_1d') or 0)
        return 0.5 * p5 + 0.3 * p3 + 0.2 * p1

    ranked = sorted(day_opens, key=_composite, reverse=True)[:n]

    result = []
    for t in ranked:
        p5 = float(t.get('predicted_5d') or t.get('predicted_return_5d') or 0)
        p3 = float(t.get('predicted_3d') or 0)
        p1 = float(t.get('predicted_1d') or 0)
        result.append({
            'ticker':          t.get('ticker'),
            'signal':          t.get('signal', 'NEUTRAL'),
            'composite_score': round(_composite(t), 6),
            'predicted_1d':    round(p1, 6),
            'predicted_3d':    round(p3, 6),
            'predicted_5d':    round(p5, 6),
            'confidence':      t.get('confidence', 0),
            'post_count_1d':   t.get('post_count_1d', 0),
            'date':            t.get('date'),
        })

    return {'date': latest_date, 'tickers': result, 'total': len(result)}


@router.get('/top-predictions')
def get_top_predictions():
    trades = _load_trade_log(200)
    opens  = [t for t in trades if t.get('action') == 'OPEN']
    opens.sort(
        key=lambda x: float(x.get('predicted_return_5d') or x.get('predicted_5d') or 0),
        reverse=True,
    )
    return opens[:10]


@router.get('/predictions')
def get_predictions(ticker: str = None, n: int = 30):
    """
    Return the latest day's OPEN signals with multi-horizon predictions.
    If ticker is specified, return single-ticker format for the dashboard.
    """
    trades = _load_trade_log(300)
    opens  = [t for t in trades if t.get('action') == 'OPEN']
    if not opens:
        if ticker:
            return {
                '1D': {'pred': 0.0, 'conf': 48},
                '3D': {'pred': 0.0, 'conf': 49},
                '5D': {'pred': 0.0, 'conf': 50},
                'density_passed': False,
                'post_count_1d':  0,
                'signal':         'NEUTRAL',
                'ticker':         ticker.upper(),
                'message':        'No signals logged yet',
            }
        return []

    latest_date = max(t.get('date', '') for t in opens)
    day_opens   = [t for t in opens if t.get('date') == latest_date]

    bullish = sorted(
        [t for t in day_opens if t.get('signal') == 'BULLISH'],
        key=lambda x: x.get('predicted_return_5d', 0), reverse=True,
    )
    neutral = sorted(
        [t for t in day_opens if t.get('signal', 'NEUTRAL') == 'NEUTRAL'],
        key=lambda x: abs(x.get('predicted_return_5d', 0)), reverse=True,
    )
    bearish = sorted(
        [t for t in day_opens if t.get('signal') == 'BEARISH'],
        key=lambda x: x.get('predicted_return_5d', 0),
    )

    if ticker:
        ticker = ticker.upper()
        match  = next(
            (s for s in bullish + bearish + neutral if s.get('ticker') == ticker),
            None,
        )
        if match:
            fv_        = (match.get('feature_vector_14') or match.get('feature_vector_11')
                          or match.get('feature_vector') or {})
            pred_5d    = float(match.get('predicted_5d') or match.get('predicted_return_5d') or 0) * 100
            pred_3d    = float(match.get('predicted_3d') or 0) * 100
            pred_1d    = float(match.get('predicted_1d') or 0) * 100
            conf_5d    = int(float(match.get('confidence') or 0) * 100)
            post_count = int(match.get('post_count_1d') or fv_.get('post_count_1d') or 0)
            return {
                '1D': {'pred': round(pred_1d, 2), 'conf': max(int(conf_5d * 0.85), 40)},
                '3D': {'pred': round(pred_3d, 2), 'conf': max(int(conf_5d * 0.92), 45)},
                '5D': {'pred': round(pred_5d, 2), 'conf': conf_5d},
                'density_passed': post_count >= 10,
                'post_count_1d':  post_count,
                'signal':         match.get('signal', 'NEUTRAL'),
                'ticker':         ticker,
            }
        else:
            return {
                '1D': {'pred': 0.0, 'conf': 48},
                '3D': {'pred': 0.0, 'conf': 49},
                '5D': {'pred': 0.0, 'conf': 50},
                'density_passed': False,
                'post_count_1d':  0,
                'signal':         'NEUTRAL',
                'ticker':         ticker,
                'message':        "Ticker not in today's signals or density gate not met",
            }

    formatted = bullish + neutral + bearish
    return {
        'date':    latest_date,
        'bullish': bullish,
        'bearish': bearish,
        'neutral': neutral,
        'total':   len(formatted),
    }


@router.get('/shap/{ticker}')
def get_shap_values(ticker: str):
    """
    SHAP feature contributions for the latest OPEN signal for this ticker.
    Groups by source family: Reddit / News / StockTwits / Market.
    Positive SHAP = pushed prediction bullish. Negative = bearish.
    """
    import pickle
    import shap
    import pandas as pd

    model_path = Path('models/registry/model_5d.pkl')
    arch_path  = Path('experiments/phase3_locked_architecture.json')
    log_path   = Path('logs/paper_trades.jsonl')

    if not model_path.exists():
        return {'error': 'model_5d.pkl not found'}
    if not arch_path.exists():
        return {'error': 'phase3_locked_architecture.json not found'}
    if not log_path.exists():
        return {'error': 'no signals logged'}

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(arch_path) as f:
        arch = json.load(f)
    features = arch['features']

    latest = None
    with open(log_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('ticker') == ticker.upper() and r.get('action') == 'OPEN':
                latest = r

    if not latest:
        return {'error': f'no OPEN signal found for {ticker}'}

    fv_raw = (latest.get('feature_vector_14')
              or latest.get('feature_vector_11')
              or latest.get('feature_vector')
              or {})
    fv = dict(fv_raw)
    for feat in features:
        if feat not in fv:
            fv[feat] = 0.0

    avail = [f for f in features if f in fv]
    X = pd.DataFrame([[fv[f] for f in avail]], columns=avail).fillna(0)

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)[0]
        base_val  = float(explainer.expected_value)
        prediction = base_val + float(sum(shap_vals))
    except Exception as e:
        return {'error': f'SHAP computation failed: {e}'}

    SOURCE_FAMILIES = {
        'reddit':     ['post_count_1d', 'mention_growth_1d', 'mention_growth_7d'],
        'news':       ['news_sentiment_1d'],
        'stocktwits': ['st_sentiment_1d', 'st_bull_pct'],
        'market':     ['returns_1d', 'returns_5d', 'returns_20d', 'rsi_14', 'atr_14',
                       'relative_volume', 'dist_from_20ma', 'dist_from_50ma'],
    }

    contributions = [
        {
            'feature':       feat,
            'shap_value':    round(float(sv), 6),
            'feature_value': round(float(fv.get(feat, 0)), 4),
            'direction':     'bullish' if sv > 0 else 'bearish',
        }
        for feat, sv in zip(avail, shap_vals)
    ]
    contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)

    family_shap = {}
    for family, family_feats in SOURCE_FAMILIES.items():
        total = sum(sv for f, sv in zip(avail, shap_vals) if f in family_feats)
        family_shap[family] = round(float(total), 6)

    drivers = sorted(family_shap.items(), key=lambda x: abs(x[1]), reverse=True)
    primary = drivers[0][0].title() if drivers else 'Market'
    attribution_text = (
        f"{ticker.upper()} signal driven by {primary} "
        f"(SHAP={drivers[0][1]:+.4f}). "
        f"Reddit: {family_shap.get('reddit', 0):+.4f} | "
        f"News: {family_shap.get('news', 0):+.4f} | "
        f"StockTwits: {family_shap.get('stocktwits', 0):+.4f} | "
        f"Market: {family_shap.get('market', 0):+.4f}."
    )

    total_abs  = sum(abs(v) for v in family_shap.values()) or 1.0
    reddit_pct = max(int(abs(family_shap.get('reddit', 0))     / total_abs * 100), 0)
    news_pct   = max(int(abs(family_shap.get('news', 0))       / total_abs * 100), 0)
    st_pct     = max(int(abs(family_shap.get('stocktwits', 0)) / total_abs * 100), 0)
    market_pct = max(int(abs(family_shap.get('market', 0))     / total_abs * 100), 0)
    total_pct  = reddit_pct + news_pct + st_pct + market_pct
    if total_pct > 0 and total_pct != 100:
        market_pct += (100 - total_pct)

    return _sanitize({
        'reddit_attention':  reddit_pct,
        'reddit_sentiment':  0,
        'news_sentiment':    news_pct,
        'st_sentiment':      st_pct,
        'market_technical':  market_pct,
        'ticker':            ticker.upper(),
        'date':              latest.get('date'),
        'base_value':        round(base_val, 4),
        'prediction':        round(prediction, 4),
        'signal':            latest.get('signal', 'NEUTRAL'),
        'family_shap_raw':   family_shap,
        'attribution_text':  attribution_text,
        'top_features':      contributions[:8],
    })
