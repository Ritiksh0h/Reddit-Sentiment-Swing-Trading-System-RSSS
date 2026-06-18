"""
FastAPI endpoints for paper trading monitoring.
Run: uvicorn api.main:app --reload --port 8000
"""
import json
import subprocess
import sys
from pathlib import Path

import math

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


def _sanitize(obj):
    """Recursively replace NaN/Inf floats with None for safe JSON serialization."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(i) for i in obj]
    return obj

app = FastAPI(title='RSSS Paper Trading API', version='3.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


def _load_portfolio() -> dict:
    path = Path('data/paper_portfolio.json')
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _load_trade_log(last_n: int = 50) -> list:
    path = Path('logs/paper_trades.jsonl')
    if not path.exists():
        return []
    lines = [l for l in path.read_text().strip().split('\n') if l]
    records = [json.loads(l) for l in lines]
    return records[-last_n:]


@app.get('/health')
def health():
    return {'status': 'ok', 'version': '3.0'}


@app.get('/portfolio')
def get_portfolio():
    INITIAL_CAPITAL = 10000.0

    portfolio = _load_portfolio()
    cash      = float(portfolio.get('cash', INITIAL_CAPITAL))
    positions = portfolio.get('positions', [])

    position_value = sum(float(p.get('position_dollars', 0)) for p in positions)
    equity         = cash + position_value
    total_return   = round((equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2)

    try:
        from portfolio.regime_detector import RegimeDetector
        regime_label = RegimeDetector().get_current_regime().upper()
    except Exception:
        regime_label = 'NEUTRAL'

    _REGIME_SIZING = {'POSITIVE': 100, 'NEUTRAL': 75, 'NEGATIVE': 50}
    sizing_pct     = _REGIME_SIZING.get(regime_label, 75)

    return {
        **portfolio,
        'equity':           round(equity, 2),
        'total_return_pct': total_return,
        'positions_count':  len(positions),
        'regime_label':     regime_label,
        'sizing_pct':       sizing_pct,
    }


@app.get('/positions')
def get_positions():
    return _load_portfolio().get('positions', [])


@app.get('/signals/recent')
def get_recent_signals(n: int = 20):
    trades = _load_trade_log(n * 3)
    return [t for t in trades if t.get('action') == 'OPEN'][-n:]


@app.get('/top-predictions')
def get_top_predictions():
    trades = _load_trade_log(200)
    opens  = [t for t in trades if t.get('action') == 'OPEN']
    opens.sort(key=lambda x: x.get('predicted_return_5d', 0), reverse=True)
    return opens[:10]


@app.get('/performance')
def get_performance():
    """Paper trading performance summary."""
    state  = _load_portfolio()
    closed = state.get('closed_trades', [])
    if not closed:
        return {'message': 'No closed trades yet', 'n_trades': 0}

    pnls = [t.get('pnl_pct', 0) for t in closed]
    wins = [p for p in pnls if p > 0]
    loss_sum = sum(p for p in pnls if p < 0)

    return {
        'n_trades':      len(pnls),
        'win_rate':      round(len(wins) / len(pnls), 3) if pnls else 0,
        'mean_pnl':      round(sum(pnls) / len(pnls), 4) if pnls else 0,
        'total_pnl':     round(sum(pnls), 4),
        'profit_factor': round(
            sum(wins) / abs(loss_sum), 3
        ) if loss_sum != 0 else None,
    }


@app.get('/trades/history')
def get_trade_history():
    return _load_portfolio().get('closed_trades', [])


@app.get('/log/recent')
def get_recent_log(n: int = 50):
    return _load_trade_log(n)

@app.get('/status')
def get_status():
    import json
    from pathlib import Path
    from datetime import date

    today = date.today().isoformat()

    # Check if system ran today
    log_path = Path('logs/paper_trades.jsonl')
    ran_today = False
    last_run_date = None
    if log_path.exists():
        with open(log_path) as f:
            lines = [l for l in f.readlines() if l.strip()]
        if lines:
            last_entry = json.loads(lines[-1])
            last_run_date = last_entry.get('date')
            ran_today = last_run_date == today

    # Check portfolio state
    port_path = Path('data/paper_portfolio.json')
    n_positions = 0
    cash = 0.0
    if port_path.exists():
        with open(port_path) as f:
            state = json.load(f)
        n_positions = len(state.get('positions', []))
        cash = state.get('cash', 0.0)

    # Check drift monitor log
    drift_path = Path('logs/daily_runs.log')
    skipped_today = False
    if drift_path.exists():
        content = drift_path.read_text()
        if today in content and 'SKIP_DAY' in content:
            skipped_today = True

    return {
        'date':          today,
        'ran_today':     ran_today,
        'skipped_today': skipped_today,
        'last_run_date': last_run_date,
        'n_positions':   n_positions,
        'cash':          round(cash, 2),
        'system_ok':     ran_today and not skipped_today,
    }


@app.get('/dashboard')
def serve_dashboard():
    """Serve the dashboard HTML file."""
    dashboard_path = Path(__file__).parent.parent / 'dashboard' / 'index.html'
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail='dashboard/index.html not found')
    return FileResponse(str(dashboard_path))


@app.get('/ic-monitor')
def get_ic_monitor():
    """Return IC monitor history from logs/ic_monitor.jsonl."""
    path = Path('logs/ic_monitor.jsonl')
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


@app.get('/backfill-log')
def get_backfill_log():
    """Return last 100 lines from the backfill test log."""
    path = Path('logs/backfill_test.log')
    if not path.exists():
        return {'lines': []}
    lines = path.read_text().strip().split('\n')
    return {'lines': lines[-100:]}


@app.post('/backfill')
def run_backfill_endpoint(
    start: str = Query(..., description='Start date YYYY-MM-DD'),
    end:   str = Query(..., description='End date YYYY-MM-DD'),
):
    """Trigger a backfill test run (async — returns immediately)."""
    project_root = str(Path(__file__).parent.parent)
    try:
        subprocess.Popen(
            [sys.executable, 'scripts/test_historical_run.py',
             '--start', start, '--end', end, '--no-restore'],
            cwd=project_root,
        )
        return {'status': 'started', 'start': start, 'end': end}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


@app.get('/model-metadata')
def get_model_metadata():
    """Return Phase 3 model baseline metadata."""
    path = Path('models/registry/phase3_model_baseline.json')
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@app.post('/settings')
def save_settings(settings: dict):
    """Save dashboard settings to data/dashboard_settings.json."""
    Path('data').mkdir(exist_ok=True)
    path = Path('data/dashboard_settings.json')
    with open(path, 'w') as f:
        json.dump(settings, f, indent=2)
    return {'status': 'saved'}


@app.get('/predictions')
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

    # No ticker — return structured response with per-signal lists
    formatted = bullish + neutral + bearish
    return {
        'date':    latest_date,
        'bullish': bullish,
        'bearish': bearish,
        'neutral': neutral,
        'total':   len(formatted),
    }


@app.get('/settings')
def get_settings():
    """Return current dashboard settings."""
    path = Path('data/dashboard_settings.json')
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@app.get('/shap/{ticker}')
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

    # Find latest OPEN signal for this ticker
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

    # Build feature vector — handle both old (11-feature) and new (14-feature) records
    fv_raw = (latest.get('feature_vector_14')
              or latest.get('feature_vector_11')
              or latest.get('feature_vector')
              or {})
    fv = dict(fv_raw)
    # Fill any missing features (news/ST default to 0.0 for pre-live records)
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
        # New dashboard format — percentage integers per source family
        'reddit_attention':  reddit_pct,
        'reddit_sentiment':  0,
        'news_sentiment':    news_pct,
        'st_sentiment':      st_pct,
        'market_technical':  market_pct,
        # Verbose data for debugging
        'ticker':            ticker.upper(),
        'date':              latest.get('date'),
        'base_value':        round(base_val, 4),
        'prediction':        round(prediction, 4),
        'signal':            latest.get('signal', 'NEUTRAL'),
        'family_shap_raw':   family_shap,
        'attribution_text':  attribution_text,
        'top_features':      contributions[:8],
    })


@app.get('/signal-accuracy')
def get_signal_accuracy():
    """
    Directional accuracy per horizon (1D, 3D, 5D) for completed signals.
    Only evaluates BULLISH/BEARISH signals with >= 7 calendar days elapsed.
    """
    import yfinance as yf
    import pandas as pd
    from datetime import date

    log_path = Path('logs/paper_trades.jsonl')
    if not log_path.exists():
        return {'n_evaluated': 0, 'message': 'No signals logged yet'}

    opens = []
    with open(log_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('action') == 'OPEN':
                opens.append(r)

    results = []
    for signal in opens:
        ticker   = signal.get('ticker')
        sig_date = signal.get('date')
        sig_type = signal.get('signal', 'NEUTRAL')
        pred_1d  = signal.get('predicted_1d', 0) or 0
        pred_3d  = signal.get('predicted_3d', 0) or 0
        pred_5d  = (signal.get('predicted_return_5d', 0)
                    or signal.get('predicted_5d', 0) or 0)

        if sig_type not in ('BULLISH', 'BEARISH') or not sig_date:
            continue

        try:
            signal_dt = date.fromisoformat(sig_date)
            if (date.today() - signal_dt).days < 7:
                continue
        except Exception:
            continue

        try:
            mkt = yf.download(ticker, start=sig_date,
                              auto_adjust=True, progress=False, period='10d')
            if isinstance(mkt.columns, pd.MultiIndex):
                mkt.columns = mkt.columns.get_level_values(0)
            if len(mkt) < 2:
                continue

            entry = float(mkt['Close'].iloc[0])
            actual_1d = float(mkt['Close'].iloc[1] / entry - 1) if len(mkt) >= 2 else None
            actual_3d = float(mkt['Close'].iloc[3] / entry - 1) if len(mkt) >= 4 else None
            actual_5d = float(mkt['Close'].iloc[4] / entry - 1) if len(mkt) >= 5 else None
        except Exception:
            continue

        def correct_dir(predicted, actual):
            if actual is None or predicted is None:
                return None
            return bool((predicted > 0 and actual > 0) or (predicted < 0 and actual < 0))

        results.append({
            'ticker':     ticker,
            'date':       sig_date,
            'signal':     sig_type,
            'pred_1d':    round(pred_1d, 4),
            'pred_3d':    round(pred_3d, 4),
            'pred_5d':    round(pred_5d, 4),
            'actual_1d':  round(actual_1d, 4) if actual_1d is not None else None,
            'actual_3d':  round(actual_3d, 4) if actual_3d is not None else None,
            'actual_5d':  round(actual_5d, 4) if actual_5d is not None else None,
            'correct_1d': correct_dir(pred_1d, actual_1d),
            'correct_3d': correct_dir(pred_3d, actual_3d),
            'correct_5d': correct_dir(pred_5d, actual_5d),
        })

    if not results:
        return {
            'n_evaluated': 0,
            'message': 'Need BULLISH/BEARISH signals with 7+ days elapsed',
            '1D': '—', '3D': '—', '5D': '—',
        }

    def acc(key):
        valid = [r[key] for r in results if r[key] is not None]
        return round(sum(valid) / len(valid), 3) if valid else None

    n  = len(results)
    a1 = acc('correct_1d')
    a3 = acc('correct_3d')
    a5 = acc('correct_5d')

    if a1 is not None and a5 is not None and a1 < 0.5 and a5 > 0.53:
        interpretation = '1D accuracy low, 5D high → multi-day momentum lag (normal)'
    elif a5 is not None and a5 < 0.5:
        interpretation = '5D accuracy below 50% → model not working in live conditions'
    else:
        interpretation = 'Signal accuracy healthy — continue monitoring'

    return _sanitize({
        'n_evaluated':    n,
        'accuracy_1d':    a1,
        'accuracy_3d':    a3,
        'accuracy_5d':    a5,
        '1D': f"{a1 * 100:.1f}%" if a1 is not None else '—',
        '3D': f"{a3 * 100:.1f}%" if a3 is not None else '—',
        '5D': f"{a5 * 100:.1f}%" if a5 is not None else '—',
        'mean_pred_5d':   round(sum(r['pred_5d']           for r in results) / n, 4),
        'mean_actual_5d': round(sum((r['actual_5d'] or 0)  for r in results) / n, 4),
        'interpretation': interpretation,
        'signals':        results,
    })


@app.get('/research-findings')
def get_research_findings():
    """Return source validation results. Run validate_sources.py first."""
    path = Path('experiments/source_validation/results.json')
    if not path.exists():
        return {
            'status':  'not_run',
            'message': 'Run: python experiments/source_validation/validate_sources.py',
        }
    with open(path) as f:
        data = json.load(f)
    return _sanitize(data)