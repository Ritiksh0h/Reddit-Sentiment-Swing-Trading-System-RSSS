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
    """Return all closed trades with computed PnL dollar amounts."""
    state  = _load_portfolio()
    closed = state.get('closed_trades', [])

    enriched = []
    for t in closed:
        pnl_pct     = t.get('pnl_pct', 0)
        n_shares    = t.get('n_shares', 0)
        entry_px    = t.get('entry_price', 0)
        exit_px     = t.get('exit_price', 0)
        cost_basis  = n_shares * entry_px
        pnl_dollars = round(n_shares * (exit_px - entry_px), 2)
        is_real     = abs(pnl_pct) > 0.0001

        enriched.append({
            'ticker':      t.get('ticker'),
            'entry_date':  t.get('entry_date'),
            'exit_date':   t.get('exit_date'),
            'entry_price': entry_px,
            'exit_price':  exit_px,
            'n_shares':    n_shares,
            'cost_basis':  round(cost_basis, 2),
            'pnl_pct':     round(pnl_pct * 100, 2),
            'pnl_dollars': pnl_dollars,
            'exit_reason': t.get('exit_reason'),
            'has_real_pnl': is_real,
            'result':      ('WIN'  if pnl_pct >  0.0001
                            else ('LOSS' if pnl_pct < -0.0001
                                  else 'ZERO')),
        })

    enriched.sort(key=lambda x: x['exit_date'] or '', reverse=True)

    total_pnl   = sum(t['pnl_dollars'] for t in enriched)
    real_trades = [t for t in enriched if t['has_real_pnl']]

    return {
        'trades':            enriched,
        'n_total':           len(enriched),
        'n_real':            len(real_trades),
        'total_pnl_dollars': round(total_pnl, 2),
        'note': (f'{len(enriched) - len(real_trades)} trades have zero PnL '
                 f'(exit price bug — pre-Jun 18 runs)') if enriched else '',
    }


@app.get('/backtest')
def get_backtest(ticker: str = None, year: int = None):
    """
    Return backtest results from Experiment C.
    Optionally filter by ticker and/or year.
    Source: experiments/experiment_c/results.json
    Historical simulation (2024 data) — not live trading.
    """
    results_path = Path('experiments/experiment_c/results.json')
    if not results_path.exists():
        return {'error': 'Backtest results not found', 'path': str(results_path)}

    with open(results_path) as f:
        results = json.load(f)

    trades = list(results.get('trade_log', []))

    if ticker:
        ticker = ticker.upper()
        trades = [t for t in trades if t.get('ticker') == ticker]
    if year:
        trades = [t for t in trades if t.get('entry_date', '').startswith(str(year))]

    if not trades:
        return {
            'n_trades': 0,
            'filter':   {'ticker': ticker, 'year': year},
            'message':  'No trades match the filter',
            'trades':   [],
        }

    pnls   = [t.get('pnl', t.get('gross_pnl', 0)) for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    total  = sum(pnls)

    equity = [10000.0]
    for p in pnls:
        equity.append(round(equity[-1] + p, 2))

    max_dd = 0.0
    peak   = equity[0]
    for v in equity:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd

    formatted = sorted([{
        'ticker':      t.get('ticker'),
        'entry_date':  t.get('entry_date'),
        'exit_date':   t.get('exit_date'),
        'entry_price': t.get('entry_price'),
        'exit_price':  t.get('exit_price'),
        'pred_return': round(t.get('pred_return', 0) * 100, 2),
        'pnl':         round(t.get('pnl', t.get('gross_pnl', 0)), 2),
        'exit_reason': t.get('exit_reason', 'hold_days'),
        'result':      'WIN' if t.get('pnl', t.get('gross_pnl', 0)) > 0 else 'LOSS',
    } for t in trades], key=lambda x: x['entry_date'])

    def _safe(v):
        return None if (v is None or (isinstance(v, float) and math.isnan(v))) else v

    return _sanitize({
        'source':       'Experiment C — Historical Simulation 2024',
        'filter':       {'ticker': ticker, 'year': year},
        'n_trades':     len(trades),
        'win_rate':     round(len(wins) / len(pnls), 3) if pnls else 0,
        'total_pnl':    round(total, 2),
        'mean_pnl':     round(total / len(pnls), 2) if pnls else 0,
        'max_drawdown': round(max_dd * 100, 2),
        'profit_factor': _safe(
            round(sum(wins) / abs(sum(losses)), 3) if losses else None
        ),
        'equity_curve': equity,
        'trades':       formatted,
        'full_stats': {
            'ic_test':        results.get('ic_test'),
            'sharpe_ratio':   results.get('sharpe_ratio'),
            'total_return':   results.get('total_return'),
            'spy_return':     results.get('spy_return'),
            'alpha':          results.get('alpha'),
            'n_trades_total': len(results.get('trade_log', [])),
        } if not ticker and not year else None,
    })


@app.get('/backtest-full')
def get_backtest_full(year: int = None, month: int = None):
    """
    Walk-forward backtest results for 2024-2025 out-of-sample period.
    Run scripts/run_backtest.py first to generate backtest_results.json.
    Optionally filter by year (2024/2025) and/or month (1-12).
    """
    path = Path('experiments/backtest_results.json')
    if not path.exists():
        return {
            'error':   'Backtest results not found',
            'message': 'Run: python scripts/run_backtest.py',
        }

    with open(path) as f:
        data = json.load(f)

    trades = list(data.get('trades', []))

    if year:
        trades = [t for t in trades if t.get('entry_date', '').startswith(str(year))]
    if month and year:
        prefix = f'{year}-{month:02d}'
        trades = [t for t in trades if t.get('entry_date', '').startswith(prefix)]

    if not trades:
        return {
            'n_trades': 0,
            'filter':   {'year': year, 'month': month},
            'message':  'No trades match the filter',
            'trades':   [],
        }

    pnls   = [t.get('pnl_dollars', 0) for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    total  = sum(pnls)
    wr     = round(len(wins) / len(pnls), 3) if pnls else 0

    month_key = (f'{year}-{month:02d}' if (year and month) else
                 str(year) if year else None)
    monthly_return = None
    if month_key and len(month_key) == 7:
        monthly_return = data.get('monthly_returns', {}).get(month_key)
    elif year:
        yr_data = data.get('annual_breakdown', {}).get(str(year), {})
        monthly_return = yr_data.get('return_pct')

    pf = round(sum(wins) / abs(sum(losses)), 3) if losses else None

    return _sanitize({
        'source':         'Walk-Forward Backtest 2024-2025 (SIMULATION)',
        'filter':         {'year': year, 'month': month},
        'n_trades':       len(trades),
        'win_rate':       wr,
        'total_pnl':      round(total, 2),
        'mean_pnl':       round(total / len(pnls), 2) if pnls else 0,
        'period_return':  monthly_return,
        'profit_factor':  pf,
        'trades':         trades,
        'full_stats': {
            'total_return_pct':   data.get('total_return_pct'),
            'spy_return_pct':     data.get('spy_return_pct'),
            'alpha':              data.get('alpha'),
            'sharpe_ratio':       data.get('sharpe_ratio'),
            'sortino_ratio':      data.get('sortino_ratio'),
            'max_drawdown_pct':   data.get('max_drawdown_pct'),
            'n_trades_total':     data.get('n_trades'),
            'annual_2024':        data.get('annual_breakdown', {}).get('2024'),
            'annual_2025':        data.get('annual_breakdown', {}).get('2025'),
        } if not year and not month else None,
    })


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
    Per-horizon directional accuracy (1D, 3D, 5D) from closed paper trades.
    1D/3D actual prices fetched via yfinance; 5D uses recorded pnl_pct.
    Signal lookup falls back to logs/paper_trades.jsonl OPEN records.
    Results cached in data/processed/signal_accuracy_cache.json.
    """
    import yfinance as yf
    import pandas as pd
    from datetime import datetime, timedelta, date as _date

    CACHE_PATH = Path('data/processed/signal_accuracy_cache.json')

    state  = _load_portfolio()
    closed = state.get('closed_trades', [])
    real   = [t for t in closed if abs(t.get('pnl_pct', 0)) > 0.0001]

    if not real:
        return {
            'n_evaluated': len(closed), 'n_real': 0,
            'message': 'No trades with real PnL yet',
            '1D': '—', '3D': '—', '5D': '—',
            'interpretation': 'Accumulating signals — need 10+ closed trades',
        }

    # Build signal lookup from OPEN records in the trade log
    signal_lookup: dict = {}
    log_path = Path('logs/paper_trades.jsonl')
    if log_path.exists():
        for line in log_path.read_text().strip().split('\n'):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if r.get('action') == 'OPEN':
                    signal_lookup[f"{r['ticker']}_{r['date']}"] = r.get('signal', 'BULLISH')
            except Exception:
                pass

    # Load price cache (avoids re-fetching on every request)
    cache: dict = {}
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH) as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    today      = _date.today()
    needs_save = False

    for trade in real:
        ticker     = trade.get('ticker', '')
        entry_date = trade.get('entry_date', '')
        if not ticker or not entry_date:
            continue
        cache_key = f'{ticker}_{entry_date}'
        if cache_key in cache:
            continue
        if (today - _date.fromisoformat(entry_date)).days < 7:
            continue
        try:
            hist = yf.download(
                ticker,
                start=entry_date,
                end=(datetime.fromisoformat(entry_date) + timedelta(days=20)).strftime('%Y-%m-%d'),
                auto_adjust=True,
                progress=False,
            )
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            if len(hist) < 2:
                continue
            c0  = float(hist['Close'].iloc[0])
            r1d = (float(hist['Close'].iloc[1]) - c0) / c0 if len(hist) > 1 else None
            r3d = (float(hist['Close'].iloc[3]) - c0) / c0 if len(hist) > 3 else None
            sig = signal_lookup.get(cache_key, 'BULLISH')
            cache[cache_key] = {
                'ticker':     ticker,
                'entry_date': entry_date,
                'signal':     sig,
                'r1d':        round(r1d, 6) if r1d is not None else None,
                'r3d':        round(r3d, 6) if r3d is not None else None,
            }
            needs_save = True
        except Exception as e:
            logger.warning(f'signal_accuracy_fetch {ticker} {entry_date}: {e}')

    if needs_save:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, 'w') as f:
            json.dump(cache, f, indent=2)

    def _accuracy(pairs):
        evaluated = [(r, s) for r, s in pairs if r is not None]
        if not evaluated:
            return None, 0
        correct = sum(
            1 for r, s in evaluated
            if (s == 'BEARISH' and r < 0) or (s != 'BEARISH' and r > 0)
        )
        return round(correct / len(evaluated), 3), len(evaluated)

    pairs_1d, pairs_3d, pairs_5d = [], [], []
    for trade in real:
        key   = f'{trade["ticker"]}_{trade["entry_date"]}'
        entry = cache.get(key, {})
        sig   = entry.get('signal') or signal_lookup.get(key, 'BULLISH')
        pairs_1d.append((entry.get('r1d'), sig))
        pairs_3d.append((entry.get('r3d'), sig))
        pairs_5d.append((trade.get('pnl_pct'), sig))

    acc_1d, n_1d = _accuracy(pairs_1d)
    acc_3d, n_3d = _accuracy(pairs_3d)
    acc_5d, n_5d = _accuracy(pairs_5d)

    def _fmt(acc):
        return '—' if acc is None else f'{round(acc * 100, 1)}%'

    pnls     = [t.get('pnl_pct', 0) for t in real]
    mean_pnl = round(sum(pnls) / len(pnls) * 100, 2) if pnls else 0
    wins     = sum(1 for p in pnls if p > 0)

    lag = (acc_1d is not None and acc_5d is not None and acc_5d > acc_1d + 0.05)
    interpretation = (
        f'{wins}/{len(real)} trades profitable. Mean PnL={mean_pnl:+.2f}%. '
        + ('1D low + 5D high = multi-day lag in signal (expected).'
           if lag else 'Too few trades for conclusions — need 30+.')
    )

    return _sanitize({
        'n_evaluated':    len(real),
        'n_zero_pnl':     len(closed) - len(real),
        'win_rate':       acc_5d or 0.0,
        'mean_pnl_pct':   mean_pnl,
        '1D':             _fmt(acc_1d),
        '3D':             _fmt(acc_3d),
        '5D':             _fmt(acc_5d),
        'n_evaluated_1d': n_1d,
        'n_evaluated_3d': n_3d,
        'n_evaluated_5d': n_5d,
        'interpretation': interpretation,
        'trades':         real,
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