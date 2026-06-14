"""
FastAPI endpoints for paper trading monitoring.
Run: uvicorn api.main:app --reload --port 8000
"""
import json
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

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
    return _load_portfolio()


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


@app.get('/settings')
def get_settings():
    """Return current dashboard settings."""
    path = Path('data/dashboard_settings.json')
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)