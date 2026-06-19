"""
RSSS API — health and settings routes.
"""
import json
from datetime import date
from pathlib import Path

from fastapi import APIRouter

router = APIRouter()


@router.get('/health')
def health():
    return {'status': 'ok', 'version': '3.0'}


@router.get('/status')
def get_status():
    today = date.today().isoformat()

    log_path = Path('logs/paper_trades.jsonl')
    ran_today = False
    last_run_date = None
    if log_path.exists():
        with open(log_path) as f:
            lines = [ln for ln in f.readlines() if ln.strip()]
        if lines:
            last_entry = json.loads(lines[-1])
            last_run_date = last_entry.get('date')
            ran_today = last_run_date == today

    port_path = Path('data/live/paper_portfolio.json')
    n_positions = 0
    cash = 0.0
    if port_path.exists():
        with open(port_path) as f:
            state = json.load(f)
        n_positions = len(state.get('positions', []))
        cash = state.get('cash', 0.0)

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


@router.get('/settings')
def get_settings():
    """Return current dashboard settings."""
    path = Path('data/dashboard_settings.json')
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@router.post('/settings')
def save_settings(settings: dict):
    """Save dashboard settings to data/dashboard_settings.json."""
    Path('data').mkdir(exist_ok=True)
    path = Path('data/dashboard_settings.json')
    with open(path, 'w') as f:
        json.dump(settings, f, indent=2)
    return {'status': 'saved'}
