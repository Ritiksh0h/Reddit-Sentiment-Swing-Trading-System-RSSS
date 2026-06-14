"""
FastAPI endpoints for paper trading monitoring.
Run: uvicorn api.main:app --reload --port 8000
"""
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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
