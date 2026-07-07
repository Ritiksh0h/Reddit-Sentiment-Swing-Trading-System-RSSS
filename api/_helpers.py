"""
RSSS API — shared helpers used across route modules.
"""
import json
import math
from pathlib import Path


def _sanitize(obj):
    """Recursively replace NaN/Inf floats with None for safe JSON serialization."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(i) for i in obj]
    return obj


def _load_portfolio() -> dict:
    """DB-first (Railway has no local file); local JSON fallback for dev."""
    try:
        from api.db import load_portfolio_state
        remote = load_portfolio_state()
        if remote:
            return remote
    except Exception:
        pass

    path = Path('data/live/paper_portfolio.json')
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _load_trade_log(last_n: int = 50) -> list:
    """
    Read trade log. DB-first when DB_URL is set (Railway), JSONL fallback for local dev.
    If DB has no rows yet (fresh Railway deploy), tries JSONL as a backstop.
    """
    try:
        from api.db import load_trades
        db_records = load_trades(last_n)
        if db_records:
            return db_records
    except Exception:
        pass

    path = Path('logs/paper_trades.jsonl')
    if not path.exists():
        return []
    lines = [ln for ln in path.read_text().strip().split('\n') if ln]
    records = [json.loads(ln) for ln in lines]
    return records[-last_n:]
