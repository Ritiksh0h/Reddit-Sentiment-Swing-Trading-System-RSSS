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
    path = Path('data/live/paper_portfolio.json')
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _load_trade_log(last_n: int = 50) -> list:
    path = Path('logs/paper_trades.jsonl')
    if not path.exists():
        return []
    lines = [ln for ln in path.read_text().strip().split('\n') if ln]
    records = [json.loads(ln) for ln in lines]
    return records[-last_n:]
