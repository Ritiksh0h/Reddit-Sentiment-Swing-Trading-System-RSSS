"""
RSSS API — health and settings routes.
"""
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path

from fastapi import APIRouter

router = APIRouter()


@router.get('/health')
def health():
    return {'status': 'ok', 'version': '3.0'}


@router.get('/status')
def get_status():
    today = date.today().isoformat()

    log_path = Path('logs/daily_runs.log')
    ran_today = False
    last_run_date = None
    skipped_today = False
    if log_path.exists():
        all_lines = log_path.read_text().splitlines()
        for line in reversed(all_lines):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                last_run_date = entry.get('date')
                if last_run_date == today:
                    ran_today = True
                break
            except Exception:
                if line.startswith(today):
                    ran_today = True
                    last_run_date = today
                break
        if any(today in ln and 'SKIP_DAY' in ln for ln in all_lines):
            skipped_today = True

    port_path = Path('data/live/paper_portfolio.json')
    n_positions = 0
    cash = 0.0
    if port_path.exists():
        with open(port_path) as f:
            state = json.load(f)
        n_positions = len(state.get('positions', []))
        cash = state.get('cash', 0.0)

    # Latest SPY daily return from paper_performance.jsonl
    spy_return_today = None
    perf_path = Path('data/live/paper_performance.jsonl')
    if perf_path.exists():
        try:
            lines = [ln for ln in perf_path.read_text().splitlines() if ln.strip()]
            if lines:
                last_snap = json.loads(lines[-1])
                spy_return_today = last_snap.get('spy_return_today')
        except Exception:
            pass

    return {
        'date':             today,
        'ran_today':        ran_today,
        'skipped_today':    skipped_today,
        'last_run_date':    last_run_date,
        'n_positions':      n_positions,
        'cash':             round(cash, 2),
        'system_ok':        ran_today,
        'spy_return_today': spy_return_today,
    }


@router.get('/settings')
def get_settings():
    """Return current dashboard settings."""
    path = Path('data/dashboard_settings.json')
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@router.get('/daily-report')
def get_daily_report():
    """
    Today's daily run report.
    Source priority: MongoDB daily_run_reports → log file parsing → paper_trades.jsonl.
    """
    today = date.today().isoformat()

    # ── Try MongoDB first (Railway / production) ─────────────────────────────
    try:
        from api.db import get_mongo_db
        mdb = get_mongo_db()
        if mdb is not None:
            doc = mdb['daily_run_reports'].find_one({'date': today}, {'_id': 0})
            if doc:
                # Normalize to the same response shape as the log-parser below
                reddit_info = doc.get('reddit', {})
                signals     = doc.get('signals', [])
                actions     = doc.get('actions', [])
                regime      = doc.get('regime', {})
                passed = [{'ticker': s.get('ticker'), 'posts': s.get('post_count_1d', 0)}
                          for s in signals if s.get('passed_density')]
                return {
                    'date':   today,
                    'source': 'mongodb',
                    'reddit': {
                        'total_posts':         reddit_info.get('total_posts', 0),
                        'tickers_found':       len(reddit_info.get('tickers', {})),
                        'subreddit_breakdown': {},
                    },
                    'density_gate': {'passed': passed, 'failed': []},
                    'ma_filter':    {'blocked': []},
                    'signals':  [{'ticker': s.get('ticker'), 'signal': s.get('signal'),
                                  'pred_1d': s.get('predicted_1d'), 'pred_3d': s.get('predicted_3d'),
                                  'pred_5d': s.get('predicted_5d') or s.get('predicted_return_5d'),
                                  'confidence': s.get('confidence'), 'post_count': s.get('post_count_1d')}
                                 for s in signals],
                    'actions':  actions,
                    'regime':   {'label': regime.get('label', 'UNKNOWN'), 'multiplier': regime.get('multiplier')},
                    'vix_percentile': None,
                }
    except Exception:
        pass

    # ── Parse daily_runs.log for today's lines ──────────────────────────────
    log_path = Path('logs/daily_runs.log')
    today_lines: list[str] = []
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            if today in line:
                today_lines.append(line)

    # Reddit totals
    total_posts = 0
    tickers_found = 0
    subreddit_breakdown: dict[str, int] = {}

    # Density gate results
    density_passed: list[dict] = []
    density_failed: list[dict] = []

    # MA filter blocks
    ma_blocked: list[dict] = []

    # Regime / VIX
    regime_label = None
    regime_multiplier = None
    vix_percentile = None

    for line in today_lines:
        # "[PASS] MU     posts=40"
        m = re.search(r'\[PASS\]\s+(\w+)\s+posts=(\d+)', line)
        if m:
            density_passed.append({'ticker': m.group(1), 'posts': int(m.group(2))})
            continue

        # "[FAIL] RKLB   posts=4"
        m = re.search(r'\[FAIL\]\s+(\w+)\s+posts=(\d+)', line)
        if m:
            density_failed.append({'ticker': m.group(1), 'posts': int(m.group(2))})
            continue

        # "ma_filter_skip ticker=NVDA price=195.74 ma20=208.79"
        m = re.search(r'ma_filter_skip ticker=(\w+)\s+price=([\d.]+)\s+ma20=([\d.]+)', line)
        if m:
            price = float(m.group(2))
            ma20  = float(m.group(3))
            gap   = round((price - ma20) / ma20 * 100, 2)
            ma_blocked.append({'ticker': m.group(1), 'price': price, 'ma20': ma20, 'gap_pct': gap})
            continue

        # "Reddit data ready: N tickers"
        m = re.search(r'Reddit data ready: (\d+) tickers', line)
        if m:
            tickers_found = int(m.group(1))
            continue

        # subreddit post counts e.g. "wallstreetbets=142 stocks=88 ..."
        for sub in ['wallstreetbets', 'stocks', 'investing', 'options', 'SecurityAnalysis']:
            m = re.search(rf'{sub}=(\d+)', line)
            if m:
                subreddit_breakdown[sub] = int(m.group(1))

        # total posts: "post_count_total=N"
        m = re.search(r'post_count_total=(\d+)', line)
        if m:
            total_posts = int(m.group(1))

        # regime: "regime=POSITIVE multiplier=1.0"
        m = re.search(r'regime[=_](\w+)', line, re.IGNORECASE)
        if m and regime_label is None:
            regime_label = m.group(1).upper()
        m = re.search(r'multiplier=([\d.]+)', line)
        if m and regime_multiplier is None:
            regime_multiplier = float(m.group(1))

        # vix percentile
        m = re.search(r'vix_pct[a-z_]*=([\d.]+)', line, re.IGNORECASE)
        if m and vix_percentile is None:
            vix_percentile = float(m.group(1))

    # Derive total_posts from subreddit sum if not found directly
    if total_posts == 0 and subreddit_breakdown:
        total_posts = sum(subreddit_breakdown.values())

    # ── Parse paper_trades.jsonl for today's signals and actions ───────────
    # Always parse this file — it is the primary source on Railway where
    # daily_runs.log may not exist.  Also acts as a fallback when the log
    # file had no entries for today (e.g. first run of the day).
    trades_path = Path('logs/paper_trades.jsonl')
    signals: list[dict] = []
    actions: list[dict] = []
    trades_tickers_seen: set[str] = set()

    if trades_path.exists():
        for line in trades_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get('date') != today:
                continue

            action = rec.get('action', '')
            ticker = rec.get('ticker', '')

            if action == 'OPEN':
                signals.append({
                    'ticker':     ticker,
                    'signal':     rec.get('signal', 'BULLISH'),
                    'pred_1d':    rec.get('predicted_1d'),
                    'pred_3d':    rec.get('predicted_3d'),
                    'pred_5d':    rec.get('predicted_5d'),
                    'confidence': rec.get('confidence'),
                    'post_count': rec.get('post_count_1d'),
                })
                trades_tickers_seen.add(ticker)
                # Only BUY actions for BULLISH signals
                if rec.get('signal') == 'BULLISH':
                    actions.append({
                        'action': 'BUY',
                        'ticker': ticker,
                        'price':  rec.get('entry_price'),
                        'shares': rec.get('shares'),
                    })
            elif action in ('CLOSE', 'EXIT'):
                actions.append({
                    'action':  action,
                    'ticker':  ticker,
                    'price':   rec.get('exit_price'),
                    'pnl_pct': rec.get('pnl_pct'),
                })

    # When the log file was absent or had no today-entries, supplement
    # density_gate.passed from paper_trades so the Railway dashboard
    # can show qualifying tickers even without daily_runs.log.
    if not density_passed and trades_tickers_seen:
        for sig in signals:
            pc = sig.get('post_count') or 0
            density_passed.append({'ticker': sig['ticker'], 'posts': pc})

    # Similarly fill tickers_found from trades if log didn't provide it
    if not tickers_found and trades_tickers_seen:
        tickers_found = len(trades_tickers_seen)

    return {
        'date':          today,
        'reddit': {
            'total_posts':         total_posts,
            'tickers_found':       tickers_found or len(density_passed) + len(density_failed),
            'subreddit_breakdown': subreddit_breakdown,
        },
        'density_gate': {
            'passed': density_passed,
            'failed': density_failed,
        },
        'ma_filter': {
            'blocked': ma_blocked,
        },
        'signals':  signals,
        'actions':  actions,
        'regime': {
            'label':      regime_label or 'UNKNOWN',
            'multiplier': regime_multiplier,
        },
        'vix_percentile': vix_percentile,
    }


@router.post('/settings')
def save_settings(settings: dict):
    """Save dashboard settings to data/dashboard_settings.json."""
    Path('data').mkdir(exist_ok=True)
    path = Path('data/dashboard_settings.json')
    with open(path, 'w') as f:
        json.dump(settings, f, indent=2)
    return {'status': 'saved'}
