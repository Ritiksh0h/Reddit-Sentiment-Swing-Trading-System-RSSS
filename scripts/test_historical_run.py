"""
Historical backfill test for Phase 4 pipeline.
Uses real market data + synthetic Reddit counts to simulate past trading days.

Usage:
    python scripts/test_historical_run.py                    # last 30 days
    python scripts/test_historical_run.py --days 10          # last 10 days
    python scripts/test_historical_run.py --start 2026-05-01 --end 2026-05-31
    python scripts/test_historical_run.py --days 30 --no-restore  # keep results
"""
import argparse
import json
import logging
import random
import shutil
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, '.')

Path('logs').mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/backfill_test.log'),
    ]
)
logger = logging.getLogger('backfill_test')

# ── Tickers ────────────────────────────────────────────────────────────────
HIGH_ACTIVITY = {'TSLA', 'NVDA', 'AMD', 'AAPL', 'GME', 'PLTR', 'COIN'}
MED_ACTIVITY  = {'META', 'MSFT', 'AMZN', 'NFLX', 'UBER', 'SNAP', 'PYPL'}
ALL_TICKERS   = list(HIGH_ACTIVITY | MED_ACTIVITY | {
    'AMC', 'NIO', 'BA', 'BB', 'NOK', 'SPCE', 'BABA', 'DKNG',
})


# ── Trading day calendar ────────────────────────────────────────────────────
def get_trading_days(start: date, end: date) -> list:
    """Return weekdays (Mon-Fri) between start and end inclusive."""
    days, current = [], start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


# ── Synthetic Reddit counts ────────────────────────────────────────────────
def build_synthetic_reddit_counts(trading_date: date) -> dict:
    """
    Build realistic synthetic reddit_counts for a historical date.
    Uses a seeded RNG so results are deterministic per date.

    High-activity tickers: 25-120 posts/day
    Med-activity tickers:  8-45 posts/day
    Low-activity tickers:  2-20 posts/day
    """
    rng = random.Random(int(trading_date.strftime('%Y%m%d')))
    counts = {}
    for ticker in ALL_TICKERS:
        if ticker in HIGH_ACTIVITY:
            post_count = rng.randint(25, 120)
        elif ticker in MED_ACTIVITY:
            post_count = rng.randint(8, 45)
        else:
            post_count = rng.randint(2, 20)

        counts[ticker] = {
            'post_count_1d':     post_count,
            'mention_growth_1d': round(rng.uniform(0.7, 1.8), 3),
            'mention_growth_7d': round(rng.uniform(0.5, 2.2), 3),
        }
    return counts


# ── State backup / restore ─────────────────────────────────────────────────
BACKUP_DIR = Path('data/backfill_backup')

STATE_FILES = [
    ('data/paper_portfolio.json',    BACKUP_DIR / 'paper_portfolio.json'),
    ('logs/paper_trades.jsonl',      BACKUP_DIR / 'paper_trades.jsonl'),
    ('data/paper_performance.jsonl', BACKUP_DIR / 'paper_performance.jsonl'),
    ('data/mention_history.json',    BACKUP_DIR / 'mention_history.json'),
]


def backup_state():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for src_str, dst in STATE_FILES:
        src = Path(src_str)
        if src.exists():
            shutil.copy(src, dst)
            logger.info(f'Backed up {src_str}')
        if src.exists():
            src.unlink()
    logger.info('State cleared — fresh $10,000 portfolio')


def restore_state():
    for src_str, backup in STATE_FILES:
        if backup.exists():
            Path(src_str).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(backup, src_str)
            logger.info(f'Restored {src_str}')
    logger.info('Original state restored')


# ── Main backfill loop ─────────────────────────────────────────────────────
def run_backfill(start: date, end: date) -> list:
    from scripts.daily_run import run

    trading_days = get_trading_days(start, end)
    logger.info(f'Simulating {len(trading_days)} trading days: {start} → {end}')

    results = []
    for trading_date in trading_days:
        date_str      = trading_date.isoformat()
        reddit_counts = build_synthetic_reddit_counts(trading_date)

        try:
            summary = run(reddit_counts=reddit_counts, today=date_str)
            results.append({
                'date':    date_str,
                'actions': summary['actions'],
                'skipped': summary.get('skipped', False),
                'reason':  summary.get('reason'),
            })
            n_open  = len([a for a in summary['actions'] if 'OPEN'  in a])
            n_close = len([a for a in summary['actions'] if 'CLOSE' in a])
            skip    = f' SKIPPED({summary.get("reason")})' if summary.get('skipped') else ''
            logger.info(f'{date_str}: +{n_open} opened  -{n_close} closed{skip}')

        except Exception as e:
            logger.error(f'{date_str}: ERROR — {e}')
            results.append({'date': date_str, 'actions': [],
                            'skipped': True, 'reason': f'error: {e}'})
    return results


# ── Results analysis ───────────────────────────────────────────────────────
def analyze_results(results: list) -> None:
    total   = len(results)
    skipped = sum(1 for r in results if r['skipped'])
    active  = total - skipped
    opens   = sum(len([a for a in r['actions'] if 'OPEN'  in a]) for r in results)
    closes  = sum(len([a for a in r['actions'] if 'CLOSE' in a]) for r in results)

    print('\n' + '=' * 60)
    print('BACKFILL TEST RESULTS')
    print('=' * 60)
    print(f'Days simulated:    {total}')
    print(f'Active days:       {active}')
    print(f'Skipped days:      {skipped}')
    print(f'Positions opened:  {opens}')
    print(f'Positions closed:  {closes}')
    print()

    # Portfolio state
    port_path = Path('data/paper_portfolio.json')
    if port_path.exists():
        with open(port_path) as f:
            state = json.load(f)
        cash   = state.get('cash', 10000)
        pos    = state.get('positions', [])
        closed = state.get('closed_trades', [])
        print(f'Final cash:        ${cash:,.2f}')
        print(f'Open positions:    {len(pos)}')
        print(f'Closed trades:     {len(closed)}')
        if closed:
            pnls = [t.get('pnl_pct', 0) for t in closed]
            wins = [p for p in pnls if p > 0]
            print(f'Win rate:          {len(wins)/len(pnls):.1%}')
            print(f'Mean PnL/trade:    {sum(pnls)/len(pnls):+.2%}')

    # Log line count
    log_path = Path('logs/paper_trades.jsonl')
    if log_path.exists():
        n = sum(1 for line in open(log_path) if line.strip())
        print(f'Execution log:     {n} entries')

    # Day-by-day activity
    print()
    print('Day-by-day activity (trades and skips only):')
    for r in results:
        n_opens  = len([a for a in r['actions'] if 'OPEN'  in a])
        n_closes = len([a for a in r['actions'] if 'CLOSE' in a])
        if n_opens > 0 or n_closes > 0 or r['skipped']:
            skip = f'  ← SKIP({r["reason"]})' if r['skipped'] else ''
            print(f'  {r["date"]}: +{n_opens} opened  -{n_closes} closed{skip}')
    print('=' * 60)


# ── Entry point ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Test Phase 4 pipeline against historical dates'
    )
    parser.add_argument('--days',       type=int, default=30)
    parser.add_argument('--start',      type=str, default=None)
    parser.add_argument('--end',        type=str, default=None)
    parser.add_argument('--no-restore', action='store_true',
                        help='Keep backfill results instead of restoring original state')
    args = parser.parse_args()

    end_date   = date.fromisoformat(args.end)   if args.end   \
                 else date.today() - timedelta(days=1)
    start_date = date.fromisoformat(args.start) if args.start \
                 else end_date - timedelta(days=args.days)

    logger.info(f'Backfill test: {start_date} → {end_date}')
    logger.warning('Backing up current state before test...')

    backup_state()
    try:
        results = run_backfill(start=start_date, end=end_date)
        analyze_results(results)
    finally:
        if not args.no_restore:
            restore_state()
            logger.info('State restored. Run with --no-restore to keep results.')
        else:
            logger.info('--no-restore set. Backfill results kept in data/ and logs/.')


if __name__ == '__main__':
    main()
