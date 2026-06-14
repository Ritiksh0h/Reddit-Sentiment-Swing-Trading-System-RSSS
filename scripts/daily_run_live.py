"""
Live daily run — fetches real Reddit data then runs portfolio engine.
Called by scheduler at 08:30 ET each weekday.

Usage:
    python scripts/daily_run_live.py
    python scripts/daily_run_live.py --dry-run  (logs signals but no trades)
    python scripts/daily_run_live.py --date 2024-03-15  (override date for testing)
"""
import argparse
import json
import logging
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, '.')

Path('logs').mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/daily_runs.log'),
    ]
)
logger = logging.getLogger('daily_run_live')


def ensure_api_running(port: int = 8000) -> None:
    """
    Start uvicorn API server if not already running on the given port.
    Called at the start of every daily run so the status endpoint
    is always available after the pipeline executes.

    Uses socket check to avoid launching a duplicate process.
    Uvicorn runs in background — does not block the pipeline.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        already_running = s.connect_ex(('localhost', port)) == 0

    if already_running:
        logger.info(f'api_server_already_running port={port}')
        return

    project_root = str(Path(__file__).parent.parent)
    subprocess.Popen(
        [
            sys.executable, '-m', 'uvicorn',
            'api.main:app',
            '--port', str(port),
            '--log-level', 'warning',
        ],
        cwd=project_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    logger.info(f'api_server_started port={port}')


def main():
    ensure_api_running()

    parser = argparse.ArgumentParser(description='RSSS live daily run')
    parser.add_argument('--dry-run', action='store_true',
                        help='Log signals but do not execute trades')
    parser.add_argument('--date', type=str, default=None,
                        help='Override date YYYY-MM-DD for testing')
    args = parser.parse_args()

    today = args.date or datetime.now(timezone.utc).strftime('%Y-%m-%d')
    logger.info(f'=== RSSS Daily Run Starting date={today} ===')

    # ── Step 1: Fetch live Reddit data ────────────────────────────────────
    logger.info('Fetching live Reddit data...')
    try:
        from data.reddit_live_fetcher import fetch_recent_posts, compute_mention_growth

        reddit_counts_raw = fetch_recent_posts(hours_back=24)

        if not reddit_counts_raw:
            logger.warning('No Reddit data fetched — api_anomaly handler will trigger')
            reddit_counts = {}
        else:
            reddit_counts = {}
            for ticker, data in reddit_counts_raw.items():
                growth = compute_mention_growth(
                    ticker=ticker,
                    current_count=data['post_count_1d'],
                )
                reddit_counts[ticker] = {
                    'post_count_1d':     data['post_count_1d'],
                    'mention_growth_1d': growth['mention_growth_1d'],
                    'mention_growth_7d': growth['mention_growth_7d'],
                }

            logger.info(f'Reddit data ready: {len(reddit_counts)} tickers')

    except Exception as e:
        logger.error(f'Reddit fetch failed: {e}')
        reddit_counts = {}

    # ── Step 2a: Dry run — signals only ───────────────────────────────────
    if args.dry_run:
        logger.info('DRY RUN MODE — signals logged, no trades executed')
        try:
            from portfolio.signal_generator import generate_signals, load_model
            model   = load_model()
            signals = generate_signals(reddit_counts, model, today)
            logger.info(f'Dry run: {len(signals)} qualifying signals')
            for s in signals[:5]:
                logger.info(f'  {s.ticker}: pred={s.predicted_return:.3f} '
                            f'price={s.price:.2f} posts={s.post_count_1d}')
        except Exception as e:
            logger.error(f'Dry run signal generation failed: {e}')
        return

    # ── Step 2b: Full run ─────────────────────────────────────────────────
    from scripts.daily_run import run
    summary = run(reddit_counts=reddit_counts, today=today)

    logger.info('=== Daily Run Complete ===')
    logger.info(f'Actions: {summary["actions"]}')
    if summary.get('skipped'):
        logger.warning(f'Run skipped — reason: {summary["reason"]}')

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
