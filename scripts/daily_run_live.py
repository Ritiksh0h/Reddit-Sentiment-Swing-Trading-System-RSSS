"""
Live daily run — fetches real Reddit data then runs portfolio engine.
Called by scheduler at 08:30 ET each weekday.

Usage:
    python scripts/daily_run_live.py
    python scripts/daily_run_live.py --dry-run  (logs signals but no trades)
    python scripts/daily_run_live.py --date 2024-03-15  (override date for testing)
"""
import os
# Load .env first so HF_TOKEN is available before transformers import.
from dotenv import load_dotenv
load_dotenv()

# Prevent OpenMP/OMP thread conflicts between XGBoost and uvicorn
# on Python 3.13 + macOS — must be set before any xgboost import.
os.environ['OMP_NUM_THREADS']      = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

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
    """
    Run one full cycle of the RSSS live pipeline.

    Flow:
        1. Fetch Reddit post counts + mention growth (Arctic Shift API)
        2. Fetch yfinance news + FinBERT sentiment scores
        3. Fetch StockTwits bullish/bearish tags
        4. Merge all three sources into a unified ticker dict
        5. Run scripts/daily_run.py → density gate → XGBoost → trades
        6. Append today's feature vectors to the live feature store
        7. Fill any pending t+5 price targets from prior days

    --dry-run logs signals only; no trades are written to paper_trades.jsonl.
    """
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

    # ── Step 1b: Fetch yfinance news sentiment ────────────────────────────
    logger.info('Fetching yfinance news...')
    try:
        from data.news_fetcher import fetch_yfinance_news
        news_data = fetch_yfinance_news()
        logger.info(f'yfinance news ready: {len(news_data)} tickers')
    except Exception as e:
        logger.warning(f'yfinance news fetch failed: {e} — continuing without news')
        news_data = {}

    # ── Step 1c: Fetch StockTwits sentiment ───────────────────────────────
    logger.info('Fetching StockTwits...')
    try:
        from data.stocktwits_fetcher import fetch_stocktwits
        st_data = fetch_stocktwits()
        logger.info(f'StockTwits ready: {len(st_data)} tickers')
    except Exception as e:
        logger.warning(f'StockTwits fetch failed: {e} — continuing without StockTwits')
        st_data = {}

    # ── Step 1d: Merge all sources into unified reddit_counts ─────────────
    # reddit_counts is the dict passed to daily_run.run()
    # Add news and StockTwits fields to each ticker's entry
    # Graceful: missing sources default to neutral values (0.0)
    for ticker in list(reddit_counts.keys()):
        news = news_data.get(ticker, {})
        reddit_counts[ticker]['news_count_1d']     = news.get('news_count_1d', 0)
        reddit_counts[ticker]['news_sentiment_1d'] = news.get('news_sentiment_1d', 0.0)

        st = st_data.get(ticker, {})
        reddit_counts[ticker]['st_count_1d']     = st.get('st_count_1d', 0)
        reddit_counts[ticker]['st_sentiment_1d'] = st.get('st_sentiment_1d', 0.0)
        reddit_counts[ticker]['st_bull_pct']     = st.get('st_bull_pct', 0.5)

    # Also add tickers found ONLY in news or StockTwits (not in Reddit)
    all_tickers = set(news_data.keys()) | set(st_data.keys())
    for ticker in all_tickers:
        if ticker not in reddit_counts:
            news = news_data.get(ticker, {})
            st   = st_data.get(ticker, {})
            reddit_counts[ticker] = {
                'post_count_1d':     0,
                'mention_growth_1d': 1.0,
                'mention_growth_7d': 1.0,
                'news_count_1d':     news.get('news_count_1d', 0),
                'news_sentiment_1d': news.get('news_sentiment_1d', 0.0),
                'st_count_1d':       st.get('st_count_1d', 0),
                'st_sentiment_1d':   st.get('st_sentiment_1d', 0.0),
                'st_bull_pct':       st.get('st_bull_pct', 0.5),
            }

    logger.info(f'Combined data: {len(reddit_counts)} tickers across all sources')

    # ── Step 2a: Dry run — signals only ───────────────────────────────────
    if args.dry_run:
        logger.info('DRY RUN MODE — signals logged, no trades executed')
        try:
            from portfolio.signal_generator import generate_signals, load_models
            models  = load_models()
            signals = generate_signals(
                reddit_counts=reddit_counts,
                models=models,
                today=today,
                news_data=news_data,
                stocktwits_data=st_data,
            )
            logger.info(f'Dry run: {len(signals)} qualifying signals')
            for s in signals[:5]:
                logger.info(
                    f'  {s.signal:<8} {s.ticker:<6} '
                    f'1D={s.predicted_1d:+.2%}  '
                    f'3D={s.predicted_3d:+.2%}  '
                    f'5D={s.predicted_5d:+.2%}  '
                    f'target={s.price_target_5d:.2f}  '
                    f'conf={s.confidence:.0%}  '
                    f'posts={s.post_count_1d}  '
                    f'news={s.news_count_1d}  '
                    f'st={s.st_count_1d}'
                )
        except Exception as e:
            logger.error(f'Dry run signal generation failed: {e}')
        return

    # ── Step 2b: Full run ─────────────────────────────────────────────────
    from scripts.daily_run import run
    summary = run(
        reddit_counts=reddit_counts,
        today=today,
        news_data=news_data,
        stocktwits_data=st_data,
    )

    logger.info('=== Daily Run Complete ===')
    logger.info(f'Actions: {summary["actions"]}')
    if summary.get('skipped'):
        logger.warning(f'Run skipped — reason: {summary["reason"]}')

    # ── Save live feature vectors for future retraining ───────────────────
    if not summary.get('skipped') and summary.get('actions'):
        try:
            result = subprocess.run(
                [sys.executable, 'scripts/append_live_features.py',
                 '--date', today],
                capture_output=True, text=True, cwd='.'
            )
            if result.returncode == 0:
                logger.info('live_features_appended')
            else:
                logger.warning(
                    f'live_features_append_failed: {result.stderr[:200]}'
                )
        except Exception as e:
            logger.warning(f'live_features_append_error: {e}')

    # ── Fill pending targets regardless of signal outcome ─────────────────
    try:
        result = subprocess.run(
            [sys.executable, 'scripts/append_live_features.py',
             '--fill-targets-only'],
            capture_output=True, text=True, cwd='.'
        )
        if result.returncode != 0:
            logger.warning(
                f'target_fill_failed: {result.stderr[:200]}'
            )
    except Exception as e:
        logger.warning(f'target_fill_error: {e}')

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
