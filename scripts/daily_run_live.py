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
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VaderSIA

        _vader = _VaderSIA()

        reddit_counts_raw = fetch_recent_posts()

        if not reddit_counts_raw:
            logger.warning('No Reddit data fetched — api_anomaly handler will trigger')
            reddit_counts = {}
        else:
            reddit_counts = {}
            for ticker, data in reddit_counts_raw.items():
                posts  = data.get('posts', [])
                growth = compute_mention_growth(
                    ticker=ticker,
                    current_count=data['post_count_1d'],
                )

                total_comments = sum(p.get('num_comments', 0) for p in posts)
                vader_scores   = [
                    _vader.polarity_scores(p.get('title', ''))['compound']
                    for p in posts if p.get('title', '').strip()
                ]
                vader_mean = round(
                    float(sum(vader_scores) / len(vader_scores)), 4
                ) if vader_scores else 0.0

                reddit_counts[ticker] = {
                    'post_count_1d':      data['post_count_1d'],
                    'mention_growth_1d':  growth['mention_growth_1d'],
                    'mention_growth_7d':  growth['mention_growth_7d'],
                    'total_comments_1d':  total_comments,
                    'vader_sentiment_1d': vader_mean,
                }

            logger.info(f'Reddit data ready: {len(reddit_counts)} tickers')

    except Exception as e:
        logger.error(f'Reddit fetch failed: {e}')
        reddit_counts = {}

    # ── Step 1b: Fetch news sentiment (Finnhub primary, yfinance fallback) ──
    logger.info('Fetching news sentiment...')
    news_data: dict = {}
    try:
        from data.finnhub_news_fetcher import fetch_live_news_sentiment
        finnhub_key = os.getenv('FINNHUB_API_KEY')
        if finnhub_key:
            from config.settings import load_tickers, TICKERS_TRADE_PATH, TICKERS_WATCH_PATH
            _all_tickers = sorted(
                set(load_tickers(TICKERS_TRADE_PATH)) |
                set(load_tickers(TICKERS_WATCH_PATH))
            )
            news_data = fetch_live_news_sentiment(_all_tickers, finnhub_key)
            logger.info(f'Finnhub news ready: {len(news_data)} tickers')
        else:
            logger.warning('FINNHUB_API_KEY not set — skipping Finnhub')
    except Exception as e:
        logger.warning(f'Finnhub news fetch failed: {e} — trying yfinance fallback')

    if not news_data or all(v.get('news_count_1d', 0) == 0 for v in news_data.values()):
        try:
            from data.news_fetcher import fetch_yfinance_news
            news_data = fetch_yfinance_news()
            logger.info(f'yfinance news fallback ready: {len(news_data)} tickers')
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
                    f'hold={s.hold_days}d horizon={s.horizon}  '
                    f'target={s.price_target_5d:.2f}  '
                    f'conf={s.confidence:.0%}  '
                    f'posts={s.post_count_1d}  '
                    f'news={s.news_count_1d}  '
                    f'st={s.st_count_1d}'
                )
        except Exception as e:
            logger.error(f'Dry run signal generation failed: {e}')
            signals = []
        save_run_to_db(
            {'actions': [], 'signals': signals, 'skipped': False, 'dry_run': True},
            today,
            reddit_counts,
        )
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
    # FIX 3: removed summary.get('actions') gate — always run on non-skipped
    # days so NEUTRAL/blocked signals still get saved to all_signals.jsonl
    # and forwarded to features_live_v2.parquet via append_live_features.py.
    if not summary.get('skipped'):
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

    # ── Persist run metadata to databases ────────────────────────────────
    save_run_to_db(summary, today, reddit_counts)

    # Make summary JSON-serializable before printing.
    # FIX 4 added summary['all_signals'] as a list of SignalRecord dataclasses
    # which json.dumps cannot serialize natively.
    import dataclasses as _dc
    summary_serializable: dict = {}
    for _k, _v in summary.items():
        if _k == 'all_signals' and isinstance(_v, list):
            summary_serializable[_k] = [
                _dc.asdict(s) if _dc.is_dataclass(s)
                else (s._asdict() if hasattr(s, '_asdict')
                else (s.__dict__ if hasattr(s, '__dict__')
                else s))
                for s in _v
            ]
        else:
            summary_serializable[_k] = _v

    print(json.dumps(summary_serializable, indent=2, default=str))


def save_run_to_db(summary: dict, today: str, reddit_counts: dict) -> None:
    """
    Write run metadata, signals, and Reddit aggregates to PostgreSQL + MongoDB.
    Wrapped entirely in try/except — DB failure must never stop trading.
    """
    source = 'github_actions' if os.getenv('CI') == 'true' else 'launchd'

    # ── PostgreSQL: daily_runs row ────────────────────────────────────────
    try:
        from api.db import insert_daily_run, insert_signal, insert_reddit_daily
        total_posts = sum(
            v.get('post_count_1d', 0) for v in reddit_counts.values()
        )
        all_signals = summary.get('all_signals', []) or summary.get('signals', [])
        insert_daily_run({
            'run_date':               today,
            'run_time':               datetime.now(timezone.utc),
            'total_posts':            total_posts,
            'tickers_found':          len(reddit_counts),
            'tickers_passed_density': sum(
                1 for v in reddit_counts.values() if v.get('post_count_1d', 0) >= 5
            ),
            'signals_generated':      len(all_signals),
            'trades_executed':        len([a for a in summary.get('actions', [])
                                          if (isinstance(a, str) and a.startswith('OPEN')) or
                                             (isinstance(a, dict) and a.get('action') == 'OPEN')]),
            'regime_label':           summary.get('regime', {}).get('label') if isinstance(summary.get('regime'), dict) else summary.get('regime'),
            'source':                 source,
        })
        logger.info('db_daily_run_saved')
    except Exception as exc:
        logger.warning('db_daily_run_failed: %s', exc)

    # ── PostgreSQL: signals rows ──────────────────────────────────────────
    try:
        from api.db import insert_signal
        all_signals = summary.get('all_signals', []) or summary.get('signals', [])
        for s in all_signals:
            if not s:
                continue
            # s may be a dataclass or dict
            d = s if isinstance(s, dict) else vars(s) if hasattr(s, '__dict__') else {}
            insert_signal({
                'run_date':        today,
                'ticker':          d.get('ticker'),
                'signal':          d.get('signal'),
                'pred_1d':         d.get('predicted_1d') or d.get('pred_1d'),
                'pred_3d':         d.get('predicted_3d') or d.get('pred_3d'),
                'pred_5d':         d.get('predicted_5d') or d.get('predicted_return_5d') or d.get('pred_5d'),
                'confidence':      d.get('confidence'),
                'post_count':      d.get('post_count_1d') or d.get('post_count'),
                'sentiment':       d.get('avg_sentiment_1d') or d.get('sentiment'),
                'composite_score': d.get('composite_score'),
                'passed_density':  (d.get('post_count_1d', 0) or 0) >= 5,
            })
        if all_signals:
            logger.info('db_signals_saved count=%d', len(all_signals))
    except Exception as exc:
        logger.warning('db_signals_failed: %s', exc)

    # ── PostgreSQL: reddit_daily rows ─────────────────────────────────────
    try:
        from api.db import insert_reddit_daily
        for ticker, vals in reddit_counts.items():
            sub = vals.get('subreddit_breakdown', {}) or {}
            insert_reddit_daily({
                'fetch_date':     today,
                'ticker':         ticker,
                'post_count_1d':  vals.get('post_count_1d', 0),
                'avg_sentiment':  vals.get('vader_sentiment_1d'),
                'wallstreetbets': sub.get('wallstreetbets'),
                'stocks':         sub.get('stocks'),
                'investing':      sub.get('investing'),
                'options':        sub.get('options'),
            })
        logger.info('db_reddit_daily_saved tickers=%d', len(reddit_counts))
    except Exception as exc:
        logger.warning('db_reddit_daily_failed: %s', exc)

    # ── MongoDB: full daily_run_reports document ──────────────────────────
    try:
        from api.db import get_mongo_db
        mdb = get_mongo_db()
        if mdb is not None:
            all_signals = summary.get('all_signals', []) or summary.get('signals', [])
            sig_dicts = []
            for s in all_signals:
                if isinstance(s, dict):
                    sig_dicts.append(s)
                elif hasattr(s, '__dict__'):
                    sig_dicts.append(vars(s))
            mdb['daily_run_reports'].replace_one(
                {'date': today},
                {
                    'date':       today,
                    'source':     source,
                    'reddit': {
                        'total_posts': sum(v.get('post_count_1d', 0) for v in reddit_counts.values()),
                        'tickers':     {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, float) or not (vv != vv)}
                                        for k, v in reddit_counts.items()},
                    },
                    'signals':    sig_dicts,
                    'actions':    [a if isinstance(a, dict) else str(a) for a in summary.get('actions', [])],
                    'regime':     summary.get('regime', {}),
                    'created_at': datetime.now(timezone.utc),
                },
                upsert=True,
            )
            logger.info('mongodb_daily_run_saved')
    except Exception as exc:
        logger.warning('mongodb_daily_run_failed: %s', exc)

    # ── MongoDB: raw Reddit post aggregates ───────────────────────────────
    try:
        from api.db import get_mongo_db
        mdb = get_mongo_db()
        if mdb is not None:
            for ticker, vals in reddit_counts.items():
                if vals.get('post_count_1d', 0) > 0:
                    mdb['reddit_posts'].insert_one({
                        'ticker':     ticker,
                        'date':       today,
                        'post_count': vals.get('post_count_1d'),
                        'sentiment':  vals.get('avg_sentiment_1d'),
                        'source':     'arctic_shift',
                        'created_at': datetime.now(timezone.utc),
                    })
    except Exception as exc:
        logger.warning('mongodb_reddit_posts_failed: %s', exc)


if __name__ == '__main__':
    main()
