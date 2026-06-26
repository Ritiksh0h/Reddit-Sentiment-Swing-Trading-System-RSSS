"""
One-time migration: flat files → PostgreSQL + MongoDB.

Safe to run multiple times — uses INSERT OR IGNORE / ON CONFLICT DO NOTHING.

Sources:
  logs/paper_trades.jsonl            → signals + trades tables (PG)
  data/live/paper_performance.jsonl  → portfolio_snapshots (PG)
  logs/ic_monitor.jsonl              → ic_monitor (PG)
  experiments/backtest_v2_results.json → backtest_results (MongoDB)
  models/training_metadata_v2.json   → model_metadata (MongoDB)

Usage:
    python scripts/migrate_to_db.py
"""
import json
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, '.')

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
log = logging.getLogger('migrate')


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        log.warning('Not found: %s', path)
        return []
    records = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception as e:
            log.warning('Line %d parse error in %s: %s', i, path.name, e)
    return records


def migrate_trades(records: list[dict]) -> tuple[int, int]:
    """Insert OPEN → signals, OPEN → trades table."""
    from api.db import insert_signal, _exec, ensure_tables
    ensure_tables()
    sig_ok = sig_err = trade_ok = trade_err = 0

    for r in records:
        action = r.get('action', '')
        date   = r.get('date', '')
        ticker = r.get('ticker', '')

        if action == 'OPEN':
            # Signals table
            fv = r.get('feature_vector_14') or r.get('feature_vector_11') or r.get('feature_vector') or {}
            ok = insert_signal({
                'run_date':        date,
                'ticker':          ticker,
                'signal':          r.get('signal', 'BULLISH'),
                'pred_1d':         r.get('predicted_1d') or r.get('pred_1d'),
                'pred_3d':         r.get('predicted_3d') or r.get('pred_3d'),
                'pred_5d':         r.get('predicted_5d') or r.get('predicted_return_5d'),
                'confidence':      r.get('confidence'),
                'post_count':      r.get('post_count_1d') or fv.get('post_count_1d'),
                'sentiment':       fv.get('avg_sentiment_1d'),
                'composite_score': None,
                'passed_density':  (r.get('post_count_1d', 0) or 0) >= 5,
            })
            if ok: sig_ok += 1
            else:  sig_err += 1

            # Trades table (OPEN half)
            fill  = r.get('fill_price') or r.get('entry_price', 0) or 0
            size  = r.get('position_size_dollars', 0) or 0
            ok2   = _exec(
                """
                INSERT INTO trades
                  (ticker, action, entry_date, entry_price, n_shares,
                   cost_basis, pred_5d, confidence)
                VALUES
                  (:ticker, :action, :entry_date, :entry_price, :n_shares,
                   :cost_basis, :pred_5d, :confidence)
                """,
                {
                    'ticker':     ticker,
                    'action':     'OPEN',
                    'entry_date': date,
                    'entry_price': fill,
                    'n_shares':   int(size / fill) if fill else 0,
                    'cost_basis': size,
                    'pred_5d':    r.get('predicted_return_5d') or r.get('predicted_5d'),
                    'confidence': r.get('confidence'),
                }
            )
            if ok2: trade_ok += 1
            else:   trade_err += 1

    return sig_ok, trade_ok


def migrate_performance(records: list[dict]) -> int:
    from api.db import insert_portfolio_snapshot, ensure_tables
    ensure_tables()
    ok = err = 0
    for r in records:
        d = r.get('date', '')[:10]
        if not d:
            continue
        pv   = r.get('portfolio_value', 0)
        sc   = r.get('starting_capital', 10000)
        pr   = r.get('portfolio_return')
        spy  = r.get('spy_return_today')
        alpha = r.get('alpha')
        result = insert_portfolio_snapshot({
            'snapshot_date':    d,
            'equity':           pv,
            'cash':             None,
            'position_value':   None,
            'total_return_pct': round(pr * 100, 4) if pr is not None else round((pv - sc) / sc * 100, 4),
            'spy_return_today': spy,
            'alpha':            alpha,
            'n_positions':      r.get('n_trades_today'),
            'regime_label':     None,
        })
        if result: ok += 1
        else:      err += 1
    log.info('portfolio_snapshots: %d ok, %d err', ok, err)
    return ok


def migrate_ic_monitor(records: list[dict]) -> int:
    from api.db import upsert_ic_monitor, ensure_tables
    ensure_tables()
    ok = err = 0
    for r in records:
        d = r.get('date', '') or r.get('check_date', '')
        result = upsert_ic_monitor({
            'check_date':           d[:10],
            'rolling_ic_30d':       r.get('rolling_ic_30d') or r.get('ic_30d'),
            'rolling_ic_7d':        r.get('rolling_ic_7d')  or r.get('ic_7d'),
            'ic_trend':             r.get('ic_trend') or r.get('status'),
            'kill_switch_triggered': r.get('kill_switch_triggered', False),
        })
        if result: ok += 1
        else:      err += 1
    log.info('ic_monitor: %d ok, %d err', ok, err)
    return ok


def migrate_mongo() -> None:
    try:
        from api.db import get_mongo_db
        mdb = get_mongo_db()
        if mdb is None:
            log.info('MongoDB not configured — skipping')
            return
    except Exception as e:
        log.warning('MongoDB unavailable: %s', e)
        return

    # Backtest results
    bt_path = Path('experiments/backtest_v2_results.json')
    if bt_path.exists():
        try:
            with open(bt_path) as f:
                bt = json.load(f)
            bt['version']    = 'v2'
            bt['created_at'] = datetime.now(timezone.utc)
            mdb['backtest_results'].replace_one(
                {'version': 'v2', 'period': bt.get('period', '2024-2025')},
                bt, upsert=True,
            )
            log.info('mongodb backtest_results saved')
        except Exception as e:
            log.warning('mongodb backtest_results failed: %s', e)

    # Model metadata
    meta_path = Path('models/training_metadata_v2.json')
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            meta['version']    = 'v2'
            meta['created_at'] = datetime.now(timezone.utc)
            mdb['model_metadata'].replace_one({'version': 'v2'}, meta, upsert=True)
            log.info('mongodb model_metadata saved')
        except Exception as e:
            log.warning('mongodb model_metadata failed: %s', e)


def main():
    log.info('=== RSSS Migration: flat files → databases ===')

    from api.db import ensure_tables
    if not ensure_tables():
        log.warning('Database unreachable — will attempt anyway (SQLite fallback)')

    # paper_trades.jsonl → signals + trades
    trades_path = Path('logs/paper_trades.jsonl')
    records = _load_jsonl(trades_path)
    log.info('paper_trades.jsonl: %d records', len(records))
    if records:
        sig_ok, trade_ok = migrate_trades(records)
        log.info('signals inserted: %d', sig_ok)
        log.info('trades inserted:  %d', trade_ok)

    # paper_performance.jsonl → portfolio_snapshots
    perf_path = Path('data/live/paper_performance.jsonl')
    perf_records = _load_jsonl(perf_path)
    log.info('paper_performance.jsonl: %d records', len(perf_records))
    if perf_records:
        migrate_performance(perf_records)

    # ic_monitor.jsonl → ic_monitor
    ic_path = Path('logs/ic_monitor.jsonl')
    ic_records = _load_jsonl(ic_path)
    log.info('ic_monitor.jsonl: %d records', len(ic_records))
    if ic_records:
        migrate_ic_monitor(ic_records)

    # MongoDB
    migrate_mongo()

    log.info('=== Migration complete ===')


if __name__ == '__main__':
    main()
