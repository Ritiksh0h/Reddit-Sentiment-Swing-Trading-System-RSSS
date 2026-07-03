"""
RSSS — database layer.

PostgreSQL (Supabase) via SQLAlchemy
  • Connection string: SUPABASE_URL env var  (preferred)
    Legacy alias:      DB_URL env var         (backward compat)
    Fallback:          sqlite:///data/rsss.db (local dev, no env var)

MongoDB Atlas via pymongo
  • Connection string: MONGODB_URL env var
  • Database name:     MONGODB_DB env var  (default: 'rsss')
  • Gracefully skipped when MONGODB_URL is absent.

All public functions degrade to no-ops / empty results when the
respective database is unavailable — DB failure must never crash
the trading pipeline.

Public API (backward-compatible with original db.py):
  ensure_tables()        → create PG tables + indexes
  load_trades(n)         → read last-n from trade_log (legacy)
  insert_trade(record)   → write to trade_log (legacy)

New structured-write helpers:
  insert_daily_run(data)
  insert_signal(data)
  insert_portfolio_snapshot(data)
  upsert_ic_monitor(data)
  insert_reddit_daily(data)

MongoDB helpers:
  get_mongo_db()            → database handle (or None)
  ensure_mongo_indexes()    → create indexes
"""
import json
import logging
import os
from datetime import date, datetime, timezone

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ── PostgreSQL / SQLite ──────────────────────────────────────────────────────

_engine       = None
_tables_ready = False

_LEGACY_DDL = """
CREATE TABLE IF NOT EXISTS trade_log (
    id         BIGSERIAL    PRIMARY KEY,
    ticker     TEXT         NOT NULL,
    trade_date TEXT         NOT NULL,
    action     TEXT         NOT NULL,
    logged_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    payload    JSONB        NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_trade_log_logged_at ON trade_log (logged_at DESC);
CREATE INDEX IF NOT EXISTS ix_trade_log_ticker     ON trade_log (ticker);
"""


def _get_pg_url() -> str | None:
    """Return connection string, trying SUPABASE_URL → DB_URL → SQLite fallback."""
    url = os.getenv('SUPABASE_URL') or os.getenv('DB_URL')
    if url:
        return url
    # Local dev fallback — SQLite
    os.makedirs('data', exist_ok=True)
    return 'sqlite:///data/rsss.db'


def _get_engine():
    """Lazy-init SQLAlchemy engine. Returns None on failure."""
    global _engine
    if _engine is not None:
        return _engine

    url = _get_pg_url()
    if url is None:
        return None

    try:
        from sqlalchemy import create_engine
        kwargs = dict(pool_pre_ping=True)
        if not url.startswith('sqlite'):
            kwargs.update(pool_size=3, max_overflow=5)
        _engine = create_engine(url, **kwargs)
        logger.info('db_engine_ready url_prefix=%s', url[:30])
        return _engine
    except Exception as exc:
        logger.warning('db_engine_init_failed: %s', exc)
        return None


# ── Structured DDL (SQLAlchemy ORM-free, raw SQL for portability) ────────────

_STRUCTURED_DDL = """
CREATE TABLE IF NOT EXISTS daily_runs (
    id                     BIGSERIAL PRIMARY KEY,
    run_date               DATE      NOT NULL,
    run_time               TIMESTAMP NOT NULL,
    total_posts            INTEGER,
    tickers_found          INTEGER,
    tickers_passed_density INTEGER,
    tickers_passed_ma      INTEGER,
    signals_generated      INTEGER,
    trades_executed        INTEGER,
    regime_label           VARCHAR(20),
    vix_percentile         REAL,
    spy_price              REAL,
    duration_seconds       REAL,
    source                 VARCHAR(30),
    created_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS signals (
    id              BIGSERIAL PRIMARY KEY,
    run_date        DATE      NOT NULL,
    ticker          VARCHAR(10) NOT NULL,
    signal          VARCHAR(10) NOT NULL,
    pred_1d         REAL,
    pred_3d         REAL,
    pred_5d         REAL,
    confidence      REAL,
    post_count      INTEGER,
    sentiment       REAL,
    regime_score    REAL,
    vix_pct         REAL,
    passed_density  BOOLEAN,
    passed_ma       BOOLEAN,
    passed_earnings BOOLEAN,
    composite_score REAL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trades (
    id           BIGSERIAL PRIMARY KEY,
    ticker       VARCHAR(10) NOT NULL,
    action       VARCHAR(10) NOT NULL,
    entry_date   DATE,
    exit_date    DATE,
    entry_price  REAL,
    exit_price   REAL,
    n_shares     INTEGER,
    cost_basis   REAL,
    pnl_pct      REAL,
    pnl_dollars  REAL,
    exit_reason  VARCHAR(30),
    hold_days    INTEGER,
    pred_5d      REAL,
    confidence   REAL,
    is_real      BOOLEAN DEFAULT FALSE,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id               BIGSERIAL PRIMARY KEY,
    snapshot_date    DATE      NOT NULL UNIQUE,
    equity           REAL,
    cash             REAL,
    position_value   REAL,
    total_return_pct REAL,
    spy_return_today REAL,
    alpha            REAL,
    n_positions      INTEGER,
    regime_label     VARCHAR(20),
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ic_monitor (
    id                    BIGSERIAL PRIMARY KEY,
    check_date            DATE NOT NULL,
    rolling_ic_30d        REAL,
    rolling_ic_7d         REAL,
    ic_trend              VARCHAR(20),
    kill_switch_triggered BOOLEAN DEFAULT FALSE,
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reddit_daily (
    id              BIGSERIAL PRIMARY KEY,
    fetch_date      DATE      NOT NULL,
    ticker          VARCHAR(10) NOT NULL,
    post_count_1d   INTEGER,
    avg_sentiment   REAL,
    sentiment_accel REAL,
    unique_authors  INTEGER,
    wallstreetbets  INTEGER,
    stocks          INTEGER,
    investing       INTEGER,
    options         INTEGER,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (fetch_date, ticker)
);
"""

# SQLite-compatible version (no BIGSERIAL, no UNIQUE ON CONFLICT)
_STRUCTURED_DDL_SQLITE = _STRUCTURED_DDL.replace(
    'BIGSERIAL', 'INTEGER'
).replace(
    'DEFAULT CURRENT_TIMESTAMP', "DEFAULT (datetime('now'))"
)


def ensure_tables() -> bool:
    """
    Create all tables and indexes. Called at API startup.
    Returns True when DB is reachable.
    """
    global _tables_ready
    if _tables_ready:
        return True

    engine = _get_engine()
    if engine is None:
        return False

    url = _get_pg_url() or ''
    is_sqlite = url.startswith('sqlite')
    legacy_ddl = _LEGACY_DDL if not is_sqlite else _LEGACY_DDL.replace(
        'BIGSERIAL', 'INTEGER'
    ).replace('TIMESTAMPTZ', 'TIMESTAMP').replace('JSONB', 'TEXT').replace(
        'DEFAULT NOW()', "DEFAULT (datetime('now'))"
    )

    try:
        from sqlalchemy import text
        ddl = _STRUCTURED_DDL_SQLITE if is_sqlite else _STRUCTURED_DDL
        with engine.connect() as conn:
            # Legacy trade_log (backward compat) — split for SQLite
            for stmt in legacy_ddl.strip().split(';'):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(text(stmt))
            # New structured tables
            for stmt in ddl.strip().split(';'):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(text(stmt))
            conn.commit()
        _tables_ready = True
        logger.info('db_tables_ready')
        return True
    except Exception as exc:
        logger.warning('db_ensure_tables_failed: %s', exc)
        return False


# ── Legacy helpers (backward compat with execution_logger.py) ────────────────

def load_trades(n: int = 500) -> list:
    """Return last *n* records from trade_log, oldest-first."""
    engine = _get_engine()
    if engine is None:
        return []
    sql = "SELECT payload FROM trade_log ORDER BY id DESC LIMIT :n"
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            rows = conn.execute(text(sql), {'n': n}).fetchall()
        results = []
        for row in reversed(rows):
            raw = row[0]
            results.append(json.loads(raw) if isinstance(raw, str) else raw)
        return results
    except Exception as exc:
        logger.warning('db_load_trades_failed: %s', exc)
        return []


def insert_trade(record: dict) -> bool:
    """Insert one record into the legacy trade_log table."""
    ensure_tables()
    engine = _get_engine()
    if engine is None:
        return False

    url = _get_pg_url() or ''
    is_sqlite = url.startswith('sqlite')
    if is_sqlite:
        sql = """
            INSERT INTO trade_log (ticker, trade_date, action, payload)
            VALUES (:ticker, :trade_date, :action, :payload)
        """
    else:
        sql = """
            INSERT INTO trade_log (ticker, trade_date, action, payload)
            VALUES (:ticker, :trade_date, :action, CAST(:payload AS JSONB))
        """
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text(sql), {
                'ticker':     record.get('ticker', ''),
                'trade_date': record.get('date', ''),
                'action':     record.get('action', ''),
                'payload':    json.dumps(record),
            })
            conn.commit()
        return True
    except Exception as exc:
        logger.warning('db_insert_trade_failed: %s', exc)
        return False


# ── Structured write helpers ─────────────────────────────────────────────────

def _exec(sql: str, params: dict) -> bool:
    """Execute one INSERT/UPDATE, returns True on success."""
    ensure_tables()
    engine = _get_engine()
    if engine is None:
        return False
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text(sql), params)
            conn.commit()
        return True
    except Exception as exc:
        logger.warning('db_exec_failed sql=%s err=%s', sql[:60], exc)
        return False


def insert_daily_run(data: dict) -> bool:
    """Insert one row into daily_runs."""
    sql = """
        INSERT INTO daily_runs
          (run_date, run_time, total_posts, tickers_found,
           tickers_passed_density, tickers_passed_ma, signals_generated, trades_executed,
           regime_label, vix_percentile, spy_price, duration_seconds, source)
        VALUES
          (:run_date, :run_time, :total_posts, :tickers_found,
           :tickers_passed_density, :tickers_passed_ma, :signals_generated, :trades_executed,
           :regime_label, :vix_percentile, :spy_price, :duration_seconds, :source)
    """
    return _exec(sql, {
        'run_date':               data.get('run_date'),
        'run_time':               data.get('run_time', datetime.now(timezone.utc)),
        'total_posts':            data.get('total_posts'),
        'tickers_found':          data.get('tickers_found'),
        'tickers_passed_density': data.get('tickers_passed_density'),
        'tickers_passed_ma':      data.get('tickers_passed_ma'),
        'signals_generated':      data.get('signals_generated'),
        'trades_executed':        data.get('trades_executed'),
        'regime_label':           data.get('regime_label'),
        'vix_percentile':         data.get('vix_percentile'),
        'spy_price':              data.get('spy_price'),
        'duration_seconds':       data.get('duration_seconds'),
        'source':                 data.get('source', 'unknown'),
    })


def insert_signal(data: dict) -> bool:
    """Insert one row into signals."""
    sql = """
        INSERT INTO signals
          (run_date, ticker, signal, pred_1d, pred_3d, pred_5d, confidence,
           post_count, sentiment, composite_score, passed_density)
        VALUES
          (:run_date, :ticker, :signal, :pred_1d, :pred_3d, :pred_5d, :confidence,
           :post_count, :sentiment, :composite_score, :passed_density)
    """
    return _exec(sql, {
        'run_date':        data.get('run_date'),
        'ticker':          data.get('ticker'),
        'signal':          data.get('signal'),
        'pred_1d':         data.get('pred_1d'),
        'pred_3d':         data.get('pred_3d'),
        'pred_5d':         data.get('pred_5d'),
        'confidence':      data.get('confidence'),
        'post_count':      data.get('post_count'),
        'sentiment':       data.get('sentiment'),
        'composite_score': data.get('composite_score'),
        'passed_density':  data.get('passed_density'),
    })


def insert_portfolio_snapshot(data: dict) -> bool:
    """
    Upsert one row into portfolio_snapshots.
    ON CONFLICT on snapshot_date → update numeric fields.
    """
    url = _get_pg_url() or ''
    if url.startswith('sqlite'):
        sql = """
            INSERT OR REPLACE INTO portfolio_snapshots
              (snapshot_date, equity, cash, position_value, total_return_pct,
               spy_return_today, alpha, n_positions, regime_label)
            VALUES
              (:snapshot_date, :equity, :cash, :position_value, :total_return_pct,
               :spy_return_today, :alpha, :n_positions, :regime_label)
        """
    else:
        sql = """
            INSERT INTO portfolio_snapshots
              (snapshot_date, equity, cash, position_value, total_return_pct,
               spy_return_today, alpha, n_positions, regime_label)
            VALUES
              (:snapshot_date, :equity, :cash, :position_value, :total_return_pct,
               :spy_return_today, :alpha, :n_positions, :regime_label)
            ON CONFLICT (snapshot_date) DO UPDATE SET
              equity           = EXCLUDED.equity,
              cash             = EXCLUDED.cash,
              position_value   = EXCLUDED.position_value,
              total_return_pct = EXCLUDED.total_return_pct,
              spy_return_today = EXCLUDED.spy_return_today,
              alpha            = EXCLUDED.alpha,
              n_positions      = EXCLUDED.n_positions,
              regime_label     = EXCLUDED.regime_label
        """
    return _exec(sql, {
        'snapshot_date':   data.get('snapshot_date'),
        'equity':          data.get('equity'),
        'cash':            data.get('cash'),
        'position_value':  data.get('position_value'),
        'total_return_pct': data.get('total_return_pct'),
        'spy_return_today': data.get('spy_return_today'),
        'alpha':           data.get('alpha'),
        'n_positions':     data.get('n_positions'),
        'regime_label':    data.get('regime_label'),
    })


def upsert_ic_monitor(data: dict) -> bool:
    """Insert one row into ic_monitor."""
    sql = """
        INSERT INTO ic_monitor
          (check_date, rolling_ic_30d, rolling_ic_7d, ic_trend, kill_switch_triggered)
        VALUES
          (:check_date, :rolling_ic_30d, :rolling_ic_7d, :ic_trend, :kill_switch_triggered)
    """
    return _exec(sql, {
        'check_date':           data.get('check_date'),
        'rolling_ic_30d':       data.get('rolling_ic_30d'),
        'rolling_ic_7d':        data.get('rolling_ic_7d'),
        'ic_trend':             data.get('ic_trend'),
        'kill_switch_triggered': data.get('kill_switch_triggered', False),
    })


def insert_reddit_daily(data: dict) -> bool:
    """Insert one row into reddit_daily (ignore duplicates)."""
    url = _get_pg_url() or ''
    if url.startswith('sqlite'):
        sql = """
            INSERT OR IGNORE INTO reddit_daily
              (fetch_date, ticker, post_count_1d, avg_sentiment,
               wallstreetbets, stocks, investing, options)
            VALUES
              (:fetch_date, :ticker, :post_count_1d, :avg_sentiment,
               :wallstreetbets, :stocks, :investing, :options)
        """
    else:
        sql = """
            INSERT INTO reddit_daily
              (fetch_date, ticker, post_count_1d, avg_sentiment,
               wallstreetbets, stocks, investing, options)
            VALUES
              (:fetch_date, :ticker, :post_count_1d, :avg_sentiment,
               :wallstreetbets, :stocks, :investing, :options)
            ON CONFLICT (fetch_date, ticker) DO NOTHING
        """
    return _exec(sql, {
        'fetch_date':     data.get('fetch_date'),
        'ticker':         data.get('ticker'),
        'post_count_1d':  data.get('post_count_1d'),
        'avg_sentiment':  data.get('avg_sentiment'),
        'wallstreetbets': data.get('wallstreetbets'),
        'stocks':         data.get('stocks'),
        'investing':      data.get('investing'),
        'options':        data.get('options'),
    })


# ── MongoDB ──────────────────────────────────────────────────────────────────

_mongo_client = None
_mongo_db     = None


def get_mongo_db():
    """Return the MongoDB database handle, or None when MONGODB_URL is absent."""
    global _mongo_client, _mongo_db
    if _mongo_db is not None:
        return _mongo_db

    url = os.getenv('MONGODB_URL')
    if not url:
        return None

    try:
        from pymongo import MongoClient
        _mongo_client = MongoClient(url, serverSelectionTimeoutMS=5000, tls=True, tlsAllowInvalidCertificates=False)
        _mongo_client.admin.command('ping')       # fast connectivity check
        db_name  = os.getenv('MONGODB_DB', 'rsss')
        _mongo_db = _mongo_client[db_name]
        logger.info('mongodb_connected db=%s', db_name)
        return _mongo_db
    except Exception as exc:
        logger.warning('mongodb_connect_failed: %s', exc)
        return None


