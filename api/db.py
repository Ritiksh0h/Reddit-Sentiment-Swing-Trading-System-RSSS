"""
PostgreSQL persistence for trade logs.

Active when DB_URL env var is set (Railway production / GitHub Actions).
Every function degrades gracefully to a no-op when DB_URL is absent,
so local development requires no database.

Table schema (auto-created on first use):

    trade_log
    ─────────
    id         BIGSERIAL PRIMARY KEY
    ticker     TEXT      NOT NULL
    trade_date TEXT      NOT NULL          -- YYYY-MM-DD
    action     TEXT      NOT NULL          -- OPEN | CLOSE_* | etc.
    logged_at  TIMESTAMPTZ DEFAULT NOW()
    payload    JSONB     NOT NULL          -- full signal record
"""
import json
import logging
import os

logger = logging.getLogger(__name__)

_engine        = None
_tables_ready  = False

_DDL = """
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


def _get_engine():
    """Lazy-init SQLAlchemy engine. Returns None when DB_URL is unset."""
    global _engine
    if _engine is not None:
        return _engine

    db_url = os.getenv('DB_URL')
    if not db_url:
        return None

    try:
        from sqlalchemy import create_engine
        _engine = create_engine(
            db_url,
            pool_pre_ping=True,   # recycle stale connections
            pool_size=3,
            max_overflow=5,
        )
        logger.info('db engine ready')
        return _engine
    except Exception as exc:
        logger.warning(f'db_engine_init_failed: {exc}')
        return None


def ensure_tables() -> bool:
    """
    Create the trade_log table and indexes if they don't exist.
    Called at API startup. Returns True when the DB is reachable.
    """
    global _tables_ready
    if _tables_ready:
        return True

    engine = _get_engine()
    if engine is None:
        return False

    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text(_DDL))
            conn.commit()
        _tables_ready = True
        logger.info('db tables ready')
        return True
    except Exception as exc:
        logger.warning(f'db_ensure_tables_failed: {exc}')
        return False


def load_trades(n: int = 500) -> list:
    """
    Return the last *n* trade records, oldest-first.
    Returns [] when the DB is unavailable.
    """
    engine = _get_engine()
    if engine is None:
        return []

    sql = "SELECT payload FROM trade_log ORDER BY id DESC LIMIT :n"
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            rows = conn.execute(text(sql), {'n': n}).fetchall()
        return [json.loads(row[0]) for row in reversed(rows)]
    except Exception as exc:
        logger.warning(f'db_load_trades_failed: {exc}')
        return []


def insert_trade(record: dict) -> bool:
    """
    Insert one record into trade_log. Returns True on success.
    Ensures the table exists before the first write.
    """
    ensure_tables()
    engine = _get_engine()
    if engine is None:
        return False

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
        logger.warning(f'db_insert_trade_failed: {exc}')
        return False
