#!/usr/bin/env python3
"""
universe_manager.py — Four-Stage Dynamic Universe Architecture.

Manages the progression of tickers through 4 stages:
  Stage 1 — MONITORING  : any Reddit mention
  Stage 2 — WATCHLIST   : passes eligibility gates
  Stage 3 — CANDIDATE   : 30-day IC probation
  Stage 4 — TRADE UNIVERSE : active trading (core + promoted)

Usage:
    python scripts/universe_manager.py --status
    python scripts/universe_manager.py --discover

MongoDB collections used:
    universe_monitoring   {ticker, date, post_count, first_seen}
    universe_watchlist    {ticker, promoted_date, price, avg_dollar_vol,
                           history_days, is_etf, post_count_7d}
    universe_candidates   {ticker, probation_start, days_observed,
                           ic_samples, avg_ic, status}

# ─── Integration note (for a future session) ────────────────────────────────
# In scripts/daily_run_live.py, when universe_manager has run for 30+ days
# and produced its first promotions, replace:
#
#   TRADE_UNIVERSE = set(load_tickers(TICKERS_TRADE_PATH))
#
# With:
#
#   from scripts.universe_manager import load_trade_universe
#   TRADE_UNIVERSE = load_trade_universe()
#
# load_trade_universe() returns CORE_UNIVERSE | promoted_candidates_from_mongodb.
# Do NOT make this change until at least one candidate has completed probation.
# ─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent

# ── Eligibility thresholds ────────────────────────────────────────────────────
MIN_PRICE            = 5.0
MIN_AVG_DOLLAR_VOL   = 10_000_000.0   # $10M average daily dollar volume
MIN_HISTORY_DAYS     = 252
MIN_WATCHLIST_DAYS   = 7              # days in watchlist before candidate
PROBATION_DAYS       = 30             # trading days on probation
MIN_IC_FOR_PROMOTE   = 0.010          # avg IC required to join trade universe
DEMOTION_LOW_POSTS_WEEKS  = 4         # weeks of post_count_7d < 5 triggers demotion
DEMOTION_LOW_POSTS_GATE   = 5
DEMOTION_LOW_VOL_DAYS     = 20
DEMOTION_LOW_VOL_GATE     = 5_000_000.0
MONITORING_MIN_POSTS_DAYS = 3         # at least 3 days with >= 3 posts in last 7
MONITORING_MIN_POSTS_DAY  = 3


# ── Load core universe ────────────────────────────────────────────────────────
def _load_core_universe() -> set[str]:
    from config.settings import load_tickers, TICKERS_TRADE_PATH
    try:
        return set(load_tickers(TICKERS_TRADE_PATH))
    except Exception as exc:
        logger.warning("load_core_universe_failed: %s", exc)
        return set()


CORE_UNIVERSE: set[str] = _load_core_universe()


def load_trade_universe() -> set[str]:
    """
    Return the full active trade universe:
    CORE_UNIVERSE (from tickers_trade.txt) PLUS any MongoDB-promoted candidates.
    Falls back to CORE_UNIVERSE when MongoDB is unavailable.
    """
    db = _get_db()
    if db is None:
        return set(CORE_UNIVERSE)

    try:
        promoted = {
            doc["ticker"]
            for doc in db["universe_candidates"].find({"status": "PROMOTED"})
        }
        return set(CORE_UNIVERSE) | promoted
    except Exception as exc:
        logger.warning("load_trade_universe_mongo_failed: %s", exc)
        return set(CORE_UNIVERSE)


# ── MongoDB helpers ───────────────────────────────────────────────────────────
def _get_db():
    """Return MongoDB database handle, or None."""
    try:
        from api.db import get_mongo_db
        return get_mongo_db()
    except Exception as exc:
        logger.warning("mongo_unavailable: %s", exc)
        return None


def _today_str() -> str:
    return date.today().isoformat()


# ── Eligibility check ─────────────────────────────────────────────────────────
def check_eligibility(ticker: str) -> dict:
    """
    Fetch yfinance data and apply Stage 2 eligibility gates.
    Returns a dict with passes=True/False and fail_reason.
    All errors return passes=False — bad tickers must not crash the pipeline.
    """
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info
        hist = yf.download(ticker, period="60d", auto_adjust=True, progress=False)

        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        if hist.empty:
            return _fail(ticker, "no_price_history")

        # Price
        price = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
        if price < MIN_PRICE:
            return _fail(ticker, f"price_too_low ({price:.2f} < {MIN_PRICE})")

        # History length
        history_days = len(hist)
        if history_days < MIN_HISTORY_DAYS:
            return _fail(ticker, f"insufficient_history ({history_days}d < {MIN_HISTORY_DAYS}d)")

        # Average dollar volume (20-day)
        if "Volume" in hist.columns and len(hist) >= 20:
            adv = float((hist["Close"] * hist["Volume"]).tail(20).mean())
        else:
            adv = 0.0
        if adv < MIN_AVG_DOLLAR_VOL:
            return _fail(ticker, f"low_dollar_volume ({adv/1e6:.1f}M < {MIN_AVG_DOLLAR_VOL/1e6:.0f}M)")

        # ETF check
        quote_type = str(info.get("quoteType", "")).upper()
        is_etf     = quote_type == "ETF"
        if is_etf:
            return _fail(ticker, "is_etf")

        return {
            "ticker":           ticker,
            "price":            round(price, 2),
            "avg_dollar_vol":   round(adv, 0),
            "history_days":     history_days,
            "is_etf":           is_etf,
            "passes":           True,
            "fail_reason":      None,
        }

    except Exception as exc:
        return _fail(ticker, f"exception: {exc}")


def _fail(ticker: str, reason: str) -> dict:
    return {
        "ticker":         ticker,
        "price":          0.0,
        "avg_dollar_vol": 0.0,
        "history_days":   0,
        "is_etf":         False,
        "passes":         False,
        "fail_reason":    reason,
    }


# ── Stage 1: scan Reddit posts ────────────────────────────────────────────────
def _get_monitoring_tickers(db) -> dict[str, list[int]]:
    """
    Return non-universe tickers that appeared in Reddit posts in the last 7 days.
    {ticker: [post_counts_per_day]}
    """
    if db is None:
        return {}

    cutoff = (date.today() - timedelta(days=7)).isoformat()
    try:
        docs = list(
            db["reddit_posts"].find(
                {"date": {"$gte": cutoff}},
                {"ticker": 1, "date": 1, "post_count": 1, "_id": 0},
            )
        )
    except Exception as exc:
        logger.warning("monitoring_query_failed: %s", exc)
        return {}

    from collections import defaultdict
    counts: dict[str, list[int]] = defaultdict(list)
    for doc in docs:
        t = doc.get("ticker", "").upper()
        if not t or t in CORE_UNIVERSE:
            continue
        counts[t].append(int(doc.get("post_count", 0)))

    return dict(counts)


def _ticker_days_with_posts(counts: list[int], min_posts: int) -> int:
    return sum(1 for c in counts if c >= min_posts)


# ── --discover mode ───────────────────────────────────────────────────────────
def cmd_discover(db) -> None:
    today = _today_str()
    print(f"\n=== Universe Discovery — {today} ===\n")

    # ── 1. Scan Stage 1 mentions ──────────────────────────────────────────────
    monitoring = _get_monitoring_tickers(db)
    if not monitoring:
        print("Stage 1: no non-universe Reddit mentions found in last 7 days.")
        if db is None:
            print("  (MongoDB unavailable — no data to scan)")
        return

    print(f"Stage 1 MONITORING: {len(monitoring)} non-universe tickers seen in last 7d")

    # Sort by total posts
    ranked = sorted(monitoring.items(), key=lambda x: sum(x[1]), reverse=True)
    for ticker, counts in ranked[:10]:
        active_days = _ticker_days_with_posts(counts, MONITORING_MIN_POSTS_DAY)
        print(f"  {ticker:<8} total={sum(counts):>4} posts  "
              f"active_days={active_days}/7")

    # ── 2. Stage 2: check eligibility for sustained tickers ──────────────────
    candidates_for_watchlist = [
        t for t, counts in monitoring.items()
        if _ticker_days_with_posts(counts, MONITORING_MIN_POSTS_DAY) >= MONITORING_MIN_POSTS_DAYS
    ]

    print(f"\nStage 2 eligibility check: {len(candidates_for_watchlist)} tickers")
    eligible: list[dict] = []

    for ticker in candidates_for_watchlist:
        elig = check_eligibility(ticker)
        status = "✓ ELIGIBLE" if elig["passes"] else f"✗ {elig['fail_reason']}"
        print(f"  {ticker:<8} {status}")
        if elig["passes"]:
            eligible.append({**elig, "post_count_7d": sum(monitoring[ticker])})

    if not eligible:
        print("\n  No tickers passed Stage 2 eligibility.")
    else:
        print(f"\n  {len(eligible)} tickers eligible for watchlist.")

    # ── 3. Update watchlist in MongoDB ───────────────────────────────────────
    if db is not None and eligible:
        existing_watchlist = {
            doc["ticker"]
            for doc in db["universe_watchlist"].find({}, {"ticker": 1, "_id": 0})
        }
        newly_added = 0
        for elig in eligible:
            if elig["ticker"] not in existing_watchlist:
                db["universe_watchlist"].insert_one({
                    "ticker":         elig["ticker"],
                    "promoted_date":  today,
                    "price":          elig["price"],
                    "avg_dollar_vol": elig["avg_dollar_vol"],
                    "history_days":   elig["history_days"],
                    "is_etf":         elig["is_etf"],
                    "post_count_7d":  elig["post_count_7d"],
                    "created_at":     datetime.now(timezone.utc).isoformat(),
                })
                logger.info("watchlist_added ticker=%s", elig["ticker"])
                newly_added += 1
            else:
                # Update post count
                db["universe_watchlist"].update_one(
                    {"ticker": elig["ticker"]},
                    {"$set": {"post_count_7d": elig["post_count_7d"],
                               "updated_at": today}},
                )
        print(f"  Watchlist updated: {newly_added} new additions.")

    # ── 4. Stage 3: promote long-watchlist tickers to probation ──────────────
    if db is None:
        return

    try:
        watchlist_docs = list(db["universe_watchlist"].find({}))
    except Exception as exc:
        logger.warning("watchlist_query_failed: %s", exc)
        return

    promoted_to_candidate = 0
    cutoff_date = (date.today() - timedelta(days=MIN_WATCHLIST_DAYS)).isoformat()
    existing_candidates = {
        doc["ticker"]
        for doc in db["universe_candidates"].find({}, {"ticker": 1, "_id": 0})
    }

    for doc in watchlist_docs:
        ticker     = doc.get("ticker", "")
        prom_date  = doc.get("promoted_date", today)

        if ticker in existing_candidates or ticker in CORE_UNIVERSE:
            continue
        if prom_date > cutoff_date:
            continue

        # Re-check eligibility before promoting
        elig = check_eligibility(ticker)
        if not elig["passes"]:
            logger.info(
                "watchlist_demote ticker=%s reason=%s",
                ticker, elig["fail_reason"]
            )
            db["universe_watchlist"].delete_one({"ticker": ticker})
            continue

        db["universe_candidates"].insert_one({
            "ticker":          ticker,
            "probation_start": today,
            "days_observed":   0,
            "ic_samples":      [],
            "avg_ic":          0.0,
            "status":          "PROBATION",
            "created_at":      datetime.now(timezone.utc).isoformat(),
        })
        promoted_to_candidate += 1
        logger.info("candidate_added ticker=%s", ticker)

    if promoted_to_candidate:
        print(f"\nStage 3: {promoted_to_candidate} tickers moved to probation.")

    # ── 5. Stage 4: graduate or demote probation tickers ─────────────────────
    _check_probation_outcomes(db, today)


def _check_probation_outcomes(db, today: str) -> None:
    try:
        candidates = list(db["universe_candidates"].find({"status": "PROBATION"}))
    except Exception as exc:
        logger.warning("candidates_query_failed: %s", exc)
        return

    for doc in candidates:
        ticker = doc.get("ticker", "")
        days   = int(doc.get("days_observed", 0))
        avg_ic = float(doc.get("avg_ic", 0.0))

        if days < PROBATION_DAYS:
            continue

        if avg_ic >= MIN_IC_FOR_PROMOTE:
            db["universe_candidates"].update_one(
                {"ticker": ticker},
                {"$set": {"status": "PROMOTED", "promoted_date": today}},
            )
            logger.info(
                "candidate_promoted ticker=%s avg_ic=%.4f", ticker, avg_ic
            )
            print(f"\n  PROMOTED: {ticker} → Stage 4  (avg_ic={avg_ic:.4f})")
        else:
            db["universe_candidates"].update_one(
                {"ticker": ticker},
                {"$set": {"status": "DEMOTED", "demoted_date": today,
                           "demote_reason": f"avg_ic={avg_ic:.4f} < {MIN_IC_FOR_PROMOTE}"}},
            )
            logger.info(
                "candidate_demoted ticker=%s avg_ic=%.4f", ticker, avg_ic
            )
            print(f"\n  DEMOTED: {ticker} → Watchlist  (avg_ic={avg_ic:.4f} < {MIN_IC_FOR_PROMOTE})")


# ── --status mode ─────────────────────────────────────────────────────────────
def cmd_status(db) -> None:
    today = _today_str()
    W = 56

    print()
    print("═" * W)
    print(f"RSSS Universe Status — {today}")
    print("═" * W)

    # ── Stage 4 ───────────────────────────────────────────────────────────────
    promoted: list[str] = []
    if db is not None:
        try:
            promoted = [
                doc["ticker"]
                for doc in db["universe_candidates"].find({"status": "PROMOTED"})
            ]
        except Exception:
            pass

    print(f"\nStage 4  TRADE UNIVERSE   "
          f"({len(CORE_UNIVERSE)} core + {len(promoted)} promoted)")
    core_sorted = sorted(CORE_UNIVERSE)
    line = "  Core:     "
    for t in core_sorted:
        if len(line) + len(t) + 1 > W - 2:
            print(line)
            line = "            "
        line += t + " "
    print(line)
    if promoted:
        print(f"  Promoted: {' '.join(sorted(promoted))}")

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    print(f"\nStage 3  CANDIDATES   (probation — NOT trading)")
    if db is None:
        print("  (MongoDB unavailable)")
    else:
        try:
            cands = list(db["universe_candidates"].find({"status": "PROBATION"}))
        except Exception:
            cands = []
        if not cands:
            print("  (none)")
        for doc in cands:
            t    = doc.get("ticker", "?")
            days = int(doc.get("days_observed", 0))
            ic   = float(doc.get("avg_ic", 0.0))
            tag  = "on track" if ic >= MIN_IC_FOR_PROMOTE else "will demote"
            print(f"  {t:<8} day {days:>2}/{PROBATION_DAYS}  IC={ic:.4f}  ({tag})")

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    print(f"\nStage 2  WATCHLIST   (screened, watching)")
    if db is None:
        print("  (MongoDB unavailable)")
    else:
        try:
            wlist = list(db["universe_watchlist"].find({}))
        except Exception:
            wlist = []
        if not wlist:
            print("  (none)")
        for doc in wlist:
            t   = doc.get("ticker", "?")
            p   = doc.get("price", 0.0)
            adv = doc.get("avg_dollar_vol", 0.0)
            pc  = doc.get("post_count_7d", 0)
            print(f"  {t:<8} price=${p:.0f}  "
                  f"ADDV=${adv/1e6:.0f}M  "
                  f"posts={pc}/7d")

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    print(f"\nStage 1  MONITORING   (all Reddit mentions, last 7d)")
    monitoring = _get_monitoring_tickers(db)
    if not monitoring:
        if db is None:
            print("  (MongoDB unavailable)")
        else:
            print("  (no non-universe tickers in last 7 days)")
    else:
        ranked = sorted(monitoring.items(), key=lambda x: sum(x[1]), reverse=True)
        print("  Top 10 non-universe tickers by post count:")
        for ticker, counts in ranked[:10]:
            print(f"    {ticker:<8} {sum(counts):>4} posts  "
                  f"({len(counts)} days active)")

    print()
    print("═" * W)


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="RSSS Universe Manager")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--discover", action="store_true",
                       help="Scan Reddit, check eligibility, promote tickers")
    group.add_argument("--status",   action="store_true",
                       help="Print current universe state across all 4 stages")
    args = parser.parse_args()

    db = _get_db()
    if db is None:
        logger.warning("MongoDB unavailable — running in read-only / core-only mode")

    if args.status:
        cmd_status(db)
    elif args.discover:
        cmd_discover(db)


if __name__ == "__main__":
    main()
