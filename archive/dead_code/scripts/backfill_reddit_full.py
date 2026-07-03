"""
backfill_reddit_full.py
=======================
Fetches complete historical Reddit data from Arctic Shift API
(with PullPush fallback) for 5 financial subreddits from
2019-01-01 to today.

Two collection modes:
  subreddit  Fetch all posts from each subreddit (default, fast ~2h)
  ticker     Search per company-name alias for each tracked ticker
             (targeted, more precise, use for single-ticker backfills)

Outputs:  data/raw/reddit_full_v2.parquet
Progress: data/raw/reddit_backfill_progress.json

NEVER overwrites: data/raw/merged_with_sentiment_full.parquet
"""

import argparse
import json
import logging
import math
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ARCTIC_SHIFT_BASE = "https://arctic-shift.photon-reddit.com"
PULLPUSH_BASE     = "https://api.pullpush.io"

REGISTRY_PATH = "config/ticker_registry.json"

SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "options",
    "valueinvesting",
]

START_DATE = "2019-01-01"
END_DATE   = datetime.now(timezone.utc).strftime("%Y-%m-%d")

FIELDS = "author,selftext,title,score,num_comments,created_utc,subreddit,id"

BATCH_DAYS     = 3    # days per API call
DELAY_SEC      = 1.5  # polite delay between requests
MAX_RETRIES    = 3
LIMIT_PER_CALL = 100  # Arctic Shift max per request
SAVE_EVERY     = 50   # save progress every N windows

OUTPUT_PATH   = "data/raw/reddit_full_v2.parquet"
PROGRESS_PATH = "data/raw/reddit_backfill_progress.json"
PROTECTED_FILE = "data/raw/merged_with_sentiment_full.parquet"

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Ticker registry
# ─────────────────────────────────────────────────────────────────────────────

def load_registry() -> dict:
    """Load ticker registry from config/ticker_registry.json."""
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def get_active_tickers(registry: dict, tier: str | None = None) -> list[str]:
    """
    Return active ticker symbols from registry.
    tier: 'trade', 'watch', or None (all active).
    """
    return [
        symbol
        for symbol, cfg in registry["tickers"].items()
        if cfg.get("active", True)
        and (tier is None or cfg.get("tier") == tier)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Progress tracking
# ─────────────────────────────────────────────────────────────────────────────

def load_progress() -> dict:
    """Load progress from JSON checkpoint file."""
    if Path(PROGRESS_PATH).exists():
        with open(PROGRESS_PATH, "r") as f:
            return json.load(f)
    return {
        "completed_windows": [],
        "total_records": 0,
        "last_updated": None,
        "mode": None,
        "tickers_covered": {},
    }


def save_progress(progress: dict) -> None:
    """Persist progress checkpoint to disk."""
    progress["last_updated"] = datetime.now(timezone.utc).isoformat()
    Path(PROGRESS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


def window_key(subreddit: str, after: str, before: str) -> str:
    return f"{subreddit}_{after}_{before}"


def ticker_window_key(ticker: str, after: str, before: str) -> str:
    return f"ticker_{ticker}_{after}_{before}"


# ─────────────────────────────────────────────────────────────────────────────
# Arctic Shift fetcher
# ─────────────────────────────────────────────────────────────────────────────

def fetch_arctic_shift(
    subreddit: str,
    after: str,
    before: str,
    limit: int = LIMIT_PER_CALL,
    query: str | None = None,
) -> list[dict]:
    """
    Fetch posts from Arctic Shift /api/posts/search.
    query: company name string for ticker-mode search (maps to ?q=).
    Falls back to PullPush on persistent failure.
    """
    params: dict = {
        "subreddit": subreddit,
        "after":     after,
        "before":    before,
        "limit":     limit,
        "fields":    FIELDS,
        "sort":      "asc",
    }
    if query:
        params["q"] = query

    url = f"{ARCTIC_SHIFT_BASE}/api/posts/search"

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=30)

            remaining = int(resp.headers.get("X-RateLimit-Remaining", 99))
            if remaining < 5:
                logger.warning("Rate limit low — sleeping 30s")
                time.sleep(30)

            if resp.status_code == 200:
                return resp.json().get("data", [])

            elif resp.status_code == 429:
                reset_wait = int(resp.headers.get("X-RateLimit-Reset", 60))
                logger.warning(
                    f"429 rate_limit subreddit={subreddit} waiting={reset_wait}s"
                )
                time.sleep(reset_wait)

            else:
                logger.warning(
                    f"arctic_shift_error status={resp.status_code} "
                    f"subreddit={subreddit} after={after}"
                )
                break

        except Exception as e:
            logger.warning(
                f"arctic_shift_failed attempt={attempt+1}/{MAX_RETRIES}: {e}"
            )
            time.sleep(5 * (attempt + 1))

    logger.info(f"Falling back to PullPush: {subreddit} {after}→{before}")
    return fetch_pullpush(subreddit, after, before, limit, query)


# ─────────────────────────────────────────────────────────────────────────────
# PullPush fallback
# ─────────────────────────────────────────────────────────────────────────────

def fetch_pullpush(
    subreddit: str,
    after: str,
    before: str,
    limit: int = LIMIT_PER_CALL,
    query: str | None = None,
) -> list[dict]:
    """PullPush /reddit/search/submission/ fallback."""
    after_ts  = int(datetime.strptime(after,  "%Y-%m-%d").replace(
        tzinfo=timezone.utc).timestamp())
    before_ts = int(datetime.strptime(before, "%Y-%m-%d").replace(
        tzinfo=timezone.utc).timestamp())

    params: dict = {
        "subreddit": subreddit,
        "after":     after_ts,
        "before":    before_ts,
        "size":      min(limit, 100),
        "sort":      "asc",
    }
    if query:
        params["q"] = query

    try:
        resp = requests.get(
            f"{PULLPUSH_BASE}/reddit/search/submission/",
            params=params,
            timeout=30,
        )
        if resp.status_code == 200:
            raw = resp.json().get("data", [])
            return [
                {
                    "id":           p.get("id", ""),
                    "author":       p.get("author", ""),
                    "selftext":     p.get("selftext", ""),
                    "title":        p.get("title", ""),
                    "score":        p.get("score", 0),
                    "num_comments": p.get("num_comments", 0),
                    "created_utc":  int(p.get("created_utc", 0)),
                    "subreddit":    p.get("subreddit", subreddit),
                }
                for p in raw
            ]
        else:
            logger.error(
                f"pullpush_error status={resp.status_code} subreddit={subreddit}"
            )
    except Exception as e:
        logger.error(f"pullpush_failed: {e}")

    return []


# ─────────────────────────────────────────────────────────────────────────────
# Date window generator
# ─────────────────────────────────────────────────────────────────────────────

def date_windows(
    start: str,
    end: str,
    days: int = BATCH_DAYS,
) -> list[tuple[str, str]]:
    """Generate (after, before) date pairs each `days` wide."""
    windows = []
    current = datetime.strptime(start, "%Y-%m-%d")
    end_dt  = datetime.strptime(end,   "%Y-%m-%d")
    while current < end_dt:
        window_end = min(current + timedelta(days=days), end_dt)
        windows.append((
            current.strftime("%Y-%m-%d"),
            window_end.strftime("%Y-%m-%d"),
        ))
        current = window_end
    return windows


# ─────────────────────────────────────────────────────────────────────────────
# Record normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalise_record(p: dict, subreddit: str) -> dict:
    """Convert a raw API record to the canonical output schema."""
    raw_body   = p.get("selftext", "") or ""
    raw_author = p.get("author",   "") or ""

    body   = "" if raw_body   in ("[removed]", "[deleted]") else raw_body.strip()
    author = "deleted" if raw_author in ("[deleted]", "") else raw_author.strip()

    created_utc = int(p.get("created_utc", 0))
    dt = datetime.fromtimestamp(created_utc, tz=timezone.utc)

    return {
        "id":           str(p.get("id", "")),
        "subreddit":    p.get("subreddit", subreddit),
        "author":       author,
        "title":        (p.get("title", "") or "").strip(),
        "selftext":     body,
        "score":        int(p.get("score", 0)),
        "num_comments": int(p.get("num_comments", 0)),
        "created_utc":  created_utc,
        "timestamp":    dt,
        "date":         dt.strftime("%Y-%m-%d"),
        "year":         dt.year,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Estimate and warn
# ─────────────────────────────────────────────────────────────────────────────

def print_estimate(
    start: str,
    end: str,
    subreddits: list[str],
    mode: str,
    registry: dict | None = None,
) -> None:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")
    total_days   = (end_dt - start_dt).days
    windows_each = math.ceil(total_days / BATCH_DAYS)

    if mode == "ticker":
        n_tickers = len(get_active_tickers(registry)) if registry else "?"
        n_tickers_val = len(get_active_tickers(registry)) if registry else 38
        avg_aliases   = 2
        avg_subs      = 3
        total_calls   = windows_each * n_tickers_val * avg_aliases * avg_subs
        est_hours     = (total_calls * DELAY_SEC) / 3600
        est_records   = "varies (targeted, fewer irrelevant posts)"
    else:
        total_calls = windows_each * len(subreddits)
        est_hours   = (total_calls * DELAY_SEC) / 3600
        est_records = "1–2 million posts"
        n_tickers   = "N/A (subreddit mode)"

    print()
    print("=" * 60)
    print(f"Estimating backfill scope — mode={mode}:")
    print(f"  Subreddits:        {len(subreddits) if mode == 'subreddit' else 'per-ticker'}")
    print(f"  Date range:        {start} → {end} ({total_days} days)")
    print(f"  Windows each:      ~{windows_each} ({BATCH_DAYS} days each)")
    print(f"  Total API calls:   ~{total_calls:,}")
    print(f"  At {DELAY_SEC}s/call:     ~{est_hours:.1f} hours")
    print(f"  Estimated records: {est_records}")
    if mode == "ticker":
        print(f"  Tickers:           {n_tickers}")
    print()
    print(f"  Note: ~{total_calls:,} API calls to Arctic Shift.")
    print("  Be considerate — run during off-peak hours.")
    print("  The script can be interrupted and resumed safely.")
    print("=" * 60)
    print()
    try:
        input("Press ENTER to continue or Ctrl+C to cancel... ")
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# Main collection loop
# ─────────────────────────────────────────────────────────────────────────────

def run_backfill(
    start: str,
    end: str,
    subreddits: list[str],
    mode: str = "subreddit",
    resume: bool = False,
    test_mode: bool = False,
    ticker_filter: str | None = None,
    registry: dict | None = None,
) -> None:
    """
    Main backfill loop.

    subreddit mode: iterates (subreddit × date-window), no text filter.
    ticker mode:    iterates (ticker × date-window), searches each
                    company-name alias per subreddit using ?q=.
    ticker_filter:  run ticker mode for a single ticker (used by --add-ticker).
    """
    if test_mode:
        start      = "2024-01-01"
        end        = "2024-01-08"
        subreddits = ["wallstreetbets"]
        mode       = "subreddit"  # test always uses fast subreddit mode
        logger.info("TEST MODE: fetching Jan 1–7 2024, wallstreetbets only")

    progress = load_progress() if resume else {
        "completed_windows": [],
        "total_records": 0,
        "last_updated": None,
        "mode": mode,
        "tickers_covered": {},
    }
    progress["mode"] = mode
    completed: set[str] = set(progress["completed_windows"])
    all_records: list[dict] = []

    # ── Build work list ────────────────────────────────────────────────────────
    if mode == "ticker":
        reg = registry or load_registry()
        symbols = (
            [ticker_filter]
            if ticker_filter
            else get_active_tickers(reg)
        )
        # Enforce added_date floor per ticker
        work: list[tuple] = []
        for sym in symbols:
            cfg = reg["tickers"].get(sym, {})
            added = cfg.get("added_date", START_DATE)
            effective_start = max(start, added)  # YYYY-MM-DD string comparison
            for (after, before) in date_windows(effective_start, end):
                work.append((sym, after, before))
    else:
        work = [
            (sub, after, before)
            for sub in subreddits
            for (after, before) in date_windows(start, end)
        ]

    total_windows = len(work)
    start_time    = time.time()
    windows_done  = 0

    logger.info(
        f"Total windows to process: {total_windows} (mode={mode})"
    )

    for item in work:
        if mode == "ticker":
            ticker, after, before = item
            key = ticker_window_key(ticker, after, before)
        else:
            subreddit, after, before = item
            key = window_key(subreddit, after, before)

        if key in completed:
            windows_done += 1
            continue

        # ── Fetch ──────────────────────────────────────────────────────────────
        if mode == "ticker":
            raw_records = _fetch_ticker_window(
                ticker, after, before, reg
            )
            display_label = ticker
        else:
            raw_records = fetch_arctic_shift(subreddit, after, before)
            # 0-record retry: split into 1-day sub-windows
            if len(raw_records) == 0:
                window_days = (
                    datetime.strptime(before, "%Y-%m-%d") -
                    datetime.strptime(after,  "%Y-%m-%d")
                ).days
                if window_days > 1:
                    logger.info(
                        f"0 records for {key} — splitting into daily windows"
                    )
                    for (sa, sb) in date_windows(after, before, days=1):
                        raw_records.extend(
                            fetch_arctic_shift(subreddit, sa, sb)
                        )
                        time.sleep(DELAY_SEC)
            display_label = subreddit

        normalised = [normalise_record(r, r.get("subreddit", "")) for r in raw_records]
        all_records.extend(normalised)

        completed.add(key)
        progress["completed_windows"] = list(completed)
        progress["total_records"] = (
            progress.get("total_records", 0) + len(normalised)
        )
        windows_done += 1

        # ── Progress report every 10 windows ──────────────────────────────────
        if windows_done % 10 == 0:
            elapsed = time.time() - start_time
            rate    = windows_done / elapsed if elapsed > 0 else 0.001
            remaining = total_windows - windows_done
            eta_sec   = remaining / rate
            eta_h     = int(eta_sec // 3600)
            eta_m     = int((eta_sec % 3600) // 60)
            logger.info(
                f"{display_label} | {after} → {before} | "
                f"fetched={len(normalised)} total={progress['total_records']:,} "
                f"elapsed={int(elapsed//60)}m{int(elapsed%60)}s "
                f"eta={eta_h}h{eta_m}m"
            )

        # ── Checkpoint save every SAVE_EVERY windows ───────────────────────────
        if windows_done % SAVE_EVERY == 0:
            save_progress(progress)
            if all_records:
                _checkpoint_save(all_records, test_mode)

        time.sleep(DELAY_SEC)

    save_progress(progress)

    if test_mode:
        _print_test_results(all_records)
        return

    _final_save(all_records, progress)


def _fetch_ticker_window(
    ticker: str,
    after: str,
    before: str,
    registry: dict,
) -> list[dict]:
    """
    Fetch posts for a single ticker over a date window by searching
    each company-name alias across the ticker's subreddits.
    Deduplicates by post ID before returning.
    """
    cfg       = registry["tickers"].get(ticker, {})
    aliases   = cfg.get("company_names", [ticker])
    subs      = cfg.get("subreddits", SUBREDDITS)

    seen_ids: set[str] = set()
    records:  list[dict] = []

    for alias in aliases:
        for sub in subs:
            batch = fetch_arctic_shift(sub, after, before, query=alias)
            for r in batch:
                r_id = r.get("id", "")
                if r_id and r_id not in seen_ids:
                    seen_ids.add(r_id)
                    records.append(r)
            time.sleep(DELAY_SEC)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Save helpers
# ─────────────────────────────────────────────────────────────────────────────

def _checkpoint_save(records: list[dict], test_mode: bool) -> None:
    """Interim save to avoid data loss on interruption."""
    if test_mode or not records:
        return
    df = pd.DataFrame(records).drop_duplicates(subset=["id"])
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    if Path(OUTPUT_PATH).exists():
        existing = pd.read_parquet(OUTPUT_PATH)
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates(
            subset=["id"]
        )
    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Checkpoint saved: {len(df):,} unique records → {OUTPUT_PATH}")


def _final_save(records: list[dict], progress: dict) -> None:
    """Deduplicate, merge with any checkpoint, save final parquet."""
    if not records:
        logger.warning("No records collected — nothing to save.")
        return

    df = pd.DataFrame(records)
    if Path(OUTPUT_PATH).exists():
        existing = pd.read_parquet(OUTPUT_PATH)
        df = pd.concat([existing, df], ignore_index=True)

    df = df.drop_duplicates(subset=["id"])
    df = df.sort_values("created_utc").reset_index(drop=True)

    df["score"]        = df["score"].astype(int)
    df["num_comments"] = df["num_comments"].astype(int)
    df["created_utc"]  = df["created_utc"].astype(int)
    df["year"]         = df["year"].astype(int)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    has_body    = (df["selftext"].str.len() > 0).sum()
    body_pct    = has_body / len(df) * 100
    unique_auth = df["author"].nunique()
    date_min    = df["date"].min()
    date_max    = df["date"].max()
    size_mb     = Path(OUTPUT_PATH).stat().st_size / (1024 ** 2)

    print()
    print("=" * 42)
    print("BACKFILL COMPLETE")
    print("=" * 42)
    print(f"Total records:    {len(df):,}")
    print(f"Unique authors:   {unique_auth:,}")
    print(f"With body text:   {has_body:,} ({body_pct:.1f}%)")
    print(f"Date range:       {date_min} → {date_max}")
    print(f"Subreddits:       {df['subreddit'].nunique()}")
    print(f"Output:           {OUTPUT_PATH}")
    print(f"Size:             ~{size_mb:.0f}MB")
    print("=" * 42)


# ─────────────────────────────────────────────────────────────────────────────
# Test mode output
# ─────────────────────────────────────────────────────────────────────────────

def _print_test_results(records: list[dict]) -> None:
    """Print first 3 records and validate required fields."""
    print()
    print("=" * 50)
    print("TEST MODE RESULTS")
    print("=" * 50)

    if not records:
        print("ERROR: 0 records returned. Check API connectivity.")
        return

    seen: set[str] = set()
    unique: list[dict] = []
    for r in records:
        if r["id"] not in seen:
            seen.add(r["id"])
            unique.append(r)

    print(f"Total fetched: {len(unique)} unique records")
    print()

    for i, r in enumerate(unique[:3], start=1):
        author_ok   = "✓" if r["author"] and r["author"] != "deleted" else "✗ EMPTY"
        selftext_ok = "✓" if r["selftext"] else "(no body — normal)"
        print(f"Record {i}:")
        print(f"  id:           {r['id']}")
        print(f"  author:       {r['author']}  {author_ok}")
        print(f"  title:        {r['title'][:60]}")
        print(f"  selftext:     {r['selftext'][:80] if r['selftext'] else '(empty)'}  {selftext_ok}")
        print(f"  score:        {r['score']}")
        print(f"  num_comments: {r['num_comments']}")
        print(f"  date:         {r['date']}")
        print()

    total    = len(unique)
    has_auth = sum(1 for r in unique if r["author"] != "deleted")
    has_body = sum(1 for r in unique if r["selftext"])
    print(f"Author populated:    {has_auth}/{total} ({has_auth/total*100:.1f}%)")
    print(f"Body text present:   {has_body}/{total} ({has_body/total*100:.1f}%)")
    print()
    print("NOTE: Test mode — no file saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Per-ticker coverage scan (for --status)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ticker_coverage(registry: dict) -> dict[str, dict]:
    """
    Scan reddit_full_v2.parquet to count posts mentioning each ticker.
    Matches company_names in title + selftext (case-insensitive).
    Returns {ticker: {post_count, first_date, last_date}}.
    """
    if not Path(OUTPUT_PATH).exists():
        return {}

    logger.info("Scanning parquet for per-ticker coverage (may take ~30s)...")
    df = pd.read_parquet(OUTPUT_PATH, columns=["id", "title", "selftext", "date"])
    text = (
        df["title"].fillna("") + " " + df["selftext"].fillna("")
    ).str.lower()

    coverage: dict[str, dict] = {}
    for symbol, cfg in registry.get("tickers", {}).items():
        if not cfg.get("active", True):
            continue
        names   = cfg.get("company_names", [symbol])
        pattern = "|".join(re.escape(n.lower()) for n in names)
        mask    = text.str.contains(pattern, regex=True, na=False)
        matched = df[mask]
        if len(matched) > 0:
            coverage[symbol] = {
                "post_count":       int(len(matched)),
                "first_date":       str(matched["date"].min()),
                "last_date":        str(matched["date"].max()),
                "backfill_complete": True,
            }
        else:
            coverage[symbol] = {
                "post_count":       0,
                "first_date":       None,
                "last_date":        None,
                "backfill_complete": False,
            }

    return coverage


# ─────────────────────────────────────────────────────────────────────────────
# Status report
# ─────────────────────────────────────────────────────────────────────────────

def print_status() -> None:
    """Print backfill progress and per-ticker coverage."""
    progress = load_progress()

    if not progress["completed_windows"]:
        print("No backfill progress found. Run without --status to start.")
        return

    completed = progress["completed_windows"]
    by_sub: dict[str, int] = {}
    for key in completed:
        if key.startswith("ticker_"):
            continue
        parts = key.split("_", 1)
        sub = parts[0] if parts else "unknown"
        by_sub[sub] = by_sub.get(sub, 0) + 1

    # Expected total
    total_expected = (
        math.ceil(
            (
                datetime.strptime(END_DATE, "%Y-%m-%d") -
                datetime.strptime(START_DATE, "%Y-%m-%d")
            ).days / BATCH_DAYS
        )
        * len(SUBREDDITS)
    )
    done_count  = len(completed)
    pct_done    = done_count / total_expected * 100 if total_expected else 0
    total_recs  = progress.get("total_records", 0)

    print()
    print("=" * 50)
    print(f"BACKFILL STATUS — {END_DATE}")
    print("=" * 50)
    print()
    print("Overall:")
    print(f"  Total windows:    {done_count:,} / ~{total_expected:,} ({pct_done:.0f}%)")
    print(f"  Total records:    {total_recs:,}")
    print(f"  Mode:             {progress.get('mode', 'unknown')}")
    if Path(OUTPUT_PATH).exists():
        size_mb = Path(OUTPUT_PATH).stat().st_size / (1024 ** 2)
        print(f"  Output file:      {OUTPUT_PATH} ({size_mb:.0f}MB)")
    else:
        print(f"  Output file:      not yet created")

    if by_sub:
        print()
        print("By subreddit:")
        for sub in SUBREDDITS:
            count  = by_sub.get(sub, 0)
            status = "✓" if count > 0 else "·"
            print(f"  {sub:<22} {count:>5} windows  {status}")

    # Per-ticker coverage scan
    if Path(REGISTRY_PATH).exists() and Path(OUTPUT_PATH).exists():
        print()
        try:
            registry = load_registry()
            coverage = _compute_ticker_coverage(registry)

            # Sort: trade tickers first, then watch, alphabetical within tier
            def _sort_key(sym: str) -> tuple[int, str]:
                tier = registry["tickers"].get(sym, {}).get("tier", "watch")
                return (0 if tier == "trade" else 1, sym)

            sorted_tickers = sorted(coverage.keys(), key=_sort_key)

            print("Coverage check (posts mentioning ticker):")
            print(f"  {'Ticker':<8}  {'Posts':>7}  {'First date':>10}  {'Last date':>10}  Status")
            print("  " + "─" * 52)
            for sym in sorted_tickers:
                cov = coverage[sym]
                if cov["post_count"] > 0:
                    status = "✓"
                    fd = cov["first_date"] or "—"
                    ld = cov["last_date"]  or "—"
                else:
                    status = "·  (no data)"
                    fd = ld = "—"
                print(
                    f"  {sym:<8}  {cov['post_count']:>7,}  {fd:>10}  {ld:>10}  {status}"
                )

            # Save coverage snapshot into progress for later reference
            progress["tickers_covered"] = coverage
            save_progress(progress)

        except Exception as e:
            print(f"  Coverage scan failed: {e}")

    print()
    print("=" * 50)


# ─────────────────────────────────────────────────────────────────────────────
# Add ticker command
# ─────────────────────────────────────────────────────────────────────────────

def add_ticker_to_registry(
    symbol: str,
    company_names: list[str],
    tier: str,
    start_date: str,
    subreddits: list[str] | None = None,
) -> None:
    """
    Add a new ticker to config/ticker_registry.json and the
    appropriate config/tickers_{tier}.txt file.
    Prints a summary of what changed.
    """
    registry = load_registry()

    already_exists = symbol in registry["tickers"]
    if already_exists:
        logger.warning(f"{symbol} already in registry — updating entry")

    registry["tickers"][symbol] = {
        "company_names": company_names,
        "subreddits":    subreddits or ["wallstreetbets", "stocks", "investing", "options"],
        "active":        True,
        "added_date":    start_date,
        "tier":          tier,
    }
    registry["last_updated"] = END_DATE

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    action = "Updated" if already_exists else "Added"
    logger.info(f"{action} {symbol} in {REGISTRY_PATH}")

    # Sync tickers_*.txt
    txt_path = (
        "config/tickers_trade.txt" if tier == "trade"
        else "config/tickers_watch.txt"
    )
    with open(txt_path, "r") as f:
        existing_lines = f.read().splitlines()

    tickers_in_file = {
        line.strip() for line in existing_lines
        if line.strip() and not line.strip().startswith("#")
    }
    if symbol not in tickers_in_file:
        with open(txt_path, "a") as f:
            f.write(f"{symbol}\n")
        logger.info(f"Added {symbol} to {txt_path}")
    else:
        logger.info(f"{symbol} already in {txt_path} — skipped")

    print()
    print(f"  {'Updated' if already_exists else 'Added'}  {symbol}")
    print(f"  company_names : {company_names}")
    print(f"  tier          : {tier}")
    print(f"  added_date    : {start_date}")
    print(f"  registry      : {REGISTRY_PATH}")
    print(f"  ticker file   : {txt_path}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Safety guard
# ─────────────────────────────────────────────────────────────────────────────

def _safety_check() -> None:
    """Abort if OUTPUT_PATH was somehow set to the protected file."""
    if OUTPUT_PATH == PROTECTED_FILE:
        logger.critical(
            f"OUTPUT_PATH must never equal {PROTECTED_FILE}. Aborting."
        )
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reddit full backfill — Arctic Shift + PullPush fallback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/backfill_reddit_full.py                         # subreddit mode, 2019-present
  python scripts/backfill_reddit_full.py --mode ticker           # per-ticker search mode
  python scripts/backfill_reddit_full.py --subreddit wallstreetbets
  python scripts/backfill_reddit_full.py --start 2024-01-01 --end 2024-12-31
  python scripts/backfill_reddit_full.py --status
  python scripts/backfill_reddit_full.py --resume
  python scripts/backfill_reddit_full.py --test

  # Add a new ticker and immediately backfill it
  python scripts/backfill_reddit_full.py \\
    --add-ticker PLTR \\
    --company-names "Palantir" "PLTR" \\
    --tier trade \\
    --start 2020-09-30
        """,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="subreddit",
        choices=["subreddit", "ticker"],
        help=(
            "subreddit = fetch all posts from subreddits (default, fast); "
            "ticker = search per company-name alias (precise, slower)"
        ),
    )
    parser.add_argument(
        "--subreddit",
        type=str,
        default=None,
        choices=SUBREDDITS,
        help="Fetch only this subreddit (subreddit mode only, default: all 5)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=START_DATE,
        help=f"Start date YYYY-MM-DD (default: {START_DATE})",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=END_DATE,
        help=f"End date YYYY-MM-DD (default: today = {END_DATE})",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print progress + per-ticker coverage and exit",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint (skips completed windows)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: Jan 1–7 2024, wallstreetbets only, no file saved",
    )
    # Add-ticker arguments
    parser.add_argument(
        "--add-ticker",
        type=str,
        default=None,
        metavar="SYMBOL",
        help="Add a new ticker to the registry and start its backfill",
    )
    parser.add_argument(
        "--company-names",
        nargs="+",
        default=None,
        metavar="NAME",
        help='Company name aliases for --add-ticker (e.g. "Palantir" "PLTR")',
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="watch",
        choices=["trade", "watch"],
        help="Registry tier for --add-ticker (default: watch)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _safety_check()
    args = parse_args()

    if args.status:
        print_status()
        return

    # ── Add-ticker flow ────────────────────────────────────────────────────────
    if args.add_ticker:
        symbol       = args.add_ticker.strip().upper()
        company_names = args.company_names or [symbol]
        start_date   = args.start  # --start doubles as IPO/added_date

        add_ticker_to_registry(
            symbol        = symbol,
            company_names = company_names,
            tier          = args.tier,
            start_date    = start_date,
        )

        logger.info(
            f"Starting targeted backfill for {symbol} "
            f"from {start_date} to {args.end}"
        )
        registry = load_registry()
        run_backfill(
            start         = start_date,
            end           = args.end,
            subreddits    = SUBREDDITS,
            mode          = "ticker",
            resume        = False,
            test_mode     = False,
            ticker_filter = symbol,
            registry      = registry,
        )
        return

    # ── Normal backfill ────────────────────────────────────────────────────────
    subreddits = [args.subreddit] if args.subreddit else SUBREDDITS

    registry = None
    if args.mode == "ticker":
        registry = load_registry()

    if not args.test and not args.resume:
        print_estimate(args.start, args.end, subreddits, args.mode, registry)

    run_backfill(
        start      = args.start,
        end        = args.end,
        subreddits = subreddits,
        mode       = args.mode,
        resume     = args.resume,
        test_mode  = args.test,
        registry   = registry,
    )


if __name__ == "__main__":
    main()
