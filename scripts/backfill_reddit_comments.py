"""
backfill_reddit_comments.py
===========================
Fetches historical Reddit comments from Arctic Shift API
(with PullPush fallback) for 5 financial subreddits from
2019-01-01 to today.

Comments are linked to posts via link_id (strip "t3_" prefix).
Ticker assignment happens downstream by joining:
  comments.link_id → posts.id → posts.ticker

Outputs:  data/raw/reddit_comments_v2.parquet
Progress: data/raw/reddit_comments_progress.json

NEVER overwrites:
  data/raw/merged_with_sentiment_full.parquet
  data/raw/reddit_full_v2.parquet
"""

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

# ─────────────────────────────────────────────────────────────────────────────
# Constants and config
# ─────────────────────────────────────────────────────────────────────────────

ARCTIC_SHIFT_BASE = "https://arctic-shift.photon-reddit.com"
PULLPUSH_BASE     = "https://api.pullpush.io"

SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "options",
    "valueinvesting",
]

START_DATE = "2019-01-01"
END_DATE   = datetime.now(timezone.utc).strftime("%Y-%m-%d")

COMMENT_FIELDS = "author,body,score,created_utc,subreddit,link_id,parent_id,id"

BATCH_DAYS     = 3    # days per API call
DELAY_SEC      = 1.5  # polite delay between requests
MAX_RETRIES    = 3
LIMIT_PER_CALL = 100  # Arctic Shift max per request
SAVE_EVERY     = 50   # save progress every N windows

OUTPUT_PATH   = "data/raw/reddit_comments_v2.parquet"
PROGRESS_PATH = "data/raw/reddit_comments_progress.json"

PROTECTED_FILES = {
    "data/raw/merged_with_sentiment_full.parquet",
    "data/raw/reddit_full_v2.parquet",
}

# ─────────────────────────────────────────────────────────────────────────────
# Bot filter
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_BOTS = {
    "VisualMod",
    "AutoModerator",
    "WSBVoteBot",
    "BotDefense",
    "anti-gif-bot",
    "RepostSleuthBot",
    "RemindMeBot",
    "SaveVideo",
    "transcribersofreddit",
    "sneakpeekbot",
}

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
        "total_filtered": 0,
        "last_updated": None,
    }


def save_progress(progress: dict) -> None:
    """Persist progress checkpoint to disk."""
    progress["last_updated"] = datetime.now(timezone.utc).isoformat()
    Path(PROGRESS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


def window_key(subreddit: str, after: str, before: str) -> str:
    """Unique key for a (subreddit, date-window) pair."""
    return f"{subreddit}_{after}_{before}"


# ─────────────────────────────────────────────────────────────────────────────
# Arctic Shift comment fetcher
# ─────────────────────────────────────────────────────────────────────────────

def fetch_arctic_shift(
    subreddit: str,
    after: str,
    before: str,
    limit: int = LIMIT_PER_CALL,
) -> list[dict]:
    """
    Fetch comments from Arctic Shift /api/comments/search.
    Falls back to PullPush on persistent failure.
    Returns list of raw comment dicts (no score filtering — done locally).
    """
    url = (
        f"{ARCTIC_SHIFT_BASE}/api/comments/search"
        f"?subreddit={subreddit}"
        f"&after={after}"
        f"&before={before}"
        f"&limit={limit}"
        f"&fields={COMMENT_FIELDS}"
        f"&sort=asc"
    )

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=30)

            # Honour rate-limit headers proactively
            remaining = int(resp.headers.get("X-RateLimit-Remaining", 99))
            if remaining < 5:
                logger.warning("Rate limit low — sleeping 30s")
                time.sleep(30)

            if resp.status_code == 200:
                return resp.json().get("data", [])

            elif resp.status_code == 429:
                reset_wait = int(resp.headers.get("X-RateLimit-Reset", 60))
                logger.warning(
                    f"429 rate_limit subreddit={subreddit} "
                    f"waiting={reset_wait}s"
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
    return fetch_pullpush(subreddit, after, before, limit)


# ─────────────────────────────────────────────────────────────────────────────
# PullPush comment fallback
# ─────────────────────────────────────────────────────────────────────────────

def fetch_pullpush(
    subreddit: str,
    after: str,
    before: str,
    limit: int = LIMIT_PER_CALL,
) -> list[dict]:
    """PullPush /reddit/search/comment/ fallback — normalises to Arctic Shift schema."""
    after_ts  = int(datetime.strptime(after,  "%Y-%m-%d").replace(
        tzinfo=timezone.utc).timestamp())
    before_ts = int(datetime.strptime(before, "%Y-%m-%d").replace(
        tzinfo=timezone.utc).timestamp())

    url = (
        f"{PULLPUSH_BASE}/reddit/search/comment/"
        f"?subreddit={subreddit}"
        f"&after={after_ts}"
        f"&before={before_ts}"
        f"&size={min(limit, 100)}"
        f"&sort=asc"
    )

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            raw = resp.json().get("data", [])
            return [
                {
                    "id":          p.get("id", ""),
                    "author":      p.get("author", ""),
                    "body":        p.get("body", ""),
                    "score":       p.get("score", 0),
                    "created_utc": int(p.get("created_utc", 0)),
                    "subreddit":   p.get("subreddit", subreddit),
                    "link_id":     p.get("link_id", ""),
                    "parent_id":   p.get("parent_id", ""),
                }
                for p in raw
            ]
        else:
            logger.error(
                f"pullpush_error status={resp.status_code} "
                f"subreddit={subreddit}"
            )
    except Exception as e:
        logger.error(f"pullpush_failed: {e}")

    return []


# ─────────────────────────────────────────────────────────────────────────────
# Date window generator (identical to post backfill)
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
# Record normalisation with bot/empty filter
# ─────────────────────────────────────────────────────────────────────────────

def normalise_record(p: dict, subreddit: str) -> dict | None:
    """
    Convert a raw API comment to the canonical output schema.
    Returns None for bots and empty/removed bodies — caller filters these.
    """
    raw_author = (p.get("author", "") or "").strip()
    raw_body   = (p.get("body",   "") or "").strip()

    # Bot filter
    if raw_author in KNOWN_BOTS:
        return None

    # Empty / deleted body filter
    if not raw_body or raw_body in ("[removed]", "[deleted]"):
        return None

    author = "deleted" if raw_author in ("[deleted]", "") else raw_author

    # Strip "t3_" prefix from link_id (post reference)
    link_id   = (p.get("link_id",  "") or "").replace("t3_", "")
    parent_id = (p.get("parent_id", "") or "")

    created_utc = int(p.get("created_utc", 0))
    dt = datetime.fromtimestamp(created_utc, tz=timezone.utc)

    return {
        "id":          str(p.get("id", "")),
        "link_id":     link_id,
        "parent_id":   parent_id,
        "subreddit":   p.get("subreddit", subreddit),
        "author":      author,
        "body":        raw_body,
        "score":       int(p.get("score", 0)),
        "created_utc": created_utc,
        "timestamp":   dt,
        "date":        dt.strftime("%Y-%m-%d"),
        "year":        dt.year,
    }


def normalise_batch(
    raw: list[dict],
    subreddit: str,
) -> tuple[list[dict], int, int]:
    """
    Normalise a batch of raw comment dicts.
    Returns (valid_records, bots_filtered, empty_filtered).
    """
    valid: list[dict] = []
    bots_filtered  = 0
    empty_filtered = 0

    for p in raw:
        raw_author = (p.get("author", "") or "").strip()
        raw_body   = (p.get("body",   "") or "").strip()

        if raw_author in KNOWN_BOTS:
            bots_filtered += 1
            continue
        if not raw_body or raw_body in ("[removed]", "[deleted]"):
            empty_filtered += 1
            continue

        rec = normalise_record(p, subreddit)
        if rec is not None:
            valid.append(rec)

    return valid, bots_filtered, empty_filtered


# ─────────────────────────────────────────────────────────────────────────────
# Estimate and warn
# ─────────────────────────────────────────────────────────────────────────────

def print_estimate(
    start: str,
    end: str,
    subreddits: list[str],
    strategy: str,
    yes_flag: bool = False,
) -> None:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")
    total_days    = (end_dt - start_dt).days
    windows_each  = math.ceil(total_days / BATCH_DAYS)
    total_windows = windows_each * len(subreddits)
    est_seconds   = total_windows * DELAY_SEC
    est_hours     = est_seconds / 3600

    if strategy == "top":
        est_records = "500K–1M comments"
        est_time    = f"~{est_hours:.1f} hours (score≥5 filter, ~3–4h typical)"
    else:
        est_records = "5–10 million comments"
        est_time    = f"~{est_hours:.1f} hours base + pagination overhead (~15–20h typical)"

    print()
    print("=" * 60)
    print("Estimating comment backfill scope:")
    print(f"  Strategy:          {strategy} ({'all comments' if strategy == 'full' else 'score >= 5 only'})")
    print(f"  Subreddits:        {len(subreddits)}")
    print(f"  Date range:        {start} → {end} ({total_days} days)")
    print(f"  Windows each:      ~{windows_each} ({BATCH_DAYS} days each)")
    print(f"  Total API calls:   ~{total_windows} (more due to pagination)")
    print(f"  Estimated time:    {est_time}")
    print(f"  Estimated records: {est_records}")
    print()
    print(f"  Tip: Run --strategy top first (much faster).")
    print(f"  Note: ~{total_windows:,} base API calls to Arctic Shift.")
    print("  Be considerate — run during off-peak hours.")
    print("  The script can be interrupted and resumed safely.")
    print("=" * 60)
    print()
    if yes_flag:
        return
    try:
        input("Press ENTER to continue or Ctrl+C to cancel... ")
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# Main collection loop
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_window(
    subreddit: str,
    after: str,
    before: str,
    score_min: int | None,
) -> list[dict]:
    """
    Fetch all comments for a window, paginating via 1-day sub-windows
    when a window returns exactly LIMIT_PER_CALL (possible truncation).
    score_min filter is applied locally after fetching — not sent to API.
    """
    records = fetch_arctic_shift(subreddit, after, before)

    window_days = (
        datetime.strptime(before, "%Y-%m-%d") -
        datetime.strptime(after,  "%Y-%m-%d")
    ).days

    # Pagination check uses raw count (before score filter)
    if len(records) == LIMIT_PER_CALL and window_days > 1:
        logger.info(
            f"Hit limit ({LIMIT_PER_CALL}) for {subreddit} {after}→{before} "
            f"— splitting into 1-day windows"
        )
        sub_records: list[dict] = []
        for (sa, sb) in date_windows(after, before, days=1):
            day_records = fetch_arctic_shift(subreddit, sa, sb)
            sub_records.extend(day_records)
            if len(day_records) == LIMIT_PER_CALL:
                logger.warning(
                    f"1-day window {subreddit} {sa}→{sb} still at limit "
                    f"— some comments may be missed"
                )
            time.sleep(DELAY_SEC)
        records = sub_records

    # Local score filter — applied after all records collected
    if score_min is not None:
        records = [r for r in records if r.get("score", 0) >= score_min]

    return records


def run_backfill(
    start: str,
    end: str,
    subreddits: list[str],
    strategy: str = "full",
    resume: bool = False,
    test_mode: bool = False,
) -> None:
    """
    Main backfill loop. Iterates over (subreddit × date-window),
    fetches comments, normalises with bot/empty filter, saves parquet.
    """
    score_min = 5 if strategy == "top" else None

    if test_mode:
        start      = "2024-01-01"
        end        = "2024-01-08"
        subreddits = ["wallstreetbets"]
        logger.info(
            f"TEST MODE: fetching Jan 1–7 2024, wallstreetbets only "
            f"(strategy={strategy})"
        )

    progress = load_progress() if resume else {
        "completed_windows": [],
        "total_records": 0,
        "total_filtered": 0,
        "last_updated": None,
    }
    completed: set[str] = set(progress["completed_windows"])
    all_records: list[dict] = []

    work = [
        (sub, after, before)
        for sub in subreddits
        for (after, before) in date_windows(start, end)
    ]

    total_windows  = len(work)
    start_time     = time.time()
    windows_done   = 0
    session_filtered = 0

    logger.info(f"Total windows to process: {total_windows} (strategy={strategy})")

    for subreddit, after, before in work:
        key = window_key(subreddit, after, before)

        if key in completed:
            windows_done += 1
            continue

        raw_records = _fetch_window(subreddit, after, before, score_min)

        valid, bots_n, empty_n = normalise_batch(raw_records, subreddit)
        filtered_n = bots_n + empty_n

        all_records.extend(valid)
        session_filtered += filtered_n

        completed.add(key)
        progress["completed_windows"] = list(completed)
        progress["total_records"]  = progress.get("total_records", 0)  + len(valid)
        progress["total_filtered"] = progress.get("total_filtered", 0) + filtered_n
        windows_done += 1

        # Progress report every 10 windows
        if windows_done % 10 == 0:
            elapsed = time.time() - start_time
            rate    = windows_done / elapsed if elapsed > 0 else 0.001
            remaining = total_windows - windows_done
            eta_sec   = remaining / rate
            eta_h     = int(eta_sec // 3600)
            eta_m     = int((eta_sec % 3600) // 60)
            logger.info(
                f"{subreddit} | {after} → {before} | "
                f"fetched={len(valid)} filtered={filtered_n} "
                f"total={progress['total_records']:,} "
                f"elapsed={int(elapsed//60)}m{int(elapsed%60)}s "
                f"eta={eta_h}h{eta_m}m"
            )

        # Periodic checkpoint
        if windows_done % SAVE_EVERY == 0:
            save_progress(progress)
            if all_records:
                _checkpoint_save(all_records, test_mode)

        time.sleep(DELAY_SEC)

    save_progress(progress)

    if test_mode:
        _print_test_results(all_records, session_filtered)
        return

    _final_save(all_records, progress)


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

    df["score"]       = df["score"].astype(int)
    df["created_utc"] = df["created_utc"].astype(int)
    df["year"]        = df["year"].astype(int)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    has_body    = (df["body"].str.len() > 0).sum()
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
    print(f"Filtered (total): {progress.get('total_filtered', 0):,}")
    print(f"Output:           {OUTPUT_PATH}")
    print(f"Size:             ~{size_mb:.0f}MB")
    print("=" * 42)


# ─────────────────────────────────────────────────────────────────────────────
# Test mode output
# ─────────────────────────────────────────────────────────────────────────────

def _print_test_results(records: list[dict], session_filtered: int) -> None:
    """Print first 3 records, coverage stats, and filter counts."""
    print()
    print("=" * 50)
    print("TEST MODE RESULTS")
    print("=" * 50)

    if not records:
        print("ERROR: 0 valid records returned. Check API connectivity.")
        return

    # Dedup for display
    seen: set[str] = set()
    unique: list[dict] = []
    for r in records:
        if r["id"] not in seen:
            seen.add(r["id"])
            unique.append(r)

    total = len(unique)
    print(f"Total fetched:    {total} unique comments")
    print(f"Bots filtered:    (included in session_filtered below)")
    print(f"Session filtered: {session_filtered} (bots + empty/removed)")
    print(f"Valid comments:   {total}")
    print()

    for i, r in enumerate(unique[:3], start=1):
        author_ok = "✓" if r["author"] not in ("deleted", "") else "✗ EMPTY"
        body_ok   = "✓" if r["body"] else "✗ EMPTY"
        link_ok   = "✓" if r["link_id"] and not r["link_id"].startswith("t3_") else "✗"
        print(f"Record {i}:")
        print(f"  id:        {r['id']}")
        print(f"  link_id:   {r['link_id']}  {link_ok}  (t3_ stripped)")
        print(f"  author:    {r['author']}  {author_ok}")
        print(f"  body:      {r['body'][:80]}  {body_ok}")
        print(f"  score:     {r['score']}")
        print(f"  date:      {r['date']}")
        print()

    has_auth  = sum(1 for r in unique if r["author"] not in ("deleted", ""))
    has_body  = sum(1 for r in unique if r["body"])
    has_link  = sum(1 for r in unique if r["link_id"])
    print(f"Author populated:  {has_auth}/{total} ({has_auth/total*100:.1f}%)")
    print(f"Body populated:    {has_body}/{total} ({has_body/total*100:.1f}%)")
    print(f"link_id present:   {has_link}/{total} ({has_link/total*100:.1f}%)")
    print(f"Bots excluded:     included in {session_filtered} filtered")
    print()
    print("NOTE: Test mode — no file saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Status report
# ─────────────────────────────────────────────────────────────────────────────

def print_status() -> None:
    """Print current backfill progress without running anything."""
    progress = load_progress()

    if not progress["completed_windows"]:
        print("No comment backfill progress found. Run without --status to start.")
        return

    completed = progress["completed_windows"]
    by_sub: dict[str, int] = {}
    for key in completed:
        sub = key.split("_", 1)[0]
        by_sub[sub] = by_sub.get(sub, 0) + 1

    print()
    print("=" * 48)
    print("COMMENT BACKFILL STATUS")
    print("=" * 48)
    print(f"Total windows completed: {len(completed):,}")
    print(f"Total records fetched:   {progress.get('total_records', 0):,}")
    print(f"Total filtered (bots+):  {progress.get('total_filtered', 0):,}")
    print(f"Last updated:            {progress.get('last_updated', 'never')}")
    print()
    print("By subreddit:")
    for sub, count in sorted(by_sub.items()):
        print(f"  {sub:<20} {count:>6} windows")
    print()
    if Path(OUTPUT_PATH).exists():
        size_mb = Path(OUTPUT_PATH).stat().st_size / (1024 ** 2)
        print(f"Output file: {OUTPUT_PATH} ({size_mb:.1f}MB)")
    else:
        print("Output file: not yet created")
    print("=" * 48)


# ─────────────────────────────────────────────────────────────────────────────
# Safety guard
# ─────────────────────────────────────────────────────────────────────────────

def _safety_check() -> None:
    """Abort if OUTPUT_PATH collides with a protected file."""
    if OUTPUT_PATH in PROTECTED_FILES:
        logger.critical(
            f"OUTPUT_PATH={OUTPUT_PATH!r} is a protected file. Aborting."
        )
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reddit comment backfill — Arctic Shift + PullPush fallback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/backfill_reddit_comments.py --strategy top   # Recommended first run
  python scripts/backfill_reddit_comments.py --strategy full  # All comments (~15-20h)
  python scripts/backfill_reddit_comments.py --subreddit wallstreetbets --strategy top
  python scripts/backfill_reddit_comments.py --start 2024-01-01 --end 2024-12-31
  python scripts/backfill_reddit_comments.py --status
  python scripts/backfill_reddit_comments.py --resume --strategy top
  python scripts/backfill_reddit_comments.py --test
  python scripts/backfill_reddit_comments.py --test --strategy top
        """,
    )
    parser.add_argument(
        "--subreddit",
        type=str,
        default=None,
        choices=SUBREDDITS,
        help="Fetch only this subreddit (default: all 5)",
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
        "--strategy",
        type=str,
        default="full",
        choices=["full", "top"],
        help=(
            "full = all comments (default, ~15-20h); "
            "top = score >= 5 only (~3-4h, recommended first run)"
        ),
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print progress status and exit",
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
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt (useful for scripted runs)",
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

    subreddits = [args.subreddit] if args.subreddit else SUBREDDITS

    if not args.test and not args.resume:
        print_estimate(args.start, args.end, subreddits, args.strategy, yes_flag=args.yes)

    run_backfill(
        start      = args.start,
        end        = args.end,
        subreddits = subreddits,
        strategy   = args.strategy,
        resume     = args.resume,
        test_mode  = args.test,
    )


if __name__ == "__main__":
    main()
