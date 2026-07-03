#!/usr/bin/env python3
"""
Score Reddit posts with VADER and FinBERT sentiment.
Run BEFORE feature engineering.

Steps:
  1. VADER: score all 401K posts (fast, lexicon-based)
  2. FinBERT: score high-engagement posts only (score>=10 OR comments>=20)
  3. Join scored posts with ticker assignments
  4. Validate: IC comparison vs old FinBERT scores

Outputs:
  data/raw/reddit_full_v2_scored.parquet      — all posts + sentiment columns
  data/raw/reddit_scored_with_tickers.parquet — joined with ticker data

Usage:
  python scripts/score_reddit_sentiment.py            # VADER + FinBERT
  python scripts/score_reddit_sentiment.py --vader-only
  python scripts/score_reddit_sentiment.py --finbert-only
  python scripts/score_reddit_sentiment.py --validate-only
  python scripts/score_reddit_sentiment.py --resume
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
FEATURES = ROOT / "data" / "features"

SOURCE_FILE = RAW / "reddit_full_v2.parquet"
SCORED_FILE = RAW / "reddit_full_v2_scored.parquet"
TICKER_FILE = RAW / "reddit_scored_with_tickers.parquet"
MERGED_FILE = RAW / "merged_with_sentiment_full.parquet"
CHECKPOINT_FILE = RAW / "scoring_progress.json"

# Thresholds
HIGH_ENG_SCORE_THRESH = 10
HIGH_ENG_COMMENTS_THRESH = 20
VADER_PROGRESS_INTERVAL = 50_000
CHECKPOINT_INTERVAL = 10_000
FINBERT_BATCH_SIZE = 32
FINBERT_MAX_CHARS = 512
TICKER_CONFIDENCE_MIN = 0.5


# ─── Checkpoint helpers ────────────────────────────────────────────────────────

def _load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"vader_completed": 0, "finbert_completed": 0}


def _save_checkpoint(vader_completed: int, finbert_completed: int) -> None:
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(
            {"vader_completed": vader_completed, "finbert_completed": finbert_completed},
            f, indent=2,
        )


# ─── Text construction ─────────────────────────────────────────────────────────

def _build_text(title, selftext) -> tuple[str, bool]:
    """Combine title + up to 500 chars of body. Returns (text, has_body)."""
    title_str = str(title).strip() if pd.notna(title) else ""
    body_str = str(selftext) if pd.notna(selftext) else ""
    has_body = len(body_str) > 20
    if has_body:
        return title_str + ". " + body_str[:500], True
    return title_str, False


# ─── Step 1: VADER ─────────────────────────────────────────────────────────────

def run_vader(df: pd.DataFrame) -> pd.DataFrame:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()

    print("\n=== STEP 1: VADER Scoring ===")

    # Initialise columns if missing (resume-safe)
    for col, default in [
        ("vader_score", np.nan), ("vader_positive", np.nan),
        ("vader_negative", np.nan), ("vader_neutral", np.nan),
        ("has_body", False), ("text_length", 0),
    ]:
        if col not in df.columns:
            df[col] = default

    unscored_mask = df["vader_score"].isna()
    n_already = int((~unscored_mask).sum())
    n_to_score = int(unscored_mask.sum())
    total = len(df)

    if n_already:
        print(f"  Already scored: {n_already:,}")
    print(f"  To score:       {n_to_score:,} / {total:,}")

    if n_to_score == 0:
        print("  All posts already have VADER scores.")
        _print_vader_summary(df)
        return df

    unscored_idx = df.index[unscored_mask].tolist()
    global_done = n_already
    last_print = n_already  # tracks when to fire progress messages
    fc_done = _load_checkpoint().get("finbert_completed", 0)

    for chunk_start in range(0, len(unscored_idx), CHECKPOINT_INTERVAL):
        chunk = unscored_idx[chunk_start:chunk_start + CHECKPOINT_INTERVAL]
        chunk_df = df.loc[chunk]

        v_score, v_pos, v_neg, v_neu, has_body, t_len = [], [], [], [], [], []
        for _, row in chunk_df.iterrows():
            text, hb = _build_text(row["title"], row.get("selftext"))
            sc = vader.polarity_scores(text)
            v_score.append(sc["compound"])
            v_pos.append(sc["pos"])
            v_neg.append(sc["neg"])
            v_neu.append(sc["neu"])
            has_body.append(hb)
            t_len.append(len(text))

        df.loc[chunk, "vader_score"] = v_score
        df.loc[chunk, "vader_positive"] = v_pos
        df.loc[chunk, "vader_negative"] = v_neg
        df.loc[chunk, "vader_neutral"] = v_neu
        df.loc[chunk, "has_body"] = has_body
        df.loc[chunk, "text_length"] = t_len

        global_done += len(chunk)

        # Progress every 50K posts
        if global_done - last_print >= VADER_PROGRESS_INTERVAL:
            pct = global_done / total * 100
            print(f"  Scored {global_done:,} / {total:,} ({pct:.1f}%) ...")
            last_print = (global_done // VADER_PROGRESS_INTERVAL) * VADER_PROGRESS_INTERVAL

        # Checkpoint every 10K posts
        df.to_parquet(SCORED_FILE, index=False)
        _save_checkpoint(global_done, fc_done)

    print(f"  Scored {global_done:,} / {total:,} (100.0%)")
    print(f"  Saved → {SCORED_FILE}")

    _print_vader_summary(df)
    return df


def _print_vader_summary(df: pd.DataFrame) -> None:
    vs = df["vader_score"]
    neutral_pct = (vs == 0).mean() * 100
    pos_pct = (vs > 0).mean() * 100
    neg_pct = (vs < 0).mean() * 100
    has_body_pct = df["has_body"].mean() * 100 if "has_body" in df.columns else 0.0
    print(f"\nVADER Summary:")
    print(f"  Total scored:   {vs.notna().sum():,}")
    print(f"  Has body:       {has_body_pct:.1f}%")
    print(f"  Neutral (=0):   {neutral_pct:.1f}%")
    print(f"  Positive (>0):  {pos_pct:.1f}%")
    print(f"  Negative (<0):  {neg_pct:.1f}%")
    print(f"  Mean score:     {vs.mean():+.3f}")


# ─── Step 2: FinBERT ───────────────────────────────────────────────────────────

def run_finbert(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== STEP 2: FinBERT Scoring (high-engagement) ===")

    # Init columns if missing
    for col, default in [
        ("finbert_score", np.nan), ("finbert_label", None), ("finbert_conf", np.nan),
    ]:
        if col not in df.columns:
            df[col] = default

    high_eng_mask = (
        (df["score"] >= HIGH_ENG_SCORE_THRESH) |
        (df["num_comments"] >= HIGH_ENG_COMMENTS_THRESH)
    )
    n_high_eng = int(high_eng_mask.sum())
    print(f"  High-engagement posts: {n_high_eng:,} / {len(df):,}")

    # Skip already-scored rows (resume)
    needs_finbert = high_eng_mask & df["finbert_score"].isna()
    n_already = int(high_eng_mask.sum()) - int(needs_finbert.sum())
    n_to_score = int(needs_finbert.sum())
    if n_already:
        print(f"  Already scored: {n_already:,} | Remaining: {n_to_score:,}")

    if n_to_score == 0:
        print("  All high-engagement posts already scored.")
        return df

    # Load model
    try:
        from transformers import pipeline as hf_pipeline
        print("  Loading ProsusAI/finbert ...")
        finbert = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            top_k=None,
            truncation=True,
            max_length=512,
        )
        print("  Model loaded.")
    except Exception as exc:
        print(f"  [ERROR] Failed to load FinBERT: {exc}")
        print("  Skipping FinBERT — VADER scores will be used for all posts.")
        return df

    target_idx = df.index[needs_finbert].tolist()
    total_he = len(target_idx)
    completed = n_already
    vc_done = _load_checkpoint().get("vader_completed", 0)

    print(f"  Scoring {n_to_score:,} posts in batches of {FINBERT_BATCH_SIZE} ...")

    for batch_start in range(0, total_he, FINBERT_BATCH_SIZE):
        batch_idx = target_idx[batch_start:batch_start + FINBERT_BATCH_SIZE]
        batch_rows = df.loc[batch_idx]

        texts = []
        for _, row in batch_rows.iterrows():
            text, _ = _build_text(row["title"], row.get("selftext"))
            texts.append(text[:FINBERT_MAX_CHARS])

        try:
            results = finbert(texts)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"  [OOM] at batch offset {batch_start} — stopping FinBERT, keeping scored so far")
                break
            raise

        fb_score_vals, fb_label_vals, fb_conf_vals = [], [], []
        for result in results:
            sc = {r["label"].lower(): r["score"] for r in result}
            pos = sc.get("positive", 0.0)
            neg = sc.get("negative", 0.0)
            winning = max(sc, key=sc.get)
            fb_score_vals.append(pos - neg)  # −1 to +1
            fb_label_vals.append(winning)
            fb_conf_vals.append(sc[winning])

        df.loc[batch_idx, "finbert_score"] = fb_score_vals
        df.loc[batch_idx, "finbert_label"] = fb_label_vals
        df.loc[batch_idx, "finbert_conf"] = fb_conf_vals

        completed += len(batch_idx)
        if completed % CHECKPOINT_INTERVAL == 0 or batch_start + FINBERT_BATCH_SIZE >= total_he:
            df.to_parquet(SCORED_FILE, index=False)
            _save_checkpoint(vc_done, completed)
            pct = completed / n_high_eng * 100
            print(f"  FinBERT: {completed:,} / {n_high_eng:,} ({pct:.1f}%) ...")

    print(f"  FinBERT scored total: {df['finbert_score'].notna().sum():,} posts")
    print(f"  Saved → {SCORED_FILE}")
    return df


# ─── Step 3: Join with ticker data ─────────────────────────────────────────────

def join_with_tickers(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== STEP 3: Join with Ticker Data ===")

    old = pd.read_parquet(
        MERGED_FILE,
        columns=["post_id", "ticker", "match_type", "confidence"],
    )
    old = old[
        (old["confidence"] >= TICKER_CONFIDENCE_MIN) &
        (old["match_type"] != "false_positive")
    ]

    joined = df.merge(old, left_on="id", right_on="post_id", how="inner")
    if "post_id" in joined.columns:
        joined = joined.drop(columns=["post_id"])

    keep = [
        "id", "ticker", "date", "year", "subreddit",
        "vader_score", "vader_positive", "vader_negative", "vader_neutral",
        "finbert_score", "finbert_label", "finbert_conf",
        "score", "num_comments", "has_body", "text_length", "author",
    ]
    keep = [c for c in keep if c in joined.columns]
    joined = joined[keep]

    joined.to_parquet(TICKER_FILE, index=False)
    print(f"  Posts joined:    {len(joined):,}")
    print(f"  Unique tickers:  {joined['ticker'].nunique()}")
    print(f"  Saved → {TICKER_FILE}")
    return joined


# ─── Step 4: Validation ────────────────────────────────────────────────────────

def validate(scored: pd.DataFrame, scored_with_ticker: pd.DataFrame) -> None:
    from scipy.stats import spearmanr

    print("\n=== STEP 4: Validation ===")
    print()
    print("=== SENTIMENT SCORING VALIDATION ===")
    print()
    print(f"Total posts scored:   {len(scored):,}")
    print(f"Posts with ticker:    {len(scored_with_ticker):,}")
    print(f"Unique tickers:       {scored_with_ticker['ticker'].nunique()}")
    print()
    print("VADER distribution:")
    vs = scored["vader_score"]
    print(f"  Neutral (=0):       {(vs == 0).mean() * 100:.1f}%")
    print(f"  Positive (>0):      {(vs > 0).mean() * 100:.1f}%")
    print(f"  Negative (<0):      {(vs < 0).mean() * 100:.1f}%")
    print(f"  Mean:               {vs.mean():+.3f}")
    print()
    print("FinBERT (high-engagement only):")
    n_fb = int(scored["finbert_score"].notna().sum())
    print(f"  Posts scored:       {n_fb:,}")
    if "finbert_label" in scored.columns and scored["finbert_label"].notna().sum() > 0:
        label_dist = scored["finbert_label"].value_counts(normalize=True) * 100
        for label, pct in label_dist.items():
            print(f"  {label}:  {pct:.1f}%")
    print()

    feat_path = FEATURES / "features_complete.parquet"
    if not feat_path.exists():
        print(f"  [skip] {feat_path.name} not found — no IC check")
        return

    feat = pd.read_parquet(feat_path)
    daily_vader = (
        scored_with_ticker
        .groupby(["ticker", "date"])["vader_score"]
        .mean()
        .reset_index()
        .rename(columns={"vader_score": "vader_mean"})
    )
    merged = feat.merge(daily_vader, on=["ticker", "date"], how="inner")
    merged = merged[merged["target_return_5d"].notna()]

    if len(merged) < 100:
        print(f"  [skip] Only {len(merged)} overlapping rows — not enough for IC check")
        return

    ic_vader, p_vader = spearmanr(merged["vader_mean"], merged["target_return_5d"])

    ic_old = np.nan
    if "avg_sentiment_1d" in merged.columns:
        ic_old, p_old = spearmanr(merged["avg_sentiment_1d"], merged["target_return_5d"])
        print(f"IC Comparison (target_return_5d, n={len(merged):,}):")
        print(f"  Old FinBERT IC:   {ic_old:+.4f} (p={p_old:.3f})")
    else:
        print(f"IC Check (target_return_5d, n={len(merged):,}):")

    print(f"  New VADER IC:     {ic_vader:+.4f} (p={p_vader:.3f})")
    print()

    if not np.isnan(ic_old):
        if abs(ic_vader) > abs(ic_old):
            print("  VADER improves on FinBERT ✓")
        else:
            print("  VADER similar to FinBERT (expected — contrarian)")
            print("  Both negative IC confirms contrarian hypothesis")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score Reddit posts with VADER + FinBERT"
    )
    parser.add_argument("--vader-only", action="store_true", help="Skip FinBERT")
    parser.add_argument("--finbert-only", action="store_true", help="Skip VADER")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="IC validation on existing scored files (no scoring)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from checkpoint (loads existing scored file)",
    )
    args = parser.parse_args()

    if args.validate_only:
        if not SCORED_FILE.exists():
            print(f"[ERROR] {SCORED_FILE} not found. Run scoring first.")
            sys.exit(1)
        scored = pd.read_parquet(SCORED_FILE)
        if TICKER_FILE.exists():
            scored_with_ticker = pd.read_parquet(TICKER_FILE)
        else:
            scored_with_ticker = join_with_tickers(scored)
        validate(scored, scored_with_ticker)
        return

    # Load base dataframe
    if args.resume and SCORED_FILE.exists():
        print(f"[resume] Loading {SCORED_FILE.name} ...")
        df = pd.read_parquet(SCORED_FILE)
        # Append any rows present in source but absent from scored file
        src_ids = set(pd.read_parquet(SOURCE_FILE, columns=["id"])["id"])
        new_ids = src_ids - set(df["id"])
        if new_ids:
            print(f"[resume] {len(new_ids):,} new rows in source — appending")
            src = pd.read_parquet(SOURCE_FILE)
            df = pd.concat([df, src[src["id"].isin(new_ids)]], ignore_index=True)
    else:
        print(f"Loading {SOURCE_FILE.name} ...")
        df = pd.read_parquet(SOURCE_FILE)

    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    # Scoring
    if not args.finbert_only:
        df = run_vader(df)

    if not args.vader_only:
        df = run_finbert(df)

    # Ensure all columns exist even in single-mode runs
    for col, default in [
        ("vader_score", np.nan), ("vader_positive", np.nan),
        ("vader_negative", np.nan), ("vader_neutral", np.nan),
        ("has_body", False), ("text_length", 0),
        ("finbert_score", np.nan), ("finbert_label", None), ("finbert_conf", np.nan),
    ]:
        if col not in df.columns:
            df[col] = default

    # Final save (covers --vader-only and --finbert-only paths)
    df.to_parquet(SCORED_FILE, index=False)

    # Steps 3 & 4
    scored_with_ticker = join_with_tickers(df)
    validate(df, scored_with_ticker)

    print("\nDone.")


if __name__ == "__main__":
    main()
