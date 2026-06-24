"""
Label high-engagement Reddit posts with DeepSeek as training data
for a financial sentiment classifier.

Steps
  1. Select 10K diverse posts from TRADE_TICKERS (high-engagement, balanced)
  2. Label each post BULLISH / BEARISH / NEUTRAL via DeepSeek API
  3. Train a LogisticRegression on those 10K labels
  4. Score all 401K posts in reddit_full_v2_scored.parquet

Usage:
    python scripts/label_with_deepseek.py [flags]

Flags:
    --select-only   Select 10K posts, write deepseek_sample_10k.parquet, stop
    --label-only    Load sample, call DeepSeek, write deepseek_labels_10k.parquet, stop
    --train-only    Load existing labels, train classifier, score 401K posts
    --resume        Skip already-labeled IDs (requires deepseek_progress.json)
    --test          Label 10 posts, print results, verify API works
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

load_dotenv()
sys.path.insert(0, ".")

# ─── paths ────────────────────────────────────────────────────────────────────
ROOT          = Path(".")
FULL_PATH     = ROOT / "data/raw/reddit_full_v2_scored.parquet"
TICKER_PATH   = ROOT / "data/raw/reddit_scored_with_tickers.parquet"
REGISTRY_PATH = ROOT / "config/ticker_registry.json"
SAMPLE_PATH   = ROOT / "data/raw/deepseek_sample_10k.parquet"
PROGRESS_PATH = ROOT / "data/raw/deepseek_progress.json"
PARTIAL_PATH  = ROOT / "data/raw/deepseek_labels_partial.parquet"
LABELS_PATH   = ROOT / "data/raw/deepseek_labels_10k.parquet"
MODEL_PATH    = ROOT / "models/deepseek_classifier.pkl"

TARGET_N         = 10_000
RATE_SLEEP       = 1.1
CHECKPOINT_EVERY = 500
RETRY_LIMIT      = 3

TRADE_TICKERS = [
    "NVDA", "TSLA", "AAPL", "AMD", "AMZN", "GOOG", "META",
    "MARA", "MSFT", "NFLX", "COIN", "PLTR", "QQQ", "SOFI",
    "HOOD", "SPY", "GME", "UBER", "MU",
]

SUBREDDIT_ENCODE = {
    "wallstreetbets": 0,
    "stocks": 1,
    "investing": 2,
    "options": 3,
    "ValueInvesting": 4,
}

VALID_LABELS = {"BULLISH", "BEARISH", "NEUTRAL"}


# ─── step 1: select posts ─────────────────────────────────────────────────────

def select_posts() -> pd.DataFrame:
    """
    Join ticker file with full text file, apply engagement + diversity filters,
    return ~10K selected posts. Saves to SAMPLE_PATH.
    """
    print("Loading data...")
    full    = pd.read_parquet(FULL_PATH)
    tickers = pd.read_parquet(TICKER_PATH)
    print(f"  full corpus : {len(full):,} rows")
    print(f"  ticker file : {len(tickers):,} rows")

    # ticker file has no title/selftext — join with full to get text columns
    merged = tickers[["id", "ticker"]].merge(
        full[["id", "title", "selftext", "created_utc", "subreddit",
              "score", "num_comments", "has_body", "text_length",
              "vader_score", "finbert_score", "year", "date"]],
        on="id",
        how="inner",
    )
    print(f"  after join  : {len(merged):,} rows")

    # filter: TRADE_TICKERS only
    merged = merged[merged["ticker"].isin(TRADE_TICKERS)].copy()
    print(f"  TRADE filter: {len(merged):,} rows")

    # filter: high engagement
    merged = merged[(merged["score"] >= 50) | (merged["num_comments"] >= 50)].copy()
    print(f"  engagement  : {len(merged):,} rows")

    # filter: meaningful title
    merged = merged[merged["title"].str.len() >= 10].copy()
    print(f"  title len   : {len(merged):,} rows")

    # deduplicate on post id — ticker file maps one post to many tickers
    merged = merged.sort_values("score", ascending=False).drop_duplicates(
        subset="id", keep="first"
    )
    print(f"  unique posts: {len(merged):,}")

    # vader sentiment bins for diversity (33% from each tercile)
    merged["vader_bin"] = pd.qcut(
        merged["vader_score"].fillna(0),
        q=3,
        labels=["low", "mid", "high"],
        duplicates="drop",
    )

    available_years  = sorted(merged["year"].dropna().unique())
    n_years          = len(available_years)
    target_per_year  = TARGET_N // n_years             # ~500 per year
    target_per_tk    = max(1, target_per_year // len(TRADE_TICKERS))
    bin_n            = max(1, target_per_tk // 3)

    samples = []
    for year in available_years:
        year_df = merged[merged["year"] == year]
        for ticker in TRADE_TICKERS:
            tk_df = year_df[year_df["ticker"] == ticker]
            if tk_df.empty:
                continue
            for bin_label in ("low", "mid", "high"):
                bin_df = tk_df[tk_df["vader_bin"] == bin_label]
                if bin_df.empty:
                    continue
                n = min(bin_n, len(bin_df))
                samples.append(bin_df.sample(n=n, random_state=42))

    selected = pd.concat(samples, ignore_index=True).drop_duplicates(subset="id")

    if len(selected) > TARGET_N:
        selected = selected.sample(n=TARGET_N, random_state=42)
    elif len(selected) < TARGET_N:
        remaining = merged[~merged["id"].isin(selected["id"])]
        needed    = TARGET_N - len(selected)
        extra     = remaining.sample(n=min(needed, len(remaining)), random_state=42)
        selected  = pd.concat([selected, extra], ignore_index=True)

    selected = selected.reset_index(drop=True)
    print(f"\nSelected {len(selected):,} posts")
    print("Year distribution:")
    print(selected["year"].value_counts().sort_index().to_string())
    print("\nTicker distribution (top 10):")
    print(selected["ticker"].value_counts().head(10).to_string())

    selected.to_parquet(SAMPLE_PATH, index=False)
    print(f"\nSaved → {SAMPLE_PATH}")
    return selected


# ─── step 2: deepseek labeling ────────────────────────────────────────────────

def _get_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not found — check .env")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def label_post(client: OpenAI, title: str, body: str, ticker: str) -> dict:
    """
    Call DeepSeek to classify one post. Returns {'label': str, 'error': str|None}.
    Retries up to RETRY_LIMIT times; on 429 sleeps 60s; fallback = NEUTRAL.
    """
    title = title.strip()
    body_clean = str(body).strip() if body and len(str(body).strip()) > 20 else ""

    prompt = (
        f"Classify this Reddit post's sentiment "
        f"toward ${ticker} stock price in the next 1-5 days.\n\n"
        f"Rules:\n"
        f"- BULLISH = author expects price to GO UP\n"
        f"  (buying calls, bought shares, positive outlook,\n"
        f"   price target increase, good earnings)\n"
        f"- BEARISH = author expects price to GO DOWN\n"
        f"  (buying puts, shorting, negative outlook,\n"
        f"   guidance cut, bad earnings, selling)\n"
        f"- NEUTRAL = no clear price direction signal\n"
        f"  (general question, news without opinion,\n"
        f"   unclear intent)\n\n"
        f"Examples:\n"
        f'"Long $NVDA calls" → BULLISH\n'
        f'"Bought puts on TSLA before earnings" → BEARISH\n'
        f'"NVDA cuts guidance for Q4" → BEARISH\n'
        f'"AAPL beats earnings, stock up 5%" → BULLISH\n'
        f'"What do you think about NVDA?" → NEUTRAL\n'
        f'"NVIDIA to acquire Mellanox" → NEUTRAL\n\n'
        f"Now classify this post:\n"
        f"Title: {title}\n"
        + (f"Body: {body_clean[:200]}\n" if body_clean else "")
        + f"\nReply with exactly one word — BULLISH, BEARISH, or NEUTRAL:"
    )

    last_error: str | None = None
    for attempt in range(RETRY_LIMIT):
        try:
            resp  = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system",
                     "content": "You are a financial sentiment classifier. Reply with exactly one word."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=10,
                temperature=0.1,
            )
            raw   = resp.choices[0].message.content.strip().upper()
            label = raw if raw in VALID_LABELS else next(
                (lbl for lbl in VALID_LABELS if lbl in raw), "NEUTRAL"
            )
            return {"label": label, "error": None}

        except Exception as exc:
            last_error = str(exc)
            if "402" in last_error or "Insufficient Balance" in last_error:
                raise RuntimeError(
                    "DeepSeek account has insufficient balance — "
                    "top up at platform.deepseek.com before running."
                ) from exc
            if "429" in last_error:
                print(f"    Rate limited — sleeping 60s (attempt {attempt + 1}/{RETRY_LIMIT})")
                time.sleep(60)
            else:
                print(f"    API error (attempt {attempt + 1}/{RETRY_LIMIT}): {last_error[:120]}")
                if attempt < RETRY_LIMIT - 1:
                    time.sleep(RATE_SLEEP * 2)

    return {"label": "NEUTRAL", "error": last_error}


def _load_progress() -> dict:
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {"completed_ids": [], "total_labeled": 0, "last_updated": ""}


def _save_progress(progress: dict) -> None:
    progress["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f)


def _load_partial_labels() -> dict[str, str]:
    """Return {id: label} from checkpoint parquet if it exists."""
    if not PARTIAL_PATH.exists():
        return {}
    df = pd.read_parquet(PARTIAL_PATH, columns=["id", "deepseek_label"])
    return dict(zip(df["id"], df["deepseek_label"]))


def label_posts(df: pd.DataFrame, resume: bool = False,
                test_mode: bool = False) -> pd.DataFrame:
    """
    Label posts in df with DeepSeek. Returns df with 'deepseek_label' column.
    Checkpoints every CHECKPOINT_EVERY posts to PARTIAL_PATH + PROGRESS_PATH.
    """
    client   = _get_client()
    progress = _load_progress() if resume else {"completed_ids": [], "total_labeled": 0}
    done_ids = set(progress["completed_ids"])

    # recover labels from previous partial save
    partial_labels: dict[str, str] = _load_partial_labels() if resume else {}

    work_df = df.head(10) if test_mode else df
    todo    = work_df[~work_df["id"].isin(done_ids)].copy()
    print(f"Posts to label: {len(todo):,}  (already done: {len(done_ids):,})")

    new_results: list[dict] = []
    counts = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0}
    start  = time.time()

    for i, row in enumerate(todo.itertuples(index=False), start=1):
        result = label_post(client, str(row.title),
                            str(getattr(row, "selftext", "") or ""), row.ticker)
        label = result["label"] or "NEUTRAL"
        counts[label] = counts.get(label, 0) + 1

        new_results.append({"id": row.id, "deepseek_label": label})
        done_ids.add(row.id)
        partial_labels[row.id] = label

        if i % CHECKPOINT_EVERY == 0 or i == len(todo):
            # persist progress + partial labels
            progress["completed_ids"]  = list(done_ids)
            progress["total_labeled"]  = len(done_ids)
            _save_progress(progress)

            # save partial labels so resume can recover them
            _save_partial(work_df, partial_labels)

            elapsed   = time.time() - start
            rate      = i / elapsed if elapsed > 0 else 1
            remaining = len(todo) - i
            eta_s     = int(remaining / rate) if rate > 0 else 0
            tot       = sum(counts.values()) or 1
            print(
                f"Labeled {i:,} / {len(todo):,} ({i / len(todo) * 100:.1f}%) | "
                f"BULLISH={counts['BULLISH'] / tot * 100:.0f}% "
                f"BEARISH={counts['BEARISH'] / tot * 100:.0f}% "
                f"NEUTRAL={counts['NEUTRAL'] / tot * 100:.0f}% | "
                f"eta={eta_s // 3600}h{(eta_s % 3600) // 60:02d}m"
            )

        if test_mode:
            pass  # no sleep in test mode
        else:
            time.sleep(RATE_SLEEP)

    # attach labels to df
    label_map = {**partial_labels}
    df = df.copy()
    df["deepseek_label"] = df["id"].map(label_map).fillna("NEUTRAL")

    if test_mode:
        print("\nTest results:")
        for _, row in df.head(10).iterrows():
            print(f"  [{row['deepseek_label']}] {row['title'][:80]}")
        print("\nDeepSeek API working ✓")

    return df


def _save_partial(df: pd.DataFrame, label_map: dict[str, str]) -> None:
    """Save checkpoint parquet so --resume can recover labels."""
    labeled_ids = list(label_map.keys())
    out = df[df["id"].isin(labeled_ids)].copy()
    out["deepseek_label"] = out["id"].map(label_map)
    out[["id", "ticker", "date", "title", "deepseek_label"]].to_parquet(
        PARTIAL_PATH, index=False
    )


# ─── step 3+4: assemble output parquet ────────────────────────────────────────

def save_labels(df: pd.DataFrame) -> None:
    """Write deepseek_labels_10k.parquet with the required schema."""
    out = pd.DataFrame({
        "id":             df["id"],
        "ticker":         df["ticker"],
        "date":           df["date"],
        "title":          df["title"],
        "body_preview":   df.get("selftext", pd.Series("", index=df.index)).fillna("").str[:100],
        "score":          df["score"],
        "num_comments":   df["num_comments"],
        "deepseek_label": df["deepseek_label"],
        "vader_score":    df["vader_score"],
        "finbert_score":  df["finbert_score"],
    })
    out.to_parquet(LABELS_PATH, index=False)
    print(f"\nSaved → {LABELS_PATH}")

    # summary
    total = len(out)
    counts = out["deepseek_label"].value_counts()
    vader_agree  = _agreement(out, "vader_score",   "deepseek_label")
    finbert_agree = _agreement(out, "finbert_score", "deepseek_label")

    print()
    print("══════════════════════════════════════")
    print("DEEPSEEK LABELING COMPLETE")
    print("══════════════════════════════════════")
    print(f"Total labeled:  {total:,}")
    for lbl in ("BULLISH", "BEARISH", "NEUTRAL"):
        n = counts.get(lbl, 0)
        print(f"{lbl:8s}:  {n / total * 100:.0f}% ({n:,} posts)")
    print()
    print(f"Agreement with VADER:   {vader_agree:.1f}%")
    print(f"Agreement with FinBERT: {finbert_agree:.1f}%")
    print()
    print(f"Saved: {LABELS_PATH}")


def _agreement(df: pd.DataFrame, score_col: str, label_col: str) -> float:
    """Directional agreement between a continuous score and categorical labels."""
    mask  = df[score_col].notna() & df[label_col].notna()
    sub   = df[mask].copy()
    if sub.empty:
        return 0.0
    pred = pd.cut(sub[score_col], bins=[-np.inf, -0.05, 0.05, np.inf],
                  labels=["BEARISH", "NEUTRAL", "BULLISH"])
    return (pred == sub[label_col]).mean() * 100


# ─── step 5: train classifier ─────────────────────────────────────────────────

def _make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix for logistic regression."""
    feats = pd.DataFrame(index=df.index)
    feats["vader_score"]   = df["vader_score"].fillna(0)
    feats["finbert_score"] = df["finbert_score"].fillna(0)
    # roberta_score not present in current dataset — keep as placeholder
    feats["roberta_score"] = df.get("roberta_score",
                                    pd.Series(0.0, index=df.index)).fillna(0)
    feats["title_length"]  = df["title"].str.split().str.len().fillna(0)
    feats["has_body"]      = df["has_body"].astype(int)
    feats["score"]         = np.log1p(df["score"].fillna(0))
    feats["num_comments"]  = np.log1p(df["num_comments"].fillna(0))
    feats["hour_of_day"]   = (df["created_utc"] % 86400) // 3600
    feats["subreddit_enc"] = df["subreddit"].map(SUBREDDIT_ENCODE).fillna(0)
    return feats.astype(float)


def train_classifier() -> None:
    """
    Train LogisticRegression on deepseek_labels_10k.parquet, then score
    all 401K posts and write deepseek_pred_label + deepseek_pred_confidence
    back to reddit_full_v2_scored.parquet.
    """
    print("\nLoading labeled data...")
    labeled = pd.read_parquet(LABELS_PATH)
    labeled = labeled[labeled["deepseek_label"].isin(VALID_LABELS)].copy()
    print(f"  Labeled rows: {len(labeled):,}")

    print("Loading full corpus...")
    full = pd.read_parquet(FULL_PATH)
    print(f"  Full corpus : {len(full):,} rows")

    # labeled parquet has a subset of columns — join with full to get all features
    join_cols = ["id", "created_utc", "has_body", "subreddit",
                 "score", "num_comments", "vader_score", "finbert_score", "title"]
    labeled_full = labeled.merge(full[join_cols], on="id", how="left",
                                 suffixes=("_lab", ""))
    # prefer right-side (full corpus) columns when available
    for col in ("score", "num_comments", "vader_score", "finbert_score", "title"):
        lab_col = col + "_lab"
        if lab_col in labeled_full.columns:
            labeled_full[col] = labeled_full[col].fillna(labeled_full[lab_col])
            labeled_full.drop(columns=[lab_col], inplace=True)

    # sort by date to avoid time-leakage in split
    if "date" in labeled_full.columns:
        labeled_full = labeled_full.sort_values("date").reset_index(drop=True)

    X = _make_features(labeled_full)
    y = labeled_full["deepseek_label"]

    split = int(len(labeled_full) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    print("Training LogisticRegression...")
    clf = LogisticRegression(max_iter=1000, random_state=42, multi_class="ovr", C=1.0)
    clf.fit(X_tr_s, y_tr)

    preds = clf.predict(X_te_s)
    acc   = accuracy_score(y_te, preds)
    print(f"  Test accuracy: {acc:.3f}")
    print(classification_report(y_te, preds, zero_division=0))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(
        {"clf": clf, "scaler": scaler, "feature_cols": X.columns.tolist()},
        MODEL_PATH,
    )
    print(f"  Saved model → {MODEL_PATH}")

    print("\nScoring all 401K posts...")
    X_all       = _make_features(full)
    X_all_s     = scaler.transform(X_all)
    pred_labels = clf.predict(X_all_s)
    pred_probs  = clf.predict_proba(X_all_s)
    pred_conf   = pred_probs.max(axis=1)

    full["deepseek_pred_label"]      = pred_labels
    full["deepseek_pred_confidence"] = pred_conf
    full.to_parquet(FULL_PATH, index=False)
    print(f"  Updated → {FULL_PATH}")

    dist  = pd.Series(pred_labels).value_counts()
    total = len(pred_labels)
    print("\nPrediction distribution (all 401K):")
    for lbl in ("BULLISH", "BEARISH", "NEUTRAL"):
        n = dist.get(lbl, 0)
        print(f"  {lbl}: {n / total * 100:.1f}% ({n:,})")


# ─── cli ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--select-only", action="store_true",
                   help="Select 10K posts and stop")
    p.add_argument("--label-only",  action="store_true",
                   help="Run DeepSeek labeling only (needs sample parquet)")
    p.add_argument("--train-only",  action="store_true",
                   help="Train classifier on existing labels")
    p.add_argument("--resume",      action="store_true",
                   help="Skip already-labeled post IDs")
    p.add_argument("--test",        action="store_true",
                   help="Label 10 posts to verify API works")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── test mode ────────────────────────────────────────────────────────────
    if args.test:
        print("── TEST MODE (10 posts) ──")
        if SAMPLE_PATH.exists():
            sample = pd.read_parquet(SAMPLE_PATH)
            print(f"Loaded existing sample ({len(sample):,} rows)")
        else:
            sample = select_posts()
        label_posts(sample, resume=False, test_mode=True)
        return

    # ── train-only ───────────────────────────────────────────────────────────
    if args.train_only:
        if not LABELS_PATH.exists():
            print(f"ERROR: {LABELS_PATH} not found — run labeling first")
            sys.exit(1)
        train_classifier()
        return

    # ── select step ──────────────────────────────────────────────────────────
    if not args.label_only:
        if SAMPLE_PATH.exists() and args.resume:
            print(f"Reusing existing sample: {SAMPLE_PATH}")
            sample = pd.read_parquet(SAMPLE_PATH)
        else:
            sample = select_posts()

        if args.select_only:
            return
    else:
        if not SAMPLE_PATH.exists():
            print(f"ERROR: {SAMPLE_PATH} not found — run --select-only first")
            sys.exit(1)
        sample = pd.read_parquet(SAMPLE_PATH)
        print(f"Loaded existing sample: {len(sample):,} posts")

    # ── label step ───────────────────────────────────────────────────────────
    labeled = label_posts(sample, resume=args.resume, test_mode=False)
    save_labels(labeled)

    if args.label_only:
        return

    # ── train step ───────────────────────────────────────────────────────────
    train_classifier()


if __name__ == "__main__":
    main()
