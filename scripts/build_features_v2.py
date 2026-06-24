#!/usr/bin/env python3
"""
build_features_v2.py

Build data/features/features_v2.parquet from scratch.
One row = (ticker, date). 14 clean features. No leakage.

Usage:
    python scripts/build_features_v2.py

Sections:
    1  – Ticker universe
    2  – Market data (yfinance + VIX)
    3  – Reddit post features (rolling attention + sentiment)
    4  – Reddit comment features
    5  – News features (FNSPID / GDELT / Finnhub merged)
    6  – Grid assembly + target computation
    7  – Validation (IC + contrarian check)
    8  – Save to data/features/features_v2.parquet
"""
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).resolve().parent.parent
DATA       = BASE / "data"
RAW        = DATA / "raw"
PROC       = DATA / "processed"
FEAT       = DATA / "features"
MARKET_DIR = PROC / "market"
MARKET_DIR.mkdir(parents=True, exist_ok=True)
FEAT.mkdir(parents=True, exist_ok=True)

# ── Section 1 — Ticker Universe ─────────────────────────────────────────────────
TRADE_TICKERS = [
    "AAPL", "AMD",  "AMZN", "COIN", "GME",  "GOOG", "HOOD",
    "MARA", "META", "MSFT", "MU",   "NFLX", "NVDA", "PLTR",
    "QQQ",  "SOFI", "SPY",  "TSLA", "UBER",
]
WATCH_TICKERS = [
    "SNAP", "PYPL", "ROKU", "DKNG", "GS",  "JPM",
    "WMT",  "BABA", "BA",   "F",    "NIO",
]
ALL_TICKERS = TRADE_TICKERS + WATCH_TICKERS

# First trading day per ticker (no grid rows before this date)
IPO_DATES: dict[str, pd.Timestamp] = {
    "COIN": pd.Timestamp("2021-04-14"),
    "HOOD": pd.Timestamp("2021-07-29"),
    "SOFI": pd.Timestamp("2021-06-01"),
    "PLTR": pd.Timestamp("2020-09-30"),
    "DKNG": pd.Timestamp("2020-04-24"),
}

DATE_START = pd.Timestamp("2019-01-01")
DATE_END   = pd.Timestamp("2026-06-20")
MKT_START  = "2018-10-01"   # extra 3 months for rolling calcs
MKT_END    = "2026-06-21"

SAMPLE_WEIGHTS = {
    2019: 1.0, 2020: 1.0, 2021: 0.3, 2022: 1.0,
    2023: 1.0, 2024: 1.5, 2025: 1.5, 2026: 1.0,
}


# ── Section 2 — Market Data ─────────────────────────────────────────────────────

def _rolling_vix_pct(vix_arr: np.ndarray, window: int = 252) -> np.ndarray:
    """Rolling percentile rank of VIX (fraction of window < current value)."""
    n = len(vix_arr)
    out = np.full(n, 0.5)
    for i in range(n):
        start = max(0, i - window + 1)
        w = vix_arr[start : i + 1]
        if len(w) >= 20:
            out[i] = float(np.mean(w < vix_arr[i]))
    return out


def build_market_data() -> dict[str, pd.DataFrame]:
    """
    Download OHLCV for ALL_TICKERS + VIX percentile via yfinance.
    Saves per-ticker parquets to data/processed/market/.
    Returns dict ticker → DataFrame(date, close, volume, returns_1d, returns_20d,
                                     rsi_14, relative_volume, vix_percentile,
                                     close_fwd1, close_fwd3, close_fwd5).
    """
    print("=== SECTION 2: MARKET DATA ===")

    # ── Download VIX ──────────────────────────────────────────────────────────
    print("  Downloading ^VIX …")
    vix_raw = yf.download("^VIX", start=MKT_START, end=MKT_END,
                           auto_adjust=True, progress=False)
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_s = vix_raw["Close"].iloc[:, 0]
    else:
        vix_s = vix_raw["Close"]
    vix_s = vix_s.dropna().sort_index()
    vix_pct_arr = _rolling_vix_pct(vix_s.values)
    vix_pct = pd.Series(vix_pct_arr, index=vix_s.index, name="vix_percentile")

    # ── Download all tickers ──────────────────────────────────────────────────
    print(f"  Downloading {len(ALL_TICKERS)} tickers …")
    raw = yf.download(
        ALL_TICKERS, start=MKT_START, end=MKT_END,
        auto_adjust=True, progress=False,
    )

    # Flatten MultiIndex: (price_type, ticker) → "Close_NVDA"
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = ["_".join(str(c) for c in col).strip()
                       for col in raw.columns]

    # SPY 200-day MA for regime features — computed once, aligned per ticker
    if "Close_SPY" in raw.columns:
        _spy_close_full   = raw["Close_SPY"].dropna().sort_index()
    else:
        _spy_close_full   = pd.Series(dtype=float)
    _spy_200ma_full       = _spy_close_full.rolling(200, min_periods=100).mean()
    _spy_above_200ma_full = (_spy_close_full > _spy_200ma_full).astype(float)

    market_dfs: dict[str, pd.DataFrame] = {}

    for ticker in ALL_TICKERS:
        try:
            close_col  = f"Close_{ticker}"
            volume_col = f"Volume_{ticker}"
            if close_col not in raw.columns:
                print(f"  WARNING: {ticker} not in download, skipping")
                continue

            close  = raw[close_col].dropna()
            volume = raw[volume_col].reindex(close.index, fill_value=0)

            if len(close) < 30:
                print(f"  WARNING: {ticker} too few rows ({len(close)}), skipping")
                continue

            # ── Technical features (past-only) ─────────────────────────────
            returns_1d  = close.pct_change(1)
            returns_20d = close.pct_change(20)

            delta  = close.diff()
            gain   = delta.clip(lower=0).rolling(14, min_periods=14).mean()
            loss   = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
            rs     = gain / loss.replace(0.0, np.nan)
            rsi_14 = (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)

            avg_vol_20 = volume.rolling(20, min_periods=5).mean()
            rel_vol    = (volume / avg_vol_20.replace(0.0, np.nan)).fillna(1.0)

            # VIX percentile aligned to this ticker's trading dates
            vix_aligned = vix_pct.reindex(close.index, method="ffill").fillna(0.5)

            # Regime features — SPY 200MA status + VIX calm composite
            spy_above_s = (
                _spy_above_200ma_full
                .reindex(close.index, method="ffill")
                .fillna(0.0)
            )
            regime_score_s = spy_above_s * 0.6 + (1.0 - vix_aligned) * 0.4

            # ── Forward close prices for target computation ─────────────────
            close_fwd1 = close.shift(-1)
            close_fwd3 = close.shift(-3)
            close_fwd5 = close.shift(-5)

            df = pd.DataFrame(
                {
                    "date":             close.index,
                    "ticker":           ticker,
                    "close":            close.values,
                    "volume":           volume.values,
                    "returns_1d":       returns_1d.values,
                    "returns_20d":      returns_20d.values,
                    "rsi_14":           rsi_14.values,
                    "relative_volume":  rel_vol.values,
                    "vix_percentile":   vix_aligned.values,
                    "spy_above_200ma":  spy_above_s.values,
                    "regime_score":     regime_score_s.values,
                    "close_fwd1":       close_fwd1.values,
                    "close_fwd3":       close_fwd3.values,
                    "close_fwd5":       close_fwd5.values,
                }
            )
            df["date"] = pd.to_datetime(df["date"])

            # Apply IPO + date range filters
            ipo = IPO_DATES.get(ticker, DATE_START)
            df  = df[df["date"] >= max(ipo, DATE_START)]
            df  = df[df["date"] <= DATE_END]
            df  = df.reset_index(drop=True)

            # Save per-ticker parquet (columns per spec)
            save_cols = ["date", "close", "volume", "returns_1d", "returns_20d",
                         "rsi_14", "relative_volume", "vix_percentile",
                         "spy_above_200ma", "regime_score"]
            df[save_cols].to_parquet(MARKET_DIR / f"{ticker}_ohlcv.parquet", index=False)

            market_dfs[ticker] = df

        except Exception as exc:
            print(f"  WARNING: {ticker} market error — {exc}")

    print(f"  Market data ready for {len(market_dfs)} tickers")
    return market_dfs


# ── Section 3 — Reddit Post Features ───────────────────────────────────────────

def _to_date_series(col: pd.Series) -> pd.Series:
    """Coerce mixed date column (str / datetime.date / Timestamp) to Timestamp."""
    return pd.to_datetime(col, errors="coerce")


def build_all_posts() -> pd.DataFrame:
    """
    Merge reddit_scored_with_tickers (new, better-scored) with
    merged_with_sentiment_full (old historical), dedup on (id, ticker).
    Returns all_posts with columns:
        id, ticker, date, vader_score, has_body, num_comments
    filtered to ALL_TICKERS and date >= 2019-01-01.
    """
    # ── New posts ─────────────────────────────────────────────────────────────
    rt = pd.read_parquet(RAW / "reddit_scored_with_tickers.parquet",
                         columns=["id", "ticker", "date", "vader_score",
                                  "has_body", "num_comments", "score"])
    full = pd.read_parquet(
        RAW / "reddit_full_v2_scored.parquet",
        columns=["id", "finbert_score", "roberta_score", "roberta_conf",
                 "score", "num_comments", "has_body", "deepseek_pred_label"],
    )
    new_posts = rt.merge(
        full[["id", "roberta_score", "roberta_conf", "deepseek_pred_label"]],
        on="id", how="left",
    )
    new_posts["date"] = _to_date_series(new_posts["date"])

    # ── Old posts ─────────────────────────────────────────────────────────────
    old_raw = pd.read_parquet(
        RAW / "merged_with_sentiment_full.parquet",
        columns=["post_id", "ticker", "date", "sentiment_score",
                 "num_comments", "score"],
    )
    old_raw = old_raw.rename(columns={"post_id": "id", "sentiment_score": "vader_score"})
    old_raw["date"]     = _to_date_series(old_raw["date"])
    old_raw["has_body"] = True   # unknown — include in sentiment averaging

    # ── Combine & dedup (prefer new_posts for same id+ticker) ─────────────────
    keep_cols = ["id", "ticker", "date", "vader_score", "has_body", "num_comments"]
    all_posts = pd.concat(
        [new_posts[keep_cols], old_raw[keep_cols]], ignore_index=True
    )
    all_posts = all_posts.drop_duplicates(subset=["id", "ticker"], keep="first")

    # ── Filters ───────────────────────────────────────────────────────────────
    all_posts = all_posts[all_posts["ticker"].isin(ALL_TICKERS)]
    all_posts = all_posts[all_posts["date"] >= DATE_START]
    all_posts = all_posts.dropna(subset=["date"])
    all_posts["date"] = all_posts["date"].dt.normalize()  # strip time component

    print(f"  all_posts: {len(all_posts):,} rows  |  "
          f"tickers: {all_posts.ticker.nunique()}")
    return all_posts.reset_index(drop=True)


def build_reddit_post_features(all_posts: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling Reddit features per (ticker, calendar_date).
    1d window  = posts from T-1 calendar day
    3d window  = posts from T-3 to T-1 calendar days
    Uses shift(1) on a daily-reindexed series for strict no-leakage.

    Returns DataFrame with columns:
        ticker, date, post_count_1d, total_comments_1d,
        vader_sentiment_1d, vader_sentiment_3d,
        sentiment_extremity, sentiment_accel
    (abnormal_attention_1d computed later after trading-day alignment)
    """
    print("=== SECTION 3: REDDIT POST FEATURES ===")

    # Eligible posts for VADER averaging (exclude zero-vader & no-body noise)
    eligible_mask = ~((all_posts["vader_score"] == 0.0) & (~all_posts["has_body"]))

    # ── Daily aggregates per (ticker, date) ───────────────────────────────────
    grp = all_posts.groupby(["ticker", "date"])
    count_d      = grp.size().rename("count_d")
    comments_d   = grp["num_comments"].sum().rename("comments_d")

    elig = all_posts[eligible_mask]
    elig_grp    = elig.groupby(["ticker", "date"])
    vader_sum_d = elig_grp["vader_score"].sum().rename("vader_sum_d")
    vader_cnt_d = elig_grp["vader_score"].count().rename("vader_cnt_d")
    abs_sum_d   = grp["vader_score"].apply(lambda x: x.abs().sum()).rename("abs_sum_d")

    daily = (
        pd.concat([count_d, comments_d, vader_sum_d, vader_cnt_d, abs_sum_d], axis=1)
        .fillna(0)
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])

    # ── Full calendar range for rolling ───────────────────────────────────────
    # Extra 4 days before DATE_START so shift lookback doesn't lose 2019-01-01
    cal_range = pd.date_range(
        start=DATE_START - pd.Timedelta(days=10),
        end=DATE_END,
        freq="D",
    )

    # ── Per-ticker rolling windows ────────────────────────────────────────────
    records = []
    for ticker in ALL_TICKERS:
        t_daily = (daily[daily["ticker"] == ticker]
                   .set_index("date")
                   .drop(columns=["ticker"], errors="ignore"))

        # Reindex to full calendar; fill zeros (silence = no posts)
        t = t_daily.reindex(cal_range, fill_value=0.0)
        t.index.name = "date"

        # 1d window (T-1 calendar day)
        count_1d    = t["count_d"].shift(1)
        comments_1d = t["comments_d"].shift(1)
        vs_1d       = t["vader_sum_d"].shift(1)
        vc_1d       = t["vader_cnt_d"].shift(1)
        abs_1d      = t["abs_sum_d"].shift(1)

        # 3d window (rolling sum of [T-3, T-2, T-1])
        vs_3d = t["vader_sum_d"].rolling(3, min_periods=1).sum().shift(1)
        vc_3d = t["vader_cnt_d"].rolling(3, min_periods=1).sum().shift(1)

        # Sentiment means (safe div)
        vader_1d = (vs_1d / vc_1d.replace(0.0, np.nan)).fillna(0.0)
        vader_3d = (vs_3d / vc_3d.replace(0.0, np.nan)).fillna(0.0)
        extremity = (abs_1d / count_1d.replace(0.0, np.nan)).fillna(0.0)
        accel     = vader_1d - vader_3d

        feat = pd.DataFrame(
            {
                "ticker":              ticker,
                "date":                cal_range,
                "post_count_1d":       count_1d.values,
                "total_comments_1d":   comments_1d.values,
                "vader_sentiment_1d":  vader_1d.values,
                "vader_sentiment_3d":  vader_3d.values,
                "sentiment_extremity": extremity.values,
                "sentiment_accel":     accel.values,
            }
        )
        records.append(feat)

    reddit_feat = pd.concat(records, ignore_index=True)
    reddit_feat["date"] = pd.to_datetime(reddit_feat["date"])

    # Only keep dates in our final range (grid will filter to trading days)
    reddit_feat = reddit_feat[
        (reddit_feat["date"] >= DATE_START) & (reddit_feat["date"] <= DATE_END)
    ]

    print(f"  Reddit post features: {len(reddit_feat):,} rows")
    return reddit_feat.reset_index(drop=True)


# ── Section 4 — Reddit Comment Features ────────────────────────────────────────

def build_comment_features(all_posts: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-(ticker, date) comment features:
        comment_count_1d, comment_sentiment_1d
    Uses 1d window (T-1 calendar day shift) — no leakage.
    """
    print("=== SECTION 4: COMMENT FEATURES ===")

    comments = pd.read_parquet(
        RAW / "reddit_comments_v2_scored.parquet",
        columns=["id", "link_id", "date", "vader_score"],
    )
    comments["date"] = _to_date_series(comments["date"])
    comments = comments.dropna(subset=["date"])
    comments["date"] = comments["date"].dt.normalize()

    # link_id is already the post_id (no 't3_' prefix in actual data)
    # Apply str.replace anyway to handle edge cases per spec
    post_id_col = comments["link_id"].str.replace("^t3_", "", regex=True)

    # Join comments → ticker via all_posts
    post_ticker = all_posts[["id", "ticker"]].drop_duplicates(subset="id")
    merged_c = pd.merge(
        comments.assign(post_id=post_id_col),
        post_ticker.rename(columns={"id": "post_id"}),
        on="post_id", how="inner",
    )
    merged_c = merged_c[merged_c["ticker"].isin(ALL_TICKERS)]

    # Daily aggregates per (ticker, date)
    grp       = merged_c.groupby(["ticker", "date"])
    cnt_d     = grp.size().rename("comment_count_d")
    # Exclude vader==0 from sentiment mean
    elig_c    = merged_c[merged_c["vader_score"] != 0.0]
    elig_grp  = elig_c.groupby(["ticker", "date"])
    vs_d      = elig_grp["vader_score"].sum().rename("comment_vs_d")
    vc_d      = elig_grp["vader_score"].count().rename("comment_vc_d")

    daily_c = (
        pd.concat([cnt_d, vs_d, vc_d], axis=1)
        .fillna(0.0)
        .reset_index()
    )
    daily_c["date"] = pd.to_datetime(daily_c["date"])

    cal_range = pd.date_range(
        start=DATE_START - pd.Timedelta(days=10),
        end=DATE_END,
        freq="D",
    )

    records = []
    for ticker in ALL_TICKERS:
        t_daily = (daily_c[daily_c["ticker"] == ticker]
                   .set_index("date")
                   .drop(columns=["ticker"], errors="ignore"))
        t = t_daily.reindex(cal_range, fill_value=0.0)

        cnt_1d = t["comment_count_d"].shift(1)
        vs_1d  = t["comment_vs_d"].shift(1)
        vc_1d  = t["comment_vc_d"].shift(1)
        sent_1d = (vs_1d / vc_1d.replace(0.0, np.nan)).fillna(0.0)

        feat = pd.DataFrame(
            {
                "ticker":               ticker,
                "date":                 cal_range,
                "comment_count_1d":     cnt_1d.values,
                "comment_sentiment_1d": sent_1d.values,
            }
        )
        records.append(feat)

    comment_feat = pd.concat(records, ignore_index=True)
    comment_feat["date"] = pd.to_datetime(comment_feat["date"])
    comment_feat = comment_feat[
        (comment_feat["date"] >= DATE_START) & (comment_feat["date"] <= DATE_END)
    ]

    print(f"  Comment features: {len(comment_feat):,} rows")
    return comment_feat.reset_index(drop=True)


# ── Section 5 — News Features ───────────────────────────────────────────────────

def build_news_features() -> pd.DataFrame:
    """
    Merge three news sources into unified (ticker, date) news features:
        news_sentiment_1d  [-1, +1]
        news_count_1d      [int]

    Priority:
        date <= 2022-12-31            → FNSPID
        2023-01-01 to 2024-12-31      → GDELT (prefer), else FNSPID
        date >= 2025-01-01            → Finnhub

    Saves data/processed/news_features_merged.parquet.
    """
    print("=== SECTION 5: NEWS FEATURES ===")

    # ── Source A: FNSPID (2019–2023) ──────────────────────────────────────────
    fnspid = pd.read_parquet(PROC / "news_features_2019_2023.parquet",
                             columns=["ticker", "date", "news_sentiment_1d",
                                      "news_count_1d"])
    fnspid["date"] = _to_date_series(fnspid["date"])
    fnspid = fnspid.dropna(subset=["date"])
    fnspid["source"] = "fnspid"

    # ── Source B: GDELT (2023–2024) ───────────────────────────────────────────
    # gdelt_tone_mean is already in ~[-1, +1] range in this parquet
    gdelt_raw = pd.read_parquet(PROC / "gdelt_news_features.parquet",
                                columns=["ticker", "date", "gdelt_tone_mean",
                                         "gdelt_article_count"])
    gdelt_raw["date"] = _to_date_series(gdelt_raw["date"])
    gdelt_raw = gdelt_raw.dropna(subset=["date"])
    gdelt = gdelt_raw.rename(columns={
        "gdelt_tone_mean":    "news_sentiment_1d",
        "gdelt_article_count": "news_count_1d",
    })
    gdelt["source"] = "gdelt"

    # ── Source C: Finnhub (2025–2026) ─────────────────────────────────────────
    finnhub = pd.read_parquet(PROC / "finnhub_news_features.parquet",
                              columns=["ticker", "date", "news_sentiment_1d",
                                       "news_count_1d"])
    finnhub["date"] = _to_date_series(finnhub["date"])
    finnhub = finnhub.dropna(subset=["date"])
    finnhub["source"] = "finnhub"

    # ── Merge strategy ─────────────────────────────────────────────────────────
    cut1 = pd.Timestamp("2022-12-31")
    cut2 = pd.Timestamp("2025-01-01")

    fnspid_use  = fnspid[fnspid["date"] <= cut1].copy()
    gdelt_use   = gdelt[(gdelt["date"] > cut1) & (gdelt["date"] < cut2)].copy()
    # Fill 2023–2024 gaps where GDELT is missing with FNSPID
    fnspid_fill = fnspid[(fnspid["date"] > cut1) & (fnspid["date"] < cut2)].copy()
    # Merge: GDELT preferred; FNSPID fills gaps
    mid_full = pd.concat([gdelt_use, fnspid_fill], ignore_index=True)
    mid_full = mid_full.sort_values(["ticker", "date", "source"],
                                    key=lambda c: c if c.name != "source"
                                    else c.map({"gdelt": 0, "fnspid": 1}))
    mid_dedup = mid_full.drop_duplicates(subset=["ticker", "date"], keep="first")

    finnhub_use = finnhub[finnhub["date"] >= cut2].copy()

    news = pd.concat(
        [fnspid_use, mid_dedup, finnhub_use], ignore_index=True
    )
    news = news[["ticker", "date", "news_sentiment_1d", "news_count_1d"]]
    news = news[news["ticker"].isin(ALL_TICKERS)]
    news["date"] = pd.to_datetime(news["date"])

    out_path = PROC / "news_features_merged.parquet"
    news.to_parquet(out_path, index=False)
    print(f"  News features: {len(news):,} rows → {out_path.name}")
    return news.reset_index(drop=True)


# ── Section 6 — Grid Assembly ───────────────────────────────────────────────────

def build_feature_grid(
    market_dfs:   dict[str, pd.DataFrame],
    reddit_feat:  pd.DataFrame,
    comment_feat: pd.DataFrame,
    news_feat:    pd.DataFrame,
) -> pd.DataFrame:
    """
    Assemble the full (ticker, date) grid, join all features,
    compute targets (forward returns), add metadata.
    """
    print("=== SECTION 6: GRID ASSEMBLY ===")

    # ── Step 1: Trading calendar from SPY ─────────────────────────────────────
    spy_df = market_dfs.get("SPY")
    if spy_df is None:
        raise RuntimeError("SPY market data missing — cannot build trading calendar")
    trading_days = sorted(spy_df["date"].unique())
    print(f"  Trading days: {len(trading_days)} ({trading_days[0].date()} → "
          f"{trading_days[-1].date()})")

    # ── Step 2: Full (ticker, date) grid ──────────────────────────────────────
    rows = []
    for ticker in ALL_TICKERS:
        ipo = IPO_DATES.get(ticker, DATE_START)
        for d in trading_days:
            if d >= max(ipo, DATE_START):
                rows.append((ticker, d))
    grid = pd.DataFrame(rows, columns=["ticker", "date"])
    grid["date"] = pd.to_datetime(grid["date"])
    print(f"  Grid size: {len(grid):,} rows × {grid.ticker.nunique()} tickers")

    # ── Step 3: Join market features ──────────────────────────────────────────
    mkt_all = pd.concat(market_dfs.values(), ignore_index=True)
    mkt_all["date"] = pd.to_datetime(mkt_all["date"])
    mkt_cols = ["ticker", "date", "close", "volume", "returns_1d", "returns_20d",
                "rsi_14", "relative_volume", "vix_percentile",
                "spy_above_200ma", "regime_score",
                "close_fwd1", "close_fwd3", "close_fwd5"]
    grid = grid.merge(mkt_all[mkt_cols], on=["ticker", "date"], how="left")

    # ── Step 4: Join Reddit post features ────────────────────────────────────
    reddit_feat["date"] = pd.to_datetime(reddit_feat["date"])
    grid = grid.merge(reddit_feat, on=["ticker", "date"], how="left")

    reddit_fill = {
        "post_count_1d":       0.0,
        "total_comments_1d":   0.0,
        "vader_sentiment_1d":  0.0,
        "vader_sentiment_3d":  0.0,
        "sentiment_extremity": 0.0,
        "sentiment_accel":     0.0,
    }
    grid = grid.fillna(reddit_fill)

    # Compute abnormal_attention_1d on the trading-day grid
    grid = grid.sort_values(["ticker", "date"]).reset_index(drop=True)
    rolling_20d_avg = (
        grid.groupby("ticker")["post_count_1d"]
        .transform(lambda x: x.rolling(20, min_periods=5).mean())
        .fillna(1.0)
    )
    grid["abnormal_attention_1d"] = (
        grid["post_count_1d"] / (rolling_20d_avg + 1.0)
    ).clip(upper=10.0)

    # ── Step 5: Join comment features ─────────────────────────────────────────
    comment_feat["date"] = pd.to_datetime(comment_feat["date"])
    grid = grid.merge(comment_feat, on=["ticker", "date"], how="left")
    grid[["comment_count_1d", "comment_sentiment_1d"]] = (
        grid[["comment_count_1d", "comment_sentiment_1d"]].fillna(0.0)
    )

    # ── Step 6: Join news features ────────────────────────────────────────────
    news_feat["date"] = pd.to_datetime(news_feat["date"])
    grid = grid.merge(news_feat, on=["ticker", "date"], how="left")
    grid[["news_sentiment_1d", "news_count_1d"]] = (
        grid[["news_sentiment_1d", "news_count_1d"]].fillna(0.0)
    )

    # ── Step 7: Compute target returns (FUTURE data — no leakage) ────────────
    def safe_fwd_return(fwd_close: pd.Series, cur_close: pd.Series) -> pd.Series:
        return (fwd_close - cur_close) / cur_close.replace(0.0, np.nan)

    grid["target_return_1d"] = safe_fwd_return(grid["close_fwd1"], grid["close"])
    grid["target_return_3d"] = safe_fwd_return(grid["close_fwd3"], grid["close"])
    grid["target_return_5d"] = safe_fwd_return(grid["close_fwd5"], grid["close"])

    # Rows where T+5 is beyond DATE_END → NaN target (kept for live prediction)
    # This is already NaN from shift(-5) beyond the last available date

    grid = grid.drop(columns=["close_fwd1", "close_fwd3", "close_fwd5"])

    # ── Step 8: Metadata ──────────────────────────────────────────────────────
    grid["year"] = grid["date"].dt.year

    def assign_split(yr: pd.Series) -> pd.Series:
        s = pd.Series("live", index=yr.index, dtype=object)
        s[yr <= 2023] = "train"
        s[(yr >= 2024) & (yr <= 2025)] = "test"
        return s

    grid["split"] = assign_split(grid["year"])
    grid["sample_weight"] = grid["year"].map(SAMPLE_WEIGHTS).fillna(1.0)

    # ── Step 9: Quality filters ───────────────────────────────────────────────
    pre_drop = len(grid)
    grid = grid.dropna(subset=["close"])               # no market data
    grid = grid[grid["volume"] > 0]                    # no trading
    post_drop = len(grid)
    print(f"  Dropped {pre_drop - post_drop:,} rows (no market data / no trading)")

    # ── Step 10: Select final columns ─────────────────────────────────────────
    IDENTITY = ["ticker", "date", "year", "split", "sample_weight"]
    ATTENTION = ["post_count_1d", "abnormal_attention_1d", "total_comments_1d"]
    SENTIMENT = ["vader_sentiment_1d", "sentiment_extremity",
                 "sentiment_accel", "comment_sentiment_1d"]
    MARKET    = ["volume", "relative_volume", "returns_1d", "returns_20d", "rsi_14"]
    NEWS      = ["news_sentiment_1d", "news_count_1d"]
    REGIME    = ["vix_percentile", "spy_above_200ma", "regime_score"]
    TARGETS   = ["target_return_1d", "target_return_3d", "target_return_5d"]

    # Interaction feature — high VIX fear + abnormal volume = strong signal
    grid["vix_x_volume"] = grid["vix_percentile"] * grid["relative_volume"]

    INTERACTION = ["vix_x_volume"]
    FINAL_COLS = IDENTITY + ATTENTION + SENTIMENT + MARKET + NEWS + REGIME + INTERACTION + TARGETS
    # Keep close for potential debugging but not as a model feature
    grid = grid[FINAL_COLS + ["close"]].copy()

    print(f"  Final grid: {len(grid):,} rows, {grid.shape[1]} columns")
    return grid.reset_index(drop=True)


# ── Section 7 — Validation ──────────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> None:
    """Run full validation per Section 7 of the spec."""
    print()
    print("=== SECTION 7: VALIDATION ===")
    print()
    print(f"Shape:      {df.shape}")
    print(f"Tickers:    {df.ticker.nunique()}")
    print(f"Date range: {df.date.min().date()} → {df.date.max().date()}")
    print(f"Train rows: {(df.split=='train').sum():,}")
    print(f"Test rows:  {(df.split=='test').sum():,}")
    print(f"Live rows:  {(df.split=='live').sum():,}")
    print()

    # Reddit coverage
    print("Reddit coverage (post_count_1d > 0):")
    for ticker in TRADE_TICKERS:
        t = df[df.ticker == ticker]
        if len(t) == 0:
            print(f"  {ticker}: NO DATA")
            continue
        cov = (t["post_count_1d"] > 0).mean() * 100
        print(f"  {ticker}: {cov:.1f}%")
    print()

    # IC check on train split
    FEATURES_14 = [
        "post_count_1d", "abnormal_attention_1d", "total_comments_1d",
        "vader_sentiment_1d", "sentiment_extremity", "sentiment_accel",
        "comment_sentiment_1d", "volume", "relative_volume",
        "returns_1d", "returns_20d", "rsi_14",
        "news_sentiment_1d", "vix_percentile",
        "vix_x_volume", "spy_above_200ma", "regime_score",
    ]

    train = df[(df["split"] == "train") & df["target_return_5d"].notna()].copy()
    print("IC vs target_return_5d (train split):")
    print(f"  {'Feature':<25} {'IC':>8} {'p-val':>8}  Sig")
    print("  " + "-" * 50)
    ics = []
    for feat in FEATURES_14:
        if feat not in train.columns:
            print(f"  {feat:<25} MISSING")
            continue
        valid = train[[feat, "target_return_5d"]].dropna()
        if len(valid) < 100:
            print(f"  {feat:<25} {'(too few)':>8}")
            continue
        ic, pval = spearmanr(valid[feat], valid["target_return_5d"])
        sig = "✓" if pval < 0.05 else " "
        ics.append(ic)
        print(f"  {feat:<25} {ic:>+8.4f} {pval:>8.3f}  {sig}")

    mean_ic = float(np.mean(np.abs(ics))) if ics else 0.0
    print()
    print(f"  Mean |IC|: {mean_ic:.4f}")
    print(f"  Retrain threshold: 0.0846")
    if mean_ic > 0.0846:
        print("  PASS → proceed to retrain ✓")
    else:
        print("  BELOW THRESHOLD → debug before retraining")

    # Contrarian check
    print()
    print("Contrarian sentiment check:")
    try:
        # Only rows with non-zero sentiment (zero = no Reddit posts, not a sentiment signal)
        train_sent = train[
            train["vader_sentiment_1d"].notna() & (train["vader_sentiment_1d"] != 0.0)
        ].copy()
        if len(train_sent) < 200:
            print("  Too few non-zero sentiment rows to check")
        else:
            bins_series, bins = pd.qcut(
                train_sent["vader_sentiment_1d"], q=4,
                retbins=True, duplicates="drop",
            )
            n_bins = bins_series.nunique()
            all_labels = ["Q1_bearish", "Q2", "Q3", "Q4_bullish"]
            labels = all_labels[:n_bins]
            train_sent["sent_q"] = pd.qcut(
                train_sent["vader_sentiment_1d"], q=4,
                labels=labels, duplicates="drop",
            )
            q_returns = (
                train_sent.groupby("sent_q", observed=True)["target_return_5d"].mean()
            )
            print(q_returns.to_string())
            if labels[0] in q_returns.index and labels[-1] in q_returns.index:
                if q_returns[labels[0]] > q_returns[labels[-1]]:
                    print("  Contrarian confirmed ✓")
                else:
                    print("  Contrarian NOT confirmed — check sentiment")
    except Exception as exc:
        print(f"  Contrarian check error: {exc}")

    print()
    # Return mean_ic for the summary
    return mean_ic


# ── Section 8 — Save ─────────────────────────────────────────────────────────────

def save_and_summarize(df: pd.DataFrame, mean_ic: float) -> None:
    out_path = FEAT / "features_v2.parquet"
    # Never overwrite features_complete.parquet (RULE 5)
    assert str(out_path) != str(FEAT / "features_complete.parquet")

    df.to_parquet(out_path, index=False)

    train_rows = (df.split == "train").sum()
    test_rows  = (df.split == "test").sum()
    live_rows  = (df.split == "live").sum()
    status     = "PASS" if mean_ic > 0.0846 else "BELOW THRESHOLD"

    print()
    print("══════════════════════════════════════")
    print("  features_v2.parquet COMPLETE")
    print("══════════════════════════════════════")
    print(f"  Shape:        {df.shape}")
    print(f"  Tickers:      {df.ticker.nunique()}")
    print(f"  Date range:   {df.date.min().date()} → {df.date.max().date()}")
    print(f"  Train rows:   {train_rows:,}")
    print(f"  Test rows:    {test_rows:,}")
    print(f"  Live rows:    {live_rows:,}")
    print(f"  Mean |IC|:    {mean_ic:.4f}")
    print(f"  Status:       {status}")
    print(f"  Saved to:     {out_path}")
    print("══════════════════════════════════════")


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("╔══════════════════════════════════════╗")
    print("║   build_features_v2.py — starting    ║")
    print("╚══════════════════════════════════════╝")
    print()

    market_dfs   = build_market_data()
    all_posts    = build_all_posts()
    reddit_feat  = build_reddit_post_features(all_posts)
    comment_feat = build_comment_features(all_posts)
    news_feat    = build_news_features()
    grid         = build_feature_grid(market_dfs, reddit_feat, comment_feat, news_feat)
    mean_ic      = validate(grid)
    save_and_summarize(grid, mean_ic)


if __name__ == "__main__":
    main()
