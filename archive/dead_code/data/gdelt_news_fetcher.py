"""
GDELT GKG v1 news fetcher.
Covers 2023-01-01 to 2024-12-31 (2 years of daily bulk files).

Downloads daily GKG v1 CSV files, filters to finance articles,
extracts ticker mentions via company-name alias matching,
aggregates to one row per (ticker, date).

No API key required — GDELT is a free public dataset.
"""

import io
import logging
import zipfile

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# GKG v1 column schema
# ─────────────────────────────────────────────────────────────────────────────

GKG_V1_COLS = [
    "DATE", "NUMARTS", "COUNTS", "THEMES",
    "LOCATIONS", "PERSONS", "ORGANIZATIONS",
    "TONE", "CAMEOEVENTIDS", "SOURCES", "SOURCEURLS",
]

# ─────────────────────────────────────────────────────────────────────────────
# Step 2A — Ticker alias dictionary
# ─────────────────────────────────────────────────────────────────────────────
# Ambiguity rules applied here:
#   "apple"   alone → skip  (too common a word)
#   "amazon"  alone → skip  (river ambiguity)
#   "arm"     alone → skip  (body part / ARM ISA ambiguity)
#   "meta"    alone → skip  (common word)
#   "snap"    alone → skip  (common word)
#   "marathon" alone → skip (race ambiguity)
#   "block"   alone → skip  (common word)
#   "square"  alone → skip  (common word)

TICKER_ALIASES: dict[str, list[str]] = {
    "NVDA": ["nvidia"],
    "TSLA": ["tesla"],
    "AAPL": ["apple inc"],
    "AMD":  ["amd", "advanced micro devices"],
    "MSFT": ["microsoft"],
    "META": ["meta platforms", "facebook", "instagram"],
    "AMZN": ["amazon.com", "amazon inc"],
    "GOOG": ["google", "alphabet inc", "alphabet"],
    "COIN": ["coinbase"],
    "GME":  ["gamestop"],
    "PLTR": ["palantir"],
    "MARA": ["marathon digital", "mara holdings"],
    "SOFI": ["sofi technologies", "social finance"],
    "HOOD": ["robinhood"],
    "MU":   ["micron", "micron technology"],
    "NFLX": ["netflix"],
    "UBER": ["uber technologies"],
    "SPY":  ["spdr s&p 500"],
    "QQQ":  ["invesco qqq"],
    "SHOP": ["shopify"],
    "ROKU": ["roku"],
    "SNAP": ["snap inc", "snapchat"],
    "HIMS": ["hims & hers", "hims and hers"],
    "RKLB": ["rocket lab"],
    "RDDT": ["reddit"],
    "INTC": ["intel", "intel corporation"],
    "ARM":  ["arm holdings"],
    "BAC":  ["bank of america"],
    "JPM":  ["jpmorgan", "jp morgan"],
    "GS":   ["goldman sachs"],
    "SCHW": ["charles schwab"],
    "WMT":  ["walmart", "wal-mart"],
    "COST": ["costco"],
    "PYPL": ["paypal"],
    "SQ":   ["block inc", "square inc"],
    "ABBV": ["abbvie"],
    "LLY":  ["eli lilly"],
    "PFE":  ["pfizer"],
}

# Reverse lookup: alias (lowercase) → ticker symbol
ALIAS_TO_TICKER: dict[str, str] = {
    alias: ticker
    for ticker, aliases in TICKER_ALIASES.items()
    for alias in aliases
}

# ─────────────────────────────────────────────────────────────────────────────
# Step 2C — Finance filter constants
# ─────────────────────────────────────────────────────────────────────────────

FINANCE_THEMES = {
    "ECON_STOCKMARKET",
    "ECON_EARNINGSREPORT",
    "ECON_IPO",
    "ECON_BANKRUPTCY",
    "ECON_INTEREST_RATE",
    "ECON_MONOPOLY",
}

FINANCE_DOMAINS = {
    "bloomberg.com",
    "reuters.com",
    "wsj.com",
    "cnbc.com",
    "ft.com",
    "marketwatch.com",
    "barrons.com",
    "benzinga.com",
    "fool.com",
    "seekingalpha.com",
    "thestreet.com",
    "investopedia.com",
    "yahoofinance.com",
    "finance.yahoo.com",
}

# ─────────────────────────────────────────────────────────────────────────────
# Step 2B — GKG v1 file downloader
# ─────────────────────────────────────────────────────────────────────────────

def download_gkg_day(date_str: str) -> pd.DataFrame:
    """
    Download and parse one day of GKG v1 data.

    date_str: YYYYMMDD (e.g. "20240115")
    Returns DataFrame with GKG_V1_COLS, or empty DataFrame on failure.
    """
    url = f"http://data.gdeltproject.org/gkg/{date_str}.gkg.csv.zip"

    try:
        resp = requests.get(url, timeout=90)

        if resp.status_code == 404:
            logger.debug(f"gdelt_not_found date={date_str}")
            return pd.DataFrame()

        if resp.status_code != 200:
            logger.warning(
                f"gdelt_download_error status={resp.status_code} "
                f"date={date_str}"
            )
            return pd.DataFrame()

        z        = zipfile.ZipFile(io.BytesIO(resp.content))
        raw      = z.read(z.namelist()[0])

        df = pd.read_csv(
            io.BytesIO(raw),
            sep="\t",
            header=None,
            names=GKG_V1_COLS,
            dtype=str,
            quoting=3,           # QUOTE_NONE — GKG files have unquoted tabs
            encoding="latin-1",
            on_bad_lines="skip",
        )

        logger.debug(
            f"gdelt_downloaded date={date_str} rows={len(df):,}"
        )
        return df

    except zipfile.BadZipFile:
        logger.warning(f"gdelt_bad_zip date={date_str}")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"gdelt_download_failed date={date_str}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Step 2C — Finance article filter
# ─────────────────────────────────────────────────────────────────────────────

def is_finance_article(row: pd.Series) -> bool:
    """Return True if row has finance-related theme or domain."""
    themes  = str(row.get("THEMES",     "") or "")
    sources = str(row.get("SOURCEURLS", "") or "")

    has_fin_theme  = any(t in themes  for t in FINANCE_THEMES)
    has_fin_domain = any(d in sources for d in FINANCE_DOMAINS)

    return has_fin_theme or has_fin_domain


# ─────────────────────────────────────────────────────────────────────────────
# Step 2D — Ticker extraction from ORGANIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def extract_tickers(orgs_str: str) -> set[str]:
    """
    Parse semicolon-separated ORGANIZATIONS field and match to tickers
    via ALIAS_TO_TICKER (substring, case-insensitive).
    """
    if not isinstance(orgs_str, str) or not orgs_str.strip():
        return set()

    orgs_lower = orgs_str.lower()
    return {
        ticker
        for alias, ticker in ALIAS_TO_TICKER.items()
        if alias in orgs_lower
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 2E — Tone parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_tone(tone_str: str) -> float:
    """
    Extract article-level tone from GDELT TONE field.
    Format: "tone,positive,negative,polarity,..."
    Raw range roughly -10 to +10. Normalized to -1.0 to +1.0.
    """
    try:
        raw = float(str(tone_str).split(",")[0])
        return max(-1.0, min(1.0, raw / 10.0))
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Step 2F — Daily aggregation per ticker
# ─────────────────────────────────────────────────────────────────────────────

def compute_gdelt_daily_sentiment(
    df: pd.DataFrame,
    date_str: str,
) -> list[dict]:
    """
    From one day's GKG DataFrame, produce one row per ticker found.

    date_str: YYYYMMDD — converted to YYYY-MM-DD in output.
    Minimum 3 articles per ticker per day (below = too noisy, skip).
    Returns list of row dicts.
    """
    if df.empty:
        return []

    # Filter to finance articles
    fin_mask = df.apply(is_finance_article, axis=1)
    df = df[fin_mask].copy()
    if df.empty:
        return []

    # Extract tickers and normalised tone per row
    df["tickers"] = df["ORGANIZATIONS"].apply(extract_tickers)
    df["tone"]    = df["TONE"].apply(parse_tone)

    # Drop rows with no ticker match
    df = df[df["tickers"].map(len) > 0]
    if df.empty:
        return []

    # Explode: one row per (article, ticker)
    df = df.explode("tickers").rename(columns={"tickers": "ticker"})

    # Minimum 3 articles per ticker per day
    counts        = df["ticker"].value_counts()
    valid_tickers = counts[counts >= 3].index
    df = df[df["ticker"].isin(valid_tickers)]
    if df.empty:
        return []

    # Date string for output
    date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

    results = []
    for ticker, group in df.groupby("ticker"):
        tones = group["tone"].tolist()

        # Unique source domains
        sources: set[str] = set()
        for s in group["SOURCEURLS"].fillna(""):
            sources.update(s.split(";"))
        sources.discard("")

        results.append({
            "ticker":              str(ticker),
            "date":                date_fmt,
            "gdelt_article_count": len(tones),
            "gdelt_tone_mean":     float(np.mean(tones)),
            "gdelt_tone_positive": float(
                np.mean([t for t in tones if t > 0]) if any(t > 0 for t in tones) else 0.0
            ),
            "gdelt_tone_negative": float(
                np.mean([t for t in tones if t < 0]) if any(t < 0 for t in tones) else 0.0
            ),
            "gdelt_source_count":  len(sources),
        })

    return results
