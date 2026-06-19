# CLAUDE.md — Reddit Sentiment Swing Trading System (RSSS)
# Master context file for Claude Code sessions
# Read this first before every session. Never skip it.
# GitHub: https://github.com/Ritiksh0h/Reddit-Sentiment-Swing-Trading-System-RSSS

---

## What This Project Is

A quantitative swing trading research system that answers one question:

> "Can Reddit sentiment + financial news + StockTwits predict short-term
>  stock returns? Show bullish/bearish signals with 1D/3D/5D predictions
>  and confidence scores on a dashboard."

This is NOT a sentiment classifier or hype detector.
It is a time-aligned numerical compression of crowd attention + market response.

**Core finding so far:** Reddit post density (post_count_1d >= 10) is the
primary value driver — not sentiment. Market IC on all rows = 0.008 (noise).
Market IC on high-attention rows = 0.092 (real signal). Sentiment Granger
test showed 0/6 significant years for Reddit sentiment. News and StockTwits
sentiment still untested formally — that is the next sprint.

---

## Current Phase: Phase 4 — Paper Trading (LIVE)

System runs automatically three times per weekday via launchd (system in EDT, UTC-4).
Paper trading started: June 15, 2026.

```bash
curl http://localhost:8000/status        # system health
open http://localhost:8000/dashboard     # full dashboard
tail -5 logs/paper_trades.jsonl | python3 -m json.tool
```

---

## Project Structure (post-cleanup target)

```
config/
  settings.py                  ← env vars, paths, global config
  thresholds.py                ← ALL magic numbers — no exceptions
  false_positive_list.txt      ← ticker extraction blocklist
  tickers.txt                  ← tracked universe (29 tickers)

data/
  raw/
    merged_with_sentiment.parquet          ← 2019-2024, 880K rows (backup)
    merged_with_sentiment_full.parquet     ← 2019-2026, 932K rows ✓
  features/
    features_expanded.parquet             ← 2019-2024 backup (keep)
    features_full.parquet                 ← 2019-2026, 14,889 rows ✓ PRIMARY
    features_complete.parquet             ← with news+ST merged ✓
  processed/
    news_features_2019_2023.parquet       ← FNSPID output ✓
    stocktwits_features_2019_2022.parquet ← ST archive output ✓
  reddit_live_fetcher.py        ← Arctic Shift API, paginated
  stocktwits_fetcher.py         ← StockTwits free API
  news_fetcher.py               ← yfinance news + FinBERT
  mention_history.json          ← rolling 14-day post count history
  paper_portfolio.json          ← current portfolio state
  paper_performance.jsonl       ← daily PnL snapshots

experiments/
  phase3_locked_architecture.json  ← LOCKED — read-only contract
  experiment_c/                    ← winning baseline experiment
  layer1_signal_existence/         ← Reddit Granger results (0/6 years)
  layer2_regime/                   ← Regime classifier results
  layer3_model/                    ← Family validation results
  source_validation/               ← NEW: multi-source validation sprint
    validate_sources.py            ← Layer 1/2/3 across Reddit+news+ST
    results.json                   ← output (after running)
  shared/
  signal_validation_summary.py
  winner.md

models/registry/
  phase3_model.pkl              ← backward compat (= model_5d.pkl)
  model_1d.pkl                  ← 1-day horizon ✓
  model_3d.pkl                  ← 3-day horizon ✓
  model_5d.pkl                  ← 5-day horizon ✓ PRIMARY
  phase3_model_baseline.json    ← training metrics

portfolio/
  signal_generator.py           ← density gate → features → XGBoost → signals
  position_sizer.py             ← ATR-based sizing + dynamic slippage
  regime_detector.py            ← SPY rule-based regime
  portfolio_engine.py           ← position tracking, risk rules, exits
  execution_logger.py           ← append-only JSONL audit trail
  drift_monitor.py              ← API anomaly detection
  paper_trader.py               ← PnL tracking vs SPY

scripts/
  daily_run.py                  ← portfolio orchestrator
  daily_run_live.py             ← live orchestrator: Reddit+news+ST → trades
  train_phase3_model.py         ← multi-horizon training (1D/3D/5D)
  monitor_live_ic.py            ← weekly IC gate check
  fix3_switch_to_17_features.py ← Fix 3 protocol
  test_historical_run.py        ← backfill test
  merge_external_features.py    ← merges news+ST into feature store

api/
  main.py                       ← FastAPI: /status /portfolio /signals
                                   /predictions /shap/{ticker}
                                   /signal-accuracy /research-findings

dashboard/
  index.html                    ← single-file dark dashboard

logs/
  paper_trades.jsonl            ← NEVER DELETE — source of live IC
  paper_performance.jsonl       ← daily PnL snapshots
  ic_monitor.jsonl              ← weekly IC readings
  daily_runs.log
  launchd.log / launchd_error.log

tests/
  test_phase3.py                ← 15 tests, all must pass

NOTE: backtest/, features/, signals/, analytics/, reports/,
      run_pipeline.py, session_notes.md, portfolio/risk.py,
      portfolio/sizing.py, portfolio/engine.py,
      data/feature_store.py, data/reddit_loader.py,
      data/market_loader.py, scripts/phase0_validate.py,
      scripts/seed_historical.py, scripts/run_backtest.py,
      scripts/scheduler.py are ALL DELETED in the cleanup sprint.
```

---

## Locked Architecture (READ-ONLY)

NEVER modify `experiments/phase3_locked_architecture.json` manually.

```python
with open('experiments/phase3_locked_architecture.json') as f:
    ARCH = json.load(f)
```

```
Features (14):
  Market (8):     returns_1d, returns_5d, returns_20d, rsi_14, atr_14,
                  relative_volume, dist_from_20ma, dist_from_50ma
  Attention (3):  post_count_1d, mention_growth_1d, mention_growth_7d
  News (1):       news_sentiment_1d
  StockTwits (2): st_sentiment_1d, st_bull_pct

Density gate:    post_count_1d >= 10
Drop tickers:    ASTS, LCID, MSTR, RIOT, RIVN, SMCI, WMT
Hold days:       5
Take-profit cap: 15%
Max positions:   3
Min pred return: 1% (0.01)
Regime sizing:   POSITIVE=100% / NEUTRAL=75% / NEGATIVE=50%
Position sizing: ATR-based (target_risk_pct=0.02)
Slippage:        dynamic: 0.001 + 0.0005 × min(mention_growth_7d, 3.0)
Ticker cooldown: 7 days
```

---

## Model State

```
Current model:    phase3_v3_multihorizon
Features:         14 (8 market + 3 attention + 1 news + 2 StockTwits)
Train split:      2019-2023 (fallback — 2022-2023 too few rows)
Test split:       2024-2025 (two-year out-of-sample) ✓
Train rows:       ~3,200 after density gate + DROP_TICKERS
Test rows:        ~3,582 (2024: 2,187  2025: 1,395)

Model_1D: IC_test = varies  (noisy — expected)
Model_3D: IC_test = varies  (weak directional assist)
Model_5D: IC_test = 0.0796  (primary signal)
Dir accuracy 5D:  52.4%
Train/test gap:   0.35

Feature store:    data/features/features_full.parquet (14,889 rows)

IMPORTANT: news/ST retrain completed but did NOT improve IC:
  2019-2022 train (with real ST data):  IC = 0.0686
  2019-2023 train (features_full):      IC = 0.0796  ← best, current
  Reason: news coverage only 24-38%, ST only covers 2019-2022.
  Source validation sprint (PROJECT_COMPLETION_SPRINT.md) will
  formally test whether any combination beats current baseline.
```

---

## External Data Collected (historical)

```
FNSPID news (2019-2023):
  data/processed/news_features_2019_2023.parquet
  28,230 daily rows, 34 tickers
  Coverage: 24-46% of feature store rows (S&P500 only)

StockTwits archive (2019-2022):
  data/processed/stocktwits_features_2019_2022.parquet
  70,112 daily rows, 62 tickers
  Coverage: 92-95% of training rows
  Sentiment: 83% bullish, 17% bearish (inherent ST bias)
  Tags: 100% of rows have real bullish/bearish values

features_complete.parquet:
  All three sources merged (14,889 rows, 33 cols)
  news_sentiment_1d: real values for 2019-2023 (24-46% coverage)
  st_sentiment_1d:   real values for 2019-2022 (92-95% coverage)
  st_sentiment_1d:   0.0 for 2023+ (no archive data)
```

---

## Live Data Sources

```
Reddit:      Arctic Shift API (free, paginated)
             4 subreddits: wallstreetbets, stocks, investing, options

News:        yfinance t.news + FinBERT — 20-35 tickers/day

StockTwits:  Free public API — 38 tickers, native bullish/bearish tags

Market:      yfinance (period='90d' for features, '5d' for exits)
```

Tiingo is blocked (403). Do not use. yfinance is the replacement.

---

## Phase 4 Monitoring Gates

```
Green:  30-day live IC > 0.03    → continue
Amber:  30-day live IC 0.01-0.03 → watch closely
Red:    30-day live IC < 0.01    → Fix 3 after 2 consecutive weeks
```

Manual check: `python scripts/monitor_live_ic.py`

---

## Automation (macOS launchd)

System clock is EDT (UTC-4). All times below are EDT = ET.

```
com.rsss.api              → always on, RunAtLoad=true, KeepAlive=true → venv uvicorn port 8000
com.rsss.dailyrun         → 09:00 EDT Mon-Fri → daily_run_live.py
com.rsss.dailyrun.1130    → 11:30 EDT Mon-Fri → daily_run_live.py
com.rsss.dailyrun.1400    → 14:00 EDT Mon-Fri → daily_run_live.py
com.rsss.icmonitor        → 09:00 EDT Monday  → monitor_live_ic.py

Plist files: ~/Library/LaunchAgents/com.rsss.*.plist
```

Check: `launchctl list | grep rsss`

---

## Drift Monitor

```python
HISTORICAL_MEANS = {
    'post_count_1d':     53.2,
    'avg_sentiment_1d': -0.025,
    'mention_growth_7d': 0.232,
}
ALERT_LOW_MULTIPLIER  = 0.5
ALERT_HIGH_MULTIPLIER = 2.0
```

skip_day triggers ONLY on post_count undercount.
High counts = WARNING only, do not skip trading.
mention_growth_7d skipped until mention_history.json has 7+ days.

---

## Key Bugs Fixed

```
e4e7c22  Exit price: used entry_price for exits. Fix: yfinance fetch.
e3c5c0b  Backfill accuracy: is_backfill flag + date-specific fetch.
Drift 1: post_count uses max() not mean().
Drift 2: mention_growth_7d skipped while history immature.
```

---

## API Endpoints

```
GET /status                 → system health check
GET /portfolio              → current positions + PnL
GET /signals                → today's signals
GET /predictions            → 1D/3D/5D predictions bullish/bearish
GET /shap/{ticker}          → SHAP attribution per source family
GET /signal-accuracy        → 1D/3D/5D accuracy from paper trading
GET /research-findings      → source validation results.json
POST /retrain-model         → trigger retraining
```

---

## Pending Work (priority order)

```
Priority 1 — NEXT (Claude Code):
  PROJECT_COMPLETION_SPRINT.md

  Part A: Cleanup
    Delete dead folders: backtest/ features/ signals/ analytics/ reports/
    Delete dead files: run_pipeline.py, session_notes.md,
                       portfolio/risk.py, portfolio/sizing.py,
                       portfolio/engine.py, data/feature_store.py,
                       data/reddit_loader.py, data/market_loader.py,
                       scripts/phase0_validate.py, scripts/seed_historical.py,
                       scripts/run_backtest.py, scripts/scheduler.py
    Update config/thresholds.py — remove stale constants
    Rewrite README.md

  Part B: Signal Validation Sprint
    Create experiments/source_validation/validate_sources.py
    Layer 1: Annual IC per feature with REGIME LABELS per year
             2019=BULL, 2020=CRASH/RECOVERY, 2021=RETAIL BULL,
             2022=BEAR, 2023=RECOVERY, 2024=AI BULL, 2025=MIXED
    Layer 2: Granger causality for news, ST, Reddit
    Layer 3: Walk-forward IC for 8 feature combinations
    Output: experiments/source_validation/results.json
    Answers: which source (Reddit/news/ST) has causal-predictive signal?

  Part C: Dashboard Completion
    C1: SHAP attribution via /shap/{ticker} endpoint
        Shows Reddit/News/StockTwits/Market contribution per prediction
        Uses shap.TreeExplainer (shap already in requirements.txt)
    C2: Per-horizon accuracy tracker (1D, 3D, 5D separately)
        /signal-accuracy endpoint
        Color: green>=55%, amber>=50%, red<50%
        Interpretation: "1D low + 5D high = multi-day lag (normal)"
    C3: Research findings panel reads results.json
        /research-findings endpoint
        Shows IC table + regime labels + source verdicts on dashboard

Priority 2 — after signal validation:
  Retrain with winning feature combination from Layer 3
  Only if mean_ic improvement > 0.005 over current 0.0796

Priority 3 — after 30+ days live IC:
  Upgrade density gate to Reddit OR news OR StockTwits combined
  Add news/ST to drift monitor HISTORICAL_MEANS

Priority 4 — future:
  10D model (target_return_10d already in feature store)
  Entity-level LLM scoring for news (if news IC > 0.05 in 2+ years)
  VADER pre-filter for Reddit (when volume justifies it)
  Earnings calendar integration
```

---

## Hard Rules (Non-Negotiable)

```
DATA
  NEVER random train/test split
  NEVER use future data in features
  NEVER modify experiments/phase3_locked_architecture.json
  NEVER overwrite features_expanded.parquet (backup)
  NEVER delete logs/paper_trades.jsonl

MODEL
  NEVER change density gate without re-running signal validation
  NEVER retrain on 2026 data — 5-day returns incomplete
  ALWAYS save phase3_model.pkl = copy of model_5d.pkl
  NEVER trigger Fix 3 after only one Red week — require two
  NEVER retrain unless IC improvement > 0.005

PORTFOLIO
  NEVER equal-weight sizing — always ATR-based
  NEVER flat slippage — always dynamic
  NEVER open more than 3 positions simultaneously
  NEVER force trades on zero signals — hold cash is correct

LIVE SYSTEM
  ALWAYS log every signal to logs/paper_trades.jsonl
  ALWAYS dry-run before first live run after any code change
  NEVER lower monitoring gates to avoid Fix 3
  ALWAYS handle API failures gracefully — default 0.0
```

---

## Environment

```
Python:    3.13 (venv at .venv)
Activate:  source .venv/bin/activate
Project:   /Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot
Git push:  bash push.sh "[scope] description"
API:       uvicorn api.main:app --reload --port 8000
Dashboard: http://localhost:8000/dashboard
```

---

## Session Start Checklist

```bash
git pull origin main
source .venv/bin/activate

# Confirm key files
ls models/registry/model_5d.pkl
ls experiments/phase3_locked_architecture.json
ls data/features/features_full.parquet
ls data/features/features_complete.parquet

# Read locked architecture
python3 -c "
import json
with open('experiments/phase3_locked_architecture.json') as f:
    arch = json.load(f)
print('Features:', arch['features'])
print('Count:', arch.get('feature_count', len(arch['features'])))
"

# System health (for live work)
curl -s http://localhost:8000/status | python3 -m json.tool
```

---

## Session End Checklist

```bash
pytest tests/ -v --tb=short
python scripts/daily_run_live.py --dry-run
bash push.sh "[scope] what you built"
```

---

*CLAUDE.md — June 2026*
*Updated: multi-horizon models, feature store rebuilt, external data collected,*
*retrain attempted (no IC improvement), project completion sprint pending*
