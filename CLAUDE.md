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
test showed 0/6 significant years for Reddit sentiment. News + StockTwits
historical data has been merged (features_complete.parquet) but did not
improve IC (0.0686 vs 0.0796 baseline). Formal source validation is next.

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

## Project Structure

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
    features_full.parquet                 ← 2019-2026, 14,889 rows ✓ PRIMARY (phase3)
    features_complete.parquet             ← with news+ST merged ✓
    features_live_2026.parquet            ← live rows (grows daily, t+5 filled)
    features_v2.parquet                   ← V2: 53,592 rows, 27 cols (incl. regime)
  live/
    paper_portfolio.json                  ← current portfolio state
    paper_performance.jsonl               ← daily PnL snapshots
  processed/
    news_features_2019_2023.parquet       ← FNSPID output ✓
    stocktwits_features_2019_2022.parquet ← ST archive output ✓
    features_target_pending.json          ← rows awaiting t+5 price fill
  reddit_live_fetcher.py        ← Arctic Shift API, paginated
  stocktwits_fetcher.py         ← StockTwits free API
  news_fetcher.py               ← yfinance news + FinBERT
  mention_history.json          ← rolling 14-day post count history

experiments/
  phase3_locked_architecture.json  ← LOCKED — read-only contract
  experiment_c/                    ← winning baseline experiment
  layer1_signal_existence/         ← Reddit Granger results (0/6 years)
  layer2_regime/                   ← Regime classifier results
  layer3_model/                    ← Family validation results
  source_validation/               ← multi-source validation (Part B — TODO)
  shared/
    metrics.py                     ← canonical compute_ic (Spearman, import from here)
    backtest.py, trainer.py, validation_utils.py
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
  regime_detector.py            ← SPY 200MA + 60d return → POSITIVE/NEUTRAL/NEGATIVE
  portfolio_engine.py           ← position tracking, risk rules, exits
  execution_logger.py           ← append-only JSONL audit trail
  drift_monitor.py              ← API anomaly detection (skip day on undercount)
  paper_trader.py               ← PnL tracking vs SPY

scripts/
  daily_run.py                  ← portfolio orchestrator (called by daily_run_live)
  daily_run_live.py             ← live orchestrator: Reddit+news+ST → trades
  train_phase3_model.py         ← multi-horizon training (1D/3D/5D)
  monitor_live_ic.py            ← weekly IC gate check
  append_live_features.py       ← saves feature vectors + fills t+5 price targets
  fix3_switch_to_17_features.py ← Fix 3 protocol
  test_historical_run.py        ← backfill test against historical feature store
  merge_external_features.py    ← merges news+ST into feature store
  build_features_v2.py          ← V2: builds features_v2.parquet (27 cols + regime)
  train_models_v2.py            ← V2: GKX stumps with ICEarlyStopping, 16 features
  run_backtest_v2.py            ← V2: rank-based core-satellite backtest

api/
  main.py                       ← FastAPI thin entry point + /dashboard route
  _helpers.py                   ← shared helpers (_sanitize, _load_portfolio)
  routes/
    health.py                   ← /health /status /settings
    portfolio.py                ← /portfolio /positions /signals/recent /trades/history
    predictions.py              ← /predictions /top-predictions /shap/{ticker}
    performance.py              ← /signal-accuracy /ic-monitor /model-metadata /backfill
    research.py                 ← /research-findings /backtest /backtest-full

dashboard/
  index.html                    ← single-file dark dashboard

logs/
  paper_trades.jsonl            ← NEVER DELETE — source of live IC
  ic_monitor.jsonl              ← weekly IC readings
  daily_runs.log
  launchd.log / launchd_error.log

archive/
  notebooks/                    ← Colab notebooks (phase0, experiment_c, news/ST processing)

tests/
  test_phase3.py                ← 21 tests
  test_backtest.py              ← 5 tests
  (26 total — all must pass)
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

### Phase 3 — Live System (ACTIVE)

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

news/ST retrain did NOT improve IC:
  2019-2022 train (with real ST data):  IC = 0.0686
  2019-2023 train (features_full):      IC = 0.0796  ← best, current
```

### V2 Research Track — GKX Stumps + Regime Features (NOT in live system yet)

```
Architecture:  GKX stumps (Gu, Kelly & Xiu 2020)
               max_depth=1, Pseudo-Huber loss, L2 reg_lambda=5, min_child_weight=20
               Per-horizon gamma: 1D=0.0 / 3D=0.1 / 5D=0.5
               ICEarlyStopping callback (Spearman) — two-phase: scout then clean fit

Features (16): 14 phase3 features + 2 regime features:
  spy_above_200ma  float — 1.0 if SPY close > SPY 200-day MA, else 0.0
  regime_score     spy_above×0.6 + (1−vix_pct)×0.4

Feature store: data/features/features_v2.parquet (53,592 rows, 27 cols)
Density gate:  >= 5 (training) / >= 3 (backtest signal generation)
Train gated:   12,032 rows  |  Test gated: 3,797 rows

V2 model metrics (test 2024-2025):
  model_1d_v2:  IC=0.041  trees=13  dir=53.7%
  model_3d_v2:  IC=0.013  trees=1   dir=54.5%  (1 tree: gamma=0.1 limits splits)
  model_5d_v2:  IC=0.041  trees=28  dir=55.8%
  Note: lower IC vs phase3 0.0796 because GKX stumps compress predictions
        to ~mean; signal is in rank ordering, not absolute magnitude

V2 backtest (rank-based, core-satellite 70% SPY / 30% RSSS, 2024-2025 OOS):
  Signal:   composite = 0.5×pred5d + 0.3×pred3d + 0.2×pred1d
            top 2 per day, quality gates: score>0, regime_score≥0.3, rel_vol≥0.8
  Combined: +35.6%  |  SPY: +49.7%  |  Alpha: -12.2%
  Sharpe:   1.32  |  Max DD: -14.1%
  Trades:   167   |  Win rate: 57.5%  |  p=0.063 (borderline, not significant)
  Sys A = Sys B (model-reversal never fires — GKX stumps always predict positive)

V2 files:
  scripts/build_features_v2.py      ← builds features_v2.parquet (27 cols)
  scripts/train_models_v2.py        ← GKX training with ICEarlyStopping
  scripts/run_backtest_v2.py        ← rank-based core-satellite backtest
  models/model_{1d,3d,5d}_v2.json  ← XGBoost JSON format (16 features)
  models/training_metadata_v2.json  ← training metrics + status
  experiments/backtest_v2_results.json
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
Health / settings
  GET  /health              → {"status":"ok","version":"3.0"}
  GET  /status              → ran_today, n_positions, cash, system_ok
  GET  /settings            → dashboard settings (data/dashboard_settings.json)
  POST /settings            → save dashboard settings

Portfolio
  GET  /portfolio           → cash, positions, closed_trades, pnl summary
  GET  /positions           → open positions with unrealized PnL
  GET  /signals/recent      → last N signals from paper_trades.jsonl
  GET  /trades/history      → closed trades with realized PnL
  GET  /log/recent          → last N raw lines from paper_trades.jsonl

Predictions
  GET  /predictions         → 1D/3D/5D predictions for tracked tickers
  GET  /top-predictions     → top bullish + top bearish signals
  GET  /shap/{ticker}       → SHAP attribution by source family (market/attention/news/ST)

Performance
  GET  /signal-accuracy     → 1D/3D/5D directional accuracy from live trades
  GET  /ic-monitor          → IC readings from ic_monitor.jsonl
  GET  /model-metadata      → model training metrics from phase3_model_baseline.json
  GET  /backfill-log        → last N backfill results
  POST /backfill            → trigger test_historical_run.py

Research
  GET  /research-findings   → source validation results.json
  GET  /backtest            → experiment_c backtest results
  GET  /backtest-full       → full 2024-2025 simulation results

Dashboard
  GET  /dashboard           → serves dashboard/index.html
```

---

## Pending Work (priority order)

```
DONE — Cleanup Sprint (all 9 phases complete, June 2026):
  ✓ Part A: Dead code removed, config consolidated, API split into routes/
  ✓ Canonical compute_ic in experiments/shared/metrics.py
  ✓ API refactored into api/routes/ (5 route files)
  ✓ experiments/ __init__.py files added
  ✓ Live data moved to data/live/
  ✓ Docstrings added to all key portfolio + scripts functions
  ✓ README.md rewritten
  ✓ CLAUDE.md updated

DONE — V2 Research Sprint (June 2026):
  ✓ build_features_v2.py: regime features (spy_above_200ma, regime_score), 27-col store
  ✓ train_models_v2.py: GKX stumps, per-horizon gamma, ICEarlyStopping, 16 features
  ✓ run_backtest_v2.py: rank-based core-satellite (70/30), 167 trades, Sharpe 1.32
  V2 NOT deployed to live system — gate is IC improvement > 0.005 over 0.0796

Priority 1 — NEXT (Claude Code):
  Part B: Signal Validation Sprint
    Create experiments/source_validation/validate_sources.py
    Layer 1: Annual IC per feature with REGIME LABELS per year
             2019=BULL, 2020=CRASH/RECOVERY, 2021=RETAIL BULL,
             2022=BEAR, 2023=RECOVERY, 2024=AI BULL, 2025=MIXED
    Layer 2: Granger causality for news, ST, Reddit
    Layer 3: Walk-forward IC for 8 feature combinations
    Output: experiments/source_validation/results.json
    Answers: which source (Reddit/news/ST) has causal-predictive signal?

  Part C: Dashboard wiring (API endpoints already exist)
    C1: /shap/{ticker} — verify SHAP attribution renders in dashboard
    C2: /signal-accuracy — verify 1D/3D/5D accuracy panel renders
        Color: green>=55%, amber>=50%, red<50%
    C3: /research-findings — wire results.json to research panel once Part B done

Priority 2 — after signal validation:
  Retrain with winning feature combination from Part B Layer 3
  ONLY if mean_ic improvement > 0.005 over current 0.0796

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
*Updated: V2 research sprint complete — GKX stumps (16 features, regime), rank-based*
*backtest (167 trades, Sharpe 1.32, p=0.063). V2 NOT in live system (gate: IC > 0.0796+0.005).*
