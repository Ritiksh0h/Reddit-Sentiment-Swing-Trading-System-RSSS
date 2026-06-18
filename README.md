# RSSS — Reddit Sentiment Swing Trading System

A quantitative swing trading research system that predicts 1-day, 3-day, and 5-day
forward returns using Reddit crowd attention, financial news sentiment, StockTwits
sentiment, and market features. Signals run through an ATR-based portfolio engine with
regime-adjusted sizing and dynamic slippage.

**This is not a sentiment classifier or hype detector.** It is a time-aligned numerical
compression of crowd attention + market response. Reddit `post_count_1d >= 10` acts
as a universe filter — IC on all rows is 0.008 (noise); IC on high-activity rows is 0.092
(real signal). The density filter is the primary value driver.

---

## Research Question

> Can Reddit crowd attention + financial news sentiment + StockTwits sentiment
> predict short-term stock returns better than market features alone?

The signal validation layer (experiments/) tests this with Granger causality,
annual IC, and walk-forward IC across four time windows (2022–2025).

---

## Current Status

**Phase 4 — Paper Trading (live since June 2026)**

| Metric | Value |
|---|---|
| Model | XGBoost, 14 features, 5-day horizon |
| Train split | 2019–2023 (3,106 rows after density gate) |
| Test split | 2024–2025 (3,582 rows, 2-year out-of-sample) |
| IC_test (5D) | **0.0796** |
| Directional accuracy | **52.4%** |
| Train/test IC gap | 0.35 (healthy) |
| Automation | launchd, 08:30 ET Mon–Fri |

Dashboard: `http://localhost:8000/dashboard`

---

## Quick Start

```bash
# 1. Activate virtualenv
source .venv/bin/activate

# 2. Start API + dashboard
uvicorn api.main:app --reload --port 8000

# 3. Check system status
curl http://localhost:8000/status

# 4. Run dry-run (no trades logged)
python scripts/daily_run_live.py --dry-run

# 5. Run tests
pytest tests/ -v --tb=short
```

---

## Architecture

```
 Reddit (Arctic Shift API)     → post_count, mention_growth
 News (yfinance + FinBERT)     → news_sentiment_1d
 StockTwits (free public API)  → st_sentiment_1d, st_bull_pct
 yfinance (market data)        → price, volume, RSI, ATR
         │
         ▼
 pipeline/01_feature_builder.py
 → density gate (post_count >= 10)
 → 14-feature vectors per (ticker, date)
 → data/features/features_full.parquet (14,889 rows, 2019–2026)
         │
         ▼
 XGBoost regressors (Model_1D, Model_3D, Model_5D)
 → predicts forward return per horizon
 → BULLISH if pred_5d >= 3%, BEARISH if <= -3%
 → confidence = min(|pred_5d| / 0.03, 1.0)
         │
         ▼
 portfolio/signal_generator.py
 → SignalRecord: ticker, pred_1d/3d/5d, confidence, signal, price_targets
         │
         ▼
 portfolio/ (engine + sizer + regime detector)
 → ATR-based sizing, regime multiplier (POSITIVE/NEUTRAL/NEGATIVE)
 → dynamic slippage: 0.001 + 0.0005 × min(mention_growth_7d, 3.0)
 → max 3 positions, 5-day hold, 15% take-profit cap
         │
         ▼
 api/main.py (FastAPI)
 → /status, /portfolio, /signals, /predictions
         │
         ▼
 dashboard/index.html
 → equity curve, signals table, 1D/3D/5D predictions, PnL vs SPY
```

---

## Key Findings

From the signal validation layer (`experiments/`):

**Reddit attention (post_count_1d) → strong filter, not a predictor.**
- IC on all rows: 0.008 (noise)
- IC on rows with post_count >= 10: 0.092 (real signal)
- Granger causality for Reddit *sentiment* (avg_sentiment_1d): 0/6 years significant
- Reddit sentiment was dropped from the model per L1 Granger test

**Post density as a universe filter is the single biggest value driver.**
Without the density gate, the model produces noise. With it, IC triples.

**Market momentum features dominate predictions.**
`returns_5d`, `dist_from_20ma`, `dist_from_50ma` carry the most feature importance.
Reddit attention adds value by filtering to the right universe, not by predicting direction.

**News + StockTwits (historical rows default to 0.0).**
FNSPID and StockTwits S3 data not yet merged into training data. Live fetchers populate
these daily going forward. Expected IC improvement after full merge: 0.0796 → 0.09–0.11.

---

## Project Structure

```
config/
  settings.py                  ← env vars, paths
  thresholds.py                ← every magic number (canonical)
  tickers.txt                  ← tracked universe (29 tickers)
  false_positive_list.txt      ← ticker extraction blocklist

pipeline/
  01_feature_builder.py        ← builds features_full.parquet from raw parquet

data/
  reddit_live_fetcher.py       ← Arctic Shift API, paginated
  news_fetcher.py              ← yfinance news + FinBERT scoring
  stocktwits_fetcher.py        ← StockTwits free API
  mention_history.json         ← rolling 14-day post count history
  raw/                         ← merged_with_sentiment_full.parquet (932K rows)
  features/                    ← feature_full.parquet (14,889 rows, primary)
  processed/                   ← news + StockTwits parquet (Colab output)

portfolio/
  signal_generator.py          ← density gate → features → XGBoost → signals
  position_sizer.py            ← ATR-based sizing + dynamic slippage
  regime_detector.py           ← SPY 200MA + 60d return
  portfolio_engine.py          ← position tracking, exits, risk rules
  execution_logger.py          ← append-only JSONL audit trail
  drift_monitor.py             ← API anomaly detection
  paper_trader.py              ← PnL vs SPY

scripts/
  daily_run_live.py            ← live orchestrator: fetches all sources → trades
  daily_run.py                 ← portfolio orchestrator (called by live)
  train_phase3_model.py        ← trains Model_1D, Model_3D, Model_5D
  monitor_live_ic.py           ← weekly IC gate check (GREEN/AMBER/RED)
  fix3_switch_to_17_features.py
  test_historical_run.py       ← backfill test with synthetic Reddit data
  merge_external_features.py   ← merges news + StockTwits into feature store

experiments/
  phase3_locked_architecture.json  ← read-only architecture contract
  experiment_c/                    ← winning experiment (IC=0.111, 2024)
  layer1_signal_existence/         ← Granger causality results
  layer2_regime/                   ← Regime classifier results
  layer3_model/                    ← Family validation results
  source_validation/               ← multi-source validation (Part B)
  winner.md

models/registry/
  model_1d.pkl, model_3d.pkl, model_5d.pkl
  phase3_model.pkl             ← backward-compat copy of model_5d.pkl
  phase3_model_baseline.json   ← training metrics

api/
  main.py                      ← FastAPI: /status /portfolio /signals /predictions

dashboard/
  index.html                   ← single-file dark dashboard

logs/
  paper_trades.jsonl           ← execution log (NEVER DELETE)
  paper_performance.jsonl      ← daily PnL snapshots
  ic_monitor.jsonl             ← weekly IC readings
```

---

## Model Details

**14-feature locked set** (`experiments/phase3_locked_architecture.json`):

| Group | Features |
|---|---|
| Market (8) | returns_1d, returns_5d, returns_20d, rsi_14, atr_14, relative_volume, dist_from_20ma, dist_from_50ma |
| Attention (3) | post_count_1d, mention_growth_1d, mention_growth_7d |
| News (1) | news_sentiment_1d |
| StockTwits (2) | st_sentiment_1d, st_bull_pct |

**Retraining**: `python scripts/train_phase3_model.py --train-years 2019,2020,2021,2022,2023 --test-years 2024,2025`

**IC monitoring gate**: GREEN ≥ 0.03 | AMBER 0.01–0.03 | RED < 0.01 (Fix 3 after 2 consecutive red weeks)

---

## Automation

```
com.rsss.dailyrun    → 08:30 ET Mon–Fri → daily_run_live.py
com.rsss.icmonitor   → 09:00 ET Monday  → monitor_live_ic.py

Check: launchctl list | grep rsss
Reload: launchctl unload && launchctl load ~/Library/LaunchAgents/com.rsss.dailyrun.plist
```

---

## Hard Rules

- Never random train/test split — always time-based
- Never use future data in features
- Never modify `experiments/phase3_locked_architecture.json`
- Never overwrite `features_expanded.parquet` (backup)
- Never delete `logs/paper_trades.jsonl`
- Never lower the density gate (post_count >= 10)
- Never trigger Fix 3 after only one Red week — require two
- Never open more than 3 positions simultaneously
- Always ATR-based sizing, never equal-weight

---

*Phase 4 — Paper Trading | June 2026 | IC_test=0.0796 | dir_acc=52.4%*
