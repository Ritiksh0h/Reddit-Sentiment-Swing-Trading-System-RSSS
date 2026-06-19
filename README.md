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

The signal validation layer (`experiments/`) tests this with Granger causality,
annual IC, and walk-forward IC across four time windows (2022–2025).

---

## Current Status

**Phase 4 — Paper Trading (live since June 15, 2026)**

| Metric | Value |
|---|---|
| Model | XGBoost, 14 features, 5-day horizon |
| Train split | 2019–2023 (3,106 rows after density gate) |
| Test split | 2024–2025 (3,582 rows, 2-year out-of-sample) |
| IC_test (5D) | **0.0796** |
| Directional accuracy | **52.4%** |
| Train/test IC gap | 0.35 (healthy) |
| Automation | launchd, 09:00 / 11:30 / 14:00 ET Mon–Fri |

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
 → density gate (post_count_1d >= 10)
 → 14-feature vectors per (ticker, date)
 → data/features/features_full.parquet (14,889 rows, 2019–2026)
         │
         ▼
 XGBoost regressors (model_1d, model_3d, model_5d)
 → predicts forward return per horizon
 → BULLISH if pred_5d >= 1.5%, BEARISH if pred_5d <= -1.5%
 → confidence = min(|pred_5d| / 0.03, 1.0)
         │
         ▼
 portfolio/signal_generator.py
 → SignalRecord: ticker, pred_1d/3d/5d, confidence, signal, price_targets
         │
         ▼
 portfolio/ (engine + sizer + regime detector)
 → ATR-based sizing, regime multiplier (POSITIVE=100% / NEUTRAL=75% / NEGATIVE=50%)
 → dynamic slippage: 0.001 + 0.0005 × min(mention_growth_7d, 3.0)
 → max 3 positions, 5-day hold, 15% take-profit cap, -8% stop-loss
         │
         ▼
 api/main.py (FastAPI thin entry point)
 → api/routes/health.py       — /health, /status, /settings
 → api/routes/portfolio.py    — /portfolio, /positions, /signals/recent
 → api/routes/predictions.py  — /predictions, /top-predictions, /shap/{ticker}
 → api/routes/performance.py  — /signal-accuracy, /ic-monitor, /model-metadata
 → api/routes/research.py     — /research-findings, /backtest
         │
         ▼
 dashboard/index.html
 → equity curve, signals table, 1D/3D/5D predictions, PnL vs SPY
```

---

## Key Findings

From the signal validation layer (`experiments/`):

**Reddit attention (post_count_1d) → strong universe filter, not a directional predictor.**
- IC on all rows: 0.008 (noise)
- IC on rows with post_count_1d >= 10: 0.092 (real signal)
- Granger causality for Reddit *sentiment* (avg_sentiment_1d): 0/6 years significant
- Reddit sentiment was dropped from the feature set; post density is kept as a filter

**Post density is the single biggest value driver.**
Without the density gate the model produces noise. With it, IC triples.

**Market momentum features dominate predictions.**
`returns_5d`, `dist_from_20ma`, `dist_from_50ma` carry the most feature importance.
Reddit attention adds value by filtering to the right universe, not by predicting direction.

**News + StockTwits retrain did not improve IC.**
Historical data was merged (FNSPID 2019–2023, StockTwits archive 2019–2022).
Retrain on `features_complete.parquet` produced IC = 0.0686 — below the 0.0796
baseline. News coverage is only 24–38%; StockTwits covers 2019–2022 only (0.0 for 2023+).
Live fetchers populate all three sources daily for future retraining.

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
  features/                    ← features_full.parquet (14,889 rows, primary)
                                   features_complete.parquet (with news+ST merged)
                                   features_live_2026.parquet (live, grows daily)
  live/                        ← paper_portfolio.json, paper_performance.jsonl
  processed/                   ← news + StockTwits parquet (Colab output)

portfolio/
  signal_generator.py          ← density gate → features → XGBoost → signals
  position_sizer.py            ← ATR-based sizing + dynamic slippage
  regime_detector.py           ← SPY 200MA + 60d return → POSITIVE/NEUTRAL/NEGATIVE
  portfolio_engine.py          ← position tracking, exits, risk rules
  execution_logger.py          ← append-only JSONL audit trail
  drift_monitor.py             ← API anomaly detection (skip day on undercount)
  paper_trader.py              ← PnL vs SPY benchmark

scripts/
  daily_run_live.py            ← live orchestrator: Reddit+news+ST → trades
  daily_run.py                 ← portfolio orchestrator (called by daily_run_live)
  train_phase3_model.py        ← trains model_1d, model_3d, model_5d
  monitor_live_ic.py           ← weekly IC gate check (GREEN/AMBER/RED)
  append_live_features.py      ← saves feature vectors + fills t+5 price targets
  test_historical_run.py       ← backfill test against historical feature store
  merge_external_features.py   ← merges news + StockTwits into feature store

experiments/
  phase3_locked_architecture.json  ← read-only architecture contract
  experiment_c/                    ← winning experiment (IC=0.111, 2024)
  layer1_signal_existence/         ← Granger causality results
  layer2_regime/                   ← Regime classifier results
  layer3_model/                    ← Family validation results
  source_validation/               ← multi-source validation sprint
  shared/                          ← metrics.py (canonical compute_ic), backtest.py

models/registry/
  model_1d.pkl, model_3d.pkl, model_5d.pkl
  phase3_model.pkl             ← backward-compat copy of model_5d.pkl
  phase3_model_baseline.json   ← training metrics

api/
  main.py                      ← FastAPI thin entry point + /dashboard route
  _helpers.py                  ← shared helpers (_sanitize, _load_portfolio)
  routes/
    health.py                  ← /health, /status, /settings
    portfolio.py               ← /portfolio, /positions, /signals/recent, /trades/history
    predictions.py             ← /predictions, /top-predictions, /shap/{ticker}
    performance.py             ← /signal-accuracy, /ic-monitor, /model-metadata, /backfill
    research.py                ← /research-findings, /backtest, /backtest-full

dashboard/
  index.html                   ← single-file dark dashboard

logs/
  paper_trades.jsonl           ← execution log (NEVER DELETE)
  ic_monitor.jsonl             ← weekly IC readings
  daily_runs.log               ← pipeline run log

archive/
  notebooks/                   ← Colab notebooks (phase0, experiment_c, news/ST processing)
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

**IC monitoring gate**: GREEN ≥ 0.03 | AMBER 0.01–0.03 | RED < 0.01
Fix 3 triggers only after **two consecutive** red weeks — never after one.

---

## Automation

Three runs per weekday via launchd (system clock is EDT = UTC−4):

```
com.rsss.api           → always on, port 8000 (RunAtLoad + KeepAlive)
com.rsss.dailyrun      → 09:00 ET Mon–Fri → daily_run_live.py
com.rsss.dailyrun.1130 → 11:30 ET Mon–Fri → daily_run_live.py
com.rsss.dailyrun.1400 → 14:00 ET Mon–Fri → daily_run_live.py
com.rsss.icmonitor     → 09:00 ET Monday  → monitor_live_ic.py

Check:  launchctl list | grep rsss
Reload: launchctl unload ~/Library/LaunchAgents/com.rsss.dailyrun.plist \
        && launchctl load ~/Library/LaunchAgents/com.rsss.dailyrun.plist
```

---

## Hard Rules

- Never random train/test split — always time-based
- Never use future data in features
- Never modify `experiments/phase3_locked_architecture.json`
- Never overwrite `data/raw/merged_with_sentiment.parquet` (backup)
- Never delete `logs/paper_trades.jsonl`
- Never change the density gate (post_count_1d >= 10) without re-running signal validation
- Never trigger Fix 3 after only one Red week — require two
- Never open more than 3 positions simultaneously
- Always ATR-based sizing, never equal-weight
- Never retrain unless IC improvement > 0.005 over current 0.0796

---

*Phase 4 — Paper Trading | June 2026 | IC_test=0.0796 | dir_acc=52.4%*
