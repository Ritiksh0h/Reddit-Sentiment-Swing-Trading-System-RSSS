# RSSS — Reddit Sentiment Swing Trading System

A quantitative swing trading research system that analyzes Reddit crowd attention,
financial news sentiment, and market features to predict 1/3/5-day stock returns using
XGBoost (GKX-optimal stumps). Includes walk-forward validated backtesting, live paper
trading via Railway API, and an automated daily pipeline via GitHub Actions.

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost 3.2](https://img.shields.io/badge/XGBoost-3.2-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Railway](https://img.shields.io/badge/deployed-Railway-blueviolet)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

**Live Dashboard:** https://reddit-sentiment-swing-trading-system-rsss-production-032d.up.railway.app/dashboard
**API:** https://reddit-sentiment-swing-trading-system-rsss-production-032d.up.railway.app

---

## Research Question

> Can Reddit crowd attention + news sentiment predict short-term stock returns better
> than market features alone?

**Honest answer:** Weak but real signal confirmed.

- IC = 0.041 on 2-year out-of-sample test (2024–2025)
- 55.8% directional accuracy on the correct ticker universe
- Walk-forward Sharpe 0.86 across 23 quarterly folds (2019–2025)
- **Not statistically significant yet** (p = 0.063, need ~200 live trades to confirm)

---

## Current Status

**Phase 4 — Live Paper Trading (started June 25, 2026)**

| Metric | Value |
|---|---|
| Model | XGBoost GKX stumps (depth=1) |
| Features | 16 (v2 schema) |
| Training period | 2019–2023 |
| Test period | 2024–2025 (3,797 rows after density gate) |
| Test IC (5D) | 0.041 |
| Directional accuracy (5D) | 55.8% |
| Backtest trades | 167 (rank-based signals, 2024–2025 OOS) |
| Backtest Sharpe | 1.32 (2024–2025 OOS) |
| Walk-forward Sharpe | 0.86 (23 folds, 2019–2025) |
| Walk-forward folds profitable | 74% (17/23) |
| 2022 bear market return | -3.9% (controlled drawdown) |
| Win rate | 57.5% |
| Statistical significance | p = 0.063 (borderline — not confirmed) |
| Live since | June 25, 2026 (paper trading only) |
| API | Railway (always-on FastAPI) |
| Pipeline | GitHub Actions (weekdays 08:30 ET) |

---

## Key Findings

### 1. Density gate is the primary value driver

The post-count filter is the single biggest contributor — more than sentiment, more
than any individual feature.

| Subset | IC |
|---|---|
| All rows (no gate) | 0.008 — noise |
| post_count_1d ≥ 5 | 0.041 — real signal |

Without the density gate the model predicts noise. With it, IC is 5× higher.

### 2. Reddit sentiment is contrarian

High positive sentiment = a stock has already been pumped. Raw directional sentiment
has no predictive value (Granger test: 0/6 years significant). The useful signal is
`sentiment_extremity` (how unusual sentiment is), not its sign. This holds consistently
across all years 2019–2025.

### 3. Volume and VIX are the strongest individual features

| Feature | IC |
|---|---|
| volume | +0.066 (consistent every year) |
| vix_x_volume (interaction) | +0.041 (best combined feature) |
| vix_percentile | +0.037 (strongest in 2022 bear market) |

The volume × VIX interaction term outperforms either feature in isolation.

### 4. Ticker universe quality matters more than model complexity

| System | Win rate |
|---|---|
| Old system (AMC/BBBY/GME meme tickers) | ~25% live |
| Current system (NVDA/AAPL/TSLA/AMZN/etc.) | 57.5% backtest |

Choosing a stable, liquid universe is a larger performance driver than hyperparameter
tuning.

### 5. Short selling adds no value

Short win rate: ~50% (coin flip). Short avg return: +0.08% (near zero). The system
is long-only. Removing shorts improved Sharpe without reducing trade count.

### 6. GKX depth-1 stumps are the correct model for this problem

More trees → train IC rises, test IC stays flat → overfitting. GKX stumps compress
predictions near the mean — the signal lives in rank ordering, not absolute magnitude.
1–28 trees depending on horizon, confirmed by IC-based early stopping (not RMSE).

---

## Architecture

```
Data Sources
────────────
Reddit (Arctic Shift API, 48h delay)  → post_count_1d, vader_sentiment_1d,
                                         abnormal_attention_1d, sentiment_extremity,
                                         sentiment_accel, total_comments_1d
News (Finnhub API)                    → news_sentiment_1d
StockTwits (public API)               → st_sentiment_1d
yfinance (market data)                → volume, relative_volume, returns_1d, returns_20d,
                                         rsi_14, vix_percentile, vix_x_volume
Regime overlay                        → spy_above_200ma, regime_score
        │
        ▼
Feature Engineering  (scripts/build_features_v2.py)
  Density gate: post_count_1d >= 5
  16 features per (ticker, date)
  53,592 rows, 30 tickers, 2019–2025
        │
        ▼
GKX Stump Models  (scripts/train_models_v2.py)
  3 separate XGBoost regressors (1D / 3D / 5D returns)
  max_depth=1, Pseudo-Huber loss, reg_lambda=5, min_child_weight=20
  Per-horizon gamma: 1D=0.0 / 3D=0.1 / 5D=0.5
  IC-based early stopping (Spearman, not RMSE)
        │
        ▼
Signal Engine  (scripts/run_backtest_v2.py)
  Composite score: 0.5×pred5d + 0.3×pred3d + 0.2×pred1d
  Rank-based: top 2 tickers per day
  Quality gates: score > 0, regime_score ≥ 0.3, relative_volume ≥ 0.8
  Core-satellite: 70% SPY + 30% RSSS signals
  5-day hold, 15% take-profit cap, -8% stop-loss
        │
        ▼
Live Pipeline
  GitHub Actions → daily_run_live.py (08:30 ET, weekdays)
  Railway → FastAPI serving signals, portfolio state, predictions
  Logs → logs/paper_trades.jsonl (append-only, NEVER delete)
        │
        ▼
Dashboard  (dashboard/index.html)
  Equity curve, signals table, 1D/3D/5D predictions, PnL vs SPY
```

---

## Results

### Backtest (2024–2025 out-of-sample)

| | Value |
|---|---|
| System return | +35.6% |
| SPY benchmark | +49.7% (exceptional bull market) |
| Alpha | -12.2% (cash drag from selectivity + regime filter) |
| Sharpe | 1.32 |
| Trades | 167 |
| Win rate | 57.5% |
| Avg hold | 4.5 days |
| Stop-loss exits | 19 |
| Take-profit exits | 25 |
| Hold-expired exits | 123 |
| p-value | 0.063 (borderline, not significant at 95%) |
| 95% CI on win rate | [49.9%, 64.7%] |
| Trades needed for significance | ~42 more at current win rate |

Structure: 70% SPY core (passive) + 30% RSSS satellite (active signals).
Total return includes both components.

### Walk-Forward Validation (2019–2025, 23 quarterly folds)

| | Value |
|---|---|
| Pooled Sharpe | 0.86 |
| Folds profitable | 74% (17/23) |
| 2022 bear market return | -3.9% (controlled) |
| WFE ratio (OOS/IS Sharpe) | 1.25 — OOS > IS, not overfit |
| DSR gate | Fails — needs more data |

The walk-forward Sharpe (0.86) is the honest estimate of live performance. The
backtest Sharpe (1.32) covers only 2024–2025 — an exceptional bull market period.

### Model metrics (test set 2024–2025)

| Model | Trees | Test IC | Test dir. accuracy |
|---|---|---|---|
| model_1d_v2 | 13 | 0.041 | 53.7% |
| model_3d_v2 | 1 | 0.013 | 54.5% |
| model_5d_v2 | 28 | 0.041 | 55.8% |

Note: GKX stumps compress predictions near the mean. Low absolute IC is expected —
signal is in cross-sectional rank ordering, not magnitude.

### Honest caveats

- p = 0.063: the edge is not statistically confirmed at the 95% level
- Walk-forward Sharpe 0.86, not 1.32 — the backtest period was unusually bullish
- Negative alpha vs SPY (-12.2%) due to cash drag and regime selectivity
- Reddit API changed in 2023 (policy shift) — potential structural break in features
- Single bull-market test window (2024–2025) is insufficient for regime coverage
- Do not deploy with real capital until 200+ live trades are accumulated

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Ritiksh0h/Reddit-Sentiment-Swing-Trading-System-RSSS.git
cd Reddit-Sentiment-Swing-Trading-System-RSSS
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Copy env vars and fill in keys
cp .env.example .env
# Required: FINNHUB_API_KEY, DB_URL
# Optional: ALPHAVANTAGE_API_KEY, STOCKTWITS_TOKEN, HF_TOKEN

# Run tests (26 total — all must pass)
pytest tests/ -v --tb=short

# Dry-run the live pipeline (no trades logged)
python scripts/daily_run_live.py --dry-run

# Start local API + dashboard
uvicorn api.main:app --reload --port 8000
# Open: http://localhost:8000/dashboard
```

---

## Project Structure

```
scripts/
  build_features_v2.py        ← feature engineering (53,592 rows, 16 features)
  train_models_v2.py          ← GKX stump training with ICEarlyStopping
  run_backtest_v2.py          ← rank-based core-satellite backtester
  daily_run_live.py           ← live orchestrator (Reddit + news + ST → signals)
  append_live_features.py     ← feature logger + t+5 price backfill
  monitor_live_ic.py          ← weekly IC gate check (GREEN/AMBER/RED)

data/
  features/
    features_v2.parquet       ← 53,592 rows, 27 cols, 30 tickers, 2019–2025
    features_complete.parquet ← with news + ST merged (NEVER overwrite)
  live/
    paper_portfolio.json      ← live portfolio state
    paper_performance.jsonl   ← daily PnL snapshots
  raw/                        ← Reddit posts (932K rows, 2019–2026)

models/
  model_1d_v2.json            ← 13 trees, IC=0.041
  model_3d_v2.json            ← 1 tree,  IC=0.013
  model_5d_v2.json            ← 28 trees, IC=0.041
  training_metadata_v2.json   ← training metrics + retrain threshold

experiments/
  phase3_locked_architecture.json  ← read-only architecture contract
  backtest_v2_results.json         ← full trade log (167 trades)
  shared/
    metrics.py               ← canonical compute_ic (import from here)

api/routes/
  portfolio.py               ← /portfolio /positions /signals/recent
  predictions.py             ← /top-predictions /shap/{ticker}
  performance.py             ← /signal-accuracy /ic-monitor /model-metadata
  research.py                ← /research-findings /backtest

dashboard/index.html         ← single-file dark dashboard

.github/workflows/daily_run.yml  ← GitHub Actions automation (08:30 ET)
railway.toml                     ← Railway deployment config
logs/
  paper_trades.jsonl         ← execution audit trail (NEVER DELETE)
  ic_monitor.jsonl           ← weekly IC readings
```

---

## API Endpoints

```
GET  /health             → {"status":"ok","version":"3.0"}
GET  /status             → ran_today, n_positions, cash, system_ok
GET  /portfolio          → cash, positions, closed_trades, PnL summary
GET  /positions          → open positions with unrealized PnL
GET  /signals/recent     → last N signals from paper_trades.jsonl
GET  /trades/history     → closed trades with realized PnL
GET  /predictions        → 1D/3D/5D predictions for tracked tickers
GET  /top-predictions    → top bullish signals by composite score
GET  /shap/{ticker}      → SHAP attribution by source family
GET  /signal-accuracy    → 1D/3D/5D directional accuracy from live trades
GET  /ic-monitor         → IC readings from ic_monitor.jsonl
GET  /model-metadata     → training metrics from training_metadata_v2.json
GET  /backtest           → backtest_v2_results.json summary
GET  /dashboard          → dashboard/index.html
```

---

## Hard Rules

**Data integrity:**
- Never random train/test split — always time-based
- Never use future data in features
- Never modify `experiments/phase3_locked_architecture.json`
- Never overwrite `data/features/features_complete.parquet`
- Never delete `logs/paper_trades.jsonl`

**Model integrity:**
- Never trust Sharpe > 2.0 without checking for leakage
- Never retrain until 200+ live trades are accumulated
- Always use the density gate (post_count_1d >= 5)
- Never change the density gate without re-running signal validation
- Never retrain unless IC improvement > 0.005 over current baseline
- Never deploy without walk-forward validation

**Capital:**
- Always verify p-value before scaling capital
- Never open more than 3 positions simultaneously
- Always ATR-based sizing — never equal-weight
- Never force trades when signals are zero — cash is a valid position

**IC monitoring gates:**
- GREEN: 30-day live IC > 0.03 → continue
- AMBER: 30-day live IC 0.01–0.03 → watch closely
- RED: 30-day live IC < 0.01 → Fix 3 after **two consecutive** red weeks (never one)

---

## Disclaimer

This is a research and paper trading system. Not financial advice. Past backtest
performance does not guarantee future results. Do not use with real capital until
statistical significance is confirmed (p < 0.05, 200+ live trades).

The current edge (p = 0.063) is borderline. It may be real or it may be noise.
Accumulate more live data before drawing conclusions.

---

*Phase 4 — Paper Trading | June 2026*
*V2 GKX stumps (16 features, regime) | Rank-based backtest: 167 trades, Sharpe 1.32 (OOS)*
*Walk-forward Sharpe 0.86 (23 folds, 2019–2025) | Live since June 25, 2026*
