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

**Core finding (Part B validated 2026-06-29):** abnormal_attention_1d
(post count normalized by ticker's own 20d history) is the primary Reddit
value driver — not raw post_count_1d and not sentiment. vix_percentile is
the single strongest feature (5/7 years |IC|>0.05, structural signal).
VADER sentiment: 1/7 years, negative IC, no Granger signal — confirms
sentiment adds no predictive value. Attention vs sentiment walk-forward:
Market + Reddit (attention) IC=0.031 vs Market + Reddit (sentiment) IC=0.006.
No retrain triggered: best combination IC=0.033 < gate=0.062 (V2 IC + 0.005).
Current V2 model (IC=0.056) remains deployed.

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
  features_v2_with_atr.parquet          ← V2 + atr_14 + atr_pct (31 cols, 100% coverage)
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
  mention_history.json          ← rolling 30-day post count history (widened from 14 on 2026-07-07)

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
  run_backtest_v2.py            ← V2: rank-based core-satellite backtest (--atr-stops flag)
  add_atr_to_features.py        ← fetch OHLC → Wilder 14-day ATR → features_v2_with_atr.parquet
  walk_forward_validation.py    ← expanding-window WFV, ICEarlyStopping, fold model save
  walk_forward_sliding.py       ← sliding-window WFV (fixed test size), same training loop
  leakage_checks.py             ← future-leak detection: autocorr + Granger + date-order checks
  universe_manager.py           ← four-stage universe: screen → liquidity → drop → approve

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
  test_phase3.py                ← 32 tests
  test_backtest.py              ← 5 tests
  test_leakage_checks.py        ← 3 tests
  (40 total — all must pass)
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
Max positions:   dynamic — 6 (bull) / 3 (bear) / 2 (choppy) — see get_max_positions()
Min pred return: 1% (0.01)
Position sizing: fractional Kelly via compute_position() — see Dynamic Risk Budget below
ATR stop:        -(2.5 × atr_pct), clamped to [-12%, -4%]; per-position, not global -8%
Slippage:        dynamic: 0.001 + 0.0005 × min(mention_growth_7d, 3.0)
Ticker cooldown: 7 days
Starting equity: $100,000 paper
```

---

## Key Architecture Notes

```
DENSITY_GATE is defined in THREE places — kept in sync manually:
  1. config/thresholds.py:37          — config/pipeline scripts
  2. portfolio/signal_generator.py:52 — live pipeline (generate_signals) ← the one live trading uses
  3. scripts/train_models_v2.py:83    — training gate (does NOT read thresholds.py)

Editing only thresholds.py is a silent no-op for live trading.
V2 was trained at gate=5. Live gate lowered to 3 on 2026-07-07 for July
volume conditions — revert when WSB daily posts consistently exceed 300.
Training gate stays 5 (deployed models trained at 5).

mention_history.json: rolling 30-day buffer (was 14 — widened 2026-07-07 so
abnormal_attention_1d's 20-day rolling average can actually hold 20 entries).
Backfilled June 8-16 2026 via scripts/backfill_mention_history.py (June 11
unrecoverable — Arctic Shift 422s on wsb pagination in that window).
```

---

## Dynamic Risk Budget (Live System)

`compute_position()` in `portfolio/position_sizer.py` — replaces the old `compute_position_size()` for new entries.

```
Fractional Kelly sizing:
  effective_risk = BASE_RISK_PCT × regime_mult × rank_decay × conf_scale
  effective_risk = min(effective_risk, BASE_RISK_PCT_MAX)
  risk_dollars   = equity × effective_risk
  size_dollars   = risk_dollars / atr_pct   (1-ATR risk distance)
  n_shares       = floor(size_dollars / price)
  THEN hard ceiling: n_shares × stop_dist ≤ BASE_RISK_PCT_MAX × equity
    (stop_dist = entry_price × abs(stop_pct), where stop_pct = -(2.5 × atr_pct))

Key constants (config/settings.py):
  BASE_RISK_PCT       = 0.005   (0.5% equity risk target per trade)
  BASE_RISK_PCT_MAX   = 0.0075  (0.75% hard ceiling at the stop)
  ATR_STOP_MULT       = 2.5
  ATR_STOP_MIN/MAX    = -0.12 / -0.04  (stop clamped between -4% and -12%)
  ATR_STOP_DEFAULT    = -0.08  (fallback for legacy positions)
  POS_CAP_HIGH/MED/LOW = 0.20 / 0.15 / 0.10  (notional cap by regime)

Regime multipliers:
  bull/positive  → mult=1.0, cap=20%  heat_budget=6%  max_pos=6
  neutral/choppy → mult=0.3, cap=15%  heat_budget=2%  max_pos=2
  bear/negative  → mult=0.5, cap=10%  heat_budget=3%  max_pos=3

Four-gate check before opening any position (daily_run.py):
  Gate 1: get_max_positions(regime) — regime-aware slot limit
  Gate 2: n_shares > 0 — sizing produced a tradeable quantity
  Gate 3: heat_budget_allows() — sum of risk_dollars ≤ regime heat budget
  Gate 4: correlation_allows() — semi-cluster cap (NVDA/AMD/MU/INTC/ARM ≤ 2)
                                  + pairwise 60-day return correlation < 0.70

ATR stop vs fixed -8% backtest comparison (run_backtest_v2.py --atr-stops):
  Fixed -8%:  return=+36.5%  Sharpe=1.36  stop-outs=16
  ATR stops:  return=+36.5%  Sharpe=1.36  stop-outs=14
  (marginal improvement; per-position stops now carry over to live system)
```

---

## Walk-Forward Validation

Two modes in `scripts/`:

```
walk_forward_validation.py  — expanding train window (all history to date)
walk_forward_sliding.py     — sliding train window (fixed lookback, e.g. 36 months)

Both use:
  ICEarlyStopping(rounds=20, X_eval=X_train, y_eval=y_train)
    Note: eval set = training data — best_iteration is optimistic; OOS fold IC is truth
  Two-phase: scout (n_estimators=200) → best_n → clean final model (n_estimators=best_n)
  Fold model JSONs saved to experiments/walk_forward_sliding/fold_models/

get_num_boosting_rounds() WARNING: returns configured n_estimators (100), NOT actual trees.
Read actual tree count from JSON: learner.gradient_booster.model.trees[]
Actual fold model tree counts: 1–62 (ICEarlyStopping working correctly)
Speed: 4 folds × 3 horizons × 2 phases = 24 fits ≈ 10 seconds (max_depth=1 stumps)
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

### V2 Research Track — GKX Stumps (retrained 2026-06-29)

```
Architecture:  GKX stumps (Gu, Kelly & Xiu 2020)
               max_depth=1, Pseudo-Huber loss, gamma=0.0 (all horizons)
               Per-horizon L2/min_child_weight: 1D=(3.0,15) / 3D=(1.0,10) / 5D=(5.0,20)
               ICEarlyStopping — val-based, no test leakage:
                 scout fits on 2022-01→2023-06, evaluates IC on val 2023-07→2023-12
                 final model trains on full 2022-2023

Features (16): regime features DROPPED — pure market + attention + news
  post_count_1d, abnormal_attention_1d, total_comments_1d, vader_sentiment_1d,
  sentiment_extremity, sentiment_accel, volume, relative_volume,
  returns_1d, returns_20d, rsi_14, news_sentiment_1d,
  vix_percentile, vix_x_volume, dist_from_20ma_pct, pead_proxy
  (spy_above_200ma / regime_score computed at inference but NOT model inputs)

Feature store: data/features/features_v2.parquet (53,592 rows, 27 cols)
               data/features/features_v2_with_atr.parquet (31 cols — adds atr_14, atr_pct)
               Use features_v2_with_atr for ATR-stop backtest (--atr-stops flag)
Density gate:  >= 5 (training and live)
Train window:  sliding 2022-2023 (excludes COVID crash + meme-stock patterns)
  train_gated: 4,301 rows  |  val_gated: 939 rows  |  test_gated: 3,797 rows

V2 model metrics after retrain (test 2024-2025):
  model_1d_v2:  IC=+0.0346  PASS ✓
  model_3d_v2:  IC=+0.0273  PASS ✓
  model_5d_v2:  IC=+0.0562  PASS ✓  (4 unique preds — 3 stumps, depth=1)
  Gate:         test_ic > 0.025  |  retrain_threshold_ic = 0.0612 (V2 IC 0.0562 + 0.005)
  Previous collapse: gamma=0.5 on 5D → 97.6% identical predictions → fixed

V2 backtest (rank-based, core-satellite 70% SPY / 30% RSSS, 2024-2025 OOS):
  Signal:   composite = 0.5×pred5d + 0.3×pred3d + 0.2×pred1d
            top 2 per day, quality gates: score>0, regime_score≥0.3, rel_vol≥0.8
  Combined: +35.6%  |  SPY: +49.7%  |  Alpha: -12.2%
  Sharpe:   1.32  |  Max DD: -14.1%
  Trades:   167   |  Win rate: 57.5%  |  p=0.063 (borderline, not significant)

V2 files:
  scripts/build_features_v2.py       ← builds features_v2.parquet (27 cols)
  scripts/add_atr_to_features.py     ← builds features_v2_with_atr.parquet (31 cols)
  scripts/train_models_v2.py         ← GKX training with val-based ICEarlyStopping
  scripts/run_backtest_v2.py         ← rank-based core-satellite backtest (--atr-stops)
  scripts/walk_forward_validation.py ← expanding-window WFV
  scripts/walk_forward_sliding.py    ← sliding-window WFV
  models/model_{1d,3d,5d}_v2.json   ← XGBoost JSON format (16 features, retrained 2026-06-29)
  models/training_metadata_v2.json   ← training metrics + gate status (validated JSON)
  models/backup_pre_retrain_20260629/ ← backup of pre-retrain models
  experiments/backtest_v2_results.json
  experiments/walk_forward_sliding/  ← fold model JSONs + results.json
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
             NOTE: began 403-ing all tickers 2026-07-01 (new API restriction).
             st_sentiment_1d and st_count_1d will be 0 on all live records.
             Zero-IC contributor — no model impact, not urgent to fix.

Market:      yfinance (period='90d' for features, '5d' for exits)
```

Tiingo is blocked (403). Do not use. yfinance is the replacement.
StockTwits is blocked (403) as of 2026-07-01. st_* features default to 0. Zero-IC contributor — no action required.

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

compute_position() risk ceiling bug (June 2026):
  Sizing used 1-ATR as risk distance; actual stop is 2.5-ATR away.
  Without ceiling, realized risk at stop = 2.5 × BASE_RISK_PCT_MAX.
  Fix: compute stop_dist = entry_price × |stop_pct| BEFORE n_shares,
       then hard ceiling: n_shares = int(max_risk_dollars / stop_dist).

BEARISH→LONG bug (June 19, 2026):
  Old daily_run.py had no signal direction guard — BEARISH signals opened long positions.
  MU: pred_5d=-7.36%, BEARISH, $2,268 long opened, -0.15% pnl, emergency close logged.
  Fix: commit 2cedebe added `if signal.signal != 'BULLISH': continue` — long-only by design.
  Long-only is correct: no borrow cost model, no short position tracking exists.

Paper equity too low (June 2026):
  At $10k, MU at $1,132 returned 0 shares — system never opened positions.
  Fix: Changed starting equity to $100,000 across paper_portfolio.json,
       portfolio_engine.py (PortfolioState default), paper_trader.py,
       and daily_run.py starting_capital.

Supabase test pollution (June 30, 2026):
  Tests wrote to Supabase via unpatched api.db.insert_trade() and api.db._exec()
  even though JSONL writes were correctly monkeypatched. The API (running under
  .venv which has psycopg2-binary) hit Supabase first in load_trades() and returned
  54 stale 2024-dated test records before ever reaching the JSONL fallback.
  System Python (terminal without venv) lacks psycopg2-binary so load_trades()
  returns [] there — making the bug invisible in manual testing.
  Fix: both test fixtures now patch all three write paths:
    monkeypatch.setattr('portfolio.execution_logger.LOG_FILE', str(log_file))
    monkeypatch.setattr('api.db.insert_trade', lambda r: True)
    monkeypatch.setattr('api.db._exec', lambda sql, params: True)
  Rule: any test that calls log_signal() MUST patch all three paths.
  Note: GitHub Actions workflow has no pytest step (runs daily_run_live.py only).
        Railway is API-only — neither runs tests against production Supabase.
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

DONE — Part B: Signal Validation Sprint (June 2026):
  ✓ validate_sources.py updated: features_v2.parquet, V2 GKX params, V2 feature families
  ✓ StockTwits excluded (zeros 2023+ corrupt walk-forward)
  ✓ Layer 1: vix_percentile strongest (5/7 yrs); abnormal_attention > post_count (3/7 vs 0/7)
  ✓ Layer 2: all features WEAK_SIGNAL (0/7 Granger-significant except 2025 marginal)
  ✓ Layer 3: attention > sentiment (0.031 vs 0.006 walk-forward IC)
  ✓ NO RETRAIN: best walk-forward IC = 0.033 < gate (0.056 + 0.005 = 0.061)
  ✓ experiments/source_validation/results.json updated with V2 results

DONE — Dynamic Risk Budget + ATR Stops Sprint (June 2026):
  ✓ add_atr_to_features.py: Wilder 14-day ATR → features_v2_with_atr.parquet (31 cols, 100% cov)
  ✓ compute_position() in position_sizer.py: fractional Kelly with ATR-derived stop
      BASE_RISK_PCT=0.005, ceiling at BASE_RISK_PCT_MAX=0.0075, rank decay, conf scaling
  ✓ get_max_positions() / heat_budget_allows() / correlation_allows() in portfolio_engine.py
  ✓ daily_run.py: four-gate check (max_pos, n_shares>0, heat_budget, correlation)
  ✓ run_backtest_v2.py: --atr-stops flag; ATR ≈ fixed -8% (+36.5% Sharpe 1.36, stop-outs 14 vs 16)
  ✓ walk_forward_validation.py + walk_forward_sliding.py: expanding + sliding WFV
  ✓ leakage_checks.py: autocorr + Granger + date-order leak detection
  ✓ universe_manager.py: four-stage universe (screen → liquidity → drop → approve)
  ✓ compute_position() risk ceiling bug fixed (2.5× ATR stop distance, not 1×)
  ✓ Paper equity: $10,000 → $100,000 (MU at $1,132 gets 0 shares at $10k)
  ✓ Tests: 40 total (32 test_phase3 + 5 test_backtest + 3 test_leakage_checks)

Priority 1 — NEXT (Claude Code):
  Part C: Dashboard wiring (API endpoints already exist)
    C1: /shap/{ticker} — verify SHAP attribution renders in dashboard
    C2: /signal-accuracy — verify 1D/3D/5D accuracy panel renders
        Color: green>=55%, amber>=50%, red<50%
    C3: /research-findings — wire results.json to research panel (Part B done)

Priority 2 — September retrain (after 30+ days live IC):
  Retrain V2 with attention-only Reddit features (drop vader_sentiment_1d)
  ONLY if: new walk-forward IC > current V2 IC (0.056) + 0.005 = 0.061
  Also consider: drop sentiment_extremity, sentiment_accel (confirmed no signal)

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
  NEVER equal-weight sizing — always ATR-based via compute_position()
  NEVER flat slippage — always dynamic
  NEVER skip four-gate check (max_pos, n_shares, heat_budget, correlation)
  Max positions is regime-dynamic: bull=6 / bear=3 / choppy=2
  NEVER force trades on zero signals — hold cash is correct
  Long-only: BEARISH + NEUTRAL logged for IC monitoring, NEVER traded
             No short infrastructure — BEARISH = skip, not short (daily_run.py:298)

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
*Updated: Dynamic risk-budget engine live (fractional Kelly, ATR stops, heat budget, correlation gates).*
*Paper equity $100k. 40 tests passing. Walk-forward sliding + leakage_checks + universe_manager added.*
*V2 NOT in live system (gate: IC > 0.0796+0.005). get_num_boosting_rounds() misleading — read JSON for actual tree count.*
