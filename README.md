# Reddit Sentiment Swing Trading System (RSSS)

A time-aligned numerical compression of crowd attention + market response,
trained to predict short-term return distributions inside a risk-controlled portfolio engine.

**This is not a sentiment classifier. Not a stock picker. Not a hype tracker.**

---

## Current Phase: Phase 3 — Production System

Phase 2 is complete. Experiment C is the confirmed winner. Phase 3 builds the production
signal generator, portfolio engine, and API using the Experiment C architecture.

### Phase 2 Results (final)

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| IC (test 2024) | 0.111 | > 0.05 | PASS |
| Sharpe ratio | 2.829 | > 1.0 | PASS |
| Total return | 87.6% | > SPY | PASS |
| SPY 2024 | 26.1% | benchmark | — |
| QQQ 2024 | 25.5% | benchmark | — |
| Walk-forward min IC | 0.034 | ≥ 0.03 | ACCEPTABLE |

**Winner: Experiment C — Expanded Dataset + Combined Model (XGBoost, MARKET + SENTIMENT features)**

```bash
# Reproduce Phase 2 results
python experiments/experiment_c/train.py
python experiments/compare.py
python experiments/walk_forward.py
```

---

## Development Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | Done | Signal validation — IC ≥ 0.03 confirmed on WSB data |
| 1 | Done | Feature pipeline — Reddit + market features, leakage-free |
| 2 | Done | Three-experiment architecture search, winner selected |
| **3** | **Active** | Production signal generator, portfolio engine, FastAPI |
| 4 | Pending | Paper trading + live monitoring |
| 5 | Pending | Walk-forward re-training, regime adaptation |

---

## Setup

```bash
# Python 3.11+ required
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Copy env template
cp .env.example .env
```

---

## Testing

```bash
pytest tests/ -v

# Backtest engine tests (5 tests — concentration, missing price, threshold, cap)
pytest tests/test_backtest.py -v

# Leakage tests (run before any features/alignment.py change)
pytest tests/test_alignment.py -v
```

---

## Non-Negotiable Rules

1. **No data leakage** — features for date T use only data strictly before T
2. **Time-based split** — train ≤ 2023-12-31, test ≥ 2024-01-01, never shuffle
3. **FinBERT only** — no keyword heuristics, no default fill for missing sentiment
4. **Versioned models** — never overwrite a model file in-place
5. **Thresholds are locked** — never lower IC > 0.05 or Sharpe > 1.0 to manufacture a winner

See `experiments/winner.md` for the Phase 3 architecture spec.

---

## Architecture

```
config/          — settings, thresholds, ticker/FP lists
data/            — Reddit + market loaders, feature store
features/        — alignment (critical), reddit + market feature computation
pipeline/        — feature builder, baselines, training, backtests, validation
experiments/     — A/B/C architecture search, shared backtest/trainer, compare
models/          — trainer, inference, registry
signals/         — BUY/HOLD/AVOID generator, ranking, filters
portfolio/       — position engine, sizing, risk rules
backtest/        — main loop, simulator, execution model, metrics
analytics/       — reports, benchmark comparison
api/             — FastAPI endpoints
utils/           — logger, time utils, validators
tests/           — mandatory tests (alignment + backtest engine)
```
