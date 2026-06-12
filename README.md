# Reddit Sentiment Swing Trading System (RSSS)

A time-aligned numerical compression of crowd attention + market response,
trained to predict short-term return distributions inside a risk-controlled portfolio engine.

**This is not a sentiment classifier. Not a stock picker. Not a hype tracker.**

---

## Current Phase: Phase 0 — Signal Validation

Before building any infrastructure, validate that Reddit sentiment actually has a
measurable Information Coefficient (IC ≥ 0.03) against next-day returns.

```bash
# Quick start — Phase 0 validation
pip install -r requirements.txt
python scripts/phase0_validate.py --list-datasets
python scripts/phase0_validate.py --dataset RomanBlanco/reddit_wsb_2021 --debug
python scripts/phase0_validate.py --dataset Lelon/reddit-wsb-posts
```

Output: `phase0_results/ic_report.json`

- `"overall_verdict": "PROCEED"` → IC ≥ 0.03, proceed to Phase 1
- `"overall_verdict": "ABORT"` → IC < 0.03, reassess thesis

---

## Development Phases

| Phase | Status | Description |
|-------|--------|-------------|
| **0** | **Active** | Signal validation (IC check on 10 tickers) |
| 1 | Pending | Data pipeline (Reddit PRAW + market data) |
| 2 | Pending | Feature store + leakage validation |
| 3 | Pending | Baseline XGBoost models + SHAP |
| 4 | Pending | Signal engine + portfolio logic |
| 5 | Pending | Backtesting engine (2015–2024) |
| 6 | Pending | FastAPI + paper trading |

---

## Setup

```bash
# Python 3.11+ required
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Copy env template
cp .env.example .env
# Edit .env — add POLYGON_API_KEY when available
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run leakage tests (mandatory before any alignment.py change)
pytest tests/test_alignment.py -v

# With coverage
pytest tests/ --cov=. --cov-report=term-missing
```

---

## Non-Negotiable Rules

1. **No data leakage** — features for date T use only data strictly before T
2. **Time-based split** — train ≤ 2023-12-31, test ≥ 2024-01-01, never shuffle
3. **FinBERT only** — no keyword heuristics, no default fill for missing sentiment
4. **Versioned models** — never overwrite a model file in-place
5. **If backtest Sharpe > 2.5** — assume leakage first, investigate before celebrating

See `CLAUDE.md` for full specification.

---

## Architecture

```
config/          — settings, thresholds, ticker/FP lists
data/            — Reddit + market loaders, feature store
features/        — alignment (critical), reddit + market feature computation
models/          — trainer, inference, registry
signals/         — BUY/HOLD/AVOID generator, ranking, filters
portfolio/       — position engine, sizing, risk rules
backtest/        — main loop, simulator, execution model, metrics
analytics/       — reports, benchmark comparison
api/             — FastAPI endpoints
utils/           — logger, time utils, validators
tests/           — mandatory tests (alignment tests run first)
scripts/         — phase0_validate.py, seed, retrain, backtest runner
```
