# Claude Code — Cleanup Sprint
# Reddit Sentiment Swing Trading System (RSSS)
# Goal: readable, organized, maintainable codebase

---

## Rules Before Starting

- Run `pytest tests/ -v --tb=short` FIRST — confirm 26/26 passing
- After EVERY deletion or move, run pytest again
- NEVER delete logs/paper_trades.jsonl
- NEVER delete experiments/phase3_locked_architecture.json
- NEVER delete models/registry/*.pkl
- NEVER modify portfolio/ or scripts/daily_run_live.py logic
- NEVER modify config/thresholds.py constants
- Push only after final 26/26 passing

---

## Phase 1 — Delete Dead Files (no logic changes)

### 1a — Delete stub files (9-line TODO-only files)

```bash
rm models/train.py
rm models/registry.py
rm models/predict.py
rm scripts/retrain.py
rm api/routes/signals.py
rm api/routes/predictions.py
rm api/routes/performance.py
```

Verify nothing imports them:
```bash
grep -rn "from models.train\|from models.registry\|from models.predict\|from scripts.retrain\|from api.routes" \
  --include="*.py" -not -path "./.venv/*"
```
Expected: no results. If anything imports them, stop and investigate.

### 1b — Delete dead API

```bash
rm data/tiingo_fetcher.py
```

Verify nothing imports it:
```bash
grep -rn "tiingo_fetcher\|tiingo" --include="*.py" -not -path "./.venv/*"
```
Expected: no results.

### 1c — Delete superseded daily_run.py

Wait — daily_run.py is still imported by daily_run_live.py:
```bash
grep -n "from scripts.daily_run\|import daily_run" scripts/daily_run_live.py
```

If imported: do NOT delete yet. Move to Phase 3 (refactor).
If NOT imported: delete it:
```bash
rm scripts/daily_run.py
```

### 1d — Delete duplicate standalone backtest

```bash
# Verify nothing imports scripts/run_backtest.py
grep -rn "from scripts.run_backtest\|import run_backtest" \
  --include="*.py" -not -path "./.venv/*"
```
If nothing imports it:
```bash
rm scripts/run_backtest.py
```

### 1e — Delete Jupyter artifacts

```bash
rm -rf scripts/.ipynb_checkpoints/
rm -rf .ipynb_checkpoints/
rm -f Untitled.ipynb
```

Do NOT delete the actual .ipynb files — archive them instead:
```bash
mkdir -p archive/notebooks
mv phase0_colab.ipynb archive/notebooks/
mv phase0_signal_validation.ipynb archive/notebooks/
mv experiment_c_data_expansion.ipynb archive/notebooks/
mv scripts/stocktwits_archive_processing.ipynb archive/notebooks/
mv scripts/fnspid_news_processing_local.ipynb archive/notebooks/
mv scripts/fnspid_news_processing.ipynb archive/notebooks/
```

### 1f — Delete planning markdown docs (implemented, no longer needed)

```bash
rm -f ARCHITECTURE_IMPROVEMENTS.md
rm -f DASHBOARD_ADDITIONS.md
rm -f DASHBOARD_REPLACEMENT.md
rm -f LIVE_FEATURE_STORE.md
```

Keep: README.md, CLAUDE.md, experiments/winner.md

### 1g — Delete empty api/routes/ directory

```bash
rmdir api/routes/ 2>/dev/null || echo "routes/ still has files — check first"
```

### 1h — Run tests after all deletions

```bash
pytest tests/ -v --tb=short
```
Must still be 26/26. If any test fails — stop and fix before continuing.

---

## Phase 2 — Fix Stale Paths in config/settings.py

Read config/settings.py first:
```bash
cat config/settings.py
```

Find and remove/fix these four stale paths:
- `FEATURE_STORE_PATH` pointing to `data/feature_store/` (deleted)
- `PHASE0_RESULTS_PATH` pointing to `phase0_results/` (deleted)
- `REPORTS_DIR` pointing to `reports/` (deleted)
- `FEATURES_PARQUET` pointing to `data/features/features.parquet` (wrong name)

For each stale path:
- If nothing imports it: delete the line
- If something imports it: update to correct path

Correct paths are:
```python
FEATURES_FULL_PATH     = 'data/features/features_full.parquet'
FEATURES_COMPLETE_PATH = 'data/features/features_complete.parquet'
```

Remove `CLEAN_FEATURES` alias from config/thresholds.py if it just
duplicates `PHASE3_FEATURES`. Check what imports it first:
```bash
grep -rn "CLEAN_FEATURES" --include="*.py" -not -path "./.venv/*"
```
If only experiment files use it (not live code): remove the alias,
update callers to use `PHASE3_FEATURES` directly.

Run tests after:
```bash
pytest tests/ -v --tb=short
```

---

## Phase 3 — Fix compute_ic Duplication (5 definitions → 1)

`compute_ic` is defined in 5 separate files. Canonical home:
`experiments/shared/metrics.py`

### Step 3a — Add canonical compute_ic to experiments/shared/metrics.py

Read the file first, then add if not already there:

```python
def compute_ic(
    df: pd.DataFrame,
    features: list[str],
    target: str = 'target_return_5d',
    xgb_params: dict | None = None,
) -> float:
    """
    Compute Spearman IC for given feature set using XGBoost.
    Canonical implementation — import from here, never redefine.
    """
    import xgboost as xgb
    from scipy import stats

    avail = [f for f in features if f in df.columns]
    if not avail or target not in df.columns:
        return float('nan')

    params = xgb_params or dict(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.6, colsample_bytree=0.6, min_child_weight=20,
        reg_alpha=0.5, reg_lambda=2.0, random_state=42, n_jobs=1,
    )
    model = xgb.XGBRegressor(**params)
    X = df[avail].fillna(0)
    y = df[target]
    model.fit(X, y)
    pred = model.predict(X)
    return float(stats.spearmanr(pred, y).correlation)
```

### Step 3b — Replace the 5 duplicate definitions with imports

In each of these files, find the local `compute_ic` definition
and replace it with:
```python
from experiments.shared.metrics import compute_ic
```

Files to update:
- pipeline/03_train_models.py
- pipeline/05_validate_alpha.py
- pipeline/06_feature_importance.py
- experiments/shared/trainer.py
- experiments/source_validation/validate_sources.py

For each file: read it, find the local `def compute_ic(`, remove
the entire function body, add the import at the top.

Run tests after:
```bash
pytest tests/ -v --tb=short
```

---

## Phase 4 — Split api/main.py (842 lines → modules)

api/main.py has 20+ endpoints, SHAP logic, backtest aggregation,
and file I/O all in one 842-line file. Split into route modules.

### Step 4a — Read api/main.py completely before touching it

```bash
cat api/main.py
```

### Step 4b — Create route modules

Create these files with the endpoints moved from api/main.py:

**api/routes/health.py** — system status endpoints
```
GET /health
GET /status
GET /settings
POST /settings
```

**api/routes/portfolio.py** — portfolio and position endpoints
```
GET /portfolio
GET /positions
GET /signals/recent
GET /signals
GET /trades/history
```

**api/routes/predictions.py** — signal and prediction endpoints
```
GET /predictions
GET /top-predictions
GET /shap/{ticker}
```

**api/routes/performance.py** — performance and accuracy endpoints
```
GET /signal-accuracy
GET /performance
GET /ic-monitor
GET /backfill-log
POST /backfill
GET /model-metadata
```

**api/routes/research.py** — research and backtest endpoints
```
GET /research-findings
GET /backtest
GET /backtest-full
```

### Step 4c — Create the route files

Each route file should follow this pattern:

```python
"""
RSSS API — [section name] routes
"""
from fastapi import APIRouter
from pathlib import Path
import json

router = APIRouter()

# Move relevant endpoints here from api/main.py
# Keep all existing logic — just move it
```

### Step 4d — Update api/main.py to use routers

After creating all route files, update api/main.py to:

```python
"""
RSSS FastAPI application — entry point.
Routes are organized in api/routes/*.py
"""
import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path

from api.routes.health      import router as health_router
from api.routes.portfolio   import router as portfolio_router
from api.routes.predictions import router as predictions_router
from api.routes.performance import router as performance_router
from api.routes.research    import router as research_router

app = FastAPI(title='RSSS — Reddit Sentiment Swing Trading System')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(health_router)
app.include_router(portfolio_router)
app.include_router(predictions_router)
app.include_router(performance_router)
app.include_router(research_router)

@app.get('/dashboard')
def serve_dashboard():
    return FileResponse('dashboard/index.html')
```

### Step 4e — Add __init__.py to api/routes/

```bash
touch api/routes/__init__.py
```

### Step 4f — Test after split

```bash
# Start API and verify all endpoints still work
uvicorn api.main:app --port 8001 --log-level warning &
sleep 3
curl http://localhost:8001/status | python3 -m json.tool
curl http://localhost:8001/portfolio | python3 -m json.tool | head -5
curl http://localhost:8001/predictions | python3 -m json.tool | head -5
curl http://localhost:8001/signal-accuracy | python3 -m json.tool | head -5
curl http://localhost:8001/research-findings | python3 -m json.tool | head -5
kill %1  # stop test server

pytest tests/ -v --tb=short
```

All 26+ tests must pass. All 5 endpoints must return HTTP 200.

---

## Phase 5 — Add __init__.py to Experiment Subdirectories

```bash
touch experiments/layer1_signal_existence/__init__.py
touch experiments/layer2_regime/__init__.py
touch experiments/layer3_model/__init__.py
touch experiments/source_validation/__init__.py
```

---

## Phase 6 — Move Live Data Files to data/live/

```bash
mkdir -p data/live
```

These files change daily and should be separate from static data:
```bash
# Only move if they exist
mv data/paper_portfolio.json data/live/ 2>/dev/null
mv data/paper_performance.jsonl data/live/ 2>/dev/null
```

Update all references to these files. Search first:
```bash
grep -rn "paper_portfolio.json\|paper_performance.jsonl" \
  --include="*.py" -not -path "./.venv/*"
```

Update each reference from `data/paper_portfolio.json` to
`data/live/paper_portfolio.json` and similarly for performance.

Files to update include: api/main.py (or new route files),
portfolio/paper_trader.py, portfolio/execution_logger.py,
scripts/daily_run.py (if still exists), scripts/daily_run_live.py.

Run tests after:
```bash
pytest tests/ -v --tb=short
```

---

## Phase 7 — Add Docstrings to Key Functions

Add a one-line docstring to every public function that lacks one
in these files (do NOT change logic, only add documentation):

**portfolio/signal_generator.py:**
- `compute_features_live()` — explain inputs/outputs
- `generate_signals()` — explain density gate and return format

**portfolio/portfolio_engine.py:**
- `check_exits()` — document 3-priority exit order
- `load_portfolio()` / `save_portfolio()` — document file path

**portfolio/drift_monitor.py:**
- `check_drift()` — document what triggers SKIP_DAY

**portfolio/regime_detector.py:**
- `classify_regime()` — document SPY logic and return values

**scripts/daily_run_live.py:**
- `main()` — document the 3-source fetch → signal → trade flow

**scripts/append_live_features.py:**
- `fill_pending_targets()` — document the t+5 price fetch logic
- `load_today_feature_vectors()` — document source (paper_trades.jsonl)

Format:
```python
def function_name(args) -> return_type:
    """
    One sentence describing what this function does.

    Args:
        arg1: description
        arg2: description

    Returns:
        description of return value
    """
```

---

## Phase 8 — Update README.md

Rewrite README.md to reflect current state. Structure:

```markdown
# RSSS — Reddit Sentiment Swing Trading System

## What It Is
2-paragraph description. One sentence: "This is NOT a sentiment
classifier — it is a time-aligned attention + momentum system."

## Research Question
Can Reddit attention + news + StockTwits predict short-term
stock returns? Answer: [from source validation results]

## Current Status
Phase 4 — Paper Trading (live since Jun 15, 2026)
Model: XGBoost, 14 features, IC=0.0796, dir_acc=52.4%

## Architecture (8-stage pipeline)
Brief description of each stage with file references

## Quick Start
5 commands to get running

## Project Structure
CURRENT folder map — what actually exists after cleanup

## Key Findings
3-4 bullet points from source validation

## Automation Schedule (EDT)
09:00 ET / 11:30 ET / 14:00 ET Mon-Fri

## Hard Rules
Condensed from CLAUDE.md
```

---

## Phase 9 — Update CLAUDE.md

Update these sections to reflect post-cleanup reality:

1. **Project Structure** — update to new folder map
2. **API Endpoints** — update to reflect new route organization
3. **Automation** — confirm EDT times (not IST)
4. **Pending Work** — remove completed items, add real next steps:
   - AV backfill completing (~12 days)
   - Retrain after backfill
   - 60-day live data milestone
   - Docker automation

---

## Final Verification

```bash
# 1. All tests pass
pytest tests/ -v --tb=short

# 2. API starts and all endpoints work
uvicorn api.main:app --port 8001 --log-level warning &
sleep 3
for endpoint in status portfolio predictions signal-accuracy research-findings; do
  echo "Testing /$endpoint:"
  curl -s http://localhost:8001/$endpoint | python3 -m json.tool | head -3
done
kill %1

# 3. Dry run works
python scripts/daily_run_live.py --dry-run 2>&1 | \
  grep -v "httpx\|HF_TOKEN\|Loading\|huggingface\|Redirect" | tail -20

# 4. No dead imports
python3 -c "
import api.main
import portfolio.signal_generator
import portfolio.portfolio_engine
import portfolio.drift_monitor
import scripts.daily_run_live
print('All imports OK')
"

# 5. Git status clean (only expected files modified)
git status

# 6. Push
bash push.sh "[cleanup] remove dead code, split api, canonical compute_ic, docstrings, update docs"
```

---

## Expected Final Folder Structure

```
config/
  settings.py          ← fixed stale paths
  thresholds.py        ← removed CLEAN_FEATURES alias
  tickers.txt
  false_positive_list.txt

data/
  raw/                 ← parquet archives
  features/            ← feature stores
  processed/           ← news, ST processed data
  live/                ← paper_portfolio.json, paper_performance.jsonl
  reddit_live_fetcher.py
  news_fetcher.py
  stocktwits_fetcher.py
  # tiingo_fetcher.py DELETED

portfolio/             ← live trading engine (unchanged logic)
  signal_generator.py
  position_sizer.py
  regime_detector.py
  portfolio_engine.py
  execution_logger.py
  drift_monitor.py
  paper_trader.py

api/
  main.py              ← thin entry point, includes routers
  routes/
    __init__.py
    health.py
    portfolio.py
    predictions.py
    performance.py
    research.py

scripts/
  daily_run_live.py    ← live orchestrator
  monitor_live_ic.py
  append_live_features.py
  merge_external_features.py
  collect_av_news.py
  train_phase3_model.py
  fix3_switch_to_17_features.py
  test_historical_run.py
  # daily_run.py DELETED (superseded)
  # run_backtest.py DELETED (duplicate)
  # retrain.py DELETED (stub)

models/
  registry/            ← pkl files (gitignored)
  # train.py DELETED (stub)
  # registry.py DELETED (stub)
  # predict.py DELETED (stub)

experiments/           ← read-only archive
  phase3_locked_architecture.json
  winner.md
  backtest_results.json
  experiment_c/
  layer1_signal_existence/
  layer2_regime/
  layer3_model/
  shared/
  source_validation/
  signal_validation_summary.py  ← consider deleting
  # signal_validation_summary.py is a one-time script, can delete

pipeline/              ← historical research pipeline
  01_feature_builder.py
  02_run_baselines.py
  03_train_models.py
  04_run_backtests.py
  05_validate_alpha.py
  06_feature_importance.py
  feature_schema.py

utils/
  logger.py
  time_utils.py
  validators.py

tests/
  test_phase3.py
  test_backtest.py

archive/
  notebooks/           ← moved .ipynb files

logs/                  ← never touch
dashboard/
  index.html
README.md
CLAUDE.md
```

---

## Build Order (strict — do not skip steps)

```
Phase 1: Delete dead files → pytest (26/26)
Phase 2: Fix stale settings paths → pytest (26/26)
Phase 3: Canonical compute_ic → pytest (26/26)
Phase 4: Split api/main.py → pytest + endpoint test (26/26)
Phase 5: Add __init__.py → pytest (26/26)
Phase 6: Move live data files → pytest (26/26)
Phase 7: Add docstrings (no logic change) → pytest (26/26)
Phase 8: Update README.md
Phase 9: Update CLAUDE.md
Final:   Full verification → push
```

Stop at any phase where tests fail. Fix before continuing.
Never skip a test run between phases.
