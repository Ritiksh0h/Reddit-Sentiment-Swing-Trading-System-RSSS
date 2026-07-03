# Claude Code Instructions — Audit Bug Fixes

## Baseline first

```bash
python3 -m pytest tests/ -x -q --tb=short
```

All 40 must pass before touching anything.

---

## FIX 1 — CRITICAL: MAX_POSITIONS=4 blocks bull regime slots 5 and 6

**File:** `portfolio/portfolio_engine.py`
**Problem:** `check_risk_limits()` line 125 uses the module-level constant
`MAX_POSITIONS = 4` to set `max_positions_reached`. This gates the outer
`can_open_new_trades` flag in `daily_run.py`. The inner
`get_max_positions(regime_label)` check at daily_run.py:308 is never reached
when `can_open_new_trades` is already False. Bull regime allows 6 positions
but the system is silently capped at 4 in all regimes.

**Fix:** Replace the hardcoded `MAX_POSITIONS` in `check_risk_limits` with
the regime-aware `get_max_positions()`. But `check_risk_limits` doesn't
know the current regime — so add a `regime` parameter.

In `portfolio/portfolio_engine.py`, change `check_risk_limits` signature:

```python
# Before
def check_risk_limits(state: PortfolioState, today: str) -> dict:

# After
def check_risk_limits(state: PortfolioState, today: str,
                      regime: str = 'NEUTRAL') -> dict:
```

Change line 125:
```python
# Before
'max_positions_reached': state.n_open_positions() >= MAX_POSITIONS,

# After
'max_positions_reached': state.n_open_positions() >= get_max_positions(regime),
```

In `scripts/daily_run.py`, find the `check_risk_limits` call (around line 78)
and pass the regime label:

```python
# Before
limits = check_risk_limits(state, today)

# After
_regime_for_limits = regime.label if regime else 'NEUTRAL'
limits = check_risk_limits(state, today, regime=_regime_for_limits)
```

The `regime` variable is computed just above this call — verify with:
```bash
sed -n '70,85p' scripts/daily_run.py
```

Confirm `regime` is in scope at the `check_risk_limits` call. If not, the
regime detection may be lower in the function — find the call and pass
the appropriate regime string.

Verify:
```bash
python3 -m pytest tests/ -x -q --tb=short
python3 -c "from portfolio.portfolio_engine import check_risk_limits; print('OK')"
```

---

## FIX 2 — CRITICAL: monitor_signal_decay import fails silently every Monday

**File:** `scripts/daily_run_live.py` around line 109
**Problem:** `from scripts.monitor_signal_decay import run_decay_monitor`
— this file was moved to `archive/dead_code/scripts/` in the cleanup sprint.
The Monday signal-health gate never runs.

`monitor_live_ic.py` already does the rolling IC check. The decay monitor
was superseded. Simply remove the entire Monday block.

Find the block (starts with `if _today_dt.weekday() == 0:`) and delete it.
It ends after the `if decay_result['color'] == 'RED':` warning block.

Verify the exact line range first:
```bash
grep -n "monitor_signal_decay\|run_decay_monitor\|weekday.*==.*0\|Monday: running" \
  scripts/daily_run_live.py
```

Delete from `if _today_dt.weekday() == 0:` through the closing `except` of
that try block. Keep everything before and after intact.

Verify:
```bash
python3 -c "from scripts.daily_run_live import *; print('OK')"
python3 -m pytest tests/ -x -q --tb=short
```

---

## FIX 3 — CRITICAL: RegimeDetector class doesn't exist — /portfolio always NEUTRAL

**File:** `api/routes/portfolio.py` lines 61–64
**Problem:** `RegimeDetector().get_current_regime()` — this class doesn't
exist. `regime_detector.py` exposes `classify_regime()` function only.

```python
# Before (lines 61-64)
try:
    from portfolio.regime_detector import RegimeDetector
    regime_label = RegimeDetector().get_current_regime().upper()
except Exception:
    regime_label = 'NEUTRAL'

# After
try:
    from portfolio.regime_detector import classify_regime
    _rs = classify_regime()
    regime_label = _rs.label.upper()
except Exception:
    regime_label = 'NEUTRAL'
```

Verify:
```bash
python3 -c "
from portfolio.regime_detector import classify_regime
rs = classify_regime()
print('label:', rs.label, 'multiplier:', rs.multiplier)
"
python3 -m pytest tests/ -x -q --tb=short
```

---

## FIX 4+5+6 — HIGH: Wrong field names in health.py /daily-report

**File:** `api/routes/health.py` around lines 227–249
**Problem:** Three field name mismatches vs what `execution_logger.py` writes.

Find these exact lines:
```bash
grep -n "entry_price\|rec.get('shares')\|predicted_5d\|pred_5d" api/routes/health.py
```

Apply these three replacements:

```python
# Fix 4: entry_price → fill_price
'price':  rec.get('entry_price'),       # WRONG
'price':  rec.get('fill_price'),        # CORRECT

# Fix 5: shares (doesn't exist)
'shares': rec.get('shares'),            # WRONG — field doesn't exist
'shares': round(rec.get('position_size_dollars', 0) /
                rec.get('fill_price', 1), 1) if rec.get('fill_price') else None,
                                        # CORRECT — derive from size/price

# Fix 6: predicted_5d → predicted_return_5d
'pred_5d': rec.get('predicted_5d'),     # WRONG
'pred_5d': rec.get('predicted_return_5d') or rec.get('predicted_5d'),  # CORRECT
```

Verify:
```bash
python3 -c "from api.routes.health import router; print('OK')"
python3 -m pytest tests/ -x -q --tb=short
```

---

## FIX 7 — HIGH: iloc[4] vs iloc[5] — 4-day vs 5-day return mismatch

**File:** `scripts/append_live_features.py` around line 185
**Problem:** `close_t5 = float(mkt['Close'].iloc[4])` computes a 4-day
return (index 0→4 is 4 steps). `monitor_live_ic.py` correctly uses `iloc[5]`
for a true 5-day return. Training targets and IC evaluation must use the
same horizon.

```python
# Before
close_t5 = float(mkt['Close'].iloc[4])

# After
close_t5 = float(mkt['Close'].iloc[5])
```

Also update the length check immediately above it:
```python
# Before
if len(mkt) < 5:

# After
if len(mkt) < 6:
```

The variable name `close_t5` is already correct — only the index was wrong.

Verify:
```bash
python3 -c "from scripts.append_live_features import *; print('OK')"
python3 -m pytest tests/ -x -q --tb=short
```

---

## FIX 8 — HIGH: trades_executed always zero in DB

**File:** `scripts/daily_run_live.py` around line 373
**Problem:** `summary['actions']` is a list of strings like `'OPEN NVDA pred=0.023'`,
not dicts. `isinstance(a, dict)` always False.

```python
# Before
'trades_executed': len([a for a in summary.get('actions', [])
                        if isinstance(a, dict) and a.get('action') == 'OPEN']),

# After
'trades_executed': len([a for a in summary.get('actions', [])
                        if (isinstance(a, str) and a.startswith('OPEN')) or
                           (isinstance(a, dict) and a.get('action') == 'OPEN')]),
```

Verify:
```bash
python3 -m pytest tests/ -x -q --tb=short
```

---

## FIX 9 — MEDIUM: insert_daily_run missing tickers_passed_ma column

**File:** `api/db.py`
Find `insert_daily_run` function. Add `tickers_passed_ma` to the INSERT:

```bash
grep -n "tickers_passed_ma\|insert_daily_run" api/db.py | head -20
```

In the INSERT statement, add the column and bind the value from the
`run_data` dict (key: `'tickers_passed_density'` or similar — check what
`daily_run_live.py` passes). If the key doesn't exist in `run_data`,
default to `None`.

---

## FIX 10 — MEDIUM: Wrong docstring in portfolio_engine.py

**File:** `portfolio/portfolio_engine.py` line 85
```python
# Before
"""Returns a fresh $10,000 PortfolioState if the file does not exist."""

# After
"""Returns a fresh $100,000 PortfolioState if the file does not exist."""
```

---

## FIX 11 — MEDIUM: Inverted bounds check in signal_generator.py

**File:** `portfolio/signal_generator.py` around line 259
```python
# Before
if abs(len(close)) <= abs(idx):
    break

# After
if abs(idx) >= len(close):
    break
```

---

## FIX 14 — MEDIUM: paper_trader.py doesn't create data/live/ directory

**File:** `portfolio/paper_trader.py`
```bash
grep -n "mkdir\|data/live\|PERF_JSONL\|PERF_FILE" portfolio/paper_trader.py | head -10
```

Find `Path('data').mkdir(exist_ok=True)` and replace with:
```python
Path('data/live').mkdir(parents=True, exist_ok=True)
```

---

## FIX 15 — LOW: retroactive_run.py still hard-skips below-MA tickers

**File:** `scripts/retroactive_run.py`
```bash
grep -n "ma_filter\|below_ma\|20ma\|_ma20\|continue" scripts/retroactive_run.py | head -10
```

Find any `continue` that skips a ticker because it's below its 20-day MA.
Replace with the same `below_ma20 = True` flag pattern used in
`signal_generator.py` (see that file's ma_filter_flag block for the pattern).

---

## FIX 16 — LOW: TRADE_UNIVERSE = None dangerous default

**File:** `portfolio/signal_generator.py` line 64
```python
# Before
TRADE_UNIVERSE = set(load_tickers(TICKERS_TRADE_PATH)) or None

# After
TRADE_UNIVERSE = set(load_tickers(TICKERS_TRADE_PATH))
# If empty, TRADE_UNIVERSE = set() — restricts to nothing (safe default)
# The universe check at line 376: `if TRADE_UNIVERSE and ticker not in TRADE_UNIVERSE`
# already handles empty set correctly (skips all tickers, which is safe)
```

---

## Skip FIX 13 (PCR log message) — benign, log-only, no behavior impact
## Skip FIX 17 (module-level open paths) — safe given all callers use project root

---

## Final verification

```bash
python3 -m pytest tests/ -x -q --tb=short

python3 -c "
from portfolio.portfolio_engine import check_risk_limits
from portfolio.regime_detector import classify_regime
from api.routes.health import router
from api.routes.portfolio import router as pr
from scripts.daily_run_live import main
print('All imports OK')
"
```

---

## Commit

```bash
git add -A
git commit -m "[fix] 12 audit bugs — regime cap, regime detector, field names, iloc mismatch

Critical:
- Fix MAX_POSITIONS=4 blocking bull regime slots 5-6 (pass regime to check_risk_limits)
- Remove dead monitor_signal_decay import (file archived in cleanup sprint)
- Fix RegimeDetector class missing — use classify_regime() instead

High:
- Fix entry_price/shares/predicted_5d field names in health.py /daily-report
- Fix iloc[4] → iloc[5] in append_live_features.py (4-day vs 5-day return)
- Fix trades_executed always 0 (actions list is strings not dicts)

Medium/Low:
- Fix portfolio_engine docstring \$10k → \$100k
- Fix inverted bounds check in signal_generator.py pead_proxy
- Fix paper_trader.py not creating data/live/ directory
- Fix retroactive_run.py hard-skip → below_ma20 flag (consistent with live)
- Fix TRADE_UNIVERSE or None → empty set (safe default)
- Add tickers_passed_ma to insert_daily_run

All 40 tests pass"

git push origin main
```
