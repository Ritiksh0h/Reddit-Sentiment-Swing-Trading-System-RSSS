# Claude Code — Architecture Improvements
# Signal Thresholds + Stop-Loss + Regime Detector
# Reddit Sentiment Swing Trading System (RSSS)

---

## Context

Three targeted improvements to make the live system actually trade
with conviction instead of generating only NEUTRAL signals.

```
Problem 1: All signals NEUTRAL
  MIN_PRED_RET = 0.01 (1%) in signal_generator.py
  Model average prediction ≈ 1.5-2.5%
  BULLISH threshold was 3% — almost never reached
  Result: portfolio opens positions on vague low-conviction signals

Problem 2: No stop-loss
  check_exits() in portfolio_engine.py only has:
    - hold_period_expired (5 days)
    - take_profit_cap (15%)
  No stop-loss at all — bad trades held full 5 days

Problem 3: Regime detector still uses period='300d'
  The SPY retry fix added fallback periods but the primary
  is still '300d' which yfinance rejects (invalid period string)
  Should be '1y' as primary
```

---

## Session Start

```bash
git pull origin main
source .venv/bin/activate

# Check current signal threshold
grep -n "MIN_PRED_RET\|BULLISH\|BEARISH\|signal" portfolio/signal_generator.py | head -10

# Check current exits
grep -n "take_profit\|stop_loss\|hold_period\|exit_reason" portfolio/portfolio_engine.py | head -10

# Check regime detector period
grep -n "period=\|300d\|1y\|retry" portfolio/regime_detector.py | head -10

# Check current model prediction distribution
python3 -c "
import pandas as pd, pickle, json
import numpy as np
df = pd.read_parquet('data/features/features_complete.parquet')
df = df[df['post_count_1d'] >= 10]
with open('models/registry/model_5d.pkl', 'rb') as f:
    model = pickle.load(f)
with open('experiments/phase3_locked_architecture.json') as f:
    arch = json.load(f)
feats = arch['features']
avail = [f for f in feats if f in df.columns]
X = df[avail].fillna(0)
preds = model.predict(X)
print(f'Prediction distribution:')
print(f'  Mean:   {preds.mean():.4f}')
print(f'  Median: {np.median(preds):.4f}')
print(f'  Std:    {preds.std():.4f}')
print(f'  P75:    {np.percentile(preds, 75):.4f}')
print(f'  P80:    {np.percentile(preds, 80):.4f}')
print(f'  P90:    {np.percentile(preds, 90):.4f}')
print(f'  >1%:    {(preds > 0.01).mean()*100:.1f}%')
print(f'  >1.5%:  {(preds > 0.015).mean()*100:.1f}%')
print(f'  >2%:    {(preds > 0.02).mean()*100:.1f}%')
print(f'  >3%:    {(preds > 0.03).mean()*100:.1f}%')
"
```

This shows the actual prediction distribution and helps pick
the right threshold. Read all outputs before changing anything.

---

## Task 1 — Fix Signal Thresholds

### What to change in portfolio/signal_generator.py

**Step 1a — Read the file first**

```bash
cat portfolio/signal_generator.py
```

**Step 1b — Update MIN_PRED_RET and add signal classification**

Find these lines near the top:
```python
MIN_PRED_RET = 0.01
```

Replace with:
```python
# Signal thresholds — calibrated to model prediction distribution
# Model mean prediction ≈ 1.5-2.5%, so 3% threshold was never reached
# These generate BULLISH/BEARISH signals for top/bottom of predictions
MIN_PRED_RET     = 0.005   # minimum to even consider (filter noise)
BULLISH_THRESHOLD = 0.015  # pred >= 1.5% → BULLISH
BEARISH_THRESHOLD = -0.015 # pred <= -1.5% → BEARISH
```

**Step 1c — Add signal classification to SignalRecord**

Find the SignalRecord dataclass and add a `signal` field:

```python
@dataclass
class SignalRecord:
    ticker:            str
    date:              str
    predicted_return:  float
    signal:            str    # 'BULLISH' | 'BEARISH' | 'NEUTRAL'
    confidence:        float  # 0.0 to 1.0
    feature_vector:    dict
    post_count_1d:     int
    atr_14:            float
    price:             float
    signal_timestamp:  str
```

**Step 1d — Update generate_signals() to classify and compute confidence**

Find the section that filters predictions and appends to signals.
Currently:
```python
        if pred < MIN_PRED_RET:
            continue

        signals.append(SignalRecord(
            ticker=ticker,
            ...
        ))
```

Replace with:
```python
        # Filter absolute noise
        if abs(pred) < MIN_PRED_RET:
            continue

        # Classify signal
        if pred >= BULLISH_THRESHOLD:
            signal = 'BULLISH'
        elif pred <= BEARISH_THRESHOLD:
            signal = 'BEARISH'
        else:
            signal = 'NEUTRAL'

        # Confidence: how far above the threshold is the prediction?
        # Scale: threshold=1.0 → conf=0.0, 3×threshold → conf=1.0
        confidence = min(
            abs(pred) / (BULLISH_THRESHOLD * 2),
            1.0
        )

        signals.append(SignalRecord(
            ticker=ticker,
            date=today,
            predicted_return=round(pred, 6),
            signal=signal,
            confidence=round(confidence, 4),
            feature_vector=features,
            post_count_1d=post_count,
            atr_14=round(atr, 6),
            price=round(price, 4),
            signal_timestamp=ts,
        ))
```

**Step 1e — Update the sort and log to show signal breakdown**

Replace:
```python
    signals.sort(key=lambda s: s.predicted_return, reverse=True)
    logger.info(f'signals_generated count={len(signals)} date={today}')
```

With:
```python
    # Sort: BULLISH first (highest pred), then NEUTRAL, then BEARISH
    bullish = sorted([s for s in signals if s.signal == 'BULLISH'],
                     key=lambda s: s.predicted_return, reverse=True)
    neutral = sorted([s for s in signals if s.signal == 'NEUTRAL'],
                     key=lambda s: abs(s.predicted_return), reverse=True)
    bearish = sorted([s for s in signals if s.signal == 'BEARISH'],
                     key=lambda s: s.predicted_return)
    signals = bullish + neutral + bearish

    logger.info(
        f'signals_generated count={len(signals)} '
        f'bullish={len(bullish)} neutral={len(neutral)} '
        f'bearish={len(bearish)} date={today}'
    )
```

**Step 1f — Update execution_logger.py to save signal and confidence**

In portfolio/execution_logger.py, find the `log_signal()` function.
Add `signal` and `confidence` parameters with defaults so existing
callers don't break:

```python
def log_signal(
    ...,
    signal:     str   = 'NEUTRAL',   # ← add
    confidence: float = 0.0,          # ← add
    ...,
) -> None:
```

Add them to the record dict inside the function:
```python
    record = {
        ...,
        'signal':     signal,
        'confidence': confidence,
        ...,
    }
```

**Step 1g — Update daily_run.py to pass signal and confidence**

In scripts/daily_run.py, find where `log_signal()` is called for
OPEN actions. Pass the signal and confidence from the SignalRecord:

```python
log_signal(
    ...,
    signal=signal.signal,
    confidence=signal.confidence,
    ...,
)
```

---

## Task 2 — Add Stop-Loss

### What to change in portfolio/portfolio_engine.py

**Step 2a — Read the file first**

```bash
cat portfolio/portfolio_engine.py
```

**Step 2b — Add STOP_LOSS_PCT constant**

Near the top where other constants are defined, add:

```python
STOP_LOSS_PCT = -0.08   # -8% stop-loss (cuts losing trades before 5-day expire)
```

Also add it to config/thresholds.py in the Portfolio section:
```python
STOP_LOSS_PCT: float     = -0.08   # exit if unrealized loss >= 8%
```

**Step 2c — Add stop-loss check to check_exits()**

In the `check_exits()` function, find the existing exit conditions
and add the stop-loss check BEFORE the hold-period check:

```python
def check_exits(
    state: PortfolioState,
    current_prices: dict,
    today: str,
) -> list:
    """
    Check all open positions for exit conditions.

    Exit conditions (checked in priority order):
        1. Stop-loss hit (unrealized loss >= 8%) — cut losses immediately
        2. Take-profit cap hit (unrealized gain >= 15%)
        3. Hold period expired (stop_date reached)
    """
    to_close = []

    for pos in state.positions:
        price = current_prices.get(pos.ticker)
        if price is None:
            continue

        unrealized_return = (price - pos.entry_price) / pos.entry_price

        # 1. Stop-loss (highest priority — protect capital)
        if unrealized_return <= STOP_LOSS_PCT:
            to_close.append({
                'position':    pos,
                'exit_price':  price,
                'exit_date':   today,
                'exit_reason': 'stop_loss',
                'pnl_pct':     round(unrealized_return, 4),
            })
            logger.info(
                f'stop_loss_triggered ticker={pos.ticker} '
                f'unrealized={unrealized_return:.2%} '
                f'threshold={STOP_LOSS_PCT:.0%}'
            )
            continue

        # 2. Take-profit cap
        if unrealized_return >= TAKE_PROFIT_CAP:
            to_close.append({
                'position':    pos,
                'exit_price':  price,
                'exit_date':   today,
                'exit_reason': 'take_profit_cap',
                'pnl_pct':     round(unrealized_return, 4),
            })
            continue

        # 3. Hold period expired
        if today >= pos.stop_date:
            to_close.append({
                'position':    pos,
                'exit_price':  price,
                'exit_date':   today,
                'exit_reason': 'hold_period_expired',
                'pnl_pct':     round(unrealized_return, 4),
            })

    return to_close
```

---

## Task 3 — Fix Regime Detector SPY Period

### What to change in portfolio/regime_detector.py

**Step 3a — Read the file first**

```bash
cat portfolio/regime_detector.py
```

**Step 3b — Fix the SPY download period**

Find:
```python
    spy = yf.download(spy_ticker, period='300d',
                      auto_adjust=True, progress=False)
```

The issue: '300d' is not a valid yfinance period string.
Valid strings: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y'

Replace with a retry loop that tries valid periods in order:

```python
    spy = pd.DataFrame()
    for period in ['1y', '6mo', '3mo']:
        try:
            spy = yf.download(
                spy_ticker, period=period,
                auto_adjust=True, progress=False
            )
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)
            if not spy.empty and len(spy) >= 60:
                logger.debug(f'SPY data loaded period={period} rows={len(spy)}')
                break
        except Exception as e:
            logger.warning(f'SPY download failed period={period}: {e}')
            spy = pd.DataFrame()

    if spy.empty or len(spy) < 60:
        logger.warning('SPY data unavailable after all retries — defaulting to NEUTRAL')
        return RegimeState(
            label='neutral',
            multiplier=POSITION_SIZING['neutral'],
            spy_above_200ma=None,
            spy_ret_60d=None,
            rolling_30d_ic=rolling_30d_ic,
            reason='SPY data unavailable — default neutral',
        )
```

**Step 3c — Verify regime fires correctly**

After fixing, add a test:
```bash
python3 -c "
from portfolio.regime_detector import classify_regime
regime = classify_regime()
print(f'Regime: {regime.label}')
print(f'Multiplier: {regime.multiplier}')
print(f'Reason: {regime.reason}')
print(f'SPY above 200MA: {regime.spy_above_200ma}')
print(f'SPY 60d return: {regime.spy_ret_60d}')
"
```

Expected in current bull market (Jun 2026):
```
Regime: positive
Multiplier: 1.0
Reason: SPY above 200MA, 60d positive, IC positive or unknown
```

If it shows NEUTRAL — the SPY data may still be failing.
Check: `python3 -c "import yfinance as yf; print(yf.download('SPY', period='1y').tail(3))"`

---

## Task 4 — Update Tests

The stop-loss and signal classification need tests.

In tests/test_phase3.py, add:

```python
def test_stop_loss_triggers_at_8pct():
    """Stop-loss must close position when unrealized loss >= 8%."""
    from portfolio.portfolio_engine import check_exits, PortfolioState, Position
    from datetime import date, timedelta

    today = date.today().isoformat()
    future = (date.today() + timedelta(days=3)).isoformat()

    pos = Position(
        ticker='NVDA',
        entry_price=100.0,
        n_shares=10,
        entry_date=today,
        stop_date=future,       # not expired yet
        predicted_return=0.02,
        atr_14=2.0,
        regime_state='positive',
        regime_multiplier=1.0,
        feature_vector={},
        slippage_applied=0.001,
    )
    state = PortfolioState(cash=9000.0, positions=[pos])

    # Price dropped 9% → should trigger stop-loss
    exits = check_exits(state, {'NVDA': 91.0}, today)
    assert len(exits) == 1
    assert exits[0]['exit_reason'] == 'stop_loss'
    assert exits[0]['pnl_pct'] == pytest.approx(-0.09, abs=0.001)


def test_stop_loss_does_not_trigger_at_7pct():
    """Stop-loss should NOT fire at 7% loss (below 8% threshold)."""
    from portfolio.portfolio_engine import check_exits, PortfolioState, Position
    from datetime import date, timedelta

    today = date.today().isoformat()
    future = (date.today() + timedelta(days=3)).isoformat()

    pos = Position(
        ticker='TSLA',
        entry_price=100.0,
        n_shares=5,
        entry_date=today,
        stop_date=future,
        predicted_return=0.02,
        atr_14=3.0,
        regime_state='neutral',
        regime_multiplier=0.75,
        feature_vector={},
        slippage_applied=0.001,
    )
    state = PortfolioState(cash=9500.0, positions=[pos])

    # Price dropped 7% → below stop-loss threshold, should NOT trigger
    exits = check_exits(state, {'TSLA': 93.0}, today)
    assert len(exits) == 0


def test_signal_classified_bullish():
    """Predictions >= 1.5% should classify as BULLISH."""
    from portfolio.signal_generator import BULLISH_THRESHOLD, BEARISH_THRESHOLD
    pred = 0.020
    signal = 'BULLISH' if pred >= BULLISH_THRESHOLD else (
             'BEARISH' if pred <= BEARISH_THRESHOLD else 'NEUTRAL')
    assert signal == 'BULLISH'


def test_signal_classified_bearish():
    """Predictions <= -1.5% should classify as BEARISH."""
    from portfolio.signal_generator import BULLISH_THRESHOLD, BEARISH_THRESHOLD
    pred = -0.020
    signal = 'BULLISH' if pred >= BULLISH_THRESHOLD else (
             'BEARISH' if pred <= BEARISH_THRESHOLD else 'NEUTRAL')
    assert signal == 'BEARISH'


def test_signal_classified_neutral():
    """Predictions between -1.5% and +1.5% should be NEUTRAL."""
    from portfolio.signal_generator import BULLISH_THRESHOLD, BEARISH_THRESHOLD
    pred = 0.010
    signal = 'BULLISH' if pred >= BULLISH_THRESHOLD else (
             'BEARISH' if pred <= BEARISH_THRESHOLD else 'NEUTRAL')
    assert signal == 'NEUTRAL'
```

---

## Task 5 — Dry Run to Verify

```bash
# Run all tests (should be 25/25 now with new tests)
pytest tests/ -v --tb=short

# Verify regime fires correctly
python3 -c "
from portfolio.regime_detector import classify_regime
r = classify_regime()
print(f'Regime: {r.label} (multiplier={r.multiplier})')
print(f'Reason: {r.reason}')
"

# Dry run to check signal classification appears in logs
python scripts/daily_run_live.py --dry-run 2>&1 | \
  grep -v "httpx\|HF_TOKEN\|Loading\|huggingface\|Redirect" | tail -20
```

Expected dry-run output (at 18:30 IST when Reddit is active):
```
INFO  signals_generated count=N bullish=N neutral=N bearish=N
INFO  BULLISH NVDA   pred=+2.1% conf=70%  posts=45
INFO  NEUTRAL TSLA   pred=+1.2% conf=40%  posts=89
INFO  BEARISH AMC    pred=-1.8% conf=60%  posts=22
```

At midnight IST (low Reddit): 0 signals is still expected.

---

## Build Order

```bash
# Step 1: Read prediction distribution (required before setting thresholds)
python3 -c "..."  # the prediction distribution check above

# Step 2: Update portfolio/signal_generator.py (Task 1)
# Step 3: Update portfolio/portfolio_engine.py (Task 2)
# Step 4: Update portfolio/regime_detector.py (Task 3)
# Step 5: Update portfolio/execution_logger.py (Task 1f)
# Step 6: Update scripts/daily_run.py (Task 1g)
# Step 7: Add tests (Task 4)
# Step 8: Run tests
pytest tests/ -v --tb=short

# Step 9: Verify regime
python3 -c "from portfolio.regime_detector import classify_regime; r = classify_regime(); print(r.label, r.reason)"

# Step 10: Dry run
python scripts/daily_run_live.py --dry-run 2>&1 | grep -v "httpx\|HF_TOKEN\|Loading\|huggingface"

# Step 11: Push
bash push.sh "[arch] signal thresholds 1.5% + stop-loss 8% + SPY regime fix"
```

---

## Hard Rules

- NEVER set BULLISH_THRESHOLD below 1% — that's below model noise level
- NEVER set STOP_LOSS_PCT below -15% — defeats the purpose
- NEVER set STOP_LOSS_PCT above -5% — too tight, normal volatility triggers it
- The -8% stop-loss is the right balance for 5-day hold swing trading
- ALWAYS check prediction distribution before setting thresholds
  (if model rarely predicts > 2%, 1.5% is the right BULLISH threshold)
- NEVER remove the NEUTRAL classification — it's needed for low-conviction days
- Existing tests must still pass — 20/20 minimum before pushing
- The signal field addition to SignalRecord is ADDITIVE — do not remove
  existing fields (backward compat with daily_run.py position sizing)

---

## Expected Impact on Live Trading

```
Before these changes:
  - All signals: NEUTRAL (confidence=0)
  - No stop-loss: bad trades held 5 full days
  - Regime: sometimes NEUTRAL when should be POSITIVE

After these changes:
  - Signals properly classified BULLISH/BEARISH/NEUTRAL
  - Stop-loss cuts -8% losers early (saves capital)
  - Regime fires POSITIVE in current bull market (100% sizing)
  - Portfolio engine uses full conviction when signal is clear
```

---

*Architecture Improvements — June 2026*
*Signal thresholds: 1.5% | Stop-loss: -8% | Regime: SPY 1y period*
