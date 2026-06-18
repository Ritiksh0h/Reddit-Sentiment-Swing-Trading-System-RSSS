# Claude Code — Fix System: Exit Price Bug + Density Gate + Dashboard
# Reddit Sentiment Swing Trading System (RSSS)
# GitHub: https://github.com/Ritiksh0h/Reddit-Sentiment-Swing-Trading-System-RSSS

---

## Context (Read This First)

Four problems preventing the system from working end-to-end:

```
PROBLEM 1 — EXIT PRICE BUG (critical)
  8 of 9 closed trades have entry_price == exit_price → PnL = 0.00%
  Root cause: current_prices built from {signal.ticker: signal.price}
  When a ticker's position expires but it's NOT in today's signals,
  the fallback sets price = entry_price (line 150 in daily_run.py):
    current_prices.setdefault(p.ticker, p.entry_price)
  Fix: fetch real exit price from yfinance for expiring positions

PROBLEM 2 — ZERO SIGNALS (critical)
  Reddit fetches 501 posts but only 7 tickers found, none clear
  the density gate (post_count_1d >= 10).
  Root cause: posts spread across 38 tickers = 13 posts/ticker avg,
  but individual ticker counts are low.
  Fix: count posts per ticker correctly and widen to include
  mention_growth momentum from 4h window.

PROBLEM 3 — TICKER UNIVERSE MISMATCH (moderate)
  config/tickers.txt has only 10 Phase 0 tickers.
  live fetcher TRACKED_TICKERS has 38 tickers.
  dashboard has no ticker dropdown (old dashboard still active —
  new Tailwind dashboard was NOT saved to disk correctly).
  Fix: sync tickers.txt, replace dashboard with new design.

PROBLEM 4 — OLD DASHBOARD STILL ACTIVE (moderate)
  dashboard/index.html is 995 lines (old design).
  New Tailwind dashboard (888 lines) was never saved.
  The new dashboard file rsss_quantitative_dashboard.html
  is in the project root — it just needs to be copied.
```

---

## Session Start

```bash
git pull origin main
source .venv/bin/activate

# Verify current state
curl -s http://localhost:8000/status | python3 -m json.tool

# Confirm exit price bug exists
python3 -c "
import json
portfolio = json.load(open('data/paper_portfolio.json'))
trades = portfolio.get('closed_trades', [])
zero = sum(1 for t in trades if t['exit_price'] == t['entry_price'])
print(f'Zero-PnL trades: {zero}/{len(trades)}')
for t in trades:
    match = 'BUG' if t['exit_price'] == t['entry_price'] else 'OK'
    print(f'  {match}  {t[\"ticker\"]:<6}  entry={t[\"entry_price\"]:>8.2f}  exit={t[\"exit_price\"]:>8.2f}  pnl={t[\"pnl_pct\"]:+.4f}')
"

# Confirm old dashboard is active
head -3 dashboard/index.html
wc -l dashboard/index.html
# Expect: 995 lines (old), not 888 (new)

# Confirm new dashboard exists in root
ls rsss_quantitative_dashboard.html
wc -l rsss_quantitative_dashboard.html
# Expect: 888 lines
```

---

## TASK 0 — CRITICAL FIXES (do these first, before everything else)

### 0a — Fix StockTwits DNS failure:
  In data/stocktwits_fetcher.py, wrap the API call in try/except
  On NameResolutionError or any connection error:
    log WARNING "stocktwits_dns_failed — returning zeros"
    return {} empty dict (not a crash)
  System then uses st_sentiment_1d=0.0, st_bull_pct=0.5 defaults
  This unblocks the daily run immediately

### 0b — Fix SPY regime detection:
  In portfolio/regime_detector.py, find the yfinance download call
  Replace:
    mkt = yf.download('SPY', period='300d', ...)
  With:
    for period in ['1y', '6mo', '60d', '30d']:
        mkt = yf.download('SPY', period=period, ...)
        if not mkt.empty: break
    if mkt.empty:
        logger.warning('SPY data unavailable — defaulting to NEUTRAL')
        return NeutralRegime()

### 0c — Fix confidence=0.0:
  In portfolio/signal_generator.py, find where SignalRecord is created
  Replace:
    confidence=0.0
  With:
    confidence=min(abs(predicted_return) / 0.03, 1.0)
  This makes confidence scale with prediction strength
  pred=1% → conf=0.33, pred=3% → conf=1.0

### 0d — Force close expiring positions:
  Run: python scripts/daily_run.py
  Confirm UBER/TSLA/NFLX close with real exit prices

## Task 1 — Fix Exit Price Bug

### Root Cause

In `scripts/daily_run.py` line ~147-150:

```python
current_prices = {s.ticker: s.price for s in signals}
# fallback for tickers not in today's signals
for p in state.positions:
    current_prices.setdefault(p.ticker, p.entry_price)  # ← BUG
```

When a position expires (hold_period_expired) but that ticker is not
in today's signals, `current_prices` uses `entry_price` as the exit
price. This produces 0.00% PnL for every expired position that didn't
generate a new signal today — which is most of them.

### Fix

Read `scripts/daily_run.py` carefully first to understand the full
exit flow, then apply this targeted fix.

Find the section that builds `current_prices` (around line 147) and
replace the fallback with a yfinance fetch:

```python
# ── Build current prices for exit evaluation ───────────────────
current_prices = {s.ticker: s.price for s in signals}

# For positions NOT in today's signals, fetch real exit price
# instead of using entry_price as fallback (the exit price bug)
positions_needing_price = [
    p for p in state.positions
    if p.ticker not in current_prices
]

if positions_needing_price:
    import yfinance as yf
    tickers_to_fetch = [p.ticker for p in positions_needing_price]
    logger.info(f'fetching_exit_prices tickers={tickers_to_fetch}')
    try:
        # Download last 2 days for all missing tickers at once
        # (batch download is faster than one-by-one)
        raw = yf.download(
            tickers_to_fetch,
            period='2d',
            auto_adjust=True,
            progress=False,
            group_by='ticker',
        )
        for p in positions_needing_price:
            try:
                if len(tickers_to_fetch) == 1:
                    # Single ticker: columns are flat
                    price = float(raw['Close'].dropna().iloc[-1])
                else:
                    # Multi-ticker: columns are MultiIndex (ticker, field)
                    price = float(raw['Close'][p.ticker].dropna().iloc[-1])
                current_prices[p.ticker] = price
                logger.info(f'exit_price_fetched ticker={p.ticker} price={price:.4f}')
            except Exception as e:
                logger.warning(f'exit_price_fetch_failed ticker={p.ticker}: {e} '
                               f'— using entry_price as fallback')
                current_prices[p.ticker] = p.entry_price
    except Exception as e:
        logger.warning(f'batch_exit_price_fetch_failed: {e} — using entry_prices')
        for p in positions_needing_price:
            current_prices[p.ticker] = p.entry_price
```

### Verify the fix

```bash
# Check that the fix is in place
grep -n "fetching_exit_prices\|positions_needing_price\|batch_exit_price" \
    scripts/daily_run.py

# Dry run to confirm no import errors
python scripts/daily_run_live.py --dry-run 2>&1 | tail -15
```

Expected: no import errors, no crashes.

### Retroactively fix existing zero-PnL trades

The 8 existing zero-PnL trades in `data/paper_portfolio.json` need
their real exit prices fetched and PnL recalculated.

Create and run this one-time fix script:

```python
# scripts/fix_exit_prices.py
"""
One-time fix: fetch real exit prices for zero-PnL closed trades.
Run once after fixing the exit price bug in daily_run.py.
"""
import json, logging
from pathlib import Path
from datetime import datetime, timezone
import yfinance as yf

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
logger = logging.getLogger(__name__)

portfolio_path = Path('data/paper_portfolio.json')
portfolio = json.loads(portfolio_path.read_text())
trades     = portfolio.get('closed_trades', [])

fixed = 0
for trade in trades:
    if trade['exit_price'] != trade['entry_price']:
        logger.info(f'SKIP {trade["ticker"]} — already has real exit price')
        continue

    ticker     = trade['ticker']
    exit_date  = trade['exit_date']

    # Fetch price around exit date
    try:
        # Download a window around the exit date
        from datetime import timedelta
        start = (datetime.fromisoformat(exit_date) - timedelta(days=2)).date()
        end   = (datetime.fromisoformat(exit_date) + timedelta(days=2)).date()

        mkt = yf.download(ticker, start=str(start), end=str(end),
                          auto_adjust=True, progress=False)
        if isinstance(mkt.columns, __import__('pandas').MultiIndex):
            mkt.columns = mkt.columns.get_level_values(0)

        if mkt.empty:
            logger.warning(f'No market data for {ticker} around {exit_date}')
            continue

        # Use the close on exit_date or nearest available date
        mkt.index = mkt.index.astype(str)
        if exit_date in mkt.index:
            real_exit_price = float(mkt.loc[exit_date, 'Close'])
        else:
            real_exit_price = float(mkt['Close'].iloc[-1])

        real_pnl = (real_exit_price - trade['entry_price']) / trade['entry_price']

        logger.info(
            f'FIXED {ticker:<6} '
            f'entry={trade["entry_price"]:.2f} '
            f'exit_old={trade["exit_price"]:.2f} '
            f'exit_new={real_exit_price:.2f} '
            f'pnl={real_pnl*100:+.2f}%'
        )

        trade['exit_price'] = round(real_exit_price, 4)
        trade['pnl_pct']    = round(real_pnl, 4)
        fixed += 1

    except Exception as e:
        logger.error(f'Failed to fix {ticker}: {e}')

# Save fixed portfolio
portfolio_path.write_text(json.dumps(portfolio, indent=2))
logger.info(f'Fixed {fixed}/{len(trades)} trades. Saved to {portfolio_path}')

# Print summary
print()
print('=== FIXED TRADES ===')
for t in portfolio['closed_trades']:
    status = 'FIXED' if t['pnl_pct'] != 0.0 else 'ZERO'
    print(f'  {status}  {t["ticker"]:<6}  pnl={t["pnl_pct"]*100:+.2f}%')
```

```bash
python scripts/fix_exit_prices.py
```

Expected: all 8 zero-PnL trades get real prices, PnL becomes non-zero.

---

## Task 2 — Fix Zero Signals (Density Gate Not Clearing)

### Root Cause

501 posts across 38 tickers = ~13 posts/ticker average, but individual
tickers vary widely. The density gate requires 10 posts for a single
ticker. Many tickers have 2-5 posts, none reach 10 when the fetcher
runs at 18:30 IST (09:00 ET, only 30 min into trading).

### Fix A — Print per-ticker post counts in daily log

First, add diagnostic logging to see exactly what's happening:

In `scripts/daily_run.py`, find where `reddit_counts` is logged:

```python
logger.info(f'Reddit data ready: {len(reddit_counts)} tickers')
```

Replace with:

```python
logger.info(f'Reddit data ready: {len(reddit_counts)} tickers')
# Log per-ticker counts so we can see who's near the gate
for ticker, data in sorted(reddit_counts.items(),
                            key=lambda x: x[1]['post_count_1d'],
                            reverse=True)[:10]:
    count = data['post_count_1d']
    gate  = 'PASS' if count >= 10 else 'FAIL'
    logger.info(f'  [{gate}] {ticker:<6} posts={count}')
```

### Fix B — Combine Reddit + StockTwits post count for density gate

The density gate currently only counts Reddit posts. StockTwits
fetches 38 tickers daily. A ticker with 7 Reddit posts and 50 StockTwits
messages has strong crowd activity but fails the gate.

In `scripts/daily_run.py`, find where `reddit_counts` is combined
with StockTwits data and update the density gate logic:

```python
# ── Combine all sources for density evaluation ─────────────────
# A ticker qualifies if it has strong activity from ANY source
# reddit_counts already has post_count_1d from Reddit
# Boost effective count if StockTwits also has data for this ticker
for ticker in list(reddit_counts.keys()):
    st_count = stocktwits_data.get(ticker, {}).get('st_count_1d', 0)
    # If ST has 20+ messages, treat as equivalent to 5 Reddit posts
    st_equivalent = min(st_count // 20, 5)
    reddit_counts[ticker]['post_count_1d'] += st_equivalent
    if st_equivalent > 0:
        logger.debug(f'density_boost ticker={ticker} '
                     f'st_count={st_count} boost=+{st_equivalent}')

# Also add tickers from StockTwits that have no Reddit presence
# but high activity
for ticker, st_data in stocktwits_data.items():
    if ticker not in reddit_counts:
        st_count = st_data.get('st_count_1d', 0)
        if st_count >= 50:  # 50 ST messages = meaningful attention
            reddit_counts[ticker] = {
                'post_count_1d':      st_count // 20,  # equivalent posts
                'mention_growth_1d':  0.0,
                'mention_growth_7d':  0.0,
                'news_sentiment_1d':  st_data.get('st_sentiment_1d', 0.0),
                'st_sentiment_1d':    st_data.get('st_sentiment_1d', 0.0),
                'st_bull_pct':        st_data.get('st_bull_pct', 0.5),
                'news_count_1d':      0,
                'st_count_1d':        st_count,
            }
            logger.info(f'stocktwits_only_ticker ticker={ticker} '
                        f'st_count={st_count} effective_posts={st_count//20}')
```

### Verify

```bash
python scripts/daily_run_live.py --dry-run 2>&1 | grep -E "PASS|FAIL|signals_generated|density"
```

Expected: some tickers now show [PASS] and signals are generated.

---

## Task 3 — Sync config/tickers.txt With Live Fetcher

`config/tickers.txt` has only 10 Phase 0 tickers.
`data/reddit_live_fetcher.py` tracks 38 tickers.
These must match.

### Replace config/tickers.txt

```bash
cat > config/tickers.txt << 'EOF'
# config/tickers.txt
# Tracked ticker universe for RSSS Phase 3/4
# Must match TRACKED_TICKERS in data/reddit_live_fetcher.py
# Format: one ticker per line, uppercase, no whitespace

NVDA
TSLA
AMD
AAPL
GME
AMC
PLTR
MARA
COIN
META
MSFT
AMZN
GOOG
NFLX
SOFI
HOOD
ROKU
SNAP
UBER
NIO
BABA
SHOP
PYPL
DKNG
DIS
RKLB
HIMS
RDDT
SOUN
IONQ
F
BA
BB
GS
JPM
BAC
SQ
NOK
SPCE
EOF
```

Verify the count matches the live fetcher:

```bash
wc -l config/tickers.txt
python3 -c "
from data.reddit_live_fetcher import TRACKED_TICKERS
print(f'Live fetcher: {len(TRACKED_TICKERS)} tickers')
# Load tickers.txt
tickers = [l.strip() for l in open('config/tickers.txt')
           if l.strip() and not l.startswith('#')]
print(f'tickers.txt:  {len(tickers)} tickers')
missing = set(TRACKED_TICKERS) - set(tickers)
extra   = set(tickers) - set(TRACKED_TICKERS)
if missing: print(f'Missing from tickers.txt: {missing}')
if extra:   print(f'Extra in tickers.txt: {extra}')
if not missing and not extra: print('Both lists match ✓')
"
```

---

## Task 4 — Replace Dashboard With New Tailwind Design

### 4a — Verify new dashboard file exists

```bash
ls rsss_quantitative_dashboard.html
wc -l rsss_quantitative_dashboard.html
# Must be 888 lines
```

If not found:

```bash
find . -name "rsss_quantitative_dashboard*" 2>/dev/null
find ~/Downloads -name "rsss*dashboard*" 2>/dev/null
```

### 4b — Replace old dashboard

```bash
# Backup old dashboard
cp dashboard/index.html dashboard/index_backup_$(date +%Y%m%d).html

# Replace with new design
cp rsss_quantitative_dashboard.html dashboard/index.html

# Verify
wc -l dashboard/index.html
# Must be 888 lines

grep -c "tailwind\|cardBg\|ticker-select" dashboard/index.html
# Must be > 0
```

### 4c — Update ticker dropdown in new dashboard

The new dashboard has only 6 tickers hardcoded in the
`simulationData` object and `<select id="ticker-select">`.

Expand the dropdown to all 38 tracked tickers.

Find this in `dashboard/index.html`:

```html
<select id="ticker-select" onchange="onTickerChange()" ...>
    <option value="NVDA">NVDA — NVIDIA Corp.</option>
    <option value="TSLA">TSLA — Tesla, Inc.</option>
    <option value="AAPL">AAPL — Apple Inc.</option>
    <option value="AMD">AMD — Advanced Micro Devices</option>
    <option value="MSFT">MSFT — Microsoft Corp.</option>
    <option value="SPY">SPY — S&P 500 ETF</option>
</select>
```

Replace with:

```html
<select id="ticker-select" onchange="onTickerChange()" class="bg-panelBg border border-cardBorder text-white px-3 py-2 rounded-lg font-mono text-sm w-full focus:outline-none focus:border-accentBlue">
    <option value="NVDA">NVDA — NVIDIA Corp.</option>
    <option value="TSLA">TSLA — Tesla Inc.</option>
    <option value="AMD">AMD — Advanced Micro Devices</option>
    <option value="AAPL">AAPL — Apple Inc.</option>
    <option value="MSFT">MSFT — Microsoft Corp.</option>
    <option value="META">META — Meta Platforms</option>
    <option value="AMZN">AMZN — Amazon</option>
    <option value="GOOG">GOOG — Alphabet</option>
    <option value="NFLX">NFLX — Netflix</option>
    <option value="COIN">COIN — Coinbase</option>
    <option value="PLTR">PLTR — Palantir</option>
    <option value="GME">GME — GameStop</option>
    <option value="AMC">AMC — AMC Entertainment</option>
    <option value="MARA">MARA — Marathon Digital</option>
    <option value="SOFI">SOFI — SoFi Technologies</option>
    <option value="HOOD">HOOD — Robinhood</option>
    <option value="ROKU">ROKU — Roku</option>
    <option value="SNAP">SNAP — Snap Inc.</option>
    <option value="UBER">UBER — Uber</option>
    <option value="NIO">NIO — NIO Inc.</option>
    <option value="BABA">BABA — Alibaba</option>
    <option value="SHOP">SHOP — Shopify</option>
    <option value="PYPL">PYPL — PayPal</option>
    <option value="DKNG">DKNG — DraftKings</option>
    <option value="DIS">DIS — Walt Disney</option>
    <option value="RKLB">RKLB — Rocket Lab</option>
    <option value="HIMS">HIMS — Hims &amp; Hers</option>
    <option value="RDDT">RDDT — Reddit Inc.</option>
    <option value="SOUN">SOUN — SoundHound AI</option>
    <option value="IONQ">IONQ — IonQ</option>
    <option value="F">F — Ford Motor</option>
    <option value="BA">BA — Boeing</option>
    <option value="BB">BB — BlackBerry</option>
    <option value="GS">GS — Goldman Sachs</option>
    <option value="JPM">JPM — JPMorgan</option>
    <option value="BAC">BAC — Bank of America</option>
    <option value="SQ">SQ — Block Inc.</option>
    <option value="NOK">NOK — Nokia</option>
    <option value="SPCE">SPCE — Virgin Galactic</option>
</select>
```

Also add default simulation data for the new tickers in the
`simulationData` object in the JavaScript. Find the closing `};`
of the simulationData object and add defaults for all tickers
not already present:

```javascript
// Add default presets for all tickers not already in simulationData
const DEFAULT_SIM = {
    reddit_vol: 15, reddit_sent: 20, news_sent: 10, st_sent: 5,
    base_accuracy_1d: "51.0%", base_accuracy_3d: "52.0%", base_accuracy_5d: "53.5%"
};
// Called at start of onTickerChange() if preset not found:
// const preset = simulationData[selectedTicker] || DEFAULT_SIM;
```

Update `onTickerChange()` to use the default:

```javascript
function onTickerChange() {
    selectedTicker = document.getElementById('ticker-select').value;
    if (currentMode === 'sim') {
        const preset = simulationData[selectedTicker] || {
            reddit_vol: 15, reddit_sent: 20, news_sent: 10, st_sent: 5,
            base_accuracy_1d: "51.0%", base_accuracy_3d: "52.0%",
            base_accuracy_5d: "53.5%"
        };
        document.getElementById('sim-reddit-vol').value   = preset.reddit_vol;
        document.getElementById('sim-reddit-sent').value  = preset.reddit_sent;
        document.getElementById('sim-news-sent').value    = preset.news_sent;
        document.getElementById('sim-st-sent').value      = preset.st_sent;
        updateSimulation();
    } else {
        fetchLiveAPIData();
    }
}
```

### 4d — Verify dashboard loads correctly

```bash
open http://localhost:8000/dashboard
```

Check:
- Simulator mode visible by default with sliders
- Ticker dropdown shows all 38 tickers
- Moving Reddit volume slider below 10 → "DENSITY GATE NOT MET"
- Toggle to "Live RSSS API" → portfolio equity loads

---

## Task 5 — Run Tests and Full Dry Run

```bash
# Run all tests
pytest tests/ -v --tb=short

# Full dry run with new fixes active
python scripts/daily_run_live.py --dry-run 2>&1 | tee /tmp/dryrun_check.log

# Check dry run results
grep -E "PASS|FAIL|signals_generated|density|exit_price|posts=" /tmp/dryrun_check.log
```

Expected:
```
INFO  [PASS] NVDA  posts=XX
INFO  [PASS] TSLA  posts=XX
INFO  signals_generated count=N bullish=N bearish=N
```

If still zero signals after the StockTwits density boost — the issue
is that the run happens before enough posts accumulate. This is not
a bug, just timing. The 21:00 IST run should produce more signals.

---

## Task 6 — Push

```bash
bash push.sh "[fix] exit price bug + density gate boost + ticker sync + dashboard update"
```

---

## Hard Rules for This Session

- NEVER lower the density gate threshold below 5
  (lower than 5 Reddit posts = genuine noise, not a signal)
- NEVER delete closed_trades from paper_portfolio.json
  (fix_exit_prices.py patches them in place)
- ALWAYS batch yfinance download for exit prices
  (one download call for all tickers, not one per ticker)
- NEVER change the simulator JavaScript logic in the new dashboard
  (updateSimulation, toggleMode, onTickerChange core logic)
- ALWAYS run pytest before pushing
- The fix_exit_prices.py script runs ONCE — do not run it again
  after real trades start accumulating

---

## Expected State After This Session

```
Exit prices:    8 zero-PnL trades corrected with real prices
Signals:        Some tickers now clear density gate (ST boost)
                Per-ticker post counts visible in logs
Dashboard:      New Tailwind design, 38-ticker dropdown
tickers.txt:    Synced with live fetcher (38 tickers)
Tests:          20/20 passing

Tomorrow 18:30 IST:
  Automated run fires
  Per-ticker counts logged
  Signals generated if any ticker reaches 10 effective posts
  Exit prices fetched correctly for expiring positions
```

---

*System Fix Sprint — June 2026*
*Fixes: exit price bug, density gate starvation, ticker sync, dashboard*
*Critical: exit price fix retroactively corrects all 8 zero-PnL trades*
