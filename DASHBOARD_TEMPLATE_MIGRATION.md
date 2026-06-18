# Claude Code — Replace Dashboard with New Tailwind Template
# Reddit Sentiment Swing Trading System (RSSS)
# GitHub: https://github.com/Ritikshah0h/Reddit-Sentiment-Swing-Trading-System-RSSS

---

## What This Session Does

Replace `dashboard/index.html` with a new Tailwind CSS dashboard template.
The new template has three modes: Simulator (interactive sliders), Live API
(fetches from localhost:8000), and graceful fallback when API is offline.

The template has been uploaded to the project. Your job is to wire the
Live API mode to the actual RSSS endpoints and fix the response format
mismatches between what the template expects and what the API returns.

---

## Session Start

```bash
git pull origin main
source .venv/bin/activate

# Confirm API is running
curl -s http://localhost:8000/status | python3 -m json.tool

# Confirm new template exists
ls dashboard/rsss_quantitative_dashboard.html
```

If the template file is not at `dashboard/rsss_quantitative_dashboard.html`,
it may have been uploaded to a different location. Find it:

```bash
find . -name "rsss_quantitative_dashboard.html" 2>/dev/null
```

---

## Task 1 — Replace dashboard/index.html

### Step 1a — Backup the current dashboard

```bash
cp dashboard/index.html dashboard/index_backup_$(date +%Y%m%d).html
```

### Step 1b — Copy template to index.html

```bash
cp dashboard/rsss_quantitative_dashboard.html dashboard/index.html
```

### Step 1c — Verify it loads

```bash
open http://localhost:8000/dashboard
```

The dashboard should open with Simulator Mode active by default showing
mock data and interactive sliders. If it shows a blank page or 404,
check that the API serves static files from the dashboard/ directory.

---

## Task 2 — Fix Live API Response Mismatches

The template's `fetchLiveAPIData()` function expects specific response
shapes from the API. The current RSSS API returns different shapes.
Fix the API endpoints to match what the template expects.

### 2a — /portfolio endpoint

Template expects:
```json
{
  "equity": 10000.00,
  "total_return_pct": 4.25,
  "positions_count": 2,
  "regime_label": "NEUTRAL",
  "sizing_pct": 75
}
```

Read the current `/portfolio` endpoint in `api/main.py` and update its
response to include all five fields. Map existing fields:
- `equity` = total portfolio value (cash + positions market value)
- `total_return_pct` = (equity - 10000) / 10000 * 100  (starting capital = $10,000)
- `positions_count` = number of open positions
- `regime_label` = current regime string ('POSITIVE', 'NEUTRAL', 'NEGATIVE')
- `sizing_pct` = regime sizing multiplier × 100 (POSITIVE=100, NEUTRAL=75, NEGATIVE=50)

### 2b — /predictions endpoint

Template calls: `GET /predictions?ticker=NVDA`

Template expects:
```json
{
  "density_passed": true,
  "1D": {"pred": 0.45, "conf": 58},
  "3D": {"pred": 1.25, "conf": 62},
  "5D": {"pred": 2.84, "conf": 79.6}
}
```

Current `/predictions` endpoint returns a list of all signals grouped by
bullish/bearish/neutral. Update it to also accept an optional `ticker`
query parameter. When `ticker` is provided:

```python
@app.get('/predictions')
def get_predictions(ticker: str = None):
    # ... existing code to load signals ...

    if ticker:
        # Find the most recent signal for this ticker
        all_signals = bullish + bearish + neutral
        ticker_signal = next(
            (s for s in all_signals if s['ticker'] == ticker), None
        )
        if ticker_signal:
            return {
                'density_passed': True,
                '1D': {
                    'pred': round((ticker_signal.get('predicted_1d', 0) or 0) * 100, 2),
                    'conf': round((ticker_signal.get('confidence', 0) or 0) * 100, 1),
                },
                '3D': {
                    'pred': round((ticker_signal.get('predicted_3d', 0) or 0) * 100, 2),
                    'conf': round((ticker_signal.get('confidence', 0) or 0) * 100, 1),
                },
                '5D': {
                    'pred': round((ticker_signal.get('predicted_5d', 0) or 0) * 100, 2),
                    'conf': round((ticker_signal.get('confidence', 0) or 0) * 100, 1),
                },
            }
        else:
            # Ticker not in today's signals — density gate not met or not tracked
            return {
                'density_passed': False,
                '1D': {'pred': 0, 'conf': 50},
                '3D': {'pred': 0, 'conf': 50},
                '5D': {'pred': 0, 'conf': 50},
            }

    # No ticker specified — return full list (existing behaviour)
    return {'date': today, 'bullish': bullish, 'bearish': bearish,
            'neutral': neutral, 'total': len(formatted)}
```

### 2c — /shap/{ticker} endpoint

Template expects these specific keys in the response:
```json
{
  "reddit_attention": 42,
  "reddit_sentiment": 12,
  "news_sentiment": 26,
  "st_sentiment": 8,
  "market_technical": 12
}
```

The current `/shap/{ticker}` endpoint returns `family_shap` with keys
`reddit`, `news`, `stocktwits`, `market`. Update the endpoint to ALSO
include the flat keys the template expects. Add to the return dict:

```python
# Add alongside existing return fields:
fam = family_shap  # existing dict with reddit/news/stocktwits/market

# Convert raw SHAP values to percentage weights for the template
total_abs = sum(abs(v) for v in fam.values()) or 1.0
shap_pcts = {k: round(abs(v) / total_abs * 100) for k, v in fam.items()}

return {
    # ... existing fields ...
    # Flat keys for the new template
    'reddit_attention':  shap_pcts.get('reddit', 0),
    'reddit_sentiment':  0,   # not a separate family in current model
    'news_sentiment':    shap_pcts.get('news', 0),
    'st_sentiment':      shap_pcts.get('stocktwits', 0),
    'market_technical':  shap_pcts.get('market', 0),
}
```

### 2d — /signal-accuracy endpoint

Template expects:
```json
{
  "1D": "51.2%",
  "3D": "52.4%",
  "5D": "56.1%"
}
```

The current endpoint returns `accuracy_1d`, `accuracy_3d`, `accuracy_5d`
as floats (0.512, 0.524, 0.561). Add shorthand keys to the response:

```python
# Add to the return dict in get_signal_accuracy():
'1D': f"{round((accuracy_1d or 0) * 100, 1)}%" if accuracy_1d else "—",
'3D': f"{round((accuracy_3d or 0) * 100, 1)}%" if accuracy_3d else "—",
'5D': f"{round((accuracy_5d or 0) * 100, 1)}%" if accuracy_5d else "—",
```

---

## Task 3 — Update Ticker List in Dashboard

The template has a hardcoded ticker selector with 6 tickers:
NVDA, TSLA, AAPL, AMD, MSFT, SPY

Update it to show all 29 tracked tickers from the RSSS universe.

In `dashboard/index.html`, find the `<select id="ticker-select">` block
and replace the hardcoded options with all tracked tickers:

```html
<select id="ticker-select" onchange="onTickerChange()"
        class="bg-panelBg border border-cardBorder text-white px-3 py-2
               rounded-lg font-mono text-sm w-full focus:outline-none
               focus:border-accentBlue">
    <option value="NVDA">NVDA — NVIDIA Corp.</option>
    <option value="TSLA">TSLA — Tesla, Inc.</option>
    <option value="AMD">AMD — Advanced Micro Devices</option>
    <option value="AAPL">AAPL — Apple Inc.</option>
    <option value="PLTR">PLTR — Palantir Technologies</option>
    <option value="COIN">COIN — Coinbase Global</option>
    <option value="MARA">MARA — Marathon Digital</option>
    <option value="META">META — Meta Platforms</option>
    <option value="MSFT">MSFT — Microsoft Corp.</option>
    <option value="AMZN">AMZN — Amazon.com</option>
    <option value="GOOG">GOOG — Alphabet Inc.</option>
    <option value="NFLX">NFLX — Netflix Inc.</option>
    <option value="SOFI">SOFI — SoFi Technologies</option>
    <option value="HOOD">HOOD — Robinhood Markets</option>
    <option value="ROKU">ROKU — Roku Inc.</option>
    <option value="SNAP">SNAP — Snap Inc.</option>
    <option value="UBER">UBER — Uber Technologies</option>
    <option value="NIO">NIO — NIO Inc.</option>
    <option value="BABA">BABA — Alibaba Group</option>
    <option value="SHOP">SHOP — Shopify Inc.</option>
    <option value="PYPL">PYPL — PayPal Holdings</option>
    <option value="DKNG">DKNG — DraftKings Inc.</option>
    <option value="DIS">DIS — Walt Disney Co.</option>
    <option value="RKLB">RKLB — Rocket Lab USA</option>
    <option value="HIMS">HIMS — Hims &amp; Hers Health</option>
    <option value="SOUN">SOUN — SoundHound AI</option>
    <option value="IONQ">IONQ — IonQ Inc.</option>
    <option value="SQ">SQ — Block Inc.</option>
    <option value="SPY">SPY — S&amp;P 500 ETF</option>
</select>
```

Also update `simulationData` in the JavaScript to include defaults for
the tickers not currently in the mock data. Add after the existing entries:

```javascript
// Add remaining tickers with neutral defaults
const defaultSimData = {
    reddit_vol: 15, reddit_sent: 10, news_sent: 5, st_sent: 0,
    base_accuracy_1d: "51.0%", base_accuracy_3d: "52.0%", base_accuracy_5d: "53.5%"
};
const extraTickers = ['PLTR','COIN','MARA','META','AMZN','GOOG','NFLX',
    'SOFI','HOOD','ROKU','SNAP','UBER','NIO','BABA','SHOP','PYPL','DKNG',
    'DIS','RKLB','HIMS','SOUN','IONQ','SQ','RIOT'];
extraTickers.forEach(t => { simulationData[t] = {...defaultSimData}; });
```

---

## Task 4 — Update Research Validation Terminal with Real Data

The right column shows hardcoded IC values and Granger results. Update
it to load real values from `/research-findings` in Live mode.

Add to `fetchLiveAPIData()` after the accuracy fetch:

```javascript
// Load research findings
try {
    const researchRes = await fetch(`${baseUrl}/research-findings`);
    const researchData = await researchRes.json();

    if (researchData.conclusion) {
        const best = researchData.conclusion.best_combination || '—';
        const bestIC = researchData.conclusion.best_mean_ic || 0;
        const currentIC = researchData.conclusion.current_model_ic || 0.0796;

        // Update the model hierarchy section (Layer 3)
        const hierEl = document.querySelector('[data-section="model-hierarchy"]');
        if (hierEl && researchData.layer3_walkforward) {
            // Update walk-forward IC values from real results.json
            const combos = researchData.layer3_walkforward;
            const marketIC = combos['Market only']?.mean_ic || 0.008;
            const redditIC = combos['Market + Reddit']?.mean_ic || 0.071;
            const allIC    = combos['All sources']?.mean_ic || currentIC;
            // Update visible elements if they have matching IDs
        }
    }
} catch(e) {
    console.log('Research findings not available:', e);
}
```

NOTE: The research terminal content is mostly static HTML showing real
validated results. Do NOT replace the hardcoded IC values in the terminal
with placeholders — leave the static values if the API fetch fails.
The template handles this gracefully with try/catch.

---

## Task 5 — Verify All Panels in Live Mode

After wiring all endpoints, switch the dashboard to Live API mode and
verify each panel populates:

```bash
# Restart API to pick up changes
pkill -f uvicorn 2>/dev/null; sleep 1
uvicorn api.main:app --host 127.0.0.1 --port 8000 &
sleep 3

# Test each endpoint the dashboard uses
curl -s "http://localhost:8000/status" | python3 -m json.tool
curl -s "http://localhost:8000/portfolio" | python3 -m json.tool
curl -s "http://localhost:8000/predictions?ticker=NVDA" | python3 -m json.tool
curl -s "http://localhost:8000/shap/NVDA" | python3 -m json.tool
curl -s "http://localhost:8000/signal-accuracy" | python3 -m json.tool
curl -s "http://localhost:8000/research-findings" | python3 -m json.tool
```

Expected for each:
- `/status`:           has date, ran_today, system_ok
- `/portfolio`:        has equity, total_return_pct, positions_count,
                       regime_label, sizing_pct
- `/predictions?ticker=NVDA`: has density_passed, 1D, 3D, 5D dicts
- `/shap/NVDA`:        has reddit_attention, news_sentiment, st_sentiment,
                       market_technical (flat keys)
- `/signal-accuracy`:  has 1D, 3D, 5D string keys ("51.2%")
- `/research-findings`: has conclusion, layer1, layer2, layer3 sections

Open dashboard and click "Live RSSS API" button. All panels should
populate. If a panel shows stale mock data, check the browser console
for the specific fetch error.

---

## Task 6 — Run Tests + Push

```bash
pytest tests/ -v --tb=short
```

All 20 tests must pass. If any API tests fail due to response shape
changes, update the test assertions to match the new response format.

```bash
bash push.sh "[dashboard] replace with Tailwind template + wire live API endpoints"
```

---

## Hard Rules for This Session

- NEVER remove the simulator mode — it's a useful demo/testing tool
- NEVER break the graceful fallback (the catch block that shows
  mock data when API is offline)
- ALWAYS keep the existing /predictions endpoint returning the full
  bullish/bearish/neutral list when no ticker param is provided
  (backward compat with any other callers)
- NEVER hardcode portfolio values — always derive from paper_portfolio.json
- The new response fields are ADDITIVE — do not remove existing fields
  from any endpoint, only add new ones
- If /shap/{ticker} returns error (no signal for that ticker yet),
  the dashboard must handle it gracefully — show zeros, not a crash

---

## What the Dashboard Shows

```
LEFT COLUMN (4/12):
  Portfolio Balance card  ← /portfolio
  Active Positions list   ← /portfolio
  Drift Monitor           ← /status (manual check from logs)

MIDDLE COLUMN (5/12):
  Predictive Signals Engine ← /predictions?ticker=X + sliders (sim mode)
  SHAP Attribution bars     ← /shap/{ticker}
  Accountability Tracker    ← /signal-accuracy

RIGHT COLUMN (3/12):
  Signal Validation Terminal ← /research-findings (static fallback ok)
```

---

*Dashboard Template Migration — June 2026*
*Tailwind CSS + Lucide icons + Simulator + Live API mode*
*Template: dashboard/rsss_quantitative_dashboard.html*
