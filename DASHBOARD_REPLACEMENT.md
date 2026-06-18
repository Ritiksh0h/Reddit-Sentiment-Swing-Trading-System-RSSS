# Claude Code — Replace Dashboard with New Design
# Reddit Sentiment Swing Trading System (RSSS)
# GitHub: https://github.com/Ritiksh0h/Reddit-Sentiment-Swing-Trading-System-RSSS

---

## What This Session Does

Replace `dashboard/index.html` with the new Tailwind-based quantitative
dashboard design. The new design is at:
  `/mnt/user-data/uploads/rsss_quantitative_dashboard.html`
  (already in the repo root or accessible to Claude Code)

The new dashboard has:
- Simulator mode (interactive sliders, no API needed for demo)
- Live RSSS API mode (fetches from localhost:8000)
- Three-column layout: Portfolio | Signals+SHAP | Research Terminal
- Accountability tracker (1D/3D/5D accuracy)
- Signal validation terminal (IC table with regime labels)
- Drift monitor panel
- Toggle between simulator and live mode

The existing `dashboard/index.html` is replaced entirely.
Do NOT attempt to merge — full replacement only.

---

## Session Start

```bash
git pull origin main
source .venv/bin/activate

# Confirm API is running
curl -s http://localhost:8000/status | python3 -m json.tool

# Check current dashboard
wc -l dashboard/index.html

# Check new design file exists
ls rsss_quantitative_dashboard.html 2>/dev/null || \
  echo "Check file location — may be in repo root or downloads"
```

---

## Task 1 — Locate the New Dashboard File

The new design file is `rsss_quantitative_dashboard.html`.

Check where it is:

```bash
find . -name "rsss_quantitative_dashboard.html" 2>/dev/null
find ~/Downloads -name "rsss_quantitative_dashboard.html" 2>/dev/null
```

If not found in the project, it was uploaded. Use its content directly.
Do NOT download from any URL. The file content is what was provided.

---

## Task 2 — Understand the API Contract

The new dashboard's `fetchLiveAPIData()` function calls these endpoints:

```javascript
GET /status           → { date, ran_today, system_ok, ... }
GET /portfolio        → { equity, total_return_pct, positions_count,
                          regime_label, sizing_pct }
GET /predictions?ticker=NVDA → {
                          '1D': { pred: float, conf: int },
                          '3D': { pred: float, conf: int },
                          '5D': { pred: float, conf: int },
                          density_passed: bool }
GET /shap/{ticker}    → { reddit_attention: int, reddit_sentiment: int,
                          news_sentiment: int, st_sentiment: int,
                          market_technical: int }
GET /signal-accuracy  → { '1D': "54.2%", '3D': "51.8%", '5D': "56.1%" }
```

**IMPORTANT: Current API responses do NOT match this contract exactly.**
The existing endpoints return different field names and structures.
Task 3 fixes the API to match. Task 4 does the dashboard replacement.
Do Task 3 BEFORE Task 4.

---

## Task 3 — Update api/main.py to Match New Dashboard Contract

Read the current `api/main.py` first to understand what exists:

```bash
cat api/main.py
```

Then make these targeted additions/modifications:

### 3a — Fix /portfolio endpoint

The dashboard expects:
```json
{
  "equity": 10250.00,
  "total_return_pct": 2.50,
  "positions_count": 2,
  "regime_label": "NEUTRAL",
  "sizing_pct": 75
}
```

Find the existing `/portfolio` endpoint. Add these fields to its response
if they are missing:

```python
# In the /portfolio endpoint, ensure these fields are returned:
# - equity (float): total portfolio value = cash + position values
# - total_return_pct (float): (equity - initial_capital) / initial_capital * 100
# - positions_count (int): number of open positions
# - regime_label (str): 'POSITIVE' | 'NEUTRAL' | 'NEGATIVE'
# - sizing_pct (int): 100 | 75 | 50

INITIAL_CAPITAL = 10000.0  # starting paper portfolio value

# Read from data/paper_portfolio.json
portfolio = json.loads(Path('data/paper_portfolio.json').read_text())
cash = float(portfolio.get('cash', INITIAL_CAPITAL))
positions = portfolio.get('positions', [])

# Compute equity (cash + mark-to-market of positions)
# For simplicity, use entry value as proxy if live price unavailable
position_value = sum(
    float(p.get('size', 0)) for p in positions
)
equity = cash + position_value
total_return_pct = round((equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2)

# Get regime from regime_detector
try:
    from portfolio.regime_detector import RegimeDetector
    detector = RegimeDetector()
    regime   = detector.get_current_regime()
    regime_label = regime.upper() if regime else 'NEUTRAL'
except Exception:
    regime_label = 'NEUTRAL'

REGIME_SIZING = {'POSITIVE': 100, 'NEUTRAL': 75, 'NEGATIVE': 50}
sizing_pct    = REGIME_SIZING.get(regime_label, 75)

return {
    'equity':           round(equity, 2),
    'cash':             round(cash, 2),
    'total_return_pct': total_return_pct,
    'positions_count':  len(positions),
    'positions':        positions,
    'regime_label':     regime_label,
    'sizing_pct':       sizing_pct,
    # keep all existing fields too
}
```

### 3b — Fix /predictions endpoint to support ?ticker= query param

The dashboard calls `/predictions?ticker=NVDA` for a specific ticker.
Current endpoint returns all predictions.

Update the endpoint signature to accept an optional ticker:

```python
@app.get('/predictions')
def get_predictions(ticker: str = None):
    """
    Return today's predictions.
    If ticker is specified, return predictions for that ticker only
    in the format expected by the new dashboard.
    """
    # ... existing load logic ...

    if ticker:
        # Find signal for this ticker
        ticker = ticker.upper()
        all_signals = bullish + bearish + neutral
        match = next(
            (s for s in all_signals if s.get('ticker') == ticker),
            None
        )
        if match:
            pred_5d = match.get('predicted_5d', 0) * 100
            pred_3d = match.get('predicted_3d', 0) * 100
            pred_1d = match.get('predicted_1d', 0) * 100
            conf_5d = int((match.get('confidence', 0)) * 100)
            post_count = match.get('post_count_1d', 0)

            return {
                '1D': {
                    'pred': round(pred_1d, 2),
                    'conf': max(int(conf_5d * 0.85), 40)
                },
                '3D': {
                    'pred': round(pred_3d, 2),
                    'conf': max(int(conf_5d * 0.92), 45)
                },
                '5D': {
                    'pred': round(pred_5d, 2),
                    'conf': conf_5d
                },
                'density_passed': post_count >= 10,
                'post_count_1d':  post_count,
                'signal':         match.get('signal', 'NEUTRAL'),
                'ticker':         ticker,
            }
        else:
            # Ticker not in today's signals — return noise state
            return {
                '1D': {'pred': 0.0, 'conf': 48},
                '3D': {'pred': 0.0, 'conf': 49},
                '5D': {'pred': 0.0, 'conf': 50},
                'density_passed': False,
                'post_count_1d':  0,
                'signal':         'NEUTRAL',
                'ticker':         ticker,
                'message':        'Ticker not in today\'s signals or density gate not met',
            }

    # If no ticker specified, return existing full response format
    return {
        'date':    today,
        'bullish': bullish,
        'bearish': bearish,
        'neutral': neutral,
        'total':   len(formatted),
    }
```

### 3c — Fix /shap/{ticker} endpoint response format

Dashboard expects percentage integers, not raw SHAP floats:
```json
{
  "reddit_attention": 42,
  "reddit_sentiment": 12,
  "news_sentiment": 26,
  "st_sentiment": 8,
  "market_technical": 12
}
```

Find the existing `/shap/{ticker}` endpoint. After computing `family_shap`,
add this transformation before returning:

```python
# Convert SHAP values to percentage attribution
fam = family_shap  # {'reddit': float, 'news': float, 'stocktwits': float, 'market': float}

# Split reddit into attention vs sentiment components
# (post_count_1d, mention_growth = attention; if avg_sentiment = sentiment)
# Since we don't have avg_sentiment in the current 14-feature model,
# treat all reddit SHAP as attention
total_abs = sum(abs(v) for v in fam.values()) or 1.0

reddit_pct    = max(int(abs(fam.get('reddit', 0)) / total_abs * 100), 0)
news_pct      = max(int(abs(fam.get('news', 0)) / total_abs * 100), 0)
st_pct        = max(int(abs(fam.get('stocktwits', 0)) / total_abs * 100), 0)
market_pct    = max(int(abs(fam.get('market', 0)) / total_abs * 100), 0)

# Ensure percentages sum to 100
total_pct = reddit_pct + news_pct + st_pct + market_pct
if total_pct > 0 and total_pct != 100:
    market_pct += (100 - total_pct)

return _sanitize({
    # New dashboard format
    'reddit_attention':  reddit_pct,
    'reddit_sentiment':  0,           # avg_sentiment not in current model
    'news_sentiment':    news_pct,
    'st_sentiment':      st_pct,
    'market_technical':  market_pct,
    # Verbose data for debugging
    'ticker':            ticker,
    'date':              latest.get('date'),
    'base_value':        round(base_val, 4),
    'prediction':        round(prediction, 4),
    'signal':            latest.get('signal', 'NEUTRAL'),
    'family_shap_raw':   family_shap,
    'attribution_text':  attribution_text,
    'top_features':      contributions[:8],
})
```

### 3d — Fix /signal-accuracy endpoint response format

Dashboard expects:
```json
{ "1D": "54.2%", "3D": "51.8%", "5D": "56.1%" }
```

At the end of the existing `/signal-accuracy` endpoint, add:

```python
# Format for new dashboard (in addition to existing detailed response)
if not results:
    return {
        'n_evaluated': 0,
        'message': 'Need 30+ signals with 7+ days elapsed',
        '1D': '—', '3D': '—', '5D': '—'
    }

return _sanitize({
    'n_evaluated':    n,
    'accuracy_1d':    acc('correct_1d'),
    'accuracy_3d':    acc('correct_3d'),
    'accuracy_5d':    acc('correct_5d'),
    # New dashboard format (string percentages)
    '1D': f"{acc('correct_1d')*100:.1f}%" if acc('correct_1d') else '—',
    '3D': f"{acc('correct_3d')*100:.1f}%" if acc('correct_3d') else '—',
    '5D': f"{acc('correct_5d')*100:.1f}%" if acc('correct_5d') else '—',
    'mean_pred_5d':   round(sum(r['pred_5d']   for r in results) / n, 4),
    'mean_actual_5d': round(sum((r['actual_5d'] or 0) for r in results) / n, 4),
    'interpretation': (
        '1D accuracy low, 5D high → signal has multi-day lag (normal for momentum)'
        if (acc('correct_1d') or 0.5) < 0.5 and (acc('correct_5d') or 0) > 0.53
        else 'Monitoring — need more completed signals'
    ),
    'signals': results,
})
```

### 3e — Test all four endpoints

```bash
# Restart API with updated code
# (if running with --reload, changes apply automatically)

curl -s http://localhost:8000/portfolio | python3 -m json.tool
curl -s "http://localhost:8000/predictions?ticker=NVDA" | python3 -m json.tool
curl -s "http://localhost:8000/shap/NVDA" | python3 -m json.tool
curl -s http://localhost:8000/signal-accuracy | python3 -m json.tool
```

For `/shap/NVDA` and `/predictions?ticker=NVDA`:
If no signals have been logged today, the endpoints will return the
no-signal response. That is correct — simulator mode handles this.

---

## Task 4 — Replace dashboard/index.html

### 4a — Backup existing dashboard

```bash
cp dashboard/index.html dashboard/index_old.html
```

### 4b — Copy new dashboard

```bash
cp rsss_quantitative_dashboard.html dashboard/index.html
```

If the file is not in repo root, it was provided as an upload.
Write its full content directly to `dashboard/index.html`.
The file is 888 lines. Write it exactly as provided — do not modify.

### 4c — Verify it serves correctly

```bash
# Check file was written
wc -l dashboard/index.html

# Open in browser
open http://localhost:8000/dashboard
```

The dashboard should load immediately showing Simulator Mode by default.
Toggle to "Live RSSS API" to fetch real data from the running API.

---

## Task 5 — Verify End-to-End

### In Simulator Mode:
- Slider panel visible
- Changing Reddit volume slider below 10 should show "DENSITY GATE NOT MET"
- SHAP bars update in real time as sliders move
- Ticker dropdown changes presets

### In Live Mode:
- Health indicator shows "LIVE ACTIVE" (green pulse)
- Portfolio equity loads from /portfolio
- If signals exist: predictions update for selected ticker
- If no signals today: dashboard shows "density not met" state gracefully
- SHAP panel loads if any signal was ever logged for that ticker
- Accuracy tracker shows "—" until 30+ signals complete (expected)

### Research Terminal (right column):
- IC table with regime labels should match
  `experiments/source_validation/results.json`
- These values are STATIC in the new design (hardcoded)
- They should be updated to match actual results.json values

### 5a — Update hardcoded IC values to match real results

The new dashboard has hardcoded IC values in the Research Terminal.
Update them to match the actual output from
`experiments/source_validation/results.json`:

```bash
python3 -c "
import json
with open('experiments/source_validation/results.json') as f:
    r = json.load(f)

# Print Layer 1 annual IC for post_count_1d and news_sentiment_1d
layer1 = r.get('layer1_annual_ic', {})
for feature in ['post_count_1d', 'news_sentiment_1d']:
    if feature in layer1:
        print(f'{feature}:')
        for year, ic in sorted(layer1[feature].items()):
            if year.isdigit():
                print(f'  {year}: {ic}')

# Print Layer 2 Granger results
print()
print('Layer 2 Granger:')
for feat, data in r.get('layer2_granger', {}).items():
    print(f'  {feat}: {data.get(\"sig_count\",0)}/6 years, verdict={data.get(\"verdict\")}')

# Print Layer 3 walk-forward
print()
print('Layer 3 best combination:')
print(r.get('conclusion', {}).get('best_combination'))
print('Mean IC:', r.get('conclusion', {}).get('best_mean_ic'))
"
```

Then find the hardcoded values in `dashboard/index.html` and update
the 2019-2025 IC rows and Granger p-value boxes to match real numbers.

The IC table HTML is in the "Signal Validation Terminal" section
(right column). The values are inline in the div text nodes.
Find them with: `grep -n "0.0481\|0.0924\|0.0035" dashboard/index.html`

Update each year row to match the real `results.json` data.

---

## Task 6 — Clean Up

```bash
# Remove old dashboard backup if everything works
rm dashboard/index_old.html

# Remove the source file from repo root if it was copied there
rm -f rsss_quantitative_dashboard.html

# Run tests
pytest tests/ -v --tb=short

# Push
bash push.sh "[dashboard] replace with new Tailwind quantitative design"
```

---

## Hard Rules for This Session

- NEVER modify the simulator JavaScript logic
  (updateSimulation, onTickerChange, toggleMode functions)
- NEVER add --reload to the uvicorn command in any plist
- ALWAYS keep the graceful fallback in fetchLiveAPIData()
  (when API unreachable, falls back to simulator silently)
- NEVER remove _sanitize() from API endpoints
  (nan values in results.json will crash JSON serialization)
- The hardcoded IC values in the Research Terminal are intentional —
  they are the final validated research findings, not live data
- Do NOT connect the Research Terminal to /research-findings API —
  it should remain static (validated findings don't change daily)
- ALWAYS backup old dashboard before replacing

---

## Expected Final State

```
Dashboard URL:     http://localhost:8000/dashboard
Default mode:      Simulator (no API needed)
Live mode:         Toggle to fetch from localhost:8000
Layout:            3-column: Portfolio | Signals | Research
Simulator:         Interactive sliders, real XGBoost logic simulation
Live data:         /portfolio, /predictions, /shap, /signal-accuracy
Research terminal: Static validated IC findings from source_validation/
```

---

*Dashboard Replacement — June 2026*
*New design: Tailwind CSS, 3-column layout, simulator + live mode*
*API contract: /portfolio /predictions?ticker= /shap/{ticker} /signal-accuracy*
