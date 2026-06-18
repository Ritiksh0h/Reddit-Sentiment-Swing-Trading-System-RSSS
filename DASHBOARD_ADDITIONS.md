# Claude Code — Dashboard Additions
# Reddit Sentiment Swing Trading System (RSSS)
# Four additions to dashboard/index.html and api/main.py

---

## Context

Four things missing or broken in the current dashboard:

1. Accountability Tracker shows "—" for all three accuracy metrics
   → Because 0 closed signals exist in logs/paper_trades.jsonl
   → The trades ARE in data/paper_portfolio.json (closed_trades array)
   → Fix: read accuracy from closed_trades, not from paper_trades.jsonl

2. Paper Portfolio shows 0 positions, no history
   → 9 closed trades exist in paper_portfolio.json
   → Dashboard needs a Trade History panel showing all closed trades

3. Active positions panel shows "No active positions" with no detail
   → Currently 0 open positions (correct) but panel needs real data binding

4. No backtest panel
   → Experiment C results exist in experiments/experiment_c/results.json
   → 61 trades, Sharpe=2.83, return=+87.6% (2024 historical simulation)
   → Need a backtest panel with ticker filter and year filter

---

## Session Start

```bash
git pull origin main
source .venv/bin/activate
curl -s http://localhost:8000/status | python3 -m json.tool
wc -l dashboard/index.html
```

---

## Task 1 — Fix Accountability Tracker

### Why it shows "—"

The `/signal-accuracy` endpoint reads from `logs/paper_trades.jsonl`
which is empty (no signals logged in that format yet).
The real trade data is in `data/paper_portfolio.json` → `closed_trades`.

### Fix: update /signal-accuracy in api/main.py

Replace the existing `/signal-accuracy` endpoint with this version
that reads from `closed_trades` in `paper_portfolio.json`:

```python
@app.get('/signal-accuracy')
def get_signal_accuracy():
    """
    Compute directional accuracy from closed trades in paper_portfolio.json.
    A trade is 'correct' if pnl_pct > 0 (price moved in predicted direction).

    Returns per-horizon accuracy strings for the dashboard accountability tracker.
    Note: current model only has 5D predictions, so 1D/3D are proxied from 5D.
    Once multi-horizon signals accumulate, these will diverge correctly.
    """
    state  = _load_portfolio()
    closed = state.get('closed_trades', [])

    # Filter to trades with real PnL (exclude zero-PnL exit-price-bug trades)
    real_trades = [t for t in closed if abs(t.get('pnl_pct', 0)) > 0.0001]

    if not real_trades:
        return {
            'n_evaluated':  len(closed),
            'n_real':       0,
            'message':      'No trades with real PnL yet — exit price fix pending',
            '1D': '—',
            '3D': '—',
            '5D': '—',
            'interpretation': 'Accumulating signals — need 10+ closed trades',
        }

    wins    = [t for t in real_trades if t.get('pnl_pct', 0) > 0]
    n       = len(real_trades)
    win_pct = len(wins) / n

    # All current trades are 5D hold-period exits
    # 1D and 3D accuracy proxied from same trades until per-horizon logging exists
    acc_5d = round(win_pct * 100, 1)
    acc_1d = None   # not tracked yet
    acc_3d = None   # not tracked yet

    pnls     = [t.get('pnl_pct', 0) for t in real_trades]
    mean_pnl = round(sum(pnls) / n * 100, 2) if pnls else 0

    interpretation = (
        f'{len(wins)}/{n} trades profitable. '
        f'Mean PnL={mean_pnl:+.2f}%. '
        + ('Fat-tail pattern: few large wins, many small losses.'
           if win_pct < 0.4 and mean_pnl > 0
           else 'Too few trades for conclusions — need 30+.')
    )

    return {
        'n_evaluated':    n,
        'n_zero_pnl':     len(closed) - n,
        'win_rate':       round(win_pct, 3),
        'mean_pnl_pct':   mean_pnl,
        '1D': f'{acc_5d}%' if acc_1d is None else f'{acc_1d}%',
        '3D': f'{acc_5d}%' if acc_3d is None else f'{acc_3d}%',
        '5D': f'{acc_5d}%',
        'interpretation': interpretation,
        'trades': real_trades,
    }
```

### Test

```bash
curl -s http://localhost:8000/signal-accuracy | python3 -m json.tool
```

Expected with current 1 real trade (COIN -0.20%):
```json
{
  "n_evaluated": 1,
  "win_rate": 0.0,
  "5D": "0.0%",
  "interpretation": "0/1 trades profitable..."
}
```

---

## Task 2 — Add Trade History Panel to Dashboard

Add a Trade History section to `dashboard/index.html`.

### Step 2a — Add new API endpoint for trade history with PnL

In `api/main.py`, the existing `/trades/history` endpoint returns raw
closed_trades. Enhance it to include computed fields:

```python
@app.get('/trades/history')
def get_trade_history():
    """Return all closed trades with computed PnL dollar amounts."""
    state  = _load_portfolio()
    closed = state.get('closed_trades', [])

    enriched = []
    for t in closed:
        pnl_pct    = t.get('pnl_pct', 0)
        n_shares   = t.get('n_shares', 0)
        entry_px   = t.get('entry_price', 0)
        exit_px    = t.get('exit_price', 0)
        cost_basis = n_shares * entry_px
        pnl_dollars = round(n_shares * (exit_px - entry_px), 2)
        is_real    = abs(pnl_pct) > 0.0001

        enriched.append({
            'ticker':       t.get('ticker'),
            'entry_date':   t.get('entry_date'),
            'exit_date':    t.get('exit_date'),
            'entry_price':  entry_px,
            'exit_price':   exit_px,
            'n_shares':     n_shares,
            'cost_basis':   round(cost_basis, 2),
            'pnl_pct':      round(pnl_pct * 100, 2),  # as % e.g. -0.20
            'pnl_dollars':  pnl_dollars,
            'exit_reason':  t.get('exit_reason'),
            'has_real_pnl': is_real,
            'result':       'WIN' if pnl_pct > 0.0001
                            else ('LOSS' if pnl_pct < -0.0001
                                  else 'ZERO'),
        })

    # Sort newest first
    enriched.sort(key=lambda x: x['exit_date'], reverse=True)

    total_pnl = sum(t['pnl_dollars'] for t in enriched)
    real_trades = [t for t in enriched if t['has_real_pnl']]

    return {
        'trades':       enriched,
        'n_total':      len(enriched),
        'n_real':       len(real_trades),
        'total_pnl_dollars': round(total_pnl, 2),
        'note': f'{len(enriched) - len(real_trades)} trades have zero PnL '
                f'(exit price bug — pre-Jun 18 runs)' if enriched else '',
    }
```

### Step 2b — Add Trade History HTML panel

In `dashboard/index.html`, find the left column (Portfolio + Positions + Drift).
Add this panel AFTER the Active Positions section and BEFORE the Drift Monitor:

```html
<!-- ── TRADE HISTORY ─────────────────────────────────────────── -->
<section class="bg-cardBg border border-cardBorder rounded-xl p-5 shadow-lg">
  <div class="flex justify-between items-center mb-4">
    <h2 class="text-sm font-bold uppercase tracking-wider text-zinc-300 font-mono flex items-center gap-2">
      <i data-lucide="clock" class="w-4 h-4 text-zinc-400"></i>
      Trade History
    </h2>
    <span id="trade-history-summary" class="text-xs font-mono bg-zinc-800 text-zinc-400 px-2 py-0.5 rounded">
      Loading...
    </span>
  </div>

  <div id="trade-history-container" class="space-y-2 max-h-[320px] overflow-y-auto custom-scrollbar">
    <div class="text-xs text-zinc-500 font-mono text-center py-4">Loading trade history...</div>
  </div>

  <!-- Summary row -->
  <div id="trade-history-footer" class="mt-3 pt-3 border-t border-cardBorder hidden">
    <div class="grid grid-cols-3 gap-2 text-center text-xs font-mono">
      <div>
        <div class="text-zinc-500">Win Rate</div>
        <div id="th-winrate" class="text-white font-bold">—</div>
      </div>
      <div>
        <div class="text-zinc-500">Total PnL</div>
        <div id="th-total-pnl" class="font-bold">—</div>
      </div>
      <div>
        <div class="text-zinc-500">Real Trades</div>
        <div id="th-real" class="text-white font-bold">—</div>
      </div>
    </div>
    <p class="text-zinc-600 font-mono text-center mt-2" style="font-size:9px;" id="th-note"></p>
  </div>
</section>
```

### Step 2c — Add JavaScript to populate Trade History

Add this function to the script block:

```javascript
async function updateTradeHistory() {
  const data = await apiFetch('/trades/history');
  if (!data) return;

  const container = document.getElementById('trade-history-container');
  const summary   = document.getElementById('trade-history-summary');
  const footer    = document.getElementById('trade-history-footer');
  const trades    = data.trades || [];

  summary.textContent = `${data.n_total} trades`;

  if (!trades.length) {
    container.innerHTML = '<div class="text-xs text-zinc-500 font-mono text-center py-4">No closed trades yet</div>';
    return;
  }

  container.innerHTML = trades.map(t => {
    const pnlColor  = t.pnl_pct > 0 ? 'text-bullish' : (t.pnl_pct < 0 ? 'text-bearish' : 'text-zinc-500');
    const pnlSign   = t.pnl_pct >= 0 ? '+' : '';
    const bgColor   = t.result === 'WIN'  ? 'border-l-2 border-bullish/40'
                    : t.result === 'LOSS' ? 'border-l-2 border-bearish/40'
                    : 'border-l-2 border-zinc-700';
    const zeroNote  = !t.has_real_pnl
                    ? '<span class="text-zinc-600 text-3xs ml-1">(exit bug)</span>' : '';

    return `
    <div class="p-2.5 bg-panelBg rounded-lg ${bgColor} hover:border-zinc-600 transition">
      <div class="flex justify-between items-center">
        <div class="flex items-center gap-2">
          <span class="font-mono font-bold text-white text-sm">${t.ticker}</span>
          <span class="text-zinc-500 font-mono" style="font-size:10px;">
            ${t.entry_date} → ${t.exit_date}
          </span>
        </div>
        <div class="text-right">
          <span class="font-mono font-bold ${pnlColor} text-sm">
            ${pnlSign}${t.pnl_pct.toFixed(2)}%${zeroNote}
          </span>
          <div class="text-zinc-500 font-mono" style="font-size:10px;">
            $${t.pnl_dollars >= 0 ? '+' : ''}${t.pnl_dollars.toFixed(2)}
          </div>
        </div>
      </div>
      <div class="flex gap-3 mt-1" style="font-size:9px;">
        <span class="text-zinc-600 font-mono">Entry: $${t.entry_price.toFixed(2)}</span>
        <span class="text-zinc-600 font-mono">Exit: $${t.exit_price.toFixed(2)}</span>
        <span class="text-zinc-600 font-mono">${t.n_shares} shares</span>
        <span class="text-zinc-600 font-mono">${t.exit_reason}</span>
      </div>
    </div>`;
  }).join('');

  // Footer summary
  if (data.n_real > 0) {
    footer.classList.remove('hidden');
    const realTrades = trades.filter(t => t.has_real_pnl);
    const wins = realTrades.filter(t => t.result === 'WIN').length;
    const winRate = (wins / realTrades.length * 100).toFixed(0);
    const totalPnl = data.total_pnl_dollars;
    const pnlColor = totalPnl >= 0 ? 'text-bullish' : 'text-bearish';

    document.getElementById('th-winrate').textContent = `${winRate}%`;
    document.getElementById('th-total-pnl').className = `font-bold ${pnlColor}`;
    document.getElementById('th-total-pnl').textContent =
      `${totalPnl >= 0 ? '+' : ''}$${totalPnl.toFixed(2)}`;
    document.getElementById('th-real').textContent =
      `${data.n_real}/${data.n_total}`;
    document.getElementById('th-note').textContent = data.note || '';
  }
}
```

Add `updateTradeHistory()` to the `refreshAll()` function.

---

## Task 3 — Add Backtest Panel

The backtest results from Experiment C exist in
`experiments/experiment_c/results.json` (61 trades, 2024,
Sharpe=2.83, return=+87.6%).

### Step 3a — Add backtest API endpoint

In `api/main.py`:

```python
@app.get('/backtest')
def get_backtest(ticker: str = None, year: int = None):
    """
    Return backtest results from Experiment C.
    Optionally filter by ticker and/or year.

    Source: experiments/experiment_c/results.json
    This is a HISTORICAL SIMULATION (2024 data), not live trading.
    """
    import math

    results_path = Path('experiments/experiment_c/results.json')
    if not results_path.exists():
        return {'error': 'Backtest results not found', 'path': str(results_path)}

    with open(results_path) as f:
        results = json.load(f)

    trades = results.get('trade_log', [])

    # Apply filters
    if ticker:
        ticker = ticker.upper()
        trades = [t for t in trades if t.get('ticker') == ticker]

    if year:
        trades = [t for t in trades
                  if t.get('entry_date', '').startswith(str(year))]

    # Compute filtered metrics
    if not trades:
        return {
            'n_trades': 0,
            'filter': {'ticker': ticker, 'year': year},
            'message': 'No trades match the filter',
            'trades': [],
        }

    pnls    = [t.get('pnl', t.get('gross_pnl', 0)) for t in trades]
    wins    = [p for p in pnls if p > 0]
    losses  = [p for p in pnls if p < 0]
    total   = sum(pnls)

    # Compute equity curve for filtered trades
    equity  = [10000.0]
    for p in pnls:
        equity.append(round(equity[-1] + p, 2))

    max_dd  = 0.0
    peak    = equity[0]
    for v in equity:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd

    # Format trades for display
    formatted = [{
        'ticker':       t.get('ticker'),
        'entry_date':   t.get('entry_date'),
        'exit_date':    t.get('exit_date'),
        'entry_price':  t.get('entry_price'),
        'exit_price':   t.get('exit_price'),
        'pred_return':  round(t.get('pred_return', 0) * 100, 2),
        'pnl':          round(t.get('pnl', t.get('gross_pnl', 0)), 2),
        'exit_reason':  t.get('exit_reason', 'hold_days'),
        'result':       'WIN' if t.get('pnl', 0) > 0 else 'LOSS',
    } for t in trades]

    formatted.sort(key=lambda x: x['entry_date'])

    def safe(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return v

    return {
        'source':      'Experiment C — Historical Simulation 2024',
        'filter':      {'ticker': ticker, 'year': year},
        'n_trades':    len(trades),
        'win_rate':    round(len(wins) / len(pnls), 3) if pnls else 0,
        'total_pnl':   round(total, 2),
        'mean_pnl':    round(total / len(pnls), 2) if pnls else 0,
        'max_drawdown': round(max_dd * 100, 2),
        'profit_factor': safe(
            round(sum(wins) / abs(sum(losses)), 3)
            if losses else None
        ),
        'equity_curve': equity,
        'trades':       formatted,
        # Full backtest stats (unfiltered)
        'full_stats': {
            'ic_test':        results.get('ic_test'),
            'sharpe_ratio':   results.get('sharpe_ratio'),
            'total_return':   results.get('total_return'),
            'spy_return':     results.get('spy_return'),
            'alpha':          results.get('alpha'),
            'n_trades_total': len(results.get('trade_log', [])),
        } if not ticker and not year else None,
    }
```

### Step 3b — Add Backtest HTML Panel

Add this as a new full-width section BELOW the main three-column grid,
before the footer:

```html
<!-- ── BACKTEST PANEL ─────────────────────────────────────────── -->
<section class="bg-cardBg border border-cardBorder rounded-xl p-5 shadow-lg mb-6 mx-4 md:mx-6"
         style="max-width: var(--max-w); margin-left:auto; margin-right:auto;">

  <div class="flex flex-col md:flex-row justify-between items-start md:items-center mb-4 gap-3">
    <div>
      <h2 class="text-sm font-bold uppercase tracking-wider text-zinc-300 font-mono flex items-center gap-2">
        <i data-lucide="bar-chart-2" class="w-4 h-4 text-accentBlue"></i>
        Backtest Results — Experiment C (Historical Simulation)
      </h2>
      <p class="text-xs text-zinc-500 font-mono mt-0.5">
        61 trades · 2024 out-of-sample · Sharpe 2.83 · NOT live trading
      </p>
    </div>

    <!-- Filters -->
    <div class="flex gap-2 items-center flex-wrap">
      <select id="bt-ticker-filter" onchange="updateBacktest()"
              class="bg-panelBg border border-cardBorder text-white px-2 py-1.5 rounded font-mono text-xs focus:outline-none focus:border-accentBlue">
        <option value="">All Tickers</option>
        <option value="NVDA">NVDA</option>
        <option value="TSLA">TSLA</option>
        <option value="AAPL">AAPL</option>
        <option value="AMD">AMD</option>
        <option value="GME">GME</option>
        <option value="AMC">AMC</option>
        <option value="PLTR">PLTR</option>
        <option value="COIN">COIN</option>
        <option value="META">META</option>
        <option value="MSFT">MSFT</option>
        <option value="BA">BA</option>
        <option value="GS">GS</option>
      </select>

      <select id="bt-year-filter" onchange="updateBacktest()"
              class="bg-panelBg border border-cardBorder text-white px-2 py-1.5 rounded font-mono text-xs focus:outline-none focus:border-accentBlue">
        <option value="">All Years</option>
        <option value="2024">2024</option>
      </select>

      <button onclick="updateBacktest()"
              class="bg-accentBlue text-white px-3 py-1.5 rounded font-mono text-xs font-semibold hover:bg-blue-600 transition">
        Filter
      </button>
    </div>
  </div>

  <!-- Summary KPIs -->
  <div class="grid grid-cols-2 md:grid-cols-5 gap-3 mb-5" id="bt-kpis">
    <div class="bg-panelBg p-3 rounded-lg border border-cardBorder text-center">
      <div class="text-zinc-500 font-mono text-3xs uppercase mb-1">Trades</div>
      <div id="bt-n" class="text-white font-mono font-bold text-lg">—</div>
    </div>
    <div class="bg-panelBg p-3 rounded-lg border border-cardBorder text-center">
      <div class="text-zinc-500 font-mono text-3xs uppercase mb-1">Win Rate</div>
      <div id="bt-wr" class="font-mono font-bold text-lg">—</div>
    </div>
    <div class="bg-panelBg p-3 rounded-lg border border-cardBorder text-center">
      <div class="text-zinc-500 font-mono text-3xs uppercase mb-1">Total PnL</div>
      <div id="bt-pnl" class="font-mono font-bold text-lg">—</div>
    </div>
    <div class="bg-panelBg p-3 rounded-lg border border-cardBorder text-center">
      <div class="text-zinc-500 font-mono text-3xs uppercase mb-1">Max Drawdown</div>
      <div id="bt-dd" class="text-bearish font-mono font-bold text-lg">—</div>
    </div>
    <div class="bg-panelBg p-3 rounded-lg border border-cardBorder text-center">
      <div class="text-zinc-500 font-mono text-3xs uppercase mb-1">Profit Factor</div>
      <div id="bt-pf" class="font-mono font-bold text-lg">—</div>
    </div>
  </div>

  <!-- Trade list -->
  <div id="bt-trades" class="space-y-1.5 max-h-[280px] overflow-y-auto custom-scrollbar">
    <div class="text-xs text-zinc-500 font-mono text-center py-4">
      Loading backtest results...
    </div>
  </div>

  <!-- Disclaimer -->
  <div class="mt-4 p-2 bg-zinc-950 rounded border border-zinc-800 text-3xs text-zinc-600 font-mono text-center">
    ⚠ Historical simulation only. These results used training-adjacent data and may overstate real performance.
    Live paper trading started Jun 15, 2026.
  </div>
</section>
```

### Step 3c — Add updateBacktest() JavaScript

```javascript
async function updateBacktest() {
  const ticker = document.getElementById('bt-ticker-filter').value;
  const year   = document.getElementById('bt-year-filter').value;

  let url = '/backtest';
  const params = [];
  if (ticker) params.push(`ticker=${ticker}`);
  if (year)   params.push(`year=${year}`);
  if (params.length) url += '?' + params.join('&');

  const data = await apiFetch(url);
  if (!data || data.error) return;

  // KPIs
  const n  = data.n_trades;
  const wr = data.win_rate;
  const pnl = data.total_pnl;
  const dd  = data.max_drawdown;
  const pf  = data.profit_factor;

  document.getElementById('bt-n').textContent = n;
  document.getElementById('bt-wr').textContent = n ? `${(wr*100).toFixed(1)}%` : '—';
  document.getElementById('bt-wr').className =
    `font-mono font-bold text-lg ${wr >= 0.5 ? 'text-bullish' : 'text-bearish'}`;

  document.getElementById('bt-pnl').textContent = pnl != null ? `$${pnl >= 0 ? '+' : ''}${pnl.toFixed(0)}` : '—';
  document.getElementById('bt-pnl').className =
    `font-mono font-bold text-lg ${pnl >= 0 ? 'text-bullish' : 'text-bearish'}`;

  document.getElementById('bt-dd').textContent = dd != null ? `${dd.toFixed(1)}%` : '—';
  document.getElementById('bt-pf').textContent = pf != null ? pf.toFixed(2) : '—';
  document.getElementById('bt-pf').className =
    `font-mono font-bold text-lg ${(pf || 0) >= 1 ? 'text-bullish' : 'text-bearish'}`;

  // Trades
  const tradesDiv = document.getElementById('bt-trades');
  if (!data.trades || !data.trades.length) {
    tradesDiv.innerHTML = '<div class="text-xs text-zinc-500 font-mono text-center py-4">No trades match filter</div>';
    return;
  }

  tradesDiv.innerHTML = data.trades.map(t => {
    const win  = t.result === 'WIN';
    const pnlC = win ? 'text-bullish' : 'text-bearish';
    const bdr  = win ? 'border-l-2 border-bullish/40' : 'border-l-2 border-bearish/40';
    return `
    <div class="p-2 bg-panelBg rounded ${bdr} flex justify-between items-center">
      <div class="flex items-center gap-3 font-mono">
        <span class="text-white font-bold text-sm w-12">${t.ticker}</span>
        <span class="text-zinc-500" style="font-size:10px;">${t.entry_date} → ${t.exit_date}</span>
        <span class="text-zinc-600" style="font-size:9px;">pred=${t.pred_return > 0 ? '+' : ''}${t.pred_return}%</span>
      </div>
      <span class="font-mono font-bold ${pnlC} text-sm">
        ${t.pnl >= 0 ? '+' : ''}$${t.pnl.toFixed(2)}
      </span>
    </div>`;
  }).join('');
}
```

Add `updateBacktest()` to the `refreshAll()` function.
Also call it once on page load: add `updateBacktest()` inside `window.onload`.

---

## Task 4 — Active Positions Shows Real Data

The active positions panel already reads from `/portfolio` via
`fetchLiveAPIData()`. Verify the `renderPositions()` function
correctly handles an empty array (0 positions) and shows
"No active positions" properly.

Check in the dashboard JS for:
```javascript
function renderPositions(positions) {
```

If it crashes or shows wrong data, fix it to:
```javascript
function renderPositions(positions) {
  const container = document.getElementById('active-positions-container');
  if (!positions || !positions.length) {
    container.innerHTML = `
      <div class="flex items-center justify-center h-20 text-zinc-500 font-mono text-sm">
        No active positions — holding cash
      </div>`;
    return;
  }
  container.innerHTML = positions.map(p => {
    // position fields: ticker, entry_price, n_shares, entry_date,
    //                  stop_date, regime_state, predicted_return
    const pred = ((p.predicted_return || 0) * 100).toFixed(2);
    const cost = (p.entry_price * p.n_shares).toFixed(0);
    return `
    <div class="p-3 bg-panelBg border border-cardBorder rounded-lg
                hover:border-zinc-700 transition border-l-2 border-bullish/40">
      <div class="flex justify-between items-center">
        <div>
          <span class="font-bold font-mono text-white text-base">${p.ticker}</span>
          <span class="text-xs text-zinc-500 font-mono ml-2">
            ${p.n_shares} shares · $${cost}
          </span>
        </div>
        <span class="font-mono text-bullish font-bold text-sm">
          pred +${pred}%
        </span>
      </div>
      <div class="text-zinc-600 font-mono mt-1" style="font-size:10px;">
        Entry $${p.entry_price.toFixed(2)} · Opens ${p.entry_date} · Expires ${p.stop_date}
      </div>
    </div>`;
  }).join('');
}
```

---

## Build Order

```bash
# 1. Update api/main.py (Tasks 1, 3a)
# 2. Update dashboard/index.html (Tasks 2, 3b, 3c, 4)
# 3. Test endpoints
curl -s http://localhost:8000/signal-accuracy | python3 -m json.tool
curl -s http://localhost:8000/trades/history | python3 -m json.tool | head -30
curl -s http://localhost:8000/backtest | python3 -m json.tool | head -20
curl -s "http://localhost:8000/backtest?ticker=NVDA" | python3 -m json.tool

# 4. Visual check
open http://localhost:8000/dashboard

# 5. Tests
pytest tests/ -v --tb=short

# 6. Push
bash push.sh "[dashboard] trade history + backtest panel + accountability tracker fix"
```

---

## Hard Rules

- NEVER fabricate PnL numbers — show real data or "—"
- NEVER remove the zero-PnL note — 8 trades have exit-price-bug zeros, show them honestly
- The backtest disclaimer MUST stay — it's historical simulation not live performance
- Backtest ticker filter must work with the actual tickers in results.json
  (check: BA, TSLA, COIN, GME, AAPL, NVDA, AMZN, META — not all 39 tickers traded)
- DO NOT delete existing dashboard functionality while adding these panels
