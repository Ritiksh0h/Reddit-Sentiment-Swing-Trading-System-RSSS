"""
RSSS API — portfolio and trading activity routes.
"""
from fastapi import APIRouter

from api._helpers import _load_portfolio, _load_trade_log

router = APIRouter()


@router.get('/portfolio')
def get_portfolio():
    import yfinance as yf
    import pandas as pd
    from datetime import date

    INITIAL_CAPITAL = 10000.0

    portfolio = _load_portfolio()
    cash      = float(portfolio.get('cash', INITIAL_CAPITAL))
    positions = portfolio.get('positions', [])

    today = date.today()
    for p in positions:
        entry_price = float(p.get('entry_price', 0))
        n_shares    = int(p.get('n_shares', 0))
        ticker      = p.get('ticker', '')

        try:
            mkt = yf.download(ticker, period='2d', auto_adjust=True, progress=False)
            if isinstance(mkt.columns, pd.MultiIndex):
                mkt.columns = mkt.columns.get_level_values(0)
            current_price = float(mkt['Close'].dropna().iloc[-1])
        except Exception:
            current_price = entry_price

        unrealized_pct     = (current_price - entry_price) / entry_price if entry_price else 0
        unrealized_dollars = (current_price - entry_price) * n_shares

        p['current_price']       = round(current_price, 4)
        p['unrealized_pct']      = round(unrealized_pct, 4)
        p['unrealized_dollars']  = round(unrealized_dollars, 2)

        try:
            entry_dt = date.fromisoformat(p['entry_date'])
            stop_dt  = date.fromisoformat(p['stop_date'])
            p['days_held']      = (today - entry_dt).days
            p['days_remaining'] = max(0, (stop_dt - today).days)
        except Exception:
            p['days_held']      = 0
            p['days_remaining'] = 0

    position_value = sum(
        float(p.get('current_price', p.get('entry_price', 0))) * int(p.get('n_shares', 0))
        for p in positions
    )
    equity       = cash + position_value
    total_return = round((equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2)

    try:
        from portfolio.regime_detector import RegimeDetector
        regime_label = RegimeDetector().get_current_regime().upper()
    except Exception:
        regime_label = 'NEUTRAL'

    _REGIME_SIZING = {'POSITIVE': 100, 'NEUTRAL': 75, 'NEGATIVE': 50}
    sizing_pct     = _REGIME_SIZING.get(regime_label, 75)

    return {
        **portfolio,
        'equity':           round(equity, 2),
        'total_return_pct': total_return,
        'positions_count':  len(positions),
        'regime_label':     regime_label,
        'sizing_pct':       sizing_pct,
    }


@router.get('/positions')
def get_positions():
    return _load_portfolio().get('positions', [])


@router.get('/signals')
def get_signals(n: int = 50):
    """
    Return all OPEN signals from the latest daily run date.
    Falls back to the most recent n signals if no date boundary is found.
    """
    trades = _load_trade_log(n * 5)
    opens  = [t for t in trades if t.get('action') == 'OPEN']
    if not opens:
        return {'date': None, 'signals': [], 'total': 0}

    latest_date = max(t.get('date', '') for t in opens)
    day_signals = [t for t in opens if t.get('date') == latest_date]

    bullish = sorted(
        [t for t in day_signals if t.get('signal') == 'BULLISH'],
        key=lambda x: x.get('predicted_return_5d') or x.get('predicted_5d') or 0,
        reverse=True,
    )
    bearish = sorted(
        [t for t in day_signals if t.get('signal') == 'BEARISH'],
        key=lambda x: x.get('predicted_return_5d') or x.get('predicted_5d') or 0,
    )
    neutral = [t for t in day_signals if t.get('signal', 'NEUTRAL') == 'NEUTRAL']

    return {
        'date':    latest_date,
        'signals': bullish + neutral + bearish,
        'total':   len(day_signals),
        'bullish': len(bullish),
        'bearish': len(bearish),
        'neutral': len(neutral),
    }


@router.get('/signals/recent')
def get_recent_signals(n: int = 20):
    trades = _load_trade_log(n * 3)
    return [t for t in trades if t.get('action') == 'OPEN'][-n:]


@router.get('/trades/history')
def get_trade_history():
    """Return all closed trades with computed PnL dollar amounts."""
    state  = _load_portfolio()
    closed = state.get('closed_trades', [])

    enriched = []
    for t in closed:
        pnl_pct     = t.get('pnl_pct', 0)
        n_shares    = t.get('n_shares', 0)
        entry_px    = t.get('entry_price', 0)
        exit_px     = t.get('exit_price', 0)
        cost_basis  = n_shares * entry_px
        pnl_dollars = round(n_shares * (exit_px - entry_px), 2)
        is_real     = abs(pnl_pct) > 0.0001

        enriched.append({
            'ticker':       t.get('ticker'),
            'entry_date':   t.get('entry_date'),
            'exit_date':    t.get('exit_date'),
            'entry_price':  entry_px,
            'exit_price':   exit_px,
            'n_shares':     n_shares,
            'cost_basis':   round(cost_basis, 2),
            'pnl_pct':      round(pnl_pct * 100, 2),
            'pnl_dollars':  pnl_dollars,
            'exit_reason':  t.get('exit_reason'),
            'has_real_pnl': is_real,
            'result': ('WIN'  if pnl_pct >  0.0001
                       else ('LOSS' if pnl_pct < -0.0001
                             else 'ZERO')),
        })

    enriched.sort(key=lambda x: x['exit_date'] or '', reverse=True)

    total_pnl   = sum(t['pnl_dollars'] for t in enriched)
    real_trades = [t for t in enriched if t['has_real_pnl']]

    return {
        'trades':            enriched,
        'n_total':           len(enriched),
        'n_real':            len(real_trades),
        'total_pnl_dollars': round(total_pnl, 2),
        'note': (f'{len(enriched) - len(real_trades)} trades have zero PnL '
                 f'(exit price bug — pre-Jun 18 runs)') if enriched else '',
    }


@router.get('/log/recent')
def get_recent_log(n: int = 50):
    return _load_trade_log(n)
