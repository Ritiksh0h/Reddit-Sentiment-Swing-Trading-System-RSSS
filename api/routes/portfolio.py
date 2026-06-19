"""
RSSS API — portfolio and trading activity routes.
"""
from fastapi import APIRouter

from api._helpers import _load_portfolio, _load_trade_log

router = APIRouter()


@router.get('/portfolio')
def get_portfolio():
    INITIAL_CAPITAL = 10000.0

    portfolio = _load_portfolio()
    cash      = float(portfolio.get('cash', INITIAL_CAPITAL))
    positions = portfolio.get('positions', [])

    position_value = sum(float(p.get('position_dollars', 0)) for p in positions)
    equity         = cash + position_value
    total_return   = round((equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2)

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
