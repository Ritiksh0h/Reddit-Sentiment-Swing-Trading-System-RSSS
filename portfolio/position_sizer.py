"""
ATR-based position sizing.
Sizes positions inversely to volatility so that each trade
risks the same dollar amount regardless of stock volatility.

Formula:
    position_size = (target_risk_pct × portfolio_value) / (atr_14 × price)
    Apply regime_multiplier on top.
    Hard cap: 25% of portfolio per position.

Why ATR-based:
    MSTR (ATR ~8%/day) should get much smaller allocation than
    MSFT (ATR ~1%/day). Equal-weight lets one volatile trade
    blow up the weekly -7% drawdown limit.
"""


def compute_position_size(
    portfolio_value: float,
    price: float,
    atr_14: float,
    regime_multiplier: float = 1.0,
    target_risk_pct: float = 0.02,
    max_position_pct: float = 0.25,
) -> dict:
    """
    Compute dollar position size using ATR-based risk targeting.

    Returns dict with:
        position_dollars:  dollar amount to invest
        n_shares:          number of shares (floor)
        pct_of_portfolio:  fraction of portfolio used
        risk_dollars:      expected dollar risk (1 ATR move)
        regime_multiplier: multiplier applied
    """
    if atr_14 <= 0 or price <= 0:
        return {'position_dollars': 0, 'n_shares': 0,
                'pct_of_portfolio': 0, 'risk_dollars': 0,
                'regime_multiplier': regime_multiplier}

    atr_pct = atr_14 / price

    target_risk_dollars = portfolio_value * target_risk_pct
    base_position       = target_risk_dollars / atr_pct

    sized_position = base_position * regime_multiplier

    max_position     = portfolio_value * max_position_pct
    position_dollars = min(sized_position, max_position)

    n_shares         = int(position_dollars / price)
    position_dollars = n_shares * price  # actual after rounding

    return {
        'position_dollars':  round(position_dollars, 2),
        'n_shares':          n_shares,
        'pct_of_portfolio':  round(position_dollars / portfolio_value, 4),
        'risk_dollars':      round(position_dollars * atr_pct, 2),
        'regime_multiplier': regime_multiplier,
    }


def compute_position(
    equity:       float,
    entry_price:  float,
    atr_14:       float,
    confidence:   float,
    regime:       str,
    signal_rank:  int = 1,
) -> dict:
    """
    Dynamic risk-budget position sizing (TASK 3).

    Sizes the position so that the dollar risk at the ATR-derived stop equals
    BASE_RISK_PCT × equity, scaled by regime, rank, and confidence.

    Args:
        equity:       current portfolio value
        entry_price:  fill price
        atr_14:       14-day Wilder ATR in dollars
        confidence:   model confidence [0, 1]
        regime:       'bull'/'positive'/'neutral'/'choppy'/'bear'/'negative'
        signal_rank:  1 = best signal today (rank 2 gets 20% decay)

    Returns:
        n_shares, size_dollars, stop_pct, stop_price, risk_dollars
        (plus position_dollars alias for backward compat)
    """
    from config.settings import (
        BASE_RISK_PCT, BASE_RISK_PCT_MAX,
        ATR_STOP_MULT, ATR_STOP_MIN, ATR_STOP_MAX, ATR_STOP_DEFAULT,
        POS_CAP_HIGH, POS_CAP_MED, POS_CAP_LOW,
    )

    if atr_14 <= 0 or entry_price <= 0:
        return {
            'n_shares':        0,
            'size_dollars':    0.0,
            'position_dollars': 0.0,
            'stop_pct':        ATR_STOP_DEFAULT,
            'stop_price':      0.0,
            'risk_dollars':    0.0,
            'pct_of_portfolio': 0.0,
        }

    atr_pct = atr_14 / entry_price

    # Regime multiplier — accepts both short and long label formats
    _r = regime.lower()
    if _r in ('bull', 'positive'):
        regime_mult, cap_pct = 1.0, POS_CAP_HIGH
    elif _r in ('bear', 'negative'):
        regime_mult, cap_pct = 0.5, POS_CAP_LOW
    else:                              # neutral / choppy / default
        regime_mult, cap_pct = 0.3, POS_CAP_MED

    # Rank decay: rank 1 = full, rank 2 = 80%, rank 3 = 60%, floor at 40%
    rank_decay = max(0.4, 1.0 - (signal_rank - 1) * 0.2)

    # Confidence scaling: maps [0, 1] → [0.5, 1.0]
    conf_scale = 0.5 + min(max(confidence, 0.0), 1.0) * 0.5

    effective_risk = BASE_RISK_PCT * regime_mult * rank_decay * conf_scale
    effective_risk = min(effective_risk, BASE_RISK_PCT_MAX)

    risk_dollars = equity * effective_risk

    # Compute stop before share sizing — ceiling is expressed in stop-distance space,
    # not in 1-ATR space, so we need stop_dist before deciding n_shares.
    stop_pct   = max(ATR_STOP_MIN, min(ATR_STOP_MAX, -(ATR_STOP_MULT * atr_pct)))
    stop_price = round(entry_price * (1 + stop_pct), 4)
    stop_dist  = entry_price * abs(stop_pct)   # dollars lost per share at stop

    # Initial sizing: size so that 1-ATR move = risk_dollars (not 2.5-ATR move)
    size_dollars = risk_dollars / atr_pct if atr_pct > 0 else 0.0
    size_dollars = min(size_dollars, equity * cap_pct)   # notional cap
    n_shares     = int(size_dollars / entry_price)

    # Hard ceiling: actual risk at stop must not exceed BASE_RISK_PCT_MAX × equity.
    # The initial sizing uses 1-ATR as risk distance but the real stop is ATR_STOP_MULT
    # ATRs away, so without this ceiling the realized risk can be 2.5× the budget.
    max_risk_dollars = equity * BASE_RISK_PCT_MAX
    if stop_dist > 0 and n_shares * stop_dist > max_risk_dollars:
        n_shares = int(max_risk_dollars / stop_dist)

    size_dollars = n_shares * entry_price   # actual size after all adjustments
    actual_risk  = round(n_shares * stop_dist, 2)

    return {
        'n_shares':         n_shares,
        'size_dollars':     round(size_dollars, 2),
        'position_dollars': round(size_dollars, 2),   # backward compat alias
        'stop_pct':         round(stop_pct, 6),
        'stop_price':       stop_price,
        'risk_dollars':     actual_risk,
        'risk_pct':         round(actual_risk / equity, 6) if equity > 0 else 0.0,
        'pct_of_portfolio': round(size_dollars / equity, 4) if equity > 0 else 0.0,
    }


def compute_slippage(
    price: float,
    mention_growth_7d: float,
    base_slippage: float = 0.001,
) -> float:
    """
    Dynamic slippage model — attention spike adjusted.

    Formula:
        slippage = base_slippage + (0.0005 × min(mention_growth_7d, 3.0))

    Returns slippage as a fraction (e.g. 0.0015 = 0.15%).
    """
    attention_addon = 0.0005 * min(mention_growth_7d, 3.0)
    total_slippage  = base_slippage + attention_addon
    return round(total_slippage, 6)
