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
