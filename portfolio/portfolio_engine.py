"""
Portfolio engine.
Manages open positions, applies risk rules, tracks PnL.

Hard limits (non-negotiable):
  - Max 3 concurrent positions
  - Max 25% of portfolio per position (enforced in position_sizer.py)
  - 7-day per-ticker cooldown (prevents TSLA domination)
  - Take-profit cap at 15% unrealized gain
  - Daily loss limit: stop new trades if portfolio drops 3% in a day
  - Weekly loss limit: pause system if portfolio drops 7% in a week
"""
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

STATE_FILE         = 'data/live/paper_portfolio.json'
MAX_POSITIONS      = 4
TAKE_PROFIT_CAP    = 0.15
STOP_LOSS_PCT      = -0.08
TICKER_COOLDOWN    = 7
DAILY_LOSS_LIMIT   = -0.03
WEEKLY_LOSS_LIMIT  = -0.07


@dataclass
class Position:
    ticker:              str
    entry_date:          str
    entry_price:         float
    n_shares:            int
    position_dollars:    float
    stop_date:           str
    predicted_return:    float
    atr_14:              float
    slippage_applied:    float
    regime_state:        str
    regime_multiplier:   float
    feature_vector:      dict
    hold_days:           int   = 5    # 1, 3, or 5 — set at entry from winning horizon
    horizon:             str   = '5D' # '1D', '3D', or '5D'
    predicted_return_1d: float = 0.0
    predicted_return_3d: float = 0.0
    predicted_return_5d: float = 0.0
    pcr_confirmation:    str   = 'UNKNOWN'
    stop_pct:            float = STOP_LOSS_PCT  # per-position stop (ATR-based); default -8%
    risk_dollars:        float = 0.0            # dollar risk at entry: size × |stop_pct|


@dataclass
class PortfolioState:
    cash:              float = 100000.0
    positions:         list  = field(default_factory=list)
    closed_trades:     list  = field(default_factory=list)
    ticker_last_trade: dict  = field(default_factory=dict)
    daily_pnl:         dict  = field(default_factory=dict)
    created_at:        str   = ''

    def total_value(self, current_prices: dict) -> float:
        position_value = sum(
            p.n_shares * current_prices.get(p.ticker, p.entry_price)
            for p in self.positions
        )
        return self.cash + position_value

    def n_open_positions(self) -> int:
        return len(self.positions)

    def is_ticker_on_cooldown(self, ticker: str, today: str) -> bool:
        last = self.ticker_last_trade.get(ticker)
        if not last:
            return False
        days_since = (date.fromisoformat(today) - date.fromisoformat(last)).days
        return days_since < TICKER_COOLDOWN


def _state_from_dict(data: dict) -> PortfolioState:
    """Rebuild PortfolioState (incl. Position objects) from a plain dict."""
    data = dict(data)
    positions = [Position(**p) for p in data.pop('positions', [])]
    state = PortfolioState(**data)
    state.positions = positions
    return state


def load_portfolio() -> PortfolioState:
    """
    Load portfolio state — Supabase first (shared with GitHub Actions runners),
    data/live/paper_portfolio.json fallback, else fresh $100,000 state.
    """
    try:
        from api.db import load_portfolio_state
        remote = load_portfolio_state()
        if remote:
            logger.debug('portfolio_loaded_from_supabase')
            return _state_from_dict(remote)
    except Exception as e:
        logger.debug(f'supabase_portfolio_load_skipped: {e}')

    if Path(STATE_FILE).exists():
        with open(STATE_FILE) as f:
            return _state_from_dict(json.load(f))
    return PortfolioState(created_at=datetime.utcnow().isoformat())


def save_portfolio(state: PortfolioState) -> None:
    """
    Persist portfolio state to data/live/paper_portfolio.json AND Supabase
    (best effort — GitHub Actions runners rely on the Supabase copy).
    """
    Path(STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
    data = asdict(state)
    with open(STATE_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    try:
        from api.db import save_portfolio_state
        save_portfolio_state(data)
    except Exception as e:
        logger.debug(f'supabase_portfolio_save_skipped: {e}')


def check_risk_limits(state: PortfolioState, today: str,
                      regime: str = 'NEUTRAL') -> dict:
    """
    Check all portfolio-level risk limits.
    Returns dict with status and any active limits.
    """
    today_pnl  = state.daily_pnl.get(today, 0.0)
    week_start = (date.fromisoformat(today) - timedelta(days=7)).isoformat()
    weekly_pnl = sum(
        v for k, v in state.daily_pnl.items()
        if k >= week_start
    )

    limits = {
        'daily_loss_triggered':  today_pnl < DAILY_LOSS_LIMIT,
        'weekly_loss_triggered': weekly_pnl < WEEKLY_LOSS_LIMIT,
        'max_positions_reached': state.n_open_positions() >= get_max_positions(regime),
        'today_pnl_pct':         round(today_pnl, 4),
        'weekly_pnl_pct':        round(weekly_pnl, 4),
    }
    limits['can_open_new_trades'] = not (
        limits['daily_loss_triggered'] or
        limits['weekly_loss_triggered'] or
        limits['max_positions_reached']
    )

    if limits['daily_loss_triggered']:
        logger.warning(f'daily_loss_limit_hit pnl={today_pnl:.4f}')
    if limits['weekly_loss_triggered']:
        logger.warning(f'weekly_loss_limit_hit pnl={weekly_pnl:.4f}')

    return limits


def check_exits(
    state: PortfolioState,
    current_prices: dict,
    today: str,
) -> list:
    """
    Check all open positions for exit conditions.

    Args:
        state:          current portfolio with open positions list
        current_prices: {ticker: latest_close_price} fetched from yfinance
        today:          date string YYYY-MM-DD

    Returns:
        list of exit dicts — each has: position, exit_price, exit_date,
        exit_reason, pnl_pct. Only positions meeting a condition are included.

    Exit conditions (priority order):
        1. Stop-loss:   unrealized loss <= pos.stop_pct (ATR-based; default -8%)
        2. Take-profit: unrealized gain >= 15%
        3. Hold expiry: today >= stop_date (5-day hold)
    """
    to_close = []

    for pos in state.positions:
        price = current_prices.get(pos.ticker)
        if price is None:
            continue

        unrealized_return = (price - pos.entry_price) / pos.entry_price
        # Use per-position stop; fall back to STOP_LOSS_PCT for positions opened
        # before the stop_pct field was added (legacy JSON portfolio compat)
        stop = getattr(pos, 'stop_pct', STOP_LOSS_PCT)

        if unrealized_return <= stop:
            to_close.append({
                'position':    pos,
                'exit_price':  price,
                'exit_date':   today,
                'exit_reason': 'stop_loss',
                'pnl_pct':     round(unrealized_return, 4),
            })
            logger.info(
                f'stop_loss_triggered ticker={pos.ticker} '
                f'unrealized={unrealized_return:.2%} threshold={stop:.1%}'
            )
            continue

        if unrealized_return >= TAKE_PROFIT_CAP:
            to_close.append({
                'position':    pos,
                'exit_price':  price,
                'exit_date':   today,
                'exit_reason': 'take_profit_cap',
                'pnl_pct':     round(unrealized_return, 4),
            })
            continue

        if today >= pos.stop_date:
            to_close.append({
                'position':    pos,
                'exit_price':  price,
                'exit_date':   today,
                'exit_reason': 'hold_period_expired',
                'pnl_pct':     round(unrealized_return, 4),
            })

    return to_close


# ── Dynamic risk-budget helpers (TASK 3) ────────────────────────────────────────

def get_max_positions(regime: str) -> int:
    """Regime-aware max concurrent positions.

    Accepts both short ('bull'/'bear'/'choppy') and long
    ('POSITIVE'/'NEGATIVE'/'NEUTRAL') regime label formats.
    """
    from config.settings import (
        MAX_POSITIONS_BULL, MAX_POSITIONS_BEAR, MAX_POSITIONS_CHOPPY,
    )
    r = regime.lower()
    if r in ('bull', 'positive'):
        return MAX_POSITIONS_BULL
    elif r in ('bear', 'negative'):
        return MAX_POSITIONS_BEAR
    else:
        return MAX_POSITIONS_CHOPPY


def heat_budget_allows(
    current_positions: list,
    new_risk_dollars:  float,
    equity:            float,
    regime:            str,
) -> bool:
    """Return True if adding new_risk_dollars keeps total heat within budget.

    Heat = sum of risk_dollars across all open positions.
    Budget is a % of equity that varies by regime.
    """
    from config.settings import (
        HEAT_BUDGET_BULL, HEAT_BUDGET_BEAR, HEAT_BUDGET_CHOPPY,
    )
    r = regime.lower()
    if r in ('bull', 'positive'):
        budget_pct = HEAT_BUDGET_BULL
    elif r in ('bear', 'negative'):
        budget_pct = HEAT_BUDGET_BEAR
    else:
        budget_pct = HEAT_BUDGET_CHOPPY

    current_heat   = sum(getattr(p, 'risk_dollars', 0.0) for p in current_positions)
    budget_dollars = equity * budget_pct
    return (current_heat + new_risk_dollars) <= budget_dollars


def correlation_allows(
    new_ticker:      str,
    current_tickers: list,
    lookback_days:   int = 60,
) -> bool:
    """Return True if adding new_ticker passes diversification gates.

    Gate 1 — semiconductor cluster cap (NVDA/AMD/MU/INTC/ARM ≤ MAX_CORR_CLUSTER).
    Gate 2 — max pairwise 60-day return correlation < MAX_BOOK_CORR.

    Fails open (returns True) when yfinance data is unavailable — never blocks
    trading due to a network error.
    """
    import pandas as _pd
    import yfinance as _yf
    from config.settings import SEMI_CLUSTER, MAX_CORR_CLUSTER, MAX_BOOK_CORR

    if not current_tickers:
        return True

    # Gate 1: semiconductor cluster cap (no network needed)
    if new_ticker in SEMI_CLUSTER:
        cluster_count = sum(1 for t in current_tickers if t in SEMI_CLUSTER)
        if cluster_count >= MAX_CORR_CLUSTER:
            logger.info(
                f'correlation_blocked ticker={new_ticker} '
                f'reason=semi_cluster_cap count={cluster_count}'
            )
            return False

    # Gate 2: pairwise return correlation
    try:
        all_tickers = list({new_ticker, *current_tickers})
        end_dt   = _pd.Timestamp.today()
        start_dt = end_dt - _pd.Timedelta(days=lookback_days + 10)

        raw = _yf.download(
            all_tickers,
            start=start_dt.strftime('%Y-%m-%d'),
            end=end_dt.strftime('%Y-%m-%d'),
            auto_adjust=True, progress=False, threads=False,
        )
        if isinstance(raw.columns, _pd.MultiIndex):
            prices = raw['Close']
        else:
            prices = raw[['Close']].rename(columns={'Close': all_tickers[0]})

        prices = prices.dropna(how='all')
        if len(prices) < 20:
            return True  # insufficient history — fail open

        rets = prices.pct_change().dropna(how='all')

        if new_ticker not in rets.columns:
            return True

        for existing in current_tickers:
            if existing not in rets.columns:
                continue
            corr = rets[new_ticker].corr(rets[existing])
            if not _pd.isna(corr) and abs(corr) >= MAX_BOOK_CORR:
                logger.info(
                    f'correlation_blocked ticker={new_ticker} '
                    f'corr_with={existing} corr={corr:.3f}'
                )
                return False

    except Exception as exc:
        logger.warning(
            f'correlation_check_failed ticker={new_ticker}: {exc} — allowing'
        )
        return True

    return True


