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
MAX_POSITIONS      = 3
TAKE_PROFIT_CAP    = 0.15
STOP_LOSS_PCT      = -0.08
TICKER_COOLDOWN    = 7
DAILY_LOSS_LIMIT   = -0.03
WEEKLY_LOSS_LIMIT  = -0.07


@dataclass
class Position:
    ticker:            str
    entry_date:        str
    entry_price:       float
    n_shares:          int
    position_dollars:  float
    stop_date:         str
    predicted_return:  float
    atr_14:            float
    slippage_applied:  float
    regime_state:      str
    regime_multiplier: float
    feature_vector:    dict


@dataclass
class PortfolioState:
    cash:              float = 10000.0
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


def load_portfolio() -> PortfolioState:
    """
    Load portfolio state from data/live/paper_portfolio.json.
    Returns a fresh $10,000 PortfolioState if the file does not exist.
    """
    if Path(STATE_FILE).exists():
        with open(STATE_FILE) as f:
            data = json.load(f)
        positions = [Position(**p) for p in data.pop('positions', [])]
        state = PortfolioState(**data)
        state.positions = positions
        return state
    return PortfolioState(created_at=datetime.utcnow().isoformat())


def save_portfolio(state: PortfolioState) -> None:
    """
    Persist portfolio state to data/live/paper_portfolio.json.

    Args:
        state: current PortfolioState including positions, cash, and trade history
    """
    Path('data').mkdir(exist_ok=True)
    data = asdict(state)
    with open(STATE_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def check_risk_limits(state: PortfolioState, today: str) -> dict:
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
        'max_positions_reached': state.n_open_positions() >= MAX_POSITIONS,
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
        1. Stop-loss:   unrealized loss <= -8%
        2. Take-profit: unrealized gain >= 15%
        3. Hold expiry: today >= stop_date (5-day hold)
    """
    to_close = []

    for pos in state.positions:
        price = current_prices.get(pos.ticker)
        if price is None:
            continue

        unrealized_return = (price - pos.entry_price) / pos.entry_price

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
                f'unrealized={unrealized_return:.2%} threshold={STOP_LOSS_PCT:.0%}'
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
