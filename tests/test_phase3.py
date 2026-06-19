"""
Phase 3 tests — minimum 9 required per spec.
Run: pytest tests/test_phase3.py -v
"""
import json
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, '.')

from portfolio.position_sizer   import compute_position_size, compute_slippage
from portfolio.drift_monitor    import check_drift
from portfolio.execution_logger import log_signal
from portfolio.portfolio_engine import (
    PortfolioState, Position, check_risk_limits, check_exits,
    TAKE_PROFIT_CAP, TICKER_COOLDOWN,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_position(
    ticker='TSLA',
    entry_price=100.0,
    n_shares=10,
    entry_date='2024-01-01',
    stop_date='2024-01-08',
    predicted_return=0.05,
    atr_14=3.0,
    slippage_applied=0.001,
    regime_state='positive',
    regime_multiplier=1.0,
) -> Position:
    return Position(
        ticker=ticker,
        entry_date=entry_date,
        entry_price=entry_price,
        n_shares=n_shares,
        position_dollars=n_shares * entry_price,
        stop_date=stop_date,
        predicted_return=predicted_return,
        atr_14=atr_14,
        slippage_applied=slippage_applied,
        regime_state=regime_state,
        regime_multiplier=regime_multiplier,
        feature_vector={'returns_1d': 0.01},
    )


# ── 1. ATR-based sizing: high ATR → smaller position ─────────────────────────

def test_position_sizer_atr_based():
    """High ATR stock gets smaller position than low ATR stock (same price, portfolio).
    atr=10: base=20k (below 25k cap), atr=1: base=200k (capped at 25k).
    """
    high_vol = compute_position_size(
        portfolio_value=100_000, price=100.0, atr_14=10.0)
    low_vol  = compute_position_size(
        portfolio_value=100_000, price=100.0, atr_14=1.0)
    assert high_vol['position_dollars'] < low_vol['position_dollars'], (
        f'High ATR=${high_vol["position_dollars"]} should be < low ATR=${low_vol["position_dollars"]}'
    )


# ── 2. Hard 25% cap ───────────────────────────────────────────────────────────

def test_position_sizer_hard_cap():
    """Position never exceeds 25% of portfolio regardless of ATR."""
    result = compute_position_size(
        portfolio_value=100_000, price=100.0,
        atr_14=0.001,          # tiny ATR → would compute enormous position
        regime_multiplier=1.0,
    )
    assert result['pct_of_portfolio'] <= 0.25, (
        f'Position {result["pct_of_portfolio"]:.2%} exceeds 25% cap'
    )


# ── 3. Regime: negative when SPY below 200MA ─────────────────────────────────

def test_regime_detector_negative_on_downtrend():
    """SPY below 200MA → negative regime."""
    from portfolio.regime_detector import classify_regime, POSITION_SIZING
    import pandas as pd

    # Build a mock SPY series: last price below 200-day MA
    close = pd.Series([100.0] * 210)
    close.iloc[-1] = 50.0   # drop to half — below any 200MA

    mock_spy = pd.DataFrame({'Close': close, 'High': close, 'Low': close})

    with patch('portfolio.regime_detector.yf.download', return_value=mock_spy):
        result = classify_regime(rolling_30d_ic=None)

    assert result.label == 'negative', f'Expected negative, got {result.label}'
    assert result.multiplier == POSITION_SIZING['negative']


# ── 4. Dynamic slippage increases with attention ──────────────────────────────

def test_dynamic_slippage_increases_with_attention():
    """Higher mention_growth_7d → higher slippage."""
    low_attn  = compute_slippage(price=100.0, mention_growth_7d=0.1)
    high_attn = compute_slippage(price=100.0, mention_growth_7d=3.0)
    assert high_attn > low_attn, (
        f'High attention slippage {high_attn} should exceed low {low_attn}'
    )


def test_dynamic_slippage_capped_at_3x():
    """mention_growth_7d > 3.0 is clamped — slippage doesn't keep growing."""
    at_3   = compute_slippage(price=100.0, mention_growth_7d=3.0)
    at_100 = compute_slippage(price=100.0, mention_growth_7d=100.0)
    assert at_3 == at_100, 'Slippage should be capped at mention_growth_7d=3.0'


# ── 5. Ticker cooldown blocks re-entry ───────────────────────────────────────

def test_ticker_cooldown_blocks_reentry():
    """After trading TSLA, same ticker is blocked for 7 days."""
    state = PortfolioState()
    state.ticker_last_trade['TSLA'] = '2024-01-01'

    assert state.is_ticker_on_cooldown('TSLA', '2024-01-05')  # 4 days later — blocked
    assert not state.is_ticker_on_cooldown('TSLA', '2024-01-09')  # 8 days later — clear
    assert not state.is_ticker_on_cooldown('PLTR', '2024-01-05')  # different ticker


# ── 6. Take-profit cap triggers at 15% ───────────────────────────────────────

def test_take_profit_cap_triggers_at_15pct():
    """Position with 16% unrealized gain gets flagged for close."""
    pos   = _make_position(entry_price=100.0, stop_date='2024-12-31')
    state = PortfolioState(positions=[pos])

    exits = check_exits(state, {'TSLA': 116.0}, '2024-01-05')  # +16%
    assert len(exits) == 1
    assert exits[0]['exit_reason'] == 'take_profit_cap'


def test_position_below_take_profit_not_closed():
    """Position with 10% gain should NOT trigger take-profit (cap is 15%)."""
    pos   = _make_position(entry_price=100.0, stop_date='2024-12-31')
    state = PortfolioState(positions=[pos])

    exits = check_exits(state, {'TSLA': 110.0}, '2024-01-05')  # +10%
    assert len(exits) == 0


# ── 7. Drift monitor skips day on low post count ─────────────────────────────

def test_drift_monitor_skips_day_on_low_posts():
    """max post count < 3 across all tickers → skip_day=True (Reddit API down)."""
    result = check_drift({
        'NVDA': {'post_count_1d': 1, 'mention_growth_7d': 1.0},
        'TSLA': {'post_count_1d': 2, 'mention_growth_7d': 1.0},
    })
    assert result['skip_day'] is True
    assert not result['clean']


def test_drift_monitor_clean_on_normal_values():
    """max post count >= 3 → no skip, clean result."""
    result = check_drift({
        'NVDA': {'post_count_1d': 11, 'mention_growth_7d': 1.0},
        'TSLA': {'post_count_1d': 7,  'mention_growth_7d': 1.0},
        'MU':   {'post_count_1d': 3,  'mention_growth_7d': 1.0},
    })
    assert result['clean'] is True
    assert result['skip_day'] is False


# ── 8. Execution log creates file ────────────────────────────────────────────

def test_execution_log_creates_file(tmp_path, monkeypatch):
    """log_signal() appends to paper_trades.jsonl."""
    log_file = tmp_path / 'paper_trades.jsonl'
    monkeypatch.setattr('portfolio.execution_logger.LOG_FILE', str(log_file))

    log_signal(
        ticker='TSLA', date='2024-01-02',
        feature_vector={'returns_1d': 0.01},
        regime_state='positive', regime_multiplier=1.0,
        predicted_return_5d=0.04, atr_14=3.5,
        position_size_dollars=2500.0, slippage_applied=0.001,
        fill_price=210.5, signal_timestamp='2024-01-02T09:00:00Z',
        action='OPEN',
    )

    assert log_file.exists()
    record = json.loads(log_file.read_text().strip())
    assert record['ticker'] == 'TSLA'
    assert record['action'] == 'OPEN'


def test_execution_log_schema(tmp_path, monkeypatch):
    """Logged record has all required fields from architecture spec."""
    log_file = tmp_path / 'paper_trades.jsonl'
    monkeypatch.setattr('portfolio.execution_logger.LOG_FILE', str(log_file))

    log_signal(
        ticker='PLTR', date='2024-01-03',
        feature_vector={'returns_1d': 0.02},
        regime_state='neutral', regime_multiplier=0.75,
        predicted_return_5d=0.03, atr_14=1.2,
        position_size_dollars=1000.0, slippage_applied=0.0012,
        fill_price=20.1, signal_timestamp='2024-01-03T09:00:00Z',
        action='OPEN',
    )

    record = json.loads(log_file.read_text().strip())
    required = [
        'ticker', 'date', 'feature_vector_11', 'regime_state',
        'regime_multiplier', 'predicted_return_5d', 'atr_14',
        'position_size_dollars', 'slippage_applied', 'fill_price',
        'signal_timestamp', 'action',
    ]
    missing = [f for f in required if f not in record]
    assert missing == [], f'Missing fields: {missing}'


# ── 9. Max 3 positions enforced ───────────────────────────────────────────────

def test_max_3_positions_enforced():
    """Portfolio state with 3 positions reports max_positions_reached."""
    state = PortfolioState(positions=[
        _make_position('TSLA'),
        _make_position('PLTR'),
        _make_position('COIN'),
    ])
    limits = check_risk_limits(state, '2024-06-01')
    assert limits['max_positions_reached'] is True
    assert limits['can_open_new_trades'] is False


def test_hold_period_exit_triggers():
    """Position past stop_date should be flagged for exit."""
    pos   = _make_position(entry_price=100.0, stop_date='2024-01-07')
    state = PortfolioState(positions=[pos])

    exits = check_exits(state, {'TSLA': 103.0}, '2024-01-08')  # past stop
    assert len(exits) == 1
    assert exits[0]['exit_reason'] == 'hold_period_expired'


def test_position_sizer_returns_zero_on_bad_inputs():
    """Zero or negative ATR/price returns zero position."""
    result_zero_atr   = compute_position_size(100_000, 100.0, 0.0)
    result_zero_price = compute_position_size(100_000, 0.0, 2.0)
    assert result_zero_atr['position_dollars'] == 0
    assert result_zero_price['position_dollars'] == 0


# ── 10. Stop-loss ─────────────────────────────────────────────────────────────

def test_stop_loss_triggers_at_8pct():
    """Stop-loss must close position when unrealized loss >= 8%."""
    from datetime import date, timedelta
    today  = date.today().isoformat()
    future = (date.today() + timedelta(days=3)).isoformat()
    pos    = _make_position(entry_price=100.0, stop_date=future)
    state  = PortfolioState(positions=[pos])

    exits = check_exits(state, {'TSLA': 91.0}, today)   # -9% → triggers
    assert len(exits) == 1
    assert exits[0]['exit_reason'] == 'stop_loss'
    assert exits[0]['pnl_pct'] == pytest.approx(-0.09, abs=0.001)


def test_stop_loss_does_not_trigger_at_7pct():
    """Stop-loss should NOT fire at 7% loss (below 8% threshold)."""
    from datetime import date, timedelta
    today  = date.today().isoformat()
    future = (date.today() + timedelta(days=3)).isoformat()
    pos    = _make_position(entry_price=100.0, stop_date=future)
    state  = PortfolioState(positions=[pos])

    exits = check_exits(state, {'TSLA': 93.0}, today)   # -7% → below threshold
    assert len(exits) == 0


def test_stop_loss_takes_priority_over_hold_period():
    """Stop-loss fires even when hold period has also expired."""
    pos   = _make_position(entry_price=100.0, stop_date='2024-01-01')   # already expired
    state = PortfolioState(positions=[pos])

    exits = check_exits(state, {'TSLA': 88.0}, '2024-01-08')   # -12%
    assert len(exits) == 1
    assert exits[0]['exit_reason'] == 'stop_loss'


# ── 11. Signal classification thresholds ──────────────────────────────────────

def test_signal_classified_bullish():
    """Predictions >= 1.5% should classify as BULLISH."""
    from portfolio.signal_generator import BULLISH_THRESHOLD, BEARISH_THRESHOLD
    pred   = 0.020
    signal = ('BULLISH' if pred >= BULLISH_THRESHOLD
              else ('BEARISH' if pred <= BEARISH_THRESHOLD else 'NEUTRAL'))
    assert signal == 'BULLISH'


def test_signal_classified_bearish():
    """Predictions <= -1.5% should classify as BEARISH."""
    from portfolio.signal_generator import BULLISH_THRESHOLD, BEARISH_THRESHOLD
    pred   = -0.020
    signal = ('BULLISH' if pred >= BULLISH_THRESHOLD
              else ('BEARISH' if pred <= BEARISH_THRESHOLD else 'NEUTRAL'))
    assert signal == 'BEARISH'


def test_signal_classified_neutral():
    """Predictions between -1.5% and +1.5% should be NEUTRAL."""
    from portfolio.signal_generator import BULLISH_THRESHOLD, BEARISH_THRESHOLD
    pred   = 0.010
    signal = ('BULLISH' if pred >= BULLISH_THRESHOLD
              else ('BEARISH' if pred <= BEARISH_THRESHOLD else 'NEUTRAL'))
    assert signal == 'NEUTRAL'
