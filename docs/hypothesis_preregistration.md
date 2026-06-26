# RSSS Strategy Pre-Registration
# Purpose: establish hypothesis BEFORE live data
# Version: 1.0

## Strategy Definition
- System: Long-only Reddit sentiment swing trading
- Universe: TRADE tickers (from ticker_registry.json)
- Entry conditions:
    post_count_1d >= 5 (density gate)
    stock price > 20-day moving average
    rank-based composite score > 0
    regime_score >= 0.3
    relative_volume >= 0.8
- Exit: 5-day hold OR stop-loss -8% OR take-profit +10%
- Portfolio: 70% SPY core + 30% RSSS satellite
- Max positions: 4 (portfolio_engine.py)

## Model Specification (LOCKED)
- Features: 16 (from training_metadata_v2.json)
- Architecture: XGBoost depth=1, Huber loss, GKX
- Training: 2019-2023
- Test: 2024-2025

## Validated Performance (pre-registration baseline)
- Backtest trades:      139
- Backtest win rate:    57.6%
- Backtest Sharpe:      1.36
- Walk-forward Sharpe:  0.86 (23 folds, 2019-2025)
- p-value:              0.089 (NOT significant yet)
- Trades to p<0.05:     ~42 more needed

## Primary Hypothesis
H0: Win rate = 50% (no edge)
H1: Win rate > 50% (positive edge)
Test: one-sided exact binomial at alpha=0.05

## Pre-registered Significance Tests
1. Exact binomial test (one-sided)
2. Wilson score 95% confidence interval
3. Probabilistic Sharpe Ratio vs SR*=0
4. Deflated Sharpe Ratio (N=50 trials assumed)
5. Block bootstrap Sharpe (block=5 days)

## Kill-Switch Conditions (cannot be changed post-registration)
- Rolling 40-trade IC < 0.02 → pause trading
- Strategy drawdown > 8% → halt new entries
- Win rate < 50% over 30 consecutive trades → halt
- CUSUM signal decay RED for 2 consecutive weeks → pause

## Registration Date
[auto-filled by git commit timestamp]
