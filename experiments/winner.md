# Phase 2 Winner: Experiment C — Expanded Data

## Summary

Experiment C passed all three Phase 2 criteria and achieved the highest Sharpe ratio.

| Metric           | Value        | Threshold | Status |
|-----------------|-------------|-----------|--------|
| IC (test 2024)  | 0.1108       | > 0.05    | PASS |
| Sharpe ratio    | 2.829      | > 1.0     | PASS |
| Total return    | 87.6%        | > SPY     | PASS |
| SPY 2024        | 26.0%        | benchmark | —      |
| Beats SPY       | YES         | required  | PASS |
| QQQ 2024        | 28.8%        | benchmark | —      |
| Beats QQQ       | YES         | optional  | PASS |

## Architecture

**Thesis:** expanded_dataset_combined_model

### Feature Set

- **Type:** Combined model on expanded 4-subreddit dataset
- **Features:** ['rsi_14', 'atr_14', 'dist_from_20ma', 'dist_from_50ma', 'relative_volume', 'returns_1d', 'returns_5d', 'returns_20d', 'volume', 'close', 'avg_sentiment_1d', 'avg_sentiment_3d', 'avg_sentiment_hc', 'weighted_sentiment', 'bullish_ratio', 'sentiment_accel', 'sentiment_std']
- **Density filter:** post_count_1d >= 10

## Backtest Parameters (locked)

- Starting capital: $1,000
- Max positions: 3
- Hold days: 5
- Slippage: 0.1%
- Fee per leg: 0.05%
- Min predicted return to trade: 2%
- No shorting, no leverage

## Detailed Metrics

- Annualized return: 102.9%
- Max drawdown: -6.3%
- Win rate: 47.5%
- Profit factor: 5.8301
- N trades: 61
- Alpha vs SPY: 61.5%

## Phase 3 Implications

This architecture is the blueprint for the full production system:
- Signal generator should implement this experiment's pre-selection and model routing logic
- Portfolio engine should use the same 5-day hold, 3-position, slippage/fee parameters
- Feature pipeline must reproduce the same feature set without look-ahead leakage

Results file: `experiments/experiment_c/results.json`
