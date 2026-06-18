"""
Daily trading orchestrator.
Run before market open: python scripts/daily_run.py

Sequence:
    1. Load portfolio state
    2. Check risk limits (daily/weekly loss)
    3. Check regime
    4. Load Reddit data (live or cached)
    5. Data drift check
    6. Generate signals
    7. Close expired/capped positions
    8. Open new positions (if limits allow)
    9. Log everything
    10. Save portfolio state

Fallback handlers:
    API anomaly    → skip trading day
    Stale data     → halt signals
    Zero signals   → hold cash
    Model error    → skip day, alert
"""
import json
import logging
import sys
from datetime import datetime, date, timedelta, timezone

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
)
logger = logging.getLogger('daily_run')

sys.path.insert(0, '.')

from portfolio.signal_generator  import generate_signals, load_model
from portfolio.position_sizer    import compute_position_size, compute_slippage
from portfolio.regime_detector   import classify_regime
from portfolio.portfolio_engine  import (
    load_portfolio, save_portfolio, check_risk_limits, check_exits,
    Position, MAX_POSITIONS,
)
from portfolio.execution_logger  import log_signal
from portfolio.drift_monitor     import check_drift


def run(reddit_counts: dict, today: str = None) -> dict:
    """
    Main daily run function.

    Args:
        reddit_counts: {ticker: {post_count_1d, mention_growth_1d, mention_growth_7d}}
        today:         date string YYYY-MM-DD

    Returns:
        summary dict with actions taken
    """
    if today is None:
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    logger.info(f'daily_run_start date={today}')
    summary = {'date': today, 'actions': [], 'skipped': False, 'reason': None}

    # ── 1. Load state ──────────────────────────────────────────────────────
    state = load_portfolio()

    # ── 2. Risk limits ─────────────────────────────────────────────────────
    limits = check_risk_limits(state, today)
    if limits['weekly_loss_triggered']:
        logger.warning('HALT: weekly loss limit hit — pausing system')
        summary['skipped'] = True
        summary['reason']  = 'weekly_loss_limit'
        save_portfolio(state)
        return summary

    # ── 3. Regime ──────────────────────────────────────────────────────────
    try:
        regime = classify_regime(rolling_30d_ic=None)
        logger.info(f'regime label={regime.label} multiplier={regime.multiplier} '
                    f'reason={regime.reason}')
    except Exception as e:
        logger.error(f'regime_detection_failed error={e}')
        regime = None

    # ── 4. Data drift check ────────────────────────────────────────────────
    # Drift check uses MAX post count (not mean) because:
    # - Historical mean (53.2) = a busy ticker, not universe average
    # - Universe average drops near-zero when few tickers are found
    # - Max answers: "did at least one ticker have normal Reddit activity?"
    post_counts     = [v.get('post_count_1d', 0) for v in reddit_counts.values()]
    mention_growths = [
        v.get('mention_growth_7d', 0)
        for v in reddit_counts.values()
        if v.get('mention_growth_7d', 1.0) != 1.0  # exclude placeholders
    ]

    live_means = {
        'post_count_1d': max(post_counts) if post_counts else 0,
        # Only pass real values; drift_monitor skips this if history is immature
        'mention_growth_7d': (
            sum(mention_growths) / len(mention_growths)
            if mention_growths else 1.0
        ),
    }

    drift            = check_drift(live_means)
    skip_new_signals = drift['skip_day']
    if skip_new_signals:
        logger.warning(
            f'SKIP_DAY: Reddit API anomaly — skipping new signals, '
            f'but will still close expiring positions. alerts={drift["alerts"]}'
        )

    # ── 5. Close expiring positions (ALWAYS — regardless of drift) ─────────
    # Positions must be closed on schedule regardless of data quality issues.
    import yfinance as _yf
    early_prices: dict = {}
    actual_today = date.today().isoformat()
    is_backfill  = today < actual_today

    for pos in state.positions:
        try:
            if is_backfill:
                exit_date = date.fromisoformat(today)
                fetch_end = (exit_date + timedelta(days=3)).isoformat()
                mkt = _yf.download(
                    pos.ticker, start=today, end=fetch_end,
                    auto_adjust=True, progress=False,
                )
            else:
                mkt = _yf.download(
                    pos.ticker, period='5d',
                    auto_adjust=True, progress=False,
                )
            if isinstance(mkt.columns, pd.MultiIndex):
                mkt.columns = mkt.columns.get_level_values(0)
            if len(mkt) > 0:
                early_prices[pos.ticker] = float(mkt['Close'].dropna().iloc[-1])
                logger.info(
                    f'exit_price_fetched ticker={pos.ticker} '
                    f'price={early_prices[pos.ticker]:.4f}'
                )
            else:
                early_prices[pos.ticker] = pos.entry_price
                logger.warning(
                    f'exit_price_fallback ticker={pos.ticker} '
                    f'using entry_price={pos.entry_price:.4f}'
                )
        except Exception as e:
            early_prices[pos.ticker] = pos.entry_price
            logger.warning(
                f'exit_price_error ticker={pos.ticker} '
                f'error={e} using entry_price={pos.entry_price:.4f}'
            )

    to_close = check_exits(state, early_prices, today)
    for exit_info in to_close:
        pos     = exit_info['position']
        logger.info(
            f'closing_position ticker={pos.ticker} '
            f'reason={exit_info["exit_reason"]} '
            f'pnl={exit_info["pnl_pct"]:+.4f}'
        )
        proceeds = pos.n_shares * exit_info['exit_price']
        state.cash += proceeds
        state.positions = [p for p in state.positions if p.ticker != pos.ticker]
        closed = {
            'ticker':      pos.ticker,
            'entry_date':  pos.entry_date,
            'exit_date':   today,
            'entry_price': pos.entry_price,
            'exit_price':  exit_info['exit_price'],
            'n_shares':    pos.n_shares,
            'pnl_pct':     exit_info['pnl_pct'],
            'exit_reason': exit_info['exit_reason'],
        }
        state.closed_trades.append(closed)
        log_signal(
            ticker=pos.ticker, date=today,
            feature_vector=pos.feature_vector,
            regime_state=pos.regime_state,
            regime_multiplier=pos.regime_multiplier,
            predicted_return_5d=pos.predicted_return,
            atr_14=pos.atr_14,
            position_size_dollars=proceeds,
            slippage_applied=0.0,
            fill_price=exit_info['exit_price'],
            signal_timestamp=datetime.now(timezone.utc).isoformat(),
            action=f'CLOSE_{exit_info["exit_reason"].upper()}',
            notes=f'pnl={exit_info["pnl_pct"]*100:.1f}%',
        )
        summary['actions'].append(f'CLOSE {pos.ticker} ({exit_info["exit_reason"]})')

    # ── 6. Skip new signals if drift anomaly detected ──────────────────────
    if skip_new_signals:
        summary['skipped'] = True
        summary['reason']  = 'api_anomaly_new_signals_only'
        save_portfolio(state)
        return summary

    # ── 7. Load model ──────────────────────────────────────────────────────
    try:
        model = load_model()
    except FileNotFoundError as e:
        logger.error(f'MODEL_NOT_FOUND error={e}')
        summary['skipped'] = True
        summary['reason']  = 'model_not_found'
        save_portfolio(state)
        return summary
    except Exception as e:
        logger.error(f'MODEL_LOAD_FAILED error={e}')
        summary['skipped'] = True
        summary['reason']  = 'model_error'
        save_portfolio(state)
        return summary

    # ── 6. Generate signals ────────────────────────────────────────────────
    try:
        signals = generate_signals(reddit_counts, model, today)
    except Exception as e:
        logger.error(f'SIGNAL_GENERATION_FAILED error={e}')
        summary['skipped'] = True
        summary['reason']  = 'model_prediction_error'
        save_portfolio(state)
        return summary

    if not signals:
        logger.info('HOLD_CASH: no qualifying signals today')
        summary['actions'].append('hold_cash')
        summary['reason'] = 'zero_signals'
        save_portfolio(state)
        return summary

    # ── 8. Open new positions ──────────────────────────────────────────────
    # current_prices: seed from step-5 early prices, overlay with signal prices
    current_prices = dict(early_prices)
    current_prices.update({s.ticker: s.price for s in signals})

    if limits['can_open_new_trades'] and not limits['daily_loss_triggered']:
        regime_mult     = regime.multiplier if regime else 0.75
        portfolio_value = state.total_value(current_prices)

        for signal in signals:
            if state.n_open_positions() >= MAX_POSITIONS:
                break

            if state.is_ticker_on_cooldown(signal.ticker, today):
                logger.debug(f'ticker_on_cooldown ticker={signal.ticker}')
                continue

            if any(p.ticker == signal.ticker for p in state.positions):
                continue

            sizing = compute_position_size(
                portfolio_value=portfolio_value,
                price=signal.price,
                atr_14=signal.atr_14,
                regime_multiplier=regime_mult,
            )

            if sizing['n_shares'] == 0:
                continue

            slippage   = compute_slippage(
                price=signal.price,
                mention_growth_7d=signal.feature_vector.get('mention_growth_7d', 0),
            )
            fill_price = signal.price * (1 + slippage)

            # Approximate 5 trading days ≈ 7 calendar days
            stop_date = (date.fromisoformat(today) + timedelta(days=7)).isoformat()

            pos = Position(
                ticker=signal.ticker,
                entry_date=today,
                entry_price=round(fill_price, 4),
                n_shares=sizing['n_shares'],
                position_dollars=sizing['position_dollars'],
                stop_date=stop_date,
                predicted_return=signal.predicted_return,
                atr_14=signal.atr_14,
                slippage_applied=slippage,
                regime_state=regime.label if regime else 'neutral',
                regime_multiplier=regime_mult,
                feature_vector=signal.feature_vector,
            )

            cost = sizing['n_shares'] * fill_price
            state.cash -= cost
            state.positions.append(pos)
            state.ticker_last_trade[signal.ticker] = today

            log_signal(
                ticker=signal.ticker, date=today,
                feature_vector=signal.feature_vector,
                regime_state=pos.regime_state,
                regime_multiplier=regime_mult,
                predicted_return_5d=signal.predicted_return,
                atr_14=signal.atr_14,
                position_size_dollars=sizing['position_dollars'],
                slippage_applied=slippage,
                fill_price=fill_price,
                signal_timestamp=signal.signal_timestamp,
                action='OPEN',
            )

            logger.info(f'opened_position ticker={signal.ticker} '
                        f'predicted_return={signal.predicted_return:.4f} '
                        f'position_dollars={sizing["position_dollars"]}')
            summary['actions'].append(
                f'OPEN {signal.ticker} pred={signal.predicted_return:.3f}'
            )

    # ── 9. Save state ──────────────────────────────────────────────────────
    save_portfolio(state)

    # ── 10. Record daily performance snapshot ─────────────────────────────
    try:
        import yfinance as _yf
        current_prices_for_value = {}
        for pos in state.positions:
            try:
                mkt = _yf.download(pos.ticker, period='2d',
                                   auto_adjust=True, progress=False)
                if isinstance(mkt.columns, pd.MultiIndex):
                    mkt.columns = mkt.columns.get_level_values(0)
                current_prices_for_value[pos.ticker] = float(mkt['Close'].iloc[-1])
            except Exception:
                current_prices_for_value[pos.ticker] = pos.entry_price

        from portfolio.paper_trader import record_daily_snapshot
        portfolio_value = state.total_value(current_prices_for_value)
        record_daily_snapshot(
            portfolio_value=portfolio_value,
            starting_capital=10000.0,
            n_trades_today=sum(1 for a in summary['actions'] if 'OPEN' in a),
            actions=summary['actions'],
            date=today,
        )
    except Exception as e:
        logger.warning(f'performance_snapshot_failed error={e}')

    logger.info(f'daily_run_complete actions={len(summary["actions"])}')
    return summary


if __name__ == '__main__':
    test_reddit = {
        'TSLA': {'post_count_1d': 45, 'mention_growth_1d': 0.3, 'mention_growth_7d': 0.4},
        'PLTR': {'post_count_1d': 22, 'mention_growth_1d': 0.2, 'mention_growth_7d': 0.3},
        'COIN': {'post_count_1d': 18, 'mention_growth_1d': 0.15, 'mention_growth_7d': 0.25},
    }
    result = run(test_reddit)
    print(json.dumps(result, indent=2))
