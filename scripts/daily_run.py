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

from portfolio.signal_generator  import generate_signals, load_models
from portfolio.position_sizer    import compute_position_size, compute_slippage, compute_position
from portfolio.regime_detector   import classify_regime
from portfolio.portfolio_engine  import (
    load_portfolio, save_portfolio, check_risk_limits, check_exits,
    Position, MAX_POSITIONS,
    get_max_positions, heat_budget_allows, correlation_allows,
)
from portfolio.execution_logger  import log_signal
from portfolio.drift_monitor     import check_drift


def run(
    reddit_counts:   dict,
    today:           str  = None,
    news_data:       dict = None,
    stocktwits_data: dict = None,
) -> dict:
    """
    Main daily run function.

    Args:
        reddit_counts:   {ticker: {post_count_1d, mention_growth_1d, mention_growth_7d}}
        today:           date string YYYY-MM-DD
        news_data:       {ticker: {news_sentiment_1d, news_count_1d}} or None
        stocktwits_data: {ticker: {st_sentiment_1d, st_bull_pct}} or None

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
    drift            = check_drift(reddit_counts)
    skip_new_signals = drift['skip_day']
    if skip_new_signals:
        logger.warning(
            f'SKIP_DAY: Reddit API anomaly — skipping new signals, '
            f'but will still close expiring positions. alerts={drift["alerts"]}'
        )

    # Log per-ticker post counts so we can see who's near the density gate
    for ticker, data in sorted(reddit_counts.items(),
                                key=lambda x: x[1].get('post_count_1d', 0),
                                reverse=True)[:10]:
        count = data.get('post_count_1d', 0)
        gate  = 'PASS' if count >= 5 else 'FAIL'
        logger.info(f'  [{gate}] {ticker:<6} posts={count}')

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

    # ── 7. Load models ─────────────────────────────────────────────────────
    try:
        models = load_models()
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

    # ── 6b. StockTwits density boost ──────────────────────────────────────
    # Tickers with heavy StockTwits activity but low Reddit posts may still
    # have real crowd attention. st_count_1d is merged into reddit_counts by
    # daily_run_live.py. Boost effective post_count_1d by up to +5.
    for ticker in list(reddit_counts.keys()):
        st_count      = reddit_counts[ticker].get('st_count_1d', 0)
        st_equivalent = min(st_count // 20, 5)
        if st_equivalent > 0:
            reddit_counts[ticker]['post_count_1d'] += st_equivalent
            logger.debug(
                f'density_boost ticker={ticker} '
                f'st_count={st_count} boost=+{st_equivalent}'
            )

    # ── 6. Generate signals ────────────────────────────────────────────────
    try:
        signals = generate_signals(
            reddit_counts=reddit_counts,
            models=models,
            today=today,
            news_data=news_data,
            stocktwits_data=stocktwits_data,
        )
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

    # ── FIX 1: Log ALL qualifying signals to all_signals.jsonl ────────────
    # Includes NEUTRAL and BEARISH — every density-gate-passing ticker that
    # was scored by XGBoost. Required so append_live_features.py can build
    # features_live_v2.parquet regardless of whether trades fired.
    try:
        import json as _json
        from pathlib import Path as _Path
        _ALL_SIG_LOG = _Path('logs/all_signals.jsonl')
        _ALL_SIG_LOG.parent.mkdir(exist_ok=True)
        for sig in signals:
            fv = sig.feature_vector or {}
            _record = {
                'date':              today,
                'ticker':            sig.ticker,
                'action':            'SIGNAL',
                'signal':            sig.signal,
                'close_price':       sig.price,
                'predicted_return_5d': sig.predicted_5d,
                'predicted_1d':      sig.predicted_1d,
                'predicted_3d':      sig.predicted_3d,
                'confidence':        sig.confidence,
                'post_count_1d':     sig.post_count_1d,
                'news_count_1d':     sig.news_count_1d,
                'st_count_1d':       sig.st_count_1d,
                'feature_vector':    fv,
            }
            with open(_ALL_SIG_LOG, 'a') as _f:
                _f.write(_json.dumps(_record) + '\n')
        logger.info(f'all_signals_logged count={len(signals)} path={_ALL_SIG_LOG}')
    except Exception as _e:
        logger.warning(f'all_signals_log_failed (non-fatal): {_e}')

    # ── 8. Open new positions ──────────────────────────────────────────────
    # current_prices: seed from step-5 early prices, overlay with signal prices
    current_prices = dict(early_prices)
    current_prices.update({s.ticker: s.price for s in signals})

    if limits['can_open_new_trades'] and not limits['daily_loss_triggered']:
        regime_mult  = regime.multiplier if regime else 0.75
        regime_label = regime.label      if regime else 'NEUTRAL'
        portfolio_value = state.total_value(current_prices)

        # 1 trading day ≈ 3 cal days, 3TD ≈ 5, 5TD ≈ 7
        _HOLD_CAL = {1: 3, 3: 5, 5: 7}

        for _sig_rank, signal in enumerate(signals, start=1):
            # Long-only: BEARISH and NEUTRAL signals are logged but never opened
            if signal.signal != 'BULLISH':
                logger.info(
                    f'skip_non_bullish ticker={signal.ticker} '
                    f'signal={signal.signal} '
                    f'pred_5d={signal.predicted_5d:.4f} '
                    f'reason=long_only_system'
                )
                continue

            if state.n_open_positions() >= get_max_positions(regime_label):
                break

            if state.is_ticker_on_cooldown(signal.ticker, today):
                logger.debug(f'ticker_on_cooldown ticker={signal.ticker}')
                continue

            if any(p.ticker == signal.ticker for p in state.positions):
                continue

            sizing = compute_position(
                equity=portfolio_value,
                entry_price=signal.price,
                atr_14=signal.atr_14,
                confidence=signal.confidence,
                regime=regime_label,
                signal_rank=_sig_rank,
            )

            if sizing['n_shares'] == 0:
                continue

            current_tickers = [p.ticker for p in state.positions]
            if not heat_budget_allows(
                state.positions, sizing['risk_dollars'], portfolio_value, regime_label
            ):
                logger.info(f'heat_budget_blocked ticker={signal.ticker} '
                            f'risk_dollars={sizing["risk_dollars"]:.2f}')
                continue
            if not correlation_allows(signal.ticker, current_tickers):
                logger.info(f'correlation_blocked ticker={signal.ticker}')
                continue

            # Apply PCR size multiplier (CAUTION = 50% size; never blocks signal)
            pcr_mult = getattr(signal, 'pcr_size_multiplier', 1.0)
            if pcr_mult < 1.0:
                reduced_n = max(1, int(sizing['n_shares'] * pcr_mult))
                sizing = dict(sizing)
                sizing['n_shares']         = reduced_n
                sizing['position_dollars'] = reduced_n * signal.price
                logger.info(
                    f'pcr_size_reduction ticker={signal.ticker} '
                    f'pcr={getattr(signal, "pcr", None)} '
                    f'conf={getattr(signal, "pcr_confirmation", "?")} '
                    f'original_n={sizing["n_shares"]} reduced_n={reduced_n}'
                )

            slippage   = compute_slippage(
                price=signal.price,
                mention_growth_7d=signal.feature_vector.get('mention_growth_7d', 0),
            )
            fill_price = signal.price * (1 + slippage)

            cal_days  = _HOLD_CAL.get(signal.hold_days, 7)
            stop_date = (date.fromisoformat(today) + timedelta(days=cal_days)).isoformat()

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
                hold_days=signal.hold_days,
                horizon=signal.horizon,
                predicted_return_1d=signal.predicted_1d,
                predicted_return_3d=signal.predicted_3d,
                predicted_return_5d=signal.predicted_5d,
                pcr_confirmation=getattr(signal, 'pcr_confirmation', 'UNKNOWN'),
                stop_pct=sizing['stop_pct'],
                risk_dollars=sizing['risk_dollars'],
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
                predicted_return_5d=signal.predicted_5d,
                hold_days=signal.hold_days,
                horizon=signal.horizon,
                predicted_1d=signal.predicted_1d,
                predicted_3d=signal.predicted_3d,
                atr_14=signal.atr_14,
                position_size_dollars=sizing['position_dollars'],
                slippage_applied=slippage,
                fill_price=fill_price,
                signal_timestamp=signal.signal_timestamp,
                action='OPEN',
                signal=signal.signal,
                confidence=signal.confidence,
                news_count_1d=signal.news_count_1d,
                st_count_1d=signal.st_count_1d,
                pcr=getattr(signal, 'pcr', None),
                pcr_confirmation=getattr(signal, 'pcr_confirmation', 'UNKNOWN'),
                pcr_size_mult=getattr(signal, 'pcr_size_multiplier', 1.0),
                pcr_reason=getattr(signal, 'pcr_reason', ''),
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

    # FIX 4: expose all qualifying signals so save_run_to_db() can write
    # them to Supabase signals table. Includes BULLISH + NEUTRAL + BEARISH.
    summary['all_signals'] = signals
    logger.info(f'daily_run_complete actions={len(summary["actions"])}')
    return summary


if __name__ == '__main__':
    print('Run via: python scripts/daily_run_live.py')
    print('         python scripts/daily_run_live.py --dry-run')
    print('         python scripts/test_historical_run.py --days 30')
