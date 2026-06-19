"""
Data drift monitor.
Compares live feature distributions against historical training means.
Alerts if values fall outside expected ranges.

Recalibrated Jun 2026 — 20-ticker trade universe.
Real observed max: 13 posts. Mean set to 15.0.
SKIP_DAY logic: max_posts < 3 = Reddit API down.

Historical means (reference values, 20-ticker universe):
    post_count_1d:    15.0  (skip when max across all tickers < 3)
    mention_growth_7d: 0.232

Alert threshold: live mean < historical × 0.5 OR > historical × 2.0
SKIP_DAY: only when max post count < 3 (API failure, not quiet market).
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Reference values — post_count_1d skip uses max_posts < 3, not percentage
HISTORICAL_MEANS = {
    'post_count_1d':     15.0,   # observed max in 12 weeks of live trading: 13
    'mention_growth_7d':  0.232,
}

ALERT_LOW_MULTIPLIER  = 0.5
ALERT_HIGH_MULTIPLIER = 2.0


def _mention_history_is_mature(
    history_path: str = 'data/mention_history.json',
    min_days: int = 7,
) -> bool:
    """
    Returns True if mention history has accumulated at least min_days of data
    for any ticker. Until then, mention_growth_7d is a 1.0 placeholder and
    should not be compared against the historical mean (0.232).
    """
    from pathlib import Path
    import json
    from datetime import date, timedelta

    path = Path(history_path)
    if not path.exists():
        return False

    try:
        with open(path) as f:
            history = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    cutoff = (date.today() - timedelta(days=min_days)).isoformat()
    for ticker_hist in history.values():
        if any(k <= cutoff for k in ticker_hist.keys()):
            return True

    return False


def check_drift(reddit_counts: dict) -> dict:
    """
    Check live feature distributions for Reddit API anomalies.

    SKIP_DAY fires when max post count across ALL tickers < 3.
    A single ticker with 3+ posts proves the API is responding —
    a low-count day is a quiet market, not a failure.

    mention_growth_7d is checked against historical mean with a ±50%/200%
    band, but only once mention_history.json has 7+ days of real data.

    Args:
        reddit_counts: {ticker: {'post_count_1d': int, 'mention_growth_7d': float, ...}}

    Returns:
        dict with:
            clean:          bool — True if no anomalies detected
            alerts:         list of alert strings
            skip_day:       bool — True only when max post count < 3
            skipped_checks: list of features skipped with reason
    """
    alerts         = []
    skipped_checks = []

    # ── SKIP_DAY: max post count across all tickers ────────────────────────
    max_posts = max(
        (v.get('post_count_1d', 0) for v in reddit_counts.values()),
        default=0,
    )
    if max_posts < 3:
        alert_msg = (
            f'post_count_1d: max_posts={max_posts} across all tickers — '
            f'Reddit API likely down or no data collected'
        )
        logger.warning(f'drift_alert: {alert_msg}')
        return {
            'clean':          False,
            'alerts':         [alert_msg],
            'skip_day':       True,
            'skipped_checks': [],
        }

    # ── mention_growth_7d: percentage check (only when history is mature) ───
    if not _mention_history_is_mature():
        skipped_checks.append(
            'mention_growth_7d: skipped — history not yet mature '
            '(< 7 days accumulated). Placeholder value 1.0 is not comparable '
            'to historical mean 0.232.'
        )
        logger.info('drift_check_skip feature=mention_growth_7d '
                    'reason=history_immature_placeholder_active')
    else:
        real_growths = [
            v.get('mention_growth_7d', 1.0)
            for v in reddit_counts.values()
            if v.get('mention_growth_7d', 1.0) != 1.0  # exclude 1.0 placeholders
        ]
        if real_growths:
            live_growth = sum(real_growths) / len(real_growths)
            hist_mean   = HISTORICAL_MEANS['mention_growth_7d']
            low_thresh  = hist_mean * ALERT_LOW_MULTIPLIER
            high_thresh = hist_mean * ALERT_HIGH_MULTIPLIER

            if live_growth < low_thresh:
                alerts.append(
                    f'mention_growth_7d: live={live_growth:.3f} is below 50% of '
                    f'historical mean ({hist_mean:.3f}). Possible API undercount.'
                )
            elif live_growth > high_thresh:
                alerts.append(
                    f'mention_growth_7d: live={live_growth:.3f} is above 200% of '
                    f'historical mean ({hist_mean:.3f}). Possible data spike or API issue.'
                )

    if alerts:
        for alert in alerts:
            logger.warning(f'drift_alert: {alert}')
    else:
        logger.info(
            f'drift_check_clean max_posts={max_posts} '
            f'features_skipped={len(skipped_checks)}'
        )

    if skipped_checks:
        for msg in skipped_checks:
            logger.info(f'drift_check_skipped: {msg}')

    return {
        'clean':          len(alerts) == 0,
        'alerts':         alerts,
        'skip_day':       False,  # SKIP_DAY handled above with early return
        'skipped_checks': skipped_checks,
    }
