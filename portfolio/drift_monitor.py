"""
Data drift monitor.
Compares live feature distributions against historical training means.
Alerts if values fall outside expected ranges.

Historical means (from feature_stats.json, training data 2019-2023):
    post_count_1d:    53.2
    mention_growth_7d: 0.232

Alert threshold: live value < mean × 0.5 OR > mean × 2.0
Action on alert: log warning. Skip day if post_count_1d anomaly detected.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# From phase3_locked_architecture.json data_drift_monitoring.historical_means
HISTORICAL_MEANS = {
    'post_count_1d':    53.2,
    'mention_growth_7d': 0.232,
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


def check_drift(live_values: dict) -> dict:
    """
    Check live feature values against historical training means.

    Skips mention_growth_7d until data/mention_history.json has accumulated
    7+ days of real data — avoids false positives from the 1.0 placeholder
    used during the first week of paper trading.

    Args:
        live_values: dict of feature_name → observed value today

    Returns:
        dict with:
            clean:          bool — True if no anomalies detected
            alerts:         list of alert strings
            skip_day:       bool — True if post_count_1d anomaly detected
            skipped_checks: list of features skipped with reason
    """
    alerts         = []
    skipped_checks = []

    features_to_check = dict(HISTORICAL_MEANS)

    if not _mention_history_is_mature():
        features_to_check.pop('mention_growth_7d', None)
        skipped_checks.append(
            'mention_growth_7d: skipped — history not yet mature '
            '(< 7 days accumulated). Placeholder value 1.0 is not comparable '
            'to historical mean 0.232.'
        )
        logger.info('drift_check_skip feature=mention_growth_7d '
                    'reason=history_immature_placeholder_active')

    for feature, hist_mean in features_to_check.items():
        live = live_values.get(feature)
        if live is None:
            alerts.append(f'{feature}: missing from live data')
            continue

        if hist_mean < 0:
            low_thresh  = hist_mean * ALERT_HIGH_MULTIPLIER
            high_thresh = hist_mean * ALERT_LOW_MULTIPLIER
        else:
            low_thresh  = hist_mean * ALERT_LOW_MULTIPLIER
            high_thresh = hist_mean * ALERT_HIGH_MULTIPLIER

        if live < low_thresh:
            alerts.append(
                f'{feature}: live={live:.3f} is below 50% of historical '
                f'mean ({hist_mean:.3f}). Possible API undercount.'
            )
        elif live > high_thresh:
            alerts.append(
                f'{feature}: live={live:.3f} is above 200% of historical '
                f'mean ({hist_mean:.3f}). Possible data spike or API issue.'
            )

    # skip_day fires only when post_count_1d is BELOW threshold (API undercount).
    # An above-threshold spike means Reddit is unusually active — not an API failure.
    # mention_growth anomalies alone never skip the day.
    skip_day = any('post_count_1d' in a and 'below' in a for a in alerts)

    if alerts:
        for alert in alerts:
            logger.warning(f'drift_alert: {alert}')
    else:
        logger.info(f'drift_check_clean features_checked={len(features_to_check)} '
                    f'features_skipped={len(skipped_checks)}')

    if skipped_checks:
        for msg in skipped_checks:
            logger.info(f'drift_check_skipped: {msg}')

    return {
        'clean':          len(alerts) == 0,
        'alerts':         alerts,
        'skip_day':       skip_day,
        'skipped_checks': skipped_checks,
    }
