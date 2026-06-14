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


def check_drift(live_values: dict) -> dict:
    """
    Check live feature values against historical means.

    Args:
        live_values: dict of feature_name → observed mean across all tickers today

    Returns:
        dict with:
            clean:    bool — True if no anomalies detected
            alerts:   list of alert strings
            skip_day: bool — True if post_count_1d anomaly (primary signal)
    """
    alerts = []

    for feature, hist_mean in HISTORICAL_MEANS.items():
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
                f'{feature}: live={live:.3f} below 50% of historical mean '
                f'({hist_mean:.3f}). Possible API undercount.'
            )
        elif live > high_thresh:
            alerts.append(
                f'{feature}: live={live:.3f} above 200% of historical mean '
                f'({hist_mean:.3f}). Possible data spike or API issue.'
            )

    skip_day = any('post_count_1d' in a for a in alerts)

    if alerts:
        for alert in alerts:
            logger.warning(f'drift_alert: {alert}')
    else:
        logger.info(f'drift_check_clean features_checked={len(HISTORICAL_MEANS)}')

    return {
        'clean':    len(alerts) == 0,
        'alerts':   alerts,
        'skip_day': skip_day,
    }
