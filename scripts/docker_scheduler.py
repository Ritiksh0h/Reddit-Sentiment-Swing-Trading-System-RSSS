"""
Runs daily_run_live.py on schedule inside Railway container.
Only needed if using Railway for scheduling instead of GitHub Actions.

Schedule fires at 08:30 container-local time (UTC in Railway).
run_daily() guards weekends and logs ET timestamp for traceability.
"""

import logging
import subprocess
import sys
import time
from datetime import datetime

import pytz
import schedule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("scheduler")

ET = pytz.timezone("America/New_York")


def run_daily() -> None:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        logger.info("Weekend — skipping")
        return
    logger.info(f"Running daily pipeline: {now}")
    result = subprocess.run(
        [sys.executable, "scripts/daily_run_live.py"],
        capture_output=False,
    )
    if result.returncode != 0:
        logger.error(f"Pipeline failed with exit code {result.returncode}")


schedule.every().day.at("08:30").do(run_daily)

# Fire immediately if started during market hours on a weekday
_now = datetime.now(ET)
if _now.weekday() < 5 and 8 <= _now.hour <= 16:
    run_daily()

while True:
    schedule.run_pending()
    time.sleep(60)
