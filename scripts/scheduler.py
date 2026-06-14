"""
Simple daily scheduler.
Runs daily_run_live.py at 08:30 ET every weekday.
Keep this running in a terminal or tmux session.

Usage:
    python scripts/scheduler.py

Alternative: use crontab -e to add:
    30 13 * * 1-5 cd "path/to/project" && .venv/bin/python scripts/daily_run_live.py >> logs/cron.log 2>&1
"""
import subprocess
import sys
import logging
import time
from datetime import datetime

import pytz

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/scheduler.log'),
    ]
)
logger = logging.getLogger('scheduler')

ET         = pytz.timezone('America/New_York')
RUN_HOUR   = 8
RUN_MINUTE = 30


def is_weekday() -> bool:
    return datetime.now(ET).weekday() < 5  # Mon=0 … Fri=4


def run_daily_pipeline():
    logger.info('Launching daily_run_live.py')
    result = subprocess.run(
        [sys.executable, 'scripts/daily_run_live.py'],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        logger.info('Daily run completed successfully')
    else:
        logger.error(f'Daily run failed:\n{result.stderr[-1000:]}')


def main():
    from pathlib import Path
    Path('logs').mkdir(exist_ok=True)

    logger.info(f'RSSS Scheduler started — will run at {RUN_HOUR:02d}:{RUN_MINUTE:02d} ET on weekdays')
    ran_today = False

    while True:
        now_et = datetime.now(ET)

        # Reset daily flag at midnight
        if now_et.hour == 0 and now_et.minute < 1:
            ran_today = False

        # Run at scheduled time on weekdays
        if (now_et.hour == RUN_HOUR and
                now_et.minute == RUN_MINUTE and
                not ran_today and
                is_weekday()):
            run_daily_pipeline()
            ran_today = True
            logger.info('Sleeping until tomorrow...')

        time.sleep(30)  # check every 30 seconds


if __name__ == '__main__':
    main()
