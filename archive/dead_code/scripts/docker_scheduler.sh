#!/usr/bin/env bash
# Docker scheduler — runs daily_run_live.py at market open times (UTC).
# EDT = UTC-4. Market times in EDT → UTC equivalents:
#   09:00 EDT = 13:00 UTC
#   11:30 EDT = 15:30 UTC
#   14:00 EDT = 18:00 UTC
# Runs Mon-Fri only. Adjust RUN_TIMES if DST changes.

set -euo pipefail

RUN_TIMES=("13:00" "15:30" "18:00")
SCRIPT="scripts/daily_run_live.py"
PYTHON="python"
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

echo "[scheduler] started at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

while true; do
    NOW_UTC=$(date -u '+%H:%M')
    DOW=$(date -u '+%u')   # 1=Mon ... 7=Sun

    if [[ "$DOW" -le 5 ]]; then
        for TARGET in "${RUN_TIMES[@]}"; do
            if [[ "$NOW_UTC" == "$TARGET" ]]; then
                echo "[scheduler] $(date -u '+%Y-%m-%d %H:%M:%S UTC') — firing $SCRIPT"
                $PYTHON "$SCRIPT" >> "$LOG_DIR/daily_runs.log" 2>&1 || \
                    echo "[scheduler] WARNING: $SCRIPT exited with error $?"
                sleep 61   # skip duplicate fires within the same minute
                break
            fi
        done
    fi

    sleep 30
done
