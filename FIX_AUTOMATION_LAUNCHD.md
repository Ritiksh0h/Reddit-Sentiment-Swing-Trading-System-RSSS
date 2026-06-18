# Claude Code — Fix Automation + Three-Times Daily Runs
# Reddit Sentiment Swing Trading System (RSSS)
# GitHub: https://github.com/Ritikshah0h/Reddit-Sentiment-Swing-Trading-System-RSSS

---

## Context

System is on IST (UTC+5:30). Three problems identified from logs:

Problem 1: com.rsss.api plist has exit code 78 (config error) — API
           crashes on startup, dashboard unreachable after reboot.

Problem 2: com.rsss.dailyrun fires at wrong time — 03:00 IST (08:30 ET)
           when Reddit is empty. Drift monitor correctly rejects
           post_count=3 against historical mean=53.2 and skips the run.
           Should fire at 18:30 IST (09:00 ET) when Reddit is active.

Problem 3: Two new plists (1100, 1400) were not created or failed to load.
           Only 3 rsss jobs show in launchctl — need 5.

---

## Session Start

```bash
git pull origin main
source .venv/bin/activate

# See current state
launchctl list | grep rsss
ls ~/Library/LaunchAgents/com.rsss.*.plist
cat logs/api_error.log 2>/dev/null || echo "no api error log"
```

---

## Task 1 — Fix com.rsss.api plist (exit code 78)

Exit code 78 = plist configuration error. Most common causes:
- WorkingDirectory path has spaces and isn't quoted correctly in XML
- Python/uvicorn path is wrong
- Log directory doesn't exist yet

### Step 1a — Read the broken plist

```bash
cat ~/Library/LaunchAgents/com.rsss.api.plist
```

### Step 1b — Identify the error

```bash
# Unload and reload with verbose output
launchctl unload ~/Library/LaunchAgents/com.rsss.api.plist
launchctl load ~/Library/LaunchAgents/com.rsss.api.plist
launchctl list | grep rsss.api

# Check what uvicorn path actually exists
ls "/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot/.venv/bin/uvicorn"

# Check logs directory exists
ls "/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot/logs/"
```

### Step 1c — Rewrite the api plist correctly

Write this exact file to `~/Library/LaunchAgents/com.rsss.api.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.rsss.api</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot/.venv/bin/uvicorn</string>
        <string>api.main:app</string>
        <string>--host</string>
        <string>127.0.0.1</string>
        <string>--port</string>
        <string>8000</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot/logs/api.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot/logs/api_error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot/.venv/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
```

### Step 1d — Reload and verify

```bash
launchctl unload ~/Library/LaunchAgents/com.rsss.api.plist
launchctl load ~/Library/LaunchAgents/com.rsss.api.plist

# Wait 3 seconds for startup
sleep 3

# Check exit code (should be blank or 0, NOT 78)
launchctl list | grep rsss.api

# Confirm API responds
curl -s http://localhost:8000/status | python3 -m json.tool
```

Expected output:
```
{
    "date": "...",
    "ran_today": false,
    "system_ok": true or false,
    ...
}
```

If exit code is still 78 — read `logs/api_error.log` and fix the
specific error before continuing. Do NOT proceed to Task 2 until
the API is confirmed running.

---

## Task 2 — Fix dailyrun timing (wrong IST hour)

System is in IST (UTC+5:30). The current plist fires at the wrong time.

US market open = 09:30 ET = 19:00 IST
Pre-market peak Reddit activity = 09:00 ET = 18:30 IST

The current plist was written for ET and fires at 03:00 IST when
Reddit has essentially zero posts. The drift monitor correctly skips.

### Step 2a — Read current plist

```bash
cat ~/Library/LaunchAgents/com.rsss.dailyrun.plist
```

Note the current Hour and Minute values.

### Step 2b — Rewrite with correct IST times

Unload existing plist, then write the corrected version:

```bash
launchctl unload ~/Library/LaunchAgents/com.rsss.dailyrun.plist
```

Write this to `~/Library/LaunchAgents/com.rsss.dailyrun.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.rsss.dailyrun</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot/.venv/bin/python</string>
        <string>/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot/scripts/daily_run_live.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot</string>
    <key>StartCalendarInterval</key>
    <array>
        <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>30</integer></dict>
        <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>30</integer></dict>
        <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>30</integer></dict>
        <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>30</integer></dict>
        <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>30</integer></dict>
    </array>
    <key>StandardOutPath</key>
    <string>/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot/logs/launchd.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot/logs/launchd_error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot/.venv/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
```

This fires at 18:30 IST (09:00 ET) — peak Reddit activity window,
US market has just opened, WSB morning discussion is active.

```bash
launchctl load ~/Library/LaunchAgents/com.rsss.dailyrun.plist
launchctl list | grep rsss.dailyrun
```

---

## Task 3 — Add two new run times (21:00 IST and 23:30 IST)

Three runs per day capture:
```
18:30 IST = 09:00 ET  ← US market open (Task 2 above)
21:00 IST = 11:30 ET  ← US midday, lunch crowd Reddit activity
23:30 IST = 14:00 ET  ← pre-close, afternoon momentum check
```

### Step 3a — Create 21:00 IST plist (11:30 ET midday)

Write to `~/Library/LaunchAgents/com.rsss.dailyrun.1130.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.rsss.dailyrun.1130</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot/.venv/bin/python</string>
        <string>/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot/scripts/daily_run_live.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot</string>
    <key>StartCalendarInterval</key>
    <array>
        <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>21</integer><key>Minute</key><integer>0</integer></dict>
        <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>21</integer><key>Minute</key><integer>0</integer></dict>
        <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>21</integer><key>Minute</key><integer>0</integer></dict>
        <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>21</integer><key>Minute</key><integer>0</integer></dict>
        <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>21</integer><key>Minute</key><integer>0</integer></dict>
    </array>
    <key>StandardOutPath</key>
    <string>/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot/logs/launchd_1130.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot/logs/launchd_1130_error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot/.venv/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
```

### Step 3b — Create 23:30 IST plist (14:00 ET pre-close)

Write to `~/Library/LaunchAgents/com.rsss.dailyrun.1400.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.rsss.dailyrun.1400</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot/.venv/bin/python</string>
        <string>/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot/scripts/daily_run_live.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot</string>
    <key>StartCalendarInterval</key>
    <array>
        <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>23</integer><key>Minute</key><integer>30</integer></dict>
        <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>23</integer><key>Minute</key><integer>30</integer></dict>
        <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>23</integer><key>Minute</key><integer>30</integer></dict>
        <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>23</integer><key>Minute</key><integer>30</integer></dict>
        <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>23</integer><key>Minute</key><integer>30</integer></dict>
    </array>
    <key>StandardOutPath</key>
    <string>/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot/logs/launchd_1400.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot/logs/launchd_1400_error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot/.venv/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
```

### Step 3c — Load both new plists

```bash
launchctl load ~/Library/LaunchAgents/com.rsss.dailyrun.1130.plist
launchctl load ~/Library/LaunchAgents/com.rsss.dailyrun.1400.plist
```

---

## Task 4 — Fix icmonitor plist timing

The IC monitor fires at 09:00 ET Monday = 14:30 IST Monday.
Verify it's set correctly:

```bash
cat ~/Library/LaunchAgents/com.rsss.icmonitor.plist | grep -A10 StartCalendarInterval
```

If Hour is not 14 and Minute is not 30 and Weekday is not 1 (Monday),
rewrite it:

```bash
launchctl unload ~/Library/LaunchAgents/com.rsss.icmonitor.plist
```

Write to `~/Library/LaunchAgents/com.rsss.icmonitor.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.rsss.icmonitor</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot/.venv/bin/python</string>
        <string>/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot/scripts/monitor_live_ic.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Weekday</key><integer>1</integer>
        <key>Hour</key><integer>14</integer>
        <key>Minute</key><integer>30</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot/logs/ic_monitor_auto.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot/logs/ic_monitor_error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/Users/ritikshah/Downloads/desktop/data analytics/Sentiment-driven Trading Bot/.venv/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.rsss.icmonitor.plist
```

---

## Task 5 — Verify everything

```bash
# All 5 rsss jobs should show — api with a PID, others with dash
launchctl list | grep rsss
```

Expected:
```
PID     EXIT    LABEL
12345   0       com.rsss.api             ← PID present, exit 0
-       0       com.rsss.dailyrun        ← waiting for 18:30 IST
-       0       com.rsss.dailyrun.1130   ← waiting for 21:00 IST
-       0       com.rsss.dailyrun.1400   ← waiting for 23:30 IST
-       0       com.rsss.icmonitor       ← waiting for Monday 14:30 IST
```

If any job shows a non-zero exit code — read that job's error log
and fix before marking complete.

```bash
# Confirm API is alive and responding
curl -s http://localhost:8000/status | python3 -m json.tool

# Confirm dashboard loads
open http://localhost:8000/dashboard

# Manual test run to confirm no skip at this time of day
python scripts/daily_run_live.py --dry-run 2>&1 | tail -10
```

The dry-run should show:
```
posts_fetched total=300+ tickers_found=5+
Combined data: 35+ tickers
```

If post_count is still < 3 and drift monitor fires — check what
time IST it currently is. If it's before 18:00 IST Reddit will be
quiet. Wait until 18:30 IST and rerun.

---

## Task 6 — Add run schedule to CLAUDE.md

Update the Automation section in CLAUDE.md:

```
com.rsss.api             → always on, RunAtLoad=true, KeepAlive=true
com.rsss.dailyrun        → 18:30 IST Mon-Fri (09:00 ET)
com.rsss.dailyrun.1130   → 21:00 IST Mon-Fri (11:30 ET midday)
com.rsss.dailyrun.1400   → 23:30 IST Mon-Fri (14:00 ET pre-close)
com.rsss.icmonitor       → 14:30 IST Monday (09:00 ET)
```

---

## Task 7 — Push

```bash
bash push.sh "[ops] fix launchd plists — correct IST timing, three daily runs, api autostart"
```

---

## Hard Rules for This Session

- NEVER add --dry-run to any production plist
- ALWAYS include EnvironmentVariables PATH in every plist
  (launchd has a minimal PATH that won't find .venv binaries otherwise)
- ALWAYS use absolute paths in ProgramArguments — no relative paths
- NEVER change the daily_run_live.py script itself
- The drift monitor skip is CORRECT behaviour — do not lower the threshold
  The fix is the timing, not the threshold
- If API plist exit code is 78 after rewrite, check that
  logs/ directory exists before loading the plist

---

## Why the Drift Monitor Skip is Correct

DO NOT touch drift_monitor.py or its thresholds.

The skip at 03:00 IST is correct behaviour:
  - Reddit post_count at 03:00 IST = 3 posts (US sleeping)
  - Historical mean = 53.2 posts (calibrated at active hours)
  - 3 < 53.2 × 0.5 = 26.6 → correctly flagged as API anomaly
  - System correctly skips rather than trading on empty Reddit data

The fix is scheduling the run during US market hours (18:30+ IST)
when Reddit is actually active. The drift monitor is doing its job.

---

*Automation Fix — June 2026*
*IST timezone: 18:30/21:00/23:30 IST = 09:00/11:30/14:00 ET*
*5 launchd jobs total: api (always on) + 3 daily runs + icmonitor*
