# Claude Code — Docker + VPS Deployment
# Reddit Sentiment Swing Trading System (RSSS)
# Goal: always-on public URL, Mac-independent

---

## Overview

Deploy RSSS to a DigitalOcean VPS using Docker.
After this: system runs 24/7, dashboard accessible from any device,
Mac can sleep/restart without affecting anything.

Cost: $6/month (DigitalOcean Basic Droplet)
Time: 30-45 minutes

---

## Part A — Build Docker Setup (local, on Mac)

Do this first. Get Docker working locally before touching the VPS.

### Step A1 — Read the project structure before writing anything

```bash
ls -la
cat requirements.txt | head -20
cat .env | grep -v "KEY\|TOKEN\|SECRET\|PASSWORD" | head -5
python --version
```

### Step A2 — Create .dockerignore

Create `.dockerignore` in project root:

```
.git
.venv
__pycache__
*.pyc
*.pyo
.DS_Store
archive/
logs/
data/raw/
data/features/
data/processed/
data/backfill_backup/
models/registry/*.pkl
*.parquet
*.ipynb
.env
ngrok
```

Note: .env is excluded from image (mounted as volume at runtime)
Note: large data files excluded (mounted as volumes)
Note: model pkl files excluded (mounted as volume)

### Step A3 — Create Dockerfile

Create `Dockerfile` in project root:

```dockerfile
FROM python:3.13-slim

# System dependencies for XGBoost, yfinance, transformers
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code (not data/models — those are volumes)
COPY api/ api/
COPY config/ config/
COPY data/*.py data/
COPY portfolio/ portfolio/
COPY scripts/ scripts/
COPY experiments/ experiments/
COPY pipeline/ pipeline/
COPY utils/ utils/
COPY dashboard/ dashboard/
COPY tests/ tests/

# Environment variables for XGBoost stability
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Create required directories
RUN mkdir -p logs data/live data/raw data/features \
    data/processed models/registry

EXPOSE 8000
```

### Step A4 — Create docker-compose.yml

Create `docker-compose.yml` in project root:

```yaml
version: '3.8'

services:
  api:
    build: .
    container_name: rsss-api
    command: python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
      - ./experiments:/app/experiments
      - ./.env:/app/.env:ro
    env_file: .env
    environment:
      - OMP_NUM_THREADS=1
      - OPENBLAS_NUM_THREADS=1
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/status"]
      interval: 30s
      timeout: 10s
      retries: 3

  scheduler:
    build: .
    container_name: rsss-scheduler
    command: /app/scripts/docker_scheduler.sh
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
      - ./experiments:/app/experiments
      - ./.env:/app/.env:ro
    env_file: .env
    environment:
      - OMP_NUM_THREADS=1
      - OPENBLAS_NUM_THREADS=1
    restart: always
    depends_on:
      api:
        condition: service_healthy
```

### Step A5 — Create scripts/docker_scheduler.sh

This replaces launchd. Runs in UTC (VPS standard).

```bash
#!/bin/bash
# RSSS Scheduler — runs inside Docker container
# All times in UTC (VPS standard timezone)
#
# Schedule:
#   13:00 UTC = 09:00 ET (market open sweep)
#   15:30 UTC = 11:30 ET (midday)
#   18:00 UTC = 14:00 ET (pre-close)
#   13:30 UTC Monday = 09:30 ET (IC monitor)
#   18:15 UTC = 14:15 ET (AV news backfill)

set -e

echo "$(date -u) RSSS Scheduler starting..."

# Load environment
if [ -f /app/.env ]; then
    export $(grep -v '^#' /app/.env | xargs)
fi

cd /app

run_daily() {
    echo "$(date -u) Starting daily run..."
    python scripts/daily_run_live.py >> logs/scheduler.log 2>&1
    echo "$(date -u) Daily run complete"
}

run_ic_monitor() {
    echo "$(date -u) Starting IC monitor..."
    python scripts/monitor_live_ic.py >> logs/scheduler.log 2>&1
    echo "$(date -u) IC monitor complete"
}

run_av_backfill() {
    echo "$(date -u) Starting AV news backfill..."
    python scripts/collect_av_news.py \
        --backfill --start 2023-01-01 --end 2025-12-31 \
        >> logs/scheduler.log 2>&1
    echo "$(date -u) AV backfill complete"
}

while true; do
    HOUR=$(date -u +%H)
    MIN=$(date -u +%M)
    DOW=$(date -u +%u)  # 1=Monday, 5=Friday

    # Weekdays only (Mon-Fri)
    if [ "$DOW" -ge 1 ] && [ "$DOW" -le 5 ]; then

        # 09:00 ET = 13:00 UTC
        if [ "$HOUR" = "13" ] && [ "$MIN" = "00" ]; then
            run_daily
            sleep 70
        fi

        # 11:30 ET = 15:30 UTC
        if [ "$HOUR" = "15" ] && [ "$MIN" = "30" ]; then
            run_daily
            sleep 70
        fi

        # 14:00 ET = 18:00 UTC
        if [ "$HOUR" = "18" ] && [ "$MIN" = "00" ]; then
            run_daily
            sleep 70
        fi

        # AV backfill 14:15 ET = 18:15 UTC (after market run)
        if [ "$HOUR" = "18" ] && [ "$MIN" = "15" ]; then
            run_av_backfill
            sleep 70
        fi

        # IC monitor Monday 09:30 ET = 13:30 UTC
        if [ "$DOW" = "1" ] && [ "$HOUR" = "13" ] && [ "$MIN" = "30" ]; then
            run_ic_monitor
            sleep 70
        fi
    fi

    sleep 30
done
```

Make it executable:
```bash
chmod +x scripts/docker_scheduler.sh
```

### Step A6 — Fix dashboard API base URL

The dashboard currently fetches from `http://localhost:8000` hardcoded.
On VPS this will be wrong — it needs to use the server's own URL.

In `dashboard/index.html`, find the API base URL:
```bash
grep -n "localhost:8000\|apiBase\|API_BASE" dashboard/index.html | head -5
```

Replace the hardcoded localhost with dynamic origin:
```javascript
// Find this pattern:
const API_BASE = 'http://localhost:8000'
// OR:
apiBase: 'http://localhost:8000'

// Replace with:
const API_BASE = window.location.origin
// OR:
apiBase: window.location.origin
```

This makes the dashboard work on localhost AND on any VPS domain.

### Step A7 — Test Docker build locally

```bash
# Build the image
docker build -t rsss .

# Test API container only
docker run --rm -p 8001:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/experiments:/app/experiments \
  -v $(pwd)/.env:/app/.env:ro \
  --env OMP_NUM_THREADS=1 \
  rsss python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &

sleep 8
curl http://localhost:8001/status | python3 -m json.tool
curl http://localhost:8001/portfolio | python3 -m json.tool | head -5
docker stop $(docker ps -q --filter ancestor=rsss)
```

Expected: status returns JSON, portfolio shows real equity.
If it fails: check docker logs for import errors.

### Step A8 — Push all Docker files

```bash
git add Dockerfile docker-compose.yml scripts/docker_scheduler.sh \
    .dockerignore dashboard/index.html
bash push.sh "[deploy] Docker setup — Dockerfile, compose, scheduler, dashboard URL fix"
```

---

## Part B — Create DigitalOcean VPS

Do this after Docker builds successfully locally.

### Step B1 — Create Droplet

1. Go to https://cloud.digitalocean.com
2. Create → Droplets
3. Choose:
   - Region: New York (closest to US markets)
   - Image: Ubuntu 24.04 LTS
   - Size: Basic → Regular → $6/month (1GB RAM, 1 CPU, 25GB SSD)
   - Authentication: SSH Key (add your Mac's public key)
     Run on Mac: `cat ~/.ssh/id_rsa.pub`
     If no key exists: `ssh-keygen -t rsa -b 4096`
4. Hostname: rsss-server
5. Click Create Droplet
6. Note the IP address (e.g., 143.198.xx.xx)

### Step B2 — Connect to VPS

```bash
# From your Mac terminal
ssh root@YOUR_DROPLET_IP
```

### Step B3 — Install Docker on VPS

Run these commands on the VPS:

```bash
# Update system
apt-get update && apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh

# Install Docker Compose
apt-get install -y docker-compose-plugin

# Verify
docker --version
docker compose version

# Install git
apt-get install -y git
```

### Step B4 — Clone repo and configure on VPS

```bash
# Clone the repo
git clone https://github.com/Ritiksh0h/Reddit-Sentiment-Swing-Trading-System-RSSS.git
cd Reddit-Sentiment-Swing-Trading-System-RSSS

# Create .env file with your secrets
nano .env
```

Add to .env (same content as your local .env):
```
HF_TOKEN=hf_your_token
ALPHAVANTAGE_API_KEY=your_key
```

Save and exit: Ctrl+X, Y, Enter

### Step B5 — Copy data files to VPS

From your Mac (new terminal, not the SSH session):

```bash
VPS_IP=YOUR_DROPLET_IP
PROJECT=/root/Reddit-Sentiment-Swing-Trading-System-RSSS

# Copy model files (required)
scp -r models/registry/*.pkl root@$VPS_IP:$PROJECT/models/registry/

# Copy feature store (required for signal generation)
scp data/features/features_full.parquet \
    root@$VPS_IP:$PROJECT/data/features/

scp data/features/features_complete.parquet \
    root@$VPS_IP:$PROJECT/data/features/

# Copy live portfolio state
scp data/live/paper_portfolio.json \
    root@$VPS_IP:$PROJECT/data/live/

scp data/live/paper_performance.jsonl \
    root@$VPS_IP:$PROJECT/data/live/

# Copy locked architecture
scp experiments/phase3_locked_architecture.json \
    root@$VPS_IP:$PROJECT/experiments/

# Copy backfill progress
scp data/processed/av_backfill_progress.json \
    root@$VPS_IP:$PROJECT/data/processed/ 2>/dev/null || true

# Copy source validation results
scp experiments/source_validation/results.json \
    root@$VPS_IP:$PROJECT/experiments/source_validation/

# Copy tickers config
scp config/tickers.txt root@$VPS_IP:$PROJECT/config/
scp config/false_positive_list.txt root@$VPS_IP:$PROJECT/config/
```

### Step B6 — Start the system on VPS

Back in the SSH session:

```bash
cd /root/Reddit-Sentiment-Swing-Trading-System-RSSS

# Create required directories
mkdir -p logs data/live data/raw data/features data/processed \
    models/registry experiments/source_validation

# Start everything
docker compose up -d

# Check it's running
docker compose ps
docker compose logs api --tail=20
```

Expected output:
```
NAME            STATUS
rsss-api        running (healthy)
rsss-scheduler  running
```

### Step B7 — Verify the live URL

```bash
# Get your VPS IP
curl http://YOUR_DROPLET_IP:8000/status
```

Open in browser:
```
http://YOUR_DROPLET_IP:8000/dashboard
```

---

## Part C — Add Domain (optional, makes URL prettier)

Skip this if you're happy with the IP address URL.

### Option 1 — Free subdomain via DigitalOcean

In DigitalOcean dashboard:
- Networking → Domains
- Add your domain if you have one
- Create A record pointing to droplet IP

### Option 2 — Free domain via Freenom

Get a free .tk or .ml domain at freenom.com
Point it to your VPS IP.

### Option 3 — Just use the IP

`http://143.198.xx.xx:8000/dashboard` works fine.
Share this URL — it's permanent and always works.

---

## Part D — Stop Local launchd (Mac no longer needs to run the system)

After VPS is confirmed working, disable local automation:

```bash
# Unload all launchd jobs (keeps plists for backup)
launchctl unload ~/Library/LaunchAgents/com.rsss.dailyrun.plist
launchctl unload ~/Library/LaunchAgents/com.rsss.dailyrun.1130.plist
launchctl unload ~/Library/LaunchAgents/com.rsss.dailyrun.1400.plist
launchctl unload ~/Library/LaunchAgents/com.rsss.icmonitor.plist
launchctl unload ~/Library/LaunchAgents/com.rsss.api.plist
launchctl unload ~/Library/LaunchAgents/com.rsss.newscollect.plist

# Verify all stopped
launchctl list | grep rsss
# Expected: no output
```

The system now runs entirely on the VPS. Mac is free.

---

## Part E — Ongoing Management

### Check system status (from anywhere):
```bash
curl http://YOUR_VPS_IP:8000/status
```

### View logs (SSH into VPS):
```bash
ssh root@YOUR_VPS_IP
cd /root/Reddit-Sentiment-Swing-Trading-System-RSSS
docker compose logs scheduler --tail=30
tail -20 logs/scheduler.log
```

### Update code after git push:
```bash
ssh root@YOUR_VPS_IP
cd /root/Reddit-Sentiment-Swing-Trading-System-RSSS
git pull origin main
docker compose restart
```

### Check containers:
```bash
docker compose ps
docker stats --no-stream
```

---

## Hard Rules

- NEVER put secrets in Dockerfile or docker-compose.yml
- ALWAYS use .env file for HF_TOKEN and ALPHAVANTAGE_API_KEY
- NEVER commit .env to git
- The .dockerignore must exclude .env (image has no secrets)
- models/registry/*.pkl are NOT in the image — they are in volumes
- If VPS runs out of memory (1GB): stop scheduler, run manually
- ALWAYS test Docker build locally before deploying to VPS
- The scheduler uses UTC — never IST or EDT in docker_scheduler.sh

---

## Expected Final State

```
VPS (always running):
  rsss-api:       http://YOUR_IP:8000/dashboard  ← public URL
  rsss-scheduler: 3x daily runs in UTC
                  AV backfill daily
                  IC monitor weekly

Mac (optional):
  No launchd jobs running
  Dashboard: http://localhost:8000/dashboard still works locally
  Development only

Logs (on VPS):
  logs/scheduler.log   ← all scheduled run output
  logs/daily_runs.log  ← signal generation details
  logs/paper_trades.jsonl ← NEVER DELETE
```

---

*Docker + VPS Deployment — June 2026*
*DigitalOcean $6/month | Ubuntu 24.04 | Docker Compose*
*Always-on, Mac-independent, public dashboard URL*
