#!/bin/bash
echo "═══════════════════════════════════════════"
echo "  RSSS System Health Check"
echo "═══════════════════════════════════════════"

echo ""
echo "── 1. Launchd jobs (scheduled + running) ──"
launchctl list | grep -i rsss

echo ""
echo "── 2. API server responding? ──"
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:8000/status || echo "API NOT RESPONDING"

echo ""
echo "── 3. Last 3 lines of each launchd log ──"
for f in logs/launchd.log logs/launchd_1130.log logs/launchd_1400.log logs/api.log logs/ic_monitor_auto.log; do
  echo "--- $f ---"
  if [ -f "$f" ]; then tail -3 "$f"; else echo "MISSING"; fi
done

echo ""
echo "── 4. Any errors in the last 24h? ──"
for f in logs/launchd_error.log logs/launchd_1130_error.log logs/launchd_1400_error.log logs/api_error.log; do
  if [ -f "$f" ] && [ -s "$f" ]; then
    echo "--- $f (has content) ---"
    tail -5 "$f"
  fi
done

echo ""
echo "── 5. Model files present and dated ──"
ls -la models/model_*_v2.json models/training_metadata_v2.json 2>&1

echo ""
echo "── 6. Today's date vs last pipeline run ──"
echo "Today: $(date '+%Y-%m-%d')"
echo "Last record in paper_trades.jsonl:"
tail -1 logs/paper_trades.jsonl 2>/dev/null | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(' date:', d.get('date'), '| action:', d.get('action'), '| ticker:', d.get('ticker'))" 2>/dev/null || echo "  could not parse"

echo ""
echo "── 7. Portfolio state ──"
python3 -c "
import json
p = json.load(open('data/live/paper_portfolio.json'))
print(' cash:', p.get('cash'))
print(' open positions:', len(p.get('positions', [])))
print(' closed trades:', len(p.get('closed_trades', [])))
"

echo ""
echo "── 8. Live IC monitor — latest reading ──"
tail -1 logs/ic_monitor.jsonl 2>/dev/null | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(' date:', d.get('date'), '| IC:', d.get('live_ic'), '| gate:', d.get('gate'))" 2>/dev/null || echo "  no readings yet"

echo ""
echo "── 9. Tests still passing? ──"
python3 -m pytest tests/ -q --tb=line 2>&1 | tail -5

echo ""
echo "═══════════════════════════════════════════"
echo "  Done"
echo "═══════════════════════════════════════════"