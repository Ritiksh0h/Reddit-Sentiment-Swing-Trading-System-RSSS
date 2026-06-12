#!/usr/bin/env python3
"""
Module: run_pipeline.py
Purpose: Orchestrate the full Phase 1 research pipeline.
         Runs Scripts 01–06 in order with go/no-go gates at each step.
         Prints a final summary table.

Phase: 1 — Research Pipeline
Usage:
    python run_pipeline.py              # full run (skip 01 if features exist)
    python run_pipeline.py --debug      # fast debug run (small data)
    python run_pipeline.py --force-01   # always re-run feature builder
    python run_pipeline.py --start 02   # skip to a specific script
    python run_pipeline.py --only 03    # run only one script
Last modified: 2026-06-11
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import FEATURES_PARQUET, REPORTS_DIR, MODELS_DIR
from utils.logger import get_logger

log = get_logger(__name__)

PIPELINE_ROOT = Path(__file__).parent
PIPELINE_DIR = PIPELINE_ROOT / "pipeline"

SCRIPTS = [
    ("01", "01_feature_builder.py",  "Feature Builder"),
    ("02", "02_run_baselines.py",     "Baseline Strategies"),
    ("03", "03_train_models.py",      "Train XGBoost Models"),
    ("04", "04_run_backtests.py",     "ML Backtests"),
    ("05", "05_validate_alpha.py",    "Statistical Validation"),
    ("06", "06_feature_importance.py","Feature Importance (SHAP)"),
]


# ---------------------------------------------------------------------------
# Gate checks between steps
# ---------------------------------------------------------------------------

def check_gate_after_02() -> tuple[bool, str]:
    """After baselines: warn if Strategy C win_rate < 50%.

    Per spec: 'report finding before Script 03' — this is a SOFT WARNING,
    not a hard stop. Pipeline continues; human reviews the baseline HTML.
    """
    report_path = REPORTS_DIR / "baseline_report.json"
    if not report_path.exists():
        return True, "baseline_report.json not found — skipping gate check"
    with open(report_path) as f:
        report = json.load(f)
    strategies = report.get("strategies", [])
    strat_c = next((s for s in strategies if s["strategy"] == "attention_volume"), None)
    if strat_c and strat_c.get("win_rate", 1.0) < 0.50:
        win = strat_c["win_rate"]
        log.warning(
            "strategy_c_weak",
            win_rate=win,
            msg=f"Strategy C win_rate={win:.1%} < 50% — signal weaker than expected",
        )
        # Soft warning — do NOT stop, continue to ML (spec says 'report finding')
        return True, f"⚠ Strategy C win_rate={win:.1%} < 50% (soft warning — continuing)"
    return True, "OK"


def check_gate_after_03() -> tuple[bool, str]:
    """After model training: enforce Reddit-adds-value gate (hard stop per spec).

    Hard gate: Reddit must add >= 0.005 IC improvement over market-only.
    Soft warning: combined IC < min threshold (0.03) — log but don't stop.
    """
    report_path = REPORTS_DIR / "model_comparison.json"
    if not report_path.exists():
        return True, "model_comparison.json not found — skipping gate check"
    with open(report_path) as f:
        report = json.load(f)
    comparison = report.get("comparison", {})
    verdict = comparison.get("verdict", "")

    # Hard gate: Reddit not adding value → stop per spec
    if verdict == "REDDIT_NOT_ADDITIVE":
        ic_diff = comparison.get("ic_diff_combined_vs_market", 0)
        msg = (f"GATE FAILED: Reddit not additive (IC diff={ic_diff:+.4f}). "
               "Per spec: do not proceed to backtests. Reassess Reddit features.")
        return False, msg

    # Soft warning: IC below minimum — continue but flag it
    if not comparison.get("combined_above_min_threshold", True):
        ic_combined = comparison.get("ic_combined", 0)
        log.warning(
            "ic_below_threshold_soft",
            ic_combined=ic_combined,
            msg="IC below 0.03 minimum — results may be unreliable on full dataset",
        )
        # Not a hard stop; the full-dataset run may produce better IC

    return True, f"Reddit adds value (verdict={verdict})"


def check_gate_after_05() -> tuple[bool, str]:
    """After validation: enforce permutation p-value gate."""
    report_path = REPORTS_DIR / "alpha_validation.json"
    if not report_path.exists():
        return True, "alpha_validation.json not found — skipping gate check"
    with open(report_path) as f:
        report = json.load(f)
    perm = report.get("permutation_test", {})
    if perm.get("verdict") == "SPURIOUS":
        p = perm.get("p_value", 1.0)
        msg = (f"GATE FAILED: Permutation p-value={p:.4f} >= 0.05. "
               "Signal may be spurious. Do not proceed to live trading.")
        return False, msg
    return True, "OK"


POST_GATES = {
    "02": check_gate_after_02,
    "03": check_gate_after_03,
    "05": check_gate_after_05,
}


# ---------------------------------------------------------------------------
# Script runner
# ---------------------------------------------------------------------------

def run_script(script_path: Path, extra_args: list[str]) -> tuple[int, float]:
    """Run a pipeline script as a subprocess. Returns (returncode, elapsed_s)."""
    cmd = [sys.executable, str(script_path)] + extra_args
    log.info("script_start", script=script_path.name, cmd=" ".join(cmd))
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.monotonic() - t0
    return result.returncode, elapsed


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

def print_final_summary(step_results: list[dict]) -> None:
    """Print a rich summary after the full pipeline."""
    print("\n" + "=" * 60)
    print("  RSSS Phase 1 Pipeline — Final Summary")
    print("=" * 60)

    # Step table
    for r in step_results:
        status = "✅" if r["success"] else "❌"
        elapsed = f"{r['elapsed_s']:.1f}s"
        note = r.get("gate_note", "")
        print(f"  {status} {r['num']}. {r['name']:30s}  {elapsed:>8}")
        if note:
            print(f"       └─ {note}")
    print()

    # Key metrics
    try:
        comp = json.loads((REPORTS_DIR / "model_comparison.json").read_text())
        ic_market = comp["comparison"]["ic_market"]
        ic_combined = comp["comparison"]["ic_combined"]
        verdict = comp["comparison"]["verdict"]
        print(f"  Model IC (market):    {ic_market:+.4f}")
        print(f"  Model IC (combined):  {ic_combined:+.4f}  → {verdict}")
    except Exception:
        pass

    try:
        bt = json.loads((REPORTS_DIR / "backtest_report.json").read_text())
        print(f"  Backtest return:      {bt['total_return']:+.1%}")
        print(f"  Sharpe ratio:         {bt['sharpe_ratio']:.2f}")
        print(f"  Beats SPY:            {'YES' if bt['beats_spy'] else 'NO'}")
        if bt.get("leakage_flags"):
            print(f"  ⚠️  Leakage flags:    {', '.join(bt['leakage_flags'])}")
    except Exception:
        pass

    try:
        av = json.loads((REPORTS_DIR / "alpha_validation.json").read_text())
        p = av["permutation_test"]["p_value"]
        ic_std = av["bootstrap_stability"]["ic_std"]
        overall = av["overall_verdict"]
        print(f"  Permutation p-value:  {p:.4f}")
        print(f"  Bootstrap IC std:     {ic_std:.4f}")
        print(f"  Alpha validation:     {overall}")
    except Exception:
        pass

    print()
    print("  Reports:")
    for f in sorted(REPORTS_DIR.glob("*.json")):
        print(f"    {f.name}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RSSS Phase 1 Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--debug", action="store_true",
                        help="Pass --debug to each script (fast, small data)")
    parser.add_argument("--force-01", action="store_true",
                        help="Force re-run of feature builder even if features exist")
    parser.add_argument("--start", default="01",
                        help="Start from this script number (01-06)")
    parser.add_argument("--only", default=None,
                        help="Run only this script number")
    parser.add_argument("--no-gates", action="store_true",
                        help="Skip go/no-go gates (for debugging)")
    args = parser.parse_args()

    debug_args = ["--debug"] if args.debug else []

    log.info(
        "pipeline_start",
        debug=args.debug,
        start=args.start,
        only=args.only,
    )

    step_results = []
    pipeline_ok = True

    for num, filename, name in SCRIPTS:
        # Filtering
        if args.only and num != args.only:
            continue
        if not args.only and int(num) < int(args.start):
            continue

        script_path = PIPELINE_DIR / filename

        # Script 01: skip if features already exist (unless --force-01)
        if num == "01" and FEATURES_PARQUET.exists() and not args.force_01:
            log.info("script_01_skipped", reason="features.parquet already exists")
            print(f"\n  [01] Feature Builder — SKIPPED (features.parquet exists)")
            step_results.append({
                "num": num, "name": name, "success": True,
                "elapsed_s": 0, "gate_note": "skipped (cached)"
            })
            continue

        print(f"\n{'─'*60}")
        print(f"  Running [{num}] {name}...")
        print(f"{'─'*60}")

        rc, elapsed = run_script(script_path, debug_args)
        success = rc == 0

        gate_note = ""
        if not success:
            log.error("script_failed", num=num, name=name, rc=rc)
            pipeline_ok = False
            step_results.append({
                "num": num, "name": name, "success": False,
                "elapsed_s": elapsed, "gate_note": f"FAILED (exit code {rc})"
            })
            print(f"\n  ❌ Script {num} failed (exit code {rc}). Stopping pipeline.")
            break

        # Post-step gate check
        if not args.no_gates and num in POST_GATES:
            gate_ok, gate_msg = POST_GATES[num]()
            if not gate_ok:
                log.warning("gate_failed", num=num, msg=gate_msg)
                gate_note = gate_msg
                step_results.append({
                    "num": num, "name": name, "success": True,
                    "elapsed_s": elapsed, "gate_note": f"GATE: {gate_note}"
                })
                print(f"\n  ⚠️  GATE: {gate_msg}")
                print("  Pipeline paused — review findings before continuing.")
                pipeline_ok = False
                break
            else:
                gate_note = gate_msg

        step_results.append({
            "num": num, "name": name, "success": True,
            "elapsed_s": elapsed, "gate_note": gate_note
        })
        log.info("script_complete", num=num, name=name, elapsed_s=round(elapsed, 1))

    print_final_summary(step_results)

    if pipeline_ok:
        print("\n  ✅ Phase 1 pipeline complete.")
    else:
        print("\n  ⚠️  Pipeline stopped early — see gate/error above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
