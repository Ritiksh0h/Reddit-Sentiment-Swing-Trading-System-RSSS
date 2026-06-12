#!/usr/bin/env python3
"""
Module: experiments/compare.py
Purpose: Load results from all three experiments and produce final comparison.
         Determines which architecture to build the full system around.

Decision rules — winner must satisfy ALL three:
  1. Total 2024 return > SPY return
  2. Sharpe > 1.0
  3. IC > 0.05
Among qualifying experiments, highest Sharpe wins.

If no experiment passes all three criteria — report it honestly.
Do NOT lower thresholds to manufacture a winner.

Usage:
    python experiments/compare.py
    python experiments/compare.py --run-missing   # runs A and B if results missing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.thresholds import EXPERIMENT_MIN_IC, EXPERIMENT_MIN_SHARPE

try:
    import yfinance as _yf

    def _fetch_return(ticker: str, start: str = "2024-01-01", end: str = "2024-12-31") -> float:
        data = _yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if data.empty:
            return 0.0
        if isinstance(data.columns, __import__("pandas").MultiIndex):
            data.columns = data.columns.get_level_values(0)
        close = data["Close"]
        return float(close.iloc[-1] / close.iloc[0] - 1)
except ImportError:
    def _fetch_return(ticker: str, **_) -> float:
        return {"SPY": 0.2605, "QQQ": 0.2550}.get(ticker, 0.0)

EXPERIMENTS: dict[str, str] = {
    "A": "experiments/experiment_a/results.json",
    "B": "experiments/experiment_b/results.json",
    "C": "experiments/experiment_c/results.json",
}

EXPERIMENT_LABELS: dict[str, str] = {
    "A": "Filter+Market",
    "B": "Regime-Aware",
    "C": "Expanded Data",
}

SPY_RETURN_DEFAULT = 0.2605   # fallback if yfinance unavailable
QQQ_RETURN_DEFAULT = 0.2550   # fallback if yfinance unavailable

WINNER_PATH = Path(__file__).parent / "winner.md"


def load_results() -> dict[str, Optional[dict]]:
    results = {}
    for exp_id, path in EXPERIMENTS.items():
        p = Path(path)
        if p.exists():
            with open(p) as f:
                results[exp_id] = json.load(f)
        else:
            results[exp_id] = None
    return results


def run_missing_experiments() -> None:
    """Run experiments A and B if their results don't exist yet."""
    for exp_id, path in [
        ("A", "experiments/experiment_a/results.json"),
        ("B", "experiments/experiment_b/results.json"),
    ]:
        if not Path(path).exists():
            print(f"\nRunning Experiment {exp_id}...")
            script = f"experiments/experiment_{exp_id.lower()}/train.py"
            result = subprocess.run([sys.executable, script], capture_output=False)
            if result.returncode != 0:
                print(f"ERROR: Experiment {exp_id} failed.")


def passes_criteria(r: dict) -> bool:
    return (
        r.get("total_return", 0) > r.get("spy_return", SPY_RETURN_DEFAULT)
        and r.get("sharpe_ratio", 0) > EXPERIMENT_MIN_SHARPE
        and r.get("ic_test", 0) > EXPERIMENT_MIN_IC
    )


def pick_winner(results: dict[str, Optional[dict]]) -> Optional[str]:
    """Return the experiment ID of the winner, or None if no experiment qualifies."""
    qualifying = {
        exp_id: r
        for exp_id, r in results.items()
        if r is not None and passes_criteria(r)
    }
    if not qualifying:
        return None
    return max(qualifying, key=lambda k: qualifying[k].get("sharpe_ratio", 0))


def format_value(v, fmt: str = ".4f") -> str:
    if v is None:
        return "N/A"
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return str(v)


def print_comparison(
    results: dict[str, Optional[dict]],
    spy_return: float = SPY_RETURN_DEFAULT,
    qqq_return: float = QQQ_RETURN_DEFAULT,
) -> None:
    print()
    print("=" * 92)
    print("  PHASE 2 EXPERIMENT COMPARISON")
    print("=" * 92)
    print()

    header = (
        f"  {'Experiment':<26} {'IC_test':>8} {'Sharpe':>7} "
        f"{'Annual%':>9} {'TotalRet%':>10} {'Beats_SPY':>10} {'Beats_QQQ':>10}  Verdict"
    )
    print(header)
    print("  " + "-" * 88)

    for exp_id, r in results.items():
        label = f"{exp_id}: {EXPERIMENT_LABELS[exp_id]}"
        if r is None:
            print(
                f"  {label:<26} {'—':>8} {'—':>7} {'—':>9} {'—':>10} "
                f"{'—':>10} {'—':>10}  WAITING"
            )
            continue

        ic = r.get("ic_test")
        sharpe = r.get("sharpe_ratio")
        annual = r.get("annualized_return")
        total = r.get("total_return")
        beats_raw = r.get("beats_spy")
        beats_spy = beats_raw if isinstance(beats_raw, bool) else str(beats_raw).lower() == "true"
        beats_qqq = bool(total is not None and float(total) > qqq_return)

        verdict = "PASS" if passes_criteria(r) else "FAIL"

        ic_str = format_value(ic, ".4f")
        sharpe_str = format_value(sharpe, ".3f")
        annual_str = f"{float(annual)*100:.1f}%" if annual is not None else "N/A"
        total_str = f"{float(total)*100:.1f}%" if total is not None else "N/A"

        print(
            f"  {label:<26} {ic_str:>8} {sharpe_str:>7} "
            f"{annual_str:>9} {total_str:>10} "
            f"{'YES' if beats_spy else 'NO':>10} {'YES' if beats_qqq else 'NO':>10}  {verdict}"
        )

    spy_pct = f"{spy_return*100:.1f}%"
    qqq_pct = f"{qqq_return*100:.1f}%"
    print(
        f"  {'SPY 2024 (benchmark)':<26} {'—':>8} {'—':>7} "
        f"{spy_pct:>9} {spy_pct:>10} {'—':>10} {'—':>10}  benchmark"
    )
    print(
        f"  {'QQQ 2024 (benchmark)':<26} {'—':>8} {'—':>7} "
        f"{qqq_pct:>9} {qqq_pct:>10} {'—':>10} {'—':>10}  benchmark"
    )
    print("  " + "-" * 88)
    print()
    print(f"  Thresholds: IC > {EXPERIMENT_MIN_IC}, Sharpe > {EXPERIMENT_MIN_SHARPE}, Total return > SPY")
    print()


def write_winner_doc(
    winner_id: str,
    r: dict,
    spy_return: float = SPY_RETURN_DEFAULT,
    qqq_return: float = QQQ_RETURN_DEFAULT,
) -> None:
    label = EXPERIMENT_LABELS.get(winner_id, winner_id)
    beats_spy = bool(r.get("total_return", 0) > spy_return)
    beats_qqq = bool(r.get("total_return", 0) > qqq_return)
    content = f"""# Phase 2 Winner: Experiment {winner_id} — {label}

## Summary

Experiment {winner_id} passed all three Phase 2 criteria and achieved the highest Sharpe ratio.

| Metric           | Value        | Threshold | Status |
|-----------------|-------------|-----------|--------|
| IC (test 2024)  | {r.get('ic_test', 0):.4f}       | > 0.05    | {'PASS' if r.get('ic_test', 0) > 0.05 else 'FAIL'} |
| Sharpe ratio    | {r.get('sharpe_ratio', 0):.3f}      | > 1.0     | {'PASS' if r.get('sharpe_ratio', 0) > 1.0 else 'FAIL'} |
| Total return    | {r.get('total_return', 0)*100:.1f}%        | > SPY     | {'PASS' if beats_spy else 'FAIL'} |
| SPY 2024        | {spy_return*100:.1f}%        | benchmark | —      |
| Beats SPY       | {'YES' if beats_spy else 'NO'}         | required  | {'PASS' if beats_spy else 'FAIL'} |
| QQQ 2024        | {qqq_return*100:.1f}%        | benchmark | —      |
| Beats QQQ       | {'YES' if beats_qqq else 'NO'}         | optional  | {'PASS' if beats_qqq else 'INFO'} |

## Architecture

**Thesis:** {r.get('thesis', 'N/A')}

### Feature Set

"""

    if winner_id == "A":
        content += f"""- **Type:** Market-only XGBoost on attention-filtered universe
- **Features:** {r.get('features', r.get('feature_set', 'MARKET_FEATURES'))}
- **Attention filter:** post_count_1d >= {r.get('min_posts', 10)}, mention_growth_7d >= {r.get('min_growth', 0.3)}
- **Pre-selection gate:** Applied before model training and inference
"""
    elif winner_id == "B":
        content += f"""- **Type:** Regime-aware dual-model system
- **Positive-regime features:** {r.get('positive_regime_features', 'MARKET + SENTIMENT')}
- **Fallback features:** {r.get('market_features', 'MARKET')}
- **Regime params:** {r.get('regime_params', {})}
"""
    elif winner_id == "C":
        content += f"""- **Type:** Combined model on expanded 4-subreddit dataset
- **Features:** {r.get('features', 'MARKET + SENTIMENT')}
- **Density filter:** post_count_1d >= {r.get('density_filter_min_posts', 10)}
"""

    content += f"""
## Backtest Parameters (locked)

- Starting capital: $1,000
- Max positions: 3
- Hold days: 5
- Slippage: 0.1%
- Fee per leg: 0.05%
- Min predicted return to trade: 2%
- No shorting, no leverage

## Detailed Metrics

- Annualized return: {r.get('annualized_return', 0)*100:.1f}%
- Max drawdown: {r.get('max_drawdown', 0)*100:.1f}%
- Win rate: {r.get('win_rate', 0)*100:.1f}%
- Profit factor: {r.get('profit_factor', 'N/A')}
- N trades: {r.get('n_trades', 'N/A')}
- Alpha vs SPY: {r.get('alpha', 0)*100:.1f}%

## Phase 3 Implications

This architecture is the blueprint for the full production system:
- Signal generator should implement this experiment's pre-selection and model routing logic
- Portfolio engine should use the same 5-day hold, 3-position, slippage/fee parameters
- Feature pipeline must reproduce the same feature set without look-ahead leakage

Results file: `experiments/experiment_{winner_id.lower()}/results.json`
"""

    WINNER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WINNER_PATH, "w") as f:
        f.write(content)

    print(f"  Winner documentation saved to: {WINNER_PATH}")


def main(run_missing: bool = False) -> None:
    if run_missing:
        run_missing_experiments()

    print("Fetching 2024 benchmark returns...", end=" ", flush=True)
    spy_return = _fetch_return("SPY")
    qqq_return = _fetch_return("QQQ")
    print(f"SPY={spy_return*100:.1f}%  QQQ={qqq_return*100:.1f}%")

    results = load_results()

    a_ready = results.get("A") is not None
    b_ready = results.get("B") is not None
    c_ready = results.get("C") is not None

    if not a_ready and not b_ready:
        print()
        print("No experiment results found.")
        print("Run: python experiments/experiment_a/train.py")
        print("     python experiments/experiment_b/train.py")
        print("Or:  python experiments/compare.py --run-missing")
        sys.exit(1)

    print_comparison(results, spy_return=spy_return, qqq_return=qqq_return)

    winner_id = pick_winner(results)

    if winner_id is None:
        print("  WINNER: NONE")
        print()
        print("  No experiment passed all three criteria:")
        print(f"    - IC > {EXPERIMENT_MIN_IC}")
        print(f"    - Sharpe > {EXPERIMENT_MIN_SHARPE}")
        print(f"    - Total return > SPY")
        print()
        print("  CONCLUSION: All three architectures fail Phase 2 success criteria.")
        print("  This is a valid negative result.")
        print()
        print("  NEXT STEPS:")
        print("    1. Review Phase 1 findings — the non-stationarity may be fundamental")
        print("    2. Consider longer hold periods (10d, 20d) if 5d is too noisy")
        print("    3. Revisit the feature set — ATR/RSI may be insufficient alone")
        print("    4. Do NOT lower thresholds to force a winner — that is overfitting")
        print()
        if not c_ready:
            print("  NOTE: Experiment C results not yet available (waiting for Colab data).")
            print("        Run compare.py again after expanded data arrives.")
            print()

    else:
        winner_result = results[winner_id]
        label = EXPERIMENT_LABELS[winner_id]
        sharpe = winner_result.get("sharpe_ratio", 0)
        ic = winner_result.get("ic_test", 0)
        total_ret = winner_result.get("total_return", 0)
        beats_qqq = bool(total_ret > qqq_return)

        print(f"  WINNER: Experiment {winner_id} — {label}")
        print()
        print(f"  REASON: Highest qualifying Sharpe ({sharpe:.3f}) with IC={ic:.4f}")
        print(
            f"          and total return {total_ret*100:.1f}% "
            f"vs SPY {spy_return*100:.1f}% / QQQ {qqq_return*100:.1f}%"
        )
        print(f"  Beats QQQ: {'YES' if beats_qqq else 'NO'}")
        print()
        print(f"  NEXT STEP: Build Phase 3 production system using Experiment {winner_id}")
        print(f"             architecture. See experiments/winner.md for full spec.")
        print()

        write_winner_doc(winner_id, winner_result, spy_return=spy_return, qqq_return=qqq_return)

    print("=" * 92)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Phase 2 experiment results")
    parser.add_argument(
        "--run-missing",
        action="store_true",
        help="Auto-run Experiment A and B if their results don't exist",
    )
    args = parser.parse_args()
    main(run_missing=args.run_missing)
