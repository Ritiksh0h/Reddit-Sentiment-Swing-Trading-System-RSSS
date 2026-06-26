"""
Statistical validation harness for RSSS.
Computes PSR, DSR, minTRL, binomial significance.

Run after every 50 new live trades.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm, binomtest
from statsmodels.stats.proportion import proportion_confint


def load_all_trades() -> list:
    """Load all closed trades from logs."""
    trades = []
    for path in [
        'logs/paper_trades_pre_v2.jsonl',
        'logs/paper_trades.jsonl',
    ]:
        p = Path(path)
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    try:
                        t = json.loads(line)
                        if t.get('status') == 'CLOSED':
                            trades.append(t)
                    except Exception:
                        pass
    return trades


def compute_psr(
    returns: list,
    sr_benchmark: float = 0.0,
) -> float:
    """
    Probabilistic Sharpe Ratio.
    Probability that true SR > benchmark.
    """
    T = len(returns)
    if T < 10:
        return 0.0

    r      = np.array(returns)
    sr_hat = r.mean() / (r.std() + 1e-10)
    skew   = float(pd.Series(r).skew())
    kurt   = float(pd.Series(r).kurtosis()) + 3  # non-excess

    sr_var = (1 - skew * sr_hat + ((kurt - 1) / 4) * sr_hat ** 2) / (T - 1)

    psr = norm.cdf((sr_hat - sr_benchmark) / np.sqrt(max(sr_var, 1e-12)))
    return float(psr)


def compute_dsr(
    returns: list,
    n_trials: int = 50,
    trials_sr_std: float = 0.5,
) -> tuple[float, float]:
    """
    Deflated Sharpe Ratio.
    Corrects for multiple testing.
    Returns (dsr, sr0_threshold).
    """
    T = len(returns)
    if T < 10:
        return 0.0, 0.0

    euler_gamma = 0.5772156649
    z1  = norm.ppf(1 - 1 / n_trials)
    z2  = norm.ppf(1 - 1 / (n_trials * np.e))
    sr0 = trials_sr_std * ((1 - euler_gamma) * z1 + euler_gamma * z2)

    dsr = compute_psr(returns, sr_benchmark=sr0)
    return float(dsr), float(sr0)


def min_track_record_length(
    returns: list,
    sr_benchmark: float = 0.0,
    target_psr: float = 0.95,
) -> int:
    """
    Minimum number of observations needed
    to achieve PSR > target_psr.
    """
    if len(returns) < 5:
        return 999

    r      = np.array(returns)
    sr_hat = r.mean() / (r.std() + 1e-10)
    skew   = float(pd.Series(r).skew())
    kurt   = float(pd.Series(r).kurtosis()) + 3

    z     = norm.ppf(target_psr)
    num   = z ** 2 * (1 - skew * sr_hat + ((kurt - 1) / 4) * sr_hat ** 2)
    denom = (sr_hat - sr_benchmark) ** 2

    if denom <= 0:
        return 9999

    return int(np.ceil(num / denom + 1))


def block_bootstrap_sharpe(
    returns: list,
    block_size: int = 5,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> dict:
    """
    Block bootstrap for Sharpe ratio.
    Handles time-series dependence.
    block_size=5 matches 5-day holding period.
    """
    if len(returns) < 20:
        return {
            'sharpe_mean': None,
            'sharpe_ci_low': None,
            'sharpe_ci_high': None,
        }
    r = np.array(returns)
    n = len(r)
    bootstrap_sharpes = []
    for _ in range(n_bootstrap):
        indices = []
        while len(indices) < n:
            start = np.random.randint(
                0, max(1, n - block_size + 1))
            indices.extend(range(
                start, min(start + block_size, n)))
        boot_r = r[indices[:n]]
        std = boot_r.std()
        if std > 0:
            sr = (boot_r.mean() / std *
                  np.sqrt(252))
            bootstrap_sharpes.append(sr)
    if not bootstrap_sharpes:
        return {
            'sharpe_mean': None,
            'sharpe_ci_low': None,
            'sharpe_ci_high': None,
        }
    alpha = 1 - confidence
    return {
        'sharpe_mean': round(float(
            np.mean(bootstrap_sharpes)), 3),
        'sharpe_ci_low': round(float(
            np.percentile(bootstrap_sharpes,
                          alpha / 2 * 100)), 3),
        'sharpe_ci_high': round(float(
            np.percentile(bootstrap_sharpes,
                          (1 - alpha / 2) * 100)), 3),
        'n_bootstrap': n_bootstrap,
        'block_size': block_size,
    }


def run_validation():
    """Run full statistical validation."""
    trades = load_all_trades()

    print('=' * 55)
    print('RSSS STATISTICAL VALIDATION REPORT')
    print('=' * 55)
    print(f'Total closed trades: {len(trades)}')
    print()

    if len(trades) < 10:
        print('INSUFFICIENT DATA — need at least 10 trades')
        return

    returns  = [t.get('pnl_pct', 0) for t in trades]
    wins     = sum(1 for r in returns if r > 0)
    win_rate = wins / len(returns)

    # 1. Binomial test
    binom     = binomtest(wins, len(trades), 0.5, alternative='greater')
    ci_low, ci_high = proportion_confint(wins, len(trades), alpha=0.05, method='wilson')

    print('--- DIRECTIONAL ACCURACY ---')
    print(f'Win rate:    {win_rate:.1%}')
    print(f'95% CI:      {ci_low:.1%} → {ci_high:.1%}')
    print(f'p-value:     {binom.pvalue:.3f}')
    print(f'Significant: {"YES ✓" if binom.pvalue < 0.05 else "NO ✗"}')
    print()

    # 2. PSR
    psr = compute_psr(returns, sr_benchmark=0.0)
    print('--- PROBABILISTIC SHARPE RATIO ---')
    print(f'PSR (vs 0):  {psr:.3f}')
    print(f'Significant: {"YES ✓" if psr > 0.95 else "NO ✗"}')
    print()

    # 2b. Block bootstrap Sharpe
    bootstrap = block_bootstrap_sharpe(returns)
    print()
    print('--- BLOCK BOOTSTRAP SHARPE ---')
    if bootstrap['sharpe_mean'] is not None:
        print(f'Mean Sharpe:  {bootstrap["sharpe_mean"]}')
        print(f'95% CI:       {bootstrap["sharpe_ci_low"]}'
              f' → {bootstrap["sharpe_ci_high"]}')
        ci_positive = (bootstrap['sharpe_ci_low'] or 0) > 0
        print(f'Significant:  '
              f'{"YES ✓" if ci_positive else "NO ✗"}')
    else:
        print('Insufficient data')
    print()

    # 3. DSR
    dsr, sr0 = compute_dsr(returns, n_trials=50, trials_sr_std=0.5)
    print('--- DEFLATED SHARPE RATIO ---')
    print(f'N trials:        50 (configs tested)')
    print(f'SR0 (threshold): {sr0:.4f}')
    print(f'DSR:             {dsr:.3f}')
    print(f'Significant:     {"YES ✓" if dsr > 0.95 else "NO ✗"}')
    print()

    # 4. minTRL
    min_T = min_track_record_length(returns, sr_benchmark=0.0, target_psr=0.95)
    print('--- MINIMUM TRACK RECORD ---')
    print(f'Current trades:    {len(trades)}')
    print(f'Need for PSR>0.95: {min_T}')
    print(f'Still needed:      {max(0, min_T - len(trades))}')
    print()

    # 5. Overall verdict
    all_pass = binom.pvalue < 0.05 and psr > 0.95 and dsr > 0.95

    print('=' * 55)
    print(f'OVERALL STATUS: {"VALIDATED ✓" if all_pass else "NOT YET VALIDATED ✗"}')
    if not all_pass:
        print(f'Next milestone: {min_T} total trades')
        print(f'Current:        {len(trades)} trades')
        print(f'Gap:            {max(0, min_T - len(trades))} more needed')
    print('=' * 55)


if __name__ == '__main__':
    run_validation()
