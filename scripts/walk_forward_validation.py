#!/usr/bin/env python3
"""
walk_forward_validation.py — RSSS v2 expanding-window walk-forward.

Quarterly folds (63 trading days), 1-year minimum training.
Purge 5 days + embargo 5 days at each train/test boundary.

For each fold:
  1. Retrain 1D/3D/5D GKX models on expanding train slice
  2. Run rank-based core-satellite OOS backtest on fold test period
  3. Run IS backtest on last 252 training days (for WFE)
  4. Record metrics + regime label

Aggregate:
  Pooled OOS Sharpe, per-regime breakdown, 5 pass/fail gates.
  DSR confidence (Bailey & Lopez de Prado 2014).

Results: experiments/walk_forward/results.json
Models:  experiments/walk_forward/fold_models/fold_N_model_{hz}.json
"""

import json
import math
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from scipy.stats import spearmanr
from xgboost import callback as xgb_callback
import yfinance as yf

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE         = Path(__file__).resolve().parent.parent
FEAT_PATH    = BASE / "data" / "features" / "features_v2.parquet"
WF_DIR       = BASE / "experiments" / "walk_forward"
FOLD_MODELS  = WF_DIR / "fold_models"
RESULTS_PATH = WF_DIR / "results.json"

# ── Fold structure ─────────────────────────────────────────────────────────────
FOLD_STEP    = 63   # trading days per fold (1 quarter)
MIN_TRAIN    = 252  # minimum training days (1 year)
PURGE_DAYS   = 5    # days removed from end of train (label overlap)
EMBARGO_DAYS = 5    # buffer days between train end and test start

WF_START = "2019-01-01"
WF_END   = "2025-12-31"  # exclude live 2026 data

# ── Feature set (must match train_models_v2.py) ────────────────────────────────
FEATURE_COLS = [
    "post_count_1d",       "abnormal_attention_1d",
    "total_comments_1d",   "vader_sentiment_1d",
    "sentiment_extremity", "sentiment_accel",
    "volume",              "relative_volume",
    "returns_1d",          "returns_20d",
    "rsi_14",              "news_sentiment_1d",
    "vix_percentile",      "vix_x_volume",
    "spy_above_200ma",     "regime_score",
]

TARGETS = {
    "1d": "target_return_1d",
    "3d": "target_return_3d",
    "5d": "target_return_5d",
}

DENSITY_GATE = 5  # minimum post_count_1d for training

# ── GKX hyperparameters (match train_models_v2.py exactly) ────────────────────
GKX_PARAMS = dict(
    max_depth        = 1,
    learning_rate    = 0.02,
    objective        = "reg:pseudohubererror",
    subsample        = 0.7,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    random_state     = 42,
    n_jobs           = -1,
    eval_metric      = "rmse",
)

HORIZON_PARAMS = {
    "1d": {"gamma": 0.0,  "reg_lambda": 3.0, "min_child_weight": 15},
    "3d": {"gamma": 0.05, "reg_lambda": 1.0, "min_child_weight": 10},
    "5d": {"gamma": 0.5,  "reg_lambda": 5.0, "min_child_weight": 20},
}

# ── Portfolio constants ────────────────────────────────────────────────────────
TRADE_TICKERS = {
    "AAPL", "AMD",  "AMZN", "COIN", "GME",  "GOOG", "HOOD",
    "MARA", "META", "MSFT", "MU",   "NFLX", "NVDA", "PLTR",
    "QQQ",  "SOFI", "TSLA", "UBER",
}
INITIAL_CAPITAL   = 10_000.0
CORE_PCT          = 0.70
SATELLITE_PCT     = 0.30
MAX_POSITIONS     = 4
MAX_DAILY_SIGNALS = 2
MIN_ACTIVITY_GATE = 3
STOP_LOSS         = -0.08
TAKE_PROFIT       = 0.10
SLIPPAGE          = 0.0015
COOLDOWN_DAYS     = 7
MAX_HOLD_DAYS     = 5

IS_LOOKBACK = 252   # trading days used for IS backtest within each fold


# ── Regime labels ──────────────────────────────────────────────────────────────
def get_regime_label(date_str: str) -> str:
    d = date.fromisoformat(date_str)
    if d < date(2020, 2, 1):  return "bull_2019"
    if d < date(2020, 6, 1):  return "covid_crash"
    if d < date(2021, 1, 1):  return "covid_recovery"
    if d < date(2022, 1, 1):  return "meme_bull_2021"
    if d < date(2023, 1, 1):  return "bear_2022"
    if d < date(2024, 1, 1):  return "recovery_2023"
    return "ai_bull_2024"


# ── Fold generation ────────────────────────────────────────────────────────────
def generate_folds(all_dates: list[str]) -> list[dict]:
    """
    Expanding-window quarterly folds.
    Train always starts at all_dates[0].
    Train end moves forward each fold, keeping purge/embargo gaps.
    Test window is exactly FOLD_STEP trading days.
    """
    folds = []
    fold_n = 1
    ti = MIN_TRAIN  # index of test_start in all_dates

    while ti + FOLD_STEP < len(all_dates):
        eff_train_end_i = ti - EMBARGO_DAYS - PURGE_DAYS
        if eff_train_end_i < MIN_TRAIN // 2:
            ti += FOLD_STEP
            continue

        test_end_i = min(ti + FOLD_STEP, len(all_dates) - 1)

        folds.append({
            "fold":        fold_n,
            "train_start": all_dates[0],
            "train_end":   all_dates[eff_train_end_i],
            "test_start":  all_dates[ti],
            "test_end":    all_dates[test_end_i],
            "regime":      get_regime_label(all_dates[ti]),
        })
        ti += FOLD_STEP
        fold_n += 1

    return folds


# ── ICEarlyStopping (matches train_models_v2.py) ──────────────────────────────
class ICEarlyStopping(xgb_callback.TrainingCallback):
    """Stop when Spearman IC on the eval set stops improving."""

    def __init__(self, rounds: int, X_eval, y_eval) -> None:
        super().__init__()
        self.rounds          = rounds
        self.X_eval          = X_eval.values if hasattr(X_eval, "values") else X_eval
        self.y_eval          = np.asarray(y_eval)
        self.best_ic         = -np.inf
        self.best_iteration  = 0
        self._no_improve     = 0

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        pred = model.inplace_predict(self.X_eval)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ic_raw, _ = spearmanr(pred, self.y_eval)
        ic = float(ic_raw) if not np.isnan(ic_raw) else -1.0
        if ic > self.best_ic:
            self.best_ic, self.best_iteration, self._no_improve = ic, epoch, 0
        else:
            self._no_improve += 1
        return self._no_improve >= self.rounds


# ── Model training per fold ────────────────────────────────────────────────────
def train_fold_models(
    train_df: pd.DataFrame,
    fold_n:   int,
) -> tuple[dict[str, xgb.XGBRegressor], dict[str, float]]:
    """
    Retrain 1D/3D/5D GKX models on fold's expanding training window.
    Uses density gate and same two-phase ICEarlyStopping as train_models_v2.py.
    Saves fold model JSONs to experiments/walk_forward/fold_models/.
    """
    gated = train_df[train_df["post_count_1d"] >= DENSITY_GATE].copy()
    for col in FEATURE_COLS:
        if col not in gated.columns:
            gated[col] = 0.0
    gated[FEATURE_COLS] = gated[FEATURE_COLS].fillna(0.0)

    models:   dict[str, xgb.XGBRegressor] = {}
    ic_scores: dict[str, float] = {}

    for hz, target_col in TARGETS.items():
        tr = gated.dropna(subset=[target_col])
        h_params = {**GKX_PARAMS, **HORIZON_PARAMS[hz]}

        if len(tr) < 30:
            # Too few rows: trivial model
            m = xgb.XGBRegressor(n_estimators=1, **h_params)
            dummy_X = pd.DataFrame(np.zeros((max(1, len(tr)), len(FEATURE_COLS))),
                                   columns=FEATURE_COLS).astype(np.float32)
            dummy_y = np.zeros(len(dummy_X), dtype=np.float32)
            m.fit(dummy_X, dummy_y)
            models[hz]    = m
            ic_scores[hz] = 0.0
            continue

        X = tr[FEATURE_COLS].astype(np.float32)
        y = tr[target_col].values.astype(np.float32)
        w = (tr["sample_weight"].values.astype(np.float32)
             if "sample_weight" in tr.columns
             else np.ones(len(tr), dtype=np.float32))

        # Scout phase
        stopper = ICEarlyStopping(rounds=20, X_eval=X, y_eval=y)
        scout   = xgb.XGBRegressor(n_estimators=200, **h_params, callbacks=[stopper])
        scout.fit(X, y, sample_weight=w, eval_set=[(X, y)], verbose=False)
        best_n = max(1, stopper.best_iteration + 1)

        # Clean final model
        model = xgb.XGBRegressor(n_estimators=best_n, **h_params)
        model.fit(X, y, sample_weight=w)

        pred = model.predict(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ic_raw, _ = spearmanr(pred, y)
        ic_scores[hz] = float(ic_raw) if not np.isnan(ic_raw) else 0.0
        models[hz]    = model

        # Save fold model
        path = FOLD_MODELS / f"fold_{fold_n}_model_{hz}.json"
        model.save_model(str(path))

    return models, ic_scores


# ── Backtest helpers ───────────────────────────────────────────────────────────
def _composite_score(p1: float, p3: float, p5: float) -> float:
    return 0.5 * p5 + 0.3 * p3 + 0.2 * p1


def _exit_date(tdays: list[str], entry: str, hold: int) -> str:
    try:
        i = tdays.index(entry)
        return tdays[min(i + hold, len(tdays) - 1)]
    except ValueError:
        return entry if tdays else ""


# ── Per-fold simulation ────────────────────────────────────────────────────────
def simulate(
    df_slice:     pd.DataFrame,
    models:       dict[str, xgb.XGBRegressor],
    price_lut:    dict,
    vol_lut:      dict,
    fold_tdays:   list[str],
    spy_prices:   dict,
    drop_tickers: set,
) -> dict:
    """
    Core-satellite simulation on a date slice.
    Core: 70% SPY, bought at slice start, sold at end.
    Satellite: rank-based long signals (top 2/day composite score).
    Matches run_backtest_v2.py logic exactly.
    """
    if not fold_tdays:
        return {
            "n_trades": 0, "total_return_pct": 0.0, "sat_return_pct": 0.0,
            "win_rate": 0.0, "sharpe": 0.0, "max_dd_pct": 0.0,
            "daily_returns": [], "trades": [],
        }

    spy_start = spy_prices.get(fold_tdays[0], 470.0)
    spy_end   = spy_prices.get(fold_tdays[-1], spy_start)

    core_cap    = INITIAL_CAPITAL * CORE_PCT
    sat_cap     = INITIAL_CAPITAL * SATELLITE_PCT
    core_shares = core_cap / spy_start if spy_start > 0 else 0
    sat_cash    = sat_cap

    open_pos: list[dict] = []
    closed:   list[dict] = []
    cooldown: dict = {}
    equity_curve: list[float] = []

    for current_date in fold_tdays:
        # Mark-to-market
        spy_price = spy_prices.get(current_date, spy_start)
        sat_val   = sum(
            p["n"] * price_lut.get((p["ticker"], current_date), p["fill"])
            for p in open_pos
        )
        equity_curve.append(core_shares * spy_price + sat_cash + sat_val)

        # Exit checks
        to_close = []
        for pos in open_pos:
            if current_date == pos["entry_date"]:
                continue
            cur = price_lut.get((pos["ticker"], current_date))
            if cur is None:
                continue
            ret = (cur - pos["fill"]) / pos["fill"]
            if   ret <= STOP_LOSS:                  to_close.append((pos, "stop_loss"))
            elif ret >= TAKE_PROFIT:                to_close.append((pos, "take_profit"))
            elif current_date >= pos["exit_date"]:  to_close.append((pos, "hold_expired"))

        for pos, reason in to_close:
            cur   = price_lut.get((pos["ticker"], current_date), pos["fill"])
            xfill = cur * (1 - SLIPPAGE)
            sat_cash += pos["n"] * xfill
            open_pos  = [p for p in open_pos if p["ticker"] != pos["ticker"]]
            cooldown[pos["ticker"]] = current_date
            closed.append({
                "pnl_pct":    (xfill - pos["fill"]) / pos["fill"],
                "exit_reason": reason,
            })

        # Signal generation
        n_slots = MAX_POSITIONS - len(open_pos)
        if n_slots == 0:
            continue

        today_rows = df_slice[df_slice["date"] == current_date]
        if today_rows.empty:
            continue

        candidates = []
        for _, row in today_rows.iterrows():
            t = row["ticker"]
            if t not in TRADE_TICKERS or t in drop_tickers:
                continue
            if row["post_count_1d"] < MIN_ACTIVITY_GATE:
                continue
            if any(p["ticker"] == t for p in open_pos):
                continue
            le = cooldown.get(t)
            if le:
                d_diff = (date.fromisoformat(current_date) -
                          date.fromisoformat(le)).days
                if d_diff < COOLDOWN_DAYS:
                    continue

            X  = pd.DataFrame([row[FEATURE_COLS].fillna(0.0).to_dict()])
            p5 = float(models["5d"].predict(X)[0])
            p3 = float(models["3d"].predict(X)[0])
            p1 = float(models["1d"].predict(X)[0])
            score = _composite_score(p1, p3, p5)

            if score <= 0:                                         continue
            if float(row.get("regime_score",    0.3)) < 0.3:      continue
            if float(row.get("relative_volume", 1.0)) < 0.8:      continue

            cur_price = price_lut.get((t, current_date), float(row["close"]))
            candidates.append({"ticker": t, "score": score, "close": cur_price})

        candidates.sort(key=lambda x: x["score"], reverse=True)

        for c in candidates[: min(MAX_DAILY_SIGNALS, n_slots)]:
            if len(open_pos) >= MAX_POSITIONS:
                break
            t = c["ticker"]
            if any(p["ticker"] == t for p in open_pos):
                continue

            # Volatility-targeted sizing
            vol = vol_lut.get((t, current_date), 0.02)
            vol = max(min(vol, 0.05), 0.005)
            confidence  = min(abs(c["score"]) / 0.05, 1.0)
            conf_scalar = 0.5 + confidence * 0.5
            size = sat_cap * 0.15 * (0.01 / vol) * conf_scalar
            size = min(size, sat_cap * 0.25)
            size = max(size, sat_cap * 0.05)

            entry_fill = c["close"] * (1 + SLIPPAGE)
            n          = int(size / entry_fill)
            if n == 0 or n * entry_fill > sat_cash:
                continue

            sat_cash -= n * entry_fill
            open_pos.append({
                "ticker":     t,
                "entry_date": current_date,
                "exit_date":  _exit_date(fold_tdays, current_date, MAX_HOLD_DAYS),
                "fill":       entry_fill,
                "n":          n,
            })

    # Force-close remaining at period end
    last = fold_tdays[-1]
    for pos in list(open_pos):
        cur   = price_lut.get((pos["ticker"], last), pos["fill"])
        xfill = cur * (1 - SLIPPAGE)
        sat_cash += pos["n"] * xfill
        closed.append({
            "pnl_pct":    (xfill - pos["fill"]) / pos["fill"],
            "exit_reason": "end_of_period",
        })

    # Final accounting
    core_final  = core_shares * spy_end * (1 - SLIPPAGE)
    total_final = core_final + sat_cash
    total_ret   = (total_final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    sat_ret     = (sat_cash - sat_cap) / sat_cap * 100

    n_trades = len(closed)
    wins     = sum(1 for t in closed if t["pnl_pct"] > 0)
    win_rate = wins / n_trades if n_trades > 0 else 0.0

    # Daily returns from equity curve
    daily_rets: list[float] = []
    if len(equity_curve) > 1:
        for i in range(1, len(equity_curve)):
            if equity_curve[i - 1] > 0:
                daily_rets.append(
                    (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1])

    # Sharpe (annualized)
    if len(daily_rets) > 5:
        mu_d = float(np.mean(daily_rets))
        sg_d = float(np.std(daily_rets, ddof=1)) or 1e-9
        sharpe = mu_d / sg_d * math.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown
    peak, max_dd = equity_curve[0] if equity_curve else INITIAL_CAPITAL, 0.0
    for eq in equity_curve:
        peak   = max(peak, eq)
        max_dd = min(max_dd, (eq - peak) / peak * 100)

    return {
        "n_trades":         n_trades,
        "total_return_pct": round(total_ret,  3),
        "sat_return_pct":   round(sat_ret,    3),
        "win_rate":         round(win_rate,   4),
        "sharpe":           round(sharpe,     3),
        "max_dd_pct":       round(max_dd,     2),
        "daily_returns":    daily_rets,
        "trades":           closed,
    }


# ── Deflated Sharpe Ratio ─────────────────────────────────────────────────────
def deflated_sharpe_confidence(
    oos_daily_returns: np.ndarray,
    n_folds: int,
) -> float:
    """
    Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio.
    Accounts for non-normality of returns and multiple testing
    across n_folds independent fold strategies.
    Returns P(observed SR is genuine), not just due to selection.
    """
    T = len(oos_daily_returns)
    if T < 20 or n_folds < 2:
        return 0.0

    mu  = float(np.mean(oos_daily_returns))
    sig = float(np.std(oos_daily_returns, ddof=1)) or 1e-9
    sr  = mu / sig  # per-period (non-annualized)

    skew     = float(stats.skew(oos_daily_returns))
    kurt_exc = float(stats.kurtosis(oos_daily_returns))   # excess kurtosis

    # Expected maximum SR under n_folds independent Gaussian trials
    sr_star = float(stats.norm.ppf(1.0 - 1.0 / n_folds))

    # SR estimator variance (Mertens 2002 / BLP 2014)
    # var(SR) = (1 - skew × SR + (kurt_full-1)/4 × SR²) / (T-1)
    # kurt_full = kurt_exc + 3  →  (kurt_full-1)/4 = (kurt_exc+2)/4
    var_sr = (
        (1.0 - skew * sr + (kurt_exc + 2.0) / 4.0 * sr ** 2) / max(T - 1, 1)
    )
    std_sr = math.sqrt(abs(var_sr)) or 1e-9

    dsr_z = (sr - sr_star) / std_sr
    return float(stats.norm.cdf(dsr_z))


# ── Aggregate + gates ─────────────────────────────────────────────────────────
def aggregate_results(fold_results: list[dict]) -> dict:
    """Pool OOS daily returns, compute per-regime breakdown, evaluate 5 gates."""
    n = len(fold_results)

    # Pooled OOS daily returns (concatenated chronologically)
    all_daily = np.array(
        [r for fr in fold_results for r in fr["oos"]["daily_returns"]],
        dtype=float,
    )

    # Pooled Sharpe
    if len(all_daily) > 10:
        mu_d = float(np.mean(all_daily))
        sg_d = float(np.std(all_daily, ddof=1)) or 1e-9
        pooled_sharpe = round(mu_d / sg_d * math.sqrt(252), 3)
    else:
        pooled_sharpe = 0.0

    avg_return   = round(sum(fr["oos"]["total_return_pct"] for fr in fold_results) / n, 2)
    avg_win_rate = round(sum(fr["oos"]["win_rate"]         for fr in fold_results) / n, 4)
    total_trades = sum(fr["oos"]["n_trades"] for fr in fold_results)

    # Per-regime breakdown
    regime_agg: dict[str, dict] = defaultdict(lambda: {
        "folds": 0, "ret_sum": 0.0, "sharpe_sum": 0.0,
        "wins": 0, "n_trades": 0,
    })
    for fr in fold_results:
        a = regime_agg[fr["regime"]]
        a["folds"]      += 1
        a["ret_sum"]    += fr["oos"]["total_return_pct"]
        a["sharpe_sum"] += fr["oos"]["sharpe"]
        a["wins"]       += sum(1 for t in fr["oos"]["trades"] if t["pnl_pct"] > 0)
        a["n_trades"]   += fr["oos"]["n_trades"]

    regime_summary: dict[str, dict] = {}
    for regime, a in regime_agg.items():
        fn = a["folds"]
        regime_summary[regime] = {
            "n_folds":    fn,
            "avg_return": round(a["ret_sum"] / fn, 2),
            "avg_sharpe": round(a["sharpe_sum"] / fn, 3),
            "win_rate":   round(a["wins"] / max(a["n_trades"], 1), 4),
            "n_trades":   a["n_trades"],
        }

    # WFE = mean OOS sharpe / mean IS sharpe
    is_sharpes  = [fr["is_sharpe"]    for fr in fold_results if fr.get("is_sharpe", 0) != 0]
    oos_sharpes = [fr["oos"]["sharpe"] for fr in fold_results]
    if is_sharpes and float(np.mean(is_sharpes)) != 0:
        wfe = float(np.mean(oos_sharpes)) / float(np.mean(is_sharpes))
        wfe = round(float(np.clip(wfe, -2.0, 5.0)), 3)
    else:
        wfe = 0.0

    # DSR confidence
    dsr_conf = deflated_sharpe_confidence(all_daily, n)

    # Gate inputs
    pos_folds = sum(1 for fr in fold_results if fr["oos"]["total_return_pct"] > 0)
    pct_pos   = pos_folds / n

    bear_folds = [fr for fr in fold_results if fr["regime"] == "bear_2022"]
    bear_ret   = (sum(fr["oos"]["total_return_pct"] for fr in bear_folds)
                  / len(bear_folds)) if bear_folds else 0.0

    # 5 gates
    g1 = pooled_sharpe >= 0.8
    g2 = pct_pos >= 0.60
    g3 = bear_ret > -20.0
    g4 = wfe >= 0.50
    g5 = dsr_conf >= 0.95

    return {
        "n_folds":              n,
        "pooled_sharpe":        pooled_sharpe,
        "avg_return_per_fold":  avg_return,
        "avg_win_rate":         avg_win_rate,
        "total_trades":         total_trades,
        "wfe":                  wfe,
        "dsr_confidence":       round(dsr_conf, 4),
        "pct_positive_folds":   round(pct_pos, 3),
        "bear_2022_return":     round(bear_ret, 2),
        "regime_summary":       regime_summary,
        "gates": {
            "g1_pooled_sharpe_ge_0.8": bool(g1),
            "g2_60pct_profitable":     bool(g2),
            "g3_bear_2022_gt_neg20":   bool(g3),
            "g4_wfe_ge_0.5":           bool(g4),
            "g5_dsr_ge_0.95":          bool(g5),
            "overall":                 bool(g1 and g2 and g3 and g4 and g5),
        },
    }


# ── Print report ──────────────────────────────────────────────────────────────
def print_report(agg: dict) -> None:
    g = agg["gates"]
    W = 56

    def pf(passed: bool) -> str:
        return "PASS ✓" if passed else "FAIL ✗"

    REGIME_ORDER = [
        "bull_2019", "covid_crash", "covid_recovery",
        "meme_bull_2021", "bear_2022", "recovery_2023", "ai_bull_2024",
    ]

    print()
    print("═" * W)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("═" * W)
    print(f"Folds completed: {agg['n_folds']}")
    print()
    print("POOLED OOS PERFORMANCE:")
    print(f"  Sharpe:     {agg['pooled_sharpe']:>+6.2f}")
    print(f"  Return:     {agg['avg_return_per_fold']:>+6.1f}%  (avg per fold)")
    print(f"  Win rate:   {agg['avg_win_rate']:>7.1%}")
    print(f"  Trades:     {agg['total_trades']:>6,}")
    print(f"  WFE:        {agg['wfe']:>+6.2f}  (OOS/IS Sharpe)")
    print(f"  DSR conf:   {agg['dsr_confidence']:>7.1%}")
    print()
    print("PER REGIME:")
    print(f"  {'Regime':<20} {'Folds':>5}  {'Return':>7}  "
          f"{'Sharpe':>6}  {'WinRate':>7}")
    print("  " + "─" * 52)
    for regime in REGIME_ORDER:
        if regime not in agg["regime_summary"]:
            continue
        r = agg["regime_summary"][regime]
        marker = "  ← CRITICAL" if regime == "bear_2022" else ""
        print(f"  {regime:<20} {r['n_folds']:>5}  "
              f"{r['avg_return']:>+6.1f}%  {r['avg_sharpe']:>+6.2f}  "
              f"{r['win_rate']:>6.1%}{marker}")
    print()
    print("PASS/FAIL GATES:")
    print(f"  Gate 1 Pooled Sharpe ≥ 0.8:    {pf(g['g1_pooled_sharpe_ge_0.8']):>8}  "
          f"(got {agg['pooled_sharpe']:+.2f})")
    print(f"  Gate 2 ≥60% folds profitable:  {pf(g['g2_60pct_profitable']):>8}  "
          f"(got {agg['pct_positive_folds']:.0%})")
    print(f"  Gate 3 2022 bear > -20%:        {pf(g['g3_bear_2022_gt_neg20']):>8}  "
          f"(got {agg['bear_2022_return']:+.1f}%)")
    print(f"  Gate 4 WFE ≥ 0.5:               {pf(g['g4_wfe_ge_0.5']):>8}  "
          f"(got {agg['wfe']:.2f})")
    print(f"  Gate 5 DSR confidence ≥ 0.95:  {pf(g['g5_dsr_ge_0.95']):>8}  "
          f"(got {agg['dsr_confidence']:.1%})")
    print()
    overall_str = "✓ PASS" if g["overall"] else "✗ FAIL"
    print(f"  OVERALL: {overall_str}")
    print("═" * W)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.time()
    WF_DIR.mkdir(parents=True, exist_ok=True)
    FOLD_MODELS.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RSSS Walk-Forward Validation v2 — Expanding Window")
    print("=" * 60)

    # ── Load features ─────────────────────────────────────────────────────────
    print("\nLoading features_v2.parquet...")
    df = pd.read_parquet(FEAT_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df[(df["date"] >= WF_START) & (df["date"] <= WF_END)].copy()
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    all_dates = sorted(df["date"].unique().tolist())
    print(f"  {len(df):,} rows  |  {len(all_dates)} trading days  "
          f"({all_dates[0]} → {all_dates[-1]})")

    # Drop tickers (locked architecture)
    arch_path = BASE / "experiments" / "phase3_locked_architecture.json"
    try:
        with open(arch_path) as f:
            drop_tickers = set(json.load(f).get("drop_tickers", []))
    except FileNotFoundError:
        drop_tickers = set()
    print(f"  Drop tickers: {sorted(drop_tickers)}")

    # ── Download prices once ──────────────────────────────────────────────────
    sim_tickers = sorted(
        (set(df["ticker"].unique()) | {"SPY"}) - drop_tickers)
    print(f"\nDownloading prices for {len(sim_tickers)} tickers (2019-2026)...")

    raw = yf.download(
        sim_tickers, start=WF_START, end="2026-01-15",
        auto_adjust=True, progress=False, threads=True,
    )
    close_df = (raw["Close"]
                if isinstance(raw.columns, pd.MultiIndex)
                else raw[["Close"]].rename(columns={"Close": sim_tickers[0]}))

    price_lut: dict = {}
    ticker_series: dict = {}
    for t in sim_tickers:
        if t not in close_df.columns:
            continue
        s = close_df[t].dropna()
        ticker_series[t] = s
        for dt, p in s.items():
            price_lut[(t, dt.strftime("%Y-%m-%d"))] = float(p)

    # Fallback prices from feature store
    for _, row in df.iterrows():
        k = (row["ticker"], row["date"])
        if k not in price_lut:
            price_lut[k] = float(row["close"])

    print(f"  Price lookup: {len(price_lut):,} entries")

    # Volatility lookup (20-day realized vol for position sizing)
    vol_lut: dict = {}
    for t, s in ticker_series.items():
        rv = s.pct_change().rolling(20, min_periods=10).std()
        for dt, v in rv.items():
            if pd.notna(v) and v > 0:
                vol_lut[(t, dt.strftime("%Y-%m-%d"))] = float(v)
    print(f"  Vol lookup:   {len(vol_lut):,} entries")

    spy_prices = {k[1]: v for k, v in price_lut.items() if k[0] == "SPY"}

    # ── Generate folds ────────────────────────────────────────────────────────
    folds = generate_folds(all_dates)
    print(f"\nFolds generated: {len(folds)}")
    for fold in folds[:3]:
        print(f"  Fold {fold['fold']:>2}: "
              f"train {fold['train_start']}→{fold['train_end']}  "
              f"test  {fold['test_start']}→{fold['test_end']}  "
              f"[{fold['regime']}]")
    if len(folds) > 3:
        print(f"  ... ({len(folds) - 3} more)")

    # ── Walk-forward loop ─────────────────────────────────────────────────────
    fold_results: list[dict] = []
    print()

    for fold in folds:
        fn = fold["fold"]
        print(f"  Fold {fn:>2}/{len(folds)}  [{fold['regime']:<20}]  "
              f"test {fold['test_start']}→{fold['test_end']} ",
              end="", flush=True)

        train_df = df[df["date"] <= fold["train_end"]].copy()
        test_df  = df[(df["date"] >= fold["test_start"]) &
                      (df["date"] <= fold["test_end"])].copy()

        if len(train_df) < 200 or test_df.empty:
            print("  [SKIP: insufficient data]")
            continue

        # Retrain models
        models, is_ic = train_fold_models(train_df, fn)

        # OOS backtest
        oos_tdays = sorted(test_df["date"].unique().tolist())
        oos = simulate(test_df, models, price_lut, vol_lut,
                       oos_tdays, spy_prices, drop_tickers)

        # IS backtest on last IS_LOOKBACK training days (for WFE)
        train_end_idx = all_dates.index(fold["train_end"])
        is_start_idx  = max(0, train_end_idx - IS_LOOKBACK)
        is_start_date = all_dates[is_start_idx]
        is_df         = df[(df["date"] >= is_start_date) &
                           (df["date"] <= fold["train_end"])].copy()
        is_tdays      = sorted(is_df["date"].unique().tolist())
        is_sim        = simulate(is_df, models, price_lut, vol_lut,
                                 is_tdays, spy_prices, drop_tickers)

        print(f" OOS={oos['total_return_pct']:>+5.1f}%  "
              f"trades={oos['n_trades']:>3}  "
              f"SR={oos['sharpe']:>+5.2f}")

        fold_results.append({
            "fold":       fn,
            "regime":     fold["regime"],
            "train_end":  fold["train_end"],
            "test_start": fold["test_start"],
            "test_end":   fold["test_end"],
            "oos":        oos,
            "is_sharpe":  is_sim["sharpe"],
            "is_ic":      is_ic,
        })

    if not fold_results:
        print("No folds completed — exiting.")
        return

    # ── Aggregate ─────────────────────────────────────────────────────────────
    print(f"\nAggregating {len(fold_results)} fold results...")
    agg = aggregate_results(fold_results)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_results = []
    for fr in fold_results:
        rec = dict(fr)
        rec["oos"] = {k: v for k, v in fr["oos"].items()
                      if k not in ("daily_returns", "trades")}
        save_results.append(rec)

    out = {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "wf_start":      WF_START,
        "wf_end":        WF_END,
        "fold_step":     FOLD_STEP,
        "min_train":     MIN_TRAIN,
        "purge_days":    PURGE_DAYS,
        "embargo_days":  EMBARGO_DAYS,
        "is_lookback":   IS_LOOKBACK,
        "feature_count": len(FEATURE_COLS),
        "aggregate":     agg,
        "folds":         save_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved → {RESULTS_PATH}")

    elapsed = round(time.time() - t0)
    print(f"Elapsed: {elapsed // 60}m {elapsed % 60}s")

    print_report(agg)


if __name__ == "__main__":
    main()
