"""
Backfill 1D/3D predictions for pre-V2 paper trade records.

Pre-V2 OPEN records in paper_trades_pre_v2.jsonl have predicted_1d=0.0,
predicted_3d=0.0, confidence=0.0 because Phase 3 only used a 5D model.
This script looks those records up in features_v2_with_atr.parquet,
runs model_1d_v2.json and model_3d_v2.json, and writes the corrected
records into paper_trades.jsonl (which is currently empty).

Hard rules observed:
- paper_trades.jsonl is append-only; we only write, never truncate
- No random splits, no future data, no model changes
- NEVER retrain: this only re-scores existing records with existing models
"""
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent

# V2 feature columns (16) — must match train_models_v2.py order
V2_FEATURES = [
    'post_count_1d', 'abnormal_attention_1d', 'total_comments_1d',
    'vader_sentiment_1d', 'sentiment_extremity', 'sentiment_accel',
    'volume', 'relative_volume', 'returns_1d', 'returns_20d',
    'rsi_14', 'news_sentiment_1d', 'vix_percentile', 'vix_x_volume',
    'dist_from_20ma_pct', 'pead_proxy',
]

BULLISH_THRESHOLD = 0.015
BEARISH_THRESHOLD = -0.015
MIN_PRED_RET      = 0.005


def load_v2_models():
    models = {}
    for hz in ('1d', '3d', '5d'):
        path = ROOT / f'models/model_{hz}_v2.json'
        if not path.exists():
            log.error(f'Model not found: {path}')
            sys.exit(1)
        b = xgb.Booster()
        b.load_model(str(path))
        models[hz] = b
        log.info(f'Loaded {path.name}')
    return models


def load_feature_store() -> pd.DataFrame:
    path = ROOT / 'data/features/features_v2_with_atr.parquet'
    df = pd.read_parquet(path)
    df['date'] = df['date'].astype(str).str[:10]
    log.info(f'Feature store: {len(df)} rows, {df["date"].min()} → {df["date"].max()}')
    return df


def score_record(row: pd.Series, models: dict) -> dict:
    """Return {'1d': float, '3d': float, '5d': float}."""
    X = row[V2_FEATURES].fillna(0.0).values.reshape(1, -1)
    dm = xgb.DMatrix(X, feature_names=V2_FEATURES)
    return {hz: float(m.predict(dm)[0]) for hz, m in models.items()}


def derive_signal_and_conf(pred_1d, pred_3d, pred_5d):
    if max(abs(pred_1d), abs(pred_3d), abs(pred_5d)) < MIN_PRED_RET:
        return 'NEUTRAL', 0.0

    if pred_5d >= BULLISH_THRESHOLD:
        best, signal = pred_5d, 'BULLISH'
    elif pred_3d >= BULLISH_THRESHOLD:
        best, signal = pred_3d, 'BULLISH'
    elif pred_1d >= BULLISH_THRESHOLD:
        best, signal = pred_1d, 'BULLISH'
    elif pred_5d <= BEARISH_THRESHOLD:
        best, signal = pred_5d, 'BEARISH'
    elif pred_3d <= BEARISH_THRESHOLD:
        best, signal = pred_3d, 'BEARISH'
    elif pred_1d <= BEARISH_THRESHOLD:
        best, signal = pred_1d, 'BEARISH'
    else:
        best, signal = pred_5d, 'NEUTRAL'

    confidence = round(min(abs(best) / (BULLISH_THRESHOLD * 2), 1.0), 4)
    return signal, confidence


def main():
    models  = load_v2_models()
    feat_df = load_feature_store()
    feat_idx = feat_df.set_index(['date', 'ticker'])

    # Collect OPEN records that need 1D/3D backfill
    source_files = [
        ROOT / 'logs/paper_trades_pre_v2.jsonl',
        ROOT / 'data/backfill_backup/paper_trades.jsonl',
    ]
    records = []
    seen = set()
    for src in source_files:
        if not src.exists():
            continue
        for line in src.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r.get('date',''), r.get('ticker',''), r.get('action',''))
            if key in seen:
                continue
            seen.add(key)
            records.append(r)

    opens_to_fix = [r for r in records if r.get('action') == 'OPEN'
                    and (r.get('predicted_1d', 0.0) == 0.0 or r.get('predicted_1d') is None)]
    log.info(f'Total records from source files: {len(records)}, OPEN needing backfill: {len(opens_to_fix)}')

    updated = 0
    for r in opens_to_fix:
        key = (r['date'], r['ticker'])
        if key not in feat_idx.index:
            log.warning(f'  NOT IN FEATURE STORE: {key}')
            continue
        row = feat_idx.loc[key]
        preds = score_record(row, models)

        p1, p3, p5 = preds['1d'], preds['3d'], preds['5d']
        signal, conf = derive_signal_and_conf(p1, p3, p5)

        # Preserve original Phase 3 5D prediction and signal — only fill missing 1D/3D
        orig_5d = r.get('predicted_return_5d') or r.get('predicted_5d') or r.get('predicted_return', 0.0)
        orig_signal = r.get('signal') or 'NEUTRAL'

        r['predicted_1d'] = round(p1, 6)
        r['predicted_3d'] = round(p3, 6)
        # Recompute signal + confidence from Phase 3 5D using current thresholds.
        # Original Phase 3 records all had signal=NEUTRAL due to a now-fixed threshold bug.
        new_sig, conf_orig = derive_signal_and_conf(p1, p3, float(orig_5d or 0.0))
        r['confidence']   = conf_orig
        r['signal']       = new_sig
        r['backfilled_v2'] = True
        log.info(f'  {r["date"]} {r["ticker"]:6s} 1D={p1*100:+.2f}% 3D={p3*100:+.2f}% orig5D={float(orig_5d or 0)*100:+.2f}% sig={new_sig} conf={conf_orig:.2f}')
        updated += 1

    if updated == 0:
        log.info('Nothing to backfill.')
        return

    # Write all records to paper_trades.jsonl (currently empty)
    out_path = ROOT / 'logs/paper_trades.jsonl'
    existing = set()
    if out_path.exists() and out_path.stat().st_size > 0:
        for line in out_path.read_text().splitlines():
            if line.strip():
                existing.add(line.strip())

    written = 0
    with open(out_path, 'a') as f:
        for r in records:
            line = json.dumps(r, default=str)
            if line not in existing:
                f.write(line + '\n')
                written += 1

    log.info(f'Done — backfilled {updated} predictions, wrote {written} records to {out_path.name}')


if __name__ == '__main__':
    main()
