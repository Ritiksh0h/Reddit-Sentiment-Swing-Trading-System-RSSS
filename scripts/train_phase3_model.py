"""
Train Phase 3 multi-horizon XGBoost models.
Trains three separate regressors: Model_1D, Model_3D, Model_5D.

Each model predicts forward return for its horizon.
All three use the same 14-feature set.
Saves to models/registry/model_{1d,3d,5d}.pkl

Run: python scripts/train_phase3_model.py
"""
import json
import pickle
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import xgboost as xgb

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

with open('experiments/phase3_locked_architecture.json') as f:
    ARCH = json.load(f)

FEATURES     = ARCH['features']
DROP_TICKERS = set(ARCH['drop_tickers'])

XGB_PARAMS = dict(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.6, colsample_bytree=0.6, min_child_weight=20,
    reg_alpha=0.5, reg_lambda=2.0,
    random_state=42, n_jobs=-1,
    objective='reg:squarederror',
)

HORIZONS = {
    '1d': 'target_return_1d',
    '3d': 'target_return_3d',
    '5d': 'target_return_5d',
}


def train(
    feature_path: str = 'data/features/features_expanded.parquet',
    output_dir:   str = 'models/registry',
) -> dict:
    """Train all three horizon models. Returns metrics dict."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(feature_path)
    df = df[df['post_count_1d'] >= 10].copy()
    df = df[~df['ticker'].isin(DROP_TICKERS)].copy()

    # Historical data predates news/StockTwits collection — default to neutral
    NEW_FEATURES = ['news_sentiment_1d', 'st_sentiment_1d', 'st_bull_pct']
    for feat in NEW_FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0

    train_df = df[df['split'] == 'train']
    test_df  = df[df['split'] == 'test']

    avail = [f for f in FEATURES if f in train_df.columns]
    logger.info(f'Training on {len(train_df)} rows, testing on {len(test_df)} rows')
    logger.info(f'Features ({len(avail)}): {avail}')

    metrics = {}
    models  = {}

    for horizon, target_col in HORIZONS.items():
        if target_col not in train_df.columns:
            logger.warning(f'Missing target {target_col} — skipping {horizon}')
            continue

        logger.info(f'Training Model_{horizon.upper()}...')

        X_tr = train_df[avail].fillna(0)
        y_tr = train_df[target_col]
        X_te = test_df[avail].fillna(0)
        y_te = test_df[target_col]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr, y_tr)

        pred_te = model.predict(X_te)
        pred_tr = model.predict(X_tr)
        ic_te   = float(stats.spearmanr(pred_te, y_te).statistic)
        ic_tr   = float(stats.spearmanr(pred_tr, y_tr).statistic)
        dir_acc = float(np.mean(np.sign(pred_te) == np.sign(y_te)))

        logger.info(f'Model_{horizon.upper()}: IC_test={ic_te:.4f}  '
                    f'IC_train={ic_tr:.4f}  dir_acc={dir_acc:.3f}')

        model_path = Path(output_dir) / f'model_{horizon}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        models[horizon]  = model
        metrics[horizon] = {
            'ic_test':    round(ic_te, 4),
            'ic_train':   round(ic_tr, 4),
            'dir_acc':    round(dir_acc, 3),
            'model_path': str(model_path),
            'n_train':    len(train_df),
            'n_test':     len(test_df),
        }

    # Save backward-compatible model_5d as phase3_model.pkl
    if '5d' in models:
        compat_path = Path(output_dir) / 'phase3_model.pkl'
        with open(compat_path, 'wb') as f:
            pickle.dump(models['5d'], f)
        logger.info(f'Saved backward-compatible: {compat_path}')

    metadata = {
        'model_version': 'phase3_v3_multihorizon',
        'trained_at':    datetime.now(timezone.utc).isoformat(),
        'features':      avail,
        'feature_count': len(avail),
        'horizons':      metrics,
        'fix3_trigger':  'live 30-day IC < 0.01 → switch to 17 features',
    }
    meta_path = Path(output_dir) / 'phase3_model_baseline.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info('All models saved.')
    for h, m in metrics.items():
        logger.info(f'  Model_{h.upper()}: IC={m["ic_test"]:.4f}  '
                    f'DirAcc={m["dir_acc"]:.1%}')

    return metrics


if __name__ == '__main__':
    results = train()
    import json as _json
    print(_json.dumps(results, indent=2))
