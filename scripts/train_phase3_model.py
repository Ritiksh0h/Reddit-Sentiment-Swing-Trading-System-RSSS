"""
Train Phase 3 XGBoost model.
Uses locked architecture: 11 features, sentiment dropped per L1.
Saves to models/registry/phase3_model.pkl

Run once before daily_run.py. Re-run weekly for retraining.
"""
import json
import pickle
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
import xgboost as xgb

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

with open('experiments/phase3_locked_architecture.json') as f:
    ARCH = json.load(f)

FEATURES     = ARCH['features']   # 11 features
DROP_TICKERS = set(ARCH['drop_tickers'])

XGB_PARAMS = dict(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1,
    objective='reg:squarederror',
)


def train(
    feature_path: str = 'data/features/features_expanded.parquet',
    output_path:  str = 'models/registry/phase3_model.pkl',
) -> dict:
    """Train and save the Phase 3 model. Returns training metrics."""
    df = pd.read_parquet(feature_path)

    # Apply filters matching architecture
    df = df[df['post_count_1d'] >= 10].copy()
    df = df[~df['ticker'].isin(DROP_TICKERS)].copy()

    train_df = df[df['split'] == 'train']
    test_df  = df[df['split'] == 'test']

    avail = [f for f in FEATURES if f in train_df.columns]
    missing = [f for f in FEATURES if f not in train_df.columns]
    if missing:
        logger.warning(f'Features missing from parquet: {missing}')

    X_tr = train_df[avail].fillna(0)
    y_tr = train_df['target_return_5d']
    X_te = test_df[avail].fillna(0)
    y_te = test_df['target_return_5d']

    logger.info(f'Training: {len(train_df)} rows, {len(avail)} features')
    logger.info(f'Test:     {len(test_df)} rows')
    logger.info(f'Features: {avail}')

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_tr, y_tr)

    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)

    ic_train = float(stats.spearmanr(pred_tr, y_tr).statistic)
    ic_test  = float(stats.spearmanr(pred_te, y_te).statistic)
    mae_test = float(np.mean(np.abs(pred_te - y_te)))
    dir_acc  = float(np.mean(np.sign(pred_te) == np.sign(y_te)))

    logger.info(f'IC train: {ic_train:.4f}  IC test: {ic_test:.4f}')
    logger.info(f'MAE test: {mae_test:.4f}  Dir acc: {dir_acc:.3f}')

    # Save model
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

    # Save metadata alongside model
    metadata = {
        'model_id':       'phase3_xgb_5d_v1',
        'trained_at':     datetime.now(timezone.utc).isoformat(),
        'train_period':   ['2019-01-01', '2023-12-31'],
        'feature_list':   avail,
        'n_features':     len(avail),
        'n_train':        len(train_df),
        'n_test':         len(test_df),
        'ic_train':       round(ic_train, 4),
        'ic_test':        round(ic_test, 4),
        'mae_test':       round(mae_test, 6),
        'dir_acc_test':   round(dir_acc, 4),
        'xgb_params':     XGB_PARAMS,
        'drop_tickers':   list(DROP_TICKERS),
        'density_gate':   'post_count_1d >= 10',
        'sentiment_in_model': False,
    }
    meta_path = Path(output_path).with_suffix('.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f'Model saved: {output_path}')
    logger.info(f'Metadata:    {meta_path}')

    return metadata


if __name__ == '__main__':
    result = train()
    print('\n=== Phase 3 Model Training Complete ===')
    print(json.dumps(result, indent=2))
