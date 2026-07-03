"""
Fix 3: Switch from 11-feature model to 17-feature model.
Run ONLY if RED gate triggers on live IC monitor for 2 consecutive weeks.

Rationale: Granger causality (L1) was a diagnostic, not a deletion mandate.
The 17-feature Experiment C model had test IC = 0.111 vs current 0.073.
Sentiment features remain in the dataset — they just didn't pass the
causality threshold. XGBoost can still learn from them if there's signal.

Usage:
    python scripts/fix3_switch_to_17_features.py

STOP: Do not run after only one Red week. Require two consecutive.
"""
import json
import logging
import shutil
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from scipy import stats
import xgboost as xgb

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('fix3')

FEATURES_17 = [
    'returns_1d', 'returns_5d', 'returns_20d', 'rsi_14', 'atr_14',
    'relative_volume', 'dist_from_20ma', 'dist_from_50ma',
    'avg_sentiment_1d', 'avg_sentiment_3d', 'weighted_sentiment',
    'sentiment_std', 'sentiment_accel', 'bullish_ratio',
    'post_count_1d', 'mention_growth_1d', 'mention_growth_7d',
]

XGB_PARAMS_17 = dict(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1,
    objective='reg:squarederror',
)

DROP_TICKERS = ['ASTS', 'LCID', 'MSTR', 'RIOT', 'RIVN', 'SMCI', 'WMT']


def check_consecutive_red_weeks() -> bool:
    """Read ic_monitor.jsonl and confirm 2 consecutive RED weeks."""
    path = Path('logs/ic_monitor.jsonl')
    if not path.exists():
        logger.error('No ic_monitor.jsonl found. Run monitor_live_ic.py first.')
        return False

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if len(records) < 2:
        logger.error(f'Only {len(records)} IC monitor records. Need 2+ RED weeks.')
        return False

    # Check last two records
    last_two = sorted(records, key=lambda r: r.get('date', ''))[-2:]
    both_red = all(r.get('gate') == 'RED' for r in last_two)

    if not both_red:
        gates = [r.get('gate') for r in last_two]
        logger.warning(f'Last two gates: {gates}. Both must be RED to trigger Fix 3.')
    return both_red


def execute_fix3(force: bool = False):
    """Retrain with 17 features and update architecture + model files."""
    if not force and not check_consecutive_red_weeks():
        print('\nFix 3 NOT triggered. Require two consecutive RED weeks in ic_monitor.jsonl.')
        print('Run with --force to override (testing only).')
        return

    logger.warning('FIX3 EXECUTING — switching to 17-feature model')

    df    = pd.read_parquet('data/features/features_expanded.parquet')
    df    = df[df['post_count_1d'] >= 10].copy()
    df    = df[~df['ticker'].isin(DROP_TICKERS)].copy()
    train = df[df['split'] == 'train']
    test  = df[df['split'] == 'test']

    avail = [f for f in FEATURES_17 if f in train.columns]
    missing = [f for f in FEATURES_17 if f not in train.columns]
    if missing:
        logger.warning(f'Features missing from parquet: {missing}')

    logger.info(f'Training 17-feature model on {len(train)} rows...')
    model = xgb.XGBRegressor(**XGB_PARAMS_17)
    model.fit(train[avail].fillna(0), train['target_return_5d'])

    pred_tr = model.predict(train[avail].fillna(0))
    pred_te = model.predict(test[avail].fillna(0))
    ic_tr   = float(stats.spearmanr(pred_tr, train['target_return_5d']).statistic)
    ic_te   = float(stats.spearmanr(pred_te, test['target_return_5d']).statistic)

    logger.info(f'17-feature model — train IC: {ic_tr:.4f}  test IC: {ic_te:.4f}')

    # Backup the 11-feature model
    old_path = Path('models/registry/phase3_model.pkl')
    if old_path.exists():
        shutil.copy(old_path, 'models/registry/phase3_model_11feat_backup.pkl')
        logger.info('Backed up 11-feature model to phase3_model_11feat_backup.pkl')

    # Save new model
    with open('models/registry/phase3_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Save new metadata
    metadata = {
        'model_id':          'phase3_xgb_5d_v2_fix3',
        'trained_at':        datetime.now(timezone.utc).isoformat(),
        'features':          avail,
        'feature_count':     len(avail),
        'sentiment_in_model': True,
        'ic_train':          round(ic_tr, 4),
        'ic_test':           round(ic_te, 4),
        'fix3_applied':      True,
        'fix3_reason':       'Live IC < 0.01 for 2 consecutive weeks',
        'prior_model':       'phase3_model_11feat_backup.pkl',
    }
    with open('models/registry/phase3_model.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Update locked architecture file
    with open('experiments/phase3_locked_architecture.json') as f:
        arch = json.load(f)

    arch['features']           = avail
    arch['feature_count']      = len(avail)
    arch['sentiment_in_model'] = True
    arch['fix3_applied']       = True
    arch['fix3_applied_date']  = datetime.now(timezone.utc).isoformat()
    arch['fix3_reason']        = 'Live IC < 0.01 for 2 consecutive weeks'
    arch['model_version']      = 'phase3_v2_fix3'

    with open('experiments/phase3_locked_architecture.json', 'w') as f:
        json.dump(arch, f, indent=2)

    print(f'\nFix 3 complete.')
    print(f'  New model test IC: {ic_te:.4f}')
    print(f'  Features: {len(avail)} (was 11)')
    print(f'  Backup: models/registry/phase3_model_11feat_backup.pkl')
    print(f'  Architecture updated.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true',
                        help='Skip consecutive-Red-week check (testing only)')
    args = parser.parse_args()
    execute_fix3(force=args.force)
