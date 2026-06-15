# Claude Code — Feature Store Rebuild + Retrain (2022-2023 Train / 2024-2025 Test)
# Reddit Sentiment Swing Trading System (RSSS)
# GitHub: https://github.com/Ritiksh0h/Reddit-Sentiment-Swing-Trading-System-RSSS

---

## Context

New data collected: 932,484 rows covering 2019-2026.
File confirmed on disk: data/raw/merged_with_sentiment_full.parquet

Current state:
  Feature store: data/features/features_expanded.parquet (13,067 rows, 2019-2024)
  Current split: Train 2019-2023 / Test 2024
  Current model: 14 features, IC_test=0.100 (2024 only)

Target state:
  Feature store: data/features/features_full.parquet (more rows, 2019-2026)
  New split:     Train 2022-2023 / Test 2024-2025
  New models:    Model_1D, Model_3D, Model_5D retrained on new split

---

## Session Start — Verify Input File

```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('data/raw/merged_with_sentiment_full.parquet')
print(f'Rows: {len(df):,}')
print('Years:')
import numpy as np
years = pd.to_datetime(df['timestamp'], utc=True).dt.year
print(years.value_counts().sort_index())
print()
print('Columns:', df.columns.tolist())
req = ['post_id','subreddit','ticker','title','score',
       'num_comments','timestamp','sentiment_score','sentiment_label']
missing = [c for c in req if c not in df.columns]
print(f'Missing required cols: {missing if missing else \"none\"}')
"
```

Expected:
```
Rows: 932,484
Years: 2019-2026 all present
Missing required cols: none
```

If this fails — stop. Do not proceed until the file is verified.

---

## Task 1 — Rebuild Feature Store

### Step 1a — Check how pipeline/01_feature_builder.py accepts arguments

```bash
python pipeline/01_feature_builder.py --help 2>/dev/null || \
python pipeline/01_feature_builder.py --input-file --help 2>/dev/null || \
head -60 pipeline/01_feature_builder.py
```

This tells us the exact argument names before running.

### Step 1b — Run the feature builder

```bash
python pipeline/01_feature_builder.py \
  --input-file data/raw/merged_with_sentiment_full.parquet \
  --output-file data/features/features_full.parquet \
  --force-recompute
```

If the pipeline does not accept those arguments, read the script and
find the correct way to point it at the new input file. The key requirement:
  - INPUT:  data/raw/merged_with_sentiment_full.parquet
  - OUTPUT: data/features/features_full.parquet
  - Must include 2025 and 2026 rows in output

Do NOT overwrite data/features/features_expanded.parquet — keep it as backup.

### Step 1c — Verify the new feature store

```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('data/features/features_full.parquet')
print(f'Feature store rows: {len(df):,}')
print(f'Columns: {len(df.columns)}')
print()

# Year distribution
df['year'] = pd.to_datetime(df['date']).dt.year
print('Rows by year:')
print(df['year'].value_counts().sort_index())
print()

# Target columns
targets = [c for c in df.columns if 'target' in c]
print(f'Target columns: {targets}')

# Split column
if 'split' in df.columns:
    print(f'Split values: {df[\"split\"].value_counts().to_dict()}')
print()

# Feature columns
features = [c for c in df.columns if c not in ['ticker','date','split','year'] + targets]
print(f'Feature columns ({len(features)}): {features}')
"
```

Expected:
```
Feature store rows: 15,000-20,000  (more than current 13,067)
Years: 2019-2026 all present
Target columns: includes target_return_1d, target_return_3d, target_return_5d
```

---

## Task 2 — Update Train/Test Split

The existing feature store uses a 'split' column where train=2019-2023, test=2024.
We need to override this with the new split: train=2022-2023, test=2024-2025.
2019-2021 and 2026 rows are excluded from training and testing
(2026 data is too recent for 5-day forward returns to be complete).

### Add split override to scripts/train_phase3_model.py

Find the section in train_phase3_model.py that loads df and applies the split.
It currently does something like:

```python
train_df = df[df['split'] == 'train']
test_df  = df[df['split'] == 'test']
```

Replace with an explicit year-based override:

```python
# Override split with explicit year ranges
# Train: 2022-2023 (recent enough to capture current market behavior)
# Test:  2024-2025 (two-year out-of-sample validation)
# Excluded: 2019-2021 (too old, different regime), 2026 (incomplete)
df['year'] = pd.to_datetime(df['date']).dt.year

train_df = df[df['year'].isin([2022, 2023])].copy()
test_df  = df[df['year'].isin([2024, 2025])].copy()

logger.info(f'Split override: train={len(train_df)} rows '
            f'({train_df["year"].value_counts().to_dict()})')
logger.info(f'Split override: test={len(test_df)} rows '
            f'({test_df["year"].value_counts().to_dict()})')

# Minimum row check
if len(train_df) < 500:
    raise ValueError(f'Too few training rows: {len(train_df)}. '
                     'Check feature store has 2022-2023 data.')
if len(test_df) < 200:
    raise ValueError(f'Too few test rows: {len(test_df)}. '
                     'Check feature store has 2024-2025 data.')
```

### Also add --feature-path argument to train_phase3_model.py

Add argument parsing so it can accept the new feature store path:

```python
# Add at the top of the if __name__ == '__main__': block:
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--feature-path', type=str,
                    default='data/features/features_full.parquet',
                    help='Path to feature store parquet file')
parser.add_argument('--train-years', type=str, default='2022,2023',
                    help='Comma-separated training years')
parser.add_argument('--test-years', type=str, default='2024,2025',
                    help='Comma-separated test years')
args = parser.parse_args()

# Pass to train():
results = train(
    feature_path=args.feature_path,
    train_years=[int(y) for y in args.train_years.split(',')],
    test_years=[int(y) for y in args.test_years.split(',')],
)
```

Update the `train()` function signature:

```python
def train(
    feature_path: str = 'data/features/features_full.parquet',
    output_dir:   str = 'models/registry',
    train_years:  list = None,
    test_years:   list = None,
) -> dict:
    if train_years is None: train_years = [2022, 2023]
    if test_years  is None: test_years  = [2024, 2025]
    ...
```

---

## Task 3 — Retrain All Three Models

```bash
python scripts/train_phase3_model.py \
  --feature-path data/features/features_full.parquet \
  --train-years 2022,2023 \
  --test-years 2024,2025
```

### Expected output

```
Split override: train=X rows ({2022: X, 2023: X})
Split override: test=X rows  ({2024: X, 2025: X})
Training on X rows, testing on X rows
Features: 14

Model_1D: IC_test=0.03-0.07  IC_train=0.20-0.45  dir_acc=51-54%
Model_3D: IC_test=0.05-0.09  IC_train=0.25-0.50  dir_acc=52-55%
Model_5D: IC_test=0.07-0.11  IC_train=0.30-0.55  dir_acc=52-56%
```

If IC_test for all three models comes back negative or near zero:
  - This means 2022-2023 training does not generalize to 2024-2025
  - Fall back to train_years=[2019,2020,2021,2022,2023] (original split)
  - Keep test_years=[2024,2025] (two-year test stays)

### Train row count expectation

```
2022 rows: ~1,500-3,000 (after density gate post_count >= 10)
2023 rows: ~800-2,000
Total train: ~2,500-5,000 rows

2024 rows: ~1,500-3,000
2025 rows: ~800-2,500
Total test: ~2,500-5,500 rows
```

If train rows < 500 after density gate — the 2022-2023 data is too sparse.
Switch to train_years=[2019,2020,2021,2022,2023] automatically.

---

## Task 4 — Compare Results

After retraining, compare new IC against current baseline:

```bash
python3 -c "
import json

# Load old baseline
with open('models/registry/phase3_model_baseline.json') as f:
    old = json.load(f)

print('=== MODEL COMPARISON ===')
print()
print(f'{\"\":<20} {\"OLD\":>12} {\"NEW\":>12}')
print('-' * 46)

# Old baseline was single 5D model
old_ic = old.get('test_ic_2024') or old.get('horizons',{}).get('5d',{}).get('ic_test', '?')
print(f'{\"Train split\":<20} {\"2019-2023\":>12} {\"2022-2023\":>12}')
print(f'{\"Test split\":<20} {\"2024 only\":>12} {\"2024-2025\":>12}')
print(f'{\"Model_5D IC_test\":<20} {str(old_ic):>12}')
print()

# New metrics from fresh baseline
try:
    with open('models/registry/phase3_model_baseline.json') as f:
        new = json.load(f)
    horizons = new.get('horizons', {})
    for h in ['1d','3d','5d']:
        m = horizons.get(h, {})
        ic  = m.get('ic_test', '?')
        acc = m.get('dir_acc', '?')
        print(f'  Model_{h.upper()}: IC_test={ic}  dir_acc={acc}')
except Exception as e:
    print(f'Could not load new baseline: {e}')
"
```

---

## Task 5 — Run Verification Dry Run

After retraining, confirm the live pipeline still works:

```bash
# Dry run to confirm models load and generate signals
python scripts/daily_run_live.py --dry-run
```

Expected:
```
INFO  Fetching live Reddit data...
INFO  Fetching yfinance news...
INFO  Fetching StockTwits...
INFO  Combined data: 35-40 tickers
INFO  signals_generated count=X bullish=X neutral=X bearish=X
```

If model loading fails — check that `models/registry/model_1d.pkl`,
`model_3d.pkl`, `model_5d.pkl` all exist after retraining.

---

## Task 6 — Push

```bash
bash push.sh "[data] rebuild feature store 2019-2026 + retrain 2022-2023/2024-2025 split"
```

---

## Decision Tree — What to Do If Things Go Wrong

### If feature builder fails with argument error:
Read pipeline/01_feature_builder.py carefully and find the correct
argument names. Do not guess. The script may use positional args,
different flag names, or a config file instead of CLI args.

### If 2025 rows are missing from feature store:
Check that the feature builder reads all years from the input file.
It may have a hardcoded date filter. Find it and remove or extend it.

### If train rows < 500 after density gate:
Switch to: train_years = [2019, 2020, 2021, 2022, 2023]
Keep:      test_years  = [2024, 2025]
This gives more training data while still testing on two years.

### If IC_test is negative for all three models:
This means the 2022-2023 bear market trained model does not
generalize to 2024-2025 bull market. Expected and documented.
Switch to: train_years = [2019, 2020, 2021, 2022, 2023]
The original split (2019-2023 train) had IC_test=0.100.
That is your fallback.

### If pipeline/01_feature_builder.py takes too long (>2 hours):
It may be recomputing yfinance data for all tickers from scratch.
Check if there is a --skip-market-data flag or similar.
If not, let it run — the first rebuild always takes longest.

---

## Files to Create / Modify

```
CREATE:   data/features/features_full.parquet    ← rebuilt feature store
MODIFY:   scripts/train_phase3_model.py          ← year-based split override
                                                    --feature-path argument
                                                    --train-years argument
                                                    --test-years argument
UPDATE:   models/registry/model_1d.pkl           ← retrained
UPDATE:   models/registry/model_3d.pkl           ← retrained
UPDATE:   models/registry/model_5d.pkl           ← retrained
UPDATE:   models/registry/phase3_model.pkl       ← backward compat (=model_5d)
UPDATE:   models/registry/phase3_model_baseline.json ← new metrics
```

## Files NOT to touch

```
data/raw/merged_with_sentiment_expanded.parquet  ← keep as backup
data/features/features_expanded.parquet          ← keep as backup
experiments/                                     ← locked
portfolio/                                       ← unchanged
scripts/daily_run.py                             ← unchanged
scripts/daily_run_live.py                        ← unchanged
```

---

## Summary of What This Achieves

```
Before:
  Training data: 2019-2023 (11,880 rows after density gate)
  Test data:     2024 only (2,187 rows)
  Test period:   1 year out-of-sample
  IC_test:       0.100 (2024 only)

After:
  Training data: 2022-2023 (~2,500-5,000 rows after density gate)
  Test data:     2024-2025 (~2,500-5,500 rows)
  Test period:   2 years out-of-sample
  IC_test:       TBD — more robust validation
```

Two-year out-of-sample test is significantly stronger evidence
than one year. 2024 was a bull market year. 2025 had mixed regimes.
If the model holds IC across both years, the signal is genuinely robust.

---

*Feature Store Rebuild + Retrain — June 2026*
*Input: 932,484 rows (2019-2026)*
*New split: Train 2022-2023 / Test 2024-2025*
*Fallback: Train 2019-2023 / Test 2024-2025 if train rows < 500*
