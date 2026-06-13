"""
For each ticker, show:
- Number of qualifying rows (post_count >= 10)
- Raw IC per feature
- Signal consistency across years
- Whether the ticker should be in training data
"""
import pandas as pd
import numpy as np
from scipy import stats
import json

df = pd.read_parquet('data/features/features_expanded.parquet')
df10 = df[df['post_count_1d'] >= 10].copy()
df10['year'] = pd.to_datetime(df10['date']).dt.year

KEY_FEATURES = ['avg_sentiment_1d', 'weighted_sentiment',
                'mention_growth_1d', 'returns_5d', 'rsi_14', 'relative_volume']
KEY_FEATURES = [f for f in KEY_FEATURES if f in df10.columns]

print(f'{"Ticker":<8} {"n_total":>8} {"n_train":>8} {"n_test":>8} '
      f'{"IC_sent":>9} {"IC_mktm":>9} {"IC_rsi":>9} {"Verdict"}')
print('-' * 80)

ticker_verdicts = {}
for ticker, grp in df10.groupby('ticker'):
    n_total = len(grp)
    n_train = (grp['split'] == 'train').sum()
    n_test  = (grp['split'] == 'test').sum()
    if n_total < 20:
        continue

    ic_sent = stats.spearmanr(
        grp['avg_sentiment_1d'].fillna(0),
        grp['target_return_5d']
    ).correlation if 'avg_sentiment_1d' in grp.columns else float('nan')

    ic_mom = stats.spearmanr(
        grp['returns_5d'].fillna(0),
        grp['target_return_5d']
    ).correlation if 'returns_5d' in grp.columns else float('nan')

    ic_rsi = stats.spearmanr(
        grp['rsi_14'].fillna(50),
        grp['target_return_5d']
    ).correlation if 'rsi_14' in grp.columns else float('nan')

    any_positive_ic = any([
        not np.isnan(ic_sent) and ic_sent > 0.03,
        not np.isnan(ic_mom) and ic_mom > 0.03,
        not np.isnan(ic_rsi) and ic_rsi > 0.03,
    ])
    verdict = 'KEEP' if (n_train >= 50 and any_positive_ic) else 'MARGINAL'
    if n_train < 30:
        verdict = 'DROP'

    ticker_verdicts[ticker] = {
        'n_total': n_total, 'n_train': n_train, 'n_test': n_test,
        'ic_sentiment': float(ic_sent), 'ic_momentum': float(ic_mom),
        'ic_rsi': float(ic_rsi), 'verdict': verdict
    }
    print(f'{ticker:<8} {n_total:>8} {n_train:>8} {n_test:>8} '
          f'{ic_sent:>9.4f} {ic_mom:>9.4f} {ic_rsi:>9.4f}  {verdict}')

with open('reports/eda/ticker_verdicts.json', 'w') as f:
    json.dump(ticker_verdicts, f, indent=2, default=str)

keep = [t for t, v in ticker_verdicts.items() if v['verdict'] == 'KEEP']
drop = [t for t, v in ticker_verdicts.items() if v['verdict'] == 'DROP']
marg = [t for t, v in ticker_verdicts.items() if v['verdict'] == 'MARGINAL']
print(f'\nKEEP:     {len(keep)} tickers: {keep}')
print(f'MARGINAL: {len(marg)} tickers: {marg}')
print(f'DROP:     {len(drop)} tickers: {drop}')
print('\nSaved: reports/eda/ticker_verdicts.json')
