"""
Understand the non-stationarity pattern in detail.
Show sentiment IC by: year, subreddit, ticker type (meme vs tech vs finance)
"""
import pandas as pd
import numpy as np
from scipy import stats
import json

df = pd.read_parquet('data/features/features_expanded.parquet')
df = df[df['post_count_1d'] >= 10].copy()
df['year'] = pd.to_datetime(df['date']).dt.year

print('=== SENTIMENT IC BY YEAR ===')
print(f'{"Year":<6} {"n":>6} {"avg_sent":>10} {"weighted":>10} {"sent_accel":>12} {"mention_gr":>12}')
print('-' * 60)

yearly = {}
for year in sorted(df['year'].unique()):
    yr = df[df['year'] == year]
    ic_s = stats.spearmanr(yr['avg_sentiment_1d'].fillna(0), yr['target_return_5d']).correlation
    ic_w = stats.spearmanr(yr['weighted_sentiment'].fillna(0), yr['target_return_5d']).correlation if 'weighted_sentiment' in yr.columns else float('nan')
    ic_a = stats.spearmanr(yr['sentiment_accel'].fillna(0), yr['target_return_5d']).correlation if 'sentiment_accel' in yr.columns else float('nan')
    ic_m = stats.spearmanr(yr['mention_growth_1d'].fillna(0), yr['target_return_5d']).correlation if 'mention_growth_1d' in yr.columns else float('nan')
    yearly[year] = {'ic_sentiment': float(ic_s), 'ic_weighted': float(ic_w),
                    'ic_accel': float(ic_a), 'ic_mention': float(ic_m), 'n': len(yr)}
    print(f'{year:<6} {len(yr):>6} {ic_s:>10.4f} {ic_w:>10.4f} {ic_a:>12.4f} {ic_m:>12.4f}')

print()
print('=== IC TREND ===')
ics = [yearly[y]['ic_sentiment'] for y in sorted(yearly)]
years = sorted(yearly.keys())
if len(years) >= 2:
    slope = (ics[-1] - ics[0]) / max(len(ics) - 1, 1)
    print(f'Sentiment IC trend: {slope:+.4f} per year')
    print(f'Min year IC: {min(ics):.4f} ({years[ics.index(min(ics))]})')
    print(f'Max year IC: {max(ics):.4f} ({years[ics.index(max(ics))]})')

print('\n=== SUBREDDIT BREAKDOWN ===')
if 'subreddit' in df.columns:
    for sub in df['subreddit'].unique():
        s = df[df['subreddit'] == sub]
        if len(s) < 50:
            continue
        ic = stats.spearmanr(s['avg_sentiment_1d'].fillna(0), s['target_return_5d']).correlation
        print(f'  {sub:<20} n={len(s):>5}  IC={ic:.4f}')
else:
    print('  (no subreddit column in feature store)')

print('\n=== TRAIN vs TEST IC ===')
for split in ['train', 'test']:
    s = df[df['split'] == split]
    ic = stats.spearmanr(s['avg_sentiment_1d'].fillna(0), s['target_return_5d']).correlation
    ic_w = stats.spearmanr(s['weighted_sentiment'].fillna(0), s['target_return_5d']).correlation
    print(f'  {split:<6}  n={len(s):>5}  IC_avg_sent={ic:.4f}  IC_weighted={ic_w:.4f}')

with open('reports/eda/regime_analysis.json', 'w') as f:
    json.dump({'yearly': {str(k): v for k, v in yearly.items()}}, f, indent=2)
print('\nSaved: reports/eda/regime_analysis.json')
