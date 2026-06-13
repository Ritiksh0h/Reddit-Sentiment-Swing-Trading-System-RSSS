"""
Aggregate all three layer results.
Produce experiments/phase3_locked_architecture.json.
This file is the ONLY output Phase 3 consumes from this sprint.
"""
import json
from pathlib import Path

print('=' * 65)
print('SIGNAL VALIDATION SPRINT — FINAL SUMMARY')
print('=' * 65)

L1 = L2 = L3 = None

try:
    with open('experiments/layer1_signal_existence/granger_results.json') as f:
        L1 = json.load(f)
    print(f'\nL1 SIGNAL EXISTENCE:')
    print(f'  Sentiment family: {L1["sentiment_sig_years"]}/6 years significant')
    print(f'  Attention family: {L1["attention_sig_years"]}/6 years significant')
    print(f'  Proceed verdict:  {L1["proceed_to_layer2"]}')
except FileNotFoundError:
    print('\nL1 — NOT RUN')

try:
    with open('experiments/layer2_regime/regime_results.json') as f:
        L2 = json.load(f)
    print(f'\nL2 REGIME CLASSIFIER:')
    for r, s in L2['position_sizing'].items():
        print(f'  {r.upper():<10} → {s:.0%} position size')
except FileNotFoundError:
    print('\nL2 — NOT RUN')

try:
    with open('experiments/layer3_model/family_validation_results.json') as f:
        L3 = json.load(f)
    print(f'\nL3 FAMILY VALIDATION:')
    print(f'  Verdict:       {L3["verdict"]}')
    print(f'  Effective IC:  {L3["effective_ic"]["effective_ic"]:.4f}')
    print(f'  Permutation p: {L3["permutation_test"]["p_value"]:.3f}')
    print(f'  WF ICs:        {[round(x, 4) for x in L3["clean_walk_forward_ics"]]}')
    adopted_features = L3['adopted_features']
except FileNotFoundError:
    print('\nL3 — NOT RUN')
    adopted_features = []

# Determine final architecture
sentiment_in_model = True
if L1 and not L1['proceed_to_layer2']:
    sentiment_in_model = False
    print('\nNOTE: L1 found weak signal. Sentiment family dropped from model.')

if L3 and L3['verdict'] == 'REJECT':
    adopted_features = [
        'returns_1d', 'returns_5d', 'returns_20d', 'rsi_14', 'atr_14',
        'relative_volume', 'dist_from_20ma', 'dist_from_50ma',
        'avg_sentiment_1d', 'avg_sentiment_3d', 'weighted_sentiment',
        'sentiment_std', 'sentiment_accel', 'bullish_ratio',
        'post_count_1d', 'mention_growth_1d', 'mention_growth_7d'
    ]
    print('NOTE: L3 rejected pruning. Using original 17 features.')

regime_sizing = L2['position_sizing'] if L2 else {
    'positive': 1.0, 'neutral': 0.75, 'negative': 0.50
}

print(f'\n{"=" * 65}')
print('PHASE 3 LOCKED ARCHITECTURE')
print('=' * 65)

arch = {
    'features':           adopted_features,
    'sentiment_in_model': bool(sentiment_in_model),
    'density_gate':       'post_count_1d >= 10',
    'drop_tickers':       ['ASTS', 'LCID', 'MSTR', 'RIOT', 'RIVN', 'SMCI', 'WMT'],
    'regime_sizing':      regime_sizing,
    'hold_days':          5,
    'take_profit_cap':    0.15,
    'max_positions':      3,
    'paper_trading_required': bool(
        L3 and 'MARGINAL' in L3.get('verdict', '')
    ),

    'phase4_requirements': {
        'position_sizing': 'atr_based',
        'position_sizing_formula': (
            'target_risk_pct=0.02 × portfolio_value / (atr_14 × price) '
            '→ gives dollar position size. '
            'Apply regime_multiplier on top. '
            'Hard cap: never exceed 25% of portfolio in one position.'
        ),
        'slippage_model': 'dynamic_atr_based',
        'slippage_formula': (
            'base_slippage = 0.001 (0.1%) '
            '+ attention_spike_addon '
            'where attention_spike_addon = 0.0005 × min(mention_growth_7d, 3.0). '
            'Example: mention_growth_7d=2.0 → extra 0.10% slippage. '
            'Captures opening auction volatility on high-attention days.'
        ),
        'volume_constraint': (
            'Max order size = 1% of ticker ADV (20-day avg daily volume). '
            'If simulated order exceeds this → partial fill at 50% of size.'
        ),
        'execution_logging': {
            'required': True,
            'log_per_signal': [
                'ticker', 'date', 'feature_vector_10d', 'raw_finbert_scores',
                'regime_state', 'regime_multiplier', 'predicted_return_5d',
                'atr_14', 'position_size_dollars', 'slippage_applied',
                'fill_price', 'signal_timestamp',
            ],
            'log_format': 'structured JSON per trade, appended to trades.jsonl',
            'storage': 'logs/paper_trades.jsonl',
        },
        'data_drift_monitoring': {
            'required': True,
            'check_features': ['post_count_1d', 'avg_sentiment_1d', 'mention_growth_7d'],
            'alert_threshold': 'flag if live value < historical_mean × 0.5 OR > historical_mean × 2.0',
            'historical_means': {
                'post_count_1d':    53.2,
                'avg_sentiment_1d': -0.025,
                'mention_growth_7d': 0.232,
            },
            'action_on_alert': 'log warning and skip trading day',
        },
        'fallback_handlers': {
            'api_anomaly': {
                'trigger': 'live post_count_1d < historical_mean × 0.50 for >70% of tickers',
                'action':  'skip trading day entirely — log reason — hold existing positions',
            },
            'stale_market_data': {
                'trigger': 'market data timestamp > 2 hours before market open',
                'action':  'halt all new signals — log critical error',
            },
            'zero_signals': {
                'trigger': 'zero tickers pass density gate on a given day',
                'action':  'hold cash — do not force trades',
            },
            'model_prediction_error': {
                'trigger': 'XGBoost raises exception or returns NaN predictions',
                'action':  'skip trading day — alert',
            },
        },
        'nlp_upgrade': {
            'status': 'deferred_to_version_2',
            'rationale': 'Permutation p=0.130 on original model. Sentiment contributes ~0.003 IC above market baseline. Custom fine-tuning deferred until paper trading confirms live signal.',
        },
    },

    'notes': {
        'density_filter_finding':  'Market IC 0.008 → 0.092 with density gate. Filter is primary value driver.',
        'sentiment_non_stationary': 'IC: +0.086 (2019) → -0.103 (2023) → +0.115 (2024). Regime-dependent.',
        'TSLA_concentration':       '31% of rows. TSLA-specific signal may dominate. Monitor in paper trading.',
        'permutation_context':      'p=0.130 on original model. Tighter features may improve this.',
        'phase4_priority':          'ATR sizing and execution logging are highest priority. Build before first paper trade.',
    },

    'layer_results': {
        'L1_signal_existence': {
            'sentiment_sig_years': L1['sentiment_sig_years'] if L1 else None,
            'attention_sig_years': L1['attention_sig_years'] if L1 else None,
            'proceed': L1['proceed_to_layer2'] if L1 else None,
        } if L1 else 'NOT_RUN',
        'L2_regime': 'RUN' if L2 else 'NOT_RUN',
        'L3_family_validation': {
            'verdict':       L3['verdict'] if L3 else None,
            'effective_ic':  L3['effective_ic']['effective_ic'] if L3 else None,
            'perm_p_value':  L3['permutation_test']['p_value'] if L3 else None,
        } if L3 else 'NOT_RUN',
    },
}

for k in ('features', 'density_gate', 'drop_tickers', 'regime_sizing',
          'hold_days', 'take_profit_cap', 'max_positions', 'paper_trading_required'):
    print(f'  {k}: {arch[k]}')

with open('experiments/phase3_locked_architecture.json', 'w') as f:
    json.dump(arch, f, indent=2)
print('\nSaved: experiments/phase3_locked_architecture.json')
print('\nPhase 3 reads this file. Sprint is complete.')
