"""
Aggregate all experiment results and determine final Phase 3 architecture.
"""
import json
from pathlib import Path

results = {}
for i in range(1, 6):
    p = Path(f'experiments/ic_improvements/exp{i}_results.json')
    if not p.exists():
        p = Path(f'experiments/ic_improvements/exp{i}_combined_results.json')
    if p.exists():
        with open(p) as f:
            results[f'exp{i}'] = json.load(f)

print('=== PRE-PHASE 3 IC IMPROVEMENT SUMMARY ===')
print()

baseline_ic = 0.0935

improvements = {
    'exp1_neutral_target':  results.get('exp1', {}).get('improvement', 0),
    'exp2_percentile_rank': max(
        (v.get('improvement', 0) for v in results.get('exp2', {}).values()
         if isinstance(v, dict) and 'improvement' in v), default=0
    ),
    'exp3_zscore_gate': max(
        (v.get('improvement', 0) for v in results.get('exp3', {}).values()
         if isinstance(v, dict) and 'improvement' in v), default=0
    ),
    'exp4_combined':    results.get('exp4', {}).get('improvement', 0),
    'exp5_ranking_obj': results.get('exp5', {}).get('improvement', 0),
}

print(f'Baseline IC: {baseline_ic:.4f}')
print()
print(f'{"Experiment":<30} {"IC Improvement":>15}  Adopt?')
print('-' * 55)
for name, imp in improvements.items():
    adopt = 'YES' if imp > 0.005 else 'NO'
    print(f'{name:<30} {imp:>+15.4f}  {adopt}')

combined_improvement = improvements['exp4_combined']
final_ic = baseline_ic + combined_improvement

print()
print(f'Combined IC estimate:  {final_ic:.4f}')
print()

if final_ic >= 0.10:
    recommendation = 'STRONG — IC >= 0.10 achieved'
    action = 'Incorporate adopted experiments into pipeline/01_feature_builder.py'
elif final_ic > baseline_ic + 0.005:
    recommendation = 'IMPROVED — meaningful improvement'
    action = 'Incorporate adopted experiments into pipeline/01_feature_builder.py'
else:
    recommendation = 'NEUTRAL/MARGINAL — improvements below threshold'
    action = 'Keep current architecture — Phase 3 proceeds with IC=0.093'

print(f'RECOMMENDATION: {recommendation}')
print(f'ACTION:         {action}')

final_config = {
    'baseline_ic': baseline_ic,
    'final_ic': round(final_ic, 6),
    'improvements': {k: round(v, 6) for k, v in improvements.items()},
    'adopted': {k: bool(v > 0.005) for k, v in improvements.items()},
    'recommendation': recommendation,
    'action': action,
    'phase3_ready': True,
}
with open('experiments/ic_improvements/phase3_architecture.json', 'w') as f:
    json.dump(final_config, f, indent=2)
print('\nSaved: experiments/ic_improvements/phase3_architecture.json')
