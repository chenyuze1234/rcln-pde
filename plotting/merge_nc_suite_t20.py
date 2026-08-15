# 合并 T20 基线 nc_suite K=200 结果进主 JSON（同协议同 IC，直接 dict.update）
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / 'results/paper_experiments/nc_suite_k200.json'
NEW = ROOT / 'results/paper_experiments/nc_suite_t20_baselines_k200.json'

main = json.load(open(MAIN, encoding='utf-8'))
new = json.load(open(NEW, encoding='utf-8'))

assert main['K'] == new['K'] == 200, 'K mismatch'
assert main['num_ic'] == new['num_ic'], 'num_ic mismatch'

overlap = set(main['models']) & set(new['models'])
assert not overlap, f'unexpected overlap: {overlap}'

main['models'].update(new['models'])
json.dump(main, open(MAIN, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('merged models:', sorted(main['models'].keys()))
