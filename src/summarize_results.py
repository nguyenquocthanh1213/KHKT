from pathlib import Path
import csv, json, statistics

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / 'runs'
OUT = ROOT / 'results'
EXPECTED_SEEDS = (42, 123, 456)
CONFIGS = ('fedavg', 'fedprox', 'fedprox_dp', 'fedprox_dp_autoclip')
METRICS = ('accuracy', 'macro_f1', 'balanced_accuracy')


def main():
    rows = []
    missing = []
    for config in CONFIGS:
        for seed in EXPECTED_SEEDS:
            path = RUNS / f'p4_linear_{config}_seed{seed}' / 'eval' / 'eval_test.json'
            if not path.exists():
                missing.append(f'{config}/seed{seed}')
                continue
            payload = json.loads(path.read_text(encoding='utf-8'))
            result = next((r for r in payload['results'] if r.get('kind') == 'server'), None)
            if result is None:
                raise ValueError(f'No selected server result in {path}')
            rows.append({'config': config, 'seed': seed, **result})
    if missing:
        raise SystemExit('Incomplete multi-seed results; refusing to write misleading mean/std. Missing: ' + ', '.join(missing))
    OUT.mkdir(exist_ok=True)
    with (OUT / 'linear_multiseed_raw.csv').open('w', newline='', encoding='utf-8') as f:
        fields = ['config', 'seed', 'round', *METRICS, 'per_class_recall']
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow({**row, 'round': row.get('round', ''), 'per_class_recall': json.dumps(row.get('per_class_recall', []))})
    with (OUT / 'linear_multiseed_summary.csv').open('w', newline='', encoding='utf-8') as f:
        fields = ['config', 'n_seeds'] + [f'{m}_{stat}' for m in METRICS for stat in ('mean', 'std')]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for config in CONFIGS:
            subset = [r for r in rows if r['config'] == config]
            row = {'config': config, 'n_seeds': len(subset)}
            for metric in METRICS:
                values = [float(r[metric]) for r in subset]
                row[f'{metric}_mean'] = statistics.mean(values)
                row[f'{metric}_std'] = statistics.stdev(values)  # sample std, ddof=1
            w.writerow(row)
    print(f'Wrote seed-level rows and sample mean±std for {len(CONFIGS)} configs × {len(EXPECTED_SEEDS)} seeds to {OUT}')


if __name__ == '__main__':
    main()
