from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_repack_stats(root: Path, split: str, clip_dir_name: str, dataset_names: Optional[Sequence[str]] = None):
    dataset_filter = set(dataset_names) if dataset_names is not None else None
    stats = {}
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if dataset_filter is not None and dataset_dir.name not in dataset_filter:
            continue
        if dataset_dir.name.endswith('_smoke'):
            continue
        path = dataset_dir / split / clip_dir_name / 'repack_stats.json'
        if not path.exists():
            continue
        stats[dataset_dir.name] = json.loads(path.read_text())
    return stats


def aggregate_bin_counts(per_dataset_stats: Dict[str, List[Dict]]):
    per_dataset_counts = {}
    aggregate = defaultdict(int)
    for dataset_name, stats in per_dataset_stats.items():
        counts = {}
        for item in stats:
            sample_count = int(item.get('sample_count', 0))
            if sample_count <= 0:
                continue
            counts[item['bin']] = sample_count
            aggregate[item['bin']] += sample_count
        per_dataset_counts[dataset_name] = counts
    return per_dataset_counts, dict(sorted(aggregate.items(), key=lambda item: item[0]))


def compute_uniform_bin_weights(bin_counts: Dict[str, int]):
    non_empty_bins = [bin_name for bin_name, count in bin_counts.items() if count > 0]
    if len(non_empty_bins) == 0:
        raise ValueError('No non-empty bins found')
    weight = 1.0 / len(non_empty_bins)
    return {bin_name: (weight if bin_name in non_empty_bins else 0.0) for bin_name in bin_counts.keys()}


def save_bar_chart(per_dataset_counts: Dict[str, Dict[str, int]], aggregate_counts: Dict[str, int], output_path: Path):
    bins = list(aggregate_counts.keys())
    datasets = list(per_dataset_counts.keys())
    x = list(range(len(bins)))

    plt.figure(figsize=(12, 7))
    bottoms = [0] * len(bins)
    for dataset_name in datasets:
        values = [per_dataset_counts[dataset_name].get(bin_name, 0) for bin_name in bins]
        plt.bar(x, values, bottom=bottoms, label=dataset_name)
        bottoms = [b + v for b, v in zip(bottoms, values)]

    for idx, total in enumerate(bottoms):
        plt.text(idx, total, f'{total:,}', ha='center', va='bottom', fontsize=9, rotation=0)

    plt.xticks(x, bins, rotation=25, ha='right')
    plt.ylabel('Sample Count (clips)')
    plt.xlabel('Depth Bin')
    plt.title('Depth Bin Sample Counts (stacked by dataset)')
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_argparser():
    parser = argparse.ArgumentParser(description='统计 repack 后的 depth-bin 样本数并生成均衡采样权重')
    parser.add_argument('--root', required=True)
    parser.add_argument('--split', default='train')
    parser.add_argument('--clip-dir-name', default='nf1_s1')
    parser.add_argument('--datasets', nargs='*', default=None)
    parser.add_argument('--output-prefix', default='depth_bin_stats')
    return parser


def main():
    args = build_argparser().parse_args()
    root = Path(args.root).resolve()
    per_dataset_stats = load_repack_stats(root, args.split, args.clip_dir_name, args.datasets)
    if len(per_dataset_stats) == 0:
        raise FileNotFoundError(f'No repack_stats.json found under {root}')

    per_dataset_counts, aggregate_counts = aggregate_bin_counts(per_dataset_stats)
    uniform_weights = compute_uniform_bin_weights(aggregate_counts)
    total = sum(aggregate_counts.values())
    natural_prob = {
        bin_name: (count / total if total > 0 else 0.0)
        for bin_name, count in aggregate_counts.items()
    }

    payload = {
        'root': str(root),
        'split': args.split,
        'clip_dir_name': args.clip_dir_name,
        'datasets': list(per_dataset_counts.keys()),
        'bin_order': list(aggregate_counts.keys()),
        'aggregate_counts': aggregate_counts,
        'aggregate_fraction': natural_prob,
        'per_dataset_counts': per_dataset_counts,
        'recommended_strategy': 'uniform_bin',
        'recommended_bin_weights_by_bin': uniform_weights,
        'recommended_bin_weights_list': [uniform_weights[bin_name] for bin_name in aggregate_counts.keys()],
        'note': '当前 depth-bin dataloader 以 bin 为混采单元。若目标是不同深度桶均匀采样，推荐直接使用等权 bin_weights。',
    }

    json_path = root / f'{args.output_prefix}.json'
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)

    chart_path = root / 'depth_bin_sample_counts.png'
    save_bar_chart(per_dataset_counts, aggregate_counts, chart_path)

    print(json.dumps(payload, indent=2))
    print(f'saved_json={json_path}')
    print(f'saved_chart={chart_path}')


if __name__ == '__main__':
    main()
