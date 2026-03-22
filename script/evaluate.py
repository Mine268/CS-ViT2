#!/usr/bin/env python3
"""
评估脚本：从 HDF5 文件计算评估指标

用法：
    python script/evaluate.py path/to/predictions.h5
    python script/evaluate.py path/to/predictions.h5 --output metrics.json
"""

import argparse
import json
import os.path as osp
from typing import Dict

import h5py
import numpy as np
from rich.console import Console
from rich.table import Table

from src.utils.metric import build_excluded_data_source_mask


console = Console()


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def compute_metrics(h5_path: str) -> Dict[str, float]:
    """
    从 HDF5 文件计算评估指标
    """
    console.print(f"[cyan]Loading predictions from:[/cyan] {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        joint_cam_pred = f['samples/joint_cam_pred'][:]
        joint_cam_gt = f['samples/joint_cam_gt'][:]
        vert_cam_pred = f['samples/vert_cam_pred'][:]
        vert_cam_gt = f['samples/vert_cam_gt'][:]
        joint_3d_valid = f['samples/joint_3d_valid'][:]
        has_mano = f['samples/has_mano'][:]
        data_source = (
            f['samples/data_source'][:]
            if 'data_source' in f['samples']
            else None
        )

        num_samples = f['metadata'].attrs['num_samples']
        norm_by_hand = f['metadata'].attrs.get('norm_by_hand', False)

    console.print(f"[green]✓[/green] Loaded {num_samples} samples")
    console.print(f"[dim]  norm_by_hand: {norm_by_hand}[/dim]")

    keep_mask = build_excluded_data_source_mask(data_source)
    excluded_coco_samples = 0
    if keep_mask is not None:
        excluded_coco_samples = int(len(keep_mask) - np.sum(keep_mask))
        joint_cam_pred = joint_cam_pred[keep_mask]
        joint_cam_gt = joint_cam_gt[keep_mask]
        vert_cam_pred = vert_cam_pred[keep_mask]
        vert_cam_gt = vert_cam_gt[keep_mask]
        joint_3d_valid = joint_3d_valid[keep_mask]
        has_mano = has_mano[keep_mask]

    joint_valid_count = float(np.sum(joint_3d_valid))
    mano_valid_count = float(np.sum(has_mano))

    joint_diff = np.linalg.norm(joint_cam_pred - joint_cam_gt, axis=-1)
    joint_diff_masked = joint_diff * joint_3d_valid
    mpjpe = _safe_divide(np.sum(joint_diff_masked), joint_valid_count)

    vert_diff = np.linalg.norm(vert_cam_pred - vert_cam_gt, axis=-1)
    vert_diff_masked = vert_diff * has_mano[:, None]
    mpvpe = _safe_divide(np.sum(vert_diff_masked), mano_valid_count * 778.0)

    joint_rel_pred = joint_cam_pred - joint_cam_pred[:, :1, :]
    joint_rel_gt = joint_cam_gt - joint_cam_gt[:, :1, :]
    joint_rel_diff = np.linalg.norm(joint_rel_pred - joint_rel_gt, axis=-1)
    joint_rel_diff_masked = joint_rel_diff * joint_3d_valid
    rel_mpjpe = _safe_divide(np.sum(joint_rel_diff_masked), joint_valid_count)

    root_pred = joint_cam_pred[:, :1, :]
    root_gt = joint_cam_gt[:, :1, :]
    vert_rel_pred = vert_cam_pred - root_pred
    vert_rel_gt = vert_cam_gt - root_gt
    vert_rel_diff = np.linalg.norm(vert_rel_pred - vert_rel_gt, axis=-1)
    vert_rel_diff_masked = vert_rel_diff * has_mano[:, None]
    rel_mpvpe = _safe_divide(np.sum(vert_rel_diff_masked), mano_valid_count * 778.0)

    metrics = {
        "mpjpe": mpjpe,
        "mpvpe": mpvpe,
        "rel_mpjpe": rel_mpjpe,
        "rel_mpvpe": rel_mpvpe,
        "num_samples": int(joint_cam_pred.shape[0]),
        "num_samples_total": int(num_samples),
        "num_excluded_coco_wholebody": excluded_coco_samples,
        "num_valid_joints": int(joint_valid_count),
        "num_valid_hands": int(mano_valid_count),
        "norm_by_hand": bool(norm_by_hand),
    }

    return metrics


def print_metrics(metrics: Dict[str, float]):
    table = Table(title="Evaluation Metrics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="green", justify="right", width=15)

    table.add_row("MPJPE", f"{metrics['mpjpe']:.2f} mm" if not np.isnan(metrics['mpjpe']) else "nan")
    table.add_row("MPVPE", f"{metrics['mpvpe']:.2f} mm" if not np.isnan(metrics['mpvpe']) else "nan")
    table.add_row("rel-MPJPE", f"{metrics['rel_mpjpe']:.2f} mm" if not np.isnan(metrics['rel_mpjpe']) else "nan")
    table.add_row("rel-MPVPE", f"{metrics['rel_mpvpe']:.2f} mm" if not np.isnan(metrics['rel_mpvpe']) else "nan")

    table.add_section()
    table.add_row("Num Samples", str(metrics['num_samples']))
    table.add_row("Total Samples", str(metrics.get('num_samples_total', metrics['num_samples'])))
    table.add_row("Excluded COCO", str(metrics.get('num_excluded_coco_wholebody', 0)))
    table.add_row("Valid Joints", str(metrics['num_valid_joints']))
    table.add_row("Valid Hands", str(metrics['num_valid_hands']))
    table.add_row("norm_by_hand", str(metrics['norm_by_hand']))

    console.print()
    console.print(table)
    console.print()


def save_metrics(metrics: Dict[str, float], output_path: str):
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    console.print(f"[green]✓[/green] Saved metrics to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate hand pose estimation from HDF5 predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script/evaluate.py checkpoint/exp1/test_results/predictions.h5
  python script/evaluate.py predictions.h5 --output my_metrics.json
  python script/evaluate.py predictions.h5 --quiet
        """,
    )

    parser.add_argument(
        "h5_path",
        type=str,
        help="Path to HDF5 predictions file (e.g., predictions.h5)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: save to same directory as h5 file)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress table output and only save metrics",
    )

    args = parser.parse_args()

    metrics = compute_metrics(args.h5_path)

    if not args.quiet:
        print_metrics(metrics)

    output_path = args.output
    if output_path is None:
        output_path = osp.join(osp.dirname(args.h5_path), "metrics_eval.json")
    save_metrics(metrics, output_path)


if __name__ == "__main__":
    main()
