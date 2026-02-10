#!/usr/bin/env python3
"""
评估脚本：从 HDF5 文件计算评估指标

用法：
    python script/evaluate.py path/to/predictions.h5
    python script/evaluate.py path/to/predictions.h5 --output metrics.json
"""

import argparse
import json
import sys
import os.path as osp
from typing import Dict

import numpy as np
import h5py
from rich.console import Console
from rich.table import Table


console = Console()


def compute_metrics(h5_path: str) -> Dict[str, float]:
    """
    从 HDF5 文件计算评估指标

    Args:
        h5_path: HDF5 文件路径

    Returns:
        metrics: 指标字典，包含：
            - mpjpe: Mean Per-Joint Position Error (mm)
            - mpvpe: Mean Per-Vertex Position Error (mm)
            - rel_mpjpe: Root-relative MPJPE (mm)
            - rel_mpvpe: Root-relative MPVPE (mm)
            - num_samples: 样本数量
            - num_valid_joints: 有效关节数量
            - num_valid_hands: 有效手部数量
    """
    console.print(f"[cyan]Loading predictions from:[/cyan] {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        # 读取数据
        joint_cam_pred = f['samples/joint_cam_pred'][:]  # [N, 21, 3]
        joint_cam_gt = f['samples/joint_cam_gt'][:]      # [N, 21, 3]
        vert_cam_pred = f['samples/vert_cam_pred'][:]    # [N, 778, 3]
        vert_cam_gt = f['samples/vert_cam_gt'][:]        # [N, 778, 3]
        joint_valid = f['samples/joint_valid'][:]        # [N, 21]
        mano_valid = f['samples/mano_valid'][:]          # [N]

        # 读取元数据
        num_samples = f['metadata'].attrs['num_samples']
        norm_by_hand = f['metadata'].attrs.get('norm_by_hand', False)

    console.print(f"[green]✓[/green] Loaded {num_samples} samples")
    console.print(f"[dim]  norm_by_hand: {norm_by_hand}[/dim]")

    # 1. 计算 MPJPE (绝对坐标)
    joint_diff = np.linalg.norm(joint_cam_pred - joint_cam_gt, axis=-1)  # [N, 21]
    joint_diff_masked = joint_diff * joint_valid  # 应用 mask
    mpjpe = np.sum(joint_diff_masked) / np.sum(joint_valid)

    # 2. 计算 MPVPE (绝对坐标)
    vert_diff = np.linalg.norm(vert_cam_pred - vert_cam_gt, axis=-1)  # [N, 778]
    vert_diff_masked = vert_diff * mano_valid[:, None]  # 应用 mask
    mpvpe = np.sum(vert_diff_masked) / (np.sum(mano_valid) * 778)

    # 3. 计算 rel-MPJPE (相对于 root joint，即 joint_0)
    # 相对坐标：所有 joint 减去 root joint
    joint_rel_pred = joint_cam_pred - joint_cam_pred[:, :1, :]  # [N, 21, 3]
    joint_rel_gt = joint_cam_gt - joint_cam_gt[:, :1, :]        # [N, 21, 3]

    joint_rel_diff = np.linalg.norm(joint_rel_pred - joint_rel_gt, axis=-1)  # [N, 21]
    joint_rel_diff_masked = joint_rel_diff * joint_valid
    rel_mpjpe = np.sum(joint_rel_diff_masked) / np.sum(joint_valid)

    # 4. 计算 rel-MPVPE (相对于 root joint)
    # 使用 joint_cam[:, 0] 作为 root 位置
    root_pred = joint_cam_pred[:, :1, :]  # [N, 1, 3]
    root_gt = joint_cam_gt[:, :1, :]      # [N, 1, 3]

    vert_rel_pred = vert_cam_pred - root_pred  # [N, 778, 3]
    vert_rel_gt = vert_cam_gt - root_gt        # [N, 778, 3]

    vert_rel_diff = np.linalg.norm(vert_rel_pred - vert_rel_gt, axis=-1)  # [N, 778]
    vert_rel_diff_masked = vert_rel_diff * mano_valid[:, None]
    rel_mpvpe = np.sum(vert_rel_diff_masked) / (np.sum(mano_valid) * 778)

    # 5. 统计信息
    num_valid_joints = int(np.sum(joint_valid))
    num_valid_hands = int(np.sum(mano_valid))

    metrics = {
        "mpjpe": float(mpjpe),
        "mpvpe": float(mpvpe),
        "rel_mpjpe": float(rel_mpjpe),
        "rel_mpvpe": float(rel_mpvpe),
        "num_samples": int(num_samples),
        "num_valid_joints": num_valid_joints,
        "num_valid_hands": num_valid_hands,
        "norm_by_hand": bool(norm_by_hand),
    }

    return metrics


def print_metrics(metrics: Dict[str, float]):
    """
    打印指标表格

    Args:
        metrics: 指标字典
    """
    table = Table(title="Evaluation Metrics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="green", justify="right", width=15)

    # 主要指标
    table.add_row("MPJPE", f"{metrics['mpjpe']:.2f} mm")
    table.add_row("MPVPE", f"{metrics['mpvpe']:.2f} mm")
    table.add_row("rel-MPJPE", f"{metrics['rel_mpjpe']:.2f} mm")
    table.add_row("rel-MPVPE", f"{metrics['rel_mpvpe']:.2f} mm")

    # 分隔线
    table.add_section()

    # 统计信息
    table.add_row("Num Samples", str(metrics['num_samples']))
    table.add_row("Valid Joints", str(metrics['num_valid_joints']))
    table.add_row("Valid Hands", str(metrics['num_valid_hands']))
    table.add_row("norm_by_hand", str(metrics['norm_by_hand']))

    console.print()
    console.print(table)
    console.print()


def save_metrics(metrics: Dict[str, float], output_path: str):
    """
    保存指标到 JSON 文件

    Args:
        metrics: 指标字典
        output_path: 输出文件路径
    """
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    console.print(f"[green]✓[/green] Saved metrics to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate hand pose estimation from HDF5 predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python script/evaluate.py checkpoint/exp1/test_results/predictions.h5

  # Save metrics to custom path
  python script/evaluate.py predictions.h5 --output my_metrics.json

  # Quiet mode (only show final metrics)
  python script/evaluate.py predictions.h5 --quiet
        """
    )

    parser.add_argument(
        "h5_path",
        type=str,
        help="Path to HDF5 predictions file (e.g., predictions.h5)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: save to same directory as h5 file)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode: only print final metrics"
    )

    args = parser.parse_args()

    # 检查输入文件
    if not osp.exists(args.h5_path):
        console.print(f"[red]✗[/red] File not found: {args.h5_path}", style="bold red")
        sys.exit(1)

    if not args.h5_path.endswith('.h5'):
        console.print(f"[yellow]⚠[/yellow] Warning: Expected .h5 file, got: {args.h5_path}")

    # 设置输出路径
    if args.output is None:
        # 默认保存到同一目录
        h5_dir = osp.dirname(args.h5_path)
        args.output = osp.join(h5_dir, "eval_metrics.json")

    # 计算指标
    try:
        metrics = compute_metrics(args.h5_path)
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to compute metrics: {e}", style="bold red")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # 打印结果
    if not args.quiet:
        print_metrics(metrics)
    else:
        console.print(f"MPJPE: {metrics['mpjpe']:.2f} mm, "
                     f"MPVPE: {metrics['mpvpe']:.2f} mm, "
                     f"rel-MPJPE: {metrics['rel_mpjpe']:.2f} mm, "
                     f"rel-MPVPE: {metrics['rel_mpvpe']:.2f} mm")

    # 保存结果
    try:
        save_metrics(metrics, args.output)
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to save metrics: {e}", style="bold red")
        sys.exit(1)

    console.print("[green]✓[/green] Evaluation completed successfully!")


if __name__ == "__main__":
    main()
