"""
检查数据集中根关节位置是否超出heatmap范围

爹，这个脚本用于验证在当前配置下（norm_by_hand=true），
数据集中归一化后的根关节位置是否会超出heatmap定义的范围。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from pathlib import Path
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import glob

from src.data.dataloader import get_dataloader
from src.constant import NORM_STAT_NPZ


def get_hand_norm_scale(j3d: torch.Tensor, valid: torch.Tensor, norm_idx):
    """
    计算手部归一化规模（中指长度）

    Args:
        j3d: [B, T, J, 3] - 关节坐标
        valid: [B, T, J] - 有效性掩码
        norm_idx: 中指关节索引列表

    Returns:
        scale: [B, T] - 中指长度
        flag: [B, T] - 有效性标志
    """
    # 计算中指各关节间的距离
    d = j3d[:, :, norm_idx[:-1], :] - j3d[:, :, norm_idx[1:], :]
    d = torch.sum(torch.sqrt(torch.sum(d ** 2, dim=-1)), dim=-1)  # [B, T]

    # 检查所有norm关节是否有效
    flag = torch.all(valid[:, :, norm_idx] > 0.5, dim=-1).float()  # [B, T]

    return d, flag


def load_heatmap_ranges(norm_stats_path: str):
    """加载heatmap的理论范围"""
    norm_stats = np.load(norm_stats_path)
    norm_mean = norm_stats["norm_mean"]
    norm_std = norm_stats["norm_std"]
    norm_list = norm_stats["norm_list"].tolist()[0]

    # Heatmap范围 = [mean - 4*std, mean + 4*std]
    x_range = [norm_mean[0] - norm_std[0] * 4, norm_mean[0] + norm_std[0] * 4]
    y_range = [norm_mean[1] - norm_std[1] * 4, norm_mean[1] + norm_std[1] * 4]
    z_range = [norm_mean[2] - norm_std[2] * 4, norm_mean[2] + norm_std[2] * 4]

    print("=" * 60)
    print("Heatmap理论范围 (基于 norm_stats.npz)")
    print("=" * 60)
    print(f"norm_mean: {norm_mean}")
    print(f"norm_std:  {norm_std}")
    print(f"norm_list: {norm_list}")
    print()
    print(f"X轴范围: [{x_range[0]:.3f}, {x_range[1]:.3f}]  (宽度: {x_range[1]-x_range[0]:.3f})")
    print(f"Y轴范围: [{y_range[0]:.3f}, {y_range[1]:.3f}]  (宽度: {y_range[1]-y_range[0]:.3f})")
    print(f"Z轴范围: [{z_range[0]:.3f}, {z_range[1]:.3f}]  (宽度: {z_range[1]-z_range[0]:.3f})")
    print("=" * 60)
    print()

    return {
        "x_range": x_range,
        "y_range": y_range,
        "z_range": z_range,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "norm_list": norm_list
    }


def get_norm_idx_from_list(norm_list):
    """从norm_list获取关节索引"""
    from src.constant import HAND_JOINTS_ORDER
    norm_idx = [HAND_JOINTS_ORDER.index(joint_name) for joint_name in norm_list]
    return norm_idx


@hydra.main(version_base=None, config_path="../config", config_name="stage1-dino_large")
def main(cfg: DictConfig):
    # 1. 加载heatmap范围
    hm_ranges = load_heatmap_ranges(NORM_STAT_NPZ)
    norm_idx = get_norm_idx_from_list(hm_ranges["norm_list"])
    print(f"中指关节索引: {norm_idx}")
    print()

    # 2. 构建数据加载器
    print("正在构建数据加载器...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 展开训练集路径
    train_sources = []
    for src in cfg.DATA.train.source:
        matched_files = glob.glob(src)
        matched_files = sorted(matched_files)
        train_sources.extend(matched_files)

    train_loader = get_dataloader(
        url=train_sources,
        num_frames=cfg.MODEL.num_frame,
        stride=cfg.DATA.train.stride,
        batch_size=cfg.TRAIN.sample_per_device,
        num_workers=cfg.GENERAL.num_worker,
        prefetcher_factor=cfg.GENERAL.prefetch_factor,
        infinite=True
    )

    # 展开验证集路径
    val_sources = []
    for src in cfg.DATA.val.source:
        matched_files = glob.glob(src)
        matched_files = sorted(matched_files)
        val_sources.extend(matched_files)

    val_loader = get_dataloader(
        url=val_sources,
        num_frames=cfg.MODEL.num_frame,
        stride=cfg.DATA.val.stride,
        batch_size=cfg.TRAIN.sample_per_device,
        num_workers=cfg.GENERAL.num_worker,
        prefetcher_factor=cfg.GENERAL.prefetch_factor,
        infinite=False
    )

    print(f"训练集加载器已创建")
    print(f"验证集加载器已创建")
    print()

    # 3. 采样数据并统计
    max_batches = 100  # 采样100个batch

    def analyze_dataset(dataloader, dataset_name, max_batches):
        print(f"开始分析 {dataset_name}...")

        root_positions_norm = []  # 归一化后的根关节位置
        norm_scales = []  # 手部规模
        valid_samples = 0
        total_samples = 0

        for i, batch in enumerate(tqdm(dataloader, total=max_batches, desc=f"处理 {dataset_name}")):
            if i >= max_batches:
                break

            # 提取数据
            joint_cam = batch["joint_cam"].to(device)  # [B, T, J, 3]
            joint_valid = batch["joint_valid"].to(device)  # [B, T, J]

            B, T, J, _ = joint_cam.shape
            total_samples += B * T

            # 提取根关节位置
            root_cam = joint_cam[:, :, 0, :]  # [B, T, 3]

            # 计算手部规模
            norm_scale, norm_flag = get_hand_norm_scale(joint_cam, joint_valid, norm_idx)

            # 归一化根关节位置
            root_norm = root_cam / norm_scale.unsqueeze(-1)  # [B, T, 3]

            # 只收集有效样本
            valid_mask = (norm_flag > 0.5) & (joint_valid[:, :, 0] > 0.5)  # [B, T]
            valid_samples += valid_mask.sum().item()

            # 收集数据
            for b in range(B):
                for t in range(T):
                    if valid_mask[b, t]:
                        root_positions_norm.append(root_norm[b, t].cpu().numpy())
                        norm_scales.append(norm_scale[b, t].cpu().numpy())

        root_positions_norm = np.array(root_positions_norm)  # [N, 3]
        norm_scales = np.array(norm_scales)  # [N]

        print(f"\n{dataset_name} 统计结果:")
        print(f"  总样本数: {total_samples}")
        print(f"  有效样本数: {valid_samples}")
        print(f"  有效率: {valid_samples/total_samples*100:.1f}%")
        print()

        return root_positions_norm, norm_scales, valid_samples

    # 分析训练集
    train_roots, train_scales, train_valid = analyze_dataset(train_loader, "训练集", max_batches)

    # 分析验证集
    val_roots, val_scales, val_valid = analyze_dataset(val_loader, "验证集", max_batches)

    # 合并数据
    all_roots = np.concatenate([train_roots, val_roots], axis=0)  # [N, 3]
    all_scales = np.concatenate([train_scales, val_scales], axis=0)  # [N]

    print("\n" + "=" * 60)
    print("综合统计分析")
    print("=" * 60)
    print(f"总有效样本数: {len(all_roots)}")
    print()

    # 4. 统计各维度
    for dim, dim_name in enumerate(['X', 'Y', 'Z']):
        dim_data = all_roots[:, dim]

        print(f"{dim_name}轴统计:")
        print(f"  最小值: {dim_data.min():.3f}")
        print(f"  最大值: {dim_data.max():.3f}")
        print(f"  均值:   {dim_data.mean():.3f}")
        print(f"  标准差: {dim_data.std():.3f}")
        print(f"  1%分位: {np.percentile(dim_data, 1):.3f}")
        print(f"  99%分位: {np.percentile(dim_data, 99):.3f}")
        print()

    # 5. 检查超出范围的样本
    print("=" * 60)
    print("超出Heatmap范围统计")
    print("=" * 60)

    x_min, x_max = hm_ranges["x_range"]
    y_min, y_max = hm_ranges["y_range"]
    z_min, z_max = hm_ranges["z_range"]

    out_x = (all_roots[:, 0] < x_min) | (all_roots[:, 0] > x_max)
    out_y = (all_roots[:, 1] < y_min) | (all_roots[:, 1] > y_max)
    out_z = (all_roots[:, 2] < z_min) | (all_roots[:, 2] > z_max)
    out_any = out_x | out_y | out_z

    total = len(all_roots)

    print(f"X轴超出: {out_x.sum():>6} / {total} ({out_x.sum()/total*100:>5.2f}%)")
    if out_x.sum() > 0:
        out_x_vals = all_roots[out_x, 0]
        print(f"  超出范围: [{out_x_vals.min():.3f}, {out_x_vals.max():.3f}]")
        print(f"  理论范围: [{x_min:.3f}, {x_max:.3f}]")

    print(f"Y轴超出: {out_y.sum():>6} / {total} ({out_y.sum()/total*100:>5.2f}%)")
    if out_y.sum() > 0:
        out_y_vals = all_roots[out_y, 1]
        print(f"  超出范围: [{out_y_vals.min():.3f}, {out_y_vals.max():.3f}]")
        print(f"  理论范围: [{y_min:.3f}, {y_max:.3f}]")

    print(f"Z轴超出: {out_z.sum():>6} / {total} ({out_z.sum()/total*100:>5.2f}%)")
    if out_z.sum() > 0:
        out_z_vals = all_roots[out_z, 2]
        print(f"  超出范围: [{out_z_vals.min():.3f}, {out_z_vals.max():.3f}]")
        print(f"  理论范围: [{z_min:.3f}, {z_max:.3f}]")

    print(f"\n任一维度超出: {out_any.sum():>6} / {total} ({out_any.sum()/total*100:>5.2f}%)")
    print("=" * 60)
    print()

    # 6. 给出建议
    out_ratio = out_any.sum() / total * 100

    print("=" * 60)
    print("结论与建议")
    print("=" * 60)

    if out_ratio < 1.0:
        print(f"✓ 范围内样本: {(100-out_ratio):.2f}%")
        print("建议: 当前heatmap范围设置合理，保持不变")
    elif out_ratio < 5.0:
        print(f"⚠ 范围内样本: {(100-out_ratio):.2f}%")
        print("建议: 有少量样本超出，考虑扩大到 ±4.5σ 或 ±5σ")
    else:
        print(f"✗ 范围内样本: {(100-out_ratio):.2f}%")
        print("建议: 大量样本超出，需要重新计算norm_stats或调整数据预处理")

    print()
    print("手部规模统计:")
    print(f"  最小值: {all_scales.min():.1f} mm")
    print(f"  最大值: {all_scales.max():.1f} mm")
    print(f"  均值:   {all_scales.mean():.1f} mm")
    print(f"  标准差: {all_scales.std():.1f} mm")
    print("=" * 60)


if __name__ == "__main__":
    main()
