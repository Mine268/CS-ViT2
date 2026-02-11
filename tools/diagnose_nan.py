"""
诊断训练 NaN 问题的脚本

分析可能的原因：
1. norm_by_hand 中的除零问题
2. 梯度爆炸
3. 数据异常
"""

import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model.loss import BundleLoss2
from src.constant import HAND_JOINTS_ORDER

print("=" * 80)
print("NaN 诊断分析")
print("=" * 80)

# 模拟问题场景
print("\n1. norm_by_hand 除零问题分析")
print("-" * 80)

# 读取 norm_stats.npz
norm_stats = np.load('model/smplx_models/mano/norm_stats.npz')
norm_list = norm_stats["norm_list"].flatten().tolist()
norm_idx = [HAND_JOINTS_ORDER.index(x) for x in norm_list]

print(f"norm_list: {norm_list}")
print(f"norm_idx: {norm_idx}")

# 不需要创建 loss 实例，直接测试除法操作

# 测试场景1：正常的 norm_scale
print("\n场景1: 正常的 norm_scale (70mm)")
norm_scale_gt_normal = torch.tensor([[70.0]], dtype=torch.float32)
trans_gt_normal = torch.tensor([[[100.0, 200.0, 800.0]]], dtype=torch.float32)

trans_gt_normalized_normal = trans_gt_normal / norm_scale_gt_normal[..., None]
print(f"  norm_scale_gt: {norm_scale_gt_normal.item():.4f} mm")
print(f"  trans_gt (原始): {trans_gt_normal[0, 0].numpy()}")
print(f"  trans_gt (归一化): {trans_gt_normalized_normal[0, 0].numpy()}")
print(f"  是否有NaN: {torch.isnan(trans_gt_normalized_normal).any().item()}")

# 测试场景2：非常小的 norm_scale
print("\n场景2: 非常小的 norm_scale (1e-6 mm)")
norm_scale_gt_small = torch.tensor([[1e-6]], dtype=torch.float32)
trans_gt_small = torch.tensor([[[100.0, 200.0, 800.0]]], dtype=torch.float32)

trans_gt_normalized_small = trans_gt_small / norm_scale_gt_small[..., None]
print(f"  norm_scale_gt: {norm_scale_gt_small.item():.10f} mm")
print(f"  trans_gt (原始): {trans_gt_small[0, 0].numpy()}")
print(f"  trans_gt (归一化): {trans_gt_normalized_small[0, 0].numpy()}")
print(f"  是否有NaN: {torch.isnan(trans_gt_normalized_small).any().item()}")
print(f"  是否有Inf: {torch.isinf(trans_gt_normalized_small).any().item()}")
print(f"  最大值: {trans_gt_normalized_small.abs().max().item():.2e}")

# 测试场景3：零 norm_scale
print("\n场景3: 零 norm_scale (0.0 mm)")
norm_scale_gt_zero = torch.tensor([[0.0]], dtype=torch.float32)
trans_gt_zero = torch.tensor([[[100.0, 200.0, 800.0]]], dtype=torch.float32)

trans_gt_normalized_zero = trans_gt_zero / norm_scale_gt_zero[..., None]
print(f"  norm_scale_gt: {norm_scale_gt_zero.item():.4f} mm")
print(f"  trans_gt (原始): {trans_gt_zero[0, 0].numpy()}")
print(f"  trans_gt (归一化): {trans_gt_normalized_zero[0, 0].numpy()}")
print(f"  是否有NaN: {torch.isnan(trans_gt_normalized_zero).any().item()}")
print(f"  是否有Inf: {torch.isinf(trans_gt_normalized_zero).any().item()}")

# 测试场景4：负数 norm_scale（理论上不应该出现）
print("\n场景4: 负数 norm_scale (-10.0 mm)")
norm_scale_gt_neg = torch.tensor([[-10.0]], dtype=torch.float32)
trans_gt_neg = torch.tensor([[[100.0, 200.0, 800.0]]], dtype=torch.float32)

trans_gt_normalized_neg = trans_gt_neg / norm_scale_gt_neg[..., None]
print(f"  norm_scale_gt: {norm_scale_gt_neg.item():.4f} mm")
print(f"  trans_gt (原始): {trans_gt_neg[0, 0].numpy()}")
print(f"  trans_gt (归一化): {trans_gt_normalized_neg[0, 0].numpy()}")
print(f"  是否有NaN: {torch.isnan(trans_gt_normalized_neg).any().item()}")

# 分析代码中的问题
print("\n" + "=" * 80)
print("2. 代码问题定位")
print("=" * 80)

print("\n问题代码位置: src/model/loss.py:454")
print("  代码: trans_gt = trans_gt / norm_scale_gt[..., None]")
print("  问题: 缺少 epsilon 保护，当 norm_scale_gt 为 0 或非常小时会产生 Inf/NaN")

print("\n问题代码位置: src/model/loss.py:480")
print("  代码: trans_pred_scaled = trans_pred * norm_scale_gt[..., None]")
print("  问题: 如果 trans_pred 是 NaN（从 line 454 传播），这里会继续传播 NaN")

# 分析可能的数据问题
print("\n" + "=" * 80)
print("3. 可能的数据问题")
print("=" * 80)

print("\nnorm_scale 可能为 0 或异常小的情况：")
print("  1. GT 关节标注错误（所有 norm_idx 关节重合）")
print("  2. GT 关节缺失（joint_valid 为 0，但仍然计算了 norm_scale）")
print("  3. 数据预处理错误（关节坐标单位错误）")
print("  4. 数值精度问题（bf16 下溢）")

# 分析训练日志
print("\n" + "=" * 80)
print("4. 训练日志分析")
print("=" * 80)

print("\n从日志 checkpoint/2026-02-11/16-26-11_stage1-dino_large/log.txt：")
print("  Step 1060: 正常")
print("    total=2.0351")
print("    loss_trans=15.3273")
print("    loss_joint_img=53.7714")
print("    lr=8.4800e-05")
print("")
print("  Step 1070: 全部 NaN")
print("    total=nan")
print("    所有 loss 都是 nan")
print("    lr=8.5600e-05 (warmup 阶段)")
print("")
print("  分析: NaN 突然出现，所有 loss 同时变为 NaN")
print("  推断: 某个 batch 中出现了 norm_scale_gt ≈ 0 的样本")

# 修复建议
print("\n" + "=" * 80)
print("5. 修复建议")
print("=" * 80)

epsilon = 1e-6
print(f"\n建议1: 添加 epsilon 保护（epsilon={epsilon}）")
print("  修改 src/model/loss.py:454")
print(f"  从: trans_gt = trans_gt / norm_scale_gt[..., None]")
print(f"  到: trans_gt = trans_gt / (norm_scale_gt[..., None] + {epsilon})")

print(f"\n建议2: 添加 epsilon 保护到反归一化")
print("  修改 src/model/loss.py:480")
print(f"  从: trans_pred_scaled = trans_pred * norm_scale_gt[..., None]")
print(f"  到: trans_pred_scaled = trans_pred * (norm_scale_gt[..., None] + {epsilon})")
print("  注: 这个修改影响较小，但为了对称性建议保持一致")

print(f"\n建议3: 在 get_hand_norm_scale 返回时添加保护")
print("  修改 src/model/loss.py:392")
print("  添加: d = torch.clamp(d, min=1e-6)  # 防止 norm_scale 过小")

print("\n建议4: 添加数据验证")
print("  在训练循环中添加:")
print("  if torch.isnan(loss).any():")
print("      print(f'NaN detected at step {step}')")
print("      print(f'norm_scale_gt: {norm_scale_gt}')")
print("      print(f'norm_valid_gt: {norm_valid_gt}')")
print("      raise RuntimeError('NaN in loss')")

print("\n建议5: 检查数据集")
print("  运行数据检查脚本，找出 norm_scale 异常的样本：")
print("  - norm_scale < 1mm (异常小)")
print("  - norm_scale > 200mm (异常大)")
print("  - norm_scale = 0 (完全错误)")

# 测试修复后的效果
print("\n" + "=" * 80)
print("6. 修复效果测试")
print("=" * 80)

epsilon = 1e-6
print(f"\n使用 epsilon={epsilon} 的保护:")

print("\n场景: norm_scale_gt = 0")
norm_scale_gt_zero = torch.tensor([[0.0]], dtype=torch.float32)
trans_gt_zero = torch.tensor([[[100.0, 200.0, 800.0]]], dtype=torch.float32)

# 原始方法（会产生 Inf/NaN）
trans_normalized_old = trans_gt_zero / norm_scale_gt_zero[..., None]
print(f"  原始方法: {trans_normalized_old[0, 0].numpy()}")
print(f"  是否有NaN/Inf: {torch.isnan(trans_normalized_old).any().item() or torch.isinf(trans_normalized_old).any().item()}")

# 修复后（加 epsilon）
trans_normalized_new = trans_gt_zero / (norm_scale_gt_zero[..., None] + epsilon)
print(f"  修复方法: {trans_normalized_new[0, 0].numpy()}")
print(f"  是否有NaN/Inf: {torch.isnan(trans_normalized_new).any().item() or torch.isinf(trans_normalized_new).any().item()}")
print(f"  值是否合理: 是 (除以一个很小的数，得到很大的值，但仍然是有限的)")

print("\n场景: norm_scale_gt = 1e-8")
norm_scale_gt_tiny = torch.tensor([[1e-8]], dtype=torch.float32)
trans_gt_tiny = torch.tensor([[[100.0, 200.0, 800.0]]], dtype=torch.float32)

# 原始方法
trans_normalized_old = trans_gt_tiny / norm_scale_gt_tiny[..., None]
print(f"  原始方法: {trans_normalized_old[0, 0].numpy()}")
print(f"  最大值: {trans_normalized_old.abs().max().item():.2e}")

# 修复后
trans_normalized_new = trans_gt_tiny / (norm_scale_gt_tiny[..., None] + epsilon)
print(f"  修复方法: {trans_normalized_new[0, 0].numpy()}")
print(f"  最大值: {trans_normalized_new.abs().max().item():.2e}")

print("\n" + "=" * 80)
print("总结")
print("=" * 80)

print("\n根本原因:")
print("  在 norm_by_hand=true 时，trans_gt 直接除以 norm_scale_gt")
print("  如果 norm_scale_gt 为 0 或非常小（数据异常或标注错误）")
print("  会产生 Inf/NaN，进而传播到所有 loss")

print("\n立即修复:")
print("  在 src/model/loss.py:454 和 480 添加 epsilon 保护")
print(f"  epsilon 建议值: 1e-6 (对于 mm 单位的坐标)")

print("\n后续改进:")
print("  1. 添加数据验证，过滤 norm_scale 异常的样本")
print("  2. 在训练循环中添加 NaN 检测和调试信息")
print("  3. 检查数据集质量，找出标注错误的样本")

print("\n" + "=" * 80)
print("诊断完成")
print("=" * 80)
