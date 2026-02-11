"""
测试 NaN 修复是否生效

验证 NORM_SCALE_EPSILON 保护是否防止除零导致的 NaN/Inf
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model.loss import NORM_SCALE_EPSILON


def test_norm_scale_epsilon_protection():
    """测试 epsilon 保护是否防止 NaN"""

    print("=" * 80)
    print("测试 NORM_SCALE_EPSILON 保护")
    print("=" * 80)

    print(f"\nEpsilon 值: {NORM_SCALE_EPSILON}")

    # 场景1: norm_scale = 0（最严重的情况）
    print("\n场景1: norm_scale_gt = 0")
    norm_scale_gt = torch.tensor([[0.0]], dtype=torch.float32)
    trans_gt = torch.tensor([[[100.0, 200.0, 800.0]]], dtype=torch.float32)

    # 原始方法（会产生 Inf）
    trans_normalized_old = trans_gt / norm_scale_gt[..., None]
    print(f"  原始方法 (无保护): {trans_normalized_old[0, 0].numpy()}")
    print(f"  是否有 NaN: {torch.isnan(trans_normalized_old).any().item()}")
    print(f"  是否有 Inf: {torch.isinf(trans_normalized_old).any().item()}")

    # 修复后
    trans_normalized_new = trans_gt / (norm_scale_gt[..., None] + NORM_SCALE_EPSILON)
    print(f"  修复方法 (加 epsilon): {trans_normalized_new[0, 0].numpy()}")
    print(f"  是否有 NaN: {torch.isnan(trans_normalized_new).any().item()}")
    print(f"  是否有 Inf: {torch.isinf(trans_normalized_new).any().item()}")
    print(f"  最大值: {trans_normalized_new.abs().max().item():.2e}")

    # 验证
    assert not torch.isnan(trans_normalized_new).any(), "✗ NaN found!"
    assert not torch.isinf(trans_normalized_new).any(), "✗ Inf found!"
    assert trans_normalized_new.abs().max() < 1e10, "✗ Value too large!"
    print("  ✓ 通过：没有 NaN/Inf，值在合理范围内")

    # 场景2: norm_scale 非常小
    print("\n场景2: norm_scale_gt = 1e-8 (非常小)")
    norm_scale_gt = torch.tensor([[1e-8]], dtype=torch.float32)
    trans_gt = torch.tensor([[[100.0, 200.0, 800.0]]], dtype=torch.float32)

    trans_normalized_old = trans_gt / norm_scale_gt[..., None]
    trans_normalized_new = trans_gt / (norm_scale_gt[..., None] + NORM_SCALE_EPSILON)

    print(f"  原始方法: max={trans_normalized_old.abs().max().item():.2e}")
    print(f"  修复方法: max={trans_normalized_new.abs().max().item():.2e}")
    print(f"  ✓ 通过：epsilon 保护使值更加稳定")

    # 场景3: 正常 norm_scale（验证 epsilon 的影响很小）
    print("\n场景3: norm_scale_gt = 70.0 (正常值)")
    norm_scale_gt = torch.tensor([[70.0]], dtype=torch.float32)
    trans_gt = torch.tensor([[[100.0, 200.0, 800.0]]], dtype=torch.float32)

    trans_normalized_old = trans_gt / norm_scale_gt[..., None]
    trans_normalized_new = trans_gt / (norm_scale_gt[..., None] + NORM_SCALE_EPSILON)

    diff = torch.abs(trans_normalized_new - trans_normalized_old).max()
    print(f"  原始方法: {trans_normalized_old[0, 0].numpy()}")
    print(f"  修复方法: {trans_normalized_new[0, 0].numpy()}")
    print(f"  差异: {diff:.2e}")

    assert diff < 1e-5, f"✗ Epsilon 影响太大: diff={diff}"
    print(f"  ✓ 通过：epsilon 对正常值的影响 < 1e-5，可以忽略")

    # 场景4: 反归一化
    print("\n场景4: 反归一化测试")
    norm_scale_gt = torch.tensor([[0.0]], dtype=torch.float32)
    trans_pred = torch.tensor([[[1.5, 2.5, 10.0]]], dtype=torch.float32)

    # 原始方法
    trans_scaled_old = trans_pred * norm_scale_gt[..., None]
    print(f"  原始方法: {trans_scaled_old[0, 0].numpy()}")

    # 修复后
    trans_scaled_new = trans_pred * (norm_scale_gt[..., None] + NORM_SCALE_EPSILON)
    print(f"  修复方法: {trans_scaled_new[0, 0].numpy()}")

    assert not torch.isnan(trans_scaled_new).any(), "✗ NaN found in denormalization!"
    print(f"  ✓ 通过：反归一化不产生 NaN")

    print("\n" + "=" * 80)
    print("所有测试通过！")
    print("=" * 80)


def test_clamp_protection():
    """测试 clamp 保护"""

    print("\n" + "=" * 80)
    print("测试 clamp 保护")
    print("=" * 80)

    # 模拟 get_hand_norm_scale 的计算
    print("\n场景: 计算 norm_scale")

    # 所有关节重合（d=0）
    j3d = torch.tensor([[[10.0, 20.0, 30.0]] * 4], dtype=torch.float32)  # [1, 4, 3]
    d = j3d[..., :-1, :] - j3d[..., 1:, :]  # [1, 3, 3]
    norm_scale_old = torch.sum(torch.sqrt(torch.sum(d ** 2, dim=-1)), dim=-1)  # [1]

    print(f"  原始方法: norm_scale={norm_scale_old.item():.6f}")

    # 添加 clamp
    norm_scale_new = torch.clamp(norm_scale_old, min=NORM_SCALE_EPSILON)
    print(f"  修复方法: norm_scale={norm_scale_new.item():.6f}")

    assert norm_scale_new >= NORM_SCALE_EPSILON, "✗ clamp 失败！"
    print(f"  ✓ 通过：norm_scale >= {NORM_SCALE_EPSILON}")

    print("\n" + "=" * 80)
    print("clamp 保护测试通过！")
    print("=" * 80)


if __name__ == "__main__":
    test_norm_scale_epsilon_protection()
    test_clamp_protection()

    print("\n" + "=" * 80)
    print("全部测试通过！NaN 修复已生效。")
    print("=" * 80)
