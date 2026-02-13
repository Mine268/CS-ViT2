"""
发现 RobustL1Loss 的关键bug

torch.where 会计算两个分支，即使最后只选择其中一个！
这可能导致在 bf16 下产生 NaN
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model.loss import RobustL1Loss

print("=" * 80)
print("RobustL1Loss 关键 Bug 分析")
print("=" * 80)

# 问题：torch.where 会计算所有分支
print("\n关键发现：torch.where 的行为")
print("-" * 80)

print("\ntorch.where(condition, x, y) 的行为:")
print("  1. x 和 y 都会被计算")
print("  2. 然后根据 condition 选择结果")
print("  3. 即使某个分支不会被选中，也会被计算")

print("\n这意味着：")
print("  - 当 abs_diff < delta 时（应该用 L1）")
print("  - loss_log 分支仍然会被计算")
print("  - 如果 loss_log 计算产生 NaN，整个结果会是 NaN")

# 复现问题
print("\n" + "=" * 80)
print("复现问题")
print("=" * 80)

robust_loss = RobustL1Loss(delta=84.0, reduction='none')

# 场景：abs_diff < delta，理论上应该只用 L1
pred = torch.tensor([0.0, 100.0], dtype=torch.float32)
target = torch.tensor([50.0, 150.0], dtype=torch.float32)  # abs_diff = [50, 50]

abs_diff = torch.abs(pred - target)
print(f"\nabs_diff: {abs_diff}")
print(f"delta: {robust_loss.delta}")
print(f"abs_diff < delta: {abs_diff < robust_loss.delta}")  # 都是 True

# 手动计算 loss_log 分支（即使不应该被使用）
inside_mask = abs_diff < robust_loss.delta
outside_diff = abs_diff - robust_loss.delta  # [50-84, 50-84] = [-34, -34]

print(f"\noutside_diff (abs_diff - delta): {outside_diff}")
print(f"outside_diff 是负数: {(outside_diff < 0).all().item()}")

ratio = outside_diff / robust_loss.delta  # [-34/84, -34/84] ≈ [-0.405, -0.405]
print(f"ratio (outside_diff / delta): {ratio}")

# log1p 对负数的行为
log_value = torch.log1p(ratio)
print(f"log1p(ratio): {log_value}")
print(f"是否有 NaN: {torch.isnan(log_value).any().item()}")

# 实际 loss 计算
loss = robust_loss(pred, target)
print(f"\n实际 loss: {loss}")
print(f"是否有 NaN: {torch.isnan(loss).any().item()}")

# 关键测试：在 bf16 下
print("\n" + "=" * 80)
print("在 bf16 下测试")
print("=" * 80)

pred_bf16 = pred.to(torch.bfloat16).to(torch.float32)
target_bf16 = target.to(torch.bfloat16).to(torch.float32)

loss_bf16 = robust_loss(pred_bf16, target_bf16)
print(f"bf16 下的 loss: {loss_bf16}")
print(f"是否有 NaN: {torch.isnan(loss_bf16).any().item()}")

# 极端情况：outside_diff / delta 接近 -1
print("\n" + "=" * 80)
print("极端情况：ratio 接近 -1")
print("=" * 80)

# abs_diff 接近 0，outside_diff 接近 -delta
pred = torch.tensor([0.0], dtype=torch.float32)
target = torch.tensor([0.01], dtype=torch.float32)  # abs_diff = 0.01，远小于 delta

abs_diff = torch.abs(pred - target)
outside_diff = abs_diff - robust_loss.delta  # 0.01 - 84 = -83.99
ratio = outside_diff / robust_loss.delta  # -83.99 / 84 ≈ -0.9999

print(f"abs_diff: {abs_diff.item()}")
print(f"outside_diff: {outside_diff.item()}")
print(f"ratio: {ratio.item()}")
print(f"log1p(ratio): {torch.log1p(ratio).item()}")

# 当 ratio 接近 -1 时，log1p(ratio) 接近 -inf
print(f"\nlog1p(-0.99): {torch.log1p(torch.tensor(-0.99)).item()}")
print(f"log1p(-0.999): {torch.log1p(torch.tensor(-0.999)).item()}")
print(f"log1p(-0.9999): {torch.log1p(torch.tensor(-0.9999)).item()}")
print(f"log1p(-1.0): {torch.log1p(torch.tensor(-1.0)).item()}")
print(f"log1p(-1.001): {torch.log1p(torch.tensor(-1.001)).item()}")  # NaN!

# 测试 where 是否会避免计算不需要的分支
print("\n" + "=" * 80)
print("测试 torch.where 的计算行为")
print("=" * 80)

def test_where_computation():
    """测试 where 是否会计算所有分支"""
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])

    # 这个会产生除零
    z = torch.tensor([5.0, 0.0])

    # 即使 condition 为 False，x/z 仍然会被计算
    result = torch.where(torch.tensor([True, False]), x / z, y)

    print(f"x: {x}")
    print(f"y: {y}")
    print(f"z: {z}")
    print(f"x / z: {x / z}")  # [0.2, inf]
    print(f"result (where): {result}")
    return result

result = test_where_computation()
print(f"是否有 Inf: {torch.isinf(result).any().item()}")

# 总结
print("\n" + "=" * 80)
print("问题总结")
print("=" * 80)

print("\n发现的问题：")
print("  1. torch.where 会计算所有分支，即使不使用")
print("  2. 当 abs_diff < delta 时:")
print("     - inside_mask = True，应该使用 loss_l1")
print("     - 但 loss_log 仍然会被计算")
print("     - outside_diff 会是负数")
print("     - outside_diff / delta 会是负数")
print("     - 如果接近 -1，log1p 会产生 -inf")
print("     - 在 bf16 精度下，可能触发 NaN")

print("\n为什么换成 L1 就不会 NaN：")
print("  - L1 loss 只是简单的 abs(pred - target)")
print("  - 没有除法，没有 log 运算")
print("  - 对数值误差不敏感")

print("\n解决方案：")
print("  方案1: 换回 L1（简单但失去鲁棒性）")
print("  方案2: 修复 RobustL1Loss 实现：")
print("    - 避免计算不需要的分支")
print("    - 或者添加数值保护")

print("\n" + "=" * 80)
