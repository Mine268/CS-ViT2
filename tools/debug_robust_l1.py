"""
调试 RobustL1Loss 的 NaN 问题

测试各种场景，找出导致 NaN 的原因
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model.loss import RobustL1Loss

print("=" * 80)
print("RobustL1Loss NaN 问题调试")
print("=" * 80)

# 创建 loss 实例
robust_loss = RobustL1Loss(delta=84.0, reduction='none')

print(f"\nRobustL1Loss 配置:")
print(f"  delta: {robust_loss.delta}")
print(f"  reduction: {robust_loss.reduction}")

# 测试场景1：正常值
print("\n场景1: 正常值")
pred = torch.tensor([100.0, 200.0, 300.0], dtype=torch.float32)
target = torch.tensor([105.0, 190.0, 310.0], dtype=torch.float32)

loss = robust_loss(pred, target)
print(f"  pred: {pred}")
print(f"  target: {target}")
print(f"  loss: {loss}")
print(f"  是否有 NaN: {torch.isnan(loss).any().item()}")

# 测试场景2：大误差（超过 delta）
print("\n场景2: 大误差（超过 delta=84）")
pred = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
target = torch.tensor([100.0, 200.0, 500.0], dtype=torch.float32)

loss = robust_loss(pred, target)
abs_diff = torch.abs(pred - target)
print(f"  pred: {pred}")
print(f"  target: {target}")
print(f"  abs_diff: {abs_diff}")
print(f"  loss: {loss}")
print(f"  是否有 NaN: {torch.isnan(loss).any().item()}")

# 测试场景3：极大误差
print("\n场景3: 极大误差")
pred = torch.tensor([0.0], dtype=torch.float32)
target = torch.tensor([10000.0], dtype=torch.float32)

loss = robust_loss(pred, target)
abs_diff = torch.abs(pred - target)
outside_diff = abs_diff - robust_loss.delta
ratio = outside_diff / robust_loss.delta
log_value = torch.log1p(ratio)

print(f"  pred: {pred.item()}")
print(f"  target: {target.item()}")
print(f"  abs_diff: {abs_diff.item()}")
print(f"  outside_diff: {outside_diff.item()}")
print(f"  ratio (outside_diff/delta): {ratio.item():.6f}")
print(f"  log1p(ratio): {log_value.item():.6f}")
print(f"  loss: {loss.item():.6f}")
print(f"  是否有 NaN: {torch.isnan(loss).any().item()}")

# 测试场景4：负数（理论上 abs 后不应该有负数）
print("\n场景4: 输入包含 NaN")
pred = torch.tensor([float('nan'), 100.0], dtype=torch.float32)
target = torch.tensor([105.0, 95.0], dtype=torch.float32)

loss = robust_loss(pred, target)
print(f"  pred: {pred}")
print(f"  target: {target}")
print(f"  loss: {loss}")
print(f"  是否有 NaN: {torch.isnan(loss).any().item()}")

# 测试场景5：bf16 混合精度
print("\n场景5: bf16 混合精度")
pred_f32 = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
target_f32 = torch.tensor([100.0, 200.0, 500.0], dtype=torch.float32)

# 转换为 bf16
pred_bf16 = pred_f32.to(torch.bfloat16)
target_bf16 = target_f32.to(torch.bfloat16)

# 计算 loss (RobustL1Loss 内部是 float32 计算)
loss_bf16 = robust_loss(pred_bf16.to(torch.float32), target_bf16.to(torch.float32))

print(f"  pred (bf16): {pred_bf16}")
print(f"  target (bf16): {target_bf16}")
print(f"  loss: {loss_bf16}")
print(f"  是否有 NaN: {torch.isnan(loss_bf16).any().item()}")

# 测试场景6：检查 torch.where 的行为
print("\n场景6: torch.where 在边界条件下的行为")

# 测试在 delta 附近的值
test_values = torch.tensor([
    83.0,  # 刚好小于 delta
    84.0,  # 等于 delta
    85.0,  # 刚好大于 delta
], dtype=torch.float32)

pred = torch.zeros_like(test_values)
target = test_values

loss = robust_loss(pred, target)
abs_diff = torch.abs(pred - target)
inside_mask = abs_diff < robust_loss.delta

print(f"  abs_diff: {abs_diff}")
print(f"  inside_mask: {inside_mask}")
print(f"  loss: {loss}")
print(f"  是否有 NaN: {torch.isnan(loss).any().item()}")

# 测试场景7：检查除零问题
print("\n场景7: 检查除零（delta=0）")

robust_loss_zero = RobustL1Loss(delta=0.0, reduction='none')
pred = torch.tensor([0.0, 100.0], dtype=torch.float32)
target = torch.tensor([50.0, 200.0], dtype=torch.float32)

try:
    loss = robust_loss_zero(pred, target)
    print(f"  loss: {loss}")
    print(f"  是否有 NaN: {torch.isnan(loss).any().item()}")
    print(f"  是否有 Inf: {torch.isinf(loss).any().item()}")
except Exception as e:
    print(f"  错误: {e}")

# 测试场景8：对比 L1 Loss
print("\n场景8: 对比 L1 Loss")

pred = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
target = torch.tensor([100.0, 200.0, 500.0], dtype=torch.float32)

# L1 Loss
l1_loss = torch.nn.L1Loss(reduction='none')
loss_l1 = l1_loss(pred, target)

# RobustL1 Loss
loss_robust = robust_loss(pred, target)

print(f"  pred: {pred}")
print(f"  target: {target}")
print(f"  L1 loss: {loss_l1}")
print(f"  RobustL1 loss: {loss_robust}")
print(f"  L1 是否有 NaN: {torch.isnan(loss_l1).any().item()}")
print(f"  RobustL1 是否有 NaN: {torch.isnan(loss_robust).any().item()}")

# 测试场景9：检查 log1p 的输入范围
print("\n场景9: 检查 log1p 的输入范围")

# log1p(x) 对于 x < -1 会产生 NaN
test_log_inputs = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=torch.float32)
log_outputs = torch.log1p(test_log_inputs)

print(f"  输入 x: {test_log_inputs}")
print(f"  log1p(x): {log_outputs}")
print(f"  是否有 NaN: {torch.isnan(log_outputs).any().item()}")

# 分析 RobustL1Loss 中 log1p 的输入
print("\n分析 RobustL1Loss 中的计算:")
pred = torch.tensor([0.0], dtype=torch.float32)
target = torch.tensor([500.0], dtype=torch.float32)

abs_diff = torch.abs(pred - target)
outside_diff = abs_diff - robust_loss.delta  # 应该 >= 0
ratio = outside_diff / robust_loss.delta      # 应该 >= 0

print(f"  abs_diff: {abs_diff.item()}")
print(f"  outside_diff: {outside_diff.item()} (应该 >= 0)")
print(f"  ratio: {ratio.item()} (应该 >= 0)")
print(f"  log1p(ratio): {torch.log1p(ratio).item()}")

# 检查可能的负数情况
print("\n检查是否可能出现负数:")
print(f"  abs_diff 是否可能为负: 否 (abs 保证 >= 0)")
print(f"  outside_diff 是否可能为负: 是! (当 abs_diff < delta)")
print(f"  但是 outside_diff 只在 abs_diff >= delta 时使用")
print(f"  所以 outside_diff / delta 应该总是 >= 0")

# 测试场景10：模拟训练中的实际数据
print("\n场景10: 模拟训练中的实际数据 (bf16)")

# 模拟重投影误差
pred_img = torch.randn(32, 1, 21, 2, dtype=torch.float32) * 100  # [B, T, J, 2]
target_img = pred_img + torch.randn_like(pred_img) * 50  # 添加噪声

# 转换为 bf16（模拟混合精度训练）
pred_img_bf16 = pred_img.to(torch.bfloat16).to(torch.float32)
target_img_bf16 = target_img.to(torch.bfloat16).to(torch.float32)

loss = robust_loss(pred_img_bf16, target_img_bf16)

print(f"  shape: {loss.shape}")
print(f"  loss 统计:")
print(f"    min: {loss.min().item():.4f}")
print(f"    max: {loss.max().item():.4f}")
print(f"    mean: {loss.mean().item():.4f}")
print(f"  是否有 NaN: {torch.isnan(loss).any().item()}")
print(f"  是否有 Inf: {torch.isinf(loss).any().item()}")

# 总结
print("\n" + "=" * 80)
print("调试总结")
print("=" * 80)

print("\n可能的 NaN 来源:")
print("  1. 输入本身包含 NaN（前向传播错误）")
print("  2. log1p 的输入 < -1（理论上不应该发生）")
print("  3. delta = 0 导致除零")
print("  4. bf16 精度损失导致的数值问题")
print("  5. torch.where 在某些条件下的行为")

print("\n需要检查:")
print("  - 在实际训练中，loss 的输入是否已经包含 NaN")
print("  - delta 的值是否正确传递")
print("  - 是否在 bf16 下计算 loss")

print("\n" + "=" * 80)
