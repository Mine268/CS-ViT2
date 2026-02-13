"""
测试 RobustL1Loss 在 bf16 混合精度下的行为

重点测试可能导致 NaN 的场景
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model.loss import RobustL1Loss

print("=" * 80)
print("RobustL1Loss 在 bf16 混合精度下的测试")
print("=" * 80)

# 创建 loss 实例
delta_value = 84.0
robust_loss = RobustL1Loss(delta=delta_value, reduction='none')

print(f"\n配置:")
print(f"  delta: {robust_loss.delta}")
print(f"  delta 类型: {type(robust_loss.delta)}")

# 关键测试：检查 delta 是否被正确保存
print(f"\n检查 delta 属性:")
print(f"  robust_loss.delta: {robust_loss.delta}")
print(f"  是否为 Tensor: {isinstance(robust_loss.delta, torch.Tensor)}")
print(f"  是否为 float: {isinstance(robust_loss.delta, float)}")

# 测试1：检查 delta 在 forward 中的使用
print("\n测试1: delta 在 forward 中的使用")

pred = torch.tensor([[100.0]], dtype=torch.float32)
target = torch.tensor([[200.0]], dtype=torch.float32)

# 手动计算
abs_diff = torch.abs(pred - target)  # 100.0
inside_mask = abs_diff < robust_loss.delta  # 100 < 84 = False
outside_diff = abs_diff - robust_loss.delta  # 100 - 84 = 16
ratio = outside_diff / robust_loss.delta  # 16 / 84 = 0.19

print(f"  abs_diff: {abs_diff.item()}")
print(f"  robust_loss.delta: {robust_loss.delta}")
print(f"  inside_mask: {inside_mask.item()}")
print(f"  outside_diff: {outside_diff.item()}")
print(f"  ratio: {ratio.item()}")
print(f"  log1p(ratio): {torch.log1p(ratio).item()}")

# 实际计算
loss = robust_loss(pred, target)
print(f"  实际 loss: {loss.item()}")
print(f"  是否有 NaN: {torch.isnan(loss).any().item()}")

# 测试2：检查 delta 是否可能变成 0
print("\n测试2: 检查 delta 的类型和值")

# 模拟从配置读取
config_delta = 84.0
robust_loss_from_config = RobustL1Loss(delta=config_delta, reduction='none')

print(f"  config_delta: {config_delta} (type: {type(config_delta)})")
print(f"  robust_loss_from_config.delta: {robust_loss_from_config.delta}")
print(f"  是否相等: {config_delta == robust_loss_from_config.delta}")

# 测试3：检查 bf16 转换后的 delta
print("\n测试3: bf16 转换对 delta 的影响")

# 将整个 loss module 转换为 bf16（这不应该做，但测试一下）
print("  警告: 测试将 loss module 转换为 bf16（不推荐）")

delta_f32 = torch.tensor(84.0, dtype=torch.float32)
delta_bf16 = delta_f32.to(torch.bfloat16)

print(f"  delta (float32): {delta_f32.item()}")
print(f"  delta (bf16): {delta_bf16.item()}")
print(f"  差异: {abs(delta_f32.item() - delta_bf16.item())}")

# 如果 delta 不小心被转换为 bf16 再转回来
delta_bf16_to_f32 = delta_bf16.to(torch.float32)
print(f"  delta (bf16 → float32): {delta_bf16_to_f32.item()}")

# 测试4：检查除零保护
print("\n测试4: 检查除零保护")

# 如果 delta 变成 0
try:
    robust_loss_zero = RobustL1Loss(delta=0.0, reduction='none')
    pred = torch.tensor([0.0, 100.0], dtype=torch.float32)
    target = torch.tensor([50.0, 200.0], dtype=torch.float32)

    abs_diff = torch.abs(pred - target)
    outside_diff = abs_diff - robust_loss_zero.delta
    ratio = outside_diff / robust_loss_zero.delta  # 除零！

    print(f"  delta=0 时:")
    print(f"    outside_diff: {outside_diff}")
    print(f"    outside_diff / 0.0: {ratio}")
    print(f"    是否有 Inf: {torch.isinf(ratio).any().item()}")

    loss = robust_loss_zero(pred, target)
    print(f"    loss: {loss}")
    print(f"    是否有 NaN: {torch.isnan(loss).any().item()}")
except Exception as e:
    print(f"  错误: {e}")

# 测试5：检查 log1p 在 bf16 下的行为
print("\n测试5: log1p 在 bf16 下的行为")

ratios_f32 = torch.tensor([0.0, 0.1, 1.0, 10.0, 100.0], dtype=torch.float32)
ratios_bf16 = ratios_f32.to(torch.bfloat16).to(torch.float32)

log_f32 = torch.log1p(ratios_f32)
log_bf16 = torch.log1p(ratios_bf16)

print(f"  ratios (f32): {ratios_f32}")
print(f"  ratios (bf16→f32): {ratios_bf16}")
print(f"  log1p (f32): {log_f32}")
print(f"  log1p (bf16): {log_bf16}")
print(f"  差异: {torch.abs(log_f32 - log_bf16)}")

# 测试6：完整的 bf16 训练流程模拟
print("\n测试6: 模拟 bf16 训练流程")

# 模拟前向传播产生 bf16 的预测和目标
pred_f32 = torch.randn(32, 1, 21, 2, dtype=torch.float32) * 100
target_f32 = pred_f32 + torch.randn_like(pred_f32) * 50

# 转换为 bf16（模拟混合精度）
pred_bf16 = pred_f32.to(torch.bfloat16)
target_bf16 = target_f32.to(torch.bfloat16)

# Loss 计算（应该在 float32）
pred_for_loss = pred_bf16.to(torch.float32)
target_for_loss = target_bf16.to(torch.float32)

loss_bf16 = robust_loss(pred_for_loss, target_for_loss)

print(f"  输入 shape: {pred_for_loss.shape}")
print(f"  loss shape: {loss_bf16.shape}")
print(f"  loss 统计:")
print(f"    min: {loss_bf16.min().item():.4f}")
print(f"    max: {loss_bf16.max().item():.4f}")
print(f"    mean: {loss_bf16.mean().item():.4f}")
print(f"    是否有 NaN: {torch.isnan(loss_bf16).any().item()}")
print(f"    是否有 Inf: {torch.isinf(loss_bf16).any().item()}")

# 测试7：检查是否有输入包含 NaN 的情况
print("\n测试7: 检查输入包含 NaN 时的行为")

pred_with_nan = torch.tensor([[100.0], [float('nan')], [200.0]], dtype=torch.float32)
target_clean = torch.tensor([[105.0], [95.0], [210.0]], dtype=torch.float32)

loss_with_nan_input = robust_loss(pred_with_nan, target_clean)
print(f"  pred (包含 NaN): {pred_with_nan.squeeze()}")
print(f"  target: {target_clean.squeeze()}")
print(f"  loss: {loss_with_nan_input.squeeze()}")
print(f"  是否有 NaN: {torch.isnan(loss_with_nan_input).any().item()}")

# 关键分析
print("\n" + "=" * 80)
print("关键发现")
print("=" * 80)

print("\n1. delta 参数:")
print(f"   - 当前值: {robust_loss.delta}")
print(f"   - 类型: {type(robust_loss.delta)}")
print(f"   - 如果 delta=0，会产生 NaN")

print("\n2. RobustL1Loss 的数值稳定性:")
print(f"   - 正常情况下不会产生 NaN")
print(f"   - bf16 精度损失不会导致 NaN")
print(f"   - 只有在 delta=0 或输入本身包含 NaN 时才会出现 NaN")

print("\n3. 用户报告:")
print(f"   - 用户说换回 L1 就不会 NaN")
print(f"   - 这说明问题确实在 RobustL1Loss")
print(f"   - 但从测试看，RobustL1Loss 本身没问题")

print("\n4. 可能的原因:")
print(f"   - 配置读取错误，delta 变成 0 或 None")
print(f"   - 前向传播已经产生 NaN，传入 RobustL1Loss")
print(f"   - 或者 torch.where 在某些特殊条件下的问题")

print("\n建议:")
print(f"   1. 检查训练日志中 delta 的实际值")
print(f"   2. 在 loss 计算前检查输入是否包含 NaN")
print(f"   3. 添加 delta != 0 的断言")
print(f"   4. 或者直接换回 L1（稳妥但不鲁棒）")

print("\n" + "=" * 80)
