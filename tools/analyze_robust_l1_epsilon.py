"""
仔细分析 RobustL1Loss 需要 epsilon 保护的位置

既然 float32 下也会 NaN，说明不是精度问题，而是实现问题
"""

import torch
import numpy as np

print("=" * 80)
print("RobustL1Loss 数值稳定性分析（float32）")
print("=" * 80)

# 当前实现
print("\n当前 RobustL1Loss 实现:")
print("-" * 80)
print("""
def forward(self, pred, target):
    abs_diff = torch.abs(pred - target)

    inside_mask = abs_diff < self.delta
    loss_l1 = abs_diff

    outside_diff = abs_diff - self.delta          # 问题1: 可能是负数
    loss_log = self.delta * (1.0 + torch.log1p(outside_diff / self.delta))  # 问题2: 除法, 问题3: log1p

    loss = torch.where(inside_mask, loss_l1, loss_log)
""")

# 分析每个可能的 NaN 源
print("\n" + "=" * 80)
print("可能的 NaN 来源分析")
print("=" * 80)

delta = 84.0

# 来源1: 输入本身包含 NaN
print("\n来源1: 输入本身包含 NaN")
print("-" * 80)
pred = torch.tensor([100.0, float('nan')], dtype=torch.float32)
target = torch.tensor([105.0, 95.0], dtype=torch.float32)

abs_diff = torch.abs(pred - target)
print(f"pred: {pred}")
print(f"target: {target}")
print(f"abs_diff: {abs_diff}")
print(f"→ 如果输入有 NaN，abs_diff 就有 NaN")

# 来源2: outside_diff / delta 可能导致问题
print("\n来源2: outside_diff / delta")
print("-" * 80)

test_cases = [
    ("正常情况", 100.0, 0.0),  # abs_diff = 100, outside_diff = 16
    ("abs_diff = delta", 84.0, 0.0),  # abs_diff = 84, outside_diff = 0
    ("abs_diff < delta", 50.0, 0.0),  # abs_diff = 50, outside_diff = -34
    ("abs_diff 很小", 0.001, 0.0),  # abs_diff = 0.001, outside_diff = -83.999
]

for name, pred_val, target_val in test_cases:
    pred = torch.tensor([pred_val], dtype=torch.float32)
    target = torch.tensor([target_val], dtype=torch.float32)

    abs_diff = torch.abs(pred - target)
    outside_diff = abs_diff - delta
    ratio = outside_diff / delta

    print(f"\n{name}:")
    print(f"  abs_diff: {abs_diff.item():.6f}")
    print(f"  outside_diff: {outside_diff.item():.6f}")
    print(f"  ratio: {ratio.item():.6f}")

    if ratio.item() > -1:
        log_val = torch.log1p(ratio)
        print(f"  log1p(ratio): {log_val.item():.6f}")
    else:
        print(f"  log1p(ratio): NaN (ratio <= -1)")

# 来源3: log1p 的临界点
print("\n来源3: log1p 的临界行为")
print("-" * 80)

critical_values = [
    -1.1,   # < -1, 产生 NaN
    -1.0,   # = -1, 产生 -inf
    -0.99,  # 接近 -1, 产生很大的负数
    -0.5,   # 负数但安全
    0.0,    # 零点
    1.0,    # 正数
]

print(f"\n{'输入 x':<15} {'log1p(x)':<20} {'状态':<10}")
print("-" * 45)
for x in critical_values:
    x_tensor = torch.tensor(x, dtype=torch.float32)
    log_val = torch.log1p(x_tensor)

    if torch.isnan(log_val):
        status = "NaN"
    elif torch.isinf(log_val):
        status = "Inf"
    else:
        status = "OK"

    print(f"{x:<15.2f} {log_val.item():<20.6f} {status:<10}")

# 来源4: 浮点数精度导致 ratio 略 < -1
print("\n来源4: 浮点数精度问题")
print("-" * 80)

# 模拟：abs_diff 非常接近 0 的情况
very_small_values = [1e-10, 1e-20, 1e-30, 1e-40]

print(f"\n{'abs_diff':<15} {'outside_diff':<20} {'ratio':<20} {'状态':<10}")
print("-" * 65)

for val in very_small_values:
    abs_diff = torch.tensor(val, dtype=torch.float32)
    outside_diff = abs_diff - delta
    ratio = outside_diff / delta

    # 检查 ratio 是否 < -1 (由于浮点精度)
    is_safe = ratio.item() > -1.0
    status = "安全" if is_safe else "危险"

    print(f"{val:<15.2e} {outside_diff.item():<20.6f} {ratio.item():<20.10f} {status:<10}")

# 来源5: 前向传播产生的极端值
print("\n来源5: 前向传播可能产生的极端值")
print("-" * 80)

# 如果前向传播产生了 Inf
pred = torch.tensor([float('inf'), 100.0], dtype=torch.float32)
target = torch.tensor([105.0, 95.0], dtype=torch.float32)

abs_diff = torch.abs(pred - target)
outside_diff = abs_diff - delta
ratio = outside_diff / delta

print(f"\npred 包含 Inf:")
print(f"  pred: {pred}")
print(f"  abs_diff: {abs_diff}")
print(f"  outside_diff: {outside_diff}")
print(f"  ratio: {ratio}")
print(f"  log1p(ratio): {torch.log1p(ratio)}")

# 分析需要 epsilon 保护的位置
print("\n" + "=" * 80)
print("需要 epsilon 保护的位置")
print("=" * 80)

print("\n位置1: outside_diff / self.delta")
print("-" * 80)
print("问题:")
print("  - 当 abs_diff < delta 时，outside_diff 是负数")
print("  - torch.where 会计算所有分支")
print("  - 如果 abs_diff ≈ 0，ratio ≈ -1")
print("  - 浮点精度可能导致 ratio < -1")
print("  - log1p(ratio < -1) = NaN")

print("\n建议的 epsilon 保护:")
print("  ratio = outside_diff / self.delta")
print("  ratio = torch.clamp(ratio, min=-0.99)  # 防止 < -1")

print("\n位置2: self.delta (检查是否为 0)")
print("-" * 80)
print("问题:")
print("  - 如果 delta = 0，会除零")
print("  - 虽然配置是 84.0，但需要防御性检查")

print("\n建议的保护:")
print("  assert self.delta > 0, f'delta must be positive, got {self.delta}'")

# 测试修复方案
print("\n" + "=" * 80)
print("测试修复方案")
print("=" * 80)

def robust_l1_fixed(pred, target, delta=84.0):
    """添加 epsilon 保护的 RobustL1Loss"""
    abs_diff = torch.abs(pred - target)

    inside_mask = abs_diff < delta
    loss_l1 = abs_diff

    outside_diff = abs_diff - delta
    ratio = outside_diff / delta

    # 关键修复：防止 ratio < -1
    ratio = torch.clamp(ratio, min=-0.99)  # epsilon 保护！

    loss_log = delta * (1.0 + torch.log1p(ratio))

    loss = torch.where(inside_mask, loss_l1, loss_log)
    return loss

print("\n测试修复后的版本:")
print("-" * 80)

# 极端情况：abs_diff 非常小
pred = torch.tensor([0.0, 100.0, 200.0], dtype=torch.float32)
target = torch.tensor([1e-10, 150.0, 250.0], dtype=torch.float32)

print(f"pred: {pred}")
print(f"target: {target}")

# 原始版本（可能 NaN）
abs_diff = torch.abs(pred - target)
outside_diff = abs_diff - delta
ratio_original = outside_diff / delta
log_original = torch.log1p(ratio_original)

print(f"\n原始版本:")
print(f"  abs_diff: {abs_diff}")
print(f"  ratio: {ratio_original}")
print(f"  log1p(ratio): {log_original}")
print(f"  是否有 NaN: {torch.isnan(log_original).any().item()}")

# 修复版本（有 epsilon 保护）
loss_fixed = robust_l1_fixed(pred, target, delta=delta)

print(f"\n修复版本:")
print(f"  loss: {loss_fixed}")
print(f"  是否有 NaN: {torch.isnan(loss_fixed).any().item()}")

# 对比 L1
loss_l1 = torch.abs(pred - target)
print(f"\nL1 Loss:")
print(f"  loss: {loss_l1}")
print(f"  是否有 NaN: {torch.isnan(loss_l1).any().item()}")

# 总结
print("\n" + "=" * 80)
print("总结")
print("=" * 80)

print("\n根本原因:")
print("  1. torch.where 会计算所有分支")
print("  2. 当 abs_diff < delta 时，ratio = (abs_diff - delta) / delta ≈ -1")
print("  3. 浮点精度可能导致 ratio 略 < -1")
print("  4. log1p(ratio < -1) = NaN")
print("  5. 即使最后选择 loss_l1，NaN 仍可能传播")

print("\n为什么 float32 也会 NaN:")
print("  - 不是精度问题，是数学问题")
print("  - log1p(x < -1) 在任何精度下都是 NaN")
print("  - torch.where 会强制计算所有分支")

print("\n为什么换成 L1 就不会 NaN:")
print("  - L1 = abs(pred - target)")
print("  - 没有除法，没有 log")
print("  - 数值稳定")

print("\n修复方案:")
print("  在 RobustL1Loss 中添加:")
print("  ratio = torch.clamp(ratio, min=-0.99)")
print("  确保 log1p 的参数 > -1")

print("\n" + "=" * 80)
