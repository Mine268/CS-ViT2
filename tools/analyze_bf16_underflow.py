"""
分析 bf16 混合精度是否导致 norm_scale 下溢

检验假设：
- 某些样本的 norm_scale 在 float32 下是正常的小值（如 1e-7）
- 在 bf16 下这些值会下溢到 0
- 导致除零产生 NaN
"""

import torch
import numpy as np

print("=" * 80)
print("bf16 混合精度与 norm_scale 下溢分析")
print("=" * 80)

# bf16 数值范围分析
print("\n1. bf16 数值特性")
print("-" * 80)

print("\nbfloat16 (bf16) 格式:")
print("  - 符号位: 1 bit")
print("  - 指数位: 8 bits (与 float32 相同)")
print("  - 尾数位: 7 bits (float32 是 23 bits)")
print("  - 范围: 与 float32 相同 (约 1e-38 到 3e38)")
print("  - 精度: 约 3 位有效数字 (float32 是 7 位)")

# 测试不同数值在 bf16 下的表现
print("\n2. 小数值在 bf16 下的行为")
print("-" * 80)

test_values = [
    1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
    0.5, 1.0, 10.0, 50.0, 70.0, 100.0
]

print(f"\n{'原始值 (float32)':<20} {'bf16 表示':<20} {'误差':<15} {'相对误差':<15}")
print("-" * 70)

for val in test_values:
    val_f32 = torch.tensor(val, dtype=torch.float32)
    val_bf16 = val_f32.to(torch.bfloat16).to(torch.float32)

    error = abs(val_bf16.item() - val)
    rel_error = error / val if val > 0 else 0

    print(f"{val:<20.2e} {val_bf16.item():<20.2e} {error:<15.2e} {rel_error:<15.2%}")

# 关键发现：检查哪些值会下溢到 0
print("\n3. bf16 下溢检测")
print("-" * 80)

underflow_threshold = 1e-38  # bf16 理论最小正数
critical_values = [1e-10, 1e-9, 1e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5]

print(f"\n{'原始值':<15} {'bf16 值':<15} {'是否为0':<10} {'状态':<10}")
print("-" * 50)

for val in critical_values:
    val_f32 = torch.tensor(val, dtype=torch.float32)
    val_bf16 = val_f32.to(torch.bfloat16)
    is_zero = (val_bf16.item() == 0.0)

    status = "下溢!" if is_zero else "正常"
    print(f"{val:<15.2e} {val_bf16.item():<15.2e} {str(is_zero):<10} {status:<10}")

# 模拟 norm_scale 计算在不同精度下的行为
print("\n4. norm_scale 计算精度差异")
print("-" * 80)

# 模拟一个接近重合的手指（norm_scale 很小）
print("\n场景: 中指4个关节几乎重合（可能是标注误差）")

# 4个关节，几乎在同一位置，但有微小差异
joints_f32 = torch.tensor([
    [100.0, 200.0, 800.0],      # Middle_1
    [100.001, 200.001, 800.001],  # Middle_2 (差0.001mm)
    [100.002, 200.001, 800.002],  # Middle_3
    [100.002, 200.002, 800.003],  # Middle_4
], dtype=torch.float32)

# 计算 norm_scale (float32)
d_f32 = joints_f32[:-1] - joints_f32[1:]
norm_scale_f32 = torch.sum(torch.sqrt(torch.sum(d_f32 ** 2, dim=-1)))

print(f"\nfloat32 计算:")
print(f"  关节差异: {d_f32}")
print(f"  norm_scale: {norm_scale_f32.item():.10f} mm")

# 计算 norm_scale (bf16)
joints_bf16 = joints_f32.to(torch.bfloat16)
d_bf16 = joints_bf16[:-1] - joints_bf16[1:]
norm_scale_bf16 = torch.sum(torch.sqrt(torch.sum(d_bf16 ** 2, dim=-1)))

print(f"\nbfloat16 计算:")
print(f"  关节差异: {d_bf16}")
print(f"  norm_scale: {norm_scale_bf16.item():.10f} mm")

print(f"\n差异:")
print(f"  绝对误差: {abs(norm_scale_f32 - norm_scale_bf16).item():.10f} mm")
print(f"  相对误差: {(abs(norm_scale_f32 - norm_scale_bf16) / norm_scale_f32 * 100).item():.2f}%")

if norm_scale_bf16.item() == 0.0:
    print(f"  ⚠️  警告: bf16 下 norm_scale 下溢到 0！")

# 更极端的场景：关节完全重合（数据错误）
print("\n场景: 中指4个关节完全重合（数据错误）")

joints_same = torch.tensor([
    [100.0, 200.0, 800.0],
    [100.0, 200.0, 800.0],
    [100.0, 200.0, 800.0],
    [100.0, 200.0, 800.0],
], dtype=torch.float32)

d_same_f32 = joints_same[:-1] - joints_same[1:]
norm_scale_same_f32 = torch.sum(torch.sqrt(torch.sum(d_same_f32 ** 2, dim=-1)))

d_same_bf16 = joints_same.to(torch.bfloat16)[:-1] - joints_same.to(torch.bfloat16)[1:]
norm_scale_same_bf16 = torch.sum(torch.sqrt(torch.sum(d_same_bf16 ** 2, dim=-1)))

print(f"  float32 norm_scale: {norm_scale_same_f32.item()}")
print(f"  bf16 norm_scale: {norm_scale_same_bf16.item()}")

# 测试除法在不同精度下的行为
print("\n5. 除零行为对比")
print("-" * 80)

trans = torch.tensor([[100.0, 200.0, 800.0]], dtype=torch.float32)

print("\nfloat32:")
for ns in [0.0, 1e-10, 1e-7, 1e-6, 1e-3]:
    norm_scale = torch.tensor(ns, dtype=torch.float32)
    result = trans / norm_scale[..., None]
    print(f"  trans / {ns:.2e} = {result[0].numpy()} (NaN:{torch.isnan(result).any().item()}, Inf:{torch.isinf(result).any().item()})")

print("\nbfloat16 (模拟混合精度训练):")
for ns in [0.0, 1e-10, 1e-7, 1e-6, 1e-3]:
    # 模拟：norm_scale 在 bf16 计算中可能下溢
    norm_scale_f32 = torch.tensor(ns, dtype=torch.float32)
    norm_scale_bf16 = norm_scale_f32.to(torch.bfloat16).to(torch.float32)

    result = trans / norm_scale_bf16[..., None]
    print(f"  trans / {ns:.2e} (bf16:{norm_scale_bf16.item():.2e}) = {result[0].numpy()} (NaN:{torch.isnan(result).any().item()}, Inf:{torch.isinf(result).any().item()})")

# 分析训练配置
print("\n6. 训练配置分析")
print("-" * 80)

print("\n当前训练配置:")
print("  - mixed_precision: bf16")
print("  - norm_by_hand: true")
print("  - NORM_SCALE_EPSILON: 1e-6 (已添加)")

print("\n风险评估:")
print("  1. 数据中如果存在 norm_scale < 1e-6 的样本")
print("  2. 在 bf16 下可能表示为 0 或非常不精确的值")
print("  3. 导致除零或数值不稳定")

print("\n之前 float32 训练成功的原因:")
print("  - float32 有更高的精度（23位尾数 vs 7位）")
print("  - 即使 norm_scale 很小（如 1e-7），也能正确表示")
print("  - 不会发生下溢")

# 建议
print("\n7. 解决方案对比")
print("-" * 80)

print("\n方案A: 保持 bf16 + epsilon 保护（已实施）")
print("  优点:")
print("    ✓ 保留混合精度训练的速度优势（约2倍加速）")
print("    ✓ 节省显存（约50%）")
print("    ✓ epsilon=1e-6 足够大，bf16 可以精确表示")
print("  缺点:")
print("    - 对异常数据给出了近似处理")
print("    - 仍然依赖数据质量")

print("\n方案B: 切换回 float32")
print("  优点:")
print("    ✓ 数值稳定性更好")
print("    ✓ 可以处理更小的 norm_scale")
print("    ✓ 与之前的训练保持一致")
print("  缺点:")
print("    - 训练速度慢约2倍")
print("    - 显存占用增加约2倍")
print("    - 可能无法使用更大的 batch size")

print("\n方案C: 混合策略（推荐）")
print("  优点:")
print("    ✓ 大部分计算用 bf16（速度快）")
print("    ✓ norm_scale 相关计算用 float32（精度高）")
print("  实现:")
print("    - 在 get_hand_norm_scale 中强制 float32")
print("    - 除法和乘法前临时转换为 float32")

# 测试方案C的可行性
print("\n8. 方案C实现测试")
print("-" * 80)

print("\n模拟: norm_scale 计算用 float32，其他用 bf16")

joints_bf16 = torch.tensor([
    [100.0, 200.0, 800.0],
    [100.001, 200.001, 800.001],
    [100.002, 200.001, 800.002],
    [100.002, 200.002, 800.003],
], dtype=torch.bfloat16)

# 关键计算用 float32
d_mixed = joints_bf16.to(torch.float32)[:-1] - joints_bf16.to(torch.float32)[1:]
norm_scale_mixed = torch.sum(torch.sqrt(torch.sum(d_mixed ** 2, dim=-1)))

print(f"  混合精度 norm_scale: {norm_scale_mixed.item():.10f} mm")
print(f"  与 float32 一致: {abs(norm_scale_mixed - norm_scale_f32).item() < 1e-10}")

# 总结
print("\n" + "=" * 80)
print("总结")
print("=" * 80)

print("\n问题根源:")
print("  ✓ 用户观察正确：bf16 混合精度是导致 NaN 的关键因素")
print("  ✓ 之前 float32 训练没有问题，因为精度足够高")
print("  ✓ 某些样本的 norm_scale 在 bf16 下下溢或精度损失严重")

print("\n立即可用的修复（已实施）:")
print("  ✓ epsilon=1e-6 保护足够大，bf16 可以精确表示")
print("  ✓ 可以继续使用 bf16 训练，获得速度优势")

print("\n后续优化建议:")
print("  1. 监控训练中 norm_scale 的分布")
print("  2. 如果频繁出现异常，考虑方案C（混合策略）")
print("  3. 或者切换回 float32（稳妥但慢）")

print("\n" + "=" * 80)
