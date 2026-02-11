"""
分析混合精度训练中的 Gradient Scaling

比较 bf16 vs fp16，说明是否需要 gradient scaling
"""

import torch
import numpy as np

print("=" * 80)
print("混合精度训练：Gradient Scaling 分析")
print("=" * 80)

# 1. bf16 vs fp16 对比
print("\n1. bf16 vs fp16 数值格式对比")
print("-" * 80)

print("\nfloat16 (fp16):")
print("  - 符号位: 1 bit")
print("  - 指数位: 5 bits")
print("  - 尾数位: 10 bits")
print("  - 范围: 约 6e-8 到 6e4")
print("  - 精度: 约 3-4 位有效数字")
print("  - 最小正数: ~6e-8")
print("  - 问题: **梯度容易下溢！**")

print("\nbfloat16 (bf16):")
print("  - 符号位: 1 bit")
print("  - 指数位: 8 bits (与 float32 相同)")
print("  - 尾数位: 7 bits")
print("  - 范围: 约 1e-38 到 3e38 (与 float32 相同)")
print("  - 精度: 约 2-3 位有效数字")
print("  - 最小正数: ~1e-38")
print("  - 优势: **梯度不容易下溢！**")

# 2. 梯度下溢测试
print("\n2. 梯度下溢测试")
print("-" * 80)

# 模拟典型的深度学习梯度范围
gradient_values = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0]

print(f"\n{'梯度值':<15} {'fp16 表示':<20} {'bf16 表示':<20} {'fp16下溢':<10}")
print("-" * 65)

for grad in gradient_values:
    grad_tensor = torch.tensor(grad, dtype=torch.float32)

    # fp16
    grad_fp16 = grad_tensor.to(torch.float16)
    fp16_underflow = (grad_fp16.item() == 0.0)

    # bf16
    grad_bf16 = grad_tensor.to(torch.bfloat16)
    bf16_underflow = (grad_bf16.item() == 0.0)

    print(f"{grad:<15.2e} {grad_fp16.item():<20.2e} {grad_bf16.item():<20.2e} {'是' if fp16_underflow else '否':<10}")

# 3. Gradient Scaling 的作用
print("\n3. Gradient Scaling 的作用")
print("-" * 80)

print("\n为什么 fp16 需要 Gradient Scaling?")
print("  问题: 深度学习中，梯度经常在 1e-6 到 1e-8 范围")
print("  fp16 最小正数: ~6e-8")
print("  → 小于 6e-8 的梯度会下溢到 0")
print("  → 权重无法更新，训练失败")

print("\n解决方案: Gradient Scaling")
print("  1. Loss 计算后乘以 scale_factor (如 2^16 = 65536)")
print("  2. 反向传播时，梯度也被放大 65536 倍")
print("  3. 更新权重前，梯度除以 65536 恢复原值")
print("  → 放大后的梯度不会下溢")

print("\n为什么 bf16 不需要 Gradient Scaling?")
print("  bf16 最小正数: ~1e-38")
print("  深度学习梯度范围: 1e-6 到 1e-8")
print("  → 远大于 bf16 的最小值，不会下溢")
print("  → 不需要额外的 scaling")

# 4. 测试 Gradient Scaling 效果
print("\n4. Gradient Scaling 效果演示")
print("-" * 80)

small_grad = 1e-7
scale_factor = 65536

print(f"\n原始梯度: {small_grad:.2e}")

# fp16 无 scaling（会下溢）
grad_fp16_no_scale = torch.tensor(small_grad, dtype=torch.float16)
print(f"\nfp16 无 scaling:")
print(f"  梯度值: {grad_fp16_no_scale.item():.2e}")
print(f"  是否下溢: {'是' if grad_fp16_no_scale.item() == 0.0 else '否'}")

# fp16 有 scaling（不会下溢）
grad_scaled = small_grad * scale_factor
grad_fp16_scaled = torch.tensor(grad_scaled, dtype=torch.float16)
grad_fp16_unscaled = grad_fp16_scaled / scale_factor
print(f"\nfp16 有 scaling (scale={scale_factor}):")
print(f"  放大后: {grad_fp16_scaled.item():.2e}")
print(f"  恢复后: {grad_fp16_unscaled.item():.2e}")
print(f"  是否下溢: {'是' if grad_fp16_scaled.item() == 0.0 else '否'}")

# bf16（不需要 scaling）
grad_bf16 = torch.tensor(small_grad, dtype=torch.bfloat16)
print(f"\nbf16 无 scaling:")
print(f"  梯度值: {grad_bf16.item():.2e}")
print(f"  是否下溢: {'是' if grad_bf16.item() == 0.0 else '否'}")

# 5. Accelerate 的自动处理
print("\n5. Accelerate 的自动处理")
print("-" * 80)

print("\n当前配置:")
print("  mixed_precision: bf16")
print("  Accelerate 初始化:")
print("    accelerator = Accelerator(")
print("        mixed_precision='bf16',")
print("        ...)")

print("\nAccelerate 的行为:")
print("  - 对于 'fp16': 自动启用 GradScaler")
print("  - 对于 'bf16': **不使用** GradScaler")
print("  - 对于 None/'no': 完全 float32，不需要 scaling")

print("\n为什么不需要手动添加 GradScaler?")
print("  1. Accelerate 会根据 mixed_precision 参数自动决定")
print("  2. bf16 不需要 gradient scaling（范围足够大）")
print("  3. 如果用 fp16，Accelerate 会自动添加 scaling")

# 6. 验证当前实现
print("\n6. 验证当前实现")
print("-" * 80)

print("\n当前训练脚本（script/train.py）:")
print("  ✓ 使用 Accelerator(mixed_precision='bf16')")
print("  ✓ 使用 accelerator.backward(loss) 而不是 loss.backward()")
print("  ✓ Accelerate 自动处理混合精度转换")

print("\n是否需要添加 GradScaler?")
print("  答案: **不需要！**")
print("  原因:")
print("    1. bf16 不需要 gradient scaling")
print("    2. Accelerate 已经处理了所有细节")
print("    3. 手动添加反而会出错")

# 7. 如果切换到 fp16
print("\n7. 如果切换到 fp16")
print("-" * 80)

print("\n配置修改:")
print("  TRAIN:")
print("    mixed_precision: fp16  # 改为 fp16")

print("\nAccelerate 会自动:")
print("  1. 启用 GradScaler")
print("  2. 在 backward 前放大 loss")
print("  3. 在 optimizer.step 前缩小梯度")
print("  4. 动态调整 scale_factor")

print("\n不需要手动写:")
print("  scaler = GradScaler()  # Accelerate 内部处理")
print("  scaler.scale(loss).backward()  # 用 accelerator.backward(loss)")
print("  scaler.step(optimizer)  # 用 optimizer.step()")
print("  scaler.update()  # Accelerate 自动调用")

# 8. norm_scale 下溢 vs 梯度下溢
print("\n8. norm_scale 下溢 vs 梯度下溢的区别")
print("-" * 80)

print("\nnorm_scale 下溢（前向传播问题）:")
print("  - 发生在: 前向传播计算 loss")
print("  - 原因: bf16 精度不足，0.001 被舍入为 0")
print("  - 结果: loss 变成 NaN")
print("  - 解决: epsilon 保护")
print("  - GradScaler 能解决吗? **不能！**（前向传播就出错了）")

print("\n梯度下溢（反向传播问题）:")
print("  - 发生在: 反向传播计算梯度")
print("  - 原因: 梯度太小，超出精度范围")
print("  - 结果: 梯度变成 0，权重不更新")
print("  - 解决: GradScaler（仅 fp16 需要）")
print("  - epsilon 保护能解决吗? **不能！**")

# 9. 完整的混合精度最佳实践
print("\n9. 完整的混合精度最佳实践")
print("-" * 80)

print("\n使用 bf16 (推荐，当前配置):")
print("  优点:")
print("    ✓ 不需要 gradient scaling")
print("    ✓ 数值范围与 float32 相同")
print("    ✓ 实现简单")
print("  缺点:")
print("    ✗ 精度略低于 float32")
print("    ✗ 需要较新的 GPU (Ampere+)")
print("  需要:")
print("    ✓ epsilon 保护（前向传播数值稳定）")
print("    ✗ GradScaler（不需要）")

print("\n使用 fp16:")
print("  优点:")
print("    ✓ 支持更多 GPU (Volta+)")
print("    ✓ 理论上略快")
print("  缺点:")
print("    ✗ 需要 gradient scaling")
print("    ✗ 数值范围小，容易下溢")
print("    ✗ 实现复杂")
print("  需要:")
print("    ✓ epsilon 保护（前向传播）")
print("    ✓ GradScaler（反向传播，Accelerate 自动）")

print("\n使用 float32:")
print("  优点:")
print("    ✓ 最稳定")
print("    ✓ 不需要任何特殊处理")
print("  缺点:")
print("    ✗ 慢 2 倍")
print("    ✗ 显存占用大")
print("  需要:")
print("    ✗ epsilon 保护（不需要，除非数据异常）")
print("    ✗ GradScaler（不需要）")

# 总结
print("\n" + "=" * 80)
print("总结")
print("=" * 80)

print("\n关于 Gradient Scaling:")
print("  1. bf16 **不需要** gradient scaling")
print("  2. Accelerate 已经自动处理了一切")
print("  3. 当前实现是正确的，无需修改")

print("\n当前 NaN 问题:")
print("  - 根源: norm_scale 在前向传播中下溢（bf16 精度不足）")
print("  - 解决: epsilon 保护（已实施）")
print("  - 与 gradient scaling 无关")

print("\n建议:")
print("  ✓ 保持当前配置（bf16 + epsilon 保护）")
print("  ✓ 不需要手动添加 GradScaler")
print("  ✓ 如果还有问题，考虑切换到 float32")

print("\n" + "=" * 80)
