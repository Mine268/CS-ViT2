#!/usr/bin/env python3
"""
分析 norm_by_hand 的实现逻辑

norm_by_hand 是一个归一化策略，用手部的骨骼长度来归一化 trans（根关节的位置）。
这样可以使模型输出的 trans 与手的大小无关，提高泛化性。
"""
import torch
import numpy as np

print("=" * 80)
print("norm_by_hand 实现逻辑分析")
print("=" * 80)

# ============================================================================
# 1. get_hand_norm_scale 函数分析
# ============================================================================
print("\n" + "=" * 80)
print("1. get_hand_norm_scale() - 计算手部归一化尺度")
print("=" * 80)

print("""
函数签名:
    def get_hand_norm_scale(j3d: Tensor, valid: Tensor) -> (Tensor, Tensor)

输入:
    j3d:   [..., 21, 3]  # 21个关节的3D坐标
    valid: [..., 21]     # 每个关节是否有效

输出:
    norm_scale: [...]    # 手部骨骼长度（标量）
    norm_valid: [...]    # 是否所有norm_idx关节都有效（0或1）

实现逻辑:
    1. norm_idx = [0, 5, 9, 13, 17]  # 5个指尖关节索引
    2. 计算相邻关节间的欧氏距离:
       d = j3d[norm_idx[:-1]] - j3d[norm_idx[1:]]
       即: [j0-j5, j5-j9, j9-j13, j13-j17]
    3. 累加所有距离:
       norm_scale = sum(||d_i||_2)
    4. 检查所有norm_idx关节是否都有效:
       norm_valid = all(valid[norm_idx] > 0.5) ? 1.0 : 0.0
""")

# 模拟示例
print("\n示例计算:")
print("-" * 80)

# 创建假的关节数据
joints = torch.randn(1, 1, 21, 3) * 100  # [b, t, j, 3]
joints_valid = torch.ones(1, 1, 21)      # 所有关节都有效

norm_idx = [0, 5, 9, 13, 17]

# 计算 norm_scale
d = joints[..., norm_idx[:-1], :] - joints[..., norm_idx[1:], :]
distances = torch.sqrt(torch.sum(d ** 2, dim=-1))  # [b, t, 4]
norm_scale = torch.sum(distances, dim=-1)           # [b, t]

# 计算 norm_valid
norm_valid = torch.all(joints_valid[:, :, norm_idx] > 0.5, dim=-1).float()

print(f"输入关节形状: {joints.shape}")
print(f"norm_idx: {norm_idx}")
print(f"各段骨骼长度: {distances[0, 0].numpy()}")
print(f"总骨骼长度 (norm_scale): {norm_scale[0, 0].item():.2f} mm")
print(f"是否有效 (norm_valid): {norm_valid[0, 0].item()}")

# ============================================================================
# 2. 训练时的 norm_by_hand 逻辑（BundleLoss2.forward）
# ============================================================================
print("\n" + "=" * 80)
print("2. 训练时的 norm_by_hand 逻辑（loss.py:425-498）")
print("=" * 80)

print("""
流程:
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: 计算 GT 的归一化尺度                                      │
├─────────────────────────────────────────────────────────────────┤
│ if norm_by_hand:                                                │
│     norm_scale_gt, norm_valid_gt = get_hand_norm_scale(         │
│         batch["joint_cam"][:, -1:],  # GT关节                    │
│         batch["joint_valid"][:, -1:]  # GT关节有效性             │
│     )                                                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Step 2: 归一化 trans_gt                                          │
├─────────────────────────────────────────────────────────────────┤
│ trans_gt = batch["joint_cam"][:, -1:, 0]  # 根关节位置 [b,1,3]   │
│ if norm_by_hand:                                                │
│     trans_gt = trans_gt / norm_scale_gt[..., None]              │
│     # 除以手的大小，得到归一化的trans                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Step 3: 计算 trans loss（在归一化空间）                           │
├─────────────────────────────────────────────────────────────────┤
│ loss_trans = L1(trans_pred, trans_gt)  # 归一化后的trans        │
│ loss_trans *= norm_valid_gt  # 只在归一化有效时计算              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Step 4: 反归一化用于投影loss（恢复真实尺度）                       │
├─────────────────────────────────────────────────────────────────┤
│ if norm_by_hand:                                                │
│     trans_pred_scaled = trans_pred * norm_scale_gt[..., None]   │
│     # 乘以手的大小，恢复真实的trans                               │
│ else:                                                           │
│     trans_pred_scaled = trans_pred                              │
│                                                                 │
│ joint_cam_pred = joint_rel_pred + trans_pred_scaled             │
│ # 用于计算重投影loss，需要真实尺度                                │
└─────────────────────────────────────────────────────────────────┘

关键点:
  1. trans 的监督在归一化空间进行（与手大小无关）
  2. 重投影loss在真实空间进行（需要反归一化）
  3. norm_valid_gt 用于mask无效样本
""")

# 模拟训练过程
print("\n示例计算:")
print("-" * 80)

# 模拟数据
joint_cam_gt = torch.randn(2, 1, 21, 3) * 100 + 500  # [b, t, j, 3]，深度约500mm
joint_valid_gt = torch.ones(2, 1, 21)
trans_pred = torch.randn(2, 1, 3) * 0.5  # 归一化的预测

# 计算 norm_scale_gt
norm_idx = [0, 5, 9, 13, 17]
d = joint_cam_gt[..., norm_idx[:-1], :] - joint_cam_gt[..., norm_idx[1:], :]
norm_scale_gt = torch.sum(torch.sqrt(torch.sum(d ** 2, dim=-1)), dim=-1)  # [b, t]
norm_valid_gt = torch.all(joint_valid_gt[:, :, norm_idx] > 0.5, dim=-1).float()

# 归一化 trans_gt
trans_gt_raw = joint_cam_gt[:, :, 0]  # 根关节位置
trans_gt_normalized = trans_gt_raw / norm_scale_gt[..., None]

# 反归一化 trans_pred 用于投影
trans_pred_scaled = trans_pred * norm_scale_gt[..., None]

print(f"GT手部大小 (norm_scale_gt): {norm_scale_gt[0, 0].item():.2f} mm")
print(f"GT根关节位置 (原始): {trans_gt_raw[0, 0].numpy()}")
print(f"GT根关节位置 (归一化): {trans_gt_normalized[0, 0].numpy()}")
print(f"预测trans (归一化): {trans_pred[0, 0].numpy()}")
print(f"预测trans (反归一化): {trans_pred_scaled[0, 0].numpy()}")
print(f"\n监督信号: L1(trans_pred, trans_gt_normalized) 在归一化空间")
print(f"投影计算: joint_cam_pred = joint_rel + trans_pred_scaled 在真实空间")

# ============================================================================
# 3. 推理时的 norm_by_hand 逻辑（PoseNet.predict_full）
# ============================================================================
print("\n" + "=" * 80)
print("3. 推理时的 norm_by_hand 逻辑（net.py:456-481）")
print("=" * 80)

print("""
流程:
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: 模型预测归一化的 trans                                    │
├─────────────────────────────────────────────────────────────────┤
│ pose, shape, trans_pred, _ = predict_mano_param(...)            │
│ # trans_pred 是归一化的（如果 norm_by_hand=True）                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Step 2: FK得到相对关节位置                                        │
├─────────────────────────────────────────────────────────────────┤
│ joint_rel_pred, vert_rel_pred = mano_to_pose(pose, shape)       │
│ # 相对于根关节的位置，与手大小有关                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Step 3: 计算归一化尺度（优先使用GT，fallback到pred）               │
├─────────────────────────────────────────────────────────────────┤
│ if norm_by_hand:                                                │
│     # 3a. 从GT计算                                               │
│     norm_scale_gt, norm_valid_gt = get_hand_norm_scale(         │
│         joint_cam_gt, joint_valid_gt                            │
│     )                                                            │
│                                                                 │
│     # 3b. 从预测计算（fallback）                                 │
│     norm_scale_pred, _ = get_hand_norm_scale(                   │
│         joint_rel_pred, torch.ones_like(...)                    │
│     )                                                            │
│                                                                 │
│     # 3c. 优先GT，无效时使用pred                                 │
│     norm_scale = (                                              │
│         norm_valid_gt * norm_scale_gt +                         │
│         (1 - norm_valid_gt) * norm_scale_pred                   │
│     )                                                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Step 4: 反归一化得到真实坐标                                       │
├─────────────────────────────────────────────────────────────────┤
│ trans_pred_denorm = trans_pred * norm_scale                     │
│ joint_cam_pred = joint_rel_pred + trans_pred_denorm             │
│ vert_cam_pred = vert_rel_pred + trans_pred_denorm              │
└─────────────────────────────────────────────────────────────────┘

关键点:
  1. 推理时需要GT来反归一化（测试集有GT）
  2. 如果GT无效，使用预测的关节计算norm_scale
  3. 最终输出是真实相机坐标系的joints/verts
""")

# 模拟推理过程
print("\n示例计算:")
print("-" * 80)

# 模拟预测
trans_pred = torch.randn(1, 1, 3) * 0.5  # 归一化的预测
joint_rel_pred = torch.randn(1, 1, 21, 3) * 50  # FK结果

# 情况1: GT可用
joint_cam_gt = torch.randn(1, 1, 21, 3) * 100 + 500
joint_valid_gt = torch.ones(1, 1, 21)

d = joint_cam_gt[..., norm_idx[:-1], :] - joint_cam_gt[..., norm_idx[1:], :]
norm_scale_gt = torch.sum(torch.sqrt(torch.sum(d ** 2, dim=-1)), dim=-1)
norm_valid_gt = torch.all(joint_valid_gt[:, :, norm_idx] > 0.5, dim=-1).float()

trans_denorm_with_gt = trans_pred * norm_scale_gt[..., None]

print("情况1: GT可用")
print(f"  GT手部大小: {norm_scale_gt[0, 0].item():.2f} mm")
print(f"  归一化trans: {trans_pred[0, 0].numpy()}")
print(f"  反归一化trans: {trans_denorm_with_gt[0, 0].numpy()}")

# 情况2: GT不可用，使用预测
d = joint_rel_pred[..., norm_idx[:-1], :] - joint_rel_pred[..., norm_idx[1:], :]
norm_scale_pred = torch.sum(torch.sqrt(torch.sum(d ** 2, dim=-1)), dim=-1)

trans_denorm_with_pred = trans_pred * norm_scale_pred[..., None]

print("\n情况2: GT不可用（fallback到预测）")
print(f"  预测手部大小: {norm_scale_pred[0, 0].item():.2f} mm")
print(f"  归一化trans: {trans_pred[0, 0].numpy()}")
print(f"  反归一化trans: {trans_denorm_with_pred[0, 0].numpy()}")

# ============================================================================
# 4. 潜在问题分析
# ============================================================================
print("\n" + "=" * 80)
print("4. 潜在问题和注意事项")
print("=" * 80)

print("""
问题1: norm_idx 硬编码
  - 当前: norm_idx = [0, 5, 9, 13, 17]
  - 含义: [手腕, 食指尖, 中指尖, 无名指尖, 小指尖]
  - 风险: 如果关节索引定义变化，需要同步修改

问题2: 训练和推理的不一致性
  - 训练: norm_scale 总是从GT计算（因为有GT）
  - 推理: 优先GT，但可能fallback到pred
  - 风险: 如果pred的norm_scale不准确，会影响最终精度

问题3: norm_valid 的使用
  - 训练: norm_valid_gt 用于mask loss
  - 推理: norm_valid_gt 用于选择GT或pred的norm_scale
  - 风险: 如果GT关节缺失（如遮挡），可能回退到不准确的pred

问题4: 重投影loss的依赖
  - 重投影loss需要反归一化的trans（真实尺度）
  - 如果norm_scale_gt不准确，会影响重投影loss的梯度
  - 这就是为什么loss_joint_img会突然飙升的原因之一

问题5: 网络输出空间
  - 网络输出的trans是归一化的（无量纲或相对单位）
  - 需要乘以norm_scale才能得到真实的mm单位
  - 如果网络没有学好归一化空间的映射，会导致预测不准

建议:
  1. 在训练时监控 norm_scale_gt 的分布，确保合理
  2. 在推理时优先使用GT的norm_scale（如果可用）
  3. 考虑添加 norm_scale 的监督（让模型直接预测手大小）
  4. 检查 norm_idx 是否与实际关节定义一致
""")

print("\n" + "=" * 80)
print("分析完成")
print("=" * 80)
