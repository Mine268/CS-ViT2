# norm_by_hand 实现文档

## 概述

`norm_by_hand` 是一种手部尺度归一化策略，用于提升模型的泛化性。核心思想：
- 使用手部骨骼长度作为归一化尺度
- 让模型输出的 trans（根关节位置）与手的大小无关
- 训练时在归一化空间进行监督，推理时反归一化恢复真实坐标

## 归一化尺度的计算

### norm_idx 定义

```python
# 从 model/smplx_models/mano/norm_stats.npz 加载
norm_list = ['Middle_1', 'Middle_2', 'Middle_3', 'Middle_4']  # 中指的4个关节

# 转换为索引（基于 HAND_JOINTS_ORDER）
norm_idx = [9, 10, 11, 12]
```

**关节对应**：
- 索引 9: Middle_1（中指近端关节）
- 索引 10: Middle_2（中指中间关节）
- 索引 11: Middle_3（中指远端关节）
- 索引 12: Middle_4（中指指尖）

### 计算公式

```python
def get_hand_norm_scale(j3d: Tensor, valid: Tensor) -> (Tensor, Tensor):
    """
    Args:
        j3d:   [..., 21, 3]  # 21个关节的3D坐标
        valid: [..., 21]     # 每个关节是否有效

    Returns:
        norm_scale: [...]    # 手部骨骼长度（标量）
        norm_valid: [...]    # 是否所有norm_idx关节都有效（0或1）
    """
    # 计算相邻关节间的欧氏距离
    d = j3d[..., norm_idx[:-1], :] - j3d[..., norm_idx[1:], :]
    # norm_idx[:-1] = [9, 10, 11]
    # norm_idx[1:]  = [10, 11, 12]
    # d = [j9-j10, j10-j11, j11-j12]  # 中指的3段骨骼

    # 累加所有骨骼长度
    norm_scale = sum(||d_i||_2)  # 中指总长度

    # 检查所有关节是否有效
    norm_valid = all(valid[norm_idx] > 0.5) ? 1.0 : 0.0

    return norm_scale, norm_valid
```

**关键点**：
- 计算的是中指的3段连续骨骼长度之和
- 骨骼长度由 MANO shape 唯一决定（pose 不改变骨骼固有长度）
- 典型值：成人手部 norm_scale ≈ 70-90mm

## 训练时的逻辑

### 流程图

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 从 GT 计算归一化尺度                                  │
├─────────────────────────────────────────────────────────────┤
│ norm_scale_gt, norm_valid_gt = get_hand_norm_scale(         │
│     batch["joint_cam"][:, -1:],   # GT关节                   │
│     batch["joint_valid"][:, -1:]  # GT有效性                 │
│ )                                                            │
└─────────────────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: 归一化 trans_gt                                      │
├─────────────────────────────────────────────────────────────┤
│ trans_gt = batch["joint_cam"][:, -1:, 0]  # 根关节位置       │
│ trans_gt_normalized = trans_gt / norm_scale_gt[..., None]   │
│ # 除以手的大小，得到无量纲的归一化trans                        │
└─────────────────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: 在归一化空间计算 trans loss                           │
├─────────────────────────────────────────────────────────────┤
│ loss_trans = L1(trans_pred, trans_gt_normalized)            │
│ loss_trans *= norm_valid_gt  # 只在归一化有效时计算           │
└─────────────────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: 反归一化用于重投影 loss                               │
├─────────────────────────────────────────────────────────────┤
│ trans_pred_scaled = trans_pred * norm_scale_gt[..., None]   │
│ # 乘以手的大小，恢复真实的mm单位                              │
│                                                              │
│ joint_cam_pred = joint_rel_pred + trans_pred_scaled         │
│ loss_joint_img = reproj_loss(joint_cam_pred, joint_cam_gt)  │
└─────────────────────────────────────────────────────────────┘
```

### 代码位置

- 实现：`src/model/loss.py:384-394` (get_hand_norm_scale)
- 训练逻辑：`src/model/loss.py:439-496` (BundleLoss2.forward)

### 关键设计

1. **trans 监督在归一化空间**：
   - 优点：与手大小无关，提高泛化性
   - 缺点：网络需要学习归一化空间的映射

2. **重投影 loss 在真实空间**：
   - 必须反归一化 trans_pred
   - 如果 norm_scale_gt 不准确，会导致重投影误差放大

3. **norm_valid masking**：
   - 如果 GT 关节缺失（如遮挡），norm_valid=0
   - loss_trans 不参与梯度更新

## 推理时的逻辑

### 流程图

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 模型预测归一化的 trans                                │
├─────────────────────────────────────────────────────────────┤
│ pose, shape, trans_pred, _ = predict_mano_param(...)        │
│ # trans_pred 是归一化的（无量纲）                             │
└─────────────────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: FK 得到相对关节位置                                   │
├─────────────────────────────────────────────────────────────┤
│ joint_rel_pred, vert_rel_pred = mano_to_pose(pose, shape)   │
│ # 相对于根关节的位置，包含手的大小信息                         │
└─────────────────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: 计算归一化尺度（优先 GT，fallback 到 pred）           │
├─────────────────────────────────────────────────────────────┤
│ # 3a. 从 GT 计算（如果可用）                                  │
│ norm_scale_gt, norm_valid_gt = get_hand_norm_scale(         │
│     joint_cam_gt, joint_valid_gt                            │
│ )                                                            │
│                                                              │
│ # 3b. 从预测计算（fallback）                                 │
│ norm_scale_pred, _ = get_hand_norm_scale(                   │
│     joint_rel_pred, torch.ones_like(...)                    │
│ )                                                            │
│                                                              │
│ # 3c. 优先 GT，无效时使用 pred                                │
│ norm_scale = (                                              │
│     norm_valid_gt * norm_scale_gt +                         │
│     (1 - norm_valid_gt) * norm_scale_pred                   │
│ )                                                            │
└─────────────────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: 反归一化得到真实坐标                                  │
├─────────────────────────────────────────────────────────────┤
│ trans_pred_denorm = trans_pred * norm_scale                 │
│ joint_cam_pred = joint_rel_pred + trans_pred_denorm         │
│ vert_cam_pred = vert_rel_pred + trans_pred_denorm          │
└─────────────────────────────────────────────────────────────┘
```

### 代码位置

- 实现：`src/model/net.py:251-261` (get_hand_norm_scale)
- 推理逻辑：`src/model/net.py:456-497` (predict_full)

### 关键设计

1. **测试集有 GT**：
   - 优先使用 GT 计算 norm_scale_gt
   - 保证反归一化的准确性

2. **Fallback 机制**：
   - 如果 GT 无效，使用预测的 joint_rel_pred 计算 norm_scale_pred
   - 依赖于 shape 预测的准确性

3. **最终输出**：
   - joint_cam_pred 和 vert_cam_pred 是真实相机坐标系（mm单位）
   - 可直接用于评估和可视化

## norm_scale 与 shape 的关系

### 理论分析

**norm_scale 由 shape 唯一决定**：

```python
# MANO forward kinematics
joint_rel = FK(shape, pose)

# norm_scale 计算
norm_scale = sum(||joint_rel[norm_idx[i]] - joint_rel[norm_idx[i+1]]||_2)
```

**为什么 pose 不影响 norm_scale？**

1. **骨骼的刚体性质**：
   - MANO 模型中，骨骼长度由 shape 决定
   - pose 只进行旋转变换（刚体变换）
   - 父子关节间的欧氏距离 = 骨骼固有长度

2. **中指的特殊性**：
   - norm_idx = [9, 10, 11, 12] 是连续的父子关节
   - Middle_1 → Middle_2 → Middle_3 → Middle_4 形成骨骼链
   - 旋转不改变相邻关节间的距离

**推论**：
- shape 学好了 → joint_rel 准确 → norm_scale 自动准确
- **不需要单独监督 norm_scale**

### 验证方式

```python
# 固定 shape，改变 pose
shape = torch.randn(1, 10)
pose_1 = torch.randn(1, 48)  # 手指弯曲
pose_2 = torch.randn(1, 48)  # 手指伸直

joint_rel_1 = FK(shape, pose_1)
joint_rel_2 = FK(shape, pose_2)

norm_scale_1 = get_hand_norm_scale(joint_rel_1, ...)
norm_scale_2 = get_hand_norm_scale(joint_rel_2, ...)

# 理论上：norm_scale_1 ≈ norm_scale_2（在数值误差范围内）
```

## 与重投影 loss 的关联

### 问题分析

重投影 loss 飙升的一个潜在原因：

```python
# 训练时
trans_pred_scaled = trans_pred * norm_scale_gt  # 反归一化

# 如果 norm_scale_gt 不准确（GT标注误差）：
# → trans_pred_scaled 误差被放大
# → joint_cam_pred = joint_rel_pred + trans_pred_scaled 误差大
# → loss_joint_img = reproj_loss(joint_cam_pred, joint_cam_gt) 飙升
```

### 解决方案

1. **RobustL1Loss**（已实现）：
   - 对异常的重投影误差进行梯度衰减
   - 配置：`LOSS.reproj_loss_type="robust_l1"`, `LOSS.reproj_loss_delta=84.0`
   - 详见 `docs/REPROJ_LOSS_CONFIG.md`

2. **监控 norm_scale_gt 分布**：
   - 检查是否存在异常值（过大或过小）
   - 可能的原因：GT 关节标注错误、遮挡导致的外推误差

3. **提高 shape 预测精度**：
   - 增强 lambda_shape 权重
   - shape 准确 → joint_rel 准确 → norm_scale fallback 更可靠

## 配置选项

### 启用/禁用

```yaml
# config/*.yaml
MODEL:
  norm_by_hand: true  # 启用归一化
  # norm_by_hand: false  # 禁用（直接预测真实 trans）
```

### 训练时的影响

| 配置 | trans 监督空间 | 重投影 loss | 泛化性 |
|------|---------------|-------------|--------|
| `true` | 归一化空间 | 需要反归一化 | 更好（手大小不变性） |
| `false` | 真实空间 | 直接使用 trans_pred | 较差（依赖训练集手大小分布） |

### 推理时的影响

| 配置 | 输出 trans | 依赖 GT | 说明 |
|------|-----------|---------|------|
| `true` | 反归一化后的真实值 | 是（优先）| 需要 GT 或依赖 shape 预测 |
| `false` | 模型直接输出 | 否 | 与训练一致 |

## 潜在问题和注意事项

### 1. norm_idx 硬编码

**问题**：
- 当前 norm_idx 从 `norm_stats.npz` 的 norm_list 动态生成
- 如果关节定义变化，需要同步更新

**建议**：
- 保持 HAND_JOINTS_ORDER 定义的稳定性
- 如果修改关节定义，重新生成 norm_stats.npz

### 2. 训练和推理的不一致性

**问题**：
- 训练：norm_scale 总是从 GT 计算（因为有 GT）
- 推理：优先 GT，但可能 fallback 到 pred

**影响**：
- 如果 pred 的 norm_scale 不准确，影响最终精度
- 对于没有 GT 的真实应用场景，完全依赖 shape 预测

**建议**：
- 加强 shape 监督（lambda_shape）
- 在测试时监控 norm_scale_pred 与 norm_scale_gt 的差异

### 3. norm_valid 的使用

**问题**：
- 如果 GT 关节缺失（如遮挡），norm_valid=0
- 训练时 loss_trans 被 mask 掉
- 推理时回退到可能不准确的 norm_scale_pred

**影响**：
- 遮挡情况下的 trans 预测可能不可靠

**建议**：
- 数据增强时保证中指关节的可见性
- 或者考虑使用其他手指作为备选（如食指）

### 4. 中指的代表性

**问题**：
- 为什么选择中指？是否具有代表性？

**分析**：
- 中指通常是最长的手指，长度变化较稳定
- 但不同人的手指比例可能不同
- 单一手指可能不能完全代表整个手的尺度

**建议**：
- 考虑使用多个手指的平均值（如中指+食指）
- 或者使用手掌宽度等其他特征

### 5. 数值稳定性

**问题**：
- norm_scale_gt 接近 0 时，归一化和反归一化可能出现数值问题
- **已知 Bug**：在 2026-02-11 的训练中，Step 1070 出现 NaN，原因是缺少除零保护

**修复**（已实施）：
- 在 `src/model/loss.py` 和 `src/model/net.py` 添加 `NORM_SCALE_EPSILON = 1e-6`
- 归一化：`trans_gt = trans_gt / (norm_scale_gt + NORM_SCALE_EPSILON)`
- 反归一化：`trans_pred_scaled = trans_pred * (norm_scale_gt + NORM_SCALE_EPSILON)`
- get_hand_norm_scale：`d = torch.clamp(d, min=NORM_SCALE_EPSILON)`

**详见**：[NAN_TRAINING_FIX.md](NAN_TRAINING_FIX.md)

## 相关文件

- **实现**：
  - `src/model/net.py:147` (norm_idx 初始化)
  - `src/model/net.py:251-261` (get_hand_norm_scale)
  - `src/model/net.py:456-497` (推理逻辑)
  - `src/model/loss.py:380` (norm_idx 传递)
  - `src/model/loss.py:384-394` (get_hand_norm_scale)
  - `src/model/loss.py:439-496` (训练逻辑)

- **配置**：
  - `model/smplx_models/mano/norm_stats.npz` (norm_list 定义)
  - `src/constant.py:5-27` (HAND_JOINTS_ORDER)

- **测试**：
  - `tests/test_predict_full_norm_by_hand.py`
  - `tests/check_root_joint_range.py:85-93`

- **分析工具**：
  - `tools/analyze_norm_by_hand.py` (详细分析脚本)

## 总结

norm_by_hand 是一种有效的尺度归一化策略：
- **优点**：提高泛化性，减少对训练集手大小分布的依赖
- **核心**：norm_scale 由 shape 唯一决定，通过中指的3段骨骼长度计算
- **权衡**：需要反归一化，依赖 GT 或 shape 预测的准确性
- **注意**：与重投影 loss 的交互，可能放大标注误差

GG
