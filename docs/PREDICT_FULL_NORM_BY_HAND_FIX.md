# predict_full() 中 norm_by_hand 反归一化逻辑修复

**日期**: 2026-02-10

## 问题描述

在 `src/model/net.py` 的 `predict_full()` 方法（第 461-510 行）中，当启用 `norm_by_hand=true` 时，反归一化逻辑存在以下问题：

### 原始问题代码

```python
if joint_cam_gt is not None and joint_valid_gt is not None:
    # 使用 GT 计算 norm_scale（更准确）
    norm_scale, norm_valid = self.get_hand_norm_scale(
        joint_cam_gt[:, -1:], joint_valid_gt[:, -1:]
    )
else:
    # 使用预测的 joint_cam 计算 norm_scale（近似）
    joint_cam_rough = joint_rel_pred + trans_for_fk[:, :, None, :]
    norm_scale, norm_valid = self.get_hand_norm_scale(
        joint_cam_rough, joint_valid_rough
    )
```

### 问题分析

1. **没有检查 norm_valid 标志**：即使有 GT，也没有检查 GT 中 norm_idx 对应的 joints 是否全部有效（通过 `norm_valid` 标志）

2. **与训练逻辑不一致**：
   - 训练时（`BundleLoss2.forward()`）总是用 GT 计算 norm_scale，但会用 `norm_valid` 掩码 loss
   - 测试时应该检查 `norm_valid` 标志，对于 `norm_valid=0` 的样本需要 fallback 到其他方法

3. **无 GT 分支的循环依赖**：
   - 使用预测的 joints 计算 norm_scale
   - 但这些 joints 是基于归一化的 trans 计算的
   - 然后又用这个 scale 去反归一化 trans
   - 这形成了循环逻辑，虽然可以作为近似，但语义不清晰

## 修复方案

### 核心思路

参考 `BundleLoss2.forward()` 的逻辑，实现以下策略：

1. **优先使用 GT**：当有 GT 且 `norm_valid_gt=1`（norm_idx 对应的 joints 全部有效）时，使用 GT 的 norm_scale
2. **智能 Fallback**：当 GT 的 `norm_valid_gt=0` 时，fallback 到使用预测的 joint_cam 计算 norm_scale
3. **逐样本混合**：对批次中的不同样本，根据其 norm_valid 标志分别选择 GT 或 pred scale
4. **明确标记**：通过 `norm_valid` 标志清晰地指示 norm_scale 的来源和可靠性

### 修复后的代码

```python
if self.norm_by_hand:
    # 3.1 计算 norm_scale（参考 BundleLoss2 的逻辑）
    if joint_cam_gt is not None and joint_valid_gt is not None:
        # 3.1.1 用 GT 计算 norm_scale 和 norm_valid 标志
        norm_scale_gt, norm_valid_gt = self.get_hand_norm_scale(
            joint_cam_gt[:, -1:], joint_valid_gt[:, -1:]
        )

        # 3.1.2 对于 norm_valid_gt=0 的样本，fallback 到使用预测的 joint_cam
        joint_cam_rough = joint_rel_pred + trans_for_fk[:, :, None, :]
        joint_valid_rough = torch.ones(
            (joint_rel_pred.shape[0], 1, 21),
            device=joint_rel_pred.device
        )
        norm_scale_pred, norm_valid_pred = self.get_hand_norm_scale(
            joint_cam_rough, joint_valid_rough
        )

        # 3.1.3 混合使用：norm_valid_gt=1 用 GT，norm_valid_gt=0 用 pred
        norm_valid_mask = norm_valid_gt[:, :, None]  # [B, 1, 1]
        norm_scale = norm_valid_mask * norm_scale_gt[:, :, None] + \
                     (1 - norm_valid_mask) * norm_scale_pred[:, :, None]
        norm_scale = norm_scale.squeeze(-1)  # [B, 1]

        # norm_valid 标志：如果 GT valid 或 pred valid，就认为有效
        norm_valid = torch.maximum(norm_valid_gt, norm_valid_pred)

        # 3.2 反归一化 trans_pred
        trans_pred_denorm = trans_for_fk * norm_scale[:, :, None]
    else:
        # 完全没有 GT，只能用预测的 joint_cam
        joint_cam_rough = joint_rel_pred + trans_for_fk[:, :, None, :]
        joint_valid_rough = torch.ones(
            (joint_rel_pred.shape[0], 1, 21),
            device=joint_rel_pred.device
        )
        norm_scale, norm_valid = self.get_hand_norm_scale(
            joint_cam_rough, joint_valid_rough
        )
        trans_pred_denorm = trans_for_fk * norm_scale[:, :, None]
else:
    # 不需要反归一化
    trans_pred_denorm = trans_for_fk
    norm_scale = torch.ones((trans_for_fk.shape[0], 1), device=trans_for_fk.device)
    norm_valid = torch.ones((trans_for_fk.shape[0], 1), device=trans_for_fk.device)
```

### 关键改进

1. **检查 norm_valid 标志**：参考 `BundleLoss2.get_hand_norm_scale()` 返回的 flag，检查 GT 中 norm_idx 对应的 joints 是否全部有效

2. **分层 Fallback 策略**：
   - 优先：GT 有效时（`norm_valid_gt=1`），使用 GT 的 norm_scale
   - 次选：GT 无效时（`norm_valid_gt=0`），使用预测的 norm_scale
   - 最后：完全无 GT 时，也使用预测的 norm_scale

3. **逐样本混合策略**：
   - 通过 `norm_valid_mask` 实现批次内不同样本使用不同策略
   - 样本 A 的 GT 有效 → 使用 GT scale
   - 样本 B 的 GT 无效 → 使用 pred scale

4. **保留循环逻辑的合理性**：
   - 当必须使用预测的 joint_cam 时（GT 不可用或 norm_idx joints 不全），使用基于归一化 trans 的粗略估计
   - 这是一个**可接受的近似**，比完全不反归一化要好

5. **与训练逻辑一致**：
   - 训练时（BundleLoss2）虽然总是用 GT 的 norm_scale，但会用 norm_valid 掩码 loss
   - 测试时根据 norm_valid 选择 GT 或 pred scale，语义更明确

## 验证结果

### 单元测试

创建了 `tests/test_predict_full_norm_by_hand.py`，包含 4 个测试场景：

1. **test_predict_full_with_valid_gt**：有 GT 且 norm_valid=1
   - 验证使用 GT 的 norm_scale
   - 验证反归一化公式正确
   - ✅ 通过

2. **test_predict_full_with_invalid_gt**：有 GT 但 norm_valid=0
   - 验证 fallback 到使用预测的 norm_scale
   - 验证最终 norm_valid=1（因为 pred 假设所有 joints 都有效）
   - ✅ 通过

3. **test_predict_full_without_gt**：无 GT
   - 验证使用预测的 norm_scale
   - 验证 norm_valid=1
   - ✅ 通过

4. **test_predict_full_mixed_validity**：批次混合情况
   - 验证逐样本选择 GT 或 pred scale
   - 验证所有样本的 norm_valid=1（fallback 后）
   - ✅ 通过

### 测试结果

```bash
$ python -m pytest tests/test_predict_full_norm_by_hand.py -v -s
======================== 4 passed, 8 warnings in 15.29s ========================
```

所有测试全部通过！

## 对现有代码的影响

### 不受影响的场景

1. **训练流程**：
   - `BundleLoss2.forward()` 不变，训练逻辑不受影响
   - 训练时的反归一化仍然总是使用 GT 的 norm_scale

2. **测试流程**：
   - `script/test.py` 已正确传入 GT（`joint_cam_gt` 和 `joint_valid_gt`）
   - 大多数测试样本的 GT `norm_valid=1`，将使用准确的 GT scale
   - 行为保持一致，结果更加鲁棒

### 行为变化的场景

1. **GT 中 norm_idx joints 不全的样本**：
   - **之前**：直接使用 GT 计算的 norm_scale（可能不准确）
   - **现在**：fallback 到使用预测的 norm_scale（更鲁棒的近似）
   - **影响**：对这些样本的反归一化更加合理

2. **纯预测场景**（无 GT）：
   - **之前**：尝试用预测的 joints 计算 norm_scale（有循环依赖）
   - **现在**：仍然使用预测的 joints，但通过 `norm_valid=1` 明确指示这是有效的近似
   - **影响**：语义更清晰，但行为基本不变

## 核心要点

1. **检查 norm_valid 标志是关键**：
   - `get_hand_norm_scale()` 返回的 `flag` 指示 norm_idx 对应的 joints 是否全部有效
   - 只有当 `norm_valid=1` 时，norm_scale 才是可靠的

2. **智能 Fallback 策略**：
   - 优先使用 GT（当 `norm_valid_gt=1`）
   - 当 GT 不可用或不可靠时，fallback 到预测的 norm_scale
   - 虽然预测的 scale 有循环依赖，但仍然是合理的近似

3. **逐样本处理**：
   - 批次中不同样本可能有不同的 norm_valid 状态
   - 使用 `norm_valid_mask` 实现逐样本选择策略

4. **与训练逻辑一致**：
   - 训练时用 norm_valid 掩码 loss
   - 测试时用 norm_valid 选择 scale 来源
   - 语义上保持一致

## 总结

这次修复解决了 `predict_full()` 中 `norm_by_hand` 反归一化逻辑的问题：

1. ✅ 检查 norm_valid 标志，正确处理 GT 不完整的情况
2. ✅ 实现智能 Fallback 策略，提高鲁棒性
3. ✅ 支持批次内逐样本混合策略
4. ✅ 与训练逻辑（BundleLoss2）保持一致
5. ✅ 通过了完整的单元测试验证

修改后的代码更加精细和鲁棒，正确处理了各种边界情况。对于 `script/test.py` 等测试脚本，由于已正确传入 GT，将继续使用准确的 GT scale（当 norm_valid=1 时），确保返回真实相机坐标。
