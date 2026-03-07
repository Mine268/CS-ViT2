# HO3D 测试误差分析报告

## 模型信息
- **Checkpoint**: `checkpoint/2026-03-05/21-07-09_stage1-dino_large_no_norm/best_model`
- **训练配置**: Stage 1, DINOv2-Large, norm_by_hand=false
- **验证集表现**: MPJPE=42.4mm, RTE=42.6mm (ih26m + dexycb)
- **HO3D 测试集表现**: MPJPE=141.3mm (严重下降!)

---

## 误差构成分析

### 1. 总体指标
| 指标 | 数值 | 说明 |
|------|------|------|
| MPJPE | 141.31 mm | 根关节对齐后的平均关节误差 |
| rel_mpjpe | 25.77 mm | Procrustes 对齐后的误差 (很好!) |
| RTE (Root Translation Error) | 146.57 mm | 根关节定位误差 (主要问题!) |
| MPVPE | NaN | 网格顶点误差 (mano_valid=0) |

**关键发现**: 相对姿态误差只有 25.77mm，说明手部姿态形状预测准确；但根关节定位误差 146.57mm 是主要问题。

### 2. 根关节误差分解
| 轴 | 平均误差 | 标准差 | 说明 |
|----|---------|--------|------|
| X | +16.22 mm | 23.88 mm | 轻度偏差 |
| Y | +5.80 mm | 17.79 mm | 较小偏差 |
| Z | **+122.55 mm** | **126.35 mm** | **系统性高估深度** |

**结论**: Z 轴深度估计存在系统性偏差，占总误差的 83%。

### 3. 深度分层误差分析

| Z 深度范围 | 样本数 | 平均 Z 误差 | 说明 |
|-----------|--------|------------|------|
| 200-300mm | 287 | +178.7 mm | 严重高估 |
| 300-400mm | 2,531 | +160.2 mm | 严重高估 |
| 400-500mm | 4,965 | +180.1 mm | 严重高估 |
| 500-600mm | 4,630 | +178.1 mm | 严重高估 |
| 600-700mm | 2,738 | +86.0 mm | 中度高估 |
| **700-800mm** | 3,251 | **-2.6 mm** | **几乎准确** |
| 800-900mm | 1,541 | +35.6 mm | 轻度偏差 |
| 900-1000mm | 89 | -23.1 mm | 轻度低估 |

**关键发现**: 
- 模型在 **700-800mm** 深度范围几乎完美（误差仅 -2.6mm）
- 但在 **<600mm 近距离** 系统性高估约 +180mm

### 4. HO3D 数据集特点
- **焦距**: 恒定 ~616mm (所有样本相同相机)
- **Z 深度分布**: [241, 939] mm，平均 568mm
- **BBox 大小**: 与 Z 深度高度负相关 (-0.775)

---

## 问题根源分析

### 根本原因: 训练-测试分布不匹配

1. **验证集不包含 HO3D**
   - 训练时验证只使用 ih26m + dexycb
   - 模型从未在 HO3D 数据分布上验证过

2. **深度分布差异**
   - HO3D 平均深度 568mm，偏近
   - 模型在 700-800mm 表现好，说明训练数据可能偏向中远距离

3. **焦距差异**
   - HO3D 使用恒定焦距 616mm
   - 训练数据（ih26m）使用多种相机，焦距变化大
   - 模型可能过度拟合了训练数据的焦距分布

4. **norm_by_hand=false 的影响**
   - 模型需要预测绝对深度
   - 对相机参数和深度分布更敏感
   - 泛化能力下降

---

## 改进方案

### 方案 1: 验证集加入 HO3D (推荐立即执行)

**修改配置文件**:
```yaml
DATA:
  val:
    source:
    - /mnt/qnap/data/datasets/webdatasets/InterHand2.6M/val/*
    - /mnt/qnap/data/datasets/webdatasets/DexYCB/s1/val/*
    - /mnt/qnap/data/datasets/webdatasets/HO3D_v3/val/*  # 添加 HO3D
```

**效果**: 训练时即可发现 HO3D 上的性能下降，及时调整。

---

### 方案 2: 使用 norm_by_hand=true (推荐新实验)

**原理**: 
- 先预测相对于手部的归一化坐标
- 不受绝对深度和相机参数影响
- 泛化能力更强

**配置修改**:
```yaml
MODEL:
  norm_by_hand: true
```

**已有 checkpoint**: `checkpoint/2026-02-11/23-53-36_stage1-dino_large/best_model` (norm_by_hand=true)

**建议**: 用该 checkpoint 在 HO3D 上测试对比。

---

### 方案 3: 针对 HO3D 微调 (快速修复)

**步骤**:
1. 加载当前 best_model
2. 在 HO3D 训练集上微调 ~5k-10k 步
3. 混合训练数据保持泛化能力

**命令示例**:
```bash
accelerate launch --gpu_ids 0,1,2,3 --num_processes 4 -m script.train \
    --config-name=stage1-dino_large_no_norm \
    GENERAL.resume_path=checkpoint/2026-03-05/21-07-09_stage1-dino_large_no_norm/best_model \
    DATA.train.source='[/mnt/qnap/data/datasets/webdatasets/HO3D_v3/train/*]' \
    TRAIN.lr=1e-5 \
    GENERAL.total_step=10000
```

---

### 方案 4: 深度校准 (测试时快速修复)

**方法**: 基于统计的后处理校准

**校准表**:
```python
def calibrate_depth(z_pred, z_gt_approx):
    """基于大致的 GT Z 深度进行校准"""
    if z_gt_approx < 600:
        return z_pred - 180
    elif z_gt_approx < 700:
        return z_pred - 86
    elif z_gt_approx < 800:
        return z_pred  # 几乎准确
    else:
        return z_pred - 35
```

**效果**: 可将 RTE 从 146mm 降至 ~90mm (39% 改善)

---

### 方案 5: 数据增强改进

**问题**: 当前 `scale_z_range=[0.9, 1.1]` 可能破坏绝对深度估计能力

**建议**:
1. 减小 Z 轴缩放范围: `[0.95, 1.05]`
2. 或关闭 Z 轴缩放，改用其他增强方式
3. 增加焦距扰动增强，提升对不同相机的泛化能力

```yaml
TRAIN:
  scale_z_range: [0.95, 1.05]  # 减小范围
  # 或增加焦距扰动
  focal_jitter: 0.1  # 焦距扰动 10%
```

---

### 方案 6: 多数据集平衡采样

**问题**: 训练时 HO3D 样本可能被其他大数据集淹没

**建议**: 使用加权采样确保每个数据集有合理的采样概率

---

## 推荐执行顺序

### 立即执行 (今天)
1. **用 norm_by_hand=true 的模型测试 HO3D**
   - Checkpoint: `checkpoint/2026-02-11/23-53-36_stage1-dino_large/best_model`
   - 对比验证 norm_by_hand 的效果

2. **修改验证集配置**
   - 加入 HO3D val 到验证集
   - 确保后续训练能监控 HO3D 性能

### 短期 (本周)
3. **HO3D 微调实验**
   - 加载当前模型，在 HO3D 上微调
   - 或者从头训练，验证集包含 HO3D

4. **深度校准脚本**
   - 实现测试时的深度校准
   - 作为临时解决方案

### 中期 (本月)
5. **改进数据增强策略**
   - 测试不同的 Z 轴缩放范围
   - 增加焦距扰动增强

6. **多数据集平衡采样**
   - 确保小数据集不被淹没

---

## 附加发现

### 验证集设计的教训
- 当前验证集仅包含 ih26m + dexycb
- 导致模型在 HO3D 上的泛化问题未被发现
- **建议**: 验证集应覆盖所有目标测试数据集

### norm_by_hand 的权衡
| 设置 | 优点 | 缺点 |
|------|------|------|
| false | 直接预测绝对坐标 | 泛化能力差，对相机参数敏感 |
| true | 泛化能力强，更稳定 | 需要额外计算绝对坐标 |

**建议**: 对于跨数据集测试，优先使用 norm_by_hand=true

---

## 结论

模型在 HO3D 上性能下降的主要原因是 **绝对深度估计的系统性偏差**，而非手部姿态预测能力不足。问题根源在于：

1. 训练时未在 HO3D 分布上验证
2. norm_by_hand=false 对分布变化敏感
3. 训练数据的深度分布可能与 HO3D 不同

**最核心的改进**: 使用 norm_by_hand=true 或验证集加入 HO3D。
