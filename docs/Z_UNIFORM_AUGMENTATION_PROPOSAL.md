# Z 均匀分布增强方案 (Z-Uniform Augmentation)

## 背景与动机

当前训练数据的深度分布严重不均：
- **InterHand2.6M**: 纯远距离 (900-1200mm)
- **DexYCB**: 中远距离 (600-1000mm)  
- **HO3D**: 近距离 (400-600mm)
- **HOT3D**: 极近距离 (300-500mm)

模型在 **700-800mm** 表现最好，但在 <600mm 和 >1000mm 的区间泛化能力差。

## 核心思想

通过数据增强，将训练样本的 Z 深度重新映射到一个**均匀分布**的目标范围内，让模型接触到更多样化的深度分布，减少对特定数据集的深度"偏见"。

## 数学原理

### 投影不变性
对于 3D 点 $(X, Y, Z)$，投影公式：
$$x = \frac{X \cdot f_x}{Z} + c_x$$

如果对 Z 缩放 $s$ 倍（$Z' = Z \cdot s$），同时保持 $f_x$ 不变：
$$x' = \frac{X \cdot f_x}{Z \cdot s} + c_x = \frac{x - c_x}{s} + c_x$$

**等价于投影点关于主点缩放 $1/s$。**

### 保持几何一致性的方法

**方案 A: Z + focal 联合缩放** (推荐)
- 将 Z 缩放到目标值: $Z' = Z \cdot s$
- 同步缩放 focal: $f' = f \cdot s$
- 投影保持不变：$x' = \frac{X \cdot f \cdot s}{Z \cdot s} + c_x = \frac{X \cdot f}{Z} + c_x = x$

## 实现方案

### 1. 配置参数

```yaml
TRAIN:
  # 原有 Z 缩放增强（小幅扰动）
  scale_z_range: [0.9, 1.1]
  
  # 新增 Z 均匀分布增强
  z_uniform_aug:
    enabled: true
    z_min: 300.0      # 目标深度下限 (mm)
    z_max: 1000.0     # 目标深度上限 (mm)
    prob: 0.5         # 应用概率
```

### 2. 核心代码 (preprocess_batch 中)

```python
if augmentation_flag:
    # 原有增强...
    
    # === Z 均匀分布增强 ===
    if z_uniform_aug_enabled and torch.rand(1) < z_uniform_aug_prob:
        # 获取当前根关节 Z 深度 [B, T]
        current_z = joint_cam[:, :, 0, 2]
        
        # 为目标范围内均匀采样目标 Z
        z_min, z_max = 300.0, 1000.0
        target_z = torch.rand(B, T, device=device) * (z_max - z_min) + z_min
        
        # 计算缩放因子
        scale_z_uniform = target_z / (current_z + 1e-6)
        
        # 应用 Z 缩放
        joint_cam[:, :, :, 2] *= scale_z_uniform[..., None]
        
        # 同步缩放 focal 保持投影一致
        focal = focal * scale_z_uniform[..., None]
        
        # 调整 trans_pred 相关参数...
        # trans_pred_scale_factor = scale_z_uniform
```

### 3. 与现有增强的协作

```python
# 增强应用顺序:
1. Z 均匀分布增强 (大幅改变深度分布)
   - 将 IH26M (1000mm) → 300mm (补充近距离)
   - 将 HOT3D (300mm) → 800mm (补充中距离)
   
2. 原有 scale_z_range [0.9, 1.1] (小幅扰动)
   - 在目标深度附近增加随机性
```

## 预期效果

### 深度分布变化

| 数据集 | 原始分布 | 增强后分布 |
|--------|---------|-----------|
| IH26M | 900-1200mm | 300-1000mm (均匀) |
| DexYCB | 600-1000mm | 300-1000mm (均匀) |
| HO3D | 400-600mm | 300-1000mm (均匀) |
| HOT3D | 300-500mm | 300-1000mm (均匀) |

### 模型性能预期

1. **HO3D test 误差降低**: 原本 141mm → 预期 <60mm
   - 原因是训练时会生成大量 400-600mm 的样本
   
2. **跨数据集泛化能力提升**: 
   - 不再过度依赖 700-800mm "舒适区"
   - 在整个 300-1000mm 范围都有良好表现

3. **norm_by_hand=false 的稳定性提升**:
   - 减少绝对深度预测的分布偏差

## 注意事项

### 1. 与 norm_by_hand 的兼容性
- `norm_by_hand=true`: Z 增强在归一化后失效（深度被归一化到单位球）
- `norm_by_hand=false`: Z 增强非常有效
- **建议**: 如果要使用 Z 均匀增强，保持 `norm_by_hand=false`

### 2. 目标范围选择

| 范围 | 适用场景 | 风险 |
|------|---------|------|
| [200, 1000] | 包含 HOT3D 极近距 | 200mm 可能太夸张，透视不自然 |
| [300, 1000] | **推荐** | 覆盖大部分场景，较自然 |
| [400, 1000] | 保守方案 | 无法补充 HOT3D 的 300-400mm |
| [300, 1200] | 包含 IH26M 远距 | 1200mm 样本较少，可能过拟合 |

### 3. 概率设置

- `prob=1.0`: 所有样本都应用 Z 均匀增强
- `prob=0.5`: 一半样本保持原始分布，一半应用增强
- **建议从 prob=0.5 开始实验**

### 4. 焦距变化的影响

虽然投影保持一致，但焦距变化会影响：
- 图像裁剪区域 (hand_bbox 可能需要调整)
- 深度估计的数值范围

## 实验计划

### 实验 1: 基础验证
```yaml
MODEL:
  norm_by_hand: false
TRAIN:
  z_uniform_aug:
    enabled: true
    z_min: 300.0
    z_max: 1000.0
    prob: 0.5
```
在 HO3D test 上验证是否比 baseline (141mm) 有显著提升。

### 实验 2: 范围消融
测试不同 z_min/z_max 的效果：
- [300, 800]
- [300, 1000] 
- [400, 1000]

### 实验 3: 与 norm_by_hand 对比
- 方案 A: norm_by_hand=true (无 Z 增强)
- 方案 B: norm_by_hand=false + Z 均匀增强

## 实现 TODO

- [ ] 修改 `src/data/preprocess.py` 中的 `preprocess_batch`
- [ ] 添加配置解析支持
- [ ] 验证 focal 缩放与 Z 缩放的投影一致性
- [ ] 在 HO3D test 上验证效果
- [ ] 对比 norm_by_hand=true 的效果

## 替代方案对比

| 方案 | 复杂度 | 效果预期 | 适用场景 |
|------|--------|---------|---------|
| **Z 均匀增强** | 低 | 高 | 保持 norm_by_hand=false |
| norm_by_hand=true | 低 | 高 | 放弃绝对深度估计 |
| 数据采样加权 | 中 | 中 | 不改变数据，只调整采样 |
| 完整重投影增强 | 高 | 高 | 需要修改更多代码 |

## 结论

Z 均匀分布增强是一个**简单但有效**的方案，可以在 `preprocess_batch` 中快速实现，预期能显著改善模型在 HO3D 等近距离数据集上的表现，同时保持跨数据集的泛化能力。
