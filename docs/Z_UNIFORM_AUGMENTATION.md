# Z 均匀分布增强方案 (Z-Uniform Augmentation)

## 背景

当前训练数据的深度分布严重不均：
- **InterHand2.6M**: 纯远距离 (900-1200mm)
- **DexYCB**: 中远距离 (600-1000mm)
- **HO3D**: 近距离 (400-600mm)
- **HOT3D**: 极近距离 (300-500mm)

模型在 **700-800mm** 表现最好，但在 <600mm 和 >1000mm 的区间泛化能力差。

## 核心思想

通过数据增强，将训练样本的 Z 深度重新映射到一个**均匀分布**的目标范围内，让模型接触到更多样化的深度分布。

## 关键设计：Z 均匀增强不调 focal

### 之前的错误理解

曾误认为需要调整 focal 来保持投影不变：
```python
# 错误的做法
joint_cam[..., 2] *= scale_z  # 改变 Z
focal = focal * scale_z       # 错误地调整 focal
```

### 正确的理解

**Z 均匀增强的目的是改变深度分布，投影点自然改变是正常的：**

```python
# 正确的做法
joint_cam[..., 2] *= scale_z  # 改变 Z
# focal 保持不变！
```

**原因**：
1. 投影公式 $u = X \cdot f / Z + c_x$
2. 当 Z 改变时，投影点 $u$ 自然改变
3. `trans_2d_mat` 会根据新的 3D 坐标计算 warp 矩阵
4. 图像会自动与新的 3D 坐标对齐

### 几何关系

```
原始: 3D(X,Y,Z) --投影--> 2D(u,v) --warp--> patch
        ↓
增强: 3D(X,Y,Z') --投影--> 2D(u',v') --warp--> patch'

注意:
- Z' 是新的目标深度
- u' ≠ u (投影点改变是正常的)
- patch' 与新的 3D 坐标匹配
```

## 实现方案

### 配置

```yaml
TRAIN:
  z_uniform_aug:
    enabled: true
    z_min: 250.0      # 目标深度下限 (mm)
    z_max: 1100.0     # 目标深度上限 (mm)
    prob: 0.5         # 应用概率
```

### 核心代码

```python
# === Z 均匀增强 ===
z_aug_scale = torch.ones(B, T, device=device)
if augmentation_flag and z_uniform_aug_config and z_uniform_aug_config.get("enabled", False):
    prob = z_uniform_aug_config.get("prob", 0.5)
    
    # 生成 mask，决定哪些样本应用 Z 均匀增强
    apply_z_aug = torch.rand(B, 1, device=device).expand(-1, T) < prob
    
    if apply_z_aug.any():
        z_min = z_uniform_aug_config.get("z_min", 300.0)
        z_max = z_uniform_aug_config.get("z_max", 1000.0)
        
        # 获取当前根关节 Z 深度
        joint_cam_temp = batch_origin["joint_cam"].to(device)
        root_z = joint_cam_temp[:, :, 0, 2]  # [B, T]
        
        # 为目标范围 [z_min, z_max] 均匀采样目标 Z
        target_z = torch.rand(B, T, device=device) * (z_max - z_min) + z_min
        
        # 计算 Z 缩放因子
        z_aug_scale = torch.where(
            apply_z_aug,
            target_z / (root_z + 1e-6),
            torch.ones_like(root_z)
        )
```

### 与原有增强的整合

**重要**: Z 均匀增强与 `scale_z_range` **互斥**。

如果启用了 Z 均匀增强，跳过 `scale_z_range`：

```python
# 检查是否应用了 Z 均匀增强
z_aug_applied = (z_aug_scale != 1.0).any()

if z_aug_applied:
    # 使用 Z 均匀增强的结果
    scale_z = z_aug_scale
else:
    # 使用原有的 scale_z_range
    scale_z = (
        torch.rand(B, 1, device=device).expand(-1, T)
        * (scale_z_range[1] - scale_z_range[0])
        + scale_z_range[0]
    )
```

**原因**:
- Z 均匀增强已经提供了大范围的 Z 变化 ([250, 1100]mm)
- 再叠加 `scale_z_range` 的小幅扰动 ([0.9, 1.1]) 意义不大
- 简化逻辑，避免两个 Z 相关的增强相互干扰

## 目标范围选择

### 当前数据集分布

| 数据集 | 深度范围 | 关键区间 |
|--------|---------|---------|
| InterHand2.6M | 900-1200mm | >900mm |
| DexYCB | 600-1000mm | 600-1000mm |
| HO3D | 400-600mm | 400-600mm |
| HOT3D | 300-500mm | <500mm |

### 推荐范围: [250, 1100]mm

- **下限 250mm**: 覆盖 HOT3D 的极近距离 (300mm)
- **上限 1100mm**: 覆盖 IH26M 的远距离 (1000-1200mm)
- **避免极端值**: <250mm 或 >1200mm 可能导致透视失真

## 与其他增强的兼容性

| 增强类型 | 兼容性 | 说明 |
|---------|-------|------|
| 内参归一化 | ✅ | 建议先归一化 focal，再应用 Z 均匀增强 |
| scale_z_range | ✅ | 叠加在 Z 均匀增强之上作为小幅扰动 |
| scale_f_range | ✅ | 独立，不影响 Z 增强 |
| persp_rot | ✅ | 独立，不受影响 |

## 预期效果

### 训练数据分布变化

```
原始分布:
IH26M:        [900-1200mm]          (纯远距离)
DexYCB:    [600-1000mm]             (中远距离)
HO3D:    [400-600mm]                (近距离)
HOT3D: [300-500mm]                  (极近距离)

增强后分布:
所有数据集: [250-1100mm] (均匀分布)
```

### 性能预期

| 测试集 | 当前 RTE | 预期 RTE | 改进原因 |
|--------|---------|---------|---------|
| HO3D | 146mm | ~70mm | 训练时有更多 400-600mm 样本 |
| DexYCB | 117mm | ~60mm | 整体深度分布更均衡 |

## 实验建议

1. **从保守范围开始**: [300, 1000]mm
2. **逐步扩展**: 如果效果好，扩展到 [250, 1100]mm
3. **监控验证集**: 确保所有深度区间都有样本

## 结论

Z 均匀分布增强通过改变 Z 坐标（不调 focal），让模型在训练时接触到均匀的深度分布，从而改善在近距离和远距离区间泛化能力差的问题。
