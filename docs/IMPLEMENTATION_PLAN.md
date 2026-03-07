# 内参归一化 + Z 均匀增强 实现方案

## 1. 总体架构

```
原始数据
    ↓
[1] 内参归一化 (仅 focal)
    - 将 focal 归一化到 target_focal (如 1000mm)
    - princpt 随 focal 同比缩放
    ↓
[2] Z 均匀增强
    - 将根关节 Z 深度随机映射到 [z_min, z_max] 均匀分布
    - 只改变 Z 坐标，focal 保持不变
    - 投影点自然改变，trans_2d_mat 自动处理图像变换
    ↓
[3] 原有增强 (scale_z_range 作为小幅扰动)
    ↓
输出
```

## 2. 关键设计决策

### 2.1 Z 均匀增强不调 focal

**之前的错误理解**: 认为需要调整 focal 来保持投影不变。

**正确的理解**: 
- Z 均匀增强的目的是**改变深度分布**
- 投影点自然随着 Z 改变而改变
- `trans_2d_mat` 会根据新的 3D 坐标计算 warp 矩阵
- 图像会自动与新的 3D 坐标对齐

### 2.2 执行顺序

```
1. 内参归一化 (如果启用)
   - 输入: 原始 focal, princpt
   - 输出: 归一化后的 focal, princpt
   
2. Z 均匀增强 (如果启用)
   - 输入: 归一化后的 joint_cam
   - 输出: 调整 Z 后的 joint_cam
   - focal 保持不变！
   
3. 原有增强 (scale_z_range, scale_f_range 等)
   - scale_z_range 作为叠加在 Z 均匀增强之上的小幅扰动
```

## 3. 配置设计

```yaml
TRAIN:
  # 内参归一化配置
  normalize_intrinsics:
    enabled: true
    target_focal: [1000.0, 1000.0]  # 目标焦距
    
  # Z 均匀增强配置
  z_uniform_aug:
    enabled: true
    z_min: 250.0      # 目标深度下限 (mm)，覆盖 HOT3D 的近距离
    z_max: 1100.0     # 目标深度上限 (mm)，覆盖 IH26M 的远距离
    prob: 0.5         # 应用概率
    
  # 原有配置
  scale_z_range: [0.9, 1.1]  # 小幅扰动
  scale_f_range: [0.9, 1.1]
```

## 4. 核心实现

### 4.1 函数签名修改

```python
def preprocess_batch(
    ...,
    normalize_intrinsics_config: Optional[Dict] = None,
    z_uniform_aug_config: Optional[Dict] = None,
):
```

### 4.2 内参归一化实现

```python
# === 步骤 0: 内参归一化 (仅归一化 focal) ===
if normalize_intrinsics_config and normalize_intrinsics_config.get("enabled", False):
    target_focal = normalize_intrinsics_config.get("target_focal", [1000.0, 1000.0])
    target_focal_t = torch.tensor(target_focal, device=device, dtype=focal.dtype)
    
    # 计算缩放因子
    scale_fx = target_focal_t[0] / focal[..., 0]
    scale_fy = target_focal_t[1] / focal[..., 1]
    
    # 构建内参归一化矩阵
    norm_intr_mat = torch.eye(3, device=device).expand(B, T, 3, 3).clone()
    norm_intr_mat[..., 0, 0] = scale_fx
    norm_intr_mat[..., 1, 1] = scale_fy
    norm_intr_mat[..., 0, 2] = princpt[..., 0] * (scale_fx - 1)
    norm_intr_mat[..., 1, 2] = princpt[..., 1] * (scale_fy - 1)
    
    # 合并到 trans_2d_mat
    trans_2d_mat = norm_intr_mat @ trans_2d_mat
    
    # 更新 focal 和 princpt
    focal = focal * torch.stack([scale_fx, scale_fy], dim=-1)
    princpt = princpt * torch.stack([scale_fx, scale_fy], dim=-1)
```

### 4.3 Z 均匀增强实现

```python
# === 步骤 1: Z 均匀增强 ===
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

### 4.4 与原有增强的整合

```python
# === 增强参数 ===
if augmentation_flag:
    rad = torch.rand(B, 1, device=device).expand(-1, T) * 2 * torch.pi
    
    # 原有的 scale_z_range 作为小幅扰动，叠加在 Z 均匀增强之上
    scale_z_perturb = (
        torch.rand(B, 1, device=device).expand(-1, T)
        * (scale_z_range[1] - scale_z_range[0])
        + scale_z_range[0]
    )
    
    # 总 Z 缩放 = Z 均匀增强 * 小幅扰动
    scale_z = z_aug_scale * scale_z_perturb
```

## 5. 关键修正说明

### 5.1 Z 均匀增强为什么不调 focal？

| 问题 | 答案 |
|------|------|
| **投影公式** | $u = X \cdot f / Z + c_x$ |
| **改变 Z 的目的** | 让模型在不同深度下学习 |
| **投影是否改变** | **是**，投影点会随着 Z 改变 |
| **如何处理图像** | `trans_2d_mat` 自动根据新的 3D 坐标计算 warp 矩阵 |
| **是否需要调 f** | **不需要**，focal 保持不变 |

### 5.2 正确的几何关系

```
原始: 3D(X,Y,Z) --投影--> 2D(u,v) --warp--> patch
        ↓
增强: 3D(X,Y,Z') --投影--> 2D(u',v') --warp--> patch'

注意:
- Z' 是新的目标深度
- u' ≠ u (投影点改变是正常的)
- patch' 与新的 3D 坐标匹配
```

## 6. 预期效果

### 6.1 训练数据分布变化

| 数据集 | 原始深度分布 | 增强后深度分布 |
|--------|-------------|---------------|
| InterHand2.6M | 900-1200mm | 250-1100mm (均匀) |
| DexYCB | 600-1000mm | 250-1100mm (均匀) |
| HO3D | 400-600mm | 250-1100mm (均匀) |
| HOT3D | 300-500mm | 250-1100mm (均匀) |

### 6.2 预期性能提升

| 测试集 | 当前 RTE | 预期 RTE | 改进幅度 |
|--------|---------|---------|---------|
| HO3D | 146mm | ~70mm | -50% |
| DexYCB | 117mm | ~60mm | -50% |

## 7. 实验计划

### 7.1 消融实验

1. **仅内参归一化**: 验证 focal 归一化的效果
2. **仅 Z 均匀增强**: 验证 Z 分布均衡的效果
3. **两者叠加**: 验证组合效果
4. **与 norm_by_hand 对比**: 对比两种方案的优劣

### 7.2 超参数搜索

- `target_focal`: 616 (RealSense) vs 1000 (标准)
- `z_min/z_max`: [300, 1000] vs [250, 1100] vs [200, 1200]
- `prob`: 0.3 vs 0.5 vs 0.7

## 8. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| Z 范围过大导致图像失真 | 高 | 限制 [250, 1100]，避免极端值 |
| 训练不稳定 | 中 | 从 prob=0.3 开始，逐步增加 |
| 与原有增强冲突 | 低 | Z 均匀增强与 scale_z_range 互斥 |

## 9. 实现状态

- [x] `preprocess_batch` 函数签名修改
- [x] 内参归一化实现
- [x] Z 均匀增强实现
- [ ] 配置系统集成 (Hydra)
- [ ] 单元测试
- [ ] 消融实验
