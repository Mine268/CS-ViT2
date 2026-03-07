# 相机内参归一化方案讨论与决策

## 背景

不同数据集使用不同的相机设备，内参差异巨大：

| 数据集 | Focal (fx, fy) | Princpt (cx, cy) | 图像分辨率 |
|--------|---------------|-----------------|-----------|
| InterHand2.6M | 1268mm | (174, 242) | 变化 |
| DexYCB | 616mm | (313, 243) | 640x480 |
| HO3D | 616mm | (324, 236) | 640x480 |
| HOT3D | 609mm | (703, 703) | 1408x1408 |

## 讨论：应该归一化哪些内参？

### 方案 1: 同时归一化 focal 和 princpt (cx, cy)

**想法**: 将所有数据集的内参统一到标准值：
- focal: 1000mm
- princpt: (112, 112) - 224x224 图像中心

**潜在问题**:
1. **HOT3D 的 cx, cy = (703, 703)**，要归一化到 (112, 112) 需要平移 **591 像素**
2. **透视失真**: 大的平移会导致手部形状扭曲
3. **边缘伪影**: 平移后需要填充大量黑边
4. **收益不明确**: 数据流中手部已经通过 bbox 大致居中

### 方案 2: 仅归一化 focal，保持 princpt 相对关系 (选定方案)

**决策依据**:

1. **focal 是跨数据集差异的核心**:
   - 1268mm vs 616mm 相差 2 倍
   - 直接影响 3D→2D 投影的尺度
   - 影响绝对深度 Z 的估计

2. **cx, cy 的重要性被高估**:
   - 输入数据已经是 hand-centric crop (224x224 patches)
   - 手部大致在图像中心附近
   - 绝对主点位置对相对姿态估计影响有限

3. **代价收益比**:
   - 归一化 cx, cy 需要大的透视变换
   - 引入失真和黑边的代价 > 可能的收益

## 最终实现方案

### 归一化策略

**只归一化 focal，princpt 随 focal 同比缩放**:

```python
# 目标内参
target_focal = [1000.0, 1000.0]  # 或 616.0

# 归一化计算
scale = target_focal / focal_orig
focal_new = focal_orig * scale  # = target_focal
princpt_new = princpt_orig * scale  # 保持相对位置
```

**为什么不把 princpt 设为 (112, 112)**:
- 保持 princpt 与 focal 的比例关系，避免额外的透视扭曲
- 手部已经在 crop 后的图像中心附近

### 在 preprocess_batch 中的实现

```python
def get_focal_normalization_matrix(focal_orig, princpt_orig, target_focal):
    """
    仅归一化 focal，princpt 随 focal 同比缩放
    
    变换矩阵:
    [f'/fx,   0,   cx * (f'/fx - 1)]
    [  0,   f'/fy, cy * (f'/fy - 1)]
    [  0,     0,         1         ]
    """
    B, T = focal_orig.shape[:2]
    device = focal_orig.device
    
    mat = torch.eye(3, device=device).expand(B, T, 3, 3).clone()
    
    scale_x = target_focal[0] / focal_orig[..., 0]
    scale_y = target_focal[1] / focal_orig[..., 1]
    
    mat[..., 0, 0] = scale_x
    mat[..., 1, 1] = scale_y
    mat[..., 0, 2] = princpt_orig[..., 0] * (scale_x - 1)
    mat[..., 1, 2] = princpt_orig[..., 1] * (scale_y - 1)
    
    return mat, target_focal, princpt_orig * torch.stack([scale_x, scale_y], dim=-1)
```

### 与现有增强的整合

在 `preprocess_batch` 中：
1. 先应用 focal 归一化（如果启用）
2. 再应用原有的 scale_z, scale_f 等增强
3. 合并所有变换矩阵到 `trans_2d_mat`

```python
# 1. 内参归一化
if normalize_intrinsics_enabled:
    norm_mat, focal_new, princpt_new = get_focal_normalization_matrix(
        focal, princpt, target_focal
    )
    trans_2d_mat = norm_mat @ trans_2d_mat
    focal = focal_new
    princpt = princpt_new

# 2. 其他增强...
```

## 预期效果

| 数据集 | 原始 Focal | 归一化后 | 变化 |
|--------|-----------|---------|------|
| InterHand2.6M | 1268mm | 1000mm | -21% |
| DexYCB | 616mm | 1000mm | +62% |
| HO3D | 616mm | 1000mm | +62% |
| HOT3D | 609mm | 1000mm | +64% |

**注意**: InterHand2.6M 的图像会略微"缩小"，其他数据集会略微"放大"。

## 配置

```yaml
TRAIN:
  normalize_intrinsics:
    enabled: true
    target_focal: [1000.0, 1000.0]  # 仅归一化 focal
    # 不设置 target_princpt，princpt 随 focal 缩放
```

## 决策总结

- **归一化 focal**: ✅ 必须，解决核心尺度差异
- **归一化 cx, cy**: ❌ 不进行，避免透视失真和黑边

这个方案在简化问题的同时，避免了不必要的副作用。
