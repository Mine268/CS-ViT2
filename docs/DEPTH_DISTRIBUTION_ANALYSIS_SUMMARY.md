# 深度分布问题分析与解决方案总结

## 问题发现

### 1. 现象
- 模型在验证集 (ih26m + dexycb) 上 RTE = 42mm
- 在 HO3D test 上 RTE = 141mm (差 3.3 倍)
- 在 DexYCB test 上 RTE = 116mm (差 2.8 倍)

### 2. 根本原因

**训练数据深度分布不均：**

| 数据集 | 平均深度 | 主要区间 | 700-800mm 占比 |
|--------|---------|---------|---------------|
| InterHand2.6M | **1089mm** | 900-1200mm | **0%** |
| DexYCB | 846mm | 600-1000mm | **23.1%** |
| HO3D | 531mm | 400-600mm | 18.0% |
| HOT3D | **368mm** | 300-500mm | **0%** |

**模型表现与深度的关系：**
- 700-800mm: RTE = 83-95mm (**最佳区间**)
- <500mm: RTE = 179-189mm (**灾难性**)
- >1000mm: RTE = 145-199mm (**很差**)

### 3. 为什么 DexYCB 比 HO3D 好？

**不是**因为模型在 DexYCB 上学得更好，而是因为：
- DexYCB 平均深度 840mm → 落在"舒适区"
- HO3D 平均深度 568mm → 被迫预测训练不足的近距离

## 解决方案

### 方案 1: 内参归一化

**问题**: InterHand2.6M (f=1268mm) 与 DexYCB/HO3D/HOT3D (f≈616mm) 差异巨大

**解决**: 将 focal 归一化到标准值 (1000mm)

```python
# 内参归一化矩阵
scale_fx = target_focal / focal[..., 0]
scale_fy = target_focal / focal[..., 1]

norm_intr_mat = [
    [scale_fx,    0,      cx * (scale_fx - 1)],
    [0,           scale_fy, cy * (scale_fy - 1)],
    [0,           0,      1]
]
```

**注意**: 不归一化 cx, cy（避免透视失真）

### 方案 2: Z 均匀增强

**问题**: 各数据集深度分布差异大

**解决**: 将 Z 深度随机映射到均匀分布 [250, 1100]mm

```python
# Z 均匀增强
root_z = joint_cam[:, :, 0, 2]
target_z = rand(z_min, z_max)
scale_z_uniform = target_z / root_z

# 只改变 Z，focal 保持不变！
joint_cam[..., 2] *= scale_z_uniform
```

**关键设计**: 
- **不调 focal**（投影点自然改变是正常的）
- **与 scale_z_range 互斥**: 如果启用 Z 均匀增强，跳过原有的 Z 随机缩放

## 实现状态

### 已完成
- [x] `preprocess_batch` 函数签名修改
- [x] 内参归一化实现
- [x] Z 均匀增强实现
- [x] 相关文档更新

### 待完成
- [ ] 配置系统集成 (Hydra)
- [ ] 单元测试
- [ ] 消融实验

## 配置示例

```yaml
TRAIN:
  # 内参归一化
  normalize_intrinsics:
    enabled: true
    target_focal: [1000.0, 1000.0]
  
  # Z 均匀增强
  z_uniform_aug:
    enabled: true
    z_min: 250.0
    z_max: 1100.0
    prob: 0.5
  
  # 原有增强
  scale_z_range: [0.9, 1.1]  # 小幅扰动
  scale_f_range: [0.9, 1.1]
```

## 预期效果

| 测试集 | 当前 RTE | 预期 RTE | 改进 |
|--------|---------|---------|------|
| HO3D | 141mm | ~70mm | -50% |
| DexYCB | 117mm | ~60mm | -50% |

## 验证集建议

**当前问题**: 验证集只包含 ih26m + dexycb，没有近距离验证信号

**建议**: 必须加入 HO3D val 进行监控

```yaml
DATA:
  val:
    source:
    - /path/to/InterHand2.6M/val/*
    - /path/to/DexYCB/val/*
    - /path/to/HO3D_v3/val/*  # 添加
```

## 关键文档索引

1. **误差分析**: `docs/HO3D_ERROR_ANALYSIS.md`
2. **深度分布可视化**: `docs/depth_distribution.png`
3. **实现方案**: `docs/IMPLEMENTATION_PLAN.md`
4. **Z 均匀增强**: `docs/Z_UNIFORM_AUGMENTATION.md`
5. **内参归一化**: `docs/INTRINSIC_NORMALIZATION.md`
6. **对比分析**: `docs/HO3D_DEXYCB_COMPARISON.md`

## 后续工作

1. **立即执行**:
   - 完成配置系统集成
   - 运行消融实验

2. **验证效果**:
   - 测试 HO3D val
   - 测试 DexYCB test
   - 对比 baseline

3. **进一步优化**:
   - 根据结果调整 z_min/z_max
   - 尝试不同的 target_focal
   - 考虑近距离样本采样加权
