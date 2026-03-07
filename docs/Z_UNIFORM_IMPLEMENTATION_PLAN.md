# Z 均匀化实施方案

## 目标
实现 Z 深度均匀分布增强，解决训练数据深度分布不均导致的跨数据集泛化问题。

## 实施阶段

### Phase 1: 核心实现 (preprocess.py)

#### 1.1 函数签名修改
```python
def preprocess_batch(
    ...,
    normalize_intrinsics_config: Optional[Dict] = None,
    z_uniform_aug_config: Optional[Dict] = None,
):
```

#### 1.2 添加内参归一化逻辑
位置: augmentation_flag 处理块之前
```python
# === 步骤 0: 内参归一化 (仅归一化 focal) ===
if normalize_intrinsics_config and normalize_intrinsics_config.get("enabled", False):
    target_focal = normalize_intrinsics_config.get("target_focal", [1000.0, 1000.0])
    target_focal_t = torch.tensor(target_focal, device=device, dtype=focal.dtype)
    
    scale_fx = target_focal_t[0] / focal[..., 0]
    scale_fy = target_focal_t[1] / focal[..., 1]
    
    # 构建内参归一化矩阵
    norm_intr_mat = torch.eye(3, device=device).expand(B, T, 3, 3).clone()
    norm_intr_mat[..., 0, 0] = scale_fx
    norm_intr_mat[..., 1, 1] = scale_fy
    norm_intr_mat[..., 0, 2] = princpt[..., 0] * (scale_fx - 1)
    norm_intr_mat[..., 1, 2] = princpt[..., 1] * (scale_fy - 1)
    
    trans_2d_mat = norm_intr_mat @ trans_2d_mat
    focal = focal * torch.stack([scale_fx, scale_fy], dim=-1)
    princpt = princpt * torch.stack([scale_fx, scale_fy], dim=-1)
```

#### 1.3 添加 Z 均匀增强逻辑
位置: augmentation_flag 处理块开始处
```python
# === 步骤 1: Z 均匀增强 ===
z_aug_scale = torch.ones(B, T, device=device)
if augmentation_flag and z_uniform_aug_config and z_uniform_aug_config.get("enabled", False):
    prob = z_uniform_aug_config.get("prob", 0.5)
    apply_z_aug = torch.rand(B, 1, device=device).expand(-1, T) < prob
    
    if apply_z_aug.any():
        z_min = z_uniform_aug_config.get("z_min", 300.0)
        z_max = z_uniform_aug_config.get("z_max", 1000.0)
        
        joint_cam_temp = batch_origin["joint_cam"].to(device)
        root_z = joint_cam_temp[:, :, 0, 2]
        
        target_z = torch.rand(B, T, device=device) * (z_max - z_min) + z_min
        
        z_aug_scale = torch.where(
            apply_z_aug,
            target_z / (root_z + 1e-6),
            torch.ones_like(root_z)
        )
```

#### 1.4 修改 scale_z 计算逻辑
与 scale_z_range 互斥:
```python
z_aug_applied = (z_aug_scale != 1.0).any()

if z_aug_applied:
    scale_z = z_aug_scale
else:
    scale_z = (
        torch.rand(B, 1, device=device).expand(-1, T)
        * (scale_z_range[1] - scale_z_range[0])
        + scale_z_range[0]
    )
```

### Phase 2: 配置集成

#### 2.1 修改配置文件
在 `config/stage1-dino_large_no_norm.yaml` 添加:
```yaml
TRAIN:
  normalize_intrinsics:
    enabled: true
    target_focal: [1000.0, 1000.0]
  
  z_uniform_aug:
    enabled: true
    z_min: 250.0
    z_max: 1100.0
    prob: 0.5
```

#### 2.2 修改 train.py
传递配置到 preprocess_batch:
```python
batch_processed = preprocess_batch(
    ...,
    normalize_intrinsics_config=cfg.TRAIN.get("normalize_intrinsics"),
    z_uniform_aug_config=cfg.TRAIN.get("z_uniform_aug"),
)
```

### Phase 3: 测试验证

#### 3.1 单元测试
- 测试内参归一化矩阵计算正确性
- 测试 Z 均匀增强后 joint_cam Z 值在目标范围内
- 测试与 scale_z_range 的互斥逻辑

#### 3.2 集成测试
- 小规模数据运行，检查输出 shape 和数值范围
- 可视化 warp 后的图像，检查是否有异常

#### 3.3 消融实验
1. Baseline (无增强)
2. 仅内参归一化
3. 仅 Z 均匀增强
4. 两者叠加

### Phase 4: 实验验证

#### 4.1 训练实验
- 在 HO3D train 上训练，监控 HO3D val 性能
- 对比 baseline 和增强后的效果

#### 4.2 测试验证
- HO3D test
- DexYCB test

## 时间安排

| 阶段 | 预计时间 | 依赖 |
|------|---------|------|
| Phase 1 | 1 小时 | 无 |
| Phase 2 | 30 分钟 | Phase 1 |
| Phase 3 | 1 小时 | Phase 2 |
| Phase 4 | 1-2 天 | Phase 3 |

## 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 代码 bug 导致训练失败 | 中 | 高 | 充分测试后再启动训练 |
| 效果不明显 | 中 | 中 | 准备多套超参数方案 |
| 训练不稳定 | 低 | 高 | 从保守参数开始 |

## 下一步行动

现在开始 Phase 1 的实施。是否确认开始？
