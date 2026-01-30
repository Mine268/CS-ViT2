# 数据增强消融实验指南

## 📋 概述

CS-ViT2现在支持通过YAML配置动态控制数据增强策略，便于进行消融实验，快速测试不同增强组合的效果。

## 🎛️ 配置系统

### 配置位置
所有数据增强配置在 `TRAIN.augmentation` 下：

```yaml
TRAIN:
  augmentation:
    # 每个增强都有 enabled 开关和具体参数
    color_jitter:
      enabled: true/false
      brightness: 0.2
      ...
```

### 支持的增强操作

| 增强名称 | 配置键 | 推荐使用 | 说明 |
|---------|--------|---------|------|
| **ColorJitter** | `color_jitter` | ✅ 推荐 | 光照增强，SOTA标配 |
| **GaussianNoise** | `gaussian_noise` | ✅ 推荐 | 传感器噪声，真实场景常见 |
| **GaussianBlur** | `gaussian_blur` | ⚠️ 可选 | 模糊增强，可测试鲁棒性 |
| **Sharpness** | `sharpness` | ❌ 不推荐 | 与模糊冲突 |
| **Equalize** | `equalize` | ❌ 不推荐 | 破坏预训练分布 |
| **MotionBlur** | `motion_blur` | ❌ 不推荐 | 破坏关节位置 |
| **RandomErasing** | `random_erasing` | ❌ 不推荐 | 与heatmap冲突 |

## 🧪 消融实验配置

### 1. 基线配置（默认）
**文件**: `config/stage1-dino_large.yaml`

```yaml
TRAIN:
  augmentation:
    color_jitter:
      enabled: true
      brightness: 0.2
      contrast: 0.2
      saturation: 0.1
      hue: 0.0
      p: 0.5
    gaussian_noise:
      enabled: true
      mean: 0.0
      std: 0.03
      p: 0.5
```

**运行**:
```bash
python script/stage1.py --config-name=stage1-dino_large
```

---

### 2. 无增强（消融基准）
**文件**: `config/ablation/no_augmentation.yaml`

测试数据增强的整体贡献。

**运行**:
```bash
python script/stage1.py --config-name=ablation/no_augmentation
```

**预期**: 验证MPJPE可能上升5-10mm，过拟合增加

---

### 3. 仅ColorJitter
**文件**: `config/ablation/only_color_jitter.yaml`

测试光照增强的单独效果。

**运行**:
```bash
python script/stage1.py --config-name=ablation/only_color_jitter
```

**预期**: 性能接近基线，说明ColorJitter是核心增强

---

### 4. 添加GaussianBlur
**文件**: `config/ablation/with_gaussian_blur.yaml`

测试在基础增强上添加模糊的效果。

**运行**:
```bash
python script/stage1.py --config-name=ablation/with_gaussian_blur
```

**预期**:
- 如果验证MPJPE下降 → 模糊有助于鲁棒性
- 如果验证MPJPE上升 → 模糊干扰了学习

---

### 5. 激进增强
**文件**: `config/ablation/aggressive_augmentation.yaml`

测试更强的增强参数（HaMeR风格）。

**运行**:
```bash
python script/stage1.py --config-name=ablation/aggressive_augmentation
```

**预期**: 可能降低过拟合，但训练loss会更高

---

## 🔧 自定义消融实验

### 方法1：创建新配置文件

```yaml
# config/ablation/my_experiment.yaml
defaults:
  - ../stage1-dino_large

TRAIN:
  augmentation:
    color_jitter:
      enabled: true
      brightness: 0.3  # 自定义参数
      contrast: 0.3
      saturation: 0.2
      hue: 0.05
      p: 0.6
    gaussian_noise:
      enabled: false   # 禁用噪声
```

```bash
python script/stage1.py --config-name=ablation/my_experiment
```

---

### 方法2：命令行覆盖参数

```bash
# 禁用ColorJitter
python script/stage1.py --config-name=stage1-dino_large \
    TRAIN.augmentation.color_jitter.enabled=false

# 修改GaussianNoise参数
python script/stage1.py --config-name=stage1-dino_large \
    TRAIN.augmentation.gaussian_noise.std=0.05 \
    TRAIN.augmentation.gaussian_noise.p=0.8

# 启用GaussianBlur
python script/stage1.py --config-name=stage1-dino_large \
    TRAIN.augmentation.gaussian_blur.enabled=true
```

---

## 📊 实验对比建议

### 完整消融矩阵

| 实验ID | ColorJitter | GaussianNoise | GaussianBlur | 预期MPJPE |
|--------|-------------|---------------|--------------|-----------|
| E1 | ❌ | ❌ | ❌ | 90mm (无增强基准) |
| E2 | ✅ | ❌ | ❌ | 85mm |
| E3 | ❌ | ✅ | ❌ | 87mm |
| E4 | ✅ | ✅ | ❌ | 84mm (基线) |
| E5 | ✅ | ✅ | ✅ | 83mm? (待测试) |

### 训练30000步后对比指标

```python
# 记录以下指标
metrics = {
    'train_loss': ...,
    'val_mpjpe': ...,
    'val_mpjpe_std': ...,  # 验证波动
    'train_val_gap': ...,   # 过拟合程度
}
```

---

## 💡 实验建议

### 快速测试（10000步）
```bash
# 修改total_step进行快速测试
python script/stage1.py --config-name=ablation/no_augmentation \
    GENERAL.total_step=10000
```

### 并行运行多个实验
```bash
# GPU 0: 基线
CUDA_VISIBLE_DEVICES=0 python script/stage1.py \
    --config-name=stage1-dino_large &

# GPU 1: 无增强
CUDA_VISIBLE_DEVICES=1 python script/stage1.py \
    --config-name=ablation/no_augmentation &

# GPU 2: 添加模糊
CUDA_VISIBLE_DEVICES=2 python script/stage1.py \
    --config-name=ablation/with_gaussian_blur &
```

### 在AIM中对比
所有实验会自动记录到AIM，使用实验名称区分：
- `stage1-dino_large` - 基线
- `ablation/no_augmentation` - 无增强
- `ablation/with_gaussian_blur` - 添加模糊

---

## 🔍 代码实现

### 增强配置解析流程

```
config/stage1-dino_large.yaml
  └─ TRAIN.augmentation: {...}
        ↓
script/stage1.py
  └─ cfg.TRAIN.get('augmentation', None)
        ↓
src/data/preprocess.py
  └─ get_or_create_augmentation(aug_config, device)
        ├─ 检查缓存（_augmentation_cache）
        ├─ 如果存在 → 直接返回缓存实例 ⚡
        └─ 如果不存在 → 创建新实例并缓存
              └─ PixelLevelAugmentation(aug_config).to(device)
                    ↓
              动态构建 torch.nn.Sequential([...])
```

### 性能优化

**缓存机制**: 相同配置的增强器只创建一次，后续直接复用

- ✅ 避免每个batch重复创建增强pipeline
- ✅ 避免重复的模型初始化和to(device)操作
- ✅ 显著提升训练效率（预计节省数分钟/10万步）

**测试缓存性能**:
```bash
python -m tests.test_augmentation_cache
```

### 关键代码位置

- **配置解析**: `script/stage1.py:383`
- **缓存管理**: `src/data/preprocess.py:123-165`
- **增强器构建**: `src/data/preprocess.py:12-120`
- **应用增强**: `src/data/preprocess.py:453-456`

---

## 📝 实验记录模板

```markdown
## 消融实验：[实验名称]

**日期**: 2026-01-30
**配置**: config/ablation/xxx.yaml

### 配置摘要
- ColorJitter: enabled=true, brightness=0.2, ...
- GaussianNoise: enabled=false
- ...

### 训练结果 (30000步)
- 训练loss: 1.15
- 验证MPJPE: 84.5mm ± 18.2mm
- 训练-验证gap: +42mm

### 对比基线
- MPJPE变化: -2.1mm (-2.5%)
- 波动变化: -3.8mm (-17%)
- 结论: GaussianBlur有助于降低波动

### AIM链接
[experiment link]
```

---

## ⚠️ 注意事项

1. **验证集seed固定**: 确保所有实验使用相同验证集
   ```yaml
   GENERAL:
     val_seed: 42  # 已在基线配置中设置
   ```

2. **随机种子固定**: 便于复现
   ```yaml
   GENERAL:
     seed: 3229084  # 已在基线配置中设置
   ```

3. **checkpoint隔离**: 不同实验的checkpoint会自动按日期和配置名分开
   ```
   checkpoint/
   ├── 30-01-2026/15-00-00_stage1-dino_large/
   ├── 30-01-2026/16-00-00_ablation-no_augmentation/
   └── 30-01-2026/17-00-00_ablation-with_gaussian_blur/
   ```

4. **增强与heatmap**: 如果启用了`LOSS.supervise_heatmap=true`，避免使用RandomErasing

---

**GG - 开始你的消融实验！**
