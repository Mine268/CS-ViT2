# 重投影Loss配置化实现文档

## 概述

实现了通过配置文件动态选择重投影loss类型的功能，支持标准L1和鲁棒L1两种类型。

## 修改内容

### 1. 核心代码修改

#### `src/model/loss.py`
- **修改点**: `BundleLoss2.__init__()` 参数
- **变更**:
  ```python
  # 旧参数
  robust_reproj: bool = True
  robust_reproj_delta: float = 84.0

  # 新参数
  reproj_loss_type: str = "robust_l1"
  reproj_loss_delta: float = 84.0
  ```
- **逻辑**: 根据 `reproj_loss_type` 动态选择loss函数
  - `"l1"`: 使用标准 `nn.L1Loss`
  - `"robust_l1"`: 使用 `RobustL1Loss(delta=reproj_loss_delta)`

#### `src/model/net.py`
- **修改点**: `PoseNet.__init__()` 参数列表和 `BundleLoss2` 初始化
- **新增参数**:
  ```python
  reproj_loss_type: str
  reproj_loss_delta: float
  ```
- **传递给**: `BundleLoss2(reproj_loss_type=..., reproj_loss_delta=...)`

#### `script/train.py`
- **修改点**: `setup_model()` 函数
- **新增配置读取**:
  ```python
  reproj_loss_type=cfg.LOSS.get("reproj_loss_type", "robust_l1"),
  reproj_loss_delta=cfg.LOSS.get("reproj_loss_delta", 84.0),
  ```

### 2. 配置文件修改

#### `config/stage1-dino_large.yaml`
#### `config/stage2-dino_large.yaml`

**新增配置项**:
```yaml
LOSS:
  # ... 其他配置 ...
  reproj_loss_type: "robust_l1"   # 重投影loss类型: "l1" 或 "robust_l1"
  reproj_loss_delta: 84.0         # RobustL1Loss阈值(像素), 基于训练log分析的95分位
```

### 3. 测试代码更新

- `tests/test_robust_loss_integration.py`: 更新参数名称
- `tests/test_config_reproj_loss.py`: 新增配置验证脚本

## 使用方法

### 方法1: 使用配置文件默认值

```bash
# 默认使用 robust_l1 (delta=84.0)
python script/train.py --config-name=stage1-dino_large
```

### 方法2: 命令行覆盖配置

```bash
# 切换为标准 L1 loss
python script/train.py --config-name=stage1-dino_large \
    LOSS.reproj_loss_type=l1

# 使用更保守的鲁棒阈值
python script/train.py --config-name=stage1-dino_large \
    LOSS.reproj_loss_delta=110.0

# 使用更激进的鲁棒阈值
python script/train.py --config-name=stage1-dino_large \
    LOSS.reproj_loss_delta=60.0
```

### 方法3: 修改配置文件

编辑 `config/stage1-dino_large.yaml`:
```yaml
LOSS:
  reproj_loss_type: "l1"  # 改为标准L1
  # 或
  reproj_loss_type: "robust_l1"
  reproj_loss_delta: 110.0  # 使用更保守的阈值
```

## 支持的Loss类型

| 类型 | 说明 | 适用场景 |
|------|------|---------|
| `"l1"` | 标准L1 loss | 数据质量高，无异常值 |
| `"robust_l1"` | 鲁棒L1 loss (delta内L1，超出后对数衰减) | 存在异常值，需要稳定训练 |

## Delta参数建议

基于 `checkpoint/2026-02-10/21-51-13_stage1-dino_large/log.txt` 的分析：

| Delta值 | 覆盖范围 | 特点 | 适用场景 |
|---------|---------|------|---------|
| 60 | 90分位 | 激进，快速收敛 | 数据质量较好 |
| **84** | **95分位** | **推荐，平衡性能和稳定性** | **默认选择** |
| 110 | μ+2σ | 保守，最稳定 | 数据噪声大 |

## 验证

运行测试确认配置正确：
```bash
# 验证集成
source .venv/bin/activate
PYTHONPATH=/data_1/renkaiwen/CS-ViT2:$PYTHONPATH python tests/test_robust_loss_integration.py

# 验证配置
python tests/test_config_reproj_loss.py
```

## 效果对比

| 误差(像素) | 标准L1 | RobustL1(δ=84) | 抑制率 |
|-----------|--------|----------------|--------|
| 0         | 0.00   | 0.00           | 0%     |
| 50        | 50.00  | 50.00          | 0%     |
| 84        | 84.00  | 84.00          | 0%     |
| 100       | 100.00 | 98.65          | 1.4%   |
| 200       | 200.00 | 156.87         | 21.6%  |
| 1000      | 1000.00| 292.06         | **70.8%** |

## 后向兼容性

- 旧的训练脚本和配置文件无法直接使用（缺少新参数）
- 建议更新所有配置文件添加 `reproj_loss_type` 和 `reproj_loss_delta` 参数
- 代码中有默认值，理论上可以工作，但建议显式配置

## 扩展支持

如需添加新的loss类型（如Huber、Cauchy等），修改 `src/model/loss.py` 中的 `BundleLoss2.__init__()`:

```python
if reproj_loss_type == "l1":
    self.reproj_loss_fn = self.l1
elif reproj_loss_type == "robust_l1":
    self.reproj_loss_fn = RobustL1Loss(delta=reproj_loss_delta, reduction='none')
elif reproj_loss_type == "huber":  # 新增
    self.reproj_loss_fn = nn.HuberLoss(delta=reproj_loss_delta, reduction='none')
else:
    raise ValueError(f"Unsupported reproj_loss_type: {reproj_loss_type}")
```

## 相关文件

- 实现: `src/model/loss.py:274-305`
- 集成: `src/model/net.py:70-71, 162-174`
- 训练: `script/train.py:215-216`
- 配置: `config/stage1-dino_large.yaml:124-125`, `config/stage2-dino_large.yaml:124-125`
- 测试: `tests/test_robust_loss_integration.py`, `tests/test_config_reproj_loss.py`
