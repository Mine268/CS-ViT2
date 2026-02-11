# 训练 NaN 问题修复

**日期**: 2026-02-11
**训练会话**: checkpoint/2026-02-11/16-26-11_stage1-dino_large

## 问题描述

训练在 Step 1070 时突然出现 NaN，所有 loss 值同时变为 NaN，训练无法继续。

### 症状

```
Step 1060: 正常
  total=2.0351
  loss_theta=0.1686
  loss_trans=15.3273
  loss_joint_img=53.7714

Step 1070: 全部 NaN
  total=nan
  loss_theta=nan
  loss_shape=nan
  loss_trans=nan
  loss_joint_rel=nan
  loss_joint_img=nan
```

### 配置

- `MODEL.norm_by_hand: true`
- `TRAIN.mixed_precision: bf16`
- `TRAIN.lr: 0.0001`
- `GENERAL.warmup_step: 5000`
- `LOSS.reproj_loss_type: robust_l1`

## 根本原因

### 代码缺陷

在 `src/model/loss.py:454`，trans_gt 的归一化缺少 epsilon 保护：

```python
# 问题代码
if self.norm_by_hand:
    trans_gt = trans_gt / norm_scale_gt[..., None]  # 除零风险！
```

当 `norm_scale_gt` 为 0 或非常接近 0 时：
- 除法产生 `Inf` 或非常大的值
- `Inf` 在后续计算中传播，最终产生 `NaN`
- 所有 loss 同时变为 `NaN`

### 数据异常

`norm_scale_gt` 可能为 0 或异常小的原因：
1. **GT 关节标注错误**：中指的 4 个关节（norm_idx=[9,10,11,12]）重合或非常接近
2. **GT 关节缺失**：虽然 `joint_valid` 为 0，但仍然参与了 norm_scale 计算
3. **数据预处理错误**：关节坐标单位错误（如从 mm 变成 m）
4. **数值精度问题**：bf16 混合精度训练可能导致 norm_scale 下溢

### 诊断验证

运行 `tools/diagnose_nan.py` 确认了问题：

```python
场景3: 零 norm_scale (0.0 mm)
  norm_scale_gt: 0.0000 mm
  trans_gt (原始): [100. 200. 800.]
  trans_gt (归一化): [inf inf inf]  # 产生 Inf！
  是否有Inf: True
```

## 修复方案

### 方案1：添加 epsilon 保护（推荐）

**优点**：
- 简单直接，立即生效
- 对正常数据无影响（epsilon 很小）
- 鲁棒性强，处理所有边界情况

**缺点**：
- 对异常数据给出了"错误"的归一化值（但总比 NaN 好）
- 不解决数据质量问题

#### 实现

修改 `src/model/loss.py:454` 和 `480`：

```python
# 修改前
if self.norm_by_hand:
    trans_gt = trans_gt / norm_scale_gt[..., None]

# ...

if self.norm_by_hand:
    trans_pred_scaled = trans_pred * norm_scale_gt[..., None]
```

```python
# 修改后
NORM_SCALE_EPSILON = 1e-6  # 在文件开头定义常量

if self.norm_by_hand:
    # 添加 epsilon 防止除零
    trans_gt = trans_gt / (norm_scale_gt[..., None] + NORM_SCALE_EPSILON)

# ...

if self.norm_by_hand:
    # 为了对称性，反归一化也加 epsilon（影响很小）
    trans_pred_scaled = trans_pred * (norm_scale_gt[..., None] + NORM_SCALE_EPSILON)
```

**epsilon 选择**：
- 建议值：`1e-6`
- 理由：对于 mm 单位的坐标，1e-6 mm 是极小的值（0.001 微米），对正常数据的影响可忽略
- 保护效果：即使 norm_scale_gt=0，除法也只产生有限值（如 800mm / 1e-6 = 8e+8，大但有限）

### 方案2：在 get_hand_norm_scale 中添加保护

**优点**：
- 在源头解决问题
- norm_scale 保证在合理范围内

**缺点**：
- 需要修改两处（loss.py 和 net.py）
- 仍然没有解决数据质量问题

#### 实现

修改 `src/model/loss.py:392` 和 `src/model/net.py:261`：

```python
def get_hand_norm_scale(self, j3d: torch.Tensor, valid: torch.Tensor):
    d = j3d[..., self.norm_idx[:-1], :] - j3d[..., self.norm_idx[1:], :]
    d = torch.sum(torch.sqrt(torch.sum(d ** 2, dim=-1)), dim=-1)

    # 添加：防止 norm_scale 过小
    d = torch.clamp(d, min=1e-6)

    flag = torch.all(valid[:, :, self.norm_idx] > 0.5, dim=-1).float()
    return d, flag
```

### 方案3：数据验证和过滤

**优点**：
- 从根本上解决数据质量问题
- 提高训练数据质量

**缺点**：
- 需要重新处理数据
- 可能丢失部分训练样本
- 实施周期长

#### 实现

在数据加载时添加验证：

```python
# 在 src/data/dataloader.py 或预处理中
def validate_sample(sample):
    joint_cam = sample["joint_cam"]
    joint_valid = sample["joint_valid"]

    # 计算 norm_scale
    norm_idx = [9, 10, 11, 12]
    if torch.all(joint_valid[norm_idx] > 0.5):
        d = joint_cam[norm_idx[:-1]] - joint_cam[norm_idx[1:]]
        norm_scale = torch.sum(torch.sqrt(torch.sum(d ** 2, dim=-1)))

        # 检查 norm_scale 是否合理
        if norm_scale < 1.0 or norm_scale > 200.0:
            print(f"Warning: Abnormal norm_scale={norm_scale:.4f} mm")
            return False  # 过滤掉这个样本

    return True
```

### 方案4：添加 NaN 检测和调试

**优点**：
- 及时发现问题
- 提供调试信息

**缺点**：
- 不解决问题，只是提前发现

#### 实现

在 `script/train.py` 的训练循环中：

```python
# 在 loss.backward() 之前
if torch.isnan(loss).any() or torch.isinf(loss).any():
    print(f"\nNaN/Inf detected at step {global_step}")
    print(f"norm_scale_gt: {output['debug']['norm_scale_gt']}")
    print(f"norm_valid_gt: {output['debug']['norm_valid_gt']}")
    print(f"loss components:")
    print(f"  loss_theta: {output['result']['loss_theta']}")
    print(f"  loss_shape: {output['result']['loss_shape']}")
    print(f"  loss_trans: {output['result']['loss_trans']}")
    print(f"  loss_joint_rel: {output['result']['loss_joint_rel']}")
    print(f"  loss_joint_img: {output['result']['loss_joint_img']}")

    # 保存异常 batch 用于调试
    torch.save(batch, f"debug/nan_batch_step_{global_step}.pt")
    raise RuntimeError("NaN in loss, check debug output")
```

## 推荐修复步骤

### 立即修复（必须）

1. **实施方案1**：在 `src/model/loss.py` 添加 epsilon 保护
   - 修改 line 454
   - 修改 line 480
   - 定义 `NORM_SCALE_EPSILON = 1e-6`

2. **测试验证**：
   ```bash
   # 激活环境
   source .venv/bin/activate

   # 运行单元测试
   pytest tests/test_robust_loss_integration.py -v

   # 短期训练测试（100 steps）
   python script/train.py \
       --config-name=stage1-dino_large \
       GENERAL.total_step=100
   ```

3. **重新启动训练**：
   ```bash
   # 从头开始（因为 checkpoint 已经包含 NaN）
   python script/train.py --config-name=stage1-dino_large
   ```

### 后续改进（建议）

4. **实施方案2**：在 `get_hand_norm_scale` 中添加 clamp
   - 双重保护，更加鲁棒

5. **实施方案4**：添加 NaN 检测
   - 及时发现问题
   - 提供调试信息

6. **数据质量检查**：
   - 运行数据验证脚本
   - 找出 norm_scale 异常的样本
   - 分析标注错误的原因

## 代码修改

### 修改1：src/model/loss.py

在文件开头添加常量：

```python
# 在 imports 之后，class 定义之前
NORM_SCALE_EPSILON = 1e-6  # 防止 norm_scale 除零，单位: mm
```

修改 line 454：

```python
# 修改前
if self.norm_by_hand:
    trans_gt = trans_gt / norm_scale_gt[..., None]

# 修改后
if self.norm_by_hand:
    trans_gt = trans_gt / (norm_scale_gt[..., None] + NORM_SCALE_EPSILON)
```

修改 line 480：

```python
# 修改前
if self.norm_by_hand:
    trans_pred_scaled = trans_pred * norm_scale_gt[..., None]

# 修改后
if self.norm_by_hand:
    trans_pred_scaled = trans_pred * (norm_scale_gt[..., None] + NORM_SCALE_EPSILON)
```

修改 line 392（可选，双重保护）：

```python
def get_hand_norm_scale(self, j3d: torch.Tensor, valid: torch.Tensor):
    """
    Args:
        j3d: [...,j,3]
        valid: [...,j]
        return: [...], [...]
    """
    d = j3d[..., self.norm_idx[:-1], :] - j3d[..., self.norm_idx[1:], :]
    d = torch.sum(torch.sqrt(torch.sum(d ** 2, dim=-1)), dim=-1) # [...]

    # 添加：防止 norm_scale 过小
    d = torch.clamp(d, min=NORM_SCALE_EPSILON)

    flag = torch.all(valid[:, :, self.norm_idx] > 0.5, dim=-1).float()
    return d, flag
```

### 修改2：src/model/net.py（可选，保持一致）

在文件开头添加常量：

```python
# 在 imports 之后，class 定义之前
from src.model.loss import NORM_SCALE_EPSILON
```

修改 line 261（可选，与 loss.py 保持一致）：

```python
def get_hand_norm_scale(self, j3d: torch.Tensor, valid: torch.Tensor):
    d = j3d[..., self.norm_idx[:-1], :] - j3d[..., self.norm_idx[1:], :]
    d = torch.sum(torch.sqrt(torch.sum(d ** 2, dim=-1)), dim=-1)

    # 添加：防止 norm_scale 过小
    d = torch.clamp(d, min=NORM_SCALE_EPSILON)

    flag = torch.all(valid[:, :, self.norm_idx] > 0.5, dim=-1).float()
    return d, flag
```

## 验证方法

### 1. 单元测试

创建 `tests/test_nan_fix.py`：

```python
import torch
from src.model.loss import NORM_SCALE_EPSILON

def test_norm_scale_epsilon_protection():
    """测试 epsilon 保护是否防止 NaN"""

    # 场景1: norm_scale = 0
    norm_scale_gt = torch.tensor([[0.0]], dtype=torch.float32)
    trans_gt = torch.tensor([[[100.0, 200.0, 800.0]]], dtype=torch.float32)

    # 归一化
    trans_normalized = trans_gt / (norm_scale_gt[..., None] + NORM_SCALE_EPSILON)

    # 验证
    assert not torch.isnan(trans_normalized).any(), "NaN found!"
    assert not torch.isinf(trans_normalized).any(), "Inf found!"
    assert trans_normalized.abs().max() < 1e10, "Value too large!"

    print(f"✓ 场景1通过: norm_scale=0 → trans_normalized={trans_normalized[0,0].numpy()}")

    # 场景2: 正常 norm_scale
    norm_scale_gt = torch.tensor([[70.0]], dtype=torch.float32)
    trans_gt = torch.tensor([[[100.0, 200.0, 800.0]]], dtype=torch.float32)

    trans_normalized = trans_gt / (norm_scale_gt[..., None] + NORM_SCALE_EPSILON)

    # 验证：应该与不加 epsilon 的结果几乎相同
    trans_normalized_no_eps = trans_gt / norm_scale_gt[..., None]
    diff = torch.abs(trans_normalized - trans_normalized_no_eps).max()

    assert diff < 1e-5, f"Epsilon 影响太大: diff={diff}"
    print(f"✓ 场景2通过: norm_scale=70mm → epsilon 影响 < 1e-5")

if __name__ == "__main__":
    test_norm_scale_epsilon_protection()
    print("\n所有测试通过！")
```

运行测试：

```bash
source .venv/bin/activate
python tests/test_nan_fix.py
```

### 2. 短期训练测试

```bash
# 训练 1000 步，检查是否还会 NaN
python script/train.py \
    --config-name=stage1-dino_large \
    GENERAL.total_step=1000 \
    GENERAL.checkpoint_step=500
```

监控日志中是否出现 NaN：

```bash
tail -f checkpoint/2026-02-11/*/log.txt | grep -E "(nan|NaN)"
```

### 3. 数据质量检查

创建 `tools/check_norm_scale_distribution.py`：

```python
"""检查训练数据中 norm_scale 的分布"""

import torch
import webdataset as wds
import numpy as np
from tqdm import tqdm

# 加载数据
dataset = wds.WebDataset("/mnt/qnap/data/datasets/webdatasets/InterHand2.6M/train/train-000000.tar")

norm_idx = [9, 10, 11, 12]
norm_scales = []

for sample in tqdm(dataset, total=1000):
    joint_cam = torch.from_numpy(sample["joint_cam.npy"])
    joint_valid = torch.from_numpy(sample["joint_valid.npy"])

    # 计算 norm_scale
    if torch.all(joint_valid[norm_idx] > 0.5):
        d = joint_cam[norm_idx[:-1]] - joint_cam[norm_idx[1:]]
        norm_scale = torch.sum(torch.sqrt(torch.sum(d ** 2, dim=-1)))
        norm_scales.append(norm_scale.item())

norm_scales = np.array(norm_scales)

print(f"\nnorm_scale 统计:")
print(f"  样本数: {len(norm_scales)}")
print(f"  均值: {norm_scales.mean():.2f} mm")
print(f"  标准差: {norm_scales.std():.2f} mm")
print(f"  最小值: {norm_scales.min():.2f} mm")
print(f"  最大值: {norm_scales.max():.2f} mm")
print(f"  25%: {np.percentile(norm_scales, 25):.2f} mm")
print(f"  50%: {np.percentile(norm_scales, 50):.2f} mm")
print(f"  75%: {np.percentile(norm_scales, 75):.2f} mm")

# 检查异常值
abnormal_small = norm_scales < 1.0
abnormal_large = norm_scales > 200.0

print(f"\n异常值:")
print(f"  < 1mm: {abnormal_small.sum()} ({abnormal_small.mean()*100:.2f}%)")
print(f"  > 200mm: {abnormal_large.sum()} ({abnormal_large.mean()*100:.2f}%)")

if abnormal_small.sum() > 0:
    print(f"\n异常小的值: {norm_scales[abnormal_small]}")
```

## 预期效果

### 修复前
- Step 1070 出现 NaN
- 训练无法继续
- 所有 checkpoint 都包含 NaN 权重

### 修复后
- norm_scale=0 时产生有限值（虽然很大，但可计算）
- 训练正常进行
- 后续通过数据清洗进一步提高鲁棒性

## 相关文档

- **[NORM_BY_HAND.md](NORM_BY_HAND.md)** - norm_by_hand 功能详细说明
- **[REPROJ_LOSS_CONFIG.md](REPROJ_LOSS_CONFIG.md)** - 重投影 loss 配置
- 分析工具：`tools/diagnose_nan.py`
- 测试脚本：`tests/test_nan_fix.py`（需创建）

## 总结

**根本原因**：`norm_by_hand` 功能中缺少除零保护

**立即修复**：添加 epsilon=1e-6 保护

**长期改进**：数据质量检查和过滤

**预防措施**：添加 NaN 检测和调试信息

GG
