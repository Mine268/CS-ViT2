# 测试脚本使用说明

本文档说明如何使用 `script/test.py` 进行模型测试和结果保存。

## 快速开始

### 基本用法（Stage 1）

```bash
source .venv/bin/activate

python script/test.py \
    --config-name=stage1-dino_large \
    TEST.checkpoint_path=checkpoint/07-01-2026/checkpoints/checkpoint-30000 \
    DATA.test.source='[/mnt/qnap/data/datasets/webdatasets/InterHand2.6M/test/*]'
```

### 多卡测试

```bash
accelerate launch --num_processes=4 script/test.py \
    --config-name=stage1-dino_large \
    TEST.checkpoint_path=checkpoint/07-01-2026/checkpoints/checkpoint-30000 \
    DATA.test.source='[/mnt/qnap/data/datasets/webdatasets/InterHand2.6M/test/*]'
```

### Stage 2 测试

```bash
python script/test.py \
    --config-name=stage2-dino_large \
    TEST.checkpoint_path=checkpoint/xxx/checkpoint-30000 \
    DATA.test.source='[/mnt/qnap/data/datasets/webdatasets/InterHand2.6M/test/*]'
```

## 配置参数

### 必需参数

- `TEST.checkpoint_path`: Checkpoint 路径（必须指定）
- `DATA.test.source`: 测试数据源列表（必须指定）

### 可选参数

```bash
python script/test.py \
    --config-name=stage1-dino_large \
    TEST.checkpoint_path=checkpoint/xxx/checkpoint-30000 \
    DATA.test.source='[/path/to/test/*]' \
    TEST.batch_size=32 \           # 测试 batch size（默认 16）
    TEST.max_samples=1000 \         # 限制样本数（调试用，默认 null=全部）
    TEST.vis_step=50 \              # 可视化频率（默认 100）
    TEST.enable_vis=false \         # 是否启用 AIM 可视化（默认 true）
    TEST.output_dir=output/my_test  # 输出目录（默认 output/test_results）
```

### 测试多个数据集

```bash
python script/test.py \
    --config-name=stage1-dino_large \
    TEST.checkpoint_path=checkpoint/xxx/checkpoint-30000 \
    DATA.test.source='[/mnt/qnap/data/datasets/webdatasets/InterHand2.6M/test/*,/mnt/qnap/data/datasets/webdatasets/DexYCB/s1/test/*]'
```

### 调试模式（小数据集）

```bash
python script/test.py \
    --config-name=stage1-dino_large \
    TEST.checkpoint_path=checkpoint/xxx/checkpoint-30000 \
    DATA.test.source='[/mnt/qnap/data/datasets/webdatasets/InterHand2.6M/test/*]' \
    TEST.max_samples=100 \
    TEST.batch_size=4 \
    TEST.enable_vis=false
```

## 输出文件

测试完成后，会在 `TEST.output_dir` 目录下生成以下文件：

```
output/test_results/
├── predictions_20260210_143052.h5  # HDF5 格式的预测结果
├── metrics.json                     # 快速指标摘要
└── test_config.yaml                 # 测试时使用的配置
```

### HDF5 文件结构

```
predictions.h5
├── samples/
│   ├── imgs_path          [N] (string)
│   ├── handedness         [N] (string)
│   ├── joint_cam_pred     [N, 21, 3] (float32, gzip)  # 真实相机坐标
│   ├── vert_cam_pred      [N, 778, 3] (float32, gzip)
│   ├── mano_pose_pred     [N, 48] (float32, gzip)
│   ├── mano_shape_pred    [N, 10] (float32, gzip)
│   ├── trans_pred         [N, 3] (float32, gzip)      # 原始输出（可能归一化）
│   ├── trans_pred_denorm  [N, 3] (float32, gzip)      # 反归一化后的 trans
│   ├── norm_scale         [N] (float32)               # 手部大小
│   ├── norm_valid         [N] (float32)               # norm_scale 是否有效
│   ├── joint_cam_gt       [N, 21, 3] (float32, gzip)
│   ├── vert_cam_gt        [N, 778, 3] (float32, gzip)
│   ├── mano_pose_gt       [N, 48] (float32, gzip)
│   ├── mano_shape_gt      [N, 10] (float32, gzip)
│   ├── focal              [N, 2] (float32)
│   ├── princpt            [N, 2] (float32)
│   ├── hand_bbox          [N, 4] (float32)
│   ├── joint_valid        [N, 21] (float32)
│   └── mano_valid         [N] (float32)
└── metadata/
    ├── num_samples (attr)
    ├── norm_by_hand (attr)
    ├── timestamp (attr)
    └── config (attr, JSON string)
```

## 验证结果

### 1. 读取 HDF5 文件

```python
import h5py
import numpy as np

with h5py.File("output/test_results/predictions.h5", 'r') as f:
    # 读取预测结果
    joint_pred = f['samples/joint_cam_pred'][:]  # [N, 21, 3]
    vert_pred = f['samples/vert_cam_pred'][:]    # [N, 778, 3]

    # 读取 ground truth
    joint_gt = f['samples/joint_cam_gt'][:]

    # 读取元数据
    num_samples = f['metadata'].attrs['num_samples']
    norm_by_hand = f['metadata'].attrs['norm_by_hand']

    print(f"Total samples: {num_samples}")
    print(f"norm_by_hand: {norm_by_hand}")
    print(f"Joint pred shape: {joint_pred.shape}")
```

### 2. 验证 norm_by_hand 处理

如果 `norm_by_hand=true`，验证反归一化公式：

```python
with h5py.File("output/test_results/predictions.h5", 'r') as f:
    trans_pred = f['samples/trans_pred'][:]          # 原始（可能归一化）
    trans_denorm = f['samples/trans_pred_denorm'][:] # 反归一化
    norm_scale = f['samples/norm_scale'][:]
    norm_by_hand = f['metadata'].attrs['norm_by_hand']

    if norm_by_hand:
        # 验证反归一化公式
        expected_denorm = trans_pred * norm_scale[:, None]
        assert np.allclose(trans_denorm, expected_denorm, atol=1e-5), \
            "norm_by_hand 反归一化验证失败"
        print("✓ norm_by_hand 反归一化验证通过")
    else:
        # 如果未启用 norm_by_hand，trans_pred 和 trans_denorm 应该相同
        assert np.allclose(trans_pred, trans_denorm, atol=1e-5), \
            "trans_pred 和 trans_denorm 应该相同"
        print("✓ norm_by_hand 未启用，trans 一致")
```

### 3. 计算自定义指标

```python
import numpy as np

with h5py.File("output/test_results/predictions.h5", 'r') as f:
    joint_pred = f['samples/joint_cam_pred'][:]
    joint_gt = f['samples/joint_cam_gt'][:]
    joint_valid = f['samples/joint_valid'][:]

    # 计算 MPJPE
    joint_diff = np.linalg.norm(joint_pred - joint_gt, axis=-1)  # [N, 21]
    joint_diff_masked = joint_diff * joint_valid
    mpjpe = np.sum(joint_diff_masked) / np.sum(joint_valid)

    print(f"MPJPE: {mpjpe:.2f} mm")

    # 计算根关节对齐的 PA-MPJPE
    joint_pred_rel = joint_pred - joint_pred[:, :1]
    joint_gt_rel = joint_gt - joint_gt[:, :1]
    joint_diff_rel = np.linalg.norm(joint_pred_rel - joint_gt_rel, axis=-1)
    joint_diff_rel_masked = joint_diff_rel * joint_valid
    pa_mpjpe = np.sum(joint_diff_rel_masked) / np.sum(joint_valid)

    print(f"PA-MPJPE: {pa_mpjpe:.2f} mm")
```

### 4. 多卡一致性测试

分别用 1 卡和 4 卡运行同一测试集，确认结果一致：

```bash
# 1 卡测试
python script/test.py \
    --config-name=stage1-dino_large \
    TEST.checkpoint_path=checkpoint/xxx/checkpoint-30000 \
    DATA.test.source='[/path/to/test/*]' \
    TEST.output_dir=output/test_1gpu

# 4 卡测试
accelerate launch --num_processes=4 script/test.py \
    --config-name=stage1-dino_large \
    TEST.checkpoint_path=checkpoint/xxx/checkpoint-30000 \
    DATA.test.source='[/path/to/test/*]' \
    TEST.output_dir=output/test_4gpu
```

验证结果：

```python
import h5py
import numpy as np

with h5py.File("output/test_1gpu/predictions.h5", 'r') as f1, \
     h5py.File("output/test_4gpu/predictions.h5", 'r') as f4:

    # 比较样本数
    n1 = f1['metadata'].attrs['num_samples']
    n4 = f4['metadata'].attrs['num_samples']
    assert n1 == n4, f"样本数不一致: {n1} vs {n4}"

    # 比较预测结果（允许小误差）
    joint_pred_1 = f1['samples/joint_cam_pred'][:]
    joint_pred_4 = f4['samples/joint_cam_pred'][:]

    assert np.allclose(joint_pred_1, joint_pred_4, atol=1e-4), \
        "多卡预测结果不一致"

    print(f"✓ 多卡一致性测试通过（样本数: {n1}）")
```

## 常见问题

### 1. checkpoint 加载失败

**错误**：`RuntimeError: Error(s) in loading state_dict`

**解决**：确保 checkpoint 路径正确，指向包含 `model.safetensors` 的目录：
```bash
TEST.checkpoint_path=checkpoint/07-01-2026/22-14-27_stage1-dino_large/checkpoints/checkpoint-30000
```

### 2. 测试数据源为空

**错误**：`ValueError: No test data found`

**解决**：确保数据路径正确，使用引号包裹列表：
```bash
DATA.test.source='[/mnt/qnap/data/datasets/webdatasets/InterHand2.6M/test/*]'
```

### 3. WebDataset worker 卡死

**现象**：推理开始后没有进度

**解决**：测试脚本已设置 `num_workers=1`，如果仍有问题，检查 tar 文件是否损坏。

### 4. 内存不足

**解决**：减小 batch size 或限制样本数：
```bash
TEST.batch_size=8 TEST.max_samples=10000
```

### 5. AIM 初始化失败

**错误**：`Failed to initialize AIM`

**解决**：关闭可视化或检查 AIM 服务器：
```bash
TEST.enable_vis=false
```

## 后续分析

保存的 HDF5 文件可以用于：

1. **计算更多指标**：PCK, AUC, EPE 等
2. **按子集分析**：按 handedness, dataset 分组统计
3. **错误分析**：找出误差最大的样本
4. **可视化**：生成预测结果的可视化图像
5. **模型对比**：对比不同 checkpoint 的性能

所有这些分析都可以离线进行，不需要重新运行推理。

## 示例分析脚本

```python
# analyze_results.py
import h5py
import numpy as np
import matplotlib.pyplot as plt

def analyze_by_handedness(h5_path):
    """按左右手分析误差"""
    with h5py.File(h5_path, 'r') as f:
        joint_pred = f['samples/joint_cam_pred'][:]
        joint_gt = f['samples/joint_cam_gt'][:]
        joint_valid = f['samples/joint_valid'][:]
        handedness = f['samples/handedness'][:].astype(str)

        # 计算误差
        errors = np.linalg.norm(joint_pred - joint_gt, axis=-1)  # [N, 21]

        # 按 handedness 分组
        for hand in ['left', 'right']:
            mask = handedness == hand
            if np.sum(mask) > 0:
                hand_errors = errors[mask] * joint_valid[mask]
                hand_mpjpe = np.sum(hand_errors) / np.sum(joint_valid[mask])
                print(f"{hand.capitalize()} hand MPJPE: {hand_mpjpe:.2f} mm")

def plot_joint_errors(h5_path):
    """绘制每个关节的误差分布"""
    with h5py.File(h5_path, 'r') as f:
        joint_pred = f['samples/joint_cam_pred'][:]
        joint_gt = f['samples/joint_cam_gt'][:]
        joint_valid = f['samples/joint_valid'][:]

        # 计算每个关节的误差
        errors = np.linalg.norm(joint_pred - joint_gt, axis=-1)  # [N, 21]

        # 绘制箱线图
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.boxplot([errors[:, i][joint_valid[:, i] > 0.5] for i in range(21)])
        ax.set_xlabel('Joint Index')
        ax.set_ylabel('Error (mm)')
        ax.set_title('Joint-wise Error Distribution')
        plt.savefig('joint_errors.png')
        print("Saved plot to joint_errors.png")

if __name__ == "__main__":
    h5_path = "output/test_results/predictions.h5"
    analyze_by_handedness(h5_path)
    plot_joint_errors(h5_path)
```

## 参考

- 主训练脚本：`script/train.py`
- 模型定义：`src/model/net.py`
- 数据加载：`src/data/dataloader.py`
- 预处理：`src/data/preprocess.py`
