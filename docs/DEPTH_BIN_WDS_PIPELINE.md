# 静态深度分桶 WebDataset 流程

本文档说明如何基于现有 WebDataset 重新生成按深度区间分桶的训练数据，并使用对应的 dataloader 在训练时按深度桶均匀采样。

## 1. 结论速览

- 输入：现有 WebDataset 序列样本（例如 `InterHand2.6M/train/*.tar`）
- 处理：把每个序列样本离线打散成固定长度 clip
- 分桶：按 **最后一帧 root joint 的 Z 深度** 分到不同 bin
- 输出：`/mnt/qnap/data/datasets/webdatasets/depth-bins/<dataset>/<split>/nf{T}_s{stride}/bin_xxxx_xxxx/*.tar`
- 训练：使用新的 depth-bin dataloader，对不同深度桶做均匀混采

## 2. 为什么选“最后一帧 root Z”

当前项目中：

- Stage 1 输入通常是 `num_frame=1`
- Stage 2 虽然输入多帧，但监督与输出都集中在最后一帧

因此离线分桶时使用：

```text
root_depth_last = joint_cam[end - 1, 0, 2]
```

这样与当前训练/评估目标最一致。

## 3. 生成脚本

脚本位置：`preprocess/depth_bin_wds.py`

### 3.1 主要能力

- 读取现有 WebDataset tar
- 按 `num_frames + stride` 离线切成 clip
- 对每个 clip 计算最后一帧 root 深度
- 按静态深度边界分桶
- 为每个 bin 单独写 tar
- 自动输出 `summary.json`
- 默认 `maxsize` 为 `1.5GB`（更适合 NAS 场景，避免产生过多小 tar）

### 3.2 示例命令

```bash
source .venv/bin/activate

python preprocess/depth_bin_wds.py \
  --dataset-name InterHand2.6M \
  --split train \
  --source '/mnt/qnap/data/datasets/webdatasets/InterHand2.6M/train/*.tar' \
  --output-root /mnt/qnap/data/datasets/webdatasets/depth-bins \
  --num-frames 1 \
  --stride 1 \
  --bin-edges 0 500 700 900 1100 1000000
  --maxsize $((1536 * 1024 * 1024))
```

### 3.3 输出目录示例

```text
/mnt/qnap/data/datasets/webdatasets/depth-bins/
└── InterHand2.6M/
    └── train/
        └── nf1_s1/
            ├── bin_0000_0500/
            │   ├── 000000.tar
            │   └── ...
            ├── bin_0500_0700/
            ├── bin_0700_0900/
            ├── bin_0900_1100/
            ├── bin_1100_inf/
            └── summary.json
```

## 4. 对应 dataloader

文件位置：`src/data/depth_bin_dataloader.py`

### 4.1 提供的接口

- `collect_depth_bin_sources(...)`
  - 收集指定数据集、指定 split、指定 clip 配置下的 bin tar 路径
- `get_depth_bin_dataloader(...)`
  - 基于多个 depth bin 构造混采 dataloader

### 4.2 采样逻辑

默认 `mix_strategy="uniform_random"`：

- 每个深度桶先各自构造一个 WebDataset
- 再用 `webdataset.RandomMix` 做均匀混采
- 这样一个 batch 更容易覆盖不同深度区间

也支持：

- `mix_strategy="round_robin"`
- `mix_strategy="dataset_bin_balanced"`

### 4.3 `dataset × depth-bin` 均衡采样

当 `mix_strategy="dataset_bin_balanced"` 时：

1. 先按深度桶做等权采样
2. 再在同一个深度桶内部，对不同数据集按 `count^(-alpha)` 做受约束均衡
3. 若某个 `(dataset, bin)` 的样本数小于 `min_cell_samples`，则该 cell 不参与均衡竞争

默认参数建议：

```yaml
DATA:
  train:
    depth_bins:
      mix_strategy: dataset_bin_balanced
      dataset_balance_alpha: 0.5
      min_cell_samples: 1000
```

这比“20 个 cell 完全等权”更稳，能避免极小 cell 被过度重复采样。

### 4.4 `alpha` 的实际含义与建议

当前实现中，`dataset_balance_alpha` 只作用于 **同一个 depth bin 内部** 的 dataset 平衡：

```text
P(cell) = P(bin) * P(dataset | bin)
P(bin) = 等权
P(dataset | bin) ∝ count(dataset, bin)^(-alpha)
```

这意味着：

- `alpha = 0`
  - 同一个 bin 内，所有 active dataset 等权
- `alpha > 0`
  - 同一个 bin 内，小样本 dataset 会被抬权
- `alpha` 越大
  - bin 内越偏向小 cell

当前实验分析表明：

- 如果目标是“不同深度桶等权，同时让总体 dataset 分布尽量更均衡”
- 那么 `alpha = 0.0` 往往比 `alpha = 0.5` 更接近目标

相关可视化：

- `docs/dataset_depth_bin_sampling_weights_with_marginals.png`
- `docs/dataset_depth_bin_sampling_before_after.png`
- `docs/dataset_depth_bin_sampling_before_after_alpha0.png`

### 4.5 与 `script/train.py` 的接入

训练配置新增了：

```yaml
DATA:
  train:
    depth_bins:
      enabled: true
      root: /mnt/qnap/data/datasets/webdatasets/depth-bins
      dataset_names: [InterHand2.6M, HO3D_v3]
      split: train
      selected_bins: null
      mix_strategy: uniform_random
      bin_weights: null
      dataset_balance_alpha: 0.5
      min_cell_samples: 1000
      shardshuffle: false
      sample_shuffle: 200
```

当 `enabled=true` 时，`script/train.py` 会优先使用 depth-bin dataloader，而不再走普通的 `DATA.train.source`。

## 5. 推荐使用方式

### 5.1 Stage 1

- 先对每个训练数据集做 `num_frames=1` 的离线分桶
- 训练时混合多个数据集的同名 bin
- 保证不同深度区间被均匀采样

### 5.2 Stage 2

- 如果后续需要，可再单独生成 `num_frames=7` 等固定 clip 长度的版本
- 同样按最后一帧 root Z 分桶

## 6. 当前实现边界

当前实现是 **纯静态深度分桶**：

- 不做动态权重调整
- 不做 dataset × depth 的联合二级权重控制
- 但已经为后续在更高层做联合采样留出了接口基础

## 7. 验证状态

已完成：

- 假数据端到端测试：`tests/test_depth_bin_wds.py`
- 覆盖：输入 tar → clip 切分 → 分桶写 tar → depth-bin dataloader 读取

## 8. 相关文件

- `preprocess/depth_bin_wds.py`
- `preprocess/repack_depth_bin_wds.py`
- `src/data/depth_bin_dataloader.py`
- `tests/test_depth_bin_wds.py`
- `tests/test_depth_bin_dataset_balanced.py`
- `docs/DATALOADER_SAMPLING_STRATEGY.md`

## 9. 数据目录快速参考

- 目录结构说明：`docs/DEPTH_BINS_DATASET_LAYOUT.md`
- 外部数据目录镜像：`/mnt/qnap/data/datasets/webdatasets/depth-bins/README`

## 10. tmux 批量转换脚本

仓库内提供了一键启动的 `tmux` 脚本：

- `tools/run_depth_bin_tmux.sh`

默认会为下列 4 个训练集各开一个 `tmux` 窗口，并按**串行顺序接力执行**转换：

1. `InterHand2.6M/train/*`
2. `DexYCB/s1/train/*`
3. `HO3D_v3/train/*`
4. `HOT3D/train/*`

也就是说：4 个窗口都能看到状态，但同一时刻只会有 1 个任务真正执行，避免 4 并发同时抢磁盘和 tar I/O。

基本用法：

```bash
cd /data_1/renkaiwen/CS-ViT2
./tools/run_depth_bin_tmux.sh
```

自定义 session 名：

```bash
./tools/run_depth_bin_tmux.sh my_depthbin
```

默认日志目录：

- `logs/depth_bin_full/`

进入 session：

```bash
tmux attach -t depthbin
```

## 11. repack 脚本

- repack 说明：`docs/DEPTH_BIN_REPACK.md`
- repack 脚本：`preprocess/repack_depth_bin_wds.py`

## 12. 当前推荐的 no_norm baseline 配置

针对当前 depth-bin 训练，推荐先使用更保守的几何/像素增强基线：

```yaml
defaults:
  - augmentation: color_jitter_only

TRAIN:
  scale_z_range: [1.0, 1.0]
  scale_f_range: [1.0, 1.0]
  persp_rot_max: 0.0872664626  # 约 5°
```

其中 `color_jitter_only` 当前建议参数为：

```yaml
color_jitter:
  brightness: 0.15
  contrast: 0.15
  saturation: 0.05
  hue: 0.0
  p: 0.5
```

目的是先稳定绝对几何学习，再观察 depth-bin 采样本身带来的收益。

## 13. 完整 val 的多卡处理

当前训练脚本支持两种 val 模式：

### 13.1 在线代理验证（默认）

```yaml
DATA:
  val:
    full_eval: false
    max_val_step: 1000
```

- `val_loader` 使用 `infinite=True`
- 每个 rank 严格跑固定步数
- 适合训练中高频监控，不容易出现多卡不同步问题

### 13.2 完整 full val

```yaml
DATA:
  val:
    full_eval: true
```

完整 full val 的实现方式：

1. 启动训练时先统计每个 val tar 在当前 `num_frame/stride` 下可展开的 clip 数
2. 按 clip 数把 tar（必要时切成片段）尽量均衡分给各个 rank
3. 每个 rank 只验证自己那部分样本
4. 最后只对标量统计量做 `reduce`

这样可以保证：

- 完整 val：每个 sample 恰好一次
- 多卡加速：不同 rank 负载更均衡
- tar 数量少于 GPU 数量时也能工作
