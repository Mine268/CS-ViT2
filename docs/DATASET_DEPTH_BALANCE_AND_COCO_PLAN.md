# 当前数据集深度分布与 COCO-WholeBody 训练方案

本文档基于当前项目的实际训练源，整理 Stage 1 的 `dataset x depth-bin` 分布、可行的均衡采样方案，以及 `COCO-WholeBody` 的采样与 loss 改进建议。

## 1. 分析范围

当前 Stage 1 训练源来自：

- `InterHand2.6M/train/*`
- `DexYCB/s1/train/*`
- `HO3D_v3/train/*`
- `HOT3D/train/*`

对应配置见：

- [stage1-dino_large.yaml](/data_1/renkaiwen/CS-ViT2/config/stage1-dino_large.yaml)

说明：

- 当前主训练配置还没有把 `COCO-WholeBody` 放进默认训练源；
- 下面的深度分布统计针对当前 4 个有 3D 标注的数据集；
- `COCO-WholeBody` 的分析单独放在后半部分。

## 2. 根关节深度分布统计

### 2.1 统计口径

- 统计对象：`nf=1, stride=1` 的 Stage 1 clip
- 深度定义：最后一帧 root joint 的 `Z` 深度
- bin 定义：
  - `bin_0000_0500`
  - `bin_0500_0700`
  - `bin_0700_0900`
  - `bin_0900_1100`
  - `bin_1100_inf`

说明：

- 这里使用了外部 `depth-bins` 目录下已有的全量 `repack_stats.json`；
- 对当前 4 个 3D 数据集，这个统计足够准确地反映 Stage 1 的根深度分布；
- `DexYCB` 在外部目录中的名字是 `DexYCB_s1`，不是 `DexYCB`。

### 2.2 每个数据集的总量

全量 clip 数：

- `InterHand2.6M`: `1,636,144`
- `DexYCB`: `356,600`
- `HO3D_v3`: `83,091`
- `HOT3D`: `750,007`

对应自然占比：

- `InterHand2.6M`: `57.90%`
- `DexYCB`: `12.62%`
- `HO3D_v3`: `2.94%`
- `HOT3D`: `26.54%`

结论：

- 当前训练集的自然 dataset 分布极不均衡；
- `InterHand2.6M + HOT3D` 合计占比超过 `84%`；
- `HO3D_v3` 只有 `2.94%`，在自然采样下几乎没有话语权。

### 2.3 聚合后的深度 bin 分布

全量聚合计数：

- `bin_0000_0500`: `734,559`
- `bin_0500_0700`: `152,398`
- `bin_0700_0900`: `194,231`
- `bin_0900_1100`: `941,016`
- `bin_1100_inf`: `803,638`

对应自然占比：

- `bin_0000_0500`: `25.99%`
- `bin_0500_0700`: `5.39%`
- `bin_0700_0900`: `6.87%`
- `bin_0900_1100`: `33.30%`
- `bin_1100_inf`: `28.44%`

结论：

- 当前训练集的根深度分布不是单峰，而是明显双峰甚至多峰；
- `900mm+` 的远距离样本占比高达 `61.74%`；
- `500-900mm` 的中距离区间明显稀缺；
- 这意味着自然采样下，模型更容易被“远距离 + 特定数据源”主导。

### 2.4 `dataset x depth-bin` 结构

#### InterHand2.6M

- `bin_0700_0900`: `475`
- `bin_0900_1100`: `846,593`
- `bin_1100_inf`: `789,076`

结论：

- `InterHand2.6M` 几乎纯远距离；
- `700-900mm` 只有极少样本；
- 对近距离泛化基本没有帮助。

#### DexYCB

- `bin_0000_0500`: `4,223`
- `bin_0500_0700`: `59,854`
- `bin_0700_0900`: `183,776`
- `bin_0900_1100`: `94,185`
- `bin_1100_inf`: `14,562`

结论：

- `DexYCB` 是当前唯一覆盖全深度范围的主力数据集；
- 但核心质量区间集中在 `700-1100mm`。

#### HO3D_v3

- `bin_0000_0500`: `40,502`
- `bin_0500_0700`: `32,395`
- `bin_0700_0900`: `9,956`
- `bin_0900_1100`: `238`

结论：

- `HO3D_v3` 基本就是近距离数据集；
- `900mm+` 可以视为几乎不存在。

#### HOT3D

- `bin_0000_0500`: `689,834`
- `bin_0500_0700`: `60,149`
- `bin_0700_0900`: `24`

结论：

- `HOT3D` 几乎纯近距离；
- 对中远距离几乎没有覆盖。

### 2.5 分布层面的核心结论

当前 4 个 3D 数据集几乎是按深度自然分域的：

- `HOT3D` 负责 `0-500mm`
- `HO3D_v3` 负责 `0-700mm`
- `DexYCB` 负责 `500-1100mm`
- `InterHand2.6M` 负责 `900mm+`

这意味着：

1. 如果直接按自然分布采样，模型会学到强烈的 `dataset ↔ depth` 绑定。
2. 如果简单做 dataset 等权，也会破坏深度分布，导致某些深度区间被反复过采。
3. 因此最合理的方案是：
   先控 depth，再在 depth 内部控 dataset。

## 3. `dataset × depth-bin` 平衡采样方案

### 3.1 候选策略量化结果

我对当前已有实现：

- `mix_strategy=dataset_bin_balanced`
- `P(bin)=等权`
- `P(dataset | bin) ∝ count(dataset, bin)^(-alpha)`

做了不同 `alpha` 和 `min_cell_samples` 的量化比较。

重点看两个指标：

1. 训练时的 dataset 边缘分布会被扭成什么样；
2. 最小 cell 会被过采多少倍。

### 3.2 不推荐的设置

#### `min_cell_samples=1000`

问题非常明显：

- `DexYCB@0-500` 被过采约 `44.6x ~ 95.5x`
- `HO3D_v3@700-900` 被过采约 `28.4x ~ 46.0x`
- `DexYCB@1100+` 被过采约 `19.4x ~ 34.2x`

结论：

- `1000` 太低；
- 会把极小 cell 硬抬成高频样本，训练会非常不稳定；
- 这类过采样不是“均衡”，而是在制造重复数据。

#### `min_cell_samples=10000`

虽然比 `1000` 好，但仍偏激：

- `DexYCB@1100+` 仍会被过采 `19.4x ~ 34.2x`
- `HO3D_v3@0-500` 仍会被过采 `7x ~ 11x`

结论：

- 可以作为 second-best 方案；
- 但如果目标是“稳定提升泛化”，依然偏激进。

### 3.3 推荐设置

推荐先用：

```yaml
DATA:
  train:
    depth_bins:
      enabled: true
      root: /mnt/qnap/data/datasets/webdatasets/depth-bins
      dataset_names: [InterHand2.6M, DexYCB_s1, HO3D_v3, HOT3D]
      split: train
      mix_strategy: dataset_bin_balanced
      dataset_balance_alpha: 0.0
      min_cell_samples: 30000
      selected_bins: null
      shardshuffle: false
      sample_shuffle: 200
```

原因：

1. `min_cell_samples=30000`
   - 能去掉极小 cell；
   - 最高过采样倍率降到更可控的范围；
   - 避免用几百到几千个样本去代表一个深度区间。

2. `dataset_balance_alpha=0.0`
   - 对当前分布最稳；
   - 在保持 bin 等权的同时，不会再额外偏向极小 dataset；
   - 更适合作为第一版稳定 baseline。

这个设置下的训练 dataset 边缘分布约为：

- `DexYCB`: `36.67%`
- `InterHand2.6M`: `30.00%`
- `HO3D_v3`: `16.67%`
- `HOT3D`: `16.67%`

结论：

- 相比自然分布，`HO3D_v3` 被显著抬升；
- `InterHand2.6M` 不再绝对主导；
- 深度 bin 又保持严格等权；
- 这是当前代码框架下最均衡、最稳的一版。

### 3.4 推荐的第二阶段 ablation

如果第一阶段的 `alpha=0.0` 跑完后，近距离泛化仍然不足，第二个实验再试：

```yaml
dataset_balance_alpha: 0.25
min_cell_samples: 30000
```

这个设置下的 dataset 边缘分布约为：

- `DexYCB`: `38.998%`
- `InterHand2.6M`: `27.322%`
- `HO3D_v3`: `20.771%`
- `HOT3D`: `12.909%`

解释：

- 会继续抬高 `HO3D_v3`
- 会继续降低 `HOT3D` 和 `InterHand2.6M`
- 更偏向“补近距离 / 补 HO3D”

适用场景：

- 你主要关心 `HO3D` 或近距离域上的精度；
- 而不是总体平均最稳。

### 3.5 为什么我不建议从 `alpha=0.5` 开始

`alpha=0.5` 在当前数据上过于激进：

- `DexYCB` dataset marginal 会抬到 `40%+ ~ 58%+`
- 小 cell 的过采倍率偏大
- 很容易把“均衡采样”变成“高频重复少量样本”

因此：

- `alpha=0.5` 不适合作为第一版 scientific baseline；
- 只适合在明确确认某一深度 / 某一数据集仍显著不足时再尝试。

## 4. COCO-WholeBody 采样与 loss 方案

### 4.1 当前实现回顾

当前项目里，`COCO-WholeBody`：

- 只作为 `2D-only` 数据；
- 只走独立的 `lambda_coco_patch_2d`；
- 当前默认 `lambda_coco_patch_2d=0.0005`；
- 当前还没有采样配比控制。

详见：

- [COCO_WHOLEBODY_INTEGRATION.md](/data_1/renkaiwen/CS-ViT2/docs/COCO_WHOLEBODY_INTEGRATION.md)

### 4.2 当前 COCO 标注质量观察

对 `COCO-WholeBody/train/000000.tar` 的前 `1024` 个样本做了快速统计：

- `visible_joint_mean`: `20.92`
- `visible_joint_median`: `21`
- `P10 / P25 / P75 / P90`: 基本都是 `21`
- `visible_joint_ge_10_frac`: `99.71%`
- `intr_type`: 全是 `fixed_virtual`
- `has_intr`: 全是 `1`

结论：

1. 当前 `COCO-WholeBody` 不是“缺关节很多”的问题；
2. 按可见关节数做强过滤，收益不会很大；
3. 真正的问题更可能是：
   - 采样比例失控；
   - 虚拟内参和 3D 几何不一致；
   - 小 hand bbox 带来的高噪声 2D 监督。

### 4.3 COCO 采样建议

结论先说：

- 不要把 COCO 直接按自然样本量混进 `DATA.train.source`
- 要给 COCO 固定上限比例

推荐第一版：

- `3D : COCO = 9 : 1`
- 即 COCO 占训练样本的 `10%`

如果后续验证表明：

- in-the-wild 2D 对齐提升明显、
- 3D 指标没有被拖坏，

再试第二版：

- `3D : COCO = 7 : 1`
- 即 COCO 占 `12.5%`

不建议一开始就超过 `15%`。

原因：

- 当前 COCO 只有 2D 监督，没有真实 3D / MANO / real intrinsics；
- 它应该是 appearance/domain regularizer，而不是主监督源；
- 比例过高时，会拉偏绝对几何学习。

### 4.4 COCO loss 设计建议

当前实现已经做对的部分：

- 目标使用 `joint_patch_resized`
- 坐标在 patch 空间计算
- 单独使用 `lambda_coco_patch_2d`
- 与 3D / MANO loss 解耦

下一步最值得改的是这三点。

#### 建议 1：把 COCO patch loss 从标准 `L1` 改成 patch-space `RobustL1`

原因：

- COCO 是 2D-only；
- bbox、虚拟内参、人工标注都会引入 outlier；
- 当前直接用 `L1`，对极端误差点过于敏感。

建议：

- 只对 `COCO-WholeBody` 的 patch loss 使用 `RobustL1Loss`
- delta 不要沿用 3D reprojection 的 `84px`
- patch-space 建议从 `8px ~ 12px` 开始 sweep

推荐第一版：

- `delta = 10px`

#### 建议 2：加入 bbox-size-aware sample weight

当前快速统计里：

- `bbox sqrt(area)` 均值约 `24`
- `P10` 约 `12.5`
- `P50` 约 `20.5`
- `P90` 约 `37.7`

这说明：

- COCO 里有相当多非常小的手；
- 它们会在 preprocess 中被强行 resize 到固定 patch；
- 这些样本的监督噪声会更大。

因此建议：

- 不要让所有 COCO 样本等权参与 loss；
- 使用 `bbox` 大小做 sample weight；
- 或者至少过滤掉最小的一部分手框。

推荐第一版做法：

1. 过滤掉 `bbox sqrt(area) < 12px` 的 COCO 样本；
2. 对剩余样本按 bbox 尺寸做轻度加权；
3. 权重只作用于 COCO 2D loss，不影响 3D 数据。

#### 建议 3：加入 in-patch mask

当前 patch loss 目标是 `joint_patch_resized`，但没有额外判断：

- joint 是否明显落在 patch 外；
- bbox 是否导致坐标被强行映射到非常离谱的位置。

建议：

- 对 COCO 的 2D patch loss 增加 `in_patch_mask`
- 只监督落在 `[-m, W+m] x [-m, H+m]` 范围内的 joint

推荐 margin：

- `m = 8px`

这样可以减少：

- 边界框误差；
- 极端裁剪误差；
- 少数异常样本对训练的污染。

### 4.5 COCO 的最优训练方式建议

我不建议一开始就把 COCO 从 step 0 大比例混进去。

更稳的做法是：

#### 方案 A：延迟引入

- 前 `10% ~ 20%` 训练 steps：只用 3D 数据
- 后续再引入 COCO

目的：

- 先让绝对几何稳定
- 再用 COCO 提升外观域泛化

#### 方案 B：loss ramp-up

- `lambda_coco_patch_2d` 从 `0` 线性升到目标值

建议目标值 sweep：

- `0.0005`
- `0.0010`

推荐顺序：

1. 先跑 `0.0005`
2. 再试 `0.0010`

### 4.6 当前我最推荐的 COCO baseline

如果现在就要做一版“兼顾稳和提升精度”的 COCO 方案，我建议：

1. `3D : COCO = 9 : 1`
2. `lambda_coco_patch_2d = 0.0005`
3. COCO patch loss 改为 `RobustL1(delta=10)`
4. 过滤 `bbox sqrt(area) < 12px`
5. 加 `in_patch_mask`，margin=`8px`
6. COCO 从训练前 `10%` steps 之后再启用

## 5. 推荐的实验顺序

### 实验 1：3D-only 深度均衡 baseline

- `dataset_bin_balanced`
- `dataset_balance_alpha=0.0`
- `min_cell_samples=30000`

目标：

- 先确认 depth-bin 采样本身能不能提升跨深度泛化

### 实验 2：更强近距离补偿

- `dataset_balance_alpha=0.25`
- `min_cell_samples=30000`

目标：

- 观察 HO3D / 近距离域是否继续受益

### 实验 3：加入 COCO baseline

- 在实验 1 最优设置的基础上
- 引入 `3D : COCO = 9 : 1`
- `lambda_coco_patch_2d = 0.0005`

目标：

- 看 2D-only appearance regularization 是否能提升泛化

### 实验 4：加入 COCO robust / bbox-aware loss

- 在实验 3 最优设置的基础上
- patch-space `RobustL1(delta=10)`
- `bbox sqrt(area) < 12px` 过滤
- `in_patch_mask`

目标：

- 看 COCO 是否能从“能用”变成“稳定增益”

## 6. 结论

当前项目最值得优先做的，不是继续盲调 backbone，而是：

1. 先用 `dataset × depth-bin` 均衡采样纠正当前极强的 `dataset ↔ depth` 偏置；
2. 再用固定上限比例引入 COCO，而不是让它按自然样本量淹没训练；
3. 最后把 COCO loss 从“简单 L1 辅助项”升级成“robust + bbox-aware + in-patch mask”的 2D regularizer。

一句话总结：

- 3D 数据解决“几何覆盖”
- depth-bin 采样解决“深度偏置”
- COCO 解决“外观域泛化”
- 三者不能混成一个自然采样池，否则谁样本多谁说了算
