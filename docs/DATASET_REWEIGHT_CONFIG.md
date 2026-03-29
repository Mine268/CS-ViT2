# Dataset Reweight 配置说明

本文档概括当前项目在常规 `webdatasets2` 训练链路中新增的 `DATA.train.reweight` 配置项，以及它和 `depth_bins` 的关系。

## 1. 目标

当前数据团队给出的 `root_z` 文档建议第一阶段先做 **dataset-level reweight**，而不是直接依赖 `depth-bin` 采样。

当前项目已支持：

- 在普通 `webdatasets2` 训练源上按 dataset 分组；
- 对每个 dataset 分别建流，再通过 `webdataset.RandomMix` 按权重混采；
- 与现有 `random_clip / dense` clip 采样逻辑兼容；
- 与 `depth_bins` 配置并列，但两者不能同时启用。

## 2. 配置位置

示例：

```yaml
DATA:
  train:
    source: [
      /mnt/qnap/data/datasets/webdatasets2/AssemblyHands/train/*,
      /mnt/qnap/data/datasets/webdatasets2/COCO-WholeBody/train/*,
      /mnt/qnap/data/datasets/webdatasets2/DexYCB/s1/train/*,
      /mnt/qnap/data/datasets/webdatasets2/FreiHAND/train/*,
      /mnt/qnap/data/datasets/webdatasets2/HO3D_v3/train/*,
      /mnt/qnap/data/datasets/webdatasets2/HOT3D/train/*,
      /mnt/qnap/data/datasets/webdatasets2/InterHand2.6M/train/*,
      /mnt/qnap/data/datasets/webdatasets2/MTC/train/*,
      /mnt/qnap/data/datasets/webdatasets2/RHD/train/*,
    ]
    reweight:
      enabled: true
      split: train
      datasets:
        - name: AssemblyHands
          source: /mnt/qnap/data/datasets/webdatasets2/AssemblyHands/train/*
          weight: 0.10
        - name: COCO-WholeBody
          source: /mnt/qnap/data/datasets/webdatasets2/COCO-WholeBody/train/*
          weight: 0.10
        - name: DexYCB
          source: /mnt/qnap/data/datasets/webdatasets2/DexYCB/s1/train/*
          weight: 0.10
        - name: FreiHAND
          source: /mnt/qnap/data/datasets/webdatasets2/FreiHAND/train/*
          weight: 0.15
        - name: HO3D_v3
          source: /mnt/qnap/data/datasets/webdatasets2/HO3D_v3/train/*
          weight: 0.10
        - name: HOT3D
          source: /mnt/qnap/data/datasets/webdatasets2/HOT3D/train/*
          weight: 0.10
        - name: InterHand2.6M
          source: /mnt/qnap/data/datasets/webdatasets2/InterHand2.6M/train/*
          weight: 0.20
        - name: MTC
          source: /mnt/qnap/data/datasets/webdatasets2/MTC/train/*
          weight: 0.15
        - name: RHD
          source: /mnt/qnap/data/datasets/webdatasets2/RHD/train/*
          weight: 0.10
      shardshuffle: 128
      post_clip_shuffle: 2048
```

## 3. 生效规则

- `DATA.train.reweight.enabled=true` 时，训练 loader 会直接读取 `reweight.datasets` 中的 `name / source / weight` 条目。
- 每个 dataset 单独建立 WebDataset 流，随后按权重做 `RandomMix`。
- 权重会自动归一化，不要求手动加和为 `1.0`。
- 若某个 dataset 条目的 `source` 没有匹配到任何 tar，会直接报错。
- 这种显式条目配置可以避免依赖字符串匹配去推断 `source ↔ weight` 关系。

## 4. 与 depth_bins 的关系

- `DATA.train.reweight.enabled` 与 `DATA.train.depth_bins.enabled` 不能同时为 `true`。
- 当前 9 数据集混训默认建议走 `reweight`。
- `depth_bins` 仍只适合已经准备好外部 `/mnt/qnap/.../depth-bins` 数据的实验。

## 5. 当前默认权重

当前三份默认启用 `reweight` 的配置：

- [stage1-dino_large_no_norm.yaml](/data_1/renkaiwen/CS-ViT2/config/stage1-dino_large_no_norm.yaml)
- [stage1-swinv2_large_no_norm.yaml](/data_1/renkaiwen/CS-ViT2/config/stage1-swinv2_large_no_norm.yaml)
- [stage2-dino_large_no_norm.yaml](/data_1/renkaiwen/CS-ViT2/config/stage2-dino_large_no_norm.yaml)

默认权重与数据团队包中的 `train_weighted_sampling_weights.json` 对齐：

- `AssemblyHands`: `0.10`
- `COCO-WholeBody`: `0.10`
- `DexYCB`: `0.10`
- `FreiHAND`: `0.15`
- `HO3D_v3`: `0.10`
- `HOT3D`: `0.10`
- `InterHand2.6M`: `0.20`
- `MTC`: `0.15`
- `RHD`: `0.10`

## 6. COCO-WholeBody 注意事项

- `COCO-WholeBody` 仍然只作为 `2D-only` 数据使用。
- 若训练源包含 `COCO-WholeBody`，必须保持 `MODEL.norm_by_hand=false`。
- 若希望 `COCO-WholeBody` 在混采中真正提供监督，需要同时设置 `LOSS.lambda_coco_patch_2d > 0`。
