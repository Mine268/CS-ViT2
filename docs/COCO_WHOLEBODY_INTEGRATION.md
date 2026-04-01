# COCO-WholeBody 接入说明

本文件概括当前项目将 `COCO-WholeBody` 接入 Stage 1 / Stage 2 训练链路时的约束、loss 设计和验证口径。

## 当前实现

- `COCO-WholeBody` 只作为 `2D-only` 数据使用。
- 训练时新增 `COCO-WholeBody` 专用的 `2D patch auxiliary loss`。
- 该 loss 只对 `data_source == "COCO-WholeBody"` 的样本生效。
- 现有 `3D / MANO / reprojection` loss 继续只由可用的 3D / MANO 标注驱动。

当前默认配置状态：

- [stage1-dino_large_no_norm.yaml](/data_1/renkaiwen/CS-ViT2/config/stage1-dino_large_no_norm.yaml) 当前已将 `COCO-WholeBody` 训练源注释掉
- [stage1-swinv2_large_no_norm.yaml](/data_1/renkaiwen/CS-ViT2/config/stage1-swinv2_large_no_norm.yaml) 当前已将 `COCO-WholeBody` 训练源注释掉
- [stage2-dino_large_no_norm.yaml](/data_1/renkaiwen/CS-ViT2/config/stage2-dino_large_no_norm.yaml) 当前已将 `COCO-WholeBody` 训练源注释掉
- [stage1-dino_large.yaml](/data_1/renkaiwen/CS-ViT2/config/stage1-dino_large.yaml) 未默认加入，因为 `MODEL.norm_by_hand=true`
- [stage1-swinv2_large.yaml](/data_1/renkaiwen/CS-ViT2/config/stage1-swinv2_large.yaml) 未默认加入，因为 `MODEL.norm_by_hand=true`
- 当前代码仍支持通过 `DATA.train.reweight` 显式接回 `COCO-WholeBody`

## Loss 设计

- 新增配置项：`LOSS.lambda_coco_patch_2d`
- 默认值：`0.0005`
- 目标坐标：`joint_patch_resized`
- 有效性掩码：`joint_2d_valid`
- 额外约束：`has_intr`

计算链路：

1. `joint_cam_pred`
2. 通过 `focal / princpt` 投影到原图坐标
3. 通过 `patch_bbox` 映射到 resized patch 坐标
4. 与 `joint_patch_resized` 做 `L1 + masked mean`

注意：

- 当前版本没有为 `COCO-WholeBody` 单独启用 RobustL1。
- 当前版本已引入 dataset-level `reweight` 配置，用于控制 `COCO-WholeBody` 的混采比例。

## masked mean 收口

当前训练主损失已经统一改成“只按有效元素归一”的 `masked mean`。

这解决了一个关键问题：

- 当 batch 中混入大量 `joint_3d_valid=0` 或 `has_mano=0` 的样本时，
- 旧写法 `torch.mean(loss * mask)` 会稀释有效样本的损失，
- 现在改为按有效元素数归一后，不再被无效样本冲淡。

## norm_by_hand 约束

当前版本明确不支持：

- `COCO-WholeBody` 参与训练时同时设置 `MODEL.norm_by_hand=true`

原因：

- `COCO-WholeBody` 没有 GT 3D hand scale，
- 当前项目还没有实现该分支下稳定可靠的 2D-only 反归一化训练路径。

因此代码里加了两层保护：

- 训练启动前配置断言
- 运行时 batch 级断言

使用要求：

- 若训练源中包含 `COCO-WholeBody`，必须设置 `MODEL.norm_by_hand=false`

## 验证与测试口径

- `COCO-WholeBody` 不参与 `val/test/evaluate` 的 3D 指标计算
- 过滤目标包括：
  - `MPJPE`
  - `MPVPE`
  - `rel-MPJPE`
  - `rel-MPVPE`
  - `RTE`

当前策略是：

- 结果文件仍可保留 `COCO-WholeBody` 样本
- 但计算指标时会按 `data_source != "COCO-WholeBody"` 过滤

## 当前未做

- 没有为 COCO 单独设计 2D 可视化指标
- 没有为 COCO 单独设计 2D-only 的评估指标
