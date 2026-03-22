# V2 Dataset Migration Status

本文件概括当前项目从旧 `webdatasets` 迁移到 `webdatasets2` 的落地状态、改动范围和验证结果。

## 当前状态

已完成：
- 当前项目主数据链路已切到 V2 runtime 语义。
- Stage 1 / Stage 2 主配置已切到 `webdatasets2` 根目录。
- 当前已使用的 4 个数据集已纳入迁移范围：
  - `InterHand2.6M`
  - `DexYCB/s1`
  - `HO3D_v3`
  - `HOT3D`
- `loss / norm / metric / test / evaluate` 已改为使用细粒度 valid 字段：
  - `joint_2d_valid`
  - `joint_3d_valid`
  - `has_mano`
  - `has_intr`
- 已补上 `COCO-WholeBody` 的 Stage 1 训练侧最小接入：
  - `masked loss -> masked mean`
  - `COCO-WholeBody` 专用 `2D patch auxiliary loss`
  - `val/test/evaluate` 显式排除 `COCO-WholeBody` 的 3D 指标统计

未纳入本次范围：
- `depth_bins`

## 已改动模块

数据入口：
- `src/data/schema_v2.py`
- `src/data/dataloader.py`
- `src/data/preprocess.py`

训练与推理：
- `src/model/loss.py`
- `src/model/net.py`
- `src/utils/metric.py`
- `src/utils/vis.py`
- `script/train.py`
- `script/test.py`
- `script/evaluate.py`

配置：
- `config/stage1-dino_large.yaml`
- `config/stage2-dino_large.yaml`

回归测试同步：
- `tests/test_predict_full_norm_by_hand.py`
- `tests/test_checkpoint_regression.py`

## 语义变化

V2 运行时现在显式区分：
- `joint_2d_valid`: 2D 监督可用性
- `joint_3d_valid`: 3D 监督可用性
- `has_mano`: MANO 监督可用性
- `has_intr`: 内参可用性

兼容 alias 仍保留在 batch 中：
- `joint_valid -> joint_2d_valid`
- `mano_valid -> has_mano`
- `joint_patch_bbox -> joint_patch_origin`
- `joint_hand_bbox -> joint_hand_origin`

这些 alias 只用于兼容旧辅助代码，不再作为训练和评估依据。

## 验证产物

当前项目本地验证脚本 `tools/verify_wds_v2_local.py` 可视化输出：
- `temp/verify_wds_v2/interhand_train/`
- `temp/verify_wds_v2/dexycb_train/`
- `temp/verify_wds_v2/ho3d_train/`
- `temp/verify_wds_v2/hot3d_train/`

说明：
- processed MANO 可视化固定使用 `right MANO`；
- 原因是 preprocess 已将左右手统一归一化到 right-hand canonical space。

当前项目 loader/preprocess 摘要验证：
- `temp/v2_migration_validation/current_project_loader_summary.json`

验证内容包括：
- 原始 batch key 是否包含 V2 字段
- preprocess 后是否输出 `joint_patch_resized`
- 四个数据集的 `joint_2d_valid / joint_3d_valid / has_mano / has_intr` 是否存在
- patch tensor 和 patch-space joint tensor 形状是否正确

## 当前观察

从 `temp/v2_migration_validation/current_project_loader_summary.json` 可以确认：
- 四个数据集都能被当前项目的 loader 正常读取。
- preprocess 后都包含：
  - `joint_patch_origin`
  - `joint_patch_resized`
  - `joint_hand_origin`
  - `joint_2d_valid`
  - `joint_3d_valid`
  - `has_mano`
  - `has_intr`
- 四个数据集在抽样样本上都是 `intr_type=real`，当前迁移范围内不需要处理 `fixed_virtual` 逻辑。

## 已知未收口项

本次没有处理的旧字段引用仍存在于辅助离线脚本中：
- `script/preprocess/finger_norm.py`
- `script/preprocess/param_gaussian.py`

它们不在当前训练 / 测试 / 评估主链路内，后续如继续维护这些脚本，再单独迁移到新字段语义。

## 后续建议

`COCO-WholeBody` 当前仍有未收口项：
- `MODEL.norm_by_hand=true` 仍不支持
- 训练集混合采样策略尚未加入
- 还没有单独的 `COCO-WholeBody` 2D 指标统计
- Stage 2 暂未考虑接入 `COCO-WholeBody`
