# 文档目录

本目录包含CS-ViT2项目的训练优化和代码重构相关文档。

## 📚 文档列表

### 快速入门
- **[QUICK_START.md](QUICK_START.md)** - 快速开始指南
  - 如何验证代码
  - 如何启动训练
  - 如何监控训练进度
  - 故障排查

### 详细说明
- **[V2_DATASET_MIGRATION_STATUS.md](V2_DATASET_MIGRATION_STATUS.md)** - V2 数据集迁移状态与验证结果
  - 当前项目已切到 `webdatasets2` 的模块范围
  - `loss / norm / metric / test / evaluate` 的新 valid 语义
  - `temp/` 下的验证产物路径

- **[COCO_WHOLEBODY_INTEGRATION.md](COCO_WHOLEBODY_INTEGRATION.md)** - COCO-WholeBody 接入约束与 loss 设计
  - `COCO-WholeBody` 目前只作为 `2D-only` 数据使用
  - `lambda_coco_patch_2d` 的作用与默认值
  - `norm_by_hand=false` 的当前限制
  - 为什么 `val/test/evaluate` 要显式排除 COCO 参与 3D 指标

- **[DATASET_DEPTH_BALANCE_AND_COCO_PLAN.md](DATASET_DEPTH_BALANCE_AND_COCO_PLAN.md)** - 当前数据集深度分布与 COCO 训练方案
  - 当前 4 个 3D 数据集的全量 `dataset x depth-bin` 统计
  - `dataset_bin_balanced` 的候选超参与过采样倍率对比
  - 推荐的 `min_cell_samples / alpha` 设置
  - COCO-WholeBody 的采样比例与 loss 设计建议

- **[DATASET_REWEIGHT_CONFIG.md](DATASET_REWEIGHT_CONFIG.md)** - dataset-level reweight 配置说明
  - `DATA.train.reweight` 的配置位置与生效规则
  - 与 `depth_bins` 的互斥关系
  - 当前 train-only sample filter 的默认阈值
  - 当前三份 `no_norm` 配置的默认权重
  - `COCO-WholeBody` 在 reweight 下的注意事项

- **[ROOT_Z_PRIOR_MULTIBIN_DESIGN.md](ROOT_Z_PRIOR_MULTIBIN_DESIGN.md)** - Root-Z prior-centered multibin 设计说明
  - 仅替换 `z` 路径、保持 `x/y` 不变的 head 设计
  - `z_prior + Δlog z` 的表达与推理恢复流程
  - 当前 `stage1-dino_large_no_norm` 统计得到的 `k / d_min / d_max`
  - train-only filter 与 root-z-only stricter mask 的设计
  - 与现有 `softargmax3d` 的配置兼容方案

- **[PATCH_UV_RHO_MULTIBIN_DESIGN.md](PATCH_UV_RHO_MULTIBIN_DESIGN.md)** - Patch-UV + Rho MultiBin camera head 设计说明
  - 先在 patch 空间预测 root `uv`，再预测光心距离 `rho`
  - `uv + intrinsics + rho -> xyz` 的显式反投影路径
  - `q_ref=bbox center ray` 的动机与 full-records 数据支撑
  - 基于 `tests/temp_root_z_prior_records_stage1_full` 重统计的 `k / d_min / d_max`
  - 对应的 loss、接口与实现顺序建议

- **[CHECKPOINT_TEST_WORKFLOW.md](CHECKPOINT_TEST_WORKFLOW.md)** - 指定 checkpoint 在指定数据集上的单卡测试流程
  - 如何用 `script.test` 运行单卡测试
  - 如何用 `script.evaluate` 计算完整指标
  - HO3D / DexYCB / IH26M 的直接命令模板

- **[DEPTH_BIN_REPACK.md](DEPTH_BIN_REPACK.md)** - depth-bin 数据重整方案
  - 对每个 dataset/bin 单独 repack
  - 新目录重整后替换旧目录
  - `repack_stats.json` 与 `repack_summary.json` 说明

- **[DEPTH_BINS_DATASET_LAYOUT.md](DEPTH_BINS_DATASET_LAYOUT.md)** - depth-bins 数据目录结构说明
  - 外部 `/mnt/qnap/.../depth-bins` 的层级含义
  - `nf*_s*`、`bin_*`、`summary.json`、tar 文件说明
  - depth-bin sample 的新增字段说明

- **[DEPTH_BIN_WDS_PIPELINE.md](DEPTH_BIN_WDS_PIPELINE.md)** - 静态深度分桶 WebDataset 流程
  - 从现有 WDS 离线切 clip 并按深度分桶
  - 对应的 depth-bin dataloader 使用方式
  - `dataset × depth-bin` 受约束均衡采样
  - 目录结构、命令示例与验证状态

- **[dataset_depth_bin_sampling_weights_with_marginals.png](dataset_depth_bin_sampling_weights_with_marginals.png)** - `dataset × depth-bin` 采样权重与边缘分布
  - 展示当前 `alpha=0.5` 下的 cell 样本数、bin 边缘分布和 dataset 边缘分布

- **[dataset_depth_bin_sampling_before_after.png](dataset_depth_bin_sampling_before_after.png)** - balance 前后采样分布对比（`alpha=0.5`）
  - 对比自然样本分布与 `dataset_bin_balanced(alpha=0.5)` 的理论采样分布

- **[dataset_depth_bin_sampling_before_after_alpha0.png](dataset_depth_bin_sampling_before_after_alpha0.png)** - balance 前后采样分布对比（`alpha=0`）
  - 对比自然样本分布与 `dataset_bin_balanced(alpha=0)` 的理论采样分布

- **[DATALOADER_SAMPLING_STRATEGY.md](DATALOADER_SAMPLING_STRATEGY.md)** - 训练/验证/测试采样策略说明
  - 为什么训练不再枚举全部滑窗
  - `random_clip` 与 `dense` 的分工
  - 如何提升 batch 内随机性且避免爆内存

- **[tests/test_full_val_partition.py](../tests/test_full_val_partition.py)** - 完整 val 的 shard/clip 均衡分配测试
  - 验证多卡完整 val 的 clip 分配逻辑
  - 覆盖 tar 数小于 GPU 数时的边界情况

- **[IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md)** - 训练优化改进总结
  - 问题分析和诊断
  - 渐进式Dropout实现
  - 验证集一致性保证
  - 预期效果

- **[CODE_DESIGN.md](CODE_DESIGN.md)** - 代码设计文档
  - 架构设计原则
  - 核心组件说明
  - 数据流图
  - 最佳实践

- **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)** - 代码重构总结
  - 重构前后对比
  - 代码度量
  - 设计原则
  - 文件清单

- **[NORM_BY_HAND.md](NORM_BY_HAND.md)** - norm_by_hand 功能说明
  - 归一化尺度计算原理
  - 训练和推理逻辑
  - 与 shape 的关系
  - 与重投影 loss 的关联
  - 潜在问题和注意事项

- **[REPROJ_LOSS_CONFIG.md](REPROJ_LOSS_CONFIG.md)** - 重投影 loss 配置说明
  - RobustL1Loss 实现
  - 配置化 loss 类型选择
  - Delta 参数建议
  - 效果对比

- **[SWINV2_BACKBONE_COMPAT.md](SWINV2_BACKBONE_COMPAT.md)** - SwinV2 backbone 兼容说明
  - `SwinV2` 无 `cls token` 的输出差异
  - 为什么 `num_patch` 要按最终 stage token grid 计算
  - `stage1-swinv2_large.yaml` 的推荐配置
  - `infusion_layer=null` 的当前限制

### 更新日志
- **[CHANGELOG_2026-02-10.md](CHANGELOG_2026-02-10.md)** - 2026-02-10 更新日志
  - Stage 2 致命 bug 修复总结
  - 训练脚本统一说明
  - 性能优化列表

### Bug 修复与技术报告
- **[STAGE2_LAST_FRAME_ONLY_FIX.md](STAGE2_LAST_FRAME_ONLY_FIX.md)** - Stage 2 "只预测最后一帧" bug 修复
  - 致命 bug：batch 维度错乱
  - Loss 计算错误分析
  - 完整修复方案
  - 权重加载机制改进

- **[KORNIA_CROP_AND_RESIZE_BUG.md](KORNIA_CROP_AND_RESIZE_BUG.md)** - Kornia crop_and_resize bug 报告
  - 透视变换时的错误结果
  - 根因分析和复现代码
  - Workaround 方案

- **[PREDICT_FULL_NORM_BY_HAND_FIX.md](PREDICT_FULL_NORM_BY_HAND_FIX.md)** - predict_full() 反归一化逻辑修复
  - norm_valid 标志检查
  - 智能 Fallback 策略
  - 逐样本混合处理
  - 与训练逻辑一致性

- **[NAN_TRAINING_FIX.md](NAN_TRAINING_FIX.md)** - 训练 NaN 问题修复
  - norm_by_hand 除零问题
  - epsilon 保护实现
  - 数据异常诊断
  - 修复验证方法

- **[CENTER_CORRECTION_DESIGN.md](CENTER_CORRECTION_DESIGN.md)** - 透视归一化设计文档

- **[WHY_MIXED_PRECISION_MATTERS.md](WHY_MIXED_PRECISION_MATTERS.md)** - 混合精度训练说明

- **[ABLATION_EXPERIMENTS.md](ABLATION_EXPERIMENTS.md)** - 消融实验设计

- **[TEST_SCRIPT_USAGE.md](TEST_SCRIPT_USAGE.md)** - 测试脚本使用说明
  - 测试脚本功能介绍
  - 使用方法和参数
  - 输出格式和结果分析

- **[THESIS_CHAPTER_SUMMARY.md](THESIS_CHAPTER_SUMMARY.md)** - 论文章节总结

## 📖 阅读顺序

### 如果你想快速开始训练
1. 阅读 **[QUICK_START.md](QUICK_START.md)**
2. 阅读 **[DATALOADER_SAMPLING_STRATEGY.md](DATALOADER_SAMPLING_STRATEGY.md)**
3. 运行测试并启动训练

### 如果你想了解改进细节
1. 阅读 **[IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md)** - 了解为什么做这些改动
2. 阅读 **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)** - 了解代码如何重构
3. 阅读 **[CODE_DESIGN.md](CODE_DESIGN.md)** - 深入理解设计原则

### 如果你当前主线是 no_norm + 新 camera head
1. 阅读 **[ROOT_Z_PRIOR_MULTIBIN_DESIGN.md](ROOT_Z_PRIOR_MULTIBIN_DESIGN.md)** - 了解当前 `xy_rootz_multibin` 的设计基线
2. 阅读 **[PATCH_UV_RHO_MULTIBIN_DESIGN.md](PATCH_UV_RHO_MULTIBIN_DESIGN.md)** - 了解 `patch uv + rho` 新路径的表达、先验与超参数
3. 对照 **[DATASET_REWEIGHT_CONFIG.md](DATASET_REWEIGHT_CONFIG.md)** 和 **[DATALOADER_SAMPLING_STRATEGY.md](DATALOADER_SAMPLING_STRATEGY.md)** 检查当前训练主线

### 如果你在训练 Stage 2
1. 阅读 **[STAGE2_LAST_FRAME_ONLY_FIX.md](STAGE2_LAST_FRAME_ONLY_FIX.md)** - 了解 Stage 2 的关键修复
2. 检查配置文件中的 `num_frame` 和 `stage1_weight` 设置
3. 使用统一的 `script/train.py` 脚本

### 如果遇到问题或 bug
1. 查看 **[KORNIA_CROP_AND_RESIZE_BUG.md](KORNIA_CROP_AND_RESIZE_BUG.md)** - 透视变换相关问题
2. 查看 **[STAGE2_LAST_FRAME_ONLY_FIX.md](STAGE2_LAST_FRAME_ONLY_FIX.md)** - Stage 2 实现问题
3. 检查 **[QUICK_START.md](QUICK_START.md)** 的故障排查部分

### 如果你想贡献代码
1. 阅读 **[CODE_DESIGN.md](CODE_DESIGN.md)** - 理解架构设计
2. 参考最佳实践和代码审查清单
3. 编写测试用例

## 🔗 相关文件

- 项目根目录的 **[CLAUDE.md](../CLAUDE.md)** - 项目整体说明和开发指南
- 测试脚本: **[tests/test_progressive_dropout.py](../tests/test_progressive_dropout.py)**
- 训练脚本: **[script/train.py](../script/train.py)** - 统一的 Stage 1/2 训练脚本
- 配置文件:
  - **[config/stage1-dino_large.yaml](../config/stage1-dino_large.yaml)** - Stage 1 配置
  - **[config/stage2-dino_large.yaml](../config/stage2-dino_large.yaml)** - Stage 2 配置

## 📅 更新日期

**最后更新**: 2026-04-02

**重要更新**:
- 2026-04-02: 实现 `patch_uv_rho_multibin` camera head 的基础版本，新增设计文档，并基于 `tests/temp_root_z_prior_records_stage1_full` 重统计 `rho` 路径的 `k / d_min / d_max`
- 2026-03-30: 实现 `xy_rootz_multibin` camera head 与对应 loss/配置兼容层，当前 `*_no_norm` 配置默认仍回退到 `softargmax3d`
- 2026-03-30: 增加 `root_z` prior-centered multibin 设计文档，并根据当前 `stage1-dino_large_no_norm` 统计结果给出 `k / d_min / d_max` 推荐值
- 2026-03-29: 新增 `DATA.train.reweight` 配置与对应 loader，三份 `no_norm` 配置已对齐到 9 数据集并默认启用 dataset-level reweight
- 2026-03-22: 增加当前数据集深度分布与 COCO 训练方案文档，明确 `dataset × depth-bin` 和 COCO 采样/loss 的推荐设置
- 2026-03-22: 增加 `swinv2-large-patch4-window12-192-22k` backbone 兼容代码和 `stage1-swinv2_large.yaml`
- 2026-03-22: 增加 `COCO-WholeBody` 专用 2D patch auxiliary loss，并在 val/test/evaluate 中显式排除 COCO 的 3D 指标统计
- 2026-03-22: 完成四个已用数据集到 `webdatasets2` 的 V2 迁移，并补充 temp/ 验证产物与状态文档
- 2026-03-07: 增加 depth-bin repack 脚本与说明
- 2026-03-07: 同步 depth-bins 外部目录 README 到 docs
- 2026-03-07: 增加静态深度分桶 WebDataset 流程文档与对应 dataloader
- 2026-03-07: 增加 train/val/test 采样策略文档，训练默认切换为 `random_clip`
- 2026-02-11: norm_by_hand 功能文档和重投影 loss 配置文档完善
- 2026-02-10: Stage 2 "只预测最后一帧" bug 修复，脚本统一为 `train.py`
- 2026-02-10: predict_full() 反归一化逻辑修复，重投影 loss 配置化
- 2026-02-08: Kornia crop_and_resize bug 修复，透视变换优化
- 2026-01-30: 渐进式 Dropout 训练优化

---

**GG**
