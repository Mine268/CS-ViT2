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
2. 运行测试并启动训练

### 如果你想了解改进细节
1. 阅读 **[IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md)** - 了解为什么做这些改动
2. 阅读 **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)** - 了解代码如何重构
3. 阅读 **[CODE_DESIGN.md](CODE_DESIGN.md)** - 深入理解设计原则

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

**最后更新**: 2026-02-11

**重要更新**:
- 2026-02-11: norm_by_hand 功能文档和重投影 loss 配置文档完善
- 2026-02-10: Stage 2 "只预测最后一帧" bug 修复，脚本统一为 `train.py`
- 2026-02-10: predict_full() 反归一化逻辑修复，重投影 loss 配置化
- 2026-02-08: Kornia crop_and_resize bug 修复，透视变换优化
- 2026-01-30: 渐进式 Dropout 训练优化

---

**GG**
