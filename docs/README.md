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

## 📖 阅读顺序

### 如果你想快速开始训练
1. 阅读 **[QUICK_START.md](QUICK_START.md)**
2. 运行测试并启动训练

### 如果你想了解改进细节
1. 阅读 **[IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md)** - 了解为什么做这些改动
2. 阅读 **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)** - 了解代码如何重构
3. 阅读 **[CODE_DESIGN.md](CODE_DESIGN.md)** - 深入理解设计原则

### 如果你想贡献代码
1. 阅读 **[CODE_DESIGN.md](CODE_DESIGN.md)** - 理解架构设计
2. 参考最佳实践和代码审查清单
3. 编写测试用例

## 🔗 相关文件

- 项目根目录的 **[CLAUDE.md](../CLAUDE.md)** - 项目整体说明和开发指南
- 测试脚本: **[tests/test_progressive_dropout.py](../tests/test_progressive_dropout.py)**
- 配置文件: **[config/stage1-dino_large.yaml](../config/stage1-dino_large.yaml)**

## 📅 更新日期

2026-01-30

---

**GG**
