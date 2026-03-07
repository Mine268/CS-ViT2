# CS-ViT2 Project Guide for AI Agents

## 项目概述

CS-ViT2 是一个基于 Vision Transformer 的 **3D 手部姿态估计**项目，支持单帧（Stage 1）和时序（Stage 2）两个阶段的手部姿态预测。项目预测：
- 21 个手部关节的 3D 位置
- 778 个手部网格顶点（基于 MANO 模型）
- 完整的 MANO 参数（姿态、形状、平移）

## 技术栈

| 类别 | 技术 |
|------|------|
| 深度学习框架 | PyTorch 2.9.1 |
| 多 GPU 训练 | Hugging Face Accelerate |
| 骨干网络 | DINOv2, MAE, SwinV2 (Hugging Face Transformers) |
| 手部模型 | MANO (SMPLX 库) |
| 配置管理 | Hydra |
| 数据加载 | WebDataset |
| 实验跟踪 | AIM |
| 数据增强 | Kornia |

## 项目结构

```
CS-ViT2/
├── src/                          # 源代码
│   ├── model/                    # 神经网络模型
│   │   ├── net.py               # PoseNet: 主模型架构
│   │   ├── module.py            # ViTBackbone, HandDecoder, TemporalEncoder
│   │   ├── loss.py              # 损失函数
│   │   └── hamer_module.py      # 基础模块 (Attention, TransformerDecoder)
│   ├── data/                     # 数据加载与预处理
│   │   ├── dataloader.py        # WebDataset 数据加载器
│   │   └── preprocess.py        # 数据预处理与增强
│   ├── utils/                    # 工具函数
│   │   ├── mano.py              # MANO 模型工具
│   │   ├── metric.py            # 评估指标 (MPJPE, MPVPE)
│   │   ├── proj.py              # 3D 到 2D 投影
│   │   ├── rot.py               # 旋转矩阵工具
│   │   ├── vis.py               # 可视化
│   │   └── train_utils.py       # 训练工具
│   └── constant.py               # 常量定义
├── script/                       # 训练和测试脚本
│   ├── train.py                 # 主训练脚本（支持 Stage 1 和 Stage 2）
│   ├── test.py                  # 测试/推理脚本
│   └── evaluate.py              # 评估脚本
├── config/                       # Hydra 配置文件
│   ├── stage1-dino_large.yaml   # Stage 1 配置
│   ├── stage2-dino_large.yaml   # Stage 2 配置
│   └── augmentation/            # 数据增强配置
├── tests/                        # 单元测试
├── tools/                        # 分析与调试工具
├── model/                        # 预训练模型和 MANO 模型文件
│   ├── facebook/                # DINOv2, MAE 模型
│   ├── microsoft/               # SwinV2 模型
│   └── smplx_models/mano/       # MANO 模型文件
└── checkpoint/                   # 训练检查点（自动生成）
```

## 环境搭建

```bash
# 1. 创建虚拟环境
uv venv .venv --python=3.12
source .venv/bin/activate

# 2. 安装依赖
uv pip install -r requirements.txt

# 3. 配置 Accelerate（多 GPU 训练）
accelerate config
```

## 训练命令

### Stage 1: 单帧训练

```bash
# 基础训练
python script/train.py --config-name=stage1-dino_large

# 多 GPU 训练
accelerate launch --main_process_port 0 --gpu_ids 0,1,2,3 --num_processes 4 -m script.train \
    --config-name=stage1-dino_large \
    augmentation=color_jitter_only \
    TRAIN.sample_per_device=32 \
    GENERAL.description="'experiment description'"

# 恢复训练
python script/train.py --config-name=stage1-dino_large \
    GENERAL.resume_path=checkpoint/2026-03-07/12-00-00_stage1-dino_large/checkpoints/checkpoint-30000
```

### Stage 2: 时序训练

```bash
accelerate launch --main_process_port 0 --gpu_ids 0,1,2,3 --num_processes 4 -m script.train \
    --config-name=stage2-dino_large \
    MODEL.stage1_weight=checkpoint/2026-02-11/23-53-36_stage1-dino_large/best_model \
    GENERAL.description="'stage2 training'"
```

**重要**: Stage 2 需要指定 Stage 1 的检查点路径 (`MODEL.stage1_weight`)。

## 测试/推理命令

```bash
# 单卡测试
python script/test.py \
    --config-name=stage1-dino_large \
    TEST.checkpoint_path=checkpoint/exp/checkpoints/checkpoint-30000 \
    DATA.test.source='[/path/to/test/data/*.tar]'

# 多卡测试
accelerate launch --gpu_ids 0,1,2,3 --num_processes 4 -m script.test \
    --config-name=stage1-dino_large \
    TEST.checkpoint_path=checkpoint/exp/checkpoints/checkpoint-30000 \
    DATA.test.source='[/path/to/test/data/*.tar]'

# 限制样本数（快速验证）
python script/test.py \
    --config-name=stage1-dino_large \
    TEST.checkpoint_path=checkpoint/exp/checkpoints/best_model \
    DATA.test.source='[/path/to/test/*.tar]' \
    TEST.max_samples=100
```

## 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/test_src_model_net.py

# 运行特定测试函数
pytest tests/test_src_model_net.py::test_PoseNet1

# 详细输出
pytest tests/ -v -s
```

## 核心架构

### 两阶段训练架构

**Stage 1 (单帧)**:
- 输入: 单帧图像 [B, 1, 3, 224, 224]
- 输出: MANO 参数（每帧独立）
- 训练模块: Backbone + PerspInfoEmbedder + HandDecoder
- 配置: `MODEL.stage=stage1`, `MODEL.num_frame=1`

**Stage 2 (时序)**:
- 输入: 图像序列 [B, T, 3, 224, 224] (T=7 帧)
- 输出: **仅最后一帧**的精炼 MANO 参数 [B, 1, ...]
- 训练模块: 仅 TemporalEncoder（空间模块冻结）
- 配置: `MODEL.stage=stage2`, `MODEL.num_frame=7`

### 模型组件

1. **PoseNet** (`src/model/net.py`): 主模型类
2. **ViTBackbone** (`src/model/module.py`): Vision Transformer 骨干
3. **HandDecoder** (`src/model/module.py`): MANO 参数解码器
4. **TemporalEncoder** (`src/model/module.py`): 时序精炼模块（Stage 2）
5. **PerspInfoEmbedder** (`src/model/module.py`): 相机参数嵌入

## 配置系统 (Hydra)

配置文件位于 `config/` 目录，使用 YAML 格式。

关键配置组:
- **GENERAL**: 训练步数、日志频率、检查点频率
- **TRAIN**: 学习率、批量大小、混合精度
- **MODEL**: 模型架构参数
- **DATA**: 数据集路径
- **LOSS**: 损失函数权重
- **AIM**: 实验跟踪服务器地址

配置覆盖示例:
```bash
python script/train.py --config-name=stage1-dino_large \
    TRAIN.lr=5e-5 \
    TRAIN.sample_per_device=16 \
    GENERAL.total_step=50000
```

## 数据格式

WebDataset 格式 (`.tar` 文件)，每个样本包含:

```python
{
    "imgs": [B, T, 3, H, W],          # RGB 图像
    "joint_cam": [B, T, 21, 3],       # 3D 关节位置（相机坐标系）
    "verts_cam": [B, T, 778, 3],      # 网格顶点
    "mano_pose": [B, T, 48],          # MANO 姿态参数（轴角）
    "mano_shape": [B, T, 10],         # MANO 形状参数
    "focal": [B, T, 2],               # 焦距 (fx, fy)
    "princpt": [B, T, 2],             # 主点 (cx, cy)
    "hand_bbox": [B, T, 4],           # 边界框 (x, y, w, h)
    "timestamp": [B, T],              # 时间戳（Stage 2 需要）
}
```

## 开发规范

### 代码组织原则
- `src/model/`: 纯神经网络模块，无 I/O 或训练逻辑
- `src/data/`: 数据加载、预处理、增强
- `src/utils/`: 指标计算、可视化、坐标变换
- `script/`: 训练入口和编排逻辑
- `config/`: 所有超参数和数据集路径
- `tests/`: 单元测试
- `tools/`: 分析和调试工具脚本

### 命名约定
- 类名: `PascalCase`
- 函数/变量: `snake_case`
- 常量: `UPPER_SNAKE_CASE`
- 私有函数: `_leading_underscore`

### 注释语言
项目主要使用**中文**注释和文档字符串。

## 重要注意事项

1. **运行前激活环境**: 执行任何命令前，先运行 `source .venv/bin/activate`

2. **MANO 模型文件**: 需要专用 MANO 模型文件 (`model/smplx_models/mano/`):
   - `MANO_LEFT.pkl`, `MANO_RIGHT.pkl`
   - `mano_mean_params.npz`
   - `norm_stats.npz` 等

3. **数据集路径**: 配置文件中包含绝对路径 `/mnt/qnap/data/datasets/`，可能需要根据环境修改

4. **CUDA 设备**: 部分测试文件硬编码了设备（如 `cuda:3`），根据需要修改

5. **检查点管理**: 默认只保留最近的 3 个检查点以节省磁盘空间

6. **调试技巧**:
   - 使用 `AIM.server_url=.` 将日志写入本地而非 AIM 服务器
   - 使用 `.vscode/launch.json` 中的调试配置

7. **Stage 2 注意事项**:
   - 仅预测最后一帧（不是全部 T 帧）
   - 必须提供 Stage 1 的检查点路径

8. **batch 维度**: 单帧预测始终使用 [B, 1, ...] 形状（T=1）

9. **norm_by_hand**:
   - `true`: 按手部大小归一化坐标（更稳定）
   - `false`: 使用原始相机坐标

## 检查点结构

```
checkpoint/<date>/<time>_<config>_<hash>/
├── checkpoints/
│   ├── checkpoint-3000/          # 定期保存的检查点
│   ├── checkpoint-6000/
│   └── checkpoint-9000/
├── best_model/                    # 验证集最优模型
├── best_model_info.json           # 最优模型元数据
├── config_*.yaml                  # 保存的配置
└── log.txt                        # 训练日志
```

## 常见问题

**Q: 如何切换骨干网络？**
A: 修改配置文件中的 `MODEL.backbone.backbone_str`，并确保模型已下载到 `model/<org>/<model-name>/`，同时调整 `MODEL.handec.context_dim` 匹配骨干输出维度。

**Q: 如何添加新数据集？**
A:
1. 将数据预处理为 WebDataset 格式（`.tar` 文件）
2. 在配置文件中添加到 `DATA.train.source` 或 `DATA.val.source`
3. 确保数据格式与 `src/data/dataloader.py` 中的预期一致

**Q: 如何修改损失函数？**
A:
1. 在 `src/model/loss.py` 中实现新损失
2. 在 `src/model/net.py` 的 `PoseNet.__init__` 中实例化
3. 在配置文件中添加权重参数
4. 在 `BundleLoss2.forward` 中组合损失

# Agent 重要注意事项
1. 修改代码之前行程计划与用户讨论，严禁直接上手改代码
2. 每一次回答之后加一句 GG