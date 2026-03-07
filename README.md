# Training

## 1. 环境准备

```shell
uv venv .venv --python=3.12
source .venv/bin/activate
uv pip install -r requirements.txt
# 此时环境应该正常激活，bash前缀新增 (.venv)
```

### 1.1 训练环境配置

本项目采用`accelerate`完成多卡训练支持，在完成环境安装之后运行如下命令完成环境配置，如实填写即可：

```bash
accelerate config
```

## 2. 数据准备

```bash
cp -r /data_1/renkaiwen/CS-ViT2/model YOUR_PROJECT_ROOT
```

确保可以访问到NAS服务器地址`10.156.232.132:/qnapdata/datasets/webdatasets`，若不能请挂载。

## 3. 训练准备

### 3.1 训练日志服务器启动

本项目采用`aim`进行日志数据管理，该库采用了C-S架构，支持多个训练同时写入训练日志，支持同时监看多个训练情况。

`aim`服务端放在73服务器的`/data_1/renkaiwen/aim_logs`下，一经启动就不用管他了，启动命令如下（请确保`aim`通过包管理器安装，建议使用`uv`）：

```bash
tmux new -s aim_server
source .venv/bin/activate # 在/data_1/renkaiwen/aim_logs下创建的环境，只用安装aim就行
aim server --host 10.208.33.93
```

`aim`服务器的客户端用于将日志内容以网页的形式展示出来，同样在`/data_1/renkaiwen/aim_logs`下，运行如下命令启动服务器：
```bash
tmux new -s aim_up
source .venv/bin/activate
aim up --host 10.208.33.93 --read-only
```

此时访问`10.208.33.93:43800`可以浏览训练日志。如果出现日志不更新的情况，请kill之后重新运行上述的`aim up`命令。

针对内网地址`10.208.33.93:43800`已经配置了frp，可以进行外网访问，外网地址在群里有。

### 3.2 训练启动

训练分为两个阶段：

1. stage1：单帧训练阶段，该阶段训练backbone+persp\_info\_embedder+handec，时序模块不参与训练和推理，该阶段旨在初始化好模模型基座；
2. stage2：时序训练阶段，该阶段旨在stage1的基础上添加时序推理模块，借助时序信息提升姿态估计的准确度。

请将训练放在`tmux`环境中运行。

#### 3.2.1 stage1训练

一个标准的stage1训练启动命令如下：

```bash
accelerate launch --main_process_port 0 --gpu_ids 0,1,2,3 --num_processes 4 -m script.train \
    --config-name=stage1-dino_large \
    augmentation=color_jitter_only \
    TRAIN.sample_per_device=32 \
    TRAIN.persp_rot_max=0.5235987755982988 \
    GENERAL.description="'try debugging nan'"
```

1. setup参数：
    - `--main_process_port`：多卡训练的master端口号，设置为`0`让`accelerate`自己选择端口避免撞车
    - `--gpu_ids`：训练使用的卡的编号
    - `--num_processes`：几个卡就填几
    - `-m script.train`：以模块的方式启动巡礼啊
2. 训练配置参数：
    - `--config-name`：用于配置训练、模型、数据、测试、损失、日志的参数，指向`config`下的配置文件，目前可选：
        - `stage1-dino_large`：stage1训练的配置文件
        - `stage2-dino_large`：stage2训练的配置文件
    - 可以通过CLI参数对yaml配置文件中的内容进行覆盖，例如`stage1-dino_large`中学习率被配置为`1e-4`，可以通过参数`TRAIN.lr=1e-3`覆盖为`1e-3`
    - `augmentation=color_jitter_only`：用于配置训练时使用的数据增强手段，其他的可选项见`config/augmentation`，每一个yaml文件表示一种增强策略，直接指定对应文件名即可
    - `GENERAL.description="'try debugging nan'"`：用于对训练进行简单的描述，描述内容放在特殊引号组`"'XXX'"`中
3. 训练启动之后将创建`checkpoint/%Y-%m-%d/%H-%M-%S_${config-name}`文件夹，训练日志的文字版本、训练配置、训练中途存储的模型、状态都将存储在这里

**调试技巧1**：在调试中如果不希望调试runs的信息写入到AIM server造成干扰，可以添加参数`AIM.server_url=.`将日志写入到项目文件夹。

**调试技巧2**：调试相关的配置在`.vscode/launch.json`里面有，可以参考。

#### 3.2.2 stage2训练

一个标准的stage2训练启动命令如下：

```bash
accelerate launch --main_process_port 0 --gpu_ids 0,1,2,3 --num_processes 4 -m script.train \
    --config-name=stage2-dino_large \
    MODEL.stage1_weight=checkpoint/2026-02-11/23-53-36_stage1-dino_large/best_model \
    GENERAL.description="'stage2 training'"
```

- `--config-name`：此时配置文件使用`stage2-dino_large`，该文件是写好的，能够适配`stage1-dino_large`的训练checkpoint
- `MODEL.stage1_weight`：由于训练需要在stage1的基础上进行，所以必须通过这个参数指定权重。为了调试的方便可以将`/data_1/renkaiwen/CS-ViT2/checkpoint`下的这个文件夹拷贝走

命令的其他行为与stage1一样，不做赘述。

## 4. 测试/推理

使用 `script/test.py` 在测试集上进行模型评估。

### 4.1 基本用法

```bash
# 单卡测试
python script/test.py \
    --config-name=stage1-dino_large \
    TEST.checkpoint_path=checkpoint/exp/checkpoints/checkpoint-30000 \
    DATA.test.source='[/path/to/test/data/*.tar]'

# 多卡测试（推荐用于大批量推理）
accelerate launch --gpu_ids 0,1,2,3 --num_processes 4 -m script.test \
    --config-name=stage1-dino_large \
    TEST.checkpoint_path=checkpoint/exp/checkpoints/checkpoint-30000 \
    DATA.test.source='[/path/to/test/data/*.tar]'
```

### 4.2 必需参数

| 参数 | 说明 |
|------|------|
| `TEST.checkpoint_path` | Checkpoint 路径（如 `checkpoint/exp/checkpoints/checkpoint-30000`） |
| `DATA.test.source` | 测试数据路径，支持 glob 模式（如 `'[/mnt/data/test/*.tar]'`） |

### 4.3 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TEST.batch_size` | 32 | 测试 batch size |
| `TEST.max_samples` | null | 限制测试样本数（用于调试） |
| `TEST.output_dir` | 自动推断 | 输出目录，默认从 checkpoint_path 自动计算 |
| `TEST.enable_vis` | true | 是否启用 AIM 可视化 |
| `TEST.vis_step` | 10 | 可视化频率（每 N 个 batch） |
| `TEST.compression` | gzip | HDF5 压缩方式（gzip/lzf/null） |

### 4.4 输出结果

测试结果保存在 `checkpoint/{exp_name}/test_results/` 目录下：

| 文件 | 说明 |
|------|------|
| `predictions.h5` | 预测结果（包含 joint_cam_pred, vert_cam_pred, mano_pose_pred 等） |
| `metrics.json` | 快速评估指标（MPJPE, MPVPE） |
| `test_config.yaml` | 测试使用的配置 |

### 4.5 示例命令

```bash
# 在 HO3D 测试集上评估 Stage 1 模型
python -m script.test \
    --config-name=stage1-dino_large_no_norm \
    TEST.checkpoint_path=checkpoint/2026-03-05/21-07-09_stage1-dino_large_no_norm/best_model \
    DATA.test.source=['/mnt/qnap/data/datasets/webdatasets/HO3D_v3/evaluation/*.tar'] \
    TEST.batch_size=64

# 在 DexYCB 上评估 Stage 2 模型（多卡）
accelerate launch --gpu_ids 0,1 -m script.test \
    --config-name=stage2-dino_large \
    TEST.checkpoint_path=checkpoint/2026-02-12/stage2/checkpoints/checkpoint-30000 \
    DATA.test.source=['/path/to/dexycb/*.tar'] \
    TEST.batch_size=32

# 仅测试前 100 个样本（快速验证）
python script/test.py \
    --config-name=stage1-dino_large \
    TEST.checkpoint_path=checkpoint/exp/checkpoints/best_model \
    DATA.test.source=['/path/to/test/*.tar'] \
    TEST.max_samples=100
```

### 4.6 注意事项

1. 配置文件需与训练时使用的配置一致（Stage 1 用 `stage1-dino_large`，Stage 2 用 `stage2-dino_large`）
2. 测试时会自动关闭数据增强
3. 多卡测试会自动合并各进程的结果
4. 如需禁用 AIM 可视化，设置 `TEST.enable_vis=false` 或 `AIM.server_url=.`
