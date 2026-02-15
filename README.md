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
