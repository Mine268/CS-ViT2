# 使用指定 checkpoint 在指定数据集上进行单卡测试

本文档给出最直接的操作流程：如何使用某个 checkpoint 在某个数据集上做 **单卡 test**，以及如何基于 `predictions.h5` 计算更完整的测试指标。

## 1. 基本流程

单卡测试分两步：

1. 用 `script.test` 生成预测结果
2. 用 `script.evaluate` 计算完整指标

推荐命令格式：

```bash
cd /data_1/renkaiwen/CS-ViT2
source .venv/bin/activate

python -m script.test \
  --config-name=<config_name> \
  TEST.checkpoint_path=<checkpoint_dir> \
  DATA.test.source='[<dataset_wds_glob>]' \
  TEST.batch_size=<batch_size>
```

测试完成后，会在 checkpoint 同级目录下自动生成一个新的 `test_results_时间戳/` 目录，其中包含：

- `predictions.h5`
- `metrics.json`
- `test_config.yaml`
- `test_sources.txt`

如果要计算更完整的指标（例如 `rel_mpjpe`），继续执行：

```bash
python script/evaluate.py <test_results_dir>/predictions.h5
```

默认会在同目录下生成：

- `eval_metrics.json`

## 2. 参数含义

- `--config-name`
  - 训练该 checkpoint 时对应的配置名
  - 例如：`stage1-dino_large_no_norm`

- `TEST.checkpoint_path`
  - 指向具体 checkpoint 目录
  - 例如：
    - `checkpoint/2026-03-08/17-48-07_stage1-dino_large_no_norm_xxx/best_model`
    - `checkpoint/.../best_model_rte`
    - `checkpoint/.../best_model_rel_mpjpe`

- `DATA.test.source`
  - 目标测试集的 WebDataset tar 路径列表
  - Hydra 列表格式必须写成：
  - `'[/path/to/data/*.tar]'`

- `TEST.batch_size`
  - 单卡测试 batch size
  - IH26M 这种大集可以适当调大

## 3. 常用数据集路径

### HO3D evaluation

```text
/mnt/qnap/data/datasets/webdatasets/HO3D_v3/evaluation/*.tar
```

### DexYCB test

```text
/mnt/qnap/data/datasets/webdatasets/DexYCB/s1/test/*.tar
```

### InterHand2.6M val

```text
/mnt/qnap/data/datasets/webdatasets/InterHand2.6M/val/*.tar
```

### InterHand2.6M test

```text
/mnt/qnap/data/datasets/webdatasets/InterHand2.6M/test/*.tar
```

## 4. 示例：使用某个 `best_model` 做单卡测试

假设 checkpoint 为：

```text
checkpoint/2026-03-08/17-48-07_stage1-dino_large_no_norm_5748b860a5c24552a31bc36a/best_model
```

配置为：

```text
stage1-dino_large_no_norm
```

### 4.1 HO3D

```bash
python -m script.test \
  --config-name=stage1-dino_large_no_norm \
  TEST.checkpoint_path=checkpoint/2026-03-08/17-48-07_stage1-dino_large_no_norm_5748b860a5c24552a31bc36a/best_model \
  DATA.test.source='[/mnt/qnap/data/datasets/webdatasets/HO3D_v3/evaluation/*.tar]' \
  TEST.batch_size=128 \
  AIM.server_url=.
```

### 4.2 DexYCB

```bash
python -m script.test \
  --config-name=stage1-dino_large_no_norm \
  TEST.checkpoint_path=checkpoint/2026-03-08/17-48-07_stage1-dino_large_no_norm_5748b860a5c24552a31bc36a/best_model \
  DATA.test.source='[/mnt/qnap/data/datasets/webdatasets/DexYCB/s1/test/*.tar]' \
  TEST.batch_size=128 \
  AIM.server_url=.
```

### 4.3 InterHand2.6M test

```bash
python -m script.test \
  --config-name=stage1-dino_large_no_norm \
  TEST.checkpoint_path=checkpoint/2026-03-08/17-48-07_stage1-dino_large_no_norm_5748b860a5c24552a31bc36a/best_model \
  DATA.test.source='[/mnt/qnap/data/datasets/webdatasets/InterHand2.6M/test/*.tar]' \
  TEST.batch_size=128 \
  AIM.server_url=.
```

## 5. 示例：使用 `best_model_rte`

如果想测 `RTE best` 的模型，只要把 checkpoint 路径换成：

```text
checkpoint/.../best_model_rte
```

例如：

```bash
python -m script.test \
  --config-name=stage1-dino_large_no_norm \
  TEST.checkpoint_path=checkpoint/2026-03-08/17-48-07_stage1-dino_large_no_norm_5748b860a5c24552a31bc36a/best_model_rte \
  DATA.test.source='[/mnt/qnap/data/datasets/webdatasets/HO3D_v3/evaluation/*.tar]' \
  TEST.batch_size=128 \
  AIM.server_url=.
```

## 6. 如何计算测试指标

`script.test` 只会直接写一个快速版的 `metrics.json`，通常只包含：

- `mpjpe`
- `mpvpe`
- `num_samples`

如果你需要更完整的评估指标，执行：

```bash
python script/evaluate.py \
  checkpoint/<date>/<time>_<config>_<hash>/test_results_<timestamp>/predictions.h5
```

会得到：

- `mpjpe`
- `mpvpe`
- `rel_mpjpe`
- `rel_mpvpe`
- `num_samples`
- `num_valid_joints`
- `num_valid_hands`

## 7. 快速定位最新测试结果目录

如果 `script.test` 刚跑完，你可以这样看：

```bash
ls checkpoint/<date>/<time>_<config>_<hash>/
```

例如：

```bash
ls checkpoint/2026-03-08/17-48-07_stage1-dino_large_no_norm_5748b860a5c24552a31bc36a
```

会看到：

- `best_model/`
- `best_model_rte/`
- `best_model_rel_mpjpe/`
- `test_results_YYYY-MM-DD_HH-MM-SS/`

## 8. 常见问题

### 8.1 远程 AIM 地址报代理错误

如果测试时 AIM 远程地址不可达或被代理拦截，可直接在命令里加：

```bash
AIM.server_url=.
```

这样日志会写到本地，不影响 test 主流程。

### 8.2 多卡测试时报 `DDP` 没有 `predict_full`

这个问题已经在 `script/test.py` 修复。当前版本多卡测试会先 unwrap 模型，再访问 `predict_full()`。

### 8.3 IH26M 多卡测试汇总超时

对于大测试集（尤其 `IH26M`），推荐优先用单卡测试，避免多卡 merge 阶段出现 `all_gather` 体量过大导致超时。
