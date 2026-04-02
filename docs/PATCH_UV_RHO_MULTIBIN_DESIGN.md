# Patch-UV + Rho MultiBin Camera Head 设计说明

本文档概括当前计划中的新 `cam_head` 路径：不再直接在相机空间预测 `x / y / z`，而是先在输入 patch 空间预测 root joint 的 `uv`，再预测沿该视线的极径 `rho`，最后通过显式相机几何恢复根关节的 3D 位置。

## 0. 当前实现状态

当前代码已经完成以下基础实现：

- 新增 `cam_head_type=patch_uv_rho_multibin`
- 新增 `rho` 版先验、编码、解码工具
- 新增 `patch uv + rho -> xyz` 的显式几何恢复路径
- `uv_patch` 已从分解式 1D heatmap 改为 joint 2D heatmap
- `loss` 已接入：
  - `uv_patch` 2D heatmap CE / 回归
  - `rho` cls / residual
  - 最终 `trans_pred` 的直接监督
- 已补充基础单测与 one-step backward smoke test
- 已通过一步 train-flow smoke（`forward -> backward -> step -> vis -> save_state`）

为保持向前兼容，当前实现仍复用原有配置字段：

```yaml
MODEL:
  handec:
    root_z:
      num_bins: ...
      prior_k: ...
      d_min: ...
      d_max: ...
```

也就是说：

- 这些字段在 `patch_uv_rho_multibin` 路径下语义上已经切换为 `rho` multibin 的超参数；
- 但字段名暂时保留 `root_z.*`，以避免同时改动过多训练配置与脚本。
- 当前 `stage1/2-dino_large_no_norm.yaml` 已显式补上：

```yaml
LOSS:
  lambda_uv_patch: 1.0
```

并对主要字符串配置项补充了 “可选值 + 含义” 注释，减少对代码默认值与口头约定的依赖。
- 对 `uv_patch` 2D heatmap 的空间分辨率，当前实现不直接复用 `heatmap_resolution[0:2]` 的全部数值；
  为避免在默认 `224 patch + 512x512 uv hm` 下产生不可接受的参数量，内部采用：

```text
uv_h = min(heatmap_resolution[0], patch_h / 4)
uv_w = min(heatmap_resolution[1], patch_w / 4)
```

当前 `224 x 224` patch 下，默认会落到：

```text
56 x 56
```

## 1. 目标

当前项目在 `no_norm` 主线路径上的 camera head 演进顺序为：

- 旧路径：`softargmax3d`
  - 直接在相机空间对 `x / y / z` 做 heatmap 分类
- 当前新路径：`xy_rootz_multibin`
  - `x / y` 仍在相机空间估计
  - `z` 改为 `root_z prior-centered multibin`

本设计希望进一步改成：

- `uv_patch`：在 patch 空间估计 root joint 的 2D 位置
- `rho`：预测根关节到光心的真实距离
- `xyz_root`：由 `uv + rho + intrinsics` 显式反投影恢复

对应的设计动机有两点：

1. 当前 backbone 看到的是固定大小的裁手 patch，先在 patch 空间估计 2D 位置，比直接估计相机空间 `x / y (mm)` 更符合输入结构；
2. 当 `uv` 已经固定后，根关节在 3D 中的位置天然可以写成 “方向 + 距离” 的形式，其中方向由 `uv` 决定，距离由 `rho` 决定，比直接回归 `x / y / z` 更几何一致。

## 2. 表达形式

### 2.1 Patch 空间 2D

设输入 patch 尺寸为：

```text
patch_size = (H_patch, W_patch)
```

当前 `DINO` 主线下：

```text
H_patch = W_patch = MODEL.img_size = 224
```

root joint 的 patch 空间监督直接使用：

```text
batch["joint_patch_resized"][..., 0, :]
```

即 `preprocess_batch()` 已经生成好的、位于 resized patch 坐标系下的 root 2D。

### 2.2 原图像素坐标恢复

设 patch 空间预测为：

```text
(u_patch, v_patch)
```

设 `patch_bbox = [x1, y1, x2, y2]` 是 patch 在原图中的坐标框，则原图像素坐标为：

```text
u = x1 + u_patch * (x2 - x1) / W_patch
v = y1 + v_patch * (y2 - y1) / H_patch
```

这里必须使用 `patch_bbox`，而不是 `hand_bbox`：

- `patch_bbox` 反映的是实际裁出来送入 backbone 的 patch 区域
- `hand_bbox` 反映的是手本身的 2D 尺度，适合做几何先验

### 2.3 Ray + Rho

设内参为：

```text
focal = (fx, fy)
princpt = (cx, cy)
```

则原图像素点 `(u, v)` 对应的相机方向向量为：

```text
q = [(u - cx) / fx, (v - cy) / fy, 1]
```

其单位方向向量为：

```text
r = q / ||q||
```

若预测的是根关节到光心的真实距离 `rho`，则根关节位置为：

```text
P_root = rho * r
```

这就是本设计中的 `trans_pred`。

## 3. 为什么使用 Rho，而不是继续使用 Root-Z

当前 `root_z` 路径中，`z` 表示的是相机空间的轴向深度：

```text
P = z * q
```

新的 `rho` 路径中，`rho` 表示的是点到光心的欧氏距离：

```text
P = rho * q / ||q||
```

两者的关系是：

```text
rho = z * ||q||
z = rho / ||q||
```

因此：

- `root_z` 和 `rho` 在几何上是等价参数化；
- 但一旦我们先显式预测了 `uv`，把 3D 根关节表示为 “方向 `r` + 距离 `rho`” 更自然；
- `rho` 路径中，`uv` 只负责方向，`rho` 只负责沿该方向的距离，解耦更彻底。

需要特别注意：

- 当 `uv` 固定时，改变 `rho` 不会改变 2D 投影位置；
- 它只会改变 3D 点沿 ray 的位置。

## 4. 新 Head 的整体结构

建议新增一个新的配置类型：

```yaml
MODEL:
  handec:
    cam_head_type: patch_uv_rho_multibin
```

其内部结构为：

1. `uv_patch` head：
   - `SoftargmaxHead2D`
   - 坐标范围是 patch 像素空间
2. `rho` head：
   - `prior-centered delta-log-rho multibin + residual`
3. 几何恢复：
   - `uv_patch -> uv_img -> q -> r`
   - `trans_pred = rho_pred * r`

最终输出接口仍保持：

```text
pred_cam = [x, y, z]
```

以保证下游 `FK / reproj / test / evaluate` 基本不需要改输出协议。

## 5. Rho Prior 的设计

### 5.1 从 Z Prior 到 Rho Prior

当前 `root_z` 路径中已经存在：

```text
z_prior = k * sqrt(fx * fy) / sqrt(bbox_w * bbox_h)
```

它反映的是：

- 手在图像上越小，通常越远；
- 焦距越大，在同样距离下成像越大。

若要把它改写成 `rho` 先验，则需要再引入一个参考 ray：

```text
q_ref = [(u_ref - cx) / fx, (v_ref - cy) / fy, 1]
rho_prior = z_prior * ||q_ref||
```

如果记 `theta_ref` 为 `q_ref` 与光轴的夹角，则：

```text
||q_ref|| = 1 / cos(theta_ref)
rho_prior = z_prior / cos(theta_ref)
```

这一步是一个离轴修正，而不是新的视觉建模。

### 5.2 q_ref 的动机

引入 `q_ref` 的核心原因是：

- `rho` 是沿 ray 的距离，不只是轴向深度；
- 因此 `rho_prior` 不能只看 `bbox` 尺度，还必须知道参考 ray 离光轴有多斜。

如果没有 `q_ref`，直接把原先的 `z_prior` 拿来当 `rho_prior`：

```text
rho_prior ≈ k * sqrt(fx * fy) / sqrt(bbox_w * bbox_h)
```

那么这个先验实际上仍然是一个 `z-like prior`，没有纳入视线方向的尺度修正。

### 5.3 V1 的设计理念

V1 选择：

```text
u_ref, v_ref = hand_bbox center
```

也就是：

```text
q_ref = bbox center ray
```

这样做并不是因为 “bbox center 就是真实 root 所在位置”，而是因为：

1. 它是一个静态、训练/推理一致的参考方向；
2. 它不依赖模型当前预测，不会让 prior 变成 moving target；
3. 它与现有 `hand_bbox`-based size prior 逻辑完全兼容；
4. 第一版实现代价最低，最稳。

V1 的理念是：

- 先用一个稳定的参考 ray，把 `z_prior` 转成 `rho_prior`；
- 再由 `Δlog rho` 去学习真实 root ray 与 `q_ref` 之间的剩余偏差。

## 6. 来自 Full Records 的数据支撑

本节统计基于：

- `tests/temp_root_z_prior_records_stage1_full/`

其运行条件在：

- `seed = 123`
- `effective_shardshuffle = false`
- `effective_post_clip_shuffle = 0`

对应文件：

- `tests/temp_root_z_prior_records_stage1_full/stage1-dino_large_no_norm.run_info.json`

有效样本数：

```text
706337
```

### 6.1 bbox center ray 与真实 root ray 的差异

基于 records 中保存的：

- `hand_bbox`
- `focal / princpt`
- `joint_img(root)`

离线计算得到：

```text
angle(q_ref, q_gt):
  p50 = 3.33°
  p95 = 8.71°
  p99 = 15.86°
```

更关键的是 `rho` 先验真正关心的尺度因子：

```text
||q_gt|| / ||q_ref||:
  p50 = 1.0007
  p95 = 1.0120
  p99 = 1.0748
```

这说明：

- `bbox center ray` 不是 root ray 本身；
- 但它对应的 ray-length 修正项与真实 root ray 在大多数样本上非常接近；
- 因此它适合作为 V1 的静态先验参考方向。

### 6.2 使用 q_ref 是否有意义

比较三种量：

1. 原始 `z_prior / z_gt`
2. 做了 `q_ref` 修正后的 `rho_prior_ref / rho_gt`
3. 不加 `q_ref`、直接把 `z_prior` 当成 `rho_prior` 的 `rho_prior_noq / rho_gt`

全局结果：

```text
z_prior / z_gt:
  p50 = 1.0011
  p95 = 1.5048
  p99 = 1.8769

rho_prior_ref / rho_gt:
  p50 = 0.9996
  p95 = 1.5018
  p99 = 1.8362

rho_prior_noq / rho_gt:
  p50 = 0.9864
  p95 = 1.4812
  p99 = 1.7939
```

更直接地看目标空间：

```text
Δlog rho_ref:
  p50 = 0.0004
  p95 = 0.4116
  p99 = 0.6481

Δlog rho_noq:
  p50 = 0.0137
  p95 = 0.4419
  p99 = 0.7391
```

解释：

- 不加 `q_ref` 时，`rho_prior` 有可见的系统偏差；
- 加上 `bbox center q_ref` 后，`Δlog rho` 重新被拉回 0 中心，并且高尾更短；
- 这正是 `q_ref` 的实证动机。

## 7. 基于 Full Records 的 V1 超参数推荐

### 7.1 统计定义

基于 V1 定义：

```text
rho_gt = root_z_gt * ||q_gt||
rho_prior = k * sqrt(fx*fy) / sqrt(bbox_w*bbox_h) * ||q_ref||
Δlog rho = log(rho_gt) - log(rho_prior)
```

从 full records 离线重统计得到：

```text
k_p50 = 121.044693
Δlog rho_p01 = -0.608065
Δlog rho_p99 = 0.647695
```

按原先 root-z 路线中相同的 `±0.1 margin` 规则，推荐：

```text
d_min = -0.708065
d_max =  0.747695
```

### 7.2 推荐配置

工程上建议写成：

```yaml
MODEL:
  handec:
    root_z:
      num_bins: 8
      prior_k: 121.0
      d_min: -0.71
      d_max: 0.75
```

也就是说：

- `k` 与当前 `root_z` 路径几乎完全一致，可继续沿用 `121.0`
- `d_min / d_max` 与当前 `[-0.73, 0.74]` 很接近
- 但对 `rho` 路径而言，`d_max` 略放宽到 `0.75` 更贴近这次统计

## 8. Loss 设计建议

### 8.1 Root 2D

`uv_patch` 分支建议使用 2D heatmap CE：

```text
loss_uv_patch =
  CE(log_hm_u_patch, u_patch_gt)
  +
  CE(log_hm_v_patch, v_patch_gt)
```

监督源：

```text
batch["joint_patch_resized"][:, :, 0]
```

### 8.2 Rho

`rho` 分支沿用当前 `root_z multibin` 形式：

```text
loss_rho_cls
loss_rho_res
```

目标：

```text
rho_gt = ||joint_cam_root_gt||_2
Δlog rho = log(rho_gt) - log(rho_prior)
```

### 8.3 最终 3D Root

建议保留最终 `trans_pred` 的直接监督，作为稳定项：

```text
loss_trans = L1(trans_pred, trans_gt)
```

其中：

```text
trans_pred = rho_pred * q_pred / ||q_pred||
trans_gt = batch["joint_cam"][:, :, 0]
```

这样可以避免第一版完全依赖 `uv/rho` 两个中间变量的 supervision。

## 9. 与当前代码接口的对应关系

### 9.1 现有可复用部分

以下内容可以直接复用：

- `multibin cls + residual` 逻辑
- `num_bins`
- 现有 `RootZMultiBinHead` 的总体结构
- `BundleLoss2` 中对应的 `cls/res` loss 组织方式
- `hand_bbox` 作为 size prior 的输入来源

### 9.2 需要改动的部分

需要新增或替换的核心逻辑：

1. `module.py`
   - 新增 `patch_uv_rho_multibin` 类型
   - `uv_patch` 头输出 patch 坐标，而不是相机空间 `x/y`
   - `rho` 头输出 `rho` 而不是 `z`

2. `root_z.py`
   - 建议新增 `rho` 版编码/解码工具，而不是复用 `root_z` 语义命名
   - 建议新增：
     - `compute_rho_prior_and_geom()`
     - `encode_delta_log_rho_targets()`
     - `decode_delta_log_rho_predictions()`

3. `loss.py`
   - `xy_rootz_multibin` 分支不能直接复用
   - 需要新增：
     - `uv_patch heatmap CE`
     - `rho cls/res`
     - `trans_pred = backproject(uv_pred, rho_pred)` 的监督路径

4. `net.py`
   - `predict_full()` 中也要走新的 `uv + rho -> xyz` 几何恢复

## 10. 实现顺序建议

建议按下面顺序实现：

1. 新增 `rho` 版几何工具与编码/解码函数
2. 新增新的 `cam_head_type`
3. 在 `decode_token()` 中完成：
   - `uv_patch -> uv_img`
   - `uv_img + intrinsics -> q_pred`
   - `rho_pred + q_pred -> trans_pred`
4. 在 `loss.py` 中接入：
   - `uv_patch` heatmap CE
   - `rho cls/res`
   - `trans_pred` 直接监督
5. 新增/补齐测试

## 11. 测试建议

第一批建议至少补以下测试：

1. `rho_prior` 几何单测
   - 验证 `rho_prior = z_prior * ||q_ref||`
2. `uv_patch -> uv_img` 映射单测
   - 与 `joint_patch_resized` / `patch_bbox` 的定义一致
3. `rho encode/decode` roundtrip 单测
4. `cam_head` 输出 shape 单测
5. `one-step smoke` 测试
   - 基于 `stage1-dino_large_no_norm`
6. `full train flow smoke` 测试
   - 验证 checkpoint / loss state / forward 不崩

## 12. 当前结论

当前可以确认的 V1 设计结论：

1. 新路线应定义为 `patch uv + rho multibin + explicit backprojection`
2. `uv_patch` 使用 2D heatmap CE 监督
3. `rho_prior` 使用：

```text
rho_prior = z_prior * ||q_ref||
```

且 V1 中：

```text
q_ref = bbox center ray
```

4. 基于 full records 的推荐超参数为：

```text
prior_k = 121.0
d_min = -0.71
d_max = 0.75
num_bins = 8
```

5. 这一路径已经具备进入实现阶段的条件。
