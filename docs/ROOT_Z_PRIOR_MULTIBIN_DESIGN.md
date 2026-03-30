# Root-Z Prior-Centered MultiBin 设计说明

本文档概括当前项目计划中的 `root_z` 新路径设计：保持现有 `x/y` 估计路径不变，仅将 `z` 从原有 `softargmax3d` 中拆出，改为 `prior-centered delta-log-z multibin + residual`。

## 1. 目标

当前项目的 `trans` 估计仍沿用统一的 `softargmax3d` 相机头：

- `x / y / z` 共用一套 `SoftargmaxHead3D`
- `z` 在**线性毫米空间**上做 heatmap 分类

在当前 9 数据集混训、且已启用 `dataset-level reweight` 的前提下，`root_z` 的统计问题主要有：

1. 绝对深度范围很宽，线性毫米空间对远近距离的误差语义不一致；
2. 不同数据集的 `root_z` 呈多峰分布；
3. 仅依赖统一的连续/线性 `z` 表达，不利于稳定建模远距与近距的混合分布。

因此本设计的目标是：

- `x / y` 路径保持不动；
- 仅替换 `z` 的估计方式；
- 与现有 `softargmax3d` 路径通过配置兼容共存；
- 新路径优先支持 `norm_by_hand=false` 的 `*_no_norm` 配置。

## 1.1 当前实现状态

当前代码已经完成以下实现：

- 新增 `cam_head_type` 配置开关：
  - `softargmax3d`
  - `xy_rootz_multibin`
- 新增 `RootZMultiBinHead`
- 新增 `z_prior + Δlog z` 的共享编码/解码工具
- `loss` 已支持根据 `cam_head_type` 自动切换：
  - 旧路径：`softargmax3d`
  - 新路径：`x/y heatmap + root_z cls/res`

为保证训练配置回退兼容性，当前三份 `*_no_norm` 配置仍默认设置为：

```yaml
MODEL:
  handec:
    cam_head_type: softargmax3d
```

也就是说：

- 新路径已经可用；
- 但默认行为仍与旧训练配置一致；
- 需要显式将 `cam_head_type` 改为 `xy_rootz_multibin` 才会启用新头。

## 2. 设计概览

新路径不直接估计绝对 `z`，而是：

1. 先由 `bbox + focal` 计算一个粗略的几何先验 `z_prior`
2. 再让网络预测：
   - `Δlog z` 的 coarse bin
   - 以及该 bin 内的 residual
3. 最终恢复绝对深度：

```text
log z = log z_prior + Δlog z
z = z_prior * exp(Δlog z)
```

因此，新头学习的是：

```text
Δlog z = log(z_gt) - log(z_prior)
```

而不是直接学习：

```text
log(z_gt)
```

## 3. 为什么使用 prior-centered Δlog z

如果直接使用绝对 `log z`：

- 不同数据集的远近距离峰仍会直接叠加到同一目标空间；
- `MTC` 等远距离数据会让输入几何量出现更长的负尾；
- 仍然需要 head 自己同时学：
  - 基本 pinhole 几何
  - dataset 间尺度偏移
  - 多峰分布的分类与细化

引入 `z_prior` 后：

- 最粗的尺度趋势先由显式几何建模给出；
- head 只需要学“相对先验偏深还是偏浅，以及偏多少”；
- `Δlog z` 的分布通常更集中，更适合做固定 bins 的分类 + residual。

这和 RootNet 一类方法中的 “prior * correction” 思路一致，只是这里改写到了 `log-space residual`。

## 4. Root-Z 头的输入

### 4.1 视觉特征

- `f_vis = token_out`

即当前 `MANOTransformerDecoderHead` 解码前的 token 特征。

### 4.2 几何特征

首版推荐的几何输入不直接使用一组 raw ratio，而是只保留少量、数值更稳定的特征：

```text
f_geo = [
  log(z_prior),
  dx,
  dy,
  dw,
  dh,
  log_aspect_ratio,
]
```

其中：

```text
dx = (bbox_cx - princpt_x) / fx
dy = (bbox_cy - princpt_y) / fy
dw = bbox_w / fx
dh = bbox_h / fy
log_aspect_ratio = log(bbox_w / bbox_h)
```

设计选择：

- 使用 `preprocess_batch` 之后的 `hand_bbox / focal / princpt`
- 不在第一版加入 `data_source embedding`
- 不将 `log(bbox_w / fx)`、`log(bbox_h / fy)` 直接作为主输入
- 但保留 `dw / dh` 这两个正值尺度比特征

原因：

- `log(bbox/focal)` 虽然有深度信息，但在远距离数据集上可能出现更长的负尾；
- 直接使用 `dw / dh` 不会引入额外的长负尾；
- 用 `z_prior` 吸收主尺度趋势，再让网络用 `dw / dh` 做细化补充更合理。

## 5. z_prior 的定义

使用一个 camera-aware size prior：

```text
f_eff = sqrt(fx * fy)
s_eff = sqrt(bbox_w * bbox_h)
z_prior = k * f_eff / s_eff
```

其中：

- `k` 是从训练统计中估计出来的全局尺度常数

取 log 后得到：

```text
log z_prior = log k + log f_eff - log s_eff
```

## 6. MultiBin 表达

设：

- `t = Δlog z`
- `K` = bin 数
- `d_min`, `d_max` = `Δlog z` 的支持区间
- `Δ = (d_max - d_min) / K`

则训练时：

```text
b = clip(floor((t - d_min) / Δ), 0, K-1)
c_b = d_min + (b + 0.5) * Δ
r = (t - c_b) / Δ
```

其中：

- `b` 是 bin id
- `r` 是 bin 内 residual，理论上落在 `[-0.5, 0.5]`

推理时：

```text
b_hat = argmax(z_cls_logits)
r_hat = clamp(z_residuals[b_hat], -0.5, 0.5)
Δlogz_hat = c_{b_hat} + r_hat * Δ
logz_hat = logz_prior + Δlogz_hat
z_hat = exp(logz_hat)
```

## 7. 与现有 softmax3d 的兼容设计

建议通过配置切换 camera head 类型：

```yaml
MODEL:
  handec:
    cam_head_type: softargmax3d        # 旧路径
    # or
    cam_head_type: xy_rootz_multibin   # 新路径
```

统一输出接口仍保持：

- `pred_cam: [B, 3]`

差别在于内部实现：

### 旧路径：`softargmax3d`

- `x / y / z` 全部来自 `SoftargmaxHead3D`

### 新路径：`xy_rootz_multibin`

- `x / y` 来自 `SoftargmaxHead2D`
- `z` 来自 `RootZMultiBinHead`
- 最终拼成：

```text
pred_cam = [pred_x, pred_y, pred_z]
```

这样可以保证：

- 下游 `FK / reproj / test / evaluate` 接口基本不变；
- 仅在 head 和 loss 处做分支。

## 8. 首版配置建议

### 8.1 结构开关

```yaml
MODEL:
  handec:
    cam_head_type: xy_rootz_multibin
    root_z:
      num_bins: 8
      z_prior_type: bbox_focal_sqrt
      feature_mode: token_geom
      geom_dim: 6
      geom_hidden_dim: 256
      use_data_source_embed: false
```

### 8.2 Loss

```yaml
LOSS:
  lambda_trans: 1.0          # 仅对应 x/y
  lambda_root_z_cls: 1.0
  lambda_root_z_res: 1.0
  root_z_bin_weight: uniform
```

当前建议第一版使用：

- `uniform` bin weight

## 9. 超参数统计与当前推荐值

统计来源：

- [tests/temp_root_z_prior_stats_stage1/stage1-dino_large_no_norm.json](/data_1/renkaiwen/CS-ViT2/tests/temp_root_z_prior_stats_stage1/stage1-dino_large_no_norm.json)

该统计是基于：

- [stage1-dino_large_no_norm.yaml](/data_1/renkaiwen/CS-ViT2/config/stage1-dino_large_no_norm.yaml)
- 当前 `reweight` 数据集集合
- `random_clip`
- `clips_per_sequence=1`
- 全量 finite 扫描完成后得到

统计结果：

- `collected_valid_samples = 706,309`
- `k_median = 120.900861`
- `delta_log_z_p01 = -0.628655`
- `delta_log_z_p99 = 0.641814`

当前建议的首版配置值：

```text
k = 121.0
d_min = -0.73
d_max = 0.74
K = 8
```

若按 `+0.1` margin 规则，理论来源为：

```text
d_min = p01 - 0.1 = -0.728655
d_max = p99 + 0.1 =  0.741814
```

## 10. 当前统计的解释边界

要注意，这里的统计并不是：

- 所有可能 `nf1_s1` clip 的“自然全集统计”

而是：

- 当前训练配置实际看到的 `random_clip` 采样语义下的统计

因此它更适合回答：

- “在当前训练路径里，prior-centered `Δlog z` 的支持区间应该设多少”

而不适合回答：

- “原始数据全集的绝对 clip 分布是什么”

## 11. 当前观察到的问题

尽管全局 residual 分布已经比较集中：

- `delta_p50 = 0.0`
- `delta_p01 ≈ -0.63`
- `delta_p99 ≈ 0.64`

但一些数据集仍有更长尾：

- `AssemblyHands`: `delta_p01 = -2.470266`
- `RHD`: `delta_p99 = 1.104852`

这说明：

1. `z_prior` 已经能较好中心化主流数据集；
2. 但对少数 domain，仍存在明显 dataset-specific tail；
3. 第一版先用全局 `k / d_min / d_max` 依然是合理的；
4. 若后续需要进一步提升稳健性，可再分析：
   - 是否对 tail 做 clamp
   - 是否引入 dataset-aware 校正

## 12. 实现边界

首版建议：

- 仅在 `norm_by_hand=false` 的 `*_no_norm` 配置上开放；
- 仅替换 `z` 路径；
- 不同时重构 `root_xy`；
- 不引入 adaptive bins；
- 不引入 dataset embedding；
- 日志新增：
  - `loss_trans_xy`
  - `loss_root_z_cls`
  - `loss_root_z_res`
  - `root_z_bin_acc`
  - `root_z_mae_mm`

## 13. 当前推荐结论

如果现在要实现第一版，建议直接按下面这套走：

1. `x / y` 保持原有 `softargmax2d` 风格路径不变
2. `z` 改为 `prior-centered Δlog z multibin + residual`
3. 输入用：
   - `token_out`
   - `log z_prior, dx, dy, dw, dh, log_aspect_ratio`
4. 超参数用：
   - `k = 121.0`
   - `d_min = -0.73`
   - `d_max = 0.74`
   - `K = 8`
5. bin weight 先用 `uniform`

这是一套与当前代码结构、当前数据重权重配置以及已完成统计都相互一致的首版方案。
