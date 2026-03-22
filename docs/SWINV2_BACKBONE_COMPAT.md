# SwinV2 Backbone 兼容说明

本文件概括当前项目对 `model/microsoft/swinv2-large-patch4-window12-192-22k` 的兼容方式和配置要点。

## 当前兼容范围

- 已支持 `SwinV2Model` 作为 Stage 1 backbone 初始化与前向。
- 已适配 `SwinV2` 无 `cls token` 的输出格式。
- 已适配 `SwinV2` 最终 stage token grid，而不是 ViT 风格的原始 patch grid。

## 关键差异

`SwinV2` 与 `DINOv2 / MAE` 的核心差异：

- 没有 `cls token`
- `last_hidden_state` 是最终 stage 的 token
- 对 `swinv2-large-patch4-window12-192-22k` 且 `img_size=192` 时，最终 token grid 是 `6 x 6`
- `hidden_size = 1536`

因此：

- `backbone.drop_cls` 对 SwinV2 实际无效
- `num_patch` 不能再用 `img_size / patch_size`
- `PIE` 的 `num_token` 也必须跟随最终 stage token 数

## 当前配置

推荐使用：

- [stage1-swinv2_large.yaml](/data_1/renkaiwen/CS-ViT2/config/stage1-swinv2_large.yaml)

主要设置：

- `MODEL.img_size=192`
- `MODEL.backbone.backbone_str=model/microsoft/swinv2-large-patch4-window12-192-22k`
- `MODEL.backbone.infusion_layer=null`
- `MODEL.handec.context_dim=1536`
- `MODEL.handec.num_head=24`
- `MODEL.temporal_encoder.num_head=24`

## infusion_layer 限制

当前 `SwinV2` 最稳妥的用法是：

- `MODEL.backbone.infusion_layer=null`

原因：

- `SwinV2` 的不同 stage token 数不同，例如 `48x48 -> 24x24 -> 12x12 -> 6x6`
- 现有项目的多层融合逻辑默认各层 token grid 一致

如果强行给 `infusion_layer` 传入多分辨率 stage：

- 代码会报清晰错误，而不是静默产生错位特征

## 使用建议

- 首先用 `stage1-swinv2_large.yaml` 做单卡初始化 smoke
- 再根据显存调整 `TRAIN.sample_per_device`
- 如果后续要做 SwinV2 的多层融合，需要单独设计跨 stage 对齐逻辑
