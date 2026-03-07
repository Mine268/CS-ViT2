# 数据加载采样策略说明

本文档说明训练/验证/测试三种阶段的数据采样策略，以及为什么训练不再枚举一个序列的全部滑窗。

## 1. 结论速览

- `train`：使用 `random_clip`
- `val/test`：使用 `dense`
- 目标：提升 batch 内样本随机性，减少连续相邻帧扎堆，同时避免单纯增大 shuffle buffer 带来的内存压力

## 2. 背景问题

原先 `clip_to_t_frames()` 会把一个原始序列的所有滑窗连续展开：

```text
seq0 -> clip0, clip1, clip2, clip3, ...
seq1 -> clip0, clip1, clip2, clip3, ...
```

即使后面再做小 buffer shuffle，也很容易让一个 batch 内出现：

- 同一段视频的相邻帧
- 同一场景/手型/object 的近重复样本
- 不同数据集之间的混合不充分

## 3. 新策略

### 3.1 训练：`random_clip`

对每个原始序列，只随机采样少量 clip：

- 默认 `clips_per_sequence=1`
- Stage 1 推荐 `post_clip_shuffle=512`
- Stage 2 推荐 `post_clip_shuffle=256`
- 同时增加 `shardshuffle=32`，提升 tar 级别随机性

这样做的好处：

- 一个 batch 会覆盖更多不同序列
- 相邻帧重复显著减少
- 不需要把 shuffle buffer 提得特别大
- CPU 内存压力更可控

### 3.2 验证/测试：`dense`

验证和测试继续枚举所有滑窗：

- 结果稳定
- 指标可复现
- 与旧结果保持可比性

默认关闭 `post_clip_shuffle`，避免评估顺序被无意义打乱。

## 4. 配置示例

```yaml
DATA:
  train:
    sampling:
      mode: random_clip
      clips_per_sequence: 1
      shardshuffle: 32
      post_clip_shuffle: 512

  val:
    sampling:
      mode: dense
      shardshuffle: false
      post_clip_shuffle: 0

  test:
    sampling:
      mode: dense
      shardshuffle: false
      post_clip_shuffle: 0
```

## 5. 适用建议

### Stage 1

- 推荐：`clips_per_sequence=1`
- 如果后续想增加序列内多样性，可尝试 `2`

### Stage 2

- 推荐：`clips_per_sequence=1`
- 因为一个 clip 本身已经包含多帧时序信息，不宜再一次性从同序列取太多 clip

## 6. 注意事项

- 该策略主要影响 `train` 的 batch 组成，不改变 `val/test` 指标定义
- 若训练覆盖率担心下降，可适当增加总训练步数
- 新策略的重点是“减少每个序列一次性产出的样本数”，而不是仅靠更大的 shuffle buffer 硬打散

## 7. 相关文件

- `src/data/dataloader.py`
- `script/train.py`
- `script/test.py`
- `config/stage1-dino_large.yaml`
- `config/stage1-dino_large_no_norm.yaml`
- `config/stage2-dino_large.yaml`
- `config/stage2-dino_large_no_norm.yaml`
