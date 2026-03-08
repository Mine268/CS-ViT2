# depth-bin repack 方案

本文档说明如何对已经生成完成的 depth-bin 数据进行无解码透传式 repack。

## 1. 目标

对每个 dataset 的每个深度桶目录单独 repack：

- 不跨 dataset
- 不跨 split
- 不跨 `nf*_s*`
- 不跨 `bin_*`

也就是说，只在同一个叶子 bin 目录内部重新切 tar。

## 2. 脚本位置

- `preprocess/repack_depth_bin_wds.py`

## 3. 默认行为

脚本默认会：

1. 扫描整个 depth-bin 根目录
2. 把 repack 结果写到一个新的临时根目录（默认 `<root>_repacked_tmp`）
3. 在 repack 成功后：
   - 先把旧目录改名为 backup
   - 再把新目录改回原始根目录名称
   - 最后删除旧 backup

也就是：

```text
旧目录 -> 备份目录
新目录 -> 正式目录
删除旧备份
```

## 4. 使用示例

```bash
source .venv/bin/activate

python preprocess/repack_depth_bin_wds.py \
  --root /mnt/qnap/data/datasets/webdatasets/depth-bins \
  --maxsize $((1536 * 1024 * 1024))
```

只处理指定 dataset：

```bash
python preprocess/repack_depth_bin_wds.py \
  --root /mnt/qnap/data/datasets/webdatasets/depth-bins \
  --datasets InterHand2.6M HO3D_v3
```

只生成新目录，不执行替换：

```bash
python preprocess/repack_depth_bin_wds.py \
  --root /mnt/qnap/data/datasets/webdatasets/depth-bins \
  --no-swap
```

保留旧备份：

```bash
python preprocess/repack_depth_bin_wds.py \
  --root /mnt/qnap/data/datasets/webdatasets/depth-bins \
  --keep-backup
```

## 5. 输出内容

在 repack 后的新根目录里：

- 保留原有目录结构
- 每个 `clip_dir` 下保留原 `summary.json`
- 新增 `repack_stats.json`
- 根目录新增 `repack_summary.json`

## 6. 关键实现点

- 使用 `WebDataset(...)` 直接读取 sample bytes
- 不调用 `.decode()`
- 用 `ShardWriter(maxsize=...)` 重新切 tar
- 不改变 sample 内容，只改变 shard 划分

## 7. 验证状态

- 最小单元测试：`tests/test_repack_depth_bin_wds.py`
- 覆盖：
  - 构造假 depth-bin 根目录
  - repack 到新目录
  - 校验 sample 数一致
  - 执行最终 swap
