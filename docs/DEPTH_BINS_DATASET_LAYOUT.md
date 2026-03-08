# depth-bins 数据目录结构说明

本文档同步自 `/mnt/qnap/data/datasets/webdatasets/depth-bins/README`，用于说明静态深度分桶数据的目录层级、文件含义和 sample 字段，方便在项目仓库内直接查阅。

depth-bins 目录说明
===================

这里存放“按静态深度区间分桶”的 WebDataset 数据。
核心思路：
- 从现有 WebDataset 序列样本出发
- 先离线打散成固定长度 clip
- 再按 clip 最后一帧 root joint 的 Z 深度分桶
- 不同深度桶分别写入不同 tar 目录

当前目录层级含义
----------------

目录结构示例：

  /mnt/qnap/data/datasets/webdatasets/depth-bins/
  └── InterHand2.6M_smoke/
      └── train/
          └── nf1_s1/
              ├── bin_1100_inf/
              │   └── 000000.tar
              └── summary.json

各级目录含义：

1. 第一层：数据集名
   - 例如：`InterHand2.6M_smoke`、`HO3D_v3_smoke`
   - 表示这是哪个数据集转换出来的 depth-bin 版本
   - 其中带 `_smoke` 的是小量测试数据，不是完整训练集

2. 第二层：split
   - 例如：`train`
   - 表示训练/验证/测试划分

3. 第三层：clip 配置目录
   - 例如：`nf1_s1`
   - `nf1` = `num_frames=1`
   - `s1` = `stride=1`
   - 如果以后生成时序版本，可能会看到像 `nf7_s1` 这样的目录

4. 第四层：深度桶目录
   - 例如：`bin_0000_0500`、`bin_0500_0700`、`bin_1100_inf`
   - 表示该目录中的所有 clip 都属于这个深度区间
   - 区间依据是：clip 最后一帧 root joint 的 Z 深度（单位 mm）

深度桶命名规则
--------------

- `bin_0000_0500`：表示 0mm <= depth < 500mm
- `bin_0500_0700`：表示 500mm <= depth < 700mm
- `bin_0700_0900`：表示 700mm <= depth < 900mm
- `bin_0900_1100`：表示 900mm <= depth < 1100mm
- `bin_1100_inf`：表示 depth >= 1100mm

说明：
- 边界由转换脚本中的 `--bin-edges` 决定
- 当前默认边界通常为：`0 500 700 900 1100 1000000`

文件含义
--------

1. `000000.tar`, `000001.tar`, ...
   - 每个 tar 是该深度桶下的一部分样本
   - 同一个深度桶样本太多时，会自动切分成多个 tar

2. `summary.json`
   - 记录该数据集/split/clip 配置下的转换统计信息
   - 典型字段：
     - `dataset_name`: 数据集名
     - `split`: train/val/test
     - `num_frames`: clip 长度
     - `stride`: clip 步长
     - `bin_edges`: 分桶边界
     - `raw_samples`: 原始序列 sample 数量
     - `clips`: 展开后的 clip 总数
     - `bin_counts`: 每个深度桶里的 clip 数量

一个 depth-bin sample 里有什么
-----------------------------

每个写入 depth-bin tar 的 sample，仍然保留训练所需的核心字段，例如：
- `img_bytes.pickle`
- `imgs_path.json`
- `joint_cam.npy`
- `joint_rel.npy`
- `joint_valid.npy`
- `joint_img.npy`
- `hand_bbox.npy`
- `focal.npy`
- `princpt.npy`
- `mano_pose.npy`
- `mano_shape.npy`
- `mano_valid.npy`
- `timestamp.npy`
- `handedness.json`

另外新增两个字段：
- `depth_bin_id.npy`
  - 当前 clip 所属深度桶的整数编号
- `root_depth_last.npy`
  - 当前 clip 最后一帧 root joint 的 Z 深度（mm）

为什么要这样组织
----------------

这样组织后，可以在训练时：
- 先收集不同深度桶对应的 tar
- 再按深度桶均匀混采
- 从而减少训练分布被某些主导深度区间“带偏”的问题

当前状态
--------

目前目录下已有：
- `InterHand2.6M_smoke`
- `HO3D_v3_smoke`

它们是用于验证“离线分桶 -> 新 dataloader -> train.py 接入”链路是否正常的小量 smoke 数据。
完整数据集转换后，也会沿用相同目录结构。



说明：当前 depth-bin 转换脚本默认 `maxsize=1.5GB`，用于减少 NAS 场景下小 tar 过多导致的读取开销。
