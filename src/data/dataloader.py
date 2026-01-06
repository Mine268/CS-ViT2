"""
Webdataset Dataloader
"""
from typing import *
from functools import partial
import numpy as np
import webdataset as wds

import torch
from torch.utils.data import DataLoader
import torchvision

import kornia.geometry.transform as T

from .preprocess import preprocess_batch

NPY_KEYS = [
    "hand_bbox.npy",
    "joint_img.npy",
    "joint_hand_bbox.npy",
    "joint_cam.npy",
    "joint_rel.npy",
    "joint_valid.npy",
    "mano_pose.npy",
    "mano_shape.npy",
    "mano_valid.npy",
    "timestamp.npy",
    "focal.npy",
    "princpt.npy",
]

COLLATE_LIST_KEYS = [
    "imgs", "handedness", "__key__"
]

def clip_to_t_frames(num_frames, stride, source):
    """
    将序列样本拆分为小片小片的连续样本
    """
    for sample in source:
        imgs_path = sample["imgs_path.json"]
        img_list = sample["img_bytes.pickle"]
        handedness = sample["handedness.json"] # "l" or "r"
        total_frames = len(img_list)
        if total_frames < num_frames:
            continue

        total_samples = (total_frames - num_frames) // stride + 1

        for i in range(total_samples):
            start = i * stride
            end = start + num_frames

            # 构建输出样本 (包含 T 帧数据)
            sub_sample = {
                # 构造唯一的 Key: 原Key_切片序号
                "__key__": f"{sample['__key__']}_{i:04d}",
                "handedness": handedness,
                # 记录图片地址
                "imgs_path": imgs_path[start:end],
                # 图片字节流：直接切片 List
                "imgs_bytes": img_list[start:end],
            }
            for key in NPY_KEYS:
                if key in sample:
                    out_key = key.replace(".npy", "") # 去后缀
                    # 这里执行的是 Numpy 的第一维切片操作
                    sub_sample[out_key] = sample[key][start:end]

            yield sub_sample

def preprocess_frame(sample):
    """将图像二进制流转换为图片"""
    # 1. 图片解码: Bytes (WebP) -> PIL -> Tensor
    # Writer 中使用的是 cv2.imencode(".webp")，这里用 PIL 打开兼容性很好
    imgs_tensor = []
    for img_bytes in sample["imgs_bytes"]:
        buffer_np = np.frombuffer(img_bytes, dtype=np.uint8).copy()
        buffer = torch.from_numpy(buffer_np)
        img = torchvision.io.decode_webp(buffer)
        imgs_tensor.append(img)
    imgs_tensor = torch.stack(imgs_tensor)

    # 2. 处理其他 Numpy 字段
    result = {
        "imgs_path": sample["imgs_path"],
        "imgs": imgs_tensor,
        "handedness": sample["handedness"], # 此时还是 str, collate 时可能需要特殊处理或 drop
    }

    # 自动将所有 numpy 字段转为 Tensor
    for key in sample:
        if key not in ["__key__", "imgs_path", "imgs_bytes", "handedness"]:
            # 确保是 float32 (根据你的 writer 逻辑，大部分已经是 float32)
            val = sample[key]
            if isinstance(val, np.ndarray):
                result[key] = torch.from_numpy(val).float()
            else:
                result[key] = torch.tensor(val)

    return result

def collate_fn(batch_wds):
    """对batch数据进行重整，由于图像大小不一，特判使用List"""
    batch_filter = [b for b in batch_wds if b is not None]
    if len(batch_filter) == 0:
        return {}

    collated = {}
    for key in batch_filter[0].keys():
        if key in COLLATE_LIST_KEYS:
            collated[key] = [sample[key] for sample in batch_filter]
        else:
            values = [sample[key] for sample in batch_filter]
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            else:
                collated[key] = values

    return collated


def get_dataloader(
    url,
    num_frames: int,
    stride: int,
    batch_size: int,
    num_workers: int,
    prefetcher_factor: int,
    infinite: bool = True,
) -> wds.WebDataset:
    """获得wds数据加载器"""
    dataset = (
        wds.WebDataset(
            url,
            resampled=infinite,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
        )
        .shuffle(1000)
        .decode()
        .compose(partial(clip_to_t_frames, num_frames, stride))
        .shuffle(5000)
        .map(preprocess_frame)
        .batched(batch_size, partial=False, collation_fn=collate_fn)
        # .with_epoch(10000)
    )

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=prefetcher_factor if num_workers > 0 else None,
        pin_memory=True
    )

    return dataloader
