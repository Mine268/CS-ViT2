"""
Webdataset Dataloader
"""
from typing import *
from functools import partial
import numpy as np
import webdataset as wds

import torch
from torch.utils.data import DataLoader, get_worker_info
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

def _build_sub_sample(sample, start: int, end: int, clip_idx: int):
    imgs_path = sample["imgs_path.json"]
    img_list = sample["img_bytes.pickle"]
    handedness = sample["handedness.json"]

    sub_sample = {
        "__key__": f"{sample['__key__']}_{clip_idx:04d}",
        "handedness": handedness,
        "imgs_path": imgs_path[start:end],
        "imgs_bytes": img_list[start:end],
    }
    for key in NPY_KEYS:
        if key in sample:
            out_key = key.replace(".npy", "")
            sub_sample[out_key] = sample[key][start:end].copy()

    return sub_sample


def _select_clip_indices(
    total_frames: int,
    num_frames: int,
    stride: int,
    sampling_mode: str,
    clips_per_sequence: Optional[int],
    rng: np.random.Generator,
) -> List[int]:
    total_clips = (total_frames - num_frames) // stride + 1
    if total_clips <= 0:
        return []

    if sampling_mode == "dense":
        return list(range(total_clips))

    if sampling_mode != "random_clip":
        raise ValueError(f"Unsupported clip sampling mode: {sampling_mode}")

    if clips_per_sequence is None:
        clips_per_sequence = 1
    if clips_per_sequence <= 0:
        raise ValueError(
            f"clips_per_sequence must be positive for random_clip, got {clips_per_sequence}"
        )

    num_select = min(clips_per_sequence, total_clips)
    selected = rng.choice(total_clips, size=num_select, replace=False)
    return selected.tolist()


def clip_to_t_frames(
    num_frames,
    stride,
    source,
    sampling_mode: str = "dense",
    clips_per_sequence: Optional[int] = None,
    seed: Optional[int] = None,
):
    """
    将序列样本拆分为小片小片的连续样本

    sampling_mode:
        - dense: 枚举该序列的全部连续窗口，适合 val/test
        - random_clip: 每个原始序列只随机采样少量 clip，适合 train
    """
    worker_info = get_worker_info()
    worker_id = worker_info.id if worker_info is not None else 0
    rng_seed = None if seed is None else seed + worker_id
    rng = np.random.default_rng(rng_seed)

    for sample in source:
        img_list = sample["img_bytes.pickle"]
        total_frames = len(img_list)
        if total_frames < num_frames:
            continue

        clip_indices = _select_clip_indices(
            total_frames=total_frames,
            num_frames=num_frames,
            stride=stride,
            sampling_mode=sampling_mode,
            clips_per_sequence=clips_per_sequence,
            rng=rng,
        )

        for clip_idx in clip_indices:
            start = int(clip_idx) * stride
            end = start + num_frames
            yield _build_sub_sample(sample, start, end, int(clip_idx))

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
    seed: Optional[int] = None,
    clip_sampling_mode: str = "dense",
    clips_per_sequence: Optional[int] = None,
    shardshuffle: Union[bool, int] = False,
    post_clip_shuffle: int = 200,
) -> wds.WebDataset:
    """获得wds数据加载器

    Args:
        seed: 随机种子,用于固定shuffle顺序(主要用于验证集保证一致性)

    Note:
        对于验证集固定seed，使用整数seed而非Generator对象，
        因为Generator对象不能被pickle序列化，会导致checkpoint保存失败。
    """
    dataset = (
        wds.WebDataset(
            url,
            resampled=infinite,
            shardshuffle=shardshuffle,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
        )
        .shuffle(20, initial=seed if seed is not None else 0)
        .decode()
    )

    dataset = dataset.compose(
        partial(
            clip_to_t_frames,
            num_frames,
            stride,
            sampling_mode=clip_sampling_mode,
            clips_per_sequence=clips_per_sequence,
            seed=seed,
        )
    )

    if post_clip_shuffle > 0:
        dataset = dataset.shuffle(post_clip_shuffle, initial=seed if seed is not None else 0)

    dataset = dataset.map(preprocess_frame).batched(
        batch_size,
        partial=False,
        collation_fn=collate_fn,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=prefetcher_factor if num_workers > 0 else None,
        pin_memory=False
    )

    return dataloader
