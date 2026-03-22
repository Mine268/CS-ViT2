"""
Webdataset Dataloader
"""
from typing import *
from functools import partial
from dataclasses import dataclass
import json
import tarfile
import numpy as np
import webdataset as wds

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import torchvision

from .schema_v2 import normalize_decoded_clip_sample, slice_normalized_clip_sample


COLLATE_LIST_KEYS = {"imgs"}


@dataclass(frozen=True)
class ClipSegment:
    tar_path: str
    start_clip: int
    end_clip: int


def count_sample_clips(total_frames: int, num_frames: int, stride: int) -> int:
    total_clips = (total_frames - num_frames) // stride + 1
    return max(0, total_clips)


def _select_clip_indices(
    total_frames: int,
    num_frames: int,
    stride: int,
    sampling_mode: str,
    clips_per_sequence: Optional[int],
    rng: np.random.Generator,
) -> List[int]:
    total_clips = count_sample_clips(total_frames, num_frames, stride)
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
    default_data_source: Optional[str] = None,
    default_source_split: str = "unknown",
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

    for decoded_sample in source:
        clip_sample = normalize_decoded_clip_sample(
            decoded_sample,
            default_data_source=default_data_source,
            default_source_split=default_source_split,
        )
        total_frames = clip_sample["num_frames"]
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
            yield slice_normalized_clip_sample(
                clip_sample,
                start,
                end,
                slice_index=int(clip_idx),
            )


def estimate_wds_shard_clip_counts(
    urls: Sequence[str],
    num_frames: int,
    stride: int,
) -> List[int]:
    clip_counts: List[int] = []
    for url in urls:
        shard_clip_count = 0
        with tarfile.open(url, "r") as tf:
            for member in tf:
                if not member.isfile() or not member.name.endswith("imgs_path.json"):
                    continue
                file_obj = tf.extractfile(member)
                if file_obj is None:
                    continue
                imgs_path = json.load(file_obj)
                shard_clip_count += count_sample_clips(len(imgs_path), num_frames, stride)
        clip_counts.append(shard_clip_count)
    return clip_counts


def build_balanced_clip_segments(
    urls: Sequence[str],
    clip_counts: Sequence[int],
    num_parts: int,
) -> List[List[ClipSegment]]:
    if len(urls) != len(clip_counts):
        raise ValueError(
            f"urls and clip_counts length mismatch: {len(urls)} vs {len(clip_counts)}"
        )
    if num_parts <= 0:
        raise ValueError(f"num_parts must be positive, got {num_parts}")

    total_clips = int(sum(clip_counts))
    assignments: List[List[ClipSegment]] = [[] for _ in range(num_parts)]
    if total_clips <= 0:
        return assignments

    part_ranges = [
        (total_clips * rank // num_parts, total_clips * (rank + 1) // num_parts)
        for rank in range(num_parts)
    ]

    clip_cursor = 0
    for url, shard_clip_count in zip(urls, clip_counts):
        shard_start = clip_cursor
        shard_end = clip_cursor + int(shard_clip_count)
        clip_cursor = shard_end
        if shard_clip_count <= 0:
            continue

        for rank, (part_start, part_end) in enumerate(part_ranges):
            overlap_start = max(shard_start, part_start)
            overlap_end = min(shard_end, part_end)
            if overlap_start >= overlap_end:
                continue
            assignments[rank].append(
                ClipSegment(
                    tar_path=url,
                    start_clip=overlap_start - shard_start,
                    end_clip=overlap_end - shard_start,
                )
            )

    return assignments


class WDSClipSegmentDataset(IterableDataset):
    def __init__(
        self,
        segments: Sequence[ClipSegment],
        num_frames: int,
        stride: int,
        default_data_source: Optional[str] = None,
        default_source_split: str = "unknown",
    ):
        super().__init__()
        self.segments = list(segments)
        self.num_frames = num_frames
        self.stride = stride
        self.default_data_source = default_data_source
        self.default_source_split = default_source_split

    def __iter__(self):
        for segment in self.segments:
            dataset = wds.WebDataset(
                [segment.tar_path],
                shardshuffle=False,
                nodesplitter=lambda src: src,
                workersplitter=lambda src: src,
            ).decode()
            clip_cursor = 0
            for decoded_sample in dataset:
                clip_sample = normalize_decoded_clip_sample(
                    decoded_sample,
                    default_data_source=self.default_data_source,
                    default_source_split=self.default_source_split,
                )
                total_frames = clip_sample["num_frames"]
                total_clips = count_sample_clips(total_frames, self.num_frames, self.stride)
                local_start = max(0, segment.start_clip - clip_cursor)
                local_end = min(total_clips, segment.end_clip - clip_cursor)
                if local_start < local_end:
                    for clip_idx in range(local_start, local_end):
                        start = clip_idx * self.stride
                        end = start + self.num_frames
                        yield preprocess_frame(
                            slice_normalized_clip_sample(
                                clip_sample,
                                start,
                                end,
                                slice_index=int(clip_idx),
                            )
                        )
                clip_cursor += total_clips
                if clip_cursor >= segment.end_clip:
                    break


def get_segmented_wds_dataloader(
    segments: Sequence[ClipSegment],
    num_frames: int,
    stride: int,
    batch_size: int,
    num_workers: int,
    prefetcher_factor: int,
    default_data_source: Optional[str] = None,
    default_source_split: str = "unknown",
):
    dataset = WDSClipSegmentDataset(
        segments=segments,
        num_frames=num_frames,
        stride=stride,
        default_data_source=default_data_source,
        default_source_split=default_source_split,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetcher_factor if num_workers > 0 else None,
        collate_fn=collate_fn,
        pin_memory=False,
    )


def _decode_webp_bytes(img_bytes: bytes) -> torch.Tensor:
    buffer_np = np.frombuffer(img_bytes, dtype=np.uint8).copy()
    buffer = torch.from_numpy(buffer_np)
    try:
        return torchvision.io.decode_webp(buffer)
    except RuntimeError:
        return torchvision.io.decode_image(
            buffer,
            mode=torchvision.io.ImageReadMode.RGB,
        )


def preprocess_frame(sample):
    """将图像二进制流转换为图片"""
    imgs_tensor = []
    for img_bytes in sample["imgs_bytes"]:
        imgs_tensor.append(_decode_webp_bytes(img_bytes))
    imgs_tensor = torch.stack(imgs_tensor)

    result = {
        "__key__": sample["__key__"],
        "imgs_path": sample["imgs_path"],
        "imgs": imgs_tensor,
        "handedness": sample["handedness"],
        "data_source": sample["data_source"],
        "source_split": sample["source_split"],
        "source_index": sample["source_index"],
        "intr_type": sample["intr_type"],
        "additional_desc": sample["additional_desc"],
    }

    for key, value in sample.items():
        if key in result or key in {"num_frames", "imgs_bytes"}:
            continue
        if isinstance(value, np.ndarray):
            result[key] = torch.from_numpy(value).float()
        else:
            result[key] = value

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
    default_data_source: Optional[str] = None,
    default_source_split: str = "unknown",
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
            default_data_source=default_data_source,
            default_source_split=default_source_split,
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
        pin_memory=False,
    )

    return dataloader
