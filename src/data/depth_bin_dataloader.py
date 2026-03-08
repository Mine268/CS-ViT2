from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torchvision
import webdataset as wds
from torch.utils.data import DataLoader

from .dataloader import collate_fn


def preprocess_preclipped_sample(sample):
    imgs_tensor = []
    for img_bytes in sample["img_bytes.pickle"]:
        buffer_np = np.frombuffer(img_bytes, dtype=np.uint8).copy()
        buffer = torch.from_numpy(buffer_np)
        img = torchvision.io.decode_webp(buffer)
        imgs_tensor.append(img)
    imgs_tensor = torch.stack(imgs_tensor)

    result = {
        "imgs_path": sample["imgs_path.json"],
        "imgs": imgs_tensor,
        "handedness": sample["handedness.json"],
    }

    for key, value in sample.items():
        if key in {"__key__", "__url__", "__local_path__", "imgs_path.json", "img_bytes.pickle", "handedness.json"}:
            continue
        if key.endswith(".npy"):
            out_key = key.replace(".npy", "")
            result[out_key] = torch.from_numpy(value).float() if isinstance(value, np.ndarray) else torch.tensor(value)
        elif key.endswith(".json"):
            out_key = key.replace(".json", "")
            result[out_key] = value
        else:
            result[key] = value

    return result


def collect_depth_bin_sources(
    root: str,
    dataset_names: Sequence[str],
    split: str,
    num_frames: int,
    stride: int,
    selected_bins: Optional[Sequence[str]] = None,
) -> "OrderedDict[str, List[str]]":
    base_name = f"nf{num_frames}_s{stride}"
    collected: Dict[str, List[str]] = {}

    for dataset_name in dataset_names:
        base_dir = Path(root) / dataset_name / split / base_name
        if not base_dir.exists():
            continue
        for bin_dir in sorted(base_dir.glob("bin_*")):
            if selected_bins is not None and bin_dir.name not in selected_bins:
                continue
            urls = sorted(str(path) for path in bin_dir.glob("*.tar"))
            if len(urls) == 0:
                continue
            collected.setdefault(bin_dir.name, []).extend(urls)

    return OrderedDict(
        (bin_name, sorted(urls))
        for bin_name, urls in sorted(collected.items(), key=lambda item: item[0])
    )


def _build_single_bin_dataset(
    urls: Sequence[str],
    infinite: bool,
    seed: Optional[int],
    shardshuffle,
    sample_shuffle: int,
):
    dataset = wds.WebDataset(
        list(urls),
        resampled=infinite,
        shardshuffle=shardshuffle,
        nodesplitter=wds.split_by_node,
        workersplitter=wds.split_by_worker,
    ).decode()

    if sample_shuffle > 0:
        dataset = dataset.shuffle(sample_shuffle, initial=seed if seed is not None else 0)

    return dataset.map(preprocess_preclipped_sample)


def get_depth_bin_dataloader(
    bin_sources: Dict[str, Sequence[str]],
    batch_size: int,
    num_workers: int,
    prefetcher_factor: int,
    infinite: bool = True,
    seed: Optional[int] = None,
    bin_weights: Optional[Sequence[float]] = None,
    shardshuffle=False,
    sample_shuffle: int = 200,
    mix_strategy: str = "uniform_random",
):
    if len(bin_sources) == 0:
        raise ValueError("bin_sources 不能为空")

    datasets = []
    for idx, urls in enumerate(bin_sources.values()):
        bin_seed = None if seed is None else seed + idx * 100003
        datasets.append(
            _build_single_bin_dataset(
                urls=urls,
                infinite=infinite,
                seed=bin_seed,
                shardshuffle=shardshuffle,
                sample_shuffle=sample_shuffle,
            )
        )

    if mix_strategy == "uniform_random":
        probs = bin_weights
        if isinstance(probs, Mapping):
            probs = [float(probs[bin_name]) for bin_name in bin_sources.keys()]
        if probs is None:
            probs = [1.0 / len(datasets)] * len(datasets)
        if len(probs) != len(datasets):
            raise ValueError(
                f"bin_weights length mismatch: got {len(probs)}, expected {len(datasets)}"
            )
        total_prob = float(sum(probs))
        if total_prob <= 0:
            raise ValueError(f"bin_weights sum must be positive, got {probs}")
        probs = [float(v) / total_prob for v in probs]
        mixed_dataset = wds.RandomMix(datasets, probs=probs, longest=not infinite)
    elif mix_strategy == "round_robin":
        mixed_dataset = wds.RoundRobin(datasets, longest=not infinite)
    else:
        raise ValueError(f"Unsupported mix_strategy: {mix_strategy}")

    return DataLoader(
        mixed_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetcher_factor if num_workers > 0 else None,
        collate_fn=collate_fn,
        pin_memory=False,
    )
