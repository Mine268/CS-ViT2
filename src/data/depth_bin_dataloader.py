from __future__ import annotations

from collections import OrderedDict, defaultdict
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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


def _load_bin_sample_counts(clip_dir: Path) -> Dict[str, int]:
    repack_stats_path = clip_dir / "repack_stats.json"
    if repack_stats_path.exists():
        stats = {}
        for item in json.loads(repack_stats_path.read_text()):
            count = int(item.get("sample_count", 0))
            if count > 0:
                stats[item["bin"]] = count
        return stats

    summary_path = clip_dir / "summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text())
        return {
            bin_name: int(count)
            for bin_name, count in data.get("bin_counts", {}).items()
            if int(count) > 0
        }

    return {}


def collect_depth_bin_cell_sources(
    root: str,
    dataset_names: Sequence[str],
    split: str,
    num_frames: int,
    stride: int,
    selected_bins: Optional[Sequence[str]] = None,
    min_cell_samples: int = 0,
) -> "OrderedDict[str, Dict]":
    base_name = f"nf{num_frames}_s{stride}"
    selected_bin_set = set(selected_bins) if selected_bins is not None else None
    collected: List[Tuple[str, Dict]] = []

    for dataset_name in sorted(dataset_names):
        clip_dir = Path(root) / dataset_name / split / base_name
        if not clip_dir.exists():
            continue
        sample_counts = _load_bin_sample_counts(clip_dir)
        for bin_dir in sorted(clip_dir.glob("bin_*")):
            if not bin_dir.is_dir():
                continue
            if selected_bin_set is not None and bin_dir.name not in selected_bin_set:
                continue
            urls = sorted(str(path) for path in bin_dir.glob("*.tar"))
            if len(urls) == 0:
                continue
            sample_count = int(sample_counts.get(bin_dir.name, 0))
            if sample_count < min_cell_samples:
                continue
            cell_key = f"{bin_dir.name}::{dataset_name}"
            collected.append(
                (
                    cell_key,
                    {
                        "bin_name": bin_dir.name,
                        "dataset_name": dataset_name,
                        "urls": urls,
                        "sample_count": sample_count,
                    },
                )
            )

    return OrderedDict(sorted(collected, key=lambda item: item[0]))


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


def compute_dataset_bin_cell_weights(
    cell_sources: Mapping[str, Mapping],
    dataset_balance_alpha: float = 0.5,
) -> "OrderedDict[str, float]":
    if len(cell_sources) == 0:
        raise ValueError("cell_sources 不能为空")

    bins_to_cells: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for cell_key, meta in cell_sources.items():
        sample_count = int(meta["sample_count"])
        if sample_count <= 0:
            continue
        bins_to_cells[meta["bin_name"]].append((cell_key, sample_count))

    if len(bins_to_cells) == 0:
        raise ValueError("No active dataset-bin cells found for weighting")

    bin_weight = 1.0 / len(bins_to_cells)
    weights: Dict[str, float] = {}

    for bin_name, cells in bins_to_cells.items():
        raw_weights = []
        for _, sample_count in cells:
            raw_weights.append(float(sample_count) ** (-float(dataset_balance_alpha)))
        raw_sum = sum(raw_weights)
        if raw_sum <= 0:
            raise ValueError(f"Invalid dataset-bin raw weights in {bin_name}: {raw_weights}")
        for (cell_key, _), raw_weight in zip(cells, raw_weights):
            weights[cell_key] = bin_weight * (raw_weight / raw_sum)

    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError(f"Invalid dataset-bin weights: {weights}")

    return OrderedDict(
        (cell_key, weights[cell_key] / total_weight)
        for cell_key in sorted(weights.keys())
    )


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


def get_dataset_bin_balanced_dataloader(
    cell_sources: Mapping[str, Mapping],
    batch_size: int,
    num_workers: int,
    prefetcher_factor: int,
    infinite: bool = True,
    seed: Optional[int] = None,
    dataset_balance_alpha: float = 0.5,
    shardshuffle=False,
    sample_shuffle: int = 200,
):
    if len(cell_sources) == 0:
        raise ValueError("cell_sources 不能为空")

    cell_weights = compute_dataset_bin_cell_weights(
        cell_sources=cell_sources,
        dataset_balance_alpha=dataset_balance_alpha,
    )

    datasets = []
    probs = []
    for idx, (cell_key, meta) in enumerate(cell_sources.items()):
        cell_seed = None if seed is None else seed + idx * 100003
        datasets.append(
            _build_single_bin_dataset(
                urls=meta["urls"],
                infinite=infinite,
                seed=cell_seed,
                shardshuffle=shardshuffle,
                sample_shuffle=sample_shuffle,
            )
        )
        probs.append(float(cell_weights[cell_key]))

    mixed_dataset = wds.RandomMix(datasets, probs=probs, longest=not infinite)

    return DataLoader(
        mixed_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetcher_factor if num_workers > 0 else None,
        collate_fn=collate_fn,
        pin_memory=False,
    )
