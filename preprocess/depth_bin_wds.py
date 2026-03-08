from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import webdataset as wds
from tqdm import tqdm


BIN_INF_THRESHOLD = 999999.0


def format_bin_name(low: float, high: Optional[float]) -> str:
    low_str = f"{int(low):04d}"
    if high is None:
        return f"bin_{low_str}_inf"
    high_str = f"{int(high):04d}"
    return f"bin_{low_str}_{high_str}"


def parse_bin_edges(bin_edges: Sequence[float]) -> List[float]:
    if len(bin_edges) < 2:
        raise ValueError("bin_edges 至少需要两个边界")
    parsed = [float(v) for v in bin_edges]
    if any(parsed[i] >= parsed[i + 1] for i in range(len(parsed) - 1)):
        raise ValueError(f"bin_edges 必须严格递增，收到: {parsed}")
    return parsed


def get_depth_bin(depth: float, bin_edges: Sequence[float]) -> Tuple[str, float, Optional[float], int]:
    parsed = parse_bin_edges(bin_edges)
    bin_index = np.searchsorted(parsed, depth, side="right") - 1
    bin_index = max(0, min(bin_index, len(parsed) - 2))
    low = parsed[bin_index]
    high = parsed[bin_index + 1]
    is_last = bin_index == len(parsed) - 2
    if is_last and high >= BIN_INF_THRESHOLD:
        return format_bin_name(low, None), low, None, bin_index
    return format_bin_name(low, high), low, high, bin_index


def iter_clip_starts(total_frames: int, num_frames: int, stride: int) -> Iterable[int]:
    total_clips = (total_frames - num_frames) // stride + 1
    if total_clips <= 0:
        return []
    return range(total_clips)


def slice_frame_aligned_value(value, start: int, end: int, total_frames: int):
    if isinstance(value, np.ndarray) and value.shape[0] == total_frames:
        return value[start:end].copy()
    if isinstance(value, list) and len(value) == total_frames:
        return value[start:end]
    return value


FRAME_ALIGNED_KEYS = [
    "imgs_path.json",
    "img_bytes.pickle",
    "focal.npy",
    "hand_bbox.npy",
    "joint_cam.npy",
    "joint_hand_bbox.npy",
    "joint_img.npy",
    "joint_rel.npy",
    "joint_valid.npy",
    "mano_pose.npy",
    "mano_shape.npy",
    "mano_valid.npy",
    "princpt.npy",
    "timestamp.npy",
]


def build_clip_sample(
    sample: MutableMapping,
    start: int,
    end: int,
    clip_idx: int,
    total_frames: int,
    depth_bin_index: int,
    root_depth_last: float,
):
    clip_sample = {
        "__key__": f"{sample['__key__']}_clip_{clip_idx:04d}",
        "handedness.json": json.dumps(sample["handedness.json"]),
        "depth_bin_id.npy": np.array(depth_bin_index, dtype=np.int64),
        "root_depth_last.npy": np.array(root_depth_last, dtype=np.float32),
    }
    for key in FRAME_ALIGNED_KEYS:
        if key not in sample:
            continue
        clip_sample[key] = slice_frame_aligned_value(sample[key], start, end, total_frames)
    return clip_sample




def get_writer_for_bin(
    writers: Dict[str, wds.ShardWriter],
    output_root: Path,
    dataset_name: str,
    split: str,
    num_frames: int,
    stride: int,
    bin_name: str,
    maxcount: int,
    maxsize: int,
) -> wds.ShardWriter:
    if bin_name not in writers:
        bin_dir = output_root / dataset_name / split / f"nf{num_frames}_s{stride}" / bin_name
        bin_dir.mkdir(parents=True, exist_ok=True)
        pattern = str(bin_dir / "%06d.tar")
        writers[bin_name] = wds.ShardWriter(pattern, maxcount=maxcount, maxsize=maxsize)
    return writers[bin_name]


def convert_existing_wds_to_depth_bins(
    source_urls: Sequence[str],
    output_root: str,
    dataset_name: str,
    split: str,
    num_frames: int,
    stride: int,
    bin_edges: Sequence[float],
    maxcount: int = 5000,
    maxsize: int = 1536 * 1024 * 1024,
    limit_raw_samples: Optional[int] = None,
) -> Dict:
    if len(source_urls) == 0:
        raise ValueError("source_urls 不能为空")

    output_root_path = Path(output_root)
    parsed_bin_edges = parse_bin_edges(bin_edges)
    dataset = wds.WebDataset(list(source_urls), shardshuffle=False).decode()
    writers: Dict[str, wds.ShardWriter] = {}

    summary = {
        "dataset_name": dataset_name,
        "split": split,
        "num_frames": num_frames,
        "stride": stride,
        "bin_edges": parsed_bin_edges,
        "raw_samples": 0,
        "clips": 0,
        "bin_counts": defaultdict(int),
    }

    try:
        iterator = enumerate(dataset)
        if limit_raw_samples is not None:
            iterator = ((i, sample) for i, sample in iterator if i < limit_raw_samples)
        for _, sample in tqdm(iterator, desc=f"depth-bin:{dataset_name}/{split}", ncols=80):
            total_frames = len(sample["img_bytes.pickle"])
            if total_frames < num_frames:
                continue
            summary["raw_samples"] += 1
            for clip_idx in iter_clip_starts(total_frames, num_frames, stride):
                start = clip_idx * stride
                end = start + num_frames
                root_depth_last = float(sample["joint_cam.npy"][end - 1, 0, 2])
                bin_name, _, _, bin_index = get_depth_bin(root_depth_last, parsed_bin_edges)
                clip_sample = build_clip_sample(
                    sample=sample,
                    start=start,
                    end=end,
                    clip_idx=int(clip_idx),
                    total_frames=total_frames,
                    depth_bin_index=bin_index,
                    root_depth_last=root_depth_last,
                )
                writer = get_writer_for_bin(
                    writers=writers,
                    output_root=output_root_path,
                    dataset_name=dataset_name,
                    split=split,
                    num_frames=num_frames,
                    stride=stride,
                    bin_name=bin_name,
                    maxcount=maxcount,
                    maxsize=maxsize,
                )
                writer.write(clip_sample)
                summary["clips"] += 1
                summary["bin_counts"][bin_name] += 1
    finally:
        for writer in writers.values():
            writer.close()

    summary["bin_counts"] = dict(summary["bin_counts"])
    summary_path = (
        output_root_path / dataset_name / split / f"nf{num_frames}_s{stride}" / "summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def build_argparser():
    parser = argparse.ArgumentParser(description="将现有 WebDataset 转换为静态深度分桶数据")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--source", nargs="+", required=True, help="输入 tar 的 glob 列表")
    parser.add_argument("--output-root", default="/mnt/qnap/data/datasets/webdatasets/depth-bins")
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--bin-edges",
        nargs="+",
        type=float,
        default=[0, 500, 700, 900, 1100, 1000000],
    )
    parser.add_argument("--maxcount", type=int, default=5000)
    parser.add_argument("--maxsize", type=int, default=1536 * 1024 * 1024)
    parser.add_argument("--limit-raw-samples", type=int, default=None)
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    source_urls: List[str] = []
    for pattern in args.source:
        source_urls.extend(sorted(glob.glob(pattern)))
    if len(source_urls) == 0:
        raise FileNotFoundError(f"没有匹配到任何输入 tar: {args.source}")

    summary = convert_existing_wds_to_depth_bins(
        source_urls=source_urls,
        output_root=args.output_root,
        dataset_name=args.dataset_name,
        split=args.split,
        num_frames=args.num_frames,
        stride=args.stride,
        bin_edges=args.bin_edges,
        maxcount=args.maxcount,
        maxsize=args.maxsize,
        limit_raw_samples=args.limit_raw_samples,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
