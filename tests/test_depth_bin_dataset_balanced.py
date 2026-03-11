import json
import sys
from pathlib import Path

import cv2
import numpy as np
import webdataset as wds

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.depth_bin_dataloader import (
    collect_depth_bin_cell_sources,
    compute_dataset_bin_cell_weights,
    get_dataset_bin_balanced_dataloader,
)


def _make_webp_bytes(color: int) -> bytes:
    img = np.full((8, 8, 3), color, dtype=np.uint8)
    ok, encoded = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, 100])
    if not ok:
        raise RuntimeError("webp 编码失败")
    return encoded.tobytes()


def _write_samples(output_pattern: Path, samples):
    sink = wds.ShardWriter(str(output_pattern), maxcount=1000)
    try:
        for sample in samples:
            sink.write(sample)
    finally:
        sink.close()


def _make_clip_sample(key: str, depth: float, bin_id: int, handedness: str):
    return {
        "__key__": key,
        "imgs_path.json": [f"{key}.jpg"],
        "img_bytes.pickle": [_make_webp_bytes(50 + bin_id)],
        "handedness.json": json.dumps(handedness),
        "joint_cam.npy": np.array([[[0.0, 0.0, depth]]], dtype=np.float32),
        "joint_rel.npy": np.zeros((1, 1, 3), dtype=np.float32),
        "joint_valid.npy": np.ones((1, 1), dtype=np.float32),
        "joint_img.npy": np.zeros((1, 1, 2), dtype=np.float32),
        "joint_hand_bbox.npy": np.zeros((1, 1, 2), dtype=np.float32),
        "hand_bbox.npy": np.zeros((1, 4), dtype=np.float32),
        "mano_pose.npy": np.zeros((1, 48), dtype=np.float32),
        "mano_shape.npy": np.zeros((1, 10), dtype=np.float32),
        "mano_valid.npy": np.ones((1,), dtype=np.bool_),
        "timestamp.npy": np.array([0], dtype=np.float32),
        "focal.npy": np.ones((1, 2), dtype=np.float32) * 600.0,
        "princpt.npy": np.ones((1, 2), dtype=np.float32) * 320.0,
        "depth_bin_id.npy": np.array(bin_id, dtype=np.int64),
        "root_depth_last.npy": np.array(depth, dtype=np.float32),
    }


def _write_dataset(root: Path, dataset_name: str, clip_dir: str, bin_name: str, sample_count: int, depth: float):
    target_dir = root / dataset_name / "train" / clip_dir / bin_name
    target_dir.mkdir(parents=True, exist_ok=True)
    samples = [
        _make_clip_sample(
            key=f"{dataset_name}_{bin_name}_{idx:04d}",
            depth=depth,
            bin_id=0 if "0000_0500" in bin_name else 1,
            handedness="right",
        )
        for idx in range(sample_count)
    ]
    _write_samples(target_dir / "%06d.tar", samples)


def test_dataset_bin_balanced_weights_and_loader(tmp_path: Path):
    root = tmp_path / "depth-bins"
    clip_dir_name = "nf1_s1"

    _write_dataset(root, "DatasetA", clip_dir_name, "bin_0000_0500", 100, 400.0)
    _write_dataset(root, "DatasetB", clip_dir_name, "bin_0000_0500", 25, 420.0)
    _write_dataset(root, "DatasetB", clip_dir_name, "bin_0500_0700", 16, 650.0)
    _write_dataset(root, "DatasetA", clip_dir_name, "bin_0500_0700", 4, 640.0)

    repack_stats_a = [
        {"bin": "bin_0000_0500", "sample_count": 100},
        {"bin": "bin_0500_0700", "sample_count": 4},
    ]
    repack_stats_b = [
        {"bin": "bin_0000_0500", "sample_count": 25},
        {"bin": "bin_0500_0700", "sample_count": 16},
    ]
    (root / "DatasetA" / "train" / clip_dir_name / "repack_stats.json").write_text(json.dumps(repack_stats_a))
    (root / "DatasetB" / "train" / clip_dir_name / "repack_stats.json").write_text(json.dumps(repack_stats_b))

    cell_sources = collect_depth_bin_cell_sources(
        root=str(root),
        dataset_names=["DatasetA", "DatasetB"],
        split="train",
        num_frames=1,
        stride=1,
        min_cell_samples=10,
    )

    assert list(cell_sources.keys()) == [
        "bin_0000_0500::DatasetA",
        "bin_0000_0500::DatasetB",
        "bin_0500_0700::DatasetB",
    ]

    weights = compute_dataset_bin_cell_weights(cell_sources, dataset_balance_alpha=0.5)
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert weights["bin_0500_0700::DatasetB"] == 0.5
    assert abs(weights["bin_0000_0500::DatasetA"] - (1.0 / 6.0)) < 1e-6
    assert abs(weights["bin_0000_0500::DatasetB"] - (1.0 / 3.0)) < 1e-6

    loader = get_dataset_bin_balanced_dataloader(
        cell_sources=cell_sources,
        batch_size=2,
        num_workers=0,
        prefetcher_factor=1,
        infinite=False,
        seed=42,
        dataset_balance_alpha=0.5,
        shardshuffle=False,
        sample_shuffle=0,
    )
    batch = next(iter(loader))
    assert batch["joint_cam"].shape[0] == 2
    assert batch["joint_cam"].shape[1] == 1
    assert "depth_bin_id" in batch
    assert "root_depth_last" in batch
