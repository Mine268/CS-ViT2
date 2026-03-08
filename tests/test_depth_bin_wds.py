import json
import sys
from pathlib import Path

import cv2
import numpy as np
import webdataset as wds

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from preprocess.depth_bin_wds import convert_existing_wds_to_depth_bins
from src.data.depth_bin_dataloader import collect_depth_bin_sources, get_depth_bin_dataloader


def _make_webp_bytes(color: int) -> bytes:
    img = np.full((8, 8, 3), color, dtype=np.uint8)
    ok, encoded = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, 100])
    if not ok:
        raise RuntimeError("webp 编码失败")
    return encoded.tobytes()


def _write_input_wds(output_tar: Path):
    sample_a = {
        "__key__": "sample_a",
        "imgs_path.json": ["a_0.jpg", "a_1.jpg", "a_2.jpg"],
        "img_bytes.pickle": [_make_webp_bytes(10), _make_webp_bytes(20), _make_webp_bytes(30)],
        "handedness.json": json.dumps("right"),
        "joint_cam.npy": np.array(
            [
                [[0.0, 0.0, 420.0]],
                [[0.0, 0.0, 460.0]],
                [[0.0, 0.0, 520.0]],
            ],
            dtype=np.float32,
        ),
        "joint_rel.npy": np.zeros((3, 1, 3), dtype=np.float32),
        "joint_valid.npy": np.ones((3, 1), dtype=np.float32),
        "joint_img.npy": np.zeros((3, 1, 2), dtype=np.float32),
        "joint_hand_bbox.npy": np.zeros((3, 1, 2), dtype=np.float32),
        "hand_bbox.npy": np.zeros((3, 4), dtype=np.float32),
        "mano_pose.npy": np.zeros((3, 48), dtype=np.float32),
        "mano_shape.npy": np.zeros((3, 10), dtype=np.float32),
        "mano_valid.npy": np.ones((3,), dtype=np.bool_),
        "timestamp.npy": np.arange(3, dtype=np.float32),
        "focal.npy": np.ones((3, 2), dtype=np.float32) * 600.0,
        "princpt.npy": np.ones((3, 2), dtype=np.float32) * 320.0,
        "additional_desc.json": [{"frame": i} for i in range(3)],
    }
    sample_b = {
        "__key__": "sample_b",
        "imgs_path.json": ["b_0.jpg", "b_1.jpg", "b_2.jpg"],
        "img_bytes.pickle": [_make_webp_bytes(40), _make_webp_bytes(50), _make_webp_bytes(60)],
        "handedness.json": json.dumps("left"),
        "joint_cam.npy": np.array(
            [
                [[0.0, 0.0, 980.0]],
                [[0.0, 0.0, 1020.0]],
                [[0.0, 0.0, 1180.0]],
            ],
            dtype=np.float32,
        ),
        "joint_rel.npy": np.zeros((3, 1, 3), dtype=np.float32),
        "joint_valid.npy": np.ones((3, 1), dtype=np.float32),
        "joint_img.npy": np.zeros((3, 1, 2), dtype=np.float32),
        "joint_hand_bbox.npy": np.zeros((3, 1, 2), dtype=np.float32),
        "hand_bbox.npy": np.zeros((3, 4), dtype=np.float32),
        "mano_pose.npy": np.zeros((3, 48), dtype=np.float32),
        "mano_shape.npy": np.zeros((3, 10), dtype=np.float32),
        "mano_valid.npy": np.ones((3,), dtype=np.bool_),
        "timestamp.npy": np.arange(3, dtype=np.float32),
        "focal.npy": np.ones((3, 2), dtype=np.float32) * 600.0,
        "princpt.npy": np.ones((3, 2), dtype=np.float32) * 320.0,
        "additional_desc.json": [{"frame": i} for i in range(3)],
    }
    sink = wds.ShardWriter(str(output_tar), maxcount=1000)
    try:
        sink.write(sample_a)
        sink.write(sample_b)
    finally:
        sink.close()


def test_convert_existing_wds_to_depth_bins(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "depth_bins"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_tar = input_dir / "%06d.tar"
    _write_input_wds(input_tar)

    summary = convert_existing_wds_to_depth_bins(
        source_urls=[str(input_dir / "000000.tar")],
        output_root=str(output_dir),
        dataset_name="mockset",
        split="train",
        num_frames=1,
        stride=1,
        bin_edges=[0, 500, 900, 1100, 1000000],
        maxcount=10,
        maxsize=16 * 1024 * 1024,
    )

    assert summary["raw_samples"] == 2
    assert summary["clips"] == 6
    assert summary["bin_counts"]["bin_0000_0500"] == 2
    assert summary["bin_counts"]["bin_0500_0900"] == 1
    assert summary["bin_counts"]["bin_0900_1100"] == 2
    assert summary["bin_counts"]["bin_1100_inf"] == 1

    bin_sources = collect_depth_bin_sources(
        root=str(output_dir),
        dataset_names=["mockset"],
        split="train",
        num_frames=1,
        stride=1,
    )
    assert set(bin_sources.keys()) == {
        "bin_0000_0500",
        "bin_0500_0900",
        "bin_0900_1100",
        "bin_1100_inf",
    }

    loader = get_depth_bin_dataloader(
        bin_sources=bin_sources,
        batch_size=2,
        num_workers=0,
        prefetcher_factor=1,
        infinite=False,
        seed=42,
        shardshuffle=False,
        sample_shuffle=0,
        mix_strategy="round_robin",
    )
    batch = next(iter(loader))
    assert batch["joint_cam"].shape[0] == 2
    assert batch["joint_cam"].shape[1] == 1
    assert len(batch["imgs"]) == 2
    assert "depth_bin_id" in batch
    assert "root_depth_last" in batch
