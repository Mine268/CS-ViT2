import os
import sys
from pathlib import Path

import cv2
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.depth_bin_dataloader import collect_depth_bin_sources, get_depth_bin_dataloader
from src.data.preprocess import preprocess_batch
from tests.test_src_data_dataloader import verify_batch, verify_origin_data


DEFAULT_DEPTH_BIN_ROOTS = [
    "/mnt/qnap/data/datasets/webdatasets/depth-bins_repacked_tmp",
    "/mnt/qnap/data/datasets/webdatasets/depth-bins",
]


def _pick_available_root() -> str:
    for root in DEFAULT_DEPTH_BIN_ROOTS:
        if Path(root).exists():
            return root
    raise FileNotFoundError(
        "No depth-bin root found. Checked: " + ", ".join(DEFAULT_DEPTH_BIN_ROOTS)
    )


def test_depth_bin_dataloader_visualization_smoke():
    output_origin_dir = "tests/temp_depth_bin_origin"
    output_processed_dir = "tests/temp_depth_bin_processed"
    os.makedirs(output_origin_dir, exist_ok=True)
    os.makedirs(output_processed_dir, exist_ok=True)

    depth_bin_root = _pick_available_root()
    dataset_names = ["InterHand2.6M", "DexYCB_s1", "HO3D_v3", "HOT3D"]

    bin_sources = collect_depth_bin_sources(
        root=depth_bin_root,
        dataset_names=dataset_names,
        split="train",
        num_frames=1,
        stride=1,
    )
    assert len(bin_sources) > 0, f"No depth-bin data found under: {depth_bin_root}"

    loader = get_depth_bin_dataloader(
        bin_sources=bin_sources,
        batch_size=4,
        num_workers=0,
        prefetcher_factor=1,
        infinite=False,
        seed=42,
        shardshuffle=False,
        sample_shuffle=0,
        mix_strategy="round_robin",
    )

    batch = next(iter(loader))
    assert len(batch["imgs"]) > 0, "Failed to load images from depth-bin dataloader"
    assert "depth_bin_id" in batch, "depth_bin_id not found in batch"
    assert "root_depth_last" in batch, "root_depth_last not found in batch"

    verify_origin_data(batch, output_origin_dir, bx=0, tx=0)

    batch_processed, trans_2d_mat, _ = preprocess_batch(
        batch,
        [256, 256],
        1.1,
        [1.0, 1.0],
        [1.0, 1.0],
        0.0,
        "3",
        False,
        torch.device("cpu"),
        None,
        False,
    )
    verify_batch(
        batch_processed,
        trans_2d_mat,
        output_processed_dir,
        source_prefix="",
        bx=0,
        tx=0,
        origin_batch=batch,
    )

    expected_paths = [
        f"{output_origin_dir}/origin.png",
        f"{output_origin_dir}/joint_cam.png",
        f"{output_processed_dir}/origin.png",
        f"{output_processed_dir}/patch.png",
        f"{output_processed_dir}/bbox-joint_img.png",
    ]
    for img_path in expected_paths:
        assert os.path.exists(img_path), f"Expected visualization not found: {img_path}"
        img = cv2.imread(img_path)
        assert img is not None and img.size > 0, f"Failed to read visualization image: {img_path}"

    print(f"depth-bin visualization smoke test passed (root={depth_bin_root})")


if __name__ == "__main__":
    test_depth_bin_dataloader_visualization_smoke()
