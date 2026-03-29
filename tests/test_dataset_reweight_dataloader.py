import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import webdataset as wds
from omegaconf import OmegaConf

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from script.train import collect_reweight_dataset_config
from src.data.dataloader import (
    compute_dataset_reweight_probs,
    get_dataset_reweight_dataloader,
)
from src.data.preprocess import preprocess_batch
from tests.test_src_data_dataloader import verify_batch, verify_origin_data


DATA_SOURCE_ALIASES = {
    "assemblyhands": "AssemblyHands",
    "coco_wholebody": "COCO-WholeBody",
    "dexycb": "DexYCB",
    "freihand": "FreiHAND",
    "ho3d": "HO3D_v3",
    "hot3d": "HOT3D",
    "ih26m": "InterHand2.6M",
    "mtc": "MTC",
    "rhd": "RHD",
}


def _make_webp_bytes(color: int) -> bytes:
    img = np.full((8, 8, 3), color, dtype=np.uint8)
    ok, encoded = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, 100])
    if not ok:
        raise RuntimeError("Failed to encode test image as WebP")
    return encoded.tobytes()


def _write_samples(output_pattern: Path, samples):
    sink = wds.ShardWriter(str(output_pattern), maxcount=1000)
    try:
        for sample in samples:
            sink.write(sample)
    finally:
        sink.close()


def _make_clip_sample(key: str, color: int, data_source: str):
    return {
        "__key__": key,
        "imgs_path.json": [f"{key}.jpg"],
        "img_bytes.pickle": [_make_webp_bytes(color)],
        "handedness.json": json.dumps("right"),
        "data_source.json": json.dumps(data_source),
        "source_split.json": json.dumps("train"),
        "joint_cam.npy": np.zeros((1, 21, 3), dtype=np.float32),
        "joint_valid.npy": np.ones((1, 21), dtype=np.float32),
        "joint_2d_valid.npy": np.ones((1, 21), dtype=np.float32),
        "joint_3d_valid.npy": np.ones((1, 21), dtype=np.float32),
        "has_mano.npy": np.ones((1,), dtype=np.float32),
        "has_intr.npy": np.ones((1,), dtype=np.float32),
        "focal.npy": np.ones((1, 2), dtype=np.float32) * 600.0,
        "princpt.npy": np.ones((1, 2), dtype=np.float32) * 320.0,
    }


def _write_dataset(root: Path, dataset_name: str, color: int, sample_count: int):
    target_dir = root / dataset_name / "train"
    target_dir.mkdir(parents=True, exist_ok=True)
    samples = [
        _make_clip_sample(
            key=f"{dataset_name}_{idx:04d}",
            color=color,
            data_source=dataset_name,
        )
        for idx in range(sample_count)
    ]
    _write_samples(target_dir / "%06d.tar", samples)


def test_compute_dataset_reweight_probs_and_loader(tmp_path: Path):
    root = tmp_path / "webdatasets2"
    _write_dataset(root, "DatasetA", color=60, sample_count=2)
    _write_dataset(root, "DatasetB", color=180, sample_count=2)

    dataset_sources = OrderedDict(
        [
            ("DatasetA", [str(root / "DatasetA" / "train" / "000000.tar")]),
            ("DatasetB", [str(root / "DatasetB" / "train" / "000000.tar")]),
        ]
    )
    dataset_weights = OrderedDict(
        [
            ("DatasetA", 1.0),
            ("DatasetB", 3.0),
        ]
    )

    probs = compute_dataset_reweight_probs(
        dataset_sources=dataset_sources,
        dataset_weights=dataset_weights,
    )
    assert abs(probs["DatasetA"] - 0.25) < 1e-6
    assert abs(probs["DatasetB"] - 0.75) < 1e-6

    loader = get_dataset_reweight_dataloader(
        dataset_sources=dataset_sources,
        dataset_weights=dataset_weights,
        num_frames=1,
        stride=1,
        batch_size=2,
        num_workers=0,
        prefetcher_factor=1,
        infinite=False,
        seed=42,
        clip_sampling_mode="dense",
        clips_per_sequence=None,
        shardshuffle=False,
        post_clip_shuffle=0,
        default_source_split="train",
    )
    batch = next(iter(loader))

    assert batch["joint_cam"].shape == (2, 1, 21, 3)
    assert set(batch["data_source"]).issubset({"DatasetA", "DatasetB"})
    assert all(split == "train" for split in batch["source_split"])


def test_collect_reweight_dataset_config_with_explicit_sources(tmp_path: Path):
    root = tmp_path / "webdatasets2"
    _write_dataset(root, "DatasetA", color=60, sample_count=1)
    _write_dataset(root, "DatasetB", color=180, sample_count=1)

    reweight_cfg = OmegaConf.create(
        {
            "datasets": [
                {
                    "name": "DatasetA",
                    "source": str(root / "DatasetA" / "train" / "*.tar"),
                    "weight": 1.0,
                },
                {
                    "name": "DatasetB",
                    "source": str(root / "DatasetB" / "train" / "*.tar"),
                    "weight": 3.0,
                },
            ]
        }
    )

    dataset_sources, dataset_weights = collect_reweight_dataset_config(
        source_patterns=[],
        reweight_cfg=reweight_cfg,
    )
    assert list(dataset_sources.keys()) == ["DatasetA", "DatasetB"]
    assert list(dataset_weights.items()) == [("DatasetA", 1.0), ("DatasetB", 3.0)]


def test_dataset_reweight_visualization_smoke():
    output_origin_dir = "tests/temp_reweight_origin"
    output_processed_dir = "tests/temp_reweight_processed"
    os.makedirs(output_origin_dir, exist_ok=True)
    os.makedirs(output_processed_dir, exist_ok=True)

    cfg = OmegaConf.load("config/stage1-dino_large_no_norm.yaml")
    dataset_sources, dataset_weights = collect_reweight_dataset_config(
        source_patterns=cfg.DATA.train.source,
        reweight_cfg=cfg.DATA.train.reweight,
    )
    assert len(dataset_sources) > 0, "Failed to collect reweight dataset sources"

    loader = get_dataset_reweight_dataloader(
        dataset_sources=dataset_sources,
        dataset_weights=dataset_weights,
        num_frames=cfg.MODEL.num_frame,
        stride=cfg.DATA.train.stride,
        batch_size=8,
        num_workers=0,
        prefetcher_factor=1,
        infinite=False,
        seed=42,
        clip_sampling_mode=cfg.DATA.train.sampling.get("mode", "dense"),
        clips_per_sequence=cfg.DATA.train.sampling.get("clips_per_sequence", None),
        shardshuffle=False,
        post_clip_shuffle=0,
        default_source_split=cfg.DATA.train.reweight.get("split", "train"),
    )

    batch = next(iter(loader))
    assert len(batch["imgs"]) > 0, "Failed to load images from reweight dataloader"
    normalized_batch_sources = {
        DATA_SOURCE_ALIASES.get(str(source).lower(), str(source))
        for source in batch["data_source"]
    }
    assert normalized_batch_sources.issubset(set(dataset_weights.keys()))

    vis_bx = None
    for bx in range(len(batch["imgs"])):
        has_valid_3d = bool(torch.all(batch["joint_3d_valid"][bx, 0] > 0.5).item())
        has_mano = bool(batch["has_mano"][bx, 0] > 0.5)
        has_intr = bool(batch["has_intr"][bx, 0] > 0.5)
        if has_valid_3d and has_mano and has_intr:
            vis_bx = bx
            break
    assert vis_bx is not None, "No valid 3D+MANO+intr sample found in reweight batch"

    verify_origin_data(batch, output_origin_dir, bx=vis_bx, tx=0)

    batch_processed, trans_2d_mat, _ = preprocess_batch(
        batch,
        [cfg.MODEL.img_size, cfg.MODEL.img_size],
        cfg.TRAIN.expansion_ratio,
        [1.0, 1.0],
        [1.0, 1.0],
        0.0,
        cfg.MODEL.joint_type,
        False,
        torch.device("cpu"),
        None,
        cfg.TRAIN.get("perspective_normalization", False),
    )
    verify_batch(
        batch_processed,
        trans_2d_mat,
        output_processed_dir,
        source_prefix="",
        bx=vis_bx,
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
