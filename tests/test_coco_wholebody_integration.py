import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
import os
import sys
from accelerate import PartialState

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from script.test import compute_quick_metrics
from script.train import assert_coco_wholebody_training_compat
from src.model.loss import build_data_source_mask, joint_img_to_patch_resized, robust_masked_mean
from src.utils.metric import build_excluded_data_source_mask


@pytest.fixture(autouse=True)
def _init_accelerate_state():
    try:
        PartialState()
    except RuntimeError:
        pass


def test_robust_masked_mean_does_not_dilute_with_invalid_entries():
    loss = torch.tensor([[2.0, 4.0], [100.0, 200.0]])
    mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])

    masked_mean = robust_masked_mean(loss, mask)

    assert torch.isclose(masked_mean, torch.tensor(3.0))


def test_joint_img_to_patch_resized_matches_patch_geometry():
    joint_img = torch.tensor([[[[20.0, 30.0], [60.0, 70.0]]]])
    patch_bbox = torch.tensor([[[10.0, 20.0, 110.0, 220.0]]])

    joint_patch = joint_img_to_patch_resized(
        joint_img,
        patch_bbox,
        patch_size=(200, 100),
    )

    expected = torch.tensor([[[[10.0, 10.0], [50.0, 50.0]]]])
    assert torch.allclose(joint_patch, expected)


def test_build_data_source_mask_only_selects_coco_wholebody():
    mask = build_data_source_mask(
        ["InterHand2.6M", "COCO-WholeBody", "DexYCB"],
        target_source="COCO-WholeBody",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    expected = torch.tensor([[0.0], [1.0], [0.0]])
    assert torch.equal(mask, expected)


def test_build_excluded_data_source_mask_filters_coco_wholebody():
    keep_mask = build_excluded_data_source_mask(
        ["InterHand2.6M", "COCO-WholeBody", "DexYCB"]
    )

    assert keep_mask.tolist() == [True, False, True]


def test_assert_coco_wholebody_training_compat_rejects_norm_by_hand():
    cfg = OmegaConf.create(
        {
            "DATA": {
                "train": {
                    "source": [
                        "/mnt/qnap/data/datasets/webdatasets2/COCO-WholeBody/train/*",
                    ]
                }
            },
            "MODEL": {"norm_by_hand": True},
        }
    )

    with pytest.raises(AssertionError, match="norm_by_hand=false"):
        assert_coco_wholebody_training_compat(cfg)


def test_compute_quick_metrics_excludes_coco_wholebody(tmp_path):
    results = {
        "joint_cam_pred": np.array(
            [
                np.zeros((21, 3), dtype=np.float32),
                np.full((21, 3), 100.0, dtype=np.float32),
            ]
        ),
        "joint_cam_gt": np.zeros((2, 21, 3), dtype=np.float32),
        "vert_cam_pred": np.array(
            [
                np.zeros((778, 3), dtype=np.float32),
                np.full((778, 3), 100.0, dtype=np.float32),
            ]
        ),
        "vert_cam_gt": np.zeros((2, 778, 3), dtype=np.float32),
        "joint_3d_valid": np.ones((2, 21), dtype=np.float32),
        "has_mano": np.ones((2,), dtype=np.float32),
        "data_source": np.array(["InterHand2.6M", "COCO-WholeBody"], dtype=object),
    }

    metrics = compute_quick_metrics(results, str(tmp_path))

    assert metrics["mpjpe"] == 0.0
    assert metrics["mpvpe"] == 0.0
    assert metrics["num_samples"] == 1
    assert metrics["num_samples_total"] == 2
    assert metrics["num_excluded_coco_wholebody"] == 1
