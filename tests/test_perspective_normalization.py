import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocess import (
    apply_perspective_to_points,
    build_intrinsic_matrices,
    compute_perspective_normalization_rotation,
    preprocess_batch,
)
from src.utils.proj import proj_points_3d


def _make_synthetic_batch(
    bbox_center=(140.0, 80.0),
    princpt=(100.0, 100.0),
    focal=(1000.0, 1000.0),
    depth=500.0,
    image_hw=(200, 200),
    has_intr=1.0,
):
    device = torch.device("cpu")
    dtype = torch.float32

    cx, cy = bbox_center
    fx, fy = focal
    px, py = princpt
    height, width = image_hw

    # 21 个二维关节，围绕 bbox 中心分布，保证 hand_bbox 中心就是 bbox_center
    offsets = torch.tensor(
        [
            [0.0, 0.0],
            [-20.0, -20.0], [20.0, -20.0], [20.0, 20.0], [-20.0, 20.0],
            [-15.0, 0.0], [15.0, 0.0], [0.0, -15.0], [0.0, 15.0],
            [-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0],
            [-25.0, 0.0], [25.0, 0.0], [0.0, -25.0], [0.0, 25.0],
            [-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0],
        ],
        dtype=dtype,
    )
    joint_img = offsets + torch.tensor([cx, cy], dtype=dtype)
    joint_img = joint_img.view(1, 1, 21, 2)

    z = torch.full((1, 1, 21, 1), depth, dtype=dtype)
    x = (joint_img[..., 0:1] - px) * z / fx
    y = (joint_img[..., 1:2] - py) * z / fy
    joint_cam = torch.cat([x, y, z], dim=-1)
    joint_rel = joint_cam - joint_cam[:, :, :1]

    hand_bbox = torch.tensor(
        [[[cx - 40.0, cy - 40.0, cx + 40.0, cy + 40.0]]],
        dtype=dtype,
    )

    img = torch.zeros(1, 3, height, width, dtype=torch.uint8)

    batch_origin = {
        "__key__": ["synthetic_sample"],
        "imgs_path": [["synthetic.png"]],
        "imgs": [img],
        "handedness": ["right"],
        "data_source": ["synthetic"],
        "source_split": ["unit_test"],
        "source_index": [[{"frame_idx_within_clip": 0}]],
        "intr_type": ["real" if has_intr > 0.5 else "none"],
        "additional_desc": [[{}]],
        "hand_bbox": hand_bbox,
        "joint_img": joint_img,
        "joint_cam": joint_cam,
        "joint_rel": joint_rel,
        "joint_2d_valid": torch.ones(1, 1, 21, dtype=dtype),
        "joint_3d_valid": torch.ones(1, 1, 21, dtype=dtype),
        "joint_valid": torch.ones(1, 1, 21, dtype=dtype),
        "mano_pose": torch.zeros(1, 1, 48, dtype=dtype),
        "mano_shape": torch.zeros(1, 1, 10, dtype=dtype),
        "has_mano": torch.zeros(1, 1, dtype=dtype),
        "mano_valid": torch.zeros(1, 1, dtype=dtype),
        "has_intr": torch.full((1, 1), has_intr, dtype=dtype),
        "timestamp": torch.zeros(1, 1, dtype=dtype),
        "focal": torch.tensor([[[fx, fy]]], dtype=dtype),
        "princpt": torch.tensor([[[px, py]]], dtype=dtype),
    }
    return batch_origin, device


def test_compute_perspective_normalization_rotation_identity_at_principal_point():
    hand_bbox = torch.tensor([[[80.0, 80.0, 120.0, 120.0]]], dtype=torch.float32)
    focal = torch.tensor([[[1000.0, 1000.0]]], dtype=torch.float32)
    princpt = torch.tensor([[[100.0, 100.0]]], dtype=torch.float32)

    rot = compute_perspective_normalization_rotation(hand_bbox, focal, princpt)
    eye = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3)

    assert torch.allclose(rot, eye, atol=1e-6)


def test_compute_perspective_normalization_rotation_aligns_bbox_ray_to_optical_axis():
    hand_bbox = torch.tensor([[[120.0, 60.0, 200.0, 100.0]]], dtype=torch.float32)
    focal = torch.tensor([[[1000.0, 1000.0]]], dtype=torch.float32)
    princpt = torch.tensor([[[100.0, 100.0]]], dtype=torch.float32)

    rot = compute_perspective_normalization_rotation(hand_bbox, focal, princpt)[0, 0]

    cx_hand = (hand_bbox[0, 0, 0] + hand_bbox[0, 0, 2]) * 0.5
    cy_hand = (hand_bbox[0, 0, 1] + hand_bbox[0, 0, 3]) * 0.5
    ray = torch.tensor(
        [
            (cx_hand - princpt[0, 0, 0]) / focal[0, 0, 0],
            (cy_hand - princpt[0, 0, 1]) / focal[0, 0, 1],
            1.0,
        ],
        dtype=torch.float32,
    )
    ray = ray / torch.norm(ray)

    ray_aligned = rot @ ray
    target = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

    assert torch.allclose(ray_aligned, target, atol=1e-5)
    assert torch.allclose(rot @ rot.transpose(0, 1), torch.eye(3), atol=1e-5)


def test_perspective_normalization_maps_bbox_center_to_principal_point_in_image_space():
    batch_origin, _ = _make_synthetic_batch()
    focal = batch_origin["focal"]
    princpt = batch_origin["princpt"]
    hand_bbox = batch_origin["hand_bbox"]

    rot = compute_perspective_normalization_rotation(hand_bbox, focal, princpt)
    intr, intr_inv = build_intrinsic_matrices(focal, princpt)
    correction_2d = intr @ rot @ intr_inv

    bbox_center = (hand_bbox[..., :2] + hand_bbox[..., 2:]) * 0.5
    bbox_center_trans = apply_perspective_to_points(correction_2d, bbox_center.unsqueeze(-2))
    bbox_center_trans = bbox_center_trans.squeeze(-2)

    assert torch.allclose(bbox_center_trans, princpt, atol=1e-4)


def test_preprocess_batch_perspective_normalization_preserves_reprojection_consistency():
    batch_origin, device = _make_synthetic_batch()

    batch_processed, trans_2d_mat, correction_rot_mat = preprocess_batch(
        batch_origin=batch_origin,
        patch_size=(64, 64),
        patch_expanstion=1.2,
        scale_z_range=(1.0, 1.0),
        scale_f_range=(1.0, 1.0),
        persp_rot_max=0.0,
        joint_rep_type="3",
        augmentation_flag=False,
        device=device,
        pixel_aug=None,
        perspective_normalization=True,
    )

    assert correction_rot_mat is not None
    assert batch_processed["patches"].shape == (1, 1, 3, 64, 64)

    joint_img_reproj = proj_points_3d(
        batch_processed["joint_cam"],
        batch_processed["focal"],
        batch_processed["princpt"],
    )

    assert torch.allclose(joint_img_reproj, batch_processed["joint_img"], atol=1e-3)

    bbox_center_orig = (batch_origin["hand_bbox"][..., :2] + batch_origin["hand_bbox"][..., 2:]) * 0.5
    bbox_center_trans = apply_perspective_to_points(
        trans_2d_mat,
        bbox_center_orig.unsqueeze(-2),
    ).squeeze(-2)
    assert torch.allclose(
        bbox_center_trans,
        batch_processed["princpt"],
        atol=1e-4,
    )


def test_preprocess_batch_perspective_normalization_is_noop_without_intrinsics():
    batch_origin, device = _make_synthetic_batch(has_intr=0.0)

    batch_processed, trans_2d_mat, correction_rot_mat = preprocess_batch(
        batch_origin=batch_origin,
        patch_size=(64, 64),
        patch_expanstion=1.2,
        scale_z_range=(1.0, 1.0),
        scale_f_range=(1.0, 1.0),
        persp_rot_max=0.0,
        joint_rep_type="3",
        augmentation_flag=False,
        device=device,
        pixel_aug=None,
        perspective_normalization=True,
    )

    eye = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3)
    assert correction_rot_mat is not None
    assert torch.allclose(correction_rot_mat, eye, atol=1e-6)
    assert torch.allclose(trans_2d_mat, eye, atol=1e-6)
    assert torch.allclose(batch_processed["joint_cam"], batch_origin["joint_cam"], atol=1e-6)
    assert torch.allclose(batch_processed["joint_img"], batch_origin["joint_img"], atol=1e-6)


def test_preprocess_batch_handles_degenerate_patch_bbox_in_augmentation_path():
    batch_origin, device = _make_synthetic_batch()

    # 构造退化样本：所有 2D joint 重合，hand_bbox 也退化为单点
    joint_img = torch.full_like(batch_origin["joint_img"], 80.0)
    z = batch_origin["joint_cam"][..., 2:3]
    focal = batch_origin["focal"]
    princpt = batch_origin["princpt"]
    px = princpt[..., 0:1].unsqueeze(-2)
    py = princpt[..., 1:2].unsqueeze(-2)
    fx = focal[..., 0:1].unsqueeze(-2)
    fy = focal[..., 1:2].unsqueeze(-2)
    x = (joint_img[..., 0:1] - px) * z / fx
    y = (joint_img[..., 1:2] - py) * z / fy

    batch_origin["joint_img"] = joint_img
    batch_origin["joint_cam"] = torch.cat([x, y, z], dim=-1)
    batch_origin["joint_rel"] = batch_origin["joint_cam"] - batch_origin["joint_cam"][:, :, :1]
    batch_origin["hand_bbox"] = torch.tensor(
        [[[80.0, 80.0, 80.0, 80.0]]],
        dtype=torch.float32,
    )

    batch_processed, trans_2d_mat, _ = preprocess_batch(
        batch_origin=batch_origin,
        patch_size=(64, 64),
        patch_expanstion=2.0,
        scale_z_range=(1.0, 1.0),
        scale_f_range=(1.0, 1.0),
        persp_rot_max=0.0,
        joint_rep_type="3",
        augmentation_flag=True,
        device=device,
        pixel_aug=None,
        perspective_normalization=False,
    )

    assert batch_processed["patches"].shape == (1, 1, 3, 64, 64)
    assert torch.isfinite(trans_2d_mat).all()
    assert not torch.isnan(batch_processed["patches"]).any()
    patch_bbox = batch_processed["patch_bbox"][0, 0]
    assert float(patch_bbox[2] - patch_bbox[0]) >= 1.0
    assert float(patch_bbox[3] - patch_bbox[1]) >= 1.0
