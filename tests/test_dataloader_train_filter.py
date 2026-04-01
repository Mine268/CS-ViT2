import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataloader import build_clip_sample_filter_fn


def _make_clip(hand_bbox, valid_count, num_frames=1):
    joint_2d_valid = np.zeros((num_frames, 21), dtype=np.float32)
    joint_2d_valid[:, :valid_count] = 1.0
    hand_bbox = np.asarray(hand_bbox, dtype=np.float32).reshape(num_frames, 4)
    return {
        "hand_bbox": hand_bbox,
        "joint_2d_valid": joint_2d_valid,
    }


def test_clip_sample_filter_keeps_good_sample():
    sample_filter = build_clip_sample_filter_fn(
        {
            "enabled": True,
            "min_valid_joints_2d": 16,
            "min_hand_bbox_edge_px": 8,
            "frame_policy": "all",
        }
    )
    clip = _make_clip([[10.0, 20.0, 30.0, 45.0]], valid_count=21)
    assert sample_filter(clip) is True


def test_clip_sample_filter_rejects_few_valid_joints():
    sample_filter = build_clip_sample_filter_fn(
        {
            "enabled": True,
            "min_valid_joints_2d": 16,
            "min_hand_bbox_edge_px": 8,
            "frame_policy": "all",
        }
    )
    clip = _make_clip([[10.0, 20.0, 30.0, 45.0]], valid_count=8)
    assert sample_filter(clip) is False


def test_clip_sample_filter_rejects_small_bbox():
    sample_filter = build_clip_sample_filter_fn(
        {
            "enabled": True,
            "min_valid_joints_2d": 16,
            "min_hand_bbox_edge_px": 8,
            "frame_policy": "all",
        }
    )
    clip = _make_clip([[10.0, 20.0, 15.0, 26.0]], valid_count=21)
    assert sample_filter(clip) is False


def test_clip_sample_filter_all_policy_rejects_if_any_frame_bad():
    sample_filter = build_clip_sample_filter_fn(
        {
            "enabled": True,
            "min_valid_joints_2d": 16,
            "min_hand_bbox_edge_px": 8,
            "frame_policy": "all",
        }
    )
    clip = {
        "hand_bbox": np.array(
            [
                [10.0, 20.0, 30.0, 45.0],
                [10.0, 20.0, 14.0, 25.0],
            ],
            dtype=np.float32,
        ),
        "joint_2d_valid": np.array(
            [
                [1.0] * 21,
                [1.0] * 10 + [0.0] * 11,
            ],
            dtype=np.float32,
        ),
    }
    assert sample_filter(clip) is False
