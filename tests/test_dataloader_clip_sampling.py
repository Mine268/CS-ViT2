import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataloader import clip_to_t_frames, _select_clip_indices


def make_sample(total_frames: int):
    joint_cam = np.arange(total_frames * 2 * 3, dtype=np.float32).reshape(total_frames, 2, 3)
    joint_valid = np.ones((total_frames, 2), dtype=np.float32)
    return {
        "__key__": "sample",
        "imgs_path.json": [f"frame_{i:04d}.jpg" for i in range(total_frames)],
        "img_bytes.pickle": [b"fake"] * total_frames,
        "handedness.json": "right",
        "joint_cam.npy": joint_cam,
        "joint_valid.npy": joint_valid,
    }


def test_select_clip_indices_dense_mode():
    rng = np.random.default_rng(123)
    indices = _select_clip_indices(
        total_frames=8,
        num_frames=3,
        stride=1,
        sampling_mode="dense",
        clips_per_sequence=None,
        rng=rng,
    )
    assert indices == [0, 1, 2, 3, 4, 5]


def test_select_clip_indices_random_clip_mode():
    rng = np.random.default_rng(123)
    indices = _select_clip_indices(
        total_frames=10,
        num_frames=4,
        stride=2,
        sampling_mode="random_clip",
        clips_per_sequence=2,
        rng=rng,
    )
    assert len(indices) == 2
    assert len(set(indices)) == 2
    assert all(0 <= idx < 4 for idx in indices)


def test_clip_to_t_frames_random_clip_returns_limited_samples():
    sample = make_sample(total_frames=8)
    outputs = list(
        clip_to_t_frames(
            3,
            1,
            [sample],
            sampling_mode="random_clip",
            clips_per_sequence=2,
            seed=123,
        )
    )

    assert len(outputs) == 2
    assert len({out["__key__"] for out in outputs}) == 2
    for out in outputs:
        assert len(out["imgs_path"]) == 3
        assert out["joint_cam"].shape == (3, 2, 3)
