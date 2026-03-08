import json
import sys
from pathlib import Path

import cv2
import numpy as np
import webdataset as wds

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from preprocess.repack_depth_bin_wds import repack_depth_bin_root, swap_repacked_root


def _make_webp_bytes(color: int) -> bytes:
    img = np.full((8, 8, 3), color, dtype=np.uint8)
    ok, encoded = cv2.imencode('.webp', img, [cv2.IMWRITE_WEBP_QUALITY, 100])
    if not ok:
        raise RuntimeError('webp 编码失败')
    return encoded.tobytes()


def _write_tar(output_pattern: Path, samples):
    sink = wds.ShardWriter(str(output_pattern), maxcount=2)
    try:
        for sample in samples:
            sink.write(sample)
    finally:
        sink.close()


def _count_samples(tars):
    return sum(1 for _ in wds.WebDataset([str(p) for p in tars], shardshuffle=False))


def test_repack_depth_bin_root_and_swap(tmp_path: Path):
    source_root = tmp_path / 'depth-bins'
    clip_dir = source_root / 'MockSet' / 'train' / 'nf1_s1'
    bin_a = clip_dir / 'bin_0000_0500'
    bin_b = clip_dir / 'bin_0500_0700'
    bin_a.mkdir(parents=True, exist_ok=True)
    bin_b.mkdir(parents=True, exist_ok=True)

    samples_a = [
        {
            '__key__': f'a_{i:04d}',
            'imgs_path.json': [f'a_{i}.jpg'],
            'img_bytes.pickle': [_make_webp_bytes(20 + i)],
            'handedness.json': json.dumps('right'),
            'joint_cam.npy': np.array([[[0.0, 0.0, 400.0]]], dtype=np.float32),
            'joint_rel.npy': np.zeros((1, 1, 3), dtype=np.float32),
            'joint_valid.npy': np.ones((1, 1), dtype=np.float32),
            'joint_img.npy': np.zeros((1, 1, 2), dtype=np.float32),
            'joint_hand_bbox.npy': np.zeros((1, 1, 2), dtype=np.float32),
            'hand_bbox.npy': np.zeros((1, 4), dtype=np.float32),
            'mano_pose.npy': np.zeros((1, 48), dtype=np.float32),
            'mano_shape.npy': np.zeros((1, 10), dtype=np.float32),
            'mano_valid.npy': np.ones((1,), dtype=np.bool_),
            'timestamp.npy': np.array([i], dtype=np.float32),
            'focal.npy': np.ones((1, 2), dtype=np.float32) * 600.0,
            'princpt.npy': np.ones((1, 2), dtype=np.float32) * 320.0,
            'depth_bin_id.npy': np.array(0, dtype=np.int64),
            'root_depth_last.npy': np.array(400.0, dtype=np.float32),
        }
        for i in range(3)
    ]
    samples_b = [
        {
            '__key__': f'b_{i:04d}',
            'imgs_path.json': [f'b_{i}.jpg'],
            'img_bytes.pickle': [_make_webp_bytes(100 + i)],
            'handedness.json': json.dumps('left'),
            'joint_cam.npy': np.array([[[0.0, 0.0, 650.0]]], dtype=np.float32),
            'joint_rel.npy': np.zeros((1, 1, 3), dtype=np.float32),
            'joint_valid.npy': np.ones((1, 1), dtype=np.float32),
            'joint_img.npy': np.zeros((1, 1, 2), dtype=np.float32),
            'joint_hand_bbox.npy': np.zeros((1, 1, 2), dtype=np.float32),
            'hand_bbox.npy': np.zeros((1, 4), dtype=np.float32),
            'mano_pose.npy': np.zeros((1, 48), dtype=np.float32),
            'mano_shape.npy': np.zeros((1, 10), dtype=np.float32),
            'mano_valid.npy': np.ones((1,), dtype=np.bool_),
            'timestamp.npy': np.array([i], dtype=np.float32),
            'focal.npy': np.ones((1, 2), dtype=np.float32) * 600.0,
            'princpt.npy': np.ones((1, 2), dtype=np.float32) * 320.0,
            'depth_bin_id.npy': np.array(1, dtype=np.int64),
            'root_depth_last.npy': np.array(650.0, dtype=np.float32),
        }
        for i in range(2)
    ]

    _write_tar(bin_a / '%06d.tar', samples_a)
    _write_tar(bin_b / '%06d.tar', samples_b)

    with open(clip_dir / 'summary.json', 'w') as f:
        json.dump({'clips': 5}, f)

    target_root = tmp_path / 'depth-bins_repacked_tmp'
    summary = repack_depth_bin_root(
        source_root=source_root,
        target_root=target_root,
        maxsize=1024 * 1024,
        maxcount=100,
        verify_counts=True,
    )
    assert target_root.exists()
    assert (target_root / 'MockSet' / 'train' / 'nf1_s1' / 'summary.json').exists()
    assert (target_root / 'MockSet' / 'train' / 'nf1_s1' / 'repack_stats.json').exists()

    repacked_a = sorted((target_root / 'MockSet' / 'train' / 'nf1_s1' / 'bin_0000_0500').glob('*.tar'))
    repacked_b = sorted((target_root / 'MockSet' / 'train' / 'nf1_s1' / 'bin_0500_0700').glob('*.tar'))
    assert _count_samples(repacked_a) == 3
    assert _count_samples(repacked_b) == 2

    backup_root = swap_repacked_root(source_root, target_root, keep_backup=False)
    assert source_root.exists()
    assert not target_root.exists()
    assert not backup_root.exists()
    assert (source_root / 'MockSet' / 'train' / 'nf1_s1' / 'summary.json').exists()
