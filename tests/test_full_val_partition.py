import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataloader import build_balanced_clip_segments


def test_build_balanced_clip_segments_even_partition():
    urls = ["a.tar", "b.tar", "c.tar"]
    clip_counts = [100, 25, 75]
    segments = build_balanced_clip_segments(urls, clip_counts, num_parts=4)

    flattened = []
    for rank_segments in segments:
        for seg in rank_segments:
            flattened.append((seg.tar_path, seg.start_clip, seg.end_clip))

    assert sum(seg.end_clip - seg.start_clip for rank in segments for seg in rank) == sum(clip_counts)
    assert len(segments) == 4
    assert any(len(rank_segments) == 0 for rank_segments in segments) is False


def test_build_balanced_clip_segments_handles_fewer_tars_than_gpus():
    urls = ["big.tar", "small.tar"]
    clip_counts = [120, 20]
    segments = build_balanced_clip_segments(urls, clip_counts, num_parts=4)

    total = sum(seg.end_clip - seg.start_clip for rank in segments for seg in rank)
    assert total == 140
    assert len(segments) == 4
    assert sum(1 for rank in segments if len(rank) == 0) < 4
