from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import webdataset as wds
from tqdm import tqdm


DEFAULT_MAXSIZE = 1536 * 1024 * 1024
DEFAULT_MAXCOUNT = 1_000_000


def iter_clip_roots(root: Path, datasets: Optional[Sequence[str]] = None) -> Iterable[Path]:
    if not root.exists():
        raise FileNotFoundError(f"depth-bin root 不存在: {root}")

    dataset_filter = set(datasets) if datasets is not None else None
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if dataset_filter is not None and dataset_dir.name not in dataset_filter:
            continue
        for split_dir in sorted(dataset_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            for clip_dir in sorted(split_dir.iterdir()):
                if clip_dir.is_dir():
                    yield clip_dir


def count_samples(urls: Sequence[str]) -> int:
    return sum(1 for _ in wds.WebDataset(list(urls), shardshuffle=False))


def get_tar_size_bytes(tar_paths: Sequence[Path]) -> int:
    return sum(path.stat().st_size for path in tar_paths)


def repack_bin_dir(
    source_bin_dir: Path,
    target_bin_dir: Path,
    maxsize: int,
    maxcount: int,
    verify_counts: bool = False,
) -> Dict:
    tar_paths = sorted(source_bin_dir.glob("*.tar"))
    if len(tar_paths) == 0:
        return {
            "bin": source_bin_dir.name,
            "source_tar_count": 0,
            "target_tar_count": 0,
            "source_size_bytes": 0,
            "target_size_bytes": 0,
            "sample_count": 0,
        }

    target_bin_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(target_bin_dir / "%06d.tar")
    source_urls = [str(path) for path in tar_paths]
    sample_count = 0

    sink = wds.ShardWriter(output_pattern, maxsize=maxsize, maxcount=maxcount)
    try:
        dataset = wds.WebDataset(source_urls, shardshuffle=False)
        for sample in tqdm(dataset, desc=f"repack:{source_bin_dir.name}", ncols=80):
            sink.write(sample)
            sample_count += 1
    finally:
        sink.close()

    target_tars = sorted(target_bin_dir.glob("*.tar"))
    stats = {
        "bin": source_bin_dir.name,
        "source_tar_count": len(tar_paths),
        "target_tar_count": len(target_tars),
        "source_size_bytes": get_tar_size_bytes(tar_paths),
        "target_size_bytes": get_tar_size_bytes(target_tars),
        "sample_count": sample_count,
    }

    if verify_counts:
        target_count = count_samples([str(path) for path in target_tars])
        stats["target_sample_count"] = target_count
        if target_count != sample_count:
            raise RuntimeError(
                f"sample count mismatch in {source_bin_dir}: source={sample_count}, target={target_count}"
            )

    return stats


def copy_clip_metadata(source_clip_dir: Path, target_clip_dir: Path):
    target_clip_dir.mkdir(parents=True, exist_ok=True)
    for path in source_clip_dir.iterdir():
        if path.is_file() and path.suffix.lower() == ".json":
            shutil.copy2(path, target_clip_dir / path.name)


def write_repack_stats(target_clip_dir: Path, stats: List[Dict]):
    with open(target_clip_dir / "repack_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


def repack_depth_bin_root(
    source_root: Path,
    target_root: Path,
    datasets: Optional[Sequence[str]] = None,
    maxsize: int = DEFAULT_MAXSIZE,
    maxcount: int = DEFAULT_MAXCOUNT,
    verify_counts: bool = False,
):
    if target_root.exists():
        raise FileExistsError(f"目标目录已存在，请先删除或更换路径: {target_root}")

    all_stats = []
    for source_clip_dir in iter_clip_roots(source_root, datasets=datasets):
        relative_clip_dir = source_clip_dir.relative_to(source_root)
        target_clip_dir = target_root / relative_clip_dir
        copy_clip_metadata(source_clip_dir, target_clip_dir)

        clip_stats = []
        for source_bin_dir in sorted(source_clip_dir.glob("bin_*")):
            if not source_bin_dir.is_dir():
                continue
            target_bin_dir = target_clip_dir / source_bin_dir.name
            stats = repack_bin_dir(
                source_bin_dir=source_bin_dir,
                target_bin_dir=target_bin_dir,
                maxsize=maxsize,
                maxcount=maxcount,
                verify_counts=verify_counts,
            )
            clip_stats.append(stats)
        write_repack_stats(target_clip_dir, clip_stats)
        all_stats.append(
            {
                "clip_dir": str(relative_clip_dir),
                "bins": clip_stats,
            }
        )

    summary = {
        "source_root": str(source_root),
        "target_root": str(target_root),
        "datasets": list(datasets) if datasets is not None else None,
        "maxsize": maxsize,
        "maxcount": maxcount,
        "verify_counts": verify_counts,
        "created_at": datetime.now().isoformat(),
        "clip_dirs": all_stats,
    }
    target_root.mkdir(parents=True, exist_ok=True)
    with open(target_root / "repack_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def swap_repacked_root(source_root: Path, target_root: Path, keep_backup: bool = False) -> Path:
    if not source_root.exists():
        raise FileNotFoundError(f"源目录不存在: {source_root}")
    if not target_root.exists():
        raise FileNotFoundError(f"repack 后目录不存在: {target_root}")

    backup_root = source_root.parent / f"{source_root.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        source_root.rename(backup_root)
        target_root.rename(source_root)
    except Exception:
        if backup_root.exists() and not source_root.exists():
            backup_root.rename(source_root)
        raise

    if not keep_backup and backup_root.exists():
        shutil.rmtree(backup_root)

    return backup_root


def build_argparser():
    parser = argparse.ArgumentParser(description="对 depth-bin 数据逐 dataset/bin 进行 repack，并用新目录替换旧目录")
    parser.add_argument("--root", required=True, help="原始 depth-bin 根目录")
    parser.add_argument(
        "--target-root",
        default=None,
        help="repack 输出临时目录，默认使用 <root>_repacked_tmp",
    )
    parser.add_argument("--datasets", nargs="+", default=None, help="只处理指定 dataset")
    parser.add_argument("--maxsize", type=int, default=DEFAULT_MAXSIZE)
    parser.add_argument("--maxcount", type=int, default=DEFAULT_MAXCOUNT)
    parser.add_argument("--verify-counts", action="store_true")
    parser.add_argument(
        "--keep-backup",
        action="store_true",
        help="替换完成后保留旧目录备份，不立即删除",
    )
    parser.add_argument(
        "--no-swap",
        action="store_true",
        help="仅生成新目录，不删除旧目录，也不改名替换",
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    source_root = Path(args.root).resolve()
    target_root = Path(args.target_root).resolve() if args.target_root else source_root.parent / f"{source_root.name}_repacked_tmp"

    summary = repack_depth_bin_root(
        source_root=source_root,
        target_root=target_root,
        datasets=args.datasets,
        maxsize=args.maxsize,
        maxcount=args.maxcount,
        verify_counts=args.verify_counts,
    )
    print(json.dumps(summary, indent=2))

    if not args.no_swap:
        backup_root = swap_repacked_root(
            source_root=source_root,
            target_root=target_root,
            keep_backup=args.keep_backup,
        )
        print(
            json.dumps(
                {
                    "swapped": True,
                    "active_root": str(source_root),
                    "backup_root": str(backup_root) if backup_root.exists() else None,
                    "keep_backup": args.keep_backup,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
