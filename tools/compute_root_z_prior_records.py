from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from omegaconf import ListConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import get_dataset_reweight_dataloader
from src.model.root_z import compute_root_z_prior_and_geom


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


def canonicalize_data_source(name: str) -> str:
    return DATA_SOURCE_ALIASES.get(str(name).lower(), str(name))


def normalize_source_patterns(source_value: Any) -> List[str]:
    if isinstance(source_value, (list, tuple, ListConfig)):
        return [str(item) for item in source_value]
    return [str(source_value)]


def collect_reweight_dataset_config(
    source_patterns: List[str],
    reweight_cfg,
) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
    """
    解析显式 `reweight.datasets` 配置，返回 dataset -> shards 和 dataset -> weight。

    这里复用训练配置的 source/weight 语义，但不依赖 train.py，
    让脚本可以独立运行。
    """
    dataset_entries = reweight_cfg.get("datasets", [])
    if len(dataset_entries) > 0:
        dataset_sources: Dict[str, List[str]] = {}
        dataset_weights: Dict[str, float] = {}

        for entry in dataset_entries:
            dataset_name = str(entry.get("name", "")).strip()
            if dataset_name == "":
                raise ValueError("Each reweight dataset entry must provide a non-empty name")
            if dataset_name in dataset_sources:
                raise ValueError(f"Duplicate reweight dataset entry: {dataset_name}")

            patterns = normalize_source_patterns(entry.get("source", []))
            if len(patterns) == 0:
                raise ValueError(
                    f"Reweight dataset entry {dataset_name} must provide a non-empty source"
                )

            matched_files: List[str] = []
            for pattern_str in patterns:
                files = sorted(glob.glob(pattern_str))
                if len(files) == 0:
                    raise ValueError(
                        f"reweight dataset {dataset_name} pattern matched no files: {pattern_str}"
                    )
                matched_files.extend(files)

            dataset_sources[dataset_name] = matched_files
            dataset_weights[dataset_name] = float(entry.get("weight", 0.0))

        return dataset_sources, dataset_weights

    dataset_weights = {
        str(name): float(weight)
        for name, weight in reweight_cfg.get("weights", {}).items()
    }
    if not dataset_weights:
        raise ValueError("reweight config must provide either datasets or weights")

    dataset_sources = {name: [] for name in dataset_weights.keys()}
    for source_pattern in source_patterns:
        pattern_str = str(source_pattern)
        matched_names = [name for name in dataset_sources.keys() if name in pattern_str]
        if len(matched_names) != 1:
            raise ValueError(
                f"Source pattern must map to exactly one reweight dataset: {pattern_str} -> {matched_names}"
            )
        files = sorted(glob.glob(pattern_str))
        if len(files) == 0:
            raise ValueError(f"source pattern matched no files: {pattern_str}")
        dataset_sources[matched_names[0]].extend(files)

    missing = [name for name, files in dataset_sources.items() if len(files) == 0]
    if missing:
        raise ValueError(f"Missing source files for reweight datasets: {missing}")
    return dataset_sources, dataset_weights


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute current root-z prior values and dump sample-level records "
            "for later analysis."
        )
    )
    parser.add_argument(
        "--config",
        default="config/stage1-dino_large_no_norm.yaml",
        help="Training config that enables DATA.train.reweight.",
    )
    parser.add_argument(
        "--target-valid",
        type=int,
        default=0,
        help=(
            "Stop after collecting this many valid samples. "
            "Use 0 or a negative value to scan the whole finite loader."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size used by the analysis loader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of DataLoader workers. Keep this small for stable finite scans.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=1,
        help="DataLoader prefetch_factor.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed passed to the loader.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Number of valid samples per chunk npz file.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Print progress every N valid samples.",
    )
    parser.add_argument(
        "--out-dir",
        default="tests/temp_root_z_prior_records",
        help="Directory to store chunk npz files and summary json.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output filename prefix. Defaults to the config stem.",
    )
    return parser


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        raise ValueError("Cannot compute percentile on an empty array")
    return float(np.quantile(values, q))


def initialize_chunk_buffer() -> Dict[str, List[Any]]:
    """
    初始化一个 chunk 缓冲区。

    这里存的字段，都是之后分析 z_prior 异常样本时最常需要回看的数据：
    - 几何量：hand_bbox / focal / princpt / img_hw
    - 监督量：joint_img / joint_2d_valid / joint_3d_valid / root_z_gt
    - 计算结果：z_prior / delta_log_z / geom_feat
    - 索引信息：data_source / imgs_path
    """
    return {
        "data_source": [],
        "imgs_path": [],
        "root_z_gt": [],
        "z_prior": [],
        "delta_log_z": [],
        "hand_bbox": [],
        "focal": [],
        "princpt": [],
        "img_hw": [],
        "joint_img": [],
        "joint_2d_valid": [],
        "joint_3d_valid": [],
        "geom_feat": [],
        "valid_joint_count_2d": [],
        "valid_joint_count_3d": [],
    }


def flush_chunk(chunk: Dict[str, List[Any]], out_path: Path) -> int:
    if len(chunk["root_z_gt"]) == 0:
        return 0

    payload = {
        "data_source": np.array(chunk["data_source"], dtype=object),
        "imgs_path": np.array(chunk["imgs_path"], dtype=object),
        "root_z_gt": np.asarray(chunk["root_z_gt"], dtype=np.float32),
        "z_prior": np.asarray(chunk["z_prior"], dtype=np.float32),
        "delta_log_z": np.asarray(chunk["delta_log_z"], dtype=np.float32),
        "hand_bbox": np.asarray(chunk["hand_bbox"], dtype=np.float32),
        "focal": np.asarray(chunk["focal"], dtype=np.float32),
        "princpt": np.asarray(chunk["princpt"], dtype=np.float32),
        "img_hw": np.asarray(chunk["img_hw"], dtype=np.int32),
        "joint_img": np.asarray(chunk["joint_img"], dtype=np.float32),
        "joint_2d_valid": np.asarray(chunk["joint_2d_valid"], dtype=np.float32),
        "joint_3d_valid": np.asarray(chunk["joint_3d_valid"], dtype=np.float32),
        "geom_feat": np.asarray(chunk["geom_feat"], dtype=np.float32),
        "valid_joint_count_2d": np.asarray(chunk["valid_joint_count_2d"], dtype=np.int32),
        "valid_joint_count_3d": np.asarray(chunk["valid_joint_count_3d"], dtype=np.int32),
    }
    np.savez_compressed(out_path, **payload)
    count = int(len(chunk["root_z_gt"]))
    chunk.clear()
    chunk.update(initialize_chunk_buffer())
    return count


def summarize(
    root_z_gt_values: np.ndarray,
    z_prior_values: np.ndarray,
    delta_values: np.ndarray,
    valid_by_dataset: Dict[str, int],
    invalid_counts: Dict[str, int],
    config_path: str,
    elapsed_s: float,
) -> Dict[str, Any]:
    ratio = z_prior_values / np.clip(root_z_gt_values, a_min=1e-6, a_max=None)
    return {
        "config": config_path,
        "collected_valid_samples": int(root_z_gt_values.size),
        "elapsed_s": round(float(elapsed_s), 2),
        "global": {
            "root_z_gt_p01": round(percentile(root_z_gt_values, 0.01), 6),
            "root_z_gt_p50": round(percentile(root_z_gt_values, 0.50), 6),
            "root_z_gt_p99": round(percentile(root_z_gt_values, 0.99), 6),
            "z_prior_p01": round(percentile(z_prior_values, 0.01), 6),
            "z_prior_p50": round(percentile(z_prior_values, 0.50), 6),
            "z_prior_p99": round(percentile(z_prior_values, 0.99), 6),
            "delta_log_z_p01": round(percentile(delta_values, 0.01), 6),
            "delta_log_z_p50": round(percentile(delta_values, 0.50), 6),
            "delta_log_z_p99": round(percentile(delta_values, 0.99), 6),
            "z_prior_over_gt_q01": round(percentile(ratio, 0.01), 6),
            "z_prior_over_gt_q50": round(percentile(ratio, 0.50), 6),
            "z_prior_over_gt_q99": round(percentile(ratio, 0.99), 6),
        },
        "valid_by_dataset": dict(sorted(valid_by_dataset.items())),
        "invalid_counts": dict(sorted(invalid_counts.items())),
    }


def main() -> None:
    args = build_argparser().parse_args()
    cfg = OmegaConf.load(args.config)

    dataset_sources, dataset_weights = collect_reweight_dataset_config(
        cfg.DATA.train.source,
        cfg.DATA.train.reweight,
    )

    output_prefix = args.output_prefix or Path(args.config).stem
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 先清掉旧 chunk，避免前一次的分析结果和本次混淆。
    for old_chunk in sorted(out_dir.glob(f"{output_prefix}.chunk_*.npz")):
        old_chunk.unlink()

    loader = get_dataset_reweight_dataloader(
        dataset_sources=dataset_sources,
        dataset_weights=dataset_weights,
        num_frames=int(cfg.MODEL.num_frame),
        stride=int(cfg.DATA.train.stride),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        prefetcher_factor=int(args.prefetch_factor),
        infinite=False,
        seed=int(args.seed),
        clip_sampling_mode=cfg.DATA.train.sampling.get("mode", "dense"),
        clips_per_sequence=cfg.DATA.train.sampling.get("clips_per_sequence", None),
        shardshuffle=False,
        post_clip_shuffle=0,
        default_source_split=cfg.DATA.train.reweight.get("split", "train"),
    )

    run_info = {
        "config": args.config,
        "target_valid": int(args.target_valid),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "seed": int(args.seed),
        "chunk_size": int(args.chunk_size),
        "dataset_counts": {name: len(files) for name, files in dataset_sources.items()},
        "effective_shardshuffle": False,
        "effective_post_clip_shuffle": 0,
        "out_dir": str(out_dir.resolve()),
    }
    run_info_path = out_dir / f"{output_prefix}.run_info.json"
    run_info_path.write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "starting", **run_info}, ensure_ascii=False), flush=True)

    chunk = initialize_chunk_buffer()
    root_z_all: List[float] = []
    z_prior_all: List[float] = []
    delta_all: List[float] = []
    valid_by_dataset: Dict[str, int] = defaultdict(int)
    invalid_counts: Dict[str, int] = defaultdict(int)

    prior_k = float(cfg.MODEL.handec.root_z.prior_k)
    valid_total = 0
    chunk_idx = 0
    started_at = time.perf_counter()

    for batch_idx, batch in enumerate(loader):
        tx = -1
        hand_bbox = batch["hand_bbox"][:, tx]
        focal = batch["focal"][:, tx]
        princpt = batch["princpt"][:, tx]
        joint_img = batch["joint_img"][:, tx]
        joint_2d_valid = batch.get("joint_2d_valid", batch["joint_valid"])[:, tx]
        joint_3d_valid = batch.get("joint_3d_valid", batch["joint_valid"])[:, tx]
        joint_cam = batch["joint_cam"][:, tx]
        root_z_gt = joint_cam[:, 0, 2]
        has_intr = batch.get("has_intr", torch.ones_like(batch["timestamp"]))[:, tx]

        z_prior, log_z_prior, geom_feat = compute_root_z_prior_and_geom(
            hand_bbox=hand_bbox,
            focal=focal,
            princpt=princpt,
            prior_k=prior_k,
        )
        delta_log_z = torch.log(torch.clamp(root_z_gt, min=1e-6)) - log_z_prior

        valid_mask = (
            (joint_3d_valid[:, 0] > 0.5)
            & (has_intr > 0.5)
            & (root_z_gt > 0.0)
            & torch.isfinite(z_prior)
            & torch.isfinite(delta_log_z)
        )
        if torch.any(~valid_mask):
            invalid_counts["filtered"] += int((~valid_mask).sum().item())

        valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten().tolist()
        for idx in valid_indices:
            data_source = canonicalize_data_source(batch["data_source"][idx])
            valid_by_dataset[data_source] += 1

            img_tensor = batch["imgs"][idx][tx]
            _, img_h, img_w = img_tensor.shape

            chunk["data_source"].append(data_source)
            chunk["imgs_path"].append(batch["imgs_path"][idx][tx])
            chunk["root_z_gt"].append(float(root_z_gt[idx].item()))
            chunk["z_prior"].append(float(z_prior[idx].item()))
            chunk["delta_log_z"].append(float(delta_log_z[idx].item()))
            chunk["hand_bbox"].append(hand_bbox[idx].cpu().numpy())
            chunk["focal"].append(focal[idx].cpu().numpy())
            chunk["princpt"].append(princpt[idx].cpu().numpy())
            chunk["img_hw"].append([int(img_h), int(img_w)])
            chunk["joint_img"].append(joint_img[idx].cpu().numpy())
            chunk["joint_2d_valid"].append(joint_2d_valid[idx].cpu().numpy())
            chunk["joint_3d_valid"].append(joint_3d_valid[idx].cpu().numpy())
            chunk["geom_feat"].append(geom_feat[idx].cpu().numpy())
            chunk["valid_joint_count_2d"].append(int((joint_2d_valid[idx] > 0.5).sum().item()))
            chunk["valid_joint_count_3d"].append(int((joint_3d_valid[idx] > 0.5).sum().item()))

            root_z_all.append(float(root_z_gt[idx].item()))
            z_prior_all.append(float(z_prior[idx].item()))
            delta_all.append(float(delta_log_z[idx].item()))
            valid_total += 1

        if valid_total > 0 and valid_total % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "batch_idx": batch_idx + 1,
                        "valid_total": valid_total,
                        "elapsed_s": round(time.perf_counter() - started_at, 2),
                        "latest_data_source": chunk["data_source"][-1] if chunk["data_source"] else None,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        if len(chunk["root_z_gt"]) >= args.chunk_size:
            chunk_path = out_dir / f"{output_prefix}.chunk_{chunk_idx:04d}.npz"
            flushed = flush_chunk(chunk, chunk_path)
            print(
                json.dumps(
                    {
                        "event": "chunk_saved",
                        "chunk_idx": chunk_idx,
                        "chunk_path": str(chunk_path.resolve()),
                        "flushed_samples": flushed,
                        "valid_total": valid_total,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            chunk_idx += 1

        if args.target_valid > 0 and valid_total >= args.target_valid:
            break

    if len(chunk["root_z_gt"]) > 0:
        chunk_path = out_dir / f"{output_prefix}.chunk_{chunk_idx:04d}.npz"
        flushed = flush_chunk(chunk, chunk_path)
        print(
            json.dumps(
                {
                    "event": "chunk_saved",
                    "chunk_idx": chunk_idx,
                    "chunk_path": str(chunk_path.resolve()),
                    "flushed_samples": flushed,
                    "valid_total": valid_total,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = summarize(
        root_z_gt_values=np.asarray(root_z_all, dtype=np.float64),
        z_prior_values=np.asarray(z_prior_all, dtype=np.float64),
        delta_values=np.asarray(delta_all, dtype=np.float64),
        valid_by_dataset=valid_by_dataset,
        invalid_counts=invalid_counts,
        config_path=args.config,
        elapsed_s=time.perf_counter() - started_at,
    )
    summary_path = out_dir / f"{output_prefix}.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(str(summary_path.resolve()), flush=True)


if __name__ == "__main__":
    main()
