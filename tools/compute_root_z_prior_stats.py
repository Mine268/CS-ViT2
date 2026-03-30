from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import threading
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
) -> Tuple["Dict[str, List[str]]", "Dict[str, float]"]:
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


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        raise ValueError("Cannot compute percentile on empty array")
    return float(np.quantile(values, q))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect root-z prior statistics for the reweight dataloader and "
            "estimate k / d_min / d_max for a prior-centered delta-log-z head."
        )
    )
    parser.add_argument(
        "--config",
        default="config/stage1-dino_large_no_norm.yaml",
        help="Path to a training config that uses DATA.train.reweight.",
    )
    parser.add_argument(
        "--target-valid",
        type=int,
        default=300000,
        help=(
            "Stop after collecting this many valid root-depth samples. "
            "Use 0 or a negative value to scan the entire finite dataloader."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50000,
        help="Write partial summaries every N newly collected valid samples.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override dataloader batch size. Defaults to cfg.TRAIN.sample_per_device.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override dataloader num_workers. Defaults to cfg.GENERAL.num_worker.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help="Override dataloader prefetch_factor. Defaults to cfg.GENERAL.prefetch_factor.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed passed to the dataloader.",
    )
    parser.add_argument(
        "--out-dir",
        default="tests/temp_root_z_prior_stats",
        help="Directory to store partial and final statistics.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output filename prefix. Defaults to the config stem.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing partial npz/json under --out-dir.",
    )
    parser.add_argument(
        "--infinite",
        action="store_true",
        help=(
            "Use the training-style resampled infinite loader. "
            "Disabled by default because finite traversal is usually enough for statistics."
        ),
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=30.0,
        help="Print a heartbeat log every N seconds while the script is running.",
    )
    return parser


def summarize_stats(
    k_values: np.ndarray,
    dataset_indices: np.ndarray,
    dataset_names: List[str],
    valid_by_dataset: Dict[str, int],
    invalid_counts: Dict[str, int],
    elapsed_s: float,
    config_path: str,
    target_valid: int,
) -> Dict:
    if k_values.size == 0:
        raise ValueError("No valid samples collected")

    log_k = np.log(k_values)
    k_median = float(np.median(k_values))
    delta_log_z = log_k - math.log(k_median)

    summary = {
        "config": config_path,
        "target_valid_samples": int(target_valid),
        "collected_valid_samples": int(k_values.size),
        "elapsed_s": round(float(elapsed_s), 2),
        "recommended": {
            "k_median": round(k_median, 6),
            "delta_log_z_p01": round(percentile(delta_log_z, 0.01), 6),
            "delta_log_z_p99": round(percentile(delta_log_z, 0.99), 6),
            "delta_log_z_dmin_margin_0p1": round(percentile(delta_log_z, 0.01) - 0.1, 6),
            "delta_log_z_dmax_margin_0p1": round(percentile(delta_log_z, 0.99) + 0.1, 6),
        },
        "global_stats": {
            "k_p01": round(percentile(k_values, 0.01), 6),
            "k_p50": round(percentile(k_values, 0.50), 6),
            "k_p99": round(percentile(k_values, 0.99), 6),
            "delta_p01": round(percentile(delta_log_z, 0.01), 6),
            "delta_p50": round(percentile(delta_log_z, 0.50), 6),
            "delta_p99": round(percentile(delta_log_z, 0.99), 6),
        },
        "valid_by_dataset": dict(sorted(valid_by_dataset.items())),
        "invalid_counts": dict(sorted(invalid_counts.items())),
        "per_dataset": {},
    }

    for dataset_id, dataset_name in enumerate(dataset_names):
        mask = dataset_indices == dataset_id
        if not np.any(mask):
            continue
        dataset_k = k_values[mask]
        dataset_delta = np.log(dataset_k) - math.log(k_median)
        summary["per_dataset"][dataset_name] = {
            "count": int(dataset_k.size),
            "k_p01": round(percentile(dataset_k, 0.01), 6),
            "k_p50": round(percentile(dataset_k, 0.50), 6),
            "k_p99": round(percentile(dataset_k, 0.99), 6),
            "delta_p01": round(percentile(dataset_delta, 0.01), 6),
            "delta_p50": round(percentile(dataset_delta, 0.50), 6),
            "delta_p99": round(percentile(dataset_delta, 0.99), 6),
        }

    return summary


def save_partial(
    npz_path: Path,
    json_path: Path,
    k_values: np.ndarray,
    dataset_indices: np.ndarray,
    dataset_names: List[str],
    valid_by_dataset: Dict[str, int],
    invalid_counts: Dict[str, int],
    elapsed_s: float,
    config_path: str,
    target_valid: int,
) -> None:
    np.savez_compressed(
        npz_path,
        k_values=k_values,
        dataset_indices=dataset_indices,
        dataset_names=np.array(dataset_names, dtype=object),
    )
    summary = summarize_stats(
        k_values=k_values,
        dataset_indices=dataset_indices,
        dataset_names=dataset_names,
        valid_by_dataset=valid_by_dataset,
        invalid_counts=invalid_counts,
        elapsed_s=elapsed_s,
        config_path=config_path,
        target_valid=target_valid,
    )
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))


class ProgressHeartbeat:
    def __init__(self, interval_s: float, output_path: Path):
        self.interval_s = interval_s
        self.output_path = output_path
        self.state: Dict[str, Any] = {
            "stage": "init",
            "elapsed_s": 0.0,
            "batch_idx": 0,
            "valid_samples": 0,
            "target_valid": 0,
            "wait_started_at_s": None,
            "message": "",
        }
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            self.state.update(kwargs)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.state)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            payload = self.snapshot()
            wait_started_at_s = payload.get("wait_started_at_s", None)
            if wait_started_at_s is not None:
                payload["waiting_for_batch_s"] = round(
                    max(0.0, payload["elapsed_s"] - wait_started_at_s), 2
                )
            else:
                payload["waiting_for_batch_s"] = None
            payload["event"] = "heartbeat"
            text = json.dumps(payload, ensure_ascii=False)
            print(text, flush=True)
            self.output_path.write_text(text + "\n")

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 1.0)


def load_partial(
    npz_path: Path,
    json_path: Path,
) -> Tuple[List[float], List[int], List[str], Dict[str, int], Dict[str, int]]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Partial npz not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    k_values = data["k_values"].astype(np.float64).tolist()
    dataset_indices = data["dataset_indices"].astype(np.int32).tolist()
    dataset_names = data["dataset_names"].tolist()

    valid_by_dataset: Dict[str, int] = {}
    invalid_counts: Dict[str, int] = {}
    if json_path.exists():
        payload = json.loads(json_path.read_text())
        valid_by_dataset = {
            str(key): int(value) for key, value in payload.get("valid_by_dataset", {}).items()
        }
        invalid_counts = {
            str(key): int(value) for key, value in payload.get("invalid_counts", {}).items()
        }
    return k_values, dataset_indices, dataset_names, valid_by_dataset, invalid_counts


def main() -> None:
    args = build_argparser().parse_args()
    cfg = OmegaConf.load(args.config)

    dataset_sources, dataset_weights = collect_reweight_dataset_config(
        cfg.DATA.train.source,
        cfg.DATA.train.reweight,
    )
    dataset_names = list(dataset_weights.keys())
    dataset_to_idx = {name: idx for idx, name in enumerate(dataset_names)}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = args.output_prefix or Path(args.config).stem
    partial_npz_path = out_dir / f"{output_prefix}.partial.npz"
    partial_json_path = out_dir / f"{output_prefix}.partial.json"
    final_json_path = out_dir / f"{output_prefix}.json"
    run_info_path = out_dir / f"{output_prefix}.run_info.json"
    heartbeat_path = out_dir / f"{output_prefix}.heartbeat.jsonl"

    k_values_list: List[float] = []
    dataset_indices_list: List[int] = []
    valid_by_dataset: Dict[str, int] = defaultdict(int)
    invalid_counts: Dict[str, int] = defaultdict(int)

    if args.resume and partial_npz_path.exists():
        (
            k_values_list,
            dataset_indices_list,
            resumed_dataset_names,
            resumed_valid_by_dataset,
            resumed_invalid_counts,
        ) = load_partial(partial_npz_path, partial_json_path)
        if resumed_dataset_names != dataset_names:
            raise ValueError(
                f"Dataset names mismatch between config and partial file: "
                f"{dataset_names} vs {resumed_dataset_names}"
            )
        valid_by_dataset.update(resumed_valid_by_dataset)
        invalid_counts.update(resumed_invalid_counts)

    batch_size = args.batch_size or int(cfg.TRAIN.sample_per_device)
    num_workers = args.num_workers if args.num_workers is not None else int(cfg.GENERAL.num_worker)
    prefetch_factor = (
        args.prefetch_factor if args.prefetch_factor is not None else int(cfg.GENERAL.prefetch_factor)
    )
    global_start = time.perf_counter()
    heartbeat = ProgressHeartbeat(
        interval_s=float(args.heartbeat_seconds),
        output_path=heartbeat_path,
    )
    heartbeat.update(
        stage="config_loaded",
        elapsed_s=0.0,
        valid_samples=len(k_values_list),
        target_valid=int(args.target_valid),
        message="Config loaded and resume state restored.",
    )
    heartbeat.start()

    heartbeat.update(
        stage="building_loader",
        elapsed_s=round(time.perf_counter() - global_start, 2),
        message="Building dataloader.",
    )

    loader = get_dataset_reweight_dataloader(
        dataset_sources=dataset_sources,
        dataset_weights=dataset_weights,
        num_frames=int(cfg.MODEL.num_frame),
        stride=int(cfg.DATA.train.stride),
        batch_size=batch_size,
        num_workers=num_workers,
        prefetcher_factor=prefetch_factor,
        infinite=args.infinite,
        seed=int(args.seed),
        clip_sampling_mode=cfg.DATA.train.sampling.get("mode", "dense"),
        clips_per_sequence=cfg.DATA.train.sampling.get("clips_per_sequence", None),
        shardshuffle=False,
        post_clip_shuffle=0,
        default_source_split=cfg.DATA.train.reweight.get("split", "train"),
    )

    run_info = {
        "config": args.config,
        "target_valid_samples": int(args.target_valid),
        "full_scan_mode": bool(args.target_valid <= 0),
        "checkpoint_every": int(args.checkpoint_every),
        "resume": bool(args.resume),
        "infinite": bool(args.infinite),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "prefetch_factor": int(prefetch_factor),
        "seed": int(args.seed),
        "num_datasets": len(dataset_sources),
        "dataset_counts": {name: len(files) for name, files in dataset_sources.items()},
        "effective_shardshuffle": False,
        "effective_post_clip_shuffle": 0,
        "output_prefix": output_prefix,
        "out_dir": str(out_dir.resolve()),
    }
    run_info_path.write_text(json.dumps(run_info, ensure_ascii=False, indent=2))
    print(json.dumps({"status": "starting", **run_info}, ensure_ascii=False), flush=True)
    heartbeat.update(
        stage="loader_ready",
        elapsed_s=round(time.perf_counter() - global_start, 2),
        message="Dataloader constructed. Entering collection loop.",
    )

    valid_collected = len(k_values_list)
    target_valid = int(args.target_valid)
    full_scan_mode = target_valid <= 0
    next_checkpoint = (
        ((valid_collected // args.checkpoint_every) + 1) * args.checkpoint_every
        if args.checkpoint_every > 0
        else (target_valid if target_valid > 0 else 0)
    )
    start_time = time.perf_counter()

    for batch_idx, batch in enumerate(loader):
        heartbeat.update(
            stage="processing_batch",
            elapsed_s=round(time.perf_counter() - global_start, 2),
            batch_idx=batch_idx + 1,
            valid_samples=valid_collected,
            wait_started_at_s=None,
            message="Batch received; validating and accumulating statistics.",
        )
        tx = -1
        joint_cam = batch["joint_cam"][:, tx]
        root_z = joint_cam[:, 0, 2]
        joint_3d_valid = batch.get("joint_3d_valid", batch["joint_valid"])[:, tx, 0]
        has_intr = batch.get("has_intr", torch.ones_like(batch["timestamp"]))[:, tx]
        hand_bbox = batch["hand_bbox"][:, tx]
        focal = batch["focal"][:, tx]

        bbox_w = hand_bbox[:, 2] - hand_bbox[:, 0]
        bbox_h = hand_bbox[:, 3] - hand_bbox[:, 1]
        fx = focal[:, 0]
        fy = focal[:, 1]

        valid_mask = (
            (joint_3d_valid > 0.5)
            & (has_intr > 0.5)
            & (root_z > 0.0)
            & (bbox_w > 1e-6)
            & (bbox_h > 1e-6)
            & (fx > 1e-6)
            & (fy > 1e-6)
        )
        if torch.any(~valid_mask):
            invalid_counts["filtered"] += int((~valid_mask).sum().item())

        valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
        for idx in valid_indices.tolist():
            dataset_name = canonicalize_data_source(batch["data_source"][idx])
            if dataset_name not in dataset_to_idx:
                invalid_counts["unknown_data_source"] += 1
                continue
            z = float(root_z[idx].item())
            bbox_scale = math.sqrt(float(bbox_w[idx].item()) * float(bbox_h[idx].item()))
            focal_eff = math.sqrt(float(fx[idx].item()) * float(fy[idx].item()))
            k_value = z * bbox_scale / focal_eff

            k_values_list.append(k_value)
            dataset_indices_list.append(dataset_to_idx[dataset_name])
            valid_by_dataset[dataset_name] += 1

        valid_collected = len(k_values_list)

        heartbeat.update(
            stage="waiting_next_batch",
            elapsed_s=round(time.perf_counter() - global_start, 2),
            batch_idx=batch_idx + 1,
            valid_samples=valid_collected,
            wait_started_at_s=round(time.perf_counter() - global_start, 2),
            message="Waiting for the next batch.",
        )

        if valid_collected >= next_checkpoint or valid_collected >= args.target_valid:
            elapsed_s = time.perf_counter() - start_time
            k_np = np.asarray(k_values_list, dtype=np.float64)
            ds_np = np.asarray(dataset_indices_list, dtype=np.int32)
            heartbeat.update(
                stage="saving_partial",
                elapsed_s=round(time.perf_counter() - global_start, 2),
                batch_idx=batch_idx + 1,
                valid_samples=valid_collected,
                wait_started_at_s=None,
                message=f"Saving partial statistics to {partial_json_path.name}.",
            )
            save_partial(
                npz_path=partial_npz_path,
                json_path=partial_json_path,
                k_values=k_np,
                dataset_indices=ds_np,
                dataset_names=dataset_names,
                valid_by_dataset=valid_by_dataset,
                invalid_counts=invalid_counts,
                elapsed_s=elapsed_s,
                config_path=args.config,
                target_valid=(target_valid if target_valid > 0 else valid_collected),
            )
            progress = {
                "batch_idx": batch_idx + 1,
                "valid_samples": valid_collected,
                "target_valid": target_valid,
                "full_scan_mode": full_scan_mode,
                "elapsed_s": round(elapsed_s, 2),
                "partial_json": str(partial_json_path.resolve()),
            }
            print(json.dumps(progress, ensure_ascii=False), flush=True)
            next_checkpoint += args.checkpoint_every
            heartbeat.update(
                stage="waiting_next_batch",
                elapsed_s=round(time.perf_counter() - global_start, 2),
                batch_idx=batch_idx + 1,
                valid_samples=valid_collected,
                wait_started_at_s=round(time.perf_counter() - global_start, 2),
                message="Partial statistics saved. Waiting for the next batch.",
            )

        if (not full_scan_mode) and valid_collected >= target_valid:
            break

    elapsed_s = time.perf_counter() - start_time
    k_np = np.asarray(k_values_list, dtype=np.float64)
    ds_np = np.asarray(dataset_indices_list, dtype=np.int32)
    final_summary = summarize_stats(
        k_values=k_np,
        dataset_indices=ds_np,
        dataset_names=dataset_names,
        valid_by_dataset=valid_by_dataset,
        invalid_counts=invalid_counts,
        elapsed_s=elapsed_s,
        config_path=args.config,
        target_valid=(target_valid if target_valid > 0 else valid_collected),
    )
    final_json_path.write_text(json.dumps(final_summary, ensure_ascii=False, indent=2))
    heartbeat.update(
        stage="done",
        elapsed_s=round(time.perf_counter() - global_start, 2),
        batch_idx=valid_collected,
        valid_samples=valid_collected,
        wait_started_at_s=None,
        message=f"Final statistics saved to {final_json_path.name}.",
    )
    heartbeat.stop()

    print(json.dumps(final_summary, ensure_ascii=False), flush=True)
    print(str(final_json_path.resolve()), flush=True)


if __name__ == "__main__":
    main()
