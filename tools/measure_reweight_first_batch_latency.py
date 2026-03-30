from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from script.train import collect_reweight_dataset_config
from src.data.dataloader import get_dataset_reweight_dataloader


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure stage1/2 reweight dataloader startup latency until the first batch "
            "arrives. This is a standalone script, not a pytest test."
        )
    )
    parser.add_argument(
        "--config",
        default="config/stage1-dino_large_no_norm.yaml",
        help="Training config that enables DATA.train.reweight.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size. Defaults to cfg.TRAIN.sample_per_device.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers. Defaults to the recommended first-version value.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=1,
        help="DataLoader prefetch_factor. Defaults to the recommended first-version value.",
    )
    parser.add_argument(
        "--shardshuffle",
        type=int,
        default=128,
        help="Reweight shardshuffle override used for this measurement.",
    )
    parser.add_argument(
        "--post-clip-shuffle",
        type=int,
        default=64,
        help="Reweight post_clip_shuffle override used for this measurement.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=64,
        help="Number of batches to fetch for timing. The first batch is reported separately.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed passed to the dataloader.",
    )
    parser.add_argument(
        "--out-dir",
        default="tests/temp_first_batch_latency",
        help="Directory for heartbeat and final reports.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output filename prefix. Defaults to the config stem.",
    )
    return parser


def append_heartbeat(path: Path, payload: Dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False)
    print(text, flush=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def percentile(values, q: float) -> float:
    if len(values) == 0:
        return 0.0
    values = sorted(values)
    idx = max(0, min(len(values) - 1, int(round((len(values) - 1) * q))))
    return float(values[idx])


def main() -> None:
    args = build_argparser().parse_args()
    cfg = OmegaConf.load(args.config)

    output_prefix = args.output_prefix or Path(args.config).stem
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_path = out_dir / f"{output_prefix}.heartbeat.jsonl"
    final_path = out_dir / f"{output_prefix}.json"

    # Fresh run: overwrite previous heartbeat log for easier tail -f inspection.
    heartbeat_path.write_text("", encoding="utf-8")
    if final_path.exists():
        final_path.unlink()

    dataset_sources, dataset_weights = collect_reweight_dataset_config(
        cfg.DATA.train.source,
        cfg.DATA.train.reweight,
    )

    batch_size = args.batch_size if args.batch_size is not None else int(cfg.TRAIN.sample_per_device)
    global_start = time.perf_counter()

    append_heartbeat(
        heartbeat_path,
        {
            "stage": "starting",
            "elapsed_s": 0.0,
            "config": args.config,
            "batch_size": batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "resampled": True,
            "shardshuffle": args.shardshuffle,
            "post_clip_shuffle": args.post_clip_shuffle,
            "seed": args.seed,
            "num_steps": args.num_steps,
            "num_datasets": len(dataset_sources),
            "dataset_counts": {name: len(files) for name, files in dataset_sources.items()},
        },
    )

    append_heartbeat(
        heartbeat_path,
        {
            "stage": "building_loader",
            "elapsed_s": round(time.perf_counter() - global_start, 4),
            "message": "Constructing reweight dataloader.",
        },
    )

    t0 = time.perf_counter()
    loader = get_dataset_reweight_dataloader(
        dataset_sources=dataset_sources,
        dataset_weights=dataset_weights,
        num_frames=cfg.MODEL.num_frame,
        stride=cfg.DATA.train.stride,
        batch_size=batch_size,
        num_workers=args.num_workers,
        prefetcher_factor=args.prefetch_factor,
        infinite=True,
        seed=args.seed,
        clip_sampling_mode=cfg.DATA.train.sampling.get("mode", "dense"),
        clips_per_sequence=cfg.DATA.train.sampling.get("clips_per_sequence", None),
        shardshuffle=args.shardshuffle,
        post_clip_shuffle=args.post_clip_shuffle,
        default_source_split=cfg.DATA.train.reweight.get("split", "train"),
    )
    t1 = time.perf_counter()

    append_heartbeat(
        heartbeat_path,
        {
            "stage": "loader_ready",
            "elapsed_s": round(time.perf_counter() - global_start, 4),
            "loader_build_s": round(t1 - t0, 4),
            "message": "Dataloader constructed. Waiting for the first batch.",
        },
    )

    it = iter(loader)
    wait_start = time.perf_counter()
    append_heartbeat(
        heartbeat_path,
        {
            "stage": "waiting_first_batch",
            "elapsed_s": round(time.perf_counter() - global_start, 4),
            "wait_started_at_s": round(wait_start - global_start, 4),
        },
    )

    batch = next(it)
    wait_end = time.perf_counter()
    first_batch_wait_s = wait_end - wait_start

    append_heartbeat(
        heartbeat_path,
        {
            "stage": "measuring_following_batches",
            "elapsed_s": round(time.perf_counter() - global_start, 4),
            "first_batch_wait_s": round(first_batch_wait_s, 4),
            "completed_steps": 1,
            "target_steps": args.num_steps,
        },
    )

    per_batch_wait_s = [first_batch_wait_s]
    step_start = wait_end
    for step_idx in range(1, max(1, args.num_steps)):
        batch = next(it)
        step_end = time.perf_counter()
        per_batch_wait_s.append(step_end - step_start)
        step_start = step_end
        if (step_idx + 1) % 8 == 0 or (step_idx + 1) == args.num_steps:
            append_heartbeat(
                heartbeat_path,
                {
                    "stage": "measuring_following_batches",
                    "elapsed_s": round(time.perf_counter() - global_start, 4),
                    "completed_steps": step_idx + 1,
                    "target_steps": args.num_steps,
                    "latest_batch_wait_s": round(per_batch_wait_s[-1], 4),
                },
            )

    payload = {
        "config": args.config,
        "mode": "reweight_training_batch_latency",
        "loader_build_s": round(t1 - t0, 4),
        "first_batch_wait_s": round(first_batch_wait_s, 4),
        "total_startup_to_first_batch_s": round(wait_end - global_start, 4),
        "num_steps": int(args.num_steps),
        "steps_total_s": round(sum(per_batch_wait_s), 4),
        "steady_batch_wait_s_avg": round(sum(per_batch_wait_s[1:]) / max(1, len(per_batch_wait_s[1:])), 4),
        "steady_batch_wait_s_p50": round(percentile(per_batch_wait_s[1:], 0.50), 4),
        "steady_batch_wait_s_p95": round(percentile(per_batch_wait_s[1:], 0.95), 4),
        "effective_config": {
            "batch_size": batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "num_frame": int(cfg.MODEL.num_frame),
            "sampling_mode": str(cfg.DATA.train.sampling.get("mode", "dense")),
            "clips_per_sequence": int(cfg.DATA.train.sampling.get("clips_per_sequence", 1)),
            "reweight_shardshuffle": args.shardshuffle,
            "reweight_post_clip_shuffle": args.post_clip_shuffle,
        },
        "batch": {
            "batch_size": len(batch["imgs"]),
            "data_sources": sorted(set(str(x) for x in batch["data_source"])),
        },
        "dataset_counts": {name: len(urls) for name, urls in dataset_sources.items()},
    }

    final_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    append_heartbeat(
        heartbeat_path,
        {
            "stage": "done",
            "elapsed_s": round(time.perf_counter() - global_start, 4),
            "final_report": str(final_path.resolve()),
            "loader_build_s": payload["loader_build_s"],
            "first_batch_wait_s": payload["first_batch_wait_s"],
            "total_startup_to_first_batch_s": payload["total_startup_to_first_batch_s"],
            "steady_batch_wait_s_avg": payload["steady_batch_wait_s_avg"],
            "steady_batch_wait_s_p95": payload["steady_batch_wait_s_p95"],
        },
    )


if __name__ == "__main__":
    main()
