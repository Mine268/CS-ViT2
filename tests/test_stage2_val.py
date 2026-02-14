"""
Test that val() function works correctly with Stage 2 full-frame output.

Usage:
    CUDA_VISIBLE_DEVICES=4 python tests/test_stage2_val.py
"""
import glob as glob_mod
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    cfg = OmegaConf.load("config/stage2-dino_large.yaml")
    cfg.MODEL.stage1_weight = "checkpoint/2026-02-11/23-53-36_stage1-dino_large/best_model"

    accelerator = Accelerator(mixed_precision=cfg.TRAIN.mixed_precision)
    set_seed(cfg.GENERAL.seed)

    from script.train import setup_model, val
    from src.data.dataloader import get_dataloader

    net = setup_model(cfg)
    net = accelerator.prepare(net)

    val_sources = []
    for src in cfg.DATA.val.source:
        val_sources.extend(sorted(glob_mod.glob(src)))
    logger.info(f"Val sources: {len(val_sources)} files")

    val_loader = get_dataloader(
        url=val_sources,
        num_frames=cfg.MODEL.num_frame,
        stride=cfg.DATA.val.stride,
        batch_size=cfg.TRAIN.sample_per_device,
        num_workers=1,
        prefetcher_factor=cfg.GENERAL.prefetch_factor,
        infinite=False,
        seed=cfg.GENERAL.get("val_seed", 42),
    )

    # Run val with limit_step=50
    logger.info("Running val() with limit_step=50...")
    val_metrics = val(
        cfg=cfg,
        accelerator=accelerator,
        net=net,
        val_loader=val_loader,
        limit_step=50,
        global_step=0,
        aim_run=None,
    )

    logger.info(f"\nVal returned: {val_metrics}")
    logger.info("val() completed without error. PASSED.")


if __name__ == "__main__":
    main()
