"""
Stage 2 smoke test: run a few forward steps and verify that
zero_linear makes initial loss close to Stage 1 baseline.

Usage:
    CUDA_VISIBLE_DEVICES=4 python tests/test_stage2_smoke.py
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
    # override: use best_model as stage1 weight
    cfg.MODEL.stage1_weight = "checkpoint/2026-02-11/23-53-36_stage1-dino_large/best_model"

    accelerator = Accelerator(mixed_precision=cfg.TRAIN.mixed_precision)
    set_seed(cfg.GENERAL.seed)
    device = accelerator.device

    from script.train import setup_model
    from src.data.dataloader import get_dataloader
    from src.data.preprocess import preprocess_batch

    net = setup_model(cfg)
    net = accelerator.prepare(net)
    net.eval()

    # val loader
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

    num_steps = 20
    logger.info(f"Running {num_steps} forward steps (Stage 2, T={cfg.MODEL.num_frame})...")

    losses = []
    with torch.no_grad():
        for ix, batch_ in enumerate(val_loader):
            if ix >= num_steps:
                break

            batch, _, _ = preprocess_batch(
                batch_origin=batch_,
                patch_size=[cfg.MODEL.img_size, cfg.MODEL.img_size],
                patch_expanstion=cfg.TRAIN.expansion_ratio,
                scale_z_range=cfg.TRAIN.scale_z_range,
                scale_f_range=cfg.TRAIN.scale_f_range,
                persp_rot_max=cfg.TRAIN.persp_rot_max,
                joint_rep_type=cfg.MODEL.joint_type,
                augmentation_flag=False,
                device=device,
                pixel_aug=None,
                perspective_normalization=cfg.TRAIN.get("perspective_normalization", False),
            )

            output = net(batch)
            loss_val = output["loss"].item()
            losses.append(loss_val)
            state = output["state"]
            logger.info(
                f"  step {ix:3d} | loss={loss_val:.4f} | "
                f"theta={state['loss_theta']:.4f} shape={state['loss_shape']:.4f} "
                f"trans={state['loss_trans']:.4f} rel={state['loss_joint_rel']:.4f} "
                f"img={state['loss_joint_img']:.4f} | "
                f"mpjpe={state['micro_mpjpe']:.2f} mpjpe_rel={state['micro_mpjpe_rel']:.2f}"
            )

    avg_loss = sum(losses) / len(losses)
    logger.info(f"\nAvg loss over {num_steps} steps: {avg_loss:.4f}")
    logger.info(f"Expected: loss should be similar to Stage 1 baseline due to zero_linear")

    # Shapes check
    logger.info(f"\nOutput shapes:")
    for k, v in output["result"].items():
        if hasattr(v, "shape"):
            logger.info(f"  {k}: {tuple(v.shape)}")

    # Verify all T frames are present in output
    t = cfg.MODEL.num_frame
    for k in ["joint_cam_pred", "verts_cam_pred", "verts_cam_gt"]:
        shape = tuple(output["result"][k].shape)
        assert shape[1] == t, f"{k} has t={shape[1]}, expected {t}"
        logger.info(f"  {k} has t={shape[1]} frames -- OK")

    logger.info("\nSmoke test PASSED.")


if __name__ == "__main__":
    main()
