"""
Regression test: load Stage 1 best checkpoint and verify metrics match.
Ensures the full-frame supervision refactor doesn't break Stage 1 (T=1).

Usage:
    CUDA_VISIBLE_DEVICES=3 python tests/test_checkpoint_regression.py
"""
import glob as glob_mod
import json
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from src.model.net import PoseNet
from src.data.dataloader import get_dataloader
from src.data.preprocess import preprocess_batch
from src.utils.metric import StreamingMetricMeter

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CKPT_DIR = "checkpoint/2026-02-11/23-53-36_stage1-dino_large"
BEST_MODEL_DIR = os.path.join(CKPT_DIR, "best_model")
CONFIG_PATH = os.path.join(CKPT_DIR, "config_stage1-dino_large.yaml")
BEST_INFO_PATH = os.path.join(CKPT_DIR, "best_model_info.json")


def main():
    # 1. load config & expected metrics
    cfg = OmegaConf.load(CONFIG_PATH)
    with open(BEST_INFO_PATH) as f:
        expected = json.load(f)

    logger.info(f"Expected metrics from best_model_info.json:")
    for k in ["micro_mpjpe", "micro_mpjpe_rel", "micro_mpvpe", "micro_mpvpe_rel", "micro_rte"]:
        logger.info(f"  {k}: {expected[k]:.4f}")

    # 2. setup
    accelerator = Accelerator(mixed_precision=cfg.TRAIN.mixed_precision)
    set_seed(cfg.GENERAL.val_seed)
    device = accelerator.device

    # 3. model
    from script.train import setup_model
    net = setup_model(cfg)

    # 4. prepare & load
    net = accelerator.prepare(net)
    accelerator.load_state(BEST_MODEL_DIR, strict=False)
    net.eval()

    # 5. val dataloader
    val_sources = []
    for src in cfg.DATA.val.source:
        val_sources.extend(sorted(glob_mod.glob(src)))
    logger.info(f"Val sources: {len(val_sources)} files")

    val_loader = get_dataloader(
        url=val_sources,
        num_frames=cfg.MODEL.get("num_frame", 1),
        stride=cfg.DATA.val.stride,
        batch_size=cfg.TRAIN.sample_per_device,
        num_workers=1,
        prefetcher_factor=cfg.GENERAL.prefetch_factor,
        infinite=False,
        seed=cfg.GENERAL.get("val_seed", 42),
    )

    max_val_step = cfg.DATA.val.get("max_val_step", 1000)
    metric_meter = StreamingMetricMeter()

    # 6. eval loop
    logger.info(f"Running val for {max_val_step} steps...")
    with torch.no_grad():
        for ix, batch_ in enumerate(val_loader):
            if ix >= max_val_step:
                break

            batch, trans_2d_mat, _ = preprocess_batch(
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

            joint_cam_gt = batch["joint_cam"][:, -1:]
            joint_rel_gt = joint_cam_gt - joint_cam_gt[:, :, :1]
            verts_cam_gt = output["result"]["verts_cam_gt"][:, -1:]
            verts_rel_gt = verts_cam_gt - joint_cam_gt[:, :, :1]

            joint_cam_pred = output["result"]["joint_cam_pred"][:, -1:]
            joint_rel_pred = joint_cam_pred - joint_cam_pred[:, :, :1]
            verts_cam_pred = output["result"]["verts_cam_pred"][:, -1:]
            verts_rel_pred = verts_cam_pred - joint_cam_pred[:, :, :1]

            joint_valid = batch["joint_valid"][:, -1:]
            mano_valid = batch["mano_valid"][:, -1:]
            if "norm_idx" in output["result"]:
                norm_idx = output["result"]["norm_idx"]
                norm_valid = torch.all(batch["joint_valid"][:, -1:, norm_idx] > 0.5, dim=-1).float()
            else:
                norm_valid = torch.ones(joint_valid.shape[:2], device=joint_valid.device)

            metric_meter.update(
                joint_cam_gt, joint_rel_gt,
                verts_cam_gt, verts_rel_gt,
                joint_cam_pred, joint_rel_pred,
                verts_cam_pred, verts_rel_pred,
                mano_valid, joint_valid, norm_valid,
            )

            if ix % 100 == 0:
                logger.info(f"  step {ix}/{max_val_step}")

    # 7. compute & compare
    results = metric_meter.compute()
    logger.info(f"\nActual metrics:")
    for k, v in results.items():
        logger.info(f"  {k}: {v:.4f}")

    logger.info(f"\nComparison (expected vs actual):")
    all_pass = True
    tolerance = 5.0  # mm, accounts for single-GPU vs multi-GPU data distribution diff
    for key in ["micro_mpjpe", "micro_mpjpe_rel", "micro_mpvpe", "micro_mpvpe_rel", "micro_rte"]:
        exp_val = expected[key]
        # metric_meter uses different key names
        meter_key = key.replace("micro_", "")
        if meter_key not in results:
            # try original key
            meter_key = key
        act_val = results.get(meter_key, results.get(key, float("nan")))
        diff = abs(act_val - exp_val)
        status = "PASS" if diff < tolerance else "FAIL"
        if status == "FAIL":
            all_pass = False
        logger.info(f"  {key}: expected={exp_val:.4f}, actual={act_val:.4f}, diff={diff:.4f} [{status}]")

    if all_pass:
        logger.info("\nAll metrics match within tolerance. Regression test PASSED.")
    else:
        logger.info(f"\nSome metrics differ by more than {tolerance}mm. Check details above.")

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
