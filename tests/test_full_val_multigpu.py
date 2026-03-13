import argparse
import datetime
import json
import logging
import os
import os.path as osp
import sys

project_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, project_root)

from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from omegaconf import OmegaConf

from script.train import setup_dataloader, setup_model, val


logger = get_logger(__name__)


def build_argparser():
    parser = argparse.ArgumentParser(description="Run full validation with multi-GPU using current train.py val logic")
    parser.add_argument("--config-path", type=str, required=True, help="Path to saved training config yaml")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Checkpoint directory to load")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    return parser


def main():
    args = build_argparser().parse_args()

    cfg = OmegaConf.load(args.config_path)
    cfg.AIM.server_url = ""
    cfg.DATA.val.full_eval = True

    timeout_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=1800))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=cfg.TRAIN.get("mixed_precision", None),
        kwargs_handlers=[ddp_kwargs, timeout_kwargs],
    )

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    logger.info(accelerator.state, main_process_only=False)

    set_seed(cfg.GENERAL.get("val_seed", 42))

    _, val_loader = setup_dataloader(cfg, accelerator)
    net = setup_model(cfg)
    net = accelerator.prepare(net)
    accelerator.load_state(args.checkpoint_path, strict=False)
    net.eval()

    val_metrics = val(
        cfg=cfg,
        accelerator=accelerator,
        net=net,
        val_loader=val_loader,
        limit_step=None,
        global_step=0,
        aim_run=None,
    )

    if accelerator.is_main_process:
        logger.info("Full validation metrics:")
        for key, value in val_metrics.items():
            logger.info(f"  {key}: {value:.6f}")

        if args.output is not None:
            os.makedirs(osp.dirname(args.output), exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(val_metrics, f, indent=2)
            logger.info(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
