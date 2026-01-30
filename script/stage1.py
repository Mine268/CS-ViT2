from typing import *
import os
import os.path as osp
import shutil
import glob
import logging
from rich.logging import RichHandler
import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.utils import set_seed, broadcast_object_list
from accelerate.logging import get_logger

from aim import Run, Image

from src.data.dataloader import get_dataloader
from src.data.preprocess import preprocess_batch
from src.model.net import PoseNet
from src.utils.vis import vis
from src.utils.metric import *
from src.utils.train_utils import get_progressive_dropout


logger = get_logger(__name__)
save_dir = None


def manage_checkpoints(output_dir, keep_last_n=3):
    """只保留最近的 N 个 checkpoint"""
    ckpt_parent_dir = os.path.join(output_dir, "checkpoints")
    if not os.path.exists(ckpt_parent_dir):
        return

    # 获取所有 checkpoint 文件夹
    ckpts = [d for d in os.listdir(ckpt_parent_dir) if d.startswith("checkpoint-")]
    # 按步数排序 (假设格式为 checkpoint-1000)
    try:
        ckpts.sort(key=lambda x: int(x.split("-")[-1]))
    except ValueError:
        return # 格式不对就不管了

    if len(ckpts) > keep_last_n:
        # 删除旧的
        for ckpt_to_del in ckpts[:-keep_last_n]:
            path_to_del = os.path.join(ckpt_parent_dir, ckpt_to_del)
            if os.path.exists(path_to_del):
                shutil.rmtree(path_to_del)
                # print(f"Deleted old checkpoint: {path_to_del}")


def setup_dataloader(cfg: DictConfig):
    train_sources = []
    for src in cfg.DATA.train.source:
        matched_files = glob.glob(src)
        matched_files = sorted(matched_files)
        train_sources.extend(matched_files)
    train_loader = get_dataloader(
        url=train_sources,
        num_frames=cfg.MODEL.num_frame,
        stride=cfg.DATA.train.stride,
        batch_size=cfg.TRAIN.sample_per_device,
        num_workers=cfg.GENERAL.num_worker,
        prefetcher_factor=cfg.GENERAL.prefetch_factor,
        infinite=True,
    )
    logger.info(f"setup train loader: {train_sources}")

    val_sources = []
    for src in cfg.DATA.val.source:
        matched_files = glob.glob(src)
        matched_files = sorted(matched_files)
        val_sources.extend(matched_files)
    val_loader = get_dataloader(
        url=val_sources,
        num_frames=cfg.MODEL.num_frame,
        stride=cfg.DATA.val.stride,
        batch_size=cfg.TRAIN.sample_per_device,
        num_workers=1, # cfg.GENERAL.num_worker,
        prefetcher_factor=cfg.GENERAL.prefetch_factor,
        infinite=False,
        seed=cfg.GENERAL.get("val_seed", 42),  # 固定验证集seed确保一致性
    )
    logger.info(f"setup val loader: {val_sources}")

    return train_loader, val_loader


def setup_model(cfg: DictConfig):
    net = PoseNet(
        stage=cfg.MODEL.stage,

        backbone_str=cfg.MODEL.backbone.backbone_str,
        img_size=cfg.MODEL.img_size,
        img_mean=cfg.MODEL.img_mean,
        img_std=cfg.MODEL.img_std,
        infusion_feats_lyr=cfg.MODEL.backbone.infusion_layer,
        drop_cls=cfg.MODEL.backbone.drop_cls,
        backbone_kwargs=cfg.MODEL.backbone.get("kwargs"),

        num_handec_layer=cfg.MODEL.handec.num_layer,
        num_handec_head=cfg.MODEL.handec.num_head,
        ndim_handec_mlp=cfg.MODEL.handec.dim_mlp,
        ndim_handec_head=cfg.MODEL.handec.dim_head,
        prob_handec_dropout=cfg.MODEL.handec.dropout,
        prob_handec_emb_dropout=0.0,
        handec_emb_dropout_type="drop",
        handec_norm=cfg.MODEL.handec.norm,
        ndim_handec_norm_cond_dim=-1,
        ndim_handec_ctx=cfg.MODEL.handec.context_dim,
        handec_skip_token_embed=cfg.MODEL.handec.skip_token_embed,
        handec_mean_init=cfg.MODEL.handec.get("use_mean_init", True),
        handec_denorm_output=cfg.MODEL.handec.get("denorm_output", False),
        handec_heatmap_resulotion=cfg.MODEL.handec.get("heatmap_resolution", 1024),

        pie_type=cfg.MODEL.persp_info_embed.type,
        num_pie_sample=cfg.MODEL.persp_info_embed.num_sample,
        pie_fusion=cfg.MODEL.persp_info_embed.get("pie_fusion", "all"),

        num_temporal_head=cfg.MODEL.temporal_encoder.num_head,
        num_temporal_layer=cfg.MODEL.temporal_encoder.num_layer,
        trope_scalar=cfg.MODEL.temporal_encoder.trope_scalar,
        zero_linear=cfg.MODEL.temporal_encoder.zero_linear,

        joint_rep_type=cfg.MODEL.joint_type,

        supervise_global=cfg.LOSS.get("supervise_global", True),
        supervise_heatmap=cfg.LOSS.get("supervise_heatmap", True),
        lambda_theta=cfg.LOSS.get("lambda_theta", 2.81),
        lambda_shape=cfg.LOSS.get("lambda_shape", 1.38),
        lambda_trans=cfg.LOSS.get("lambda_trans", 0.123),
        lambda_rel=cfg.LOSS.get("lambda_rel", 0.000305),
        lambda_img=cfg.LOSS.get("lambda_img", 0.00512),
        hm_sigma=cfg.LOSS.get("heatmap_sigma", 3),

        freeze_backbone=cfg.TRAIN.backbone_lr is None,
        norm_by_hand=cfg.MODEL.get("norm_by_hand", False),
    )

    return net


def setup_optim(cfg: DictConfig, net: nn.Module):
    params = [
        {
            "params": filter(lambda p: p.requires_grad, net.get_regressor_params()),
            "lr": cfg.TRAIN.lr,
        },
    ]
    if cfg.TRAIN.backbone_lr is not None:
        params.append(
            {
                "params": filter(lambda p: p.requires_grad, net.get_backbone_params()),
                "lr": cfg.TRAIN.backbone_lr,
            }
        )

    optim = torch.optim.AdamW(
        params=params,
        weight_decay=cfg.TRAIN.weight_decay,
    )

    return optim


def setup_scheduler(cfg: DictConfig, optim: torch.optim.Optimizer):
    total_step = cfg.GENERAL.total_step
    num_warmup_step = cfg.GENERAL.warmup_step
    num_cycle = cfg.GENERAL.cosine_cycle

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optim,
        num_warmup_steps=num_warmup_step,
        num_training_steps=total_step,
        num_cycles=num_cycle,
    )

    return scheduler


@torch.inference_mode()
def val(
    cfg: DictConfig,
    accelerator: Accelerator,
    net: nn.Module,
    val_loader: Iterable,
    limit_step: Optional[int] = None,
    global_step: Optional[int] = None,
    aim_run: Optional[Run] = None,
):
    """
    多卡验证函数。
    核心思路：本地累加 Error 和 Count -> 全局 Reduce 求和 -> 计算平均值。
    """
    net.eval()
    device = accelerator.device

    metric_meter = StreamingMetricMeter()

    # 如果想要显示进度条 (仅主进程)
    # iter_wrapper = tqdm(val_loader, desc="Val") if accelerator.is_main_process else val_loader
    iter_wrapper = val_loader

    for ix, batch_ in enumerate(iter_wrapper):

        if limit_step is not None and ix >= limit_step:
            break

        batch, trans_2d_mat = preprocess_batch(
            batch_origin=batch_,
            patch_size=[cfg.MODEL.img_size, cfg.MODEL.img_size],
            patch_expanstion=cfg.TRAIN.expansion_ratio,
            scale_z_range=cfg.TRAIN.scale_z_range,
            scale_f_range=cfg.TRAIN.scale_f_range,
            persp_rot_max=cfg.TRAIN.persp_rot_max,
            joint_rep_type=cfg.MODEL.joint_type,
            augmentation_flag=False,
            device=device,
            pixel_aug=None  # 验证时不使用增强
        )

        output = net(batch)

        joint_cam_gt = batch["joint_cam"]
        joint_rel_gt = joint_cam_gt - joint_cam_gt[:, :, :1]
        verts_cam_gt = output["result"]["verts_cam_gt"]
        verts_rel_gt = verts_cam_gt - joint_cam_gt[:, :, :1]

        joint_cam_pred = output["result"]["joint_cam_pred"]
        joint_rel_pred = joint_cam_pred - joint_cam_pred[:, :, :1]
        verts_cam_pred = output["result"]["verts_cam_pred"]
        verts_rel_pred = verts_cam_pred - joint_cam_pred[:, :, :1]

        joint_valid = batch["joint_valid"]
        mano_valid = batch["mano_valid"]
        if "norm_idx" in output["result"]:
            norm_idx = output["result"]["norm_idx"]
            norm_valid = torch.all(batch["joint_valid"][:, :, norm_idx] > 0.5, dim=-1).float()
        else:
            norm_valid = torch.ones(joint_valid.shape[:2], device=joint_valid.device)

        # 计算指标
        metric_meter.update(
            joint_cam_gt,
            joint_rel_gt,
            verts_cam_gt,
            verts_rel_gt,
            joint_cam_pred,
            joint_rel_pred,
            verts_cam_pred,
            verts_rel_pred,
            mano_valid,
            joint_valid,
            norm_valid,
        )

        # 进行可视化
        if (
            accelerator.is_main_process
            and aim_run is not None
            and ix % max(100, cfg.GENERAL.vis_step // 10) == 0
        ):
            img_vis_np = vis(batch, trans_2d_mat, output["result"], 0)
            img_vis_aim = Image(img_vis_np, caption="gt/pred proj")

            aim_run.track(
                img_vis_aim,
                name="projection",
                step=global_step,
                context={"subset": "val"},
            )

    # ==========================================
    # 3. 关键同步代码：Pack -> Reduce -> Unpack
    # ==========================================

    # 定义需要同步的指标键值顺序 (必须固定顺序)
    keys_order = ["cs_mpjpe", "rs_mpjpe", "cs_mpvpe", "rs_mpvpe", "rte"]
    # 映射到输出的名称
    output_mapping = {
        "cs_mpjpe": "micro_mpjpe",
        "rs_mpjpe": "micro_mpjpe_rel",
        "cs_mpvpe": "micro_mpvpe",
        "rs_mpvpe": "micro_mpvpe_rel",
        "rte": "micro_rte",
    }

    # Step A: Pack (打包)
    # 从 Python 对象转为 Tensor，以便在 GPU 间传输
    # 5个指标 * 2个值(error, count) = 10个 float64
    local_stats = torch.zeros(len(keys_order) * 2, device=device, dtype=torch.float64)

    for i, key in enumerate(keys_order):
        # 从 metric_meter 取出累加好的 [error_sum, count_sum]
        err_sum, count_sum = metric_meter.accumulators[key]
        local_stats[2 * i] = err_sum
        local_stats[2 * i + 1] = count_sum

    # Step B: Reduce (归约)
    # 将所有 GPU 的 local_stats 相加
    global_stats = accelerator.reduce(local_stats, reduction="sum")

    # Step C: Unpack & Compute (解包并计算平均值)
    final_results = {}

    # 只需要在主进程或者所有进程都需要结果时计算
    # 这里让所有进程都计算一下，开销很小，且方便打印日志
    for i, key in enumerate(keys_order):
        total_err = global_stats[2 * i].item()
        total_cnt = global_stats[2 * i + 1].item()

        out_name = output_mapping[key]

        if total_cnt > 0:
            final_results[out_name] = total_err / total_cnt
        else:
            final_results[out_name] = 0.0

    # 4. 记录到 Aim (仅主进程)
    if aim_run is not None and accelerator.is_main_process:
        for k, v in final_results.items():
            aim_run.track(v, name=k, step=global_step, context={"subset": "val"})

    net.train()
    return final_results


def train(
    cfg: DictConfig,
    accelerator: Accelerator,
    net: nn.Module,
    optim: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    train_loader: Iterable,
    val_loader: Iterable,
    save_dir: str,
    start_step: int = 0,
    aim_run = None,
):
    # steps
    total_step: int = cfg.GENERAL.total_step
    log_step: int = cfg.GENERAL.log_step
    vis_step: int = cfg.GENERAL.vis_step
    checkpoint_step: int = cfg.GENERAL.checkpoint_step

    # deviec
    net.train()
    device = accelerator.device
    global_step = start_step

    # 创建数据增强对象（训练时使用）
    from src.data.preprocess import PixelLevelAugmentation
    from omegaconf import OmegaConf
    pixel_aug = None
    if cfg.TRAIN.get('augmentation', None) is not None:
        aug_config = OmegaConf.to_container(cfg.TRAIN.augmentation, resolve=True)
        pixel_aug = PixelLevelAugmentation(aug_config).to(device)
        pixel_aug.eval()  # 增强器始终在eval模式

    # start training
    data_iter = iter(train_loader)

    while global_step < total_step:
        # 0. 动态调整dropout率（渐进式策略）
        current_dropout = get_progressive_dropout(
            step=global_step,
            total_steps=total_step,
            warmup_steps=cfg.GENERAL.get("dropout_warmup_step", 10000),
            target_dropout=cfg.MODEL.handec.dropout
        )
        # 更新模型的dropout率
        unwrapped_net = net.module if hasattr(net, 'module') else net
        unwrapped_net.set_dropout_rate(current_dropout)

        # 1. 获取数据&增强
        batch_ = next(data_iter)
        batch, trans_2d_mat = preprocess_batch(
            batch_origin=batch_,
            patch_size=[cfg.MODEL.img_size, cfg.MODEL.img_size],
            patch_expanstion=cfg.TRAIN.expansion_ratio,
            scale_z_range=cfg.TRAIN.scale_z_range,
            scale_f_range=cfg.TRAIN.scale_f_range,
            persp_rot_max=cfg.TRAIN.persp_rot_max,
            joint_rep_type=cfg.MODEL.joint_type,
            augmentation_flag=True,
            device=device,
            pixel_aug=pixel_aug  # 传递增强对象
        )

        # 2. 计算loss
        with accelerator.accumulate(net):
            output = net(batch)
            loss = output["loss"]

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(net.parameters(), cfg.TRAIN.max_grad)

                optim.step()
                scheduler.step()
                optim.zero_grad()

                global_step += 1

                # 3. 保存模型
                if global_step % checkpoint_step == 0:
                    ckpt_path = osp.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
                    accelerator.save_state(ckpt_path)

                    if accelerator.is_main_process:
                        manage_checkpoints(save_dir, keep_last_n=3)
                        logger.info(f"Saved state to {ckpt_path}.")

                # 4. 验证集测试
                if global_step % checkpoint_step == 0:
                    logger.info("validating...")
                    val_result = val(
                        cfg,
                        accelerator,
                        net,
                        val_loader,
                        cfg.DATA.val.get("max_val_step", 1000),
                        global_step,
                        aim_run
                    )
                    logger.info(f"validation finished.")
                    for k, v in val_result.items():
                        logger.info(f"{k}={v}")

                # 5. 打印日志
                if global_step % log_step == 0:
                    state = output["state"]
                    fmt = f"{global_step}/{total_step}"

                    # 监控lr
                    current_lr = scheduler.get_last_lr()[0]
                    fmt += f" lr={current_lr:.4e}"

                    # 监控dropout率
                    fmt += f" dropout={current_dropout:.3f}"

                    # 监控loss组成
                    fmt += f" total={loss.cpu().item():.4f}"
                    for k, v in state.items():
                        fmt += f" {k}={v.cpu().item():.4f}"

                    if aim_run is not None and accelerator.is_main_process:
                        # 记录 Learning Rate
                        aim_run.track(
                            current_lr, name="lr", step=global_step, context={"subset": "train"}
                        )
                        # 记录 Dropout Rate
                        aim_run.track(
                            current_dropout,
                            name="dropout_rate",
                            step=global_step,
                            context={"subset": "train"},
                        )
                        # 记录 Total Loss
                        aim_run.track(
                            loss.item(),
                            name="loss_total",
                            step=global_step,
                            context={"subset": "train"},
                        )
                        # 记录 Loss 组件 (如 kps3d_loss, verts_loss 等)
                        for k, v in state.items():
                            aim_run.track(
                                v.item(), name=k, step=global_step, context={"subset": "train"}
                            )

                    logger.info(fmt)

                # 6. 可视化
                if (
                    accelerator.is_main_process
                    and aim_run is not None
                    and global_step % vis_step == 0
                ):
                    logger.info("visualizing the result to aim.")

                    img_vis_np = vis(batch, trans_2d_mat, output["result"], 0)
                    img_vis_aim = Image(img_vis_np, caption="gt/pred proj")

                    aim_run.track(
                        img_vis_aim,
                        name="projection",
                        step=global_step,
                        context={"subset": "train"},
                    )


@hydra.main(version_base=None, config_path="../config", config_name="default_stage1")
def main(cfg: DictConfig):
    # 1. 初始训练配置
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    timeout_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=1800))
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.TRAIN.grad_accum_step,
        mixed_precision=None,
        kwargs_handlers=[ddp_kwargs, timeout_kwargs]
    )

    log_format = "%(message)s"
    date_format = "[%X]"

    logging.basicConfig(
        format=log_format,
        datefmt=date_format,
        level=logging.INFO,
        handlers=[RichHandler(rich_tracebacks=True)],
        force=True,
    )

    save_dir_obj = [None]
    aim_run = None

    if accelerator.is_main_process:
        now = datetime.datetime.now()
        date_str = now.strftime("%d-%m-%Y")
        time_str = now.strftime("%H-%M-%S")

        try:
            config_name = HydraConfig.get().job.config_name
        except Exception:
            config_name = "debug"

        _save_dir = osp.join("checkpoint", date_str, f"{time_str}_{config_name}")
        os.makedirs(_save_dir, exist_ok=True)

        # B. 配置名为 file 的 Handler
        log_filename = osp.join(_save_dir, "log.txt")
        file_handler = logging.FileHandler(log_filename, mode="w")
        # 关键修改：手动创建 Formatter 并赋予 file_handler
        formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(formatter)
        # 将 Handler 添加到 root logger
        logging.getLogger().addHandler(file_handler)

        save_dir_obj[0] = _save_dir

        # Run看板
        aim_run = Run(
            experiment=f"{config_name}",
            repo=cfg.AIM.server_url,
        )
        aim_run["hparams"] = OmegaConf.to_container(cfg, resolve=True)
        logger.info(f'Aim run initialized in {cfg.AIM.server_url}')

    broadcast_object_list(save_dir_obj, from_process=0)
    save_dir = save_dir_obj[0]

    accelerator.wait_for_everyone()
    logger.info(accelerator.state, main_process_only=False)

    # 2. 配置种子
    set_seed(cfg.GENERAL.seed)

    # 3. 获取dataloader
    train_loader, val_loader = setup_dataloader(cfg)

    # 4. 获取模型
    net = setup_model(cfg)

    # 5. 优化器
    optim = setup_optim(cfg, net)
    scheduler = setup_scheduler(cfg, optim)

    # 6. accel, 不用处理dataloader
    net, optim, scheduler = accelerator.prepare(net, optim, scheduler)

    # 7. 训练
    start_step = 0
    resume_path = cfg.GENERAL.resume_path

    if resume_path is not None:
        accelerator.load_state(resume_path)
        logger.info(f"Resumed training from {resume_path}")

        # 解析步数
        try:
            # checkpoint-XXX
            step_str = os.path.basename(os.path.normpath(resume_path)).split("-")[-1]
            start_step = int(step_str)
        except ValueError:
            logger.warning("Warning: Could not parse step from checkpoint path, step count will be 0.")

    train(
        cfg=cfg,
        accelerator=accelerator,
        net=net,
        optim=optim,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=save_dir,
        start_step=start_step,
        aim_run=aim_run,
    )

    # close
    if accelerator.is_main_process and aim_run is not None:
        aim_run.close()


if __name__ == "__main__":
    main()
