from typing import *
import os
import os.path as osp
import shutil
import glob
import logging
import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed, broadcast_object_list
from accelerate.logging import get_logger

from aim import Run

from src.data.dataloader import get_dataloader
from src.data.preprocess import preprocess_batch
from src.model.net import PoseNet


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

        num_hf_layer=cfg.MODEL.hand_feat_extractor.num_layer,
        num_hf_head=cfg.MODEL.hand_feat_extractor.num_head,
        ndim_hf_mlp=cfg.MODEL.hand_feat_extractor.dim_mlp,
        ndim_hf_head=cfg.MODEL.hand_feat_extractor.dim_head,
        prob_hf_dropout=cfg.MODEL.hand_feat_extractor.dropout,
        prob_hf_emb_dropout=0.0,
        hf_emb_dropout_type="drop",
        hf_norm=cfg.MODEL.hand_feat_extractor.norm,
        ndim_hf_norm_cond_dim=-1,
        ndim_hf_ctx=cfg.MODEL.hand_feat_extractor.context_dim,
        hf_skip_token_embed=cfg.MODEL.hand_feat_extractor.skip_token_embed,

        pie_type=cfg.MODEL.persp_info_embed.type,
        num_pie_sample=cfg.MODEL.persp_info_embed.num_sample,
        pie_fusion=cfg.MODEL.persp_info_embed.get("pie_fusion", "all"),

        num_temporal_head=cfg.MODEL.temporal_encoder.num_head,
        num_temporal_layer=cfg.MODEL.temporal_encoder.num_layer,
        trope_scalar=cfg.MODEL.temporal_encoder.trope_scalar,
        zero_linear=cfg.MODEL.temporal_encoder.zero_linear,

        detok_joint_type=cfg.MODEL.detokenizer.joint_type,

        kps3d_loss_type=cfg.LOSS.kps3d_loss_type,
        verts_loss_type=cfg.LOSS.verts_loss_type,
    )

    return net


def setup_optim(cfg: DictConfig, net: nn.Module):
    optim = torch.optim.AdamW(
        params=[
            {
                "params": filter(lambda p: p.requires_grad, net.get_regressor_params()),
                "lr": cfg.TRAIN.lr,
            },
            {
                "params": filter(lambda p: p.requires_grad, net.get_backbone_params()),
                "lr": cfg.TRAIN.backbone_lr,
            }
        ],
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
):
    """
    多卡验证函数。
    核心思路：本地累加 Error 和 Count -> 全局 Reduce 求和 -> 计算平均值。
    """
    net.eval()
    device = accelerator.device

    # 1. 初始化累计器 (Total Error, Total Count)
    # 使用 float64 防止溢出，虽然 float32 通常也够
    metrics_accum = torch.zeros(4, device=device, dtype=torch.float64)
    # indices: [0]=joint_err_sum, [1]=joint_count, [2]=verts_err_sum, [3]=verts_count

    # 如果想要显示进度条 (仅主进程)
    # iter_wrapper = tqdm(val_loader, desc="Val") if accelerator.is_main_process else val_loader
    iter_wrapper = val_loader

    for ix, batch_ in enumerate(iter_wrapper):
        batch, _ = preprocess_batch(
            batch_origin=batch_,
            patch_size=[cfg.MODEL.img_size, cfg.MODEL.img_size],
            patch_expanstion=cfg.TRAIN.expansion_ratio,
            scale_z_range=cfg.TRAIN.scale_z_range,
            scale_f_range=cfg.TRAIN.scale_f_range,
            persp_rot_max=cfg.TRAIN.persp_rot_max,
            augmentation_flag=False,
            device=device
        )

        output = net(batch)

        # 获取预测值和真值
        joint_cam_pred = output["result"]["joint_cam_pred"] # [B, T, J, 3]
        verts_cam_pred = output["result"]["verts_cam_pred"] # [B, T, V, 3]

        joint_cam_gt = batch["joint_cam"]
        verts_cam_gt = output["result"]["verts_cam_gt"]

        # Mask: joint_valid [B, T, J], mano_valid [B, T]
        joint_valid = batch["joint_valid"]
        mano_valid = batch["mano_valid"]

        # --- A. 计算 MPJPE (Mean Per Joint Position Error) ---
        # error per joint: [B, T, J]
        joint_error = torch.sqrt(torch.sum((joint_cam_pred - joint_cam_gt) ** 2, dim=-1))

        # 应用 Mask
        mask_j = joint_valid > 0.5
        if mask_j.any():
            metrics_accum[0] += joint_error[mask_j].sum()
            metrics_accum[1] += mask_j.sum()

        # --- B. 计算 MPVPE (Mean Per Vertex Position Error) ---
        # error per vertex: [B, T, V]
        if verts_cam_pred is not None and verts_cam_gt is not None:
            verts_error = torch.sqrt(torch.sum((verts_cam_pred - verts_cam_gt) ** 2, dim=-1))

            # mano_valid 通常是整手有效性 [B, T]，需要扩展到 [B, T, V] 或者直接索引
            # 这里简单处理：找出有效的 (b,t) 索引，对这些帧的所有点求和
            mask_v = mano_valid > 0.5 # [B, T]
            if mask_v.any():
                # verts_error[mask_v] 得到 [N_valid_frames, V]
                metrics_accum[2] += verts_error[mask_v].sum()
                metrics_accum[3] += verts_error[mask_v].numel()

    # ==========================================
    # 3. 关键同步代码：Reduce
    # ==========================================
    # 将所有 GPU 的累加值相加
    # reduction="sum" 表示所有进程的值加在一起，结果广播回所有进程
    metrics_sum = accelerator.reduce(metrics_accum, reduction="sum")

    # 4. 计算最终平均值 (转回 mm)
    final_results = {}

    # MPJPE
    total_j_err, total_j_cnt = metrics_sum[0].item(), metrics_sum[1].item()
    if total_j_cnt > 0:
        final_results["mpjpe"] = total_j_err / total_j_cnt
    else:
        final_results["mpjpe"] = 0.0

    # MPVPE
    total_v_err, total_v_cnt = metrics_sum[2].item(), metrics_sum[3].item()
    if total_v_cnt > 0:
        final_results["mpvpe"] = total_v_err / total_v_cnt
    else:
        final_results["mpvpe"] = 0.0

    net.train() # 恢复训练模式
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
    checkpoint_step: int = cfg.GENERAL.checkpoint_step

    # deviec
    net.train()
    device = accelerator.device
    global_step = start_step

    # start training
    data_iter = iter(train_loader)

    while global_step < total_step:
        # 1. 获取数据&增强
        batch_ = next(data_iter)
        batch, trans_2d_mat = preprocess_batch(
            batch_origin=batch_,
            patch_size=[cfg.MODEL.img_size, cfg.MODEL.img_size],
            patch_expanstion=cfg.TRAIN.expansion_ratio,
            scale_z_range=cfg.TRAIN.scale_z_range,
            scale_f_range=cfg.TRAIN.scale_f_range,
            persp_rot_max=cfg.TRAIN.persp_rot_max,
            augmentation_flag=True,
            device=device
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
                    val_result = val(cfg, accelerator, net, val_loader)
                    logger.info(f"validation finished, mpjpe={val_result['mpjpe']}, "
                        f"mpvpe={val_result["mpvpe"]}")

                    if aim_run is not None and accelerator.is_main_process:
                        for k, v in val_result.items():
                            aim_run.track(v, name=k, step=global_step, context={"subset": "val"})

                # 5. 打印日志
                if global_step % log_step == 0:
                    state = output["state"]
                    fmt = f"{global_step}/{total_step}"

                    # 监控lr
                    current_lr = scheduler.get_last_lr()[0]
                    fmt += f" lr={current_lr:.4e}"

                    # 监控loss组成
                    fmt += f" total={loss.cpu().item():.4f}"
                    for k, v in state.items():
                        fmt += f" {k}={v.cpu().item():.4f}"

                    if aim_run is not None and accelerator.is_main_process:
                        # 记录 Learning Rate
                        aim_run.track(
                            current_lr, name="lr", step=global_step, context={"subset": "train"}
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
                if global_step % log_step == 0:
                    pass


@hydra.main(version_base=None, config_path="../config", config_name="default_stage1")
def main(cfg: DictConfig):
    # 1. 初始训练配置
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.TRAIN.grad_accum_step,
        mixed_precision=None,
        kwargs_handlers=[ddp_kwargs]
    )

    log_format = "[%(asctime)s][%(levelname)s][%(name)s] %(message)s"
    date_format = "%m/%d/%Y %H:%M:%S"

    logging.basicConfig(
        format=log_format,
        datefmt=date_format,
        level=logging.INFO,
        force=True
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
            experiment=f"{date_str}_{time_str}_{config_name}",
            repo=hydra.utils.get_original_cwd(),
        )
        aim_run["hparams"] = OmegaConf.to_container(cfg, resolve=True)
        logger.info(f"Aim run initialized in {_save_dir}")

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
