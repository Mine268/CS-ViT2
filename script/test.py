from typing import *
import os
import os.path as osp
import glob
import logging
from rich.logging import RichHandler
import datetime
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import numpy as np
import h5py
import torch
import torch.nn as nn
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed
from accelerate.logging import get_logger

from aim import Run, Image

from src.data.dataloader import get_dataloader
from src.data.dataloader import (
    estimate_wds_shard_clip_counts,
    build_balanced_clip_segments,
    get_segmented_wds_dataloader,
)
from src.data.preprocess import preprocess_batch
from src.model.net import PoseNet
from src.utils.vis import vis
from src.utils.metric import *


logger = get_logger(__name__)


def setup_test_dataloader(cfg: DictConfig, accelerator: Optional[Accelerator] = None):
    """设置测试数据加载器

    Args:
        cfg: Hydra 配置对象

    Returns:
        test_loader: WebDataset DataLoader

    Raises:
        ValueError: 如果测试数据源为空
    """
    test_sources = []
    for src in cfg.DATA.test.source:
        matched_files = glob.glob(src)
        matched_files = sorted(matched_files)
        test_sources.extend(matched_files)

    if len(test_sources) == 0:
        raise ValueError(
            "No test data found. Please specify test dataset via: "
            "DATA.test.source='[/path/to/test/data/*]'"
        )

    test_sampling_cfg = cfg.DATA.test.get("sampling", {})
    if cfg.DATA.test.get("full_eval", False):
        if accelerator is None:
            raise ValueError("full_eval test requires accelerator for shard partitioning")
        shard_clip_counts_obj = [None]
        if accelerator.is_main_process:
            shard_clip_counts_obj[0] = estimate_wds_shard_clip_counts(
                urls=test_sources,
                num_frames=cfg.MODEL.num_frame,
                stride=cfg.DATA.test.stride,
            )
        from accelerate.utils import broadcast_object_list
        broadcast_object_list(shard_clip_counts_obj, from_process=0)
        shard_clip_counts = shard_clip_counts_obj[0]
        rank_segments = build_balanced_clip_segments(
            urls=test_sources,
            clip_counts=shard_clip_counts,
            num_parts=accelerator.num_processes,
        )
        process_segments = rank_segments[accelerator.process_index]
        test_loader = get_segmented_wds_dataloader(
            segments=process_segments,
            num_frames=cfg.MODEL.num_frame,
            stride=cfg.DATA.test.stride,
            batch_size=cfg.TEST.batch_size,
            num_workers=0,
            prefetcher_factor=1,
        )
        logger.info(
            f"Setup full-eval test loader: process={accelerator.process_index}/{accelerator.num_processes} "
            f"segments={len(process_segments)} total_shards={len(test_sources)} "
            f"clip_counts={shard_clip_counts}"
        )
    else:
        test_loader = get_dataloader(
            url=test_sources,
            num_frames=cfg.MODEL.num_frame,
            stride=cfg.DATA.test.stride,
            batch_size=cfg.TEST.batch_size,
            num_workers=1,              # 单 worker 避免 tar 文件少的问题
            prefetcher_factor=1,
            infinite=False,             # 单次遍历
            seed=42,                    # 固定种子
            clip_sampling_mode=test_sampling_cfg.get("mode", "dense"),
            clips_per_sequence=test_sampling_cfg.get("clips_per_sequence", None),
            shardshuffle=test_sampling_cfg.get("shardshuffle", False),
            post_clip_shuffle=test_sampling_cfg.get("post_clip_shuffle", 200),
        )
        logger.info(f"Setup test loader: {test_sources}")

    return test_loader


def setup_model(cfg: DictConfig):
    """初始化模型结构（不加载权重）"""
    net = PoseNet(
        stage=cfg.MODEL.stage,
        stage1_weight_path=None,  # 测试时不从 stage1 加载权重

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
        reproj_loss_type=cfg.LOSS.get("reproj_loss_type", "l1"),
        reproj_loss_delta=cfg.LOSS.get("reproj_loss_delta", 84.0),
    )

    return net


@torch.inference_mode()
def test(
    cfg: DictConfig,
    accelerator: Accelerator,
    net: nn.Module,
    test_loader: Iterable,
    aim_run: Optional[Run] = None,
):
    """
    运行测试并收集预测结果

    Returns:
        local_results: 本地收集的结果字典
    """
    net.eval()
    device = accelerator.device

    local_results = {
        "sample_key": [],
        "imgs_path": [],
        "handedness": [],
        "data_source": [],
        "source_split": [],
        "intr_type": [],
        "source_index": [],
        "joint_cam_pred": [],
        "vert_cam_pred": [],
        "mano_pose_pred": [],
        "mano_shape_pred": [],
        "trans_pred": [],
        "trans_pred_denorm": [],
        "norm_scale": [],
        "norm_valid": [],
        "joint_cam_gt": [],
        "vert_cam_gt": [],
        "mano_pose_gt": [],
        "mano_shape_gt": [],
        "focal": [],
        "princpt": [],
        "hand_bbox": [],
        "joint_2d_valid": [],
        "joint_3d_valid": [],
        "has_mano": [],
        "has_intr": [],
    }

    total_samples = 0
    max_samples = cfg.TEST.max_samples

    for batch_idx, batch_ in enumerate(test_loader):
        if max_samples is not None and total_samples >= max_samples:
            break

        batch, trans_2d_mat, _ = preprocess_batch(
            batch_origin=batch_,
            patch_size=[cfg.MODEL.img_size, cfg.MODEL.img_size],
            patch_expanstion=cfg.TRAIN.expansion_ratio,
            scale_z_range=[1.0, 1.0],
            scale_f_range=[1.0, 1.0],
            persp_rot_max=0.0,
            joint_rep_type=cfg.MODEL.joint_type,
            augmentation_flag=False,
            device=device,
            pixel_aug=None,
            perspective_normalization=cfg.TRAIN.get("perspective_normalization", False),
        )

        unwrapped_net = net.module if hasattr(net, 'module') else net
        result = unwrapped_net.predict_full(
            img=batch["patches"],
            bbox=batch["patch_bbox"],
            focal=batch["focal"],
            princpt=batch["princpt"],
            timestamp=batch["timestamp"],
            joint_cam_gt=batch["joint_cam"],
            joint_3d_valid_gt=batch["joint_3d_valid"],
        )

        with torch.no_grad():
            _, verts_rel_gt = unwrapped_net.mano_to_pose(
                batch["mano_pose"][:, -1:],
                batch["mano_shape"][:, -1:],
            )
            verts_cam_gt = verts_rel_gt + batch["joint_cam"][:, -1:, :1]

        batch_size = batch["patches"].shape[0]
        for i in range(batch_size):
            if max_samples is not None and total_samples >= max_samples:
                break

            local_results["joint_cam_pred"].append(result["joint_cam_pred"][i, 0].cpu().numpy())
            local_results["vert_cam_pred"].append(result["vert_cam_pred"][i, 0].cpu().numpy())
            local_results["mano_pose_pred"].append(result["mano_pose_pred"][i, 0].cpu().numpy())
            local_results["mano_shape_pred"].append(result["mano_shape_pred"][i, 0].cpu().numpy())
            local_results["trans_pred"].append(result["trans_pred"][i, 0].cpu().numpy())
            local_results["trans_pred_denorm"].append(result["trans_pred_denorm"][i, 0].cpu().numpy())
            local_results["norm_scale"].append(result["norm_scale"][i, 0].cpu().numpy())
            local_results["norm_valid"].append(result["norm_valid"][i, 0].cpu().numpy())

            local_results["joint_cam_gt"].append(batch["joint_cam"][i, -1].cpu().numpy())
            local_results["vert_cam_gt"].append(verts_cam_gt[i, 0].cpu().numpy())
            local_results["mano_pose_gt"].append(batch["mano_pose"][i, -1].cpu().numpy())
            local_results["mano_shape_gt"].append(batch["mano_shape"][i, -1].cpu().numpy())

            local_results["focal"].append(batch["focal"][i, -1].cpu().numpy())
            local_results["princpt"].append(batch["princpt"][i, -1].cpu().numpy())
            local_results["hand_bbox"].append(batch["hand_bbox"][i, -1].cpu().numpy())
            local_results["joint_2d_valid"].append(batch["joint_2d_valid"][i, -1].cpu().numpy())
            local_results["joint_3d_valid"].append(batch["joint_3d_valid"][i, -1].cpu().numpy())
            local_results["has_mano"].append(batch["has_mano"][i, -1].cpu().numpy())
            local_results["has_intr"].append(batch["has_intr"][i, -1].cpu().numpy())

            local_results["sample_key"].append(str(batch["__key__"][i]))
            imgs_path = batch["imgs_path"][i]
            local_results["imgs_path"].append(imgs_path[-1] if isinstance(imgs_path, list) else str(imgs_path))
            local_results["handedness"].append(str(batch["handedness"][i]))
            local_results["data_source"].append(str(batch["data_source"][i]))
            local_results["source_split"].append(str(batch["source_split"][i]))
            local_results["intr_type"].append(str(batch["intr_type"][i]))
            source_index = batch["source_index"][i]
            if isinstance(source_index, list):
                source_index = source_index[-1]
            local_results["source_index"].append(json.dumps(source_index, ensure_ascii=False, sort_keys=True))

            total_samples += 1

        if (
            accelerator.is_main_process
            and aim_run is not None
            and cfg.TEST.enable_vis
            and batch_idx % cfg.TEST.vis_step == 0
        ):
            img_vis_np = vis(batch, trans_2d_mat, result, tx=0)
            img_vis_aim = Image(img_vis_np, caption="test: gt/pred proj")
            aim_run.track(
                img_vis_aim,
                name="test_vis",
                step=batch_idx,
                context={"subset": "test"},
            )

        if accelerator.is_main_process and (batch_idx + 1) % 10 == 0:
            logger.info(f"Processed {batch_idx + 1} batches, {total_samples} samples")

    logger.info(f"Local process collected {total_samples} samples")
    return local_results


def merge_results(accelerator: Accelerator, local_results: dict):
    """
    合并多卡的结果

    Args:
        accelerator: Accelerate 分布式管理器
        local_results: 本地结果字典

    Returns:
        merged_results: 合并后的结果字典（只在主进程有效）
    """
    device = accelerator.device
    merged_results = {}

    tensor_keys = [
        "joint_cam_pred", "vert_cam_pred", "mano_pose_pred", "mano_shape_pred",
        "trans_pred", "trans_pred_denorm", "norm_scale", "norm_valid",
        "joint_cam_gt", "vert_cam_gt", "mano_pose_gt", "mano_shape_gt",
        "focal", "princpt", "hand_bbox", "joint_2d_valid", "joint_3d_valid",
        "has_mano", "has_intr",
    ]

    for key in tensor_keys:
        if len(local_results[key]) > 0:
            local_tensor = torch.from_numpy(np.stack(local_results[key])).to(device)
            gathered_tensor = accelerator.gather_for_metrics(local_tensor)
            if accelerator.is_main_process:
                merged_results[key] = gathered_tensor.cpu().numpy()
        else:
            if accelerator.is_main_process:
                merged_results[key] = np.array([])

    string_keys = [
        "sample_key", "imgs_path", "handedness", "data_source",
        "source_split", "intr_type", "source_index",
    ]
    for key in string_keys:
        gathered_list = accelerator.gather_for_metrics(local_results[key])
        if accelerator.is_main_process:
            merged_results[key] = gathered_list

    return merged_results


def _flatten_string_values(values):
    flattened = []
    for value in values:
        if isinstance(value, list):
            flattened.append(value[0] if len(value) > 0 else "")
        else:
            flattened.append(str(value))
    return np.array(flattened, dtype=h5py.string_dtype(encoding='utf-8'))


def save_results_hdf5(output_path: str, results: dict, cfg: DictConfig):
    """
    保存结果到 HDF5 文件

    Args:
        output_path: HDF5 文件路径
        results: 结果字典
        cfg: 配置对象
    """
    os.makedirs(osp.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        samples_group = f.create_group("samples")
        compression = cfg.TEST.compression if cfg.TEST.compression else None

        for key in [
            "sample_key", "imgs_path", "handedness", "data_source",
            "source_split", "intr_type", "source_index",
        ]:
            samples_group.create_dataset(key, data=_flatten_string_values(results[key]))

        for key in [
            "joint_cam_pred", "vert_cam_pred", "mano_pose_pred", "mano_shape_pred",
            "trans_pred", "trans_pred_denorm", "joint_cam_gt", "vert_cam_gt",
            "mano_pose_gt", "mano_shape_gt",
        ]:
            samples_group.create_dataset(
                key,
                data=results[key].astype(np.float32),
                compression=compression,
            )

        for key in [
            "norm_scale", "norm_valid", "focal", "princpt", "hand_bbox",
            "joint_2d_valid", "joint_3d_valid", "has_mano", "has_intr",
        ]:
            samples_group.create_dataset(key, data=results[key].astype(np.float32))

        metadata_group = f.create_group("metadata")
        metadata_group.attrs["num_samples"] = len(results["imgs_path"])
        metadata_group.attrs["timestamp"] = datetime.datetime.now().isoformat()
        metadata_group.attrs["norm_by_hand"] = cfg.MODEL.norm_by_hand
        metadata_group.attrs["config"] = json.dumps(
            OmegaConf.to_container(cfg, resolve=True)
        )

    logger.info(f"Saved results to {output_path}")


def compute_quick_metrics(results: dict, output_dir: str):
    """
    计算快速摘要指标并保存

    Args:
        results: 结果字典
        output_dir: 输出目录

    Returns:
        metrics: 指标字典
    """
    joint_cam_pred = results["joint_cam_pred"]
    joint_cam_gt = results["joint_cam_gt"]
    vert_cam_pred = results["vert_cam_pred"]
    vert_cam_gt = results["vert_cam_gt"]
    joint_3d_valid = results["joint_3d_valid"]
    has_mano = results["has_mano"]

    joint_valid_count = float(np.sum(joint_3d_valid))
    mano_valid_count = float(np.sum(has_mano))

    if joint_valid_count > 0:
        joint_diff = np.linalg.norm(joint_cam_pred - joint_cam_gt, axis=-1)
        joint_diff_masked = joint_diff * joint_3d_valid
        mpjpe = float(np.sum(joint_diff_masked) / joint_valid_count)
    else:
        mpjpe = float('nan')

    if mano_valid_count > 0:
        vert_diff = np.linalg.norm(vert_cam_pred - vert_cam_gt, axis=-1)
        vert_diff_masked = vert_diff * has_mano[:, None]
        mpvpe = float(np.sum(vert_diff_masked) / (mano_valid_count * 778.0))
    else:
        mpvpe = float('nan')

    metrics = {
        "mpjpe": mpjpe,
        "mpvpe": mpvpe,
        "num_samples": int(len(joint_cam_pred)),
        "num_valid_joints": int(joint_valid_count),
        "num_valid_hands": int(mano_valid_count),
    }

    metrics_path = osp.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics: MPJPE={mpjpe}, MPVPE={mpvpe}")
    logger.info(f"Saved metrics to {metrics_path}")

    return metrics


@hydra.main(version_base=None, config_path="../config", config_name="stage1-dino_large")
def main(cfg: DictConfig):
    # 1. 检查必需参数
    if cfg.TEST.checkpoint_path is None:
        raise ValueError(
            "Must specify checkpoint path via: "
            "TEST.checkpoint_path=path/to/checkpoint"
        )

    if not cfg.DATA.test.source or len(cfg.DATA.test.source) == 0:
        raise ValueError(
            "Must specify test dataset via: "
            "DATA.test.source='[/path/to/test/data/*]'"
        )

    # 2. 初始化 Accelerator
    timeout_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=1800))
    accelerator = Accelerator(kwargs_handlers=[timeout_kwargs])

    # 3. 设置日志
    log_format = "%(message)s"
    date_format = "[%X]"

    logging.basicConfig(
        format=log_format,
        datefmt=date_format,
        level=logging.INFO,
        handlers=[RichHandler(rich_tracebacks=True)],
        force=True,
    )

    # 4. 创建输出目录（自动根据 checkpoint_path 设置）
    # 获取当前日期
    today_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if cfg.TEST.checkpoint_path:
        # 从 checkpoint_path 提取基础目录
        # 例如：checkpoint/07-01-2026/checkpoints/checkpoint-30000 → checkpoint/07-01-2026
        checkpoint_path = cfg.TEST.checkpoint_path
        # 获取 checkpoints 文件夹的父目录
        checkpoint_parent = osp.dirname(checkpoint_path)
        # 在同级创建带日期的 test_results 文件夹
        output_dir = osp.join(checkpoint_parent, f"test_results_{today_str}")
        logger.info(f"Auto-set output_dir based on checkpoint_path: {output_dir}")
    else:
        # 使用配置文件中的默认值，并添加日期后缀
        output_dir = osp.join(cfg.TEST.output_dir, f"test_results_{today_str}")

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # 保存配置
        config_save_path = osp.join(output_dir, "test_config.yaml")
        OmegaConf.save(cfg, config_save_path)
        logger.info(f"Saved config to {config_save_path}")

        # 保存测试数据源
        sources_save_path = osp.join(output_dir, "test_sources.txt")
        with open(sources_save_path, 'w') as f:
            f.write(f"# Test Date: {today_str}\n")
            f.write(f"# Number of sources: {len(cfg.DATA.test.source)}\n")
            f.write("=" * 60 + "\n\n")
            for i, src in enumerate(cfg.DATA.test.source, 1):
                f.write(f"[{i}] {src}\n")
                # 展开通配符，列出匹配的文件
                matched_files = glob.glob(src)
                matched_files = sorted(matched_files)
                if matched_files:
                    for mf in matched_files:
                        f.write(f"    - {mf}\n")
                else:
                    f.write(f"    (No files matched)\n")
                f.write("\n")
        logger.info(f"Saved test sources to {sources_save_path}")

    # 5. 初始化 AIM（可选）
    aim_run = None
    if accelerator.is_main_process and cfg.AIM.get("server_url") and cfg.TEST.enable_vis:
        try:
            aim_run = Run(
                experiment="test",
                repo=cfg.AIM.server_url,
            )
            aim_run["hparams"] = OmegaConf.to_container(cfg, resolve=True)
            logger.info(f'AIM run initialized in {cfg.AIM.server_url}')
            logger.info(f'AIM run hash: {aim_run.hash}')
        except Exception as e:
            logger.warning(f"Failed to initialize AIM: {e}")
            aim_run = None

    accelerator.wait_for_everyone()

    # 6. 设置种子
    set_seed(42)

    # 7. 加载测试数据
    test_loader = setup_test_dataloader(cfg, accelerator)

    # 8. 初始化模型
    net = setup_model(cfg)

    # 9. Accelerate prepare（不需要 optimizer）
    net = accelerator.prepare(net)

    # 10. 加载 checkpoint
    checkpoint_path = cfg.TEST.checkpoint_path
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    accelerator.load_state(checkpoint_path, strict=False)
    logger.info("Checkpoint loaded successfully")

    # 11. 执行测试
    logger.info("Starting test...")
    local_results = test(cfg, accelerator, net, test_loader, aim_run)

    # 12. 合并多卡结果
    logger.info("Merging results from all processes...")
    merged_results = merge_results(accelerator, local_results)

    # 13. 保存结果（只在主进程）
    if accelerator.is_main_process:
        # 保存 HDF5
        output_path = osp.join(output_dir, "predictions.h5")
        save_results_hdf5(output_path, merged_results, cfg)

        # 计算并保存指标
        compute_quick_metrics(merged_results, output_dir)

        logger.info("Test completed successfully!")

    # 14. 关闭 AIM
    if accelerator.is_main_process and aim_run is not None:
        aim_run.close()


if __name__ == "__main__":
    main()
