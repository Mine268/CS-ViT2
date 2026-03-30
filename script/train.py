from typing import *
import os
import os.path as osp
import shutil
import glob
import logging
from collections import OrderedDict
from rich.logging import RichHandler
import datetime
import json
import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.utils import set_seed, broadcast_object_list
from accelerate.logging import get_logger

from aim import Run, Image

from src.data.dataloader import get_dataloader
from src.data.dataloader import (
    estimate_wds_shard_clip_counts,
    build_balanced_clip_segments,
    compute_dataset_reweight_probs,
    get_dataset_reweight_dataloader,
    get_segmented_wds_dataloader,
)
from src.data.depth_bin_dataloader import (
    collect_depth_bin_sources,
    collect_depth_bin_cell_sources,
    compute_dataset_bin_cell_weights,
    get_depth_bin_dataloader,
    get_dataset_bin_balanced_dataloader,
)
from src.data.preprocess import preprocess_batch
from src.model.net import PoseNet
from src.utils.vis import vis
from src.utils.metric import *
from src.utils.train_utils import get_progressive_dropout


logger = get_logger(__name__)
save_dir = None


def _has_coco_wholebody_source(source_list: Sequence[str]) -> bool:
    return any("COCO-WholeBody" in str(source) for source in source_list)


def _normalize_source_patterns(source_value: Any) -> List[str]:
    if isinstance(source_value, (list, tuple, ListConfig)):
        return [str(item) for item in source_value]
    return [str(source_value)]


def get_effective_train_source_patterns(cfg: DictConfig) -> List[str]:
    source_patterns = [str(source) for source in cfg.DATA.train.get("source", [])]
    train_reweight_cfg = cfg.DATA.train.get("reweight", {})
    for entry in train_reweight_cfg.get("datasets", []):
        source_patterns.extend(_normalize_source_patterns(entry.get("source", [])))
    return source_patterns


def assert_coco_wholebody_training_compat(cfg: DictConfig):
    train_sources = get_effective_train_source_patterns(cfg)
    if _has_coco_wholebody_source(train_sources) and cfg.MODEL.get("norm_by_hand", False):
        raise AssertionError(
            "COCO-WholeBody training currently requires MODEL.norm_by_hand=false."
        )


def collect_reweight_dataset_sources(
    source_patterns: Sequence[str],
    dataset_names: Sequence[str],
) -> "OrderedDict[str, List[str]]":
    if len(dataset_names) == 0:
        raise ValueError("reweight.enabled=true requires a non-empty weights mapping")

    dataset_sources: "OrderedDict[str, List[str]]" = OrderedDict(
        (str(dataset_name), []) for dataset_name in dataset_names
    )
    unmatched_patterns: List[str] = []
    ambiguous_patterns: List[str] = []

    for source_pattern in source_patterns:
        pattern_str = str(source_pattern)
        matched_names = [
            dataset_name for dataset_name in dataset_sources.keys()
            if dataset_name in pattern_str
        ]
        if len(matched_names) == 0:
            unmatched_patterns.append(pattern_str)
            continue
        if len(matched_names) > 1:
            ambiguous_patterns.append(pattern_str)
            continue

        matched_files = sorted(glob.glob(pattern_str))
        if len(matched_files) == 0:
            raise ValueError(f"reweight source pattern matched no files: {pattern_str}")

        dataset_sources[matched_names[0]].extend(matched_files)

    if ambiguous_patterns:
        raise ValueError(f"Ambiguous reweight source patterns: {ambiguous_patterns}")
    if unmatched_patterns:
        raise ValueError(
            "Found source patterns that do not map to any reweight dataset: "
            f"{unmatched_patterns}"
        )

    empty_datasets = [name for name, urls in dataset_sources.items() if len(urls) == 0]
    if empty_datasets:
        raise ValueError(
            "Missing train sources for reweight datasets: "
            f"{empty_datasets}"
        )

    return dataset_sources


def collect_reweight_dataset_config(
    source_patterns: Sequence[str],
    reweight_cfg: DictConfig,
) -> Tuple["OrderedDict[str, List[str]]", "OrderedDict[str, float]"]:
    dataset_entries = reweight_cfg.get("datasets", [])
    if len(dataset_entries) > 0:
        dataset_sources: "OrderedDict[str, List[str]]" = OrderedDict()
        dataset_weights: "OrderedDict[str, float]" = OrderedDict()

        for entry in dataset_entries:
            dataset_name = str(entry.get("name", "")).strip()
            if dataset_name == "":
                raise ValueError("Each reweight dataset entry must provide a non-empty name")
            if dataset_name in dataset_sources:
                raise ValueError(f"Duplicate reweight dataset entry: {dataset_name}")

            patterns = _normalize_source_patterns(entry.get("source", []))
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

    dataset_weights = OrderedDict(
        (str(name), float(weight))
        for name, weight in reweight_cfg.get("weights", {}).items()
    )
    dataset_sources = collect_reweight_dataset_sources(
        source_patterns=source_patterns,
        dataset_names=list(dataset_weights.keys()),
    )
    return dataset_sources, dataset_weights


def save_best_model_variant(
    accelerator: Accelerator,
    output_dir: str,
    global_step: int,
    val_metrics: Dict[str, float],
    config_name: str,
    best_dir_name: str,
    metadata_filename: str,
):
    """保存指定指标对应的最优模型及其元数据。"""
    if not accelerator.is_main_process:
        return

    best_model_dir = osp.join(output_dir, best_dir_name)
    accelerator.save_state(best_model_dir)

    metadata = {
        "step": global_step,
        "config_name": config_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "best_dir_name": best_dir_name,
        **val_metrics,
    }

    metadata_path = osp.join(output_dir, metadata_filename)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Best model metadata saved to {metadata_path}")


def load_best_metric_info(
    output_dir: str,
    metric_key: str,
    metadata_filename: str,
) -> Dict:
    """加载指定指标的最优模型元数据。"""
    metadata_path = osp.join(output_dir, metadata_filename)

    if not osp.exists(metadata_path):
        return {
            "best_value": float('inf'),
            "step": 0,
        }

    try:
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            return {
                "best_value": data.get(metric_key, float('inf')),
                "step": data.get("step", 0),
            }
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load best model info from {metadata_path}: {e}")
        return {
            "best_value": float('inf'),
            "step": 0,
        }


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


def setup_dataloader(cfg: DictConfig, accelerator: Optional[Accelerator] = None):
    train_sampling_cfg = cfg.DATA.train.get("sampling", {})
    train_depth_bin_cfg = cfg.DATA.train.get("depth_bins", {})
    train_reweight_cfg = cfg.DATA.train.get("reweight", {})

    if train_depth_bin_cfg.get("enabled", False) and train_reweight_cfg.get("enabled", False):
        raise ValueError(
            "DATA.train.depth_bins.enabled and DATA.train.reweight.enabled "
            "cannot both be true"
        )

    if train_depth_bin_cfg.get("enabled", False):
        dataset_names = train_depth_bin_cfg.get("dataset_names", [])
        split = train_depth_bin_cfg.get("split", "train")
        mix_strategy = train_depth_bin_cfg.get("mix_strategy", "uniform_random")
        if mix_strategy == "dataset_bin_balanced":
            cell_sources = collect_depth_bin_cell_sources(
                root=train_depth_bin_cfg["root"],
                dataset_names=dataset_names,
                split=split,
                num_frames=cfg.MODEL.num_frame,
                stride=cfg.DATA.train.stride,
                selected_bins=train_depth_bin_cfg.get("selected_bins", None),
                min_cell_samples=train_depth_bin_cfg.get("min_cell_samples", 0),
            )
            if len(cell_sources) == 0:
                raise ValueError(
                    f"No dataset-bin cells found under root={train_depth_bin_cfg['root']} "
                    f"for datasets={dataset_names}, split={split}, "
                    f"nf={cfg.MODEL.num_frame}, stride={cfg.DATA.train.stride}"
                )
            cell_weights = compute_dataset_bin_cell_weights(
                cell_sources=cell_sources,
                dataset_balance_alpha=train_depth_bin_cfg.get("dataset_balance_alpha", 0.5),
            )
            train_loader = get_dataset_bin_balanced_dataloader(
                cell_sources=cell_sources,
                batch_size=cfg.TRAIN.sample_per_device,
                num_workers=cfg.GENERAL.num_worker,
                prefetcher_factor=cfg.GENERAL.prefetch_factor,
                infinite=True,
                seed=cfg.GENERAL.get("seed", None),
                dataset_balance_alpha=train_depth_bin_cfg.get("dataset_balance_alpha", 0.5),
                shardshuffle=train_depth_bin_cfg.get("shardshuffle", False),
                sample_shuffle=train_depth_bin_cfg.get("sample_shuffle", 200),
            )
            logger.info(
                f"setup dataset-bin-balanced train loader: root={train_depth_bin_cfg['root']} "
                f"datasets={dataset_names} cells={list(cell_sources.keys())} "
                f"weights={dict(cell_weights)}"
            )
        else:
            bin_sources = collect_depth_bin_sources(
                root=train_depth_bin_cfg["root"],
                dataset_names=dataset_names,
                split=split,
                num_frames=cfg.MODEL.num_frame,
                stride=cfg.DATA.train.stride,
                selected_bins=train_depth_bin_cfg.get("selected_bins", None),
            )
            if len(bin_sources) == 0:
                raise ValueError(
                    f"No depth-bin data found under root={train_depth_bin_cfg['root']} "
                    f"for datasets={dataset_names}, split={split}, "
                    f"nf={cfg.MODEL.num_frame}, stride={cfg.DATA.train.stride}"
                )

            bin_weights = train_depth_bin_cfg.get("bin_weights", None)
            train_loader = get_depth_bin_dataloader(
                bin_sources=bin_sources,
                batch_size=cfg.TRAIN.sample_per_device,
                num_workers=cfg.GENERAL.num_worker,
                prefetcher_factor=cfg.GENERAL.prefetch_factor,
                infinite=True,
                seed=cfg.GENERAL.get("seed", None),
                bin_weights=bin_weights,
                shardshuffle=train_depth_bin_cfg.get("shardshuffle", False),
                sample_shuffle=train_depth_bin_cfg.get("sample_shuffle", 200),
                mix_strategy=mix_strategy,
            )
            logger.info(
                f"setup depth-bin train loader: root={train_depth_bin_cfg['root']} "
                f"datasets={dataset_names} bins={list(bin_sources.keys())}"
            )
    elif train_reweight_cfg.get("enabled", False):
        dataset_sources, dataset_weights = collect_reweight_dataset_config(
            source_patterns=cfg.DATA.train.source,
            reweight_cfg=train_reweight_cfg,
        )
        normalized_weights = compute_dataset_reweight_probs(
            dataset_sources=dataset_sources,
            dataset_weights=dataset_weights,
        )
        train_loader = get_dataset_reweight_dataloader(
            dataset_sources=dataset_sources,
            dataset_weights=dataset_weights,
            num_frames=cfg.MODEL.num_frame,
            stride=cfg.DATA.train.stride,
            batch_size=cfg.TRAIN.sample_per_device,
            num_workers=cfg.GENERAL.num_worker,
            prefetcher_factor=cfg.GENERAL.prefetch_factor,
            infinite=True,
            seed=cfg.GENERAL.get("seed", None),
            clip_sampling_mode=train_sampling_cfg.get("mode", "dense"),
            clips_per_sequence=train_sampling_cfg.get("clips_per_sequence", None),
            shardshuffle=train_reweight_cfg.get(
                "shardshuffle",
                train_sampling_cfg.get("shardshuffle", False),
            ),
            post_clip_shuffle=train_reweight_cfg.get(
                "post_clip_shuffle",
                train_sampling_cfg.get("post_clip_shuffle", 200),
            ),
            default_source_split=train_reweight_cfg.get("split", "train"),
        )
        logger.info(
            "setup reweight train loader: datasets=%s weights=%s num_frames=%s stride=%s",
            {name: len(urls) for name, urls in dataset_sources.items()},
            dict(normalized_weights),
            cfg.MODEL.num_frame,
            cfg.DATA.train.stride,
        )
    else:
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
            clip_sampling_mode=train_sampling_cfg.get("mode", "dense"),
            clips_per_sequence=train_sampling_cfg.get("clips_per_sequence", None),
            shardshuffle=train_sampling_cfg.get("shardshuffle", False),
            post_clip_shuffle=train_sampling_cfg.get("post_clip_shuffle", 200),
        )
        logger.info(f"setup train loader: {train_sources}")

    val_sampling_cfg = cfg.DATA.val.get("sampling", {})
    val_sources = []
    for src in cfg.DATA.val.source:
        matched_files = glob.glob(src)
        matched_files = sorted(matched_files)
        val_sources.extend(matched_files)
    if cfg.DATA.val.get("full_eval", False):
        if accelerator is None:
            raise ValueError("full_eval val requires accelerator for shard partitioning")
        shard_clip_counts_obj = [None]
        if accelerator.is_main_process:
            shard_clip_counts_obj[0] = estimate_wds_shard_clip_counts(
                urls=val_sources,
                num_frames=cfg.MODEL.num_frame,
                stride=cfg.DATA.val.stride,
            )
        broadcast_object_list(shard_clip_counts_obj, from_process=0)
        shard_clip_counts = shard_clip_counts_obj[0]
        rank_segments = build_balanced_clip_segments(
            urls=val_sources,
            clip_counts=shard_clip_counts,
            num_parts=accelerator.num_processes,
        )
        process_segments = rank_segments[accelerator.process_index]
        val_loader = get_segmented_wds_dataloader(
            segments=process_segments,
            num_frames=cfg.MODEL.num_frame,
            stride=cfg.DATA.val.stride,
            batch_size=cfg.TRAIN.sample_per_device,
            num_workers=0,
            prefetcher_factor=cfg.GENERAL.prefetch_factor,
        )
        logger.info(
            f"setup full-eval val loader: process={accelerator.process_index}/{accelerator.num_processes} "
            f"segments={len(process_segments)} total_shards={len(val_sources)} "
            f"clip_counts={shard_clip_counts}"
        )
    else:
        val_loader = get_dataloader(
            url=val_sources,
            num_frames=cfg.MODEL.num_frame,
            stride=cfg.DATA.val.stride,
            batch_size=cfg.TRAIN.sample_per_device,
            num_workers=1, # cfg.GENERAL.num_worker,
            prefetcher_factor=cfg.GENERAL.prefetch_factor,
            infinite=True,
            seed=cfg.GENERAL.get("val_seed", 42),  # 固定验证顺序，在线验证可复现
            clip_sampling_mode=val_sampling_cfg.get("mode", "dense"),
            clips_per_sequence=val_sampling_cfg.get("clips_per_sequence", None),
            shardshuffle=val_sampling_cfg.get("shardshuffle", False),
            post_clip_shuffle=val_sampling_cfg.get("post_clip_shuffle", 200),
        )
        logger.info(f"setup val loader: {val_sources}")

    return train_loader, val_loader


def setup_model(cfg: DictConfig):
    net = PoseNet(
        stage=cfg.MODEL.stage,
        stage1_weight_path=cfg.MODEL.get("stage1_weight", None),

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
        lambda_coco_patch_2d=cfg.LOSS.get("lambda_coco_patch_2d", 0.0),
        hm_sigma=cfg.LOSS.get("heatmap_sigma", 3),
        reproj_loss_type=cfg.LOSS.get("reproj_loss_type", "robust_l1"),
        reproj_loss_delta=cfg.LOSS.get("reproj_loss_delta", 84.0),

        freeze_backbone=cfg.TRAIN.backbone_lr is None,
        norm_by_hand=cfg.MODEL.get("norm_by_hand", False),
        handec_cam_head_type=cfg.MODEL.handec.get("cam_head_type", "softargmax3d"),
        root_z_num_bins=cfg.MODEL.handec.get("root_z", {}).get("num_bins", 8),
        root_z_d_min=cfg.MODEL.handec.get("root_z", {}).get("d_min", -0.73),
        root_z_d_max=cfg.MODEL.handec.get("root_z", {}).get("d_max", 0.74),
        root_z_prior_k=cfg.MODEL.handec.get("root_z", {}).get("prior_k", 121.0),
        root_z_geom_hidden_dim=cfg.MODEL.handec.get("root_z", {}).get("geom_hidden_dim", 256),
        root_z_dropout=cfg.MODEL.handec.get("root_z", {}).get("dropout", 0.0),
        root_z_use_data_source_embed=cfg.MODEL.handec.get("root_z", {}).get("use_data_source_embed", False),
        lambda_root_z_cls=cfg.LOSS.get("lambda_root_z_cls", 1.0),
        lambda_root_z_res=cfg.LOSS.get("lambda_root_z_res", 1.0),
    )

    return net


def setup_optim(cfg: DictConfig, net: PoseNet):
    optim = torch.optim.AdamW(
        params=net.get_optim_param_dict(cfg.TRAIN.lr, cfg.TRAIN.backbone_lr),
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

    if limit_step is None:
        iter_wrapper = val_loader
    else:
        val_iter = iter(val_loader)
        iter_wrapper = (next(val_iter) for _ in range(limit_step))

    for ix, batch_ in enumerate(iter_wrapper):

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
            pixel_aug=None,  # 验证时不使用增强
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

        joint_3d_valid = batch["joint_3d_valid"][:, -1:]
        has_mano = batch["has_mano"][:, -1:]
        if "norm_idx" in output["result"]:
            norm_idx = output["result"]["norm_idx"]
            norm_valid = torch.all(batch["joint_3d_valid"][:, -1:, norm_idx] > 0.5, dim=-1).float()
        else:
            norm_valid = torch.ones(joint_3d_valid.shape[:2], device=joint_3d_valid.device)

        keep_mask_np = build_excluded_data_source_mask(batch.get("data_source"))
        if keep_mask_np is None:
            keep_mask = torch.ones(joint_3d_valid.shape[0], device=device, dtype=torch.bool)
        else:
            keep_mask = torch.as_tensor(keep_mask_np, device=device, dtype=torch.bool)
        if not torch.any(keep_mask):
            continue

        # 计算指标
        metric_meter.update(
            joint_cam_gt[keep_mask],
            joint_rel_gt[keep_mask],
            verts_cam_gt[keep_mask],
            verts_rel_gt[keep_mask],
            joint_cam_pred[keep_mask],
            joint_rel_pred[keep_mask],
            verts_cam_pred[keep_mask],
            verts_rel_pred[keep_mask],
            has_mano[keep_mask],
            joint_3d_valid[keep_mask],
            norm_valid[keep_mask],
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

    # ===== 初始化最优模型追踪器 =====
    best_trackers = {
        "micro_mpjpe": {
            "best_dir_name": "best_model",
            "metadata_filename": "best_model_info.json",
            "display_name": "MPJPE",
            "best_value": float('inf'),
            "step": 0,
        },
        "micro_rte": {
            "best_dir_name": "best_model_rte",
            "metadata_filename": "best_model_rte_info.json",
            "display_name": "RTE",
            "best_value": float('inf'),
            "step": 0,
        },
        "micro_mpjpe_rel": {
            "best_dir_name": "best_model_rel_mpjpe",
            "metadata_filename": "best_model_rel_mpjpe_info.json",
            "display_name": "rel-MPJPE",
            "best_value": float('inf'),
            "step": 0,
        },
    }
    if accelerator.is_main_process:
        for metric_key, tracker in best_trackers.items():
            loaded = load_best_metric_info(
                save_dir,
                metric_key=metric_key,
                metadata_filename=tracker["metadata_filename"],
            )
            tracker["best_value"] = loaded["best_value"]
            tracker["step"] = loaded["step"]
            logger.info(
                f"Best tracker initialized: {tracker['display_name']}="
                f"{tracker['best_value']:.4f} at step {tracker['step']}"
            )

    best_trackers_obj = [best_trackers]
    broadcast_object_list(best_trackers_obj, from_process=0)
    best_trackers = best_trackers_obj[0]
    # ===== 追踪器初始化结束 =====

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
        batch, trans_2d_mat, _ = preprocess_batch(
            batch_origin=batch_,
            patch_size=[cfg.MODEL.img_size, cfg.MODEL.img_size],
            patch_expanstion=cfg.TRAIN.expansion_ratio,
            scale_z_range=cfg.TRAIN.scale_z_range,
            scale_f_range=cfg.TRAIN.scale_f_range,
            persp_rot_max=cfg.TRAIN.persp_rot_max,
            joint_rep_type=cfg.MODEL.joint_type,
            augmentation_flag=True,
            device=device,
            pixel_aug=pixel_aug,  # 传递增强对象
            perspective_normalization=cfg.TRAIN.get("perspective_normalization", False),
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
                    val_limit_step = None if cfg.DATA.val.get("full_eval", False) else cfg.DATA.val.get("max_val_step", 1000)
                    val_result = val(
                        cfg,
                        accelerator,
                        net,
                        val_loader,
                        val_limit_step,
                        global_step,
                        aim_run
                    )
                    logger.info(f"validation finished.")
                    for k, v in val_result.items():
                        logger.info(f"{k}={v}")

                    # ===== 检查是否需要保存最优模型 =====
                    try:
                        config_name = HydraConfig.get().job.config_name
                    except Exception:
                        config_name = "unknown"

                    for metric_key, tracker in best_trackers.items():
                        current_value = val_result.get(metric_key, float('inf'))
                        best_value = tracker["best_value"]
                        display_name = tracker["display_name"]

                        if current_value < best_value:
                            tracker["best_value"] = current_value
                            tracker["step"] = global_step
                            logger.info(
                                f"\n{'='*70}\n"
                                f"NEW BEST MODEL FOUND!\n"
                                f"   {display_name} improved: {best_value:.4f} -> {current_value:.4f}\n"
                                f"   Step: {global_step}\n"
                                f"   Saving to: {osp.join(save_dir, tracker['best_dir_name'])}\n"
                                f"{'='*70}\n"
                            )
                            if accelerator.is_main_process:
                                save_best_model_variant(
                                    accelerator=accelerator,
                                    output_dir=save_dir,
                                    global_step=global_step,
                                    val_metrics=val_result,
                                    config_name=config_name,
                                    best_dir_name=tracker["best_dir_name"],
                                    metadata_filename=tracker["metadata_filename"],
                                )
                        else:
                            logger.info(
                                f"Current {display_name} ({current_value:.4f}) >= "
                                f"Best {display_name} ({best_value:.4f}), not saving {tracker['best_dir_name']}."
                            )

                    logger.info(
                        "Best metrics summary: "
                        + ", ".join(
                            f"{tracker['display_name']}={tracker['best_value']:.4f}@{tracker['step']}"
                            for tracker in best_trackers.values()
                        )
                    )
                    # ===== 最优模型检查结束 =====

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
        mixed_precision=cfg.TRAIN.get("mixed_precision", None),
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

    assert_coco_wholebody_training_compat(cfg)

    save_dir_obj = [None]
    aim_run = None

    if accelerator.is_main_process:
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")

        try:
            config_name = HydraConfig.get().job.config_name
        except Exception:
            config_name = "debug"

        # Run看板
        aim_run = Run(
            experiment=f"{config_name}",
            repo=cfg.AIM.server_url,
        )

        _save_dir = osp.join("checkpoint", date_str, f"{time_str}_{config_name}_{aim_run.hash}")
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

        aim_run["hparams"] = OmegaConf.to_container(cfg, resolve=True)
        logger.info(f'AIM run initialized in {cfg.AIM.server_url}')
        logger.info(f'AIM run hash: {aim_run.hash}')
        logger.info(f'AIM run URL: {cfg.AIM.server_url}/runs/{aim_run.hash}')

    broadcast_object_list(save_dir_obj, from_process=0)
    save_dir = save_dir_obj[0]

    if accelerator.is_main_process:
        config_save_path = osp.join(save_dir, f"config_{config_name}.yaml")
        OmegaConf.save(cfg, config_save_path)
        logger.info(f"Save config to {config_save_path}")

    accelerator.wait_for_everyone()
    logger.info(accelerator.state, main_process_only=False)

    # 2. 配置种子
    set_seed(cfg.GENERAL.seed)

    # 3. 获取dataloader
    train_loader, val_loader = setup_dataloader(cfg, accelerator)

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
        accelerator.load_state(resume_path, strict=False)
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
