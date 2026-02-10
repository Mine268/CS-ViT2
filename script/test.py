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
from src.data.preprocess import preprocess_batch
from src.model.net import PoseNet
from src.utils.vis import vis
from src.utils.metric import *


logger = get_logger(__name__)


def setup_test_dataloader(cfg: DictConfig):
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

    test_loader = get_dataloader(
        url=test_sources,
        num_frames=cfg.MODEL.num_frame,
        stride=cfg.DATA.test.stride,
        batch_size=cfg.TEST.batch_size,
        num_workers=1,              # 单 worker 避免 tar 文件少的问题
        prefetcher_factor=1,
        infinite=False,             # 单次遍历
        seed=42,                    # 固定种子
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
    主测试循环，收集预测结果和 ground truth

    Args:
        cfg: Hydra 配置对象
        accelerator: Accelerate 分布式管理器
        net: 模型
        test_loader: 测试数据加载器
        aim_run: AIM 实验跟踪对象（可选）

    Returns:
        local_results: 本地收集的结果字典
    """
    net.eval()
    device = accelerator.device

    # 初始化本地结果存储
    local_results = {
        "imgs_path": [],
        "handedness": [],
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
        "joint_valid": [],
        "mano_valid": [],
    }

    total_samples = 0
    max_samples = cfg.TEST.max_samples

    for batch_idx, batch_ in enumerate(test_loader):
        # 检查是否达到样本数限制
        if max_samples is not None and total_samples >= max_samples:
            break

        # 预处理（关闭增强）
        batch, trans_2d_mat, _ = preprocess_batch(
            batch_origin=batch_,
            patch_size=[cfg.MODEL.img_size, cfg.MODEL.img_size],
            patch_expanstion=1.0,  # 测试时不扩张
            scale_z_range=[1.0, 1.0],
            scale_f_range=[1.0, 1.0],
            persp_rot_max=0.0,
            joint_rep_type=cfg.MODEL.joint_type,
            augmentation_flag=False,  # 关闭增强
            device=device,
            pixel_aug=None,
            perspective_normalization=cfg.TRAIN.get("perspective_normalization", False),
        )

        # 推理（使用 predict_full，传入 GT 用于 norm_by_hand 反归一化）
        result = net.predict_full(
            img=batch["patches"],
            bbox=batch["patch_bbox"],
            focal=batch["focal"],
            princpt=batch["princpt"],
            timestamp=batch["timestamp"],
            joint_cam_gt=batch["joint_cam"],      # 用于计算 norm_scale
            joint_valid_gt=batch["joint_valid"],  # 用于计算 norm_scale
        )

        # 计算 GT 的 verts（使用 MANO FK）
        # 注意：需要访问 unwrapped net
        unwrapped_net = net.module if hasattr(net, 'module') else net
        with torch.no_grad():
            _, verts_rel_gt = unwrapped_net.mano_to_pose(
                batch["mano_pose"][:, -1:],
                batch["mano_shape"][:, -1:]
            )
            # verts_cam_gt = verts_rel_gt + root_joint
            verts_cam_gt = verts_rel_gt + batch["joint_cam"][:, -1:, :1]

        # 收集结果（只保存最后一帧）
        batch_size = batch["patches"].shape[0]
        for i in range(batch_size):
            # 检查是否达到样本数限制
            if max_samples is not None and total_samples >= max_samples:
                break

            # 预测结果（已经是最后一帧）
            local_results["joint_cam_pred"].append(
                result["joint_cam_pred"][i, 0].cpu().numpy()  # [21, 3]
            )
            local_results["vert_cam_pred"].append(
                result["vert_cam_pred"][i, 0].cpu().numpy()  # [778, 3]
            )
            local_results["mano_pose_pred"].append(
                result["mano_pose_pred"][i, 0].cpu().numpy()  # [48]
            )
            local_results["mano_shape_pred"].append(
                result["mano_shape_pred"][i, 0].cpu().numpy()  # [10]
            )
            local_results["trans_pred"].append(
                result["trans_pred"][i, 0].cpu().numpy()  # [3]
            )
            local_results["trans_pred_denorm"].append(
                result["trans_pred_denorm"][i, 0].cpu().numpy()  # [3]
            )
            local_results["norm_scale"].append(
                result["norm_scale"][i, 0].cpu().numpy()  # scalar
            )
            local_results["norm_valid"].append(
                result["norm_valid"][i, 0].cpu().numpy()  # scalar
            )

            # Ground truth（取最后一帧）
            local_results["joint_cam_gt"].append(
                batch["joint_cam"][i, -1].cpu().numpy()  # [21, 3]
            )
            local_results["vert_cam_gt"].append(
                verts_cam_gt[i, 0].cpu().numpy()  # [778, 3]
            )
            local_results["mano_pose_gt"].append(
                batch["mano_pose"][i, -1].cpu().numpy()  # [48]
            )
            local_results["mano_shape_gt"].append(
                batch["mano_shape"][i, -1].cpu().numpy()  # [10]
            )

            # 相机参数和其他信息（取最后一帧）
            local_results["focal"].append(
                batch["focal"][i, -1].cpu().numpy()  # [2]
            )
            local_results["princpt"].append(
                batch["princpt"][i, -1].cpu().numpy()  # [2]
            )
            local_results["hand_bbox"].append(
                batch["hand_bbox"][i, -1].cpu().numpy()  # [4]
            )
            local_results["joint_valid"].append(
                batch["joint_valid"][i, -1].cpu().numpy()  # [21]
            )
            local_results["mano_valid"].append(
                batch["mano_valid"][i, -1].cpu().numpy()  # []
            )

            # 路径和 handedness（字符串）
            # 注意：imgs_path 可能是列表或字符串
            imgs_path = batch["imgs_path"][i]
            if isinstance(imgs_path, list):
                local_results["imgs_path"].append(imgs_path[-1])  # 取最后一帧
            else:
                local_results["imgs_path"].append(str(imgs_path))

            handedness = "left" if batch["flip"][i] else "right"
            if isinstance(handedness, list):
                local_results["handedness"].append(handedness[-1])
            else:
                local_results["handedness"].append(str(handedness))

            total_samples += 1

        # 可视化
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
                context={"subset": "test"}
            )

        # 日志
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

    # 数值数据：转为 Tensor 后 gather
    tensor_keys = [
        "joint_cam_pred", "vert_cam_pred", "mano_pose_pred", "mano_shape_pred",
        "trans_pred", "trans_pred_denorm", "norm_scale", "norm_valid",
        "joint_cam_gt", "vert_cam_gt", "mano_pose_gt", "mano_shape_gt",
        "focal", "princpt", "hand_bbox", "joint_valid", "mano_valid"
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

    # 字符串数据：直接 gather（gather_for_metrics 支持字符串列表）
    string_keys = ["imgs_path", "handedness"]
    for key in string_keys:
        gathered_list = accelerator.gather_for_metrics(local_results[key])
        if accelerator.is_main_process:
            merged_results[key] = gathered_list

    return merged_results


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
        # 创建 samples 组
        samples_group = f.create_group("samples")

        # 字符串数据（需要特殊处理）
        dt_str = h5py.string_dtype(encoding='utf-8')

        # imgs_path
        imgs_path_list = results["imgs_path"]
        imgs_path_flat = []
        for path in imgs_path_list:
            if isinstance(path, list):
                imgs_path_flat.append(path[0] if len(path) > 0 else "")
            else:
                imgs_path_flat.append(str(path))
        samples_group.create_dataset(
            "imgs_path",
            data=np.array(imgs_path_flat, dtype=dt_str)
        )

        # handedness
        handedness_list = results["handedness"]
        handedness_flat = []
        for hand in handedness_list:
            if isinstance(hand, list):
                handedness_flat.append(hand[0] if len(hand) > 0 else "")
            else:
                handedness_flat.append(str(hand))
        samples_group.create_dataset(
            "handedness",
            data=np.array(handedness_flat, dtype=dt_str)
        )

        # 数值数据（使用 gzip 压缩）
        compression = cfg.TEST.compression if cfg.TEST.compression else None

        samples_group.create_dataset(
            "joint_cam_pred",
            data=results["joint_cam_pred"].astype(np.float32),
            compression=compression
        )
        samples_group.create_dataset(
            "vert_cam_pred",
            data=results["vert_cam_pred"].astype(np.float32),
            compression=compression
        )
        samples_group.create_dataset(
            "mano_pose_pred",
            data=results["mano_pose_pred"].astype(np.float32),
            compression=compression
        )
        samples_group.create_dataset(
            "mano_shape_pred",
            data=results["mano_shape_pred"].astype(np.float32),
            compression=compression
        )
        samples_group.create_dataset(
            "trans_pred",
            data=results["trans_pred"].astype(np.float32),
            compression=compression
        )
        samples_group.create_dataset(
            "trans_pred_denorm",
            data=results["trans_pred_denorm"].astype(np.float32),
            compression=compression
        )
        samples_group.create_dataset(
            "norm_scale",
            data=results["norm_scale"].astype(np.float32)
        )
        samples_group.create_dataset(
            "norm_valid",
            data=results["norm_valid"].astype(np.float32)
        )

        samples_group.create_dataset(
            "joint_cam_gt",
            data=results["joint_cam_gt"].astype(np.float32),
            compression=compression
        )
        samples_group.create_dataset(
            "vert_cam_gt",
            data=results["vert_cam_gt"].astype(np.float32),
            compression=compression
        )
        samples_group.create_dataset(
            "mano_pose_gt",
            data=results["mano_pose_gt"].astype(np.float32),
            compression=compression
        )
        samples_group.create_dataset(
            "mano_shape_gt",
            data=results["mano_shape_gt"].astype(np.float32),
            compression=compression
        )

        samples_group.create_dataset(
            "focal",
            data=results["focal"].astype(np.float32)
        )
        samples_group.create_dataset(
            "princpt",
            data=results["princpt"].astype(np.float32)
        )
        samples_group.create_dataset(
            "hand_bbox",
            data=results["hand_bbox"].astype(np.float32)
        )
        samples_group.create_dataset(
            "joint_valid",
            data=results["joint_valid"].astype(np.float32)
        )
        samples_group.create_dataset(
            "mano_valid",
            data=results["mano_valid"].astype(np.float32)
        )

        # 创建 metadata 组
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
    joint_cam_pred = results["joint_cam_pred"]  # [N, 21, 3]
    joint_cam_gt = results["joint_cam_gt"]      # [N, 21, 3]
    vert_cam_pred = results["vert_cam_pred"]    # [N, 778, 3]
    vert_cam_gt = results["vert_cam_gt"]        # [N, 778, 3]
    joint_valid = results["joint_valid"]        # [N, 21]
    mano_valid = results["mano_valid"]          # [N]

    # 计算 MPJPE
    joint_diff = np.linalg.norm(joint_cam_pred - joint_cam_gt, axis=-1)  # [N, 21]
    joint_diff_masked = joint_diff * joint_valid
    mpjpe = np.sum(joint_diff_masked) / np.sum(joint_valid)

    # 计算 MPVPE
    vert_diff = np.linalg.norm(vert_cam_pred - vert_cam_gt, axis=-1)  # [N, 778]
    vert_diff_masked = vert_diff * mano_valid[:, None]
    mpvpe = np.sum(vert_diff_masked) / (np.sum(mano_valid) * 778)

    metrics = {
        "mpjpe": float(mpjpe),
        "mpvpe": float(mpvpe),
        "num_samples": int(len(joint_cam_pred)),
    }

    # 保存到 JSON
    metrics_path = osp.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics: MPJPE={mpjpe:.2f} mm, MPVPE={mpvpe:.2f} mm")
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
    if cfg.TEST.checkpoint_path:
        # 从 checkpoint_path 提取基础目录
        # 例如：checkpoint/07-01-2026/checkpoints/checkpoint-30000 → checkpoint/07-01-2026
        checkpoint_path = cfg.TEST.checkpoint_path
        # 获取 checkpoints 文件夹的父目录
        checkpoint_parent = osp.dirname(osp.dirname(checkpoint_path))
        # 在同级创建 test_results 文件夹
        output_dir = osp.join(checkpoint_parent, "test_results")
        logger.info(f"Auto-set output_dir based on checkpoint_path: {output_dir}")
    else:
        # 使用配置文件中的默认值
        output_dir = cfg.TEST.output_dir

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # 保存配置
        config_save_path = osp.join(output_dir, "test_config.yaml")
        OmegaConf.save(cfg, config_save_path)
        logger.info(f"Saved config to {config_save_path}")

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
    test_loader = setup_test_dataloader(cfg)

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
