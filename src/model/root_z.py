from __future__ import annotations

import math
from typing import Dict, Tuple

import torch


ROOT_Z_GEOM_DIM = 6
ROOT_Z_MIN_PRIOR = 1e-6


def compute_root_z_prior_and_geom(
    hand_bbox: torch.Tensor,
    focal: torch.Tensor,
    princpt: torch.Tensor,
    prior_k: float,
    eps: float = ROOT_Z_MIN_PRIOR,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    根据 hand bbox 和相机内参构造 root-z 的显式先验与几何特征。

    这里采用的先验形式为：

        z_prior = k * sqrt(fx * fy) / sqrt(bbox_w * bbox_h)

    其直觉是：同样真实尺寸的手，图像里越小通常越远；焦距越大，
    在同样距离下的表观尺寸越大。因此先显式给出一个粗略几何先验，
    再让网络学习相对这个先验的 `Δlog z`，会比直接学习绝对深度更稳。

    Args:
        hand_bbox: [..., 4]，xyxy 格式
        focal: [..., 2]，(fx, fy)
        princpt: [..., 2]，(cx, cy)
        prior_k: 全局尺度常数
        eps: 数值稳定项，避免除零或 log(0)

    Returns:
        z_prior: [...]，正值先验深度（mm）
        log_z_prior: [...]，先验深度的对数
        geom_feat: [..., 6]
            [log_z_prior, dx, dy, dw, dh, log_aspect_ratio]
    """
    bbox_w = torch.clamp(hand_bbox[..., 2] - hand_bbox[..., 0], min=eps)
    bbox_h = torch.clamp(hand_bbox[..., 3] - hand_bbox[..., 1], min=eps)
    bbox_cx = (hand_bbox[..., 0] + hand_bbox[..., 2]) * 0.5
    bbox_cy = (hand_bbox[..., 1] + hand_bbox[..., 3]) * 0.5

    fx = torch.clamp(focal[..., 0], min=eps)
    fy = torch.clamp(focal[..., 1], min=eps)
    px = princpt[..., 0]
    py = princpt[..., 1]

    focal_eff = torch.sqrt(fx * fy)
    bbox_scale = torch.sqrt(bbox_w * bbox_h)

    z_prior = torch.clamp(prior_k * focal_eff / bbox_scale, min=eps)
    log_z_prior = torch.log(z_prior)

    dx = (bbox_cx - px) / fx
    dy = (bbox_cy - py) / fy
    dw = bbox_w / fx
    dh = bbox_h / fy
    log_aspect_ratio = torch.log(torch.clamp(bbox_w / bbox_h, min=eps))

    geom_feat = torch.stack(
        [log_z_prior, dx, dy, dw, dh, log_aspect_ratio],
        dim=-1,
    )
    return z_prior, log_z_prior, geom_feat


def encode_delta_log_z_targets(
    root_z: torch.Tensor,
    log_z_prior: torch.Tensor,
    d_min: float,
    d_max: float,
    num_bins: int,
    eps: float = ROOT_Z_MIN_PRIOR,
) -> Dict[str, torch.Tensor]:
    """
    将绝对 root-z 编码成 prior-centered `Δlog z` 的 multibin 监督目标。

    目标定义为：

        Δlog z = log(z_gt) - log(z_prior)

    然后将 `Δlog z` 裁剪到 [d_min, d_max] 区间，生成：
    - `bin_idx`: coarse 分类标签
    - `bin_center`: 该 bin 的中心
    - `residual`: bin 内残差，理论范围约为 [-0.5, 0.5]

    Args:
        root_z: [...]，GT 绝对根深度（mm）
        log_z_prior: [...]，先验深度的对数
        d_min: `Δlog z` 支持区间下界
        d_max: `Δlog z` 支持区间上界
        num_bins: bin 数
        eps: 数值稳定项

    Returns:
        包含编码结果的字典：
        - delta_log_z
        - delta_log_z_clamped
        - bin_idx
        - bin_center
        - residual
        - bin_size
    """
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive, got {num_bins}")
    if d_max <= d_min:
        raise ValueError(f"d_max must be larger than d_min, got {d_min} >= {d_max}")

    delta = (d_max - d_min) / float(num_bins)
    root_z = torch.clamp(root_z, min=eps)
    delta_log_z = torch.log(root_z) - log_z_prior
    delta_log_z_clamped = torch.clamp(delta_log_z, min=d_min, max=d_max)
    bin_idx = torch.floor((delta_log_z_clamped - d_min) / delta).long().clamp(0, num_bins - 1)
    bin_center = d_min + (bin_idx.float() + 0.5) * delta
    residual = (delta_log_z_clamped - bin_center) / delta

    return {
        "delta_log_z": delta_log_z,
        "delta_log_z_clamped": delta_log_z_clamped,
        "bin_idx": bin_idx,
        "bin_center": bin_center,
        "residual": residual,
        "bin_size": torch.full_like(delta_log_z, delta),
    }


def decode_delta_log_z_predictions(
    z_cls_logits: torch.Tensor,
    z_residuals: torch.Tensor,
    log_z_prior: torch.Tensor,
    d_min: float,
    d_max: float,
) -> Dict[str, torch.Tensor]:
    """
    将 multibin 的 root-z 预测恢复成绝对深度。

    推理流程：
    1. 取 `argmax(logits)` 得到预测 bin
    2. 读取该 bin 上的 residual，并限制在 [-0.5, 0.5]
    3. 恢复 `pred_delta_log_z`
    4. 加回 `log_z_prior`
    5. `exp` 得到最终绝对深度

    Args:
        z_cls_logits: [..., K]，分类 logits
        z_residuals: [..., K]，每个 bin 的 residual 预测
        log_z_prior: [...]，先验深度的对数
        d_min: `Δlog z` 支持区间下界
        d_max: `Δlog z` 支持区间上界

    Returns:
        包含解码结果的字典：
        - pred_bin
        - pred_residual
        - pred_delta_log_z
        - pred_log_z
        - pred_z
        - bin_centers
        - bin_size
    """
    if z_cls_logits.shape != z_residuals.shape:
        raise ValueError(
            f"logits/residuals shape mismatch: {z_cls_logits.shape} vs {z_residuals.shape}"
        )
    num_bins = z_cls_logits.shape[-1]
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive, got {num_bins}")
    if d_max <= d_min:
        raise ValueError(f"d_max must be larger than d_min, got {d_min} >= {d_max}")

    delta = (d_max - d_min) / float(num_bins)
    bin_centers = torch.linspace(
        d_min + 0.5 * delta,
        d_max - 0.5 * delta,
        steps=num_bins,
        device=z_cls_logits.device,
        dtype=z_cls_logits.dtype,
    )
    pred_bin = torch.argmax(z_cls_logits, dim=-1)
    pred_residual = z_residuals.gather(dim=-1, index=pred_bin[..., None]).squeeze(-1)
    pred_residual = torch.clamp(pred_residual, min=-0.5, max=0.5)
    pred_delta_log_z = bin_centers[pred_bin] + pred_residual * delta
    pred_log_z = log_z_prior + pred_delta_log_z
    pred_z = torch.exp(pred_log_z)

    return {
        "pred_bin": pred_bin,
        "pred_residual": pred_residual,
        "pred_delta_log_z": pred_delta_log_z,
        "pred_log_z": pred_log_z,
        "pred_z": pred_z,
        "bin_centers": bin_centers,
        "bin_size": torch.full_like(pred_log_z, delta),
    }
