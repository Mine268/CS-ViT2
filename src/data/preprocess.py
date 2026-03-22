from typing import *
import torch
from einops import rearrange
import kornia.geometry.transform as KT
import kornia.geometry.conversions as KC
import kornia.augmentation as KA

from ..utils.rot import *
from ..constant import *


class PixelLevelAugmentation(torch.nn.Module):
    """
    可配置的像素级增强策略（支持动态插拔）

    设计原则：
    - 通过配置文件控制每个增强的开关和参数
    - 便于进行消融实验，快速测试不同增强组合的效果
    - 避免破坏DINOv2预训练分布的增强
    - 避免与heatmap监督冲突的增强

    参考文献：
    - HaMeR (CVPR 2023): ColorJitter(0.4, 0.4, 0.4)
    - POEM (ICCV 2023): ColorJitter(0.3, 0.3, 0.2)
    - Hand4Whole (ECCV 2022): ColorJitter(0.2) + GaussianNoise

    Args:
        aug_config: 数据增强配置字典，包含各增强的开关和参数
    """
    def __init__(self, aug_config: Optional[Dict] = None):
        super().__init__()

        if aug_config is None:
            # 默认配置：只使用ColorJitter和GaussianNoise
            aug_config = {
                'color_jitter': {'enabled': True, 'brightness': 0.2, 'contrast': 0.2,
                                'saturation': 0.1, 'hue': 0.0, 'p': 0.5},
                'gaussian_noise': {'enabled': True, 'mean': 0.0, 'std': 0.03, 'p': 0.5},
            }

        # 动态构建增强pipeline
        transforms = []

        # 1. 光照增强 - ColorJitter
        if aug_config.get('color_jitter', {}).get('enabled', False):
            cfg = aug_config['color_jitter']
            transforms.append(KA.ColorJitter(
                brightness=cfg.get('brightness', 0.2),
                contrast=cfg.get('contrast', 0.2),
                saturation=cfg.get('saturation', 0.1),
                hue=cfg.get('hue', 0.0),
                p=cfg.get('p', 0.5)
            ))

        # 2. 传感器噪声 - GaussianNoise
        if aug_config.get('gaussian_noise', {}).get('enabled', False):
            cfg = aug_config['gaussian_noise']
            transforms.append(KA.RandomGaussianNoise(
                mean=cfg.get('mean', 0.0),
                std=cfg.get('std', 0.03),
                p=cfg.get('p', 0.5)
            ))

        # 3. 模糊增强 - GaussianBlur
        if aug_config.get('gaussian_blur', {}).get('enabled', False):
            cfg = aug_config['gaussian_blur']
            transforms.append(KA.RandomGaussianBlur(
                kernel_size=tuple(cfg.get('kernel_size', [5, 5])),
                sigma=tuple(cfg.get('sigma', [0.3, 1.0])),
                p=cfg.get('p', 0.15)
            ))

        # 4. 锐化 - Sharpness
        if aug_config.get('sharpness', {}).get('enabled', False):
            cfg = aug_config['sharpness']
            transforms.append(KA.RandomSharpness(
                sharpness=cfg.get('sharpness', 0.5),
                p=cfg.get('p', 0.3)
            ))

        # 5. 直方图均衡 - Equalize
        if aug_config.get('equalize', {}).get('enabled', False):
            cfg = aug_config['equalize']
            transforms.append(KA.RandomEqualize(
                p=cfg.get('p', 0.3)
            ))

        # 6. 运动模糊 - MotionBlur
        if aug_config.get('motion_blur', {}).get('enabled', False):
            cfg = aug_config['motion_blur']
            transforms.append(KA.RandomMotionBlur(
                kernel_size=cfg.get('kernel_size', 5),
                angle=cfg.get('angle', 45.0),
                direction=cfg.get('direction', 0.5),
                p=cfg.get('p', 0.3)
            ))

        # 7. 随机擦除 - RandomErasing（注意：与heatmap监督冲突）
        if aug_config.get('random_erasing', {}).get('enabled', False):
            cfg = aug_config['random_erasing']
            transforms.append(KA.RandomErasing(
                p=cfg.get('p', 0.3),
                scale=tuple(cfg.get('scale', [0.02, 0.1])),
                ratio=tuple(cfg.get('ratio', [0.3, 3.3]))
            ))

        # 如果没有任何增强启用，使用Identity
        if len(transforms) == 0:
            transforms.append(torch.nn.Identity())

        self.transforms = torch.nn.Sequential(*transforms)
        self.num_transforms = len(transforms)

    def forward(self, input_tensor):
        B, T, C, H, W = input_tensor.shape
        input_tensor_aug = self.transforms(
            input_tensor.reshape(B * T, C, H, W)
        ).reshape(B, T, C, H, W)
        input_tensor_aug = torch.clamp(input_tensor_aug, 0.0, 1.0)
        return input_tensor_aug


def get_trans_3d_mat(
    rad: torch.Tensor,
    scale: torch.Tensor,
    axis_angle: torch.Tensor,
) -> torch.Tensor:
    """
    生成三维空间中的旋转变换增强矩阵

    Args:
        rad: [...] 以z方向为旋转轴进行旋转的角度
        scale: [...] 对z分量进行缩放的系数
        axis_angle: [..., 3] 全场景物体以该轴角表示的旋转进行变换的旋转轴角，若为空则不进行变换

    Returns:
        mat: [..., 3, 3] 以上三个变换首先旋转然后缩放最后全局旋转，对应的变换矩阵
    """
    device = rad.device
    dtype = rad.dtype

    cos_rad = torch.cos(rad)
    sin_rad = torch.sin(rad)
    prefix_shape = rad.shape

    mat = torch.zeros(*prefix_shape, 3, 3, device=device, dtype=dtype)
    mat[..., 0, 0] = cos_rad
    mat[..., 1, 0] = sin_rad
    mat[..., 0, 1] = -sin_rad
    mat[..., 1, 1] = cos_rad
    mat[..., 2, 2] = scale

    if axis_angle is not None:
        prefix_shape = axis_angle.shape[:-1]
        axis_angle_mat = KC.axis_angle_to_rotation_matrix(
            axis_angle.reshape(-1, 3)
        ).reshape(*prefix_shape, 3, 3)
        mat = axis_angle_mat @ mat

    return mat

def get_trans_2d_mat(
    rad: torch.Tensor,
    scale_inv: torch.Tensor,
    focal_old: torch.Tensor,
    princpt_old: torch.Tensor,
    focal_new: torch.Tensor,
    princpt_new: torch.Tensor,
    axis_angle: torch.Tensor,
) -> torch.Tensor:
    """
    生成图像空间中对应的变换矩阵，其包括三维增强产生的变换以及内参增强产生的变换

    Args:
        rad: [...] 以z方向为旋转轴进行旋转的角度
        scale_inv: [...] 对z分量进行缩放的系数的倒数，因为z的远离（系数>1）对应图像的缩小
        focal_old/new: [..., 2] 旧/新内参的焦距系数
        princpt_old/new: [..., 2] 旧/新内参的主点系数
        axis_angle: [..., 3] 全场景物体以该轴角表示的旋转进行变换的旋转轴角，若为空则不进行变换

    Returns:
        mat: [..., 3, 3] 首先进行旋转，然后进行缩放，最后进行内参变换。\
            这三个变换的矩阵的乘积形成的对二维坐标的透视变换矩阵
    """
    device = rad.device
    dtype = rad.dtype
    prefix_shape = rad.shape

    # 原始内参求逆
    old_intr_inv = torch.zeros(*prefix_shape, 3, 3, device=device, dtype=dtype)
    old_intr_inv[..., 0, 0] = 1 / focal_old[..., 0]
    old_intr_inv[..., 1, 1] = 1 / focal_old[..., 1]
    old_intr_inv[..., 0, 2] = -princpt_old[..., 0] / focal_old[..., 0]
    old_intr_inv[..., 1, 2] = -princpt_old[..., 1] / focal_old[..., 1]
    old_intr_inv[..., 2, 2] = 1

    # 构造新的内参矩阵
    new_intr = torch.zeros(*prefix_shape, 3, 3, device=device, dtype=dtype)
    new_intr[..., 0, 0] = focal_new[..., 0]
    new_intr[..., 1, 1] = focal_new[..., 1]
    new_intr[..., 0, 2] = princpt_new[..., 0]
    new_intr[..., 1, 2] = princpt_new[..., 1]
    new_intr[..., 2, 2] = 1

    # 旋转矩阵，直接copy3d的，以及缩放矩阵，两个合到一起写
    cos_rad = torch.cos(rad)
    sin_rad = torch.sin(rad)
    mat = torch.zeros(*prefix_shape, 3, 3, device=device, dtype=dtype)
    mat[..., 0, 0] = cos_rad
    mat[..., 1, 0] = sin_rad
    mat[..., 0, 1] = -sin_rad
    mat[..., 1, 1] = cos_rad
    mat = mat * scale_inv[..., None, None]
    mat[..., 2, 2] = 1

    if axis_angle is not None:
        prefix_shape = axis_angle.shape[:-1]
        axis_angle_mat = KC.axis_angle_to_rotation_matrix(
            axis_angle.reshape(-1, 3)
        ).reshape(*prefix_shape, 3, 3)
        mat = axis_angle_mat @ mat

    return new_intr @ mat @ old_intr_inv

def apply_perspective_to_points(
    trans_matrix: torch.Tensor,
    points: torch.Tensor
) -> torch.Tensor:
    """
    对任意维度 D 的点集应用 (D+1)x(D+1) 的变换矩阵。
    支持透视变换（自动执行透视除法）。

    Args:
        trans_matrix: [..., D+1, D+1] 变换矩阵
        points: [..., N, D] 坐标点

    Returns:
        points: [..., N, D] 变换后的点
    """
    # 1. 维度检查
    D = points.shape[-1]
    if trans_matrix.shape[-1] != D + 1 or trans_matrix.shape[-2] != D + 1:
        raise ValueError(f"Matrix shape {trans_matrix.shape} does not match point dim {D}+1")

    # 2. 转换为齐次坐标 (..., N, D) -> (..., N, D+1)
    # 例如：[x, y] -> [x, y, 1]
    points_h = KC.convert_points_to_homogeneous(points)

    # 3. 应用矩阵变换
    # Kornia/PyTorch 中点通常是行向量，公式为: P_out = P_in * H^T
    # trans_matrix.transpose(-1, -2) 用于适应行向量乘法
    # 这一步利用广播机制支持 Batch
    points_h_transformed = points_h @ trans_matrix.transpose(-1, -2)

    # 4. 从齐次坐标还原 (..., N, D+1) -> (..., N, D)
    # 这一步会自动执行透视除法: coords / w
    # Kornia 的实现通常包含数值稳定性处理 (epsilon)
    points_transformed = KC.convert_points_from_homogeneous(points_h_transformed)

    return points_transformed


def build_intrinsic_matrices(
    focal: torch.Tensor,
    princpt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    构建内参矩阵及其逆矩阵。

    Args:
        focal: [..., 2] 焦距 (fx, fy)
        princpt: [..., 2] 主点 (cx, cy)

    Returns:
        intr: [..., 3, 3] 内参矩阵 K
        intr_inv: [..., 3, 3] 内参逆矩阵 K^{-1}
    """
    prefix_shape = focal.shape[:-1]
    device = focal.device
    dtype = focal.dtype

    intr = torch.zeros(*prefix_shape, 3, 3, device=device, dtype=dtype)
    intr[..., 0, 0] = focal[..., 0]
    intr[..., 1, 1] = focal[..., 1]
    intr[..., 0, 2] = princpt[..., 0]
    intr[..., 1, 2] = princpt[..., 1]
    intr[..., 2, 2] = 1.0

    intr_inv = torch.zeros(*prefix_shape, 3, 3, device=device, dtype=dtype)
    intr_inv[..., 0, 0] = 1.0 / focal[..., 0]
    intr_inv[..., 1, 1] = 1.0 / focal[..., 1]
    intr_inv[..., 0, 2] = -princpt[..., 0] / focal[..., 0]
    intr_inv[..., 1, 2] = -princpt[..., 1] / focal[..., 1]
    intr_inv[..., 2, 2] = 1.0

    return intr, intr_inv


def compute_perspective_normalization_rotation(
    hand_bbox: torch.Tensor,
    focal: torch.Tensor,
    princpt: torch.Tensor,
) -> torch.Tensor:
    """
    计算将 bbox 中心旋转到主点（光轴）方向的透视矫正旋转矩阵。
    满足 R @ ray_norm ≈ [0, 0, 1]，即将 bbox 中心的视线方向旋转到光轴。

    Args:
        hand_bbox: [B, T, 4] xyxy 格式的手部包围盒
        focal: [B, T, 2] 焦距 (fx, fy)
        princpt: [B, T, 2] 主点 (cx, cy)

    Returns:
        R: [B, T, 3, 3] 旋转矩阵
    """
    # 1. bbox 中心的归一化相机方向
    cx_hand = (hand_bbox[..., 0] + hand_bbox[..., 2]) * 0.5  # [B, T]
    cy_hand = (hand_bbox[..., 1] + hand_bbox[..., 3]) * 0.5  # [B, T]

    rx = (cx_hand - princpt[..., 0]) / focal[..., 0]  # [B, T]
    ry = (cy_hand - princpt[..., 1]) / focal[..., 1]  # [B, T]
    rz = torch.ones_like(rx)                            # [B, T]

    # 2. 归一化为单位向量
    norm = torch.sqrt(rx**2 + ry**2 + rz**2)
    rx = rx / norm
    ry = ry / norm
    rz = rz / norm

    # 3. 旋转轴 = cross(ray_norm, [0,0,1]) = [ry, -rx, 0]
    #    旋转角度 = arccos(rz)，用 atan2 更稳定
    ax = ry   # [B, T]
    ay = -rx  # [B, T]
    sin_angle = torch.sqrt(ax**2 + ay**2)  # = sin(arccos(rz))
    angle = torch.atan2(sin_angle, rz)     # [B, T]

    # 4. 轴角: axis_angle = (axis / |axis|) * angle
    #    退化处理: sin_angle ≈ 0 时 bbox 中心在主点上, R = I
    safe_sin = torch.clamp(sin_angle, min=1e-8)

    axis_angle = torch.stack([
        ax / safe_sin * angle,
        ay / safe_sin * angle,
        torch.zeros_like(ax),
    ], dim=-1)  # [B, T, 3]

    # sin_angle < 1e-7 → 强制零向量 → R = I
    degenerate_mask = (sin_angle < 1e-7).unsqueeze(-1).expand_as(axis_angle)
    axis_angle = axis_angle.masked_fill(degenerate_mask, 0.0)

    # 5. 轴角 → 旋转矩阵
    prefix_shape = axis_angle.shape[:-1]  # (B, T)
    R = KC.axis_angle_to_rotation_matrix(
        axis_angle.reshape(-1, 3)
    ).reshape(*prefix_shape, 3, 3)  # [B, T, 3, 3]

    return R


def _compute_resized_patch_joints(
    joint_patch_origin: torch.Tensor,
    patch_bbox: torch.Tensor,
    patch_size: Tuple[int, int],
) -> torch.Tensor:
    patch_h, patch_w = patch_size
    patch_width = torch.clamp(patch_bbox[..., 2] - patch_bbox[..., 0], min=1e-6)
    patch_height = torch.clamp(patch_bbox[..., 3] - patch_bbox[..., 1], min=1e-6)

    joint_patch_resized = torch.empty_like(joint_patch_origin)
    joint_patch_resized[..., 0] = (
        joint_patch_origin[..., 0] * patch_w / patch_width[..., None]
    )
    joint_patch_resized[..., 1] = (
        joint_patch_origin[..., 1] * patch_h / patch_height[..., None]
    )
    return joint_patch_resized


@torch.no_grad()
def preprocess_batch(
    batch_origin,
    patch_size: Tuple[int, int],
    patch_expanstion: float,
    scale_z_range: Tuple[float, float],
    scale_f_range: Tuple[float, float],
    persp_rot_max: float,
    joint_rep_type: str,
    augmentation_flag: bool,
    device: torch.device,
    pixel_aug=None,
    perspective_normalization: bool = False,
):
    """
    将wds的原始数据进行预处理和数据增强，最后送给模型

    Args:
        patch_size: 输出给模型的图像patch的大小
        patch_expansion: patch相较于正方形包围盒扩大的范围
        scale_z_range: 进行缩放/平移增强变换的系数范围
        scale_f_range: 进行内参增强变换的焦距乘数的范围
        pixel_aug: 像素级增强器对象（可选），由调用方创建和管理
        perspective_normalization: 是否进行透视归一化（将bbox中心旋转到主点）
    """
    batch_size, num_frames = batch_origin["joint_img"].shape[:2]
    trans_2d_mat = (
        torch.eye(3, device=device, dtype=torch.float32)[None, None]
        .repeat(batch_size, num_frames, 1, 1)
    )
    correction_rot_mat = None

    sample_keys = batch_origin.get(
        "__key__", [f"sample_{idx}" for idx in range(batch_size)]
    )
    imgs_path: List[List[str]] = batch_origin["imgs_path"]
    handedness: List[str] = batch_origin["handedness"]
    data_source: List[str] = batch_origin.get(
        "data_source", ["unknown" for _ in range(batch_size)]
    )
    source_split: List[str] = batch_origin.get(
        "source_split", ["unknown" for _ in range(batch_size)]
    )
    source_index = batch_origin.get(
        "source_index",
        [
            [{"frame_idx_within_clip": frame_idx} for frame_idx in range(num_frames)]
            for _ in range(batch_size)
        ],
    )
    intr_type: List[str] = batch_origin.get(
        "intr_type", ["unknown" for _ in range(batch_size)]
    )
    additional_desc = batch_origin.get(
        "additional_desc",
        [[{} for _ in range(num_frames)] for _ in range(batch_size)],
    )
    flip: List[bool] = [value == "left" for value in handedness]

    hand_bbox: torch.Tensor = batch_origin["hand_bbox"].to(device).clone()
    joint_img: torch.Tensor = batch_origin["joint_img"].to(device).clone()
    joint_cam: torch.Tensor = batch_origin["joint_cam"].to(device).clone()
    joint_rel: torch.Tensor = batch_origin["joint_rel"].to(device).clone()
    joint_2d_valid: torch.Tensor = batch_origin.get(
        "joint_2d_valid", batch_origin["joint_valid"]
    ).to(device).clone()
    joint_3d_valid: torch.Tensor = batch_origin.get(
        "joint_3d_valid", batch_origin["joint_valid"]
    ).to(device).clone()
    mano_pose: torch.Tensor = batch_origin["mano_pose"].to(device).clone()
    mano_shape: torch.Tensor = batch_origin["mano_shape"].to(device).clone()
    has_mano: torch.Tensor = batch_origin.get(
        "has_mano", batch_origin["mano_valid"]
    ).to(device).clone()
    timestamp: torch.Tensor = batch_origin["timestamp"].to(device).clone()
    focal: torch.Tensor = batch_origin["focal"].to(device).clone()
    princpt: torch.Tensor = batch_origin["princpt"].to(device).clone()
    has_intr_origin = batch_origin.get("has_intr", None)
    if has_intr_origin is None:
        has_intr = torch.ones_like(timestamp)
    else:
        has_intr = has_intr_origin.to(device).clone()
    has_intr_mask = has_intr > 0.5

    if not augmentation_flag and not perspective_normalization:
        hand_bbox_center = (hand_bbox[..., :2] + hand_bbox[..., 2:]) * 0.5
        width, height = torch.split(hand_bbox[..., 2:] - hand_bbox[..., :2], 1, dim=-1)
        half_edge_len = torch.max(width, height) * patch_expanstion * 0.5
        patch_bbox_corners = torch.stack(
            [
                hand_bbox_center - half_edge_len,
                torch.stack(
                    [
                        hand_bbox_center[..., 0] + half_edge_len[..., 0],
                        hand_bbox_center[..., 1] - half_edge_len[..., 0],
                    ],
                    dim=-1,
                ),
                hand_bbox_center + half_edge_len,
                torch.stack(
                    [
                        hand_bbox_center[..., 0] - half_edge_len[..., 0],
                        hand_bbox_center[..., 1] + half_edge_len[..., 0],
                    ],
                    dim=-1,
                ),
                ],
            dim=2,
        )
        patch_bbox: torch.Tensor = torch.cat(
            [patch_bbox_corners[:, :, 0], patch_bbox_corners[:, :, 2]], dim=-1
        )
        patches = []
        for bx, img_orig_tensor in enumerate(batch_origin["imgs"]):
            patch = KT.crop_and_resize(
                img_orig_tensor.to(device).float() / 255.0,
                patch_bbox_corners[bx],
                patch_size,
                mode="bilinear",
            )
            patches.append(patch)
        patches = torch.stack(patches)
    else:
        safe_focal = torch.where(has_intr_mask[..., None], focal, torch.ones_like(focal))
        safe_princpt = torch.where(
            has_intr_mask[..., None], princpt, torch.zeros_like(princpt)
        )

        if perspective_normalization:
            correction_rot_candidate = compute_perspective_normalization_rotation(
                hand_bbox, safe_focal, safe_princpt
            )
            eye = torch.eye(3, device=device, dtype=torch.float32)[None, None].repeat(
                batch_size, num_frames, 1, 1
            )
            correction_rot_mat = torch.where(
                has_intr_mask[..., None, None],
                correction_rot_candidate,
                eye,
            )

        if augmentation_flag:
            rad = torch.rand(batch_size, 1, device=device).expand(-1, num_frames)
            rad = rad * 2.0 * torch.pi
            scale_z = (
                torch.rand(batch_size, 1, device=device).expand(-1, num_frames)
                * (scale_z_range[1] - scale_z_range[0])
                + scale_z_range[0]
            )
            scale_f = (
                torch.rand(batch_size, 1, device=device).expand(-1, num_frames)
                * (scale_f_range[1] - scale_f_range[0])
                + scale_f_range[0]
            )
            focal_new = focal * scale_f[:, :, None]
            princpt_noise = torch.randn(batch_size, 1, 2, device=device).expand(
                -1, num_frames, -1
            )
            princpt_noise = (
                princpt_noise
                * torch.norm(princpt, dim=-1, keepdim=True)
                * 0.1111111
            )
            princpt_new = princpt_noise + princpt
            persp_dir_rad = (
                torch.rand(batch_size, 1, device=device).expand(-1, num_frames)
                * 2.0
                * torch.pi
            )
            persp_rot_rad = (
                torch.rand(batch_size, 1, device=device).expand(-1, num_frames)
                * persp_rot_max
            )
            persp_axis_angle = (
                torch.stack(
                    [
                        torch.cos(persp_dir_rad),
                        torch.sin(persp_dir_rad),
                        torch.zeros(batch_size, num_frames, device=device),
                    ],
                    dim=-1,
                )
                * persp_rot_rad[..., None]
            )
            if perspective_normalization:
                persp_axis_angle = torch.zeros(
                    batch_size, num_frames, 3, device=device, dtype=torch.float32
                )
        else:
            rad = torch.zeros(batch_size, num_frames, device=device)
            scale_z = torch.ones(batch_size, num_frames, device=device)
            focal_new = focal.clone()
            princpt_new = princpt.clone()
            persp_axis_angle = torch.zeros(
                batch_size, num_frames, 3, device=device, dtype=torch.float32
            )

        safe_focal_new = torch.where(
            has_intr_mask[..., None], focal_new, torch.ones_like(focal_new)
        )
        safe_princpt_new = torch.where(
            has_intr_mask[..., None], princpt_new, torch.zeros_like(princpt_new)
        )

        trans_3d_mat = get_trans_3d_mat(rad, scale_z, persp_axis_angle)
        trans_2d_candidate = get_trans_2d_mat(
            rad,
            1.0 / scale_z,
            safe_focal,
            safe_princpt,
            safe_focal_new,
            safe_princpt_new,
            persp_axis_angle,
        )
        trans_2d_mat = torch.where(
            has_intr_mask[..., None, None],
            trans_2d_candidate,
            trans_2d_mat,
        )

        if correction_rot_mat is not None:
            trans_3d_mat = trans_3d_mat @ correction_rot_mat
            old_intr, old_intr_inv = build_intrinsic_matrices(safe_focal, safe_princpt)
            correction_2d = old_intr @ correction_rot_mat @ old_intr_inv
            trans_2d_mat = torch.where(
                has_intr_mask[..., None, None],
                trans_2d_mat @ correction_2d,
                trans_2d_mat,
            )

        frame_has_3d = torch.any(joint_3d_valid > 0.5, dim=-1)
        joint_cam_aug = torch.einsum("...jd,...nd->...jn", joint_cam, trans_3d_mat)
        joint_rel_aug = joint_cam_aug - joint_cam_aug[:, :, :1]
        joint_cam = torch.where(frame_has_3d[..., None, None], joint_cam_aug, joint_cam)
        joint_rel = torch.where(frame_has_3d[..., None, None], joint_rel_aug, joint_rel)

        frame_has_mano = (has_mano > 0.5).reshape(-1)
        if torch.any(frame_has_mano):
            mano_pose_flat = mano_pose.reshape(-1, mano_pose.shape[-1]).clone()
            root_axis_angle = torch.zeros(
                (batch_size * num_frames, 3),
                device=device,
                dtype=mano_pose.dtype,
            )
            root_axis_angle[:, 2] = rad.reshape(-1)
            root_rot_mat = KC.axis_angle_to_rotation_matrix(root_axis_angle)
            if correction_rot_mat is not None:
                root_rot_mat = root_rot_mat @ correction_rot_mat.reshape(-1, 3, 3)
            if persp_rot_max > 0 and not perspective_normalization:
                persp_rot_mat = KC.axis_angle_to_rotation_matrix(
                    persp_axis_angle.reshape(-1, 3)
                )
                root_rot_mat = persp_rot_mat @ root_rot_mat
            mano_root_rot = KC.axis_angle_to_rotation_matrix(mano_pose_flat[:, :3])
            mano_root_rot = KC.rotation_matrix_to_axis_angle(root_rot_mat @ mano_root_rot)
            mano_pose_flat[frame_has_mano, :3] = mano_root_rot[frame_has_mano]
            mano_pose = mano_pose_flat.reshape(batch_size, num_frames, -1)

        joint_mask = (joint_2d_valid < 0.5)[..., None].expand(-1, -1, -1, 2)
        joint_img = apply_perspective_to_points(trans_2d_mat, joint_img)

        min_xy = torch.min(
            joint_img.masked_fill(joint_mask, float("inf")), dim=-2
        ).values
        max_xy = torch.max(
            joint_img.masked_fill(joint_mask, -float("inf")), dim=-2
        ).values
        hand_bbox_new = torch.cat([min_xy, max_xy], dim=-1)
        valid_joint_count = torch.sum(joint_2d_valid > 0.5, dim=-1, keepdim=True)
        no_valid_joint = valid_joint_count == 0
        hand_bbox = torch.where(no_valid_joint.expand_as(hand_bbox), hand_bbox, hand_bbox_new)

        center = (hand_bbox[..., :2] + hand_bbox[..., 2:]) * 0.5
        half_edge_len = (
            torch.max(hand_bbox[..., 2:] - hand_bbox[..., :2], dim=-1).values
            * 0.5
            * patch_expanstion
        )
        patch_bbox = torch.cat(
            [
                center[..., 0:1] - half_edge_len[..., None],
                center[..., 1:2] - half_edge_len[..., None],
                center[..., 0:1] + half_edge_len[..., None],
                center[..., 1:2] + half_edge_len[..., None],
            ],
            dim=-1,
        )

        sx = (patch_bbox[..., 2] - patch_bbox[..., 0]) / patch_size[1]
        sy = (patch_bbox[..., 3] - patch_bbox[..., 1]) / patch_size[0]
        a_inv = torch.zeros(batch_size, num_frames, 3, 3, device=device, dtype=torch.float32)
        a_inv[..., 0, 0] = 1.0 / sx
        a_inv[..., 1, 1] = 1.0 / sy
        a_inv[..., 0, 2] = -patch_bbox[..., 0] / sx
        a_inv[..., 1, 2] = -patch_bbox[..., 1] / sy
        a_inv[..., 2, 2] = 1.0
        m_crop = a_inv @ trans_2d_mat

        patches = []
        for bx, img_orig_tensor in enumerate(batch_origin["imgs"]):
            patch = KT.warp_perspective(
                img_orig_tensor.to(device).float() / 255.0,
                m_crop[bx],
                tuple(patch_size),
                mode="bilinear",
            )
            patches.append(patch)
        patches = torch.stack(patches)

        if pixel_aug is not None:
            patches = pixel_aug(patches)

        focal = torch.where(has_intr_mask[..., None], focal_new, focal)
        princpt = torch.where(has_intr_mask[..., None], princpt_new, princpt)

    joint_hand_origin = joint_img - hand_bbox[:, :, None, :2]
    joint_patch_origin = joint_img - patch_bbox[:, :, None, :2]

    for bx, do_flip in enumerate(flip):
        if not do_flip:
            continue

        _, _, _, width = batch_origin["imgs"][bx].shape
        patch_bbox_w = patch_bbox[bx, :, 2] - patch_bbox[bx, :, 0]
        hand_bbox_w = hand_bbox[bx, :, 2] - hand_bbox[bx, :, 0]

        patches[bx] = torch.flip(patches[bx], dims=[-1])
        patch_bbox[bx, :, 0], patch_bbox[bx, :, 2] = (
            width - patch_bbox[bx, :, 2],
            width - patch_bbox[bx, :, 0],
        )
        hand_bbox[bx, :, 0], hand_bbox[bx, :, 2] = (
            width - hand_bbox[bx, :, 2],
            width - hand_bbox[bx, :, 0],
        )
        joint_img[bx, :, :, 0] = width - joint_img[bx, :, :, 0]
        joint_patch_origin[bx, :, :, 0] = (
            patch_bbox_w[:, None] - joint_patch_origin[bx, :, :, 0]
        )
        joint_hand_origin[bx, :, :, 0] = (
            hand_bbox_w[:, None] - joint_hand_origin[bx, :, :, 0]
        )
        joint_cam[bx, :, :, 0] *= -1.0
        joint_rel[bx, :, :, 0] *= -1.0

        mano_pose_bx = rearrange(mano_pose[bx], "t (j d) -> t j d", d=3)
        mano_pose_bx[:, :, 1:] *= -1.0
        mano_pose[bx] = rearrange(mano_pose_bx, "t j d -> t (j d)")

        intr_mask = has_intr[bx] > 0.5
        princpt[bx, intr_mask, 0] = width - princpt[bx, intr_mask, 0]

    joint_patch_resized = _compute_resized_patch_joints(
        joint_patch_origin,
        patch_bbox,
        patch_size,
    )

    if joint_rep_type == "3":
        pass
    elif joint_rep_type == "6d":
        batch_size, num_frames = mano_pose.shape[:2]
        mano_pose = rearrange(
            mano_pose, "b t (j d) -> (b t j) d", j=MANO_JOINT_COUNT
        )
        mano_pose = KC.axis_angle_to_rotation_matrix(mano_pose)
        mano_pose = rotation_matrix_to_rotation6d(mano_pose)
        mano_pose = rearrange(
            mano_pose, "(b t j) d -> b t (j d)", b=batch_size, t=num_frames
        )
    elif joint_rep_type == "quat":
        batch_size, num_frames = mano_pose.shape[:2]
        mano_pose = rearrange(
            mano_pose, "b t (j d) -> (b t j) d", j=MANO_JOINT_COUNT
        )
        mano_pose = KC.axis_angle_to_quaternion(mano_pose)
        mano_pose = rearrange(
            mano_pose, "(b t j) d -> b t (j d)", b=batch_size, t=num_frames
        )
    else:
        raise NotImplementedError(f"Unsupported rotation type={joint_rep_type}")

    valid_mano_mask = has_mano > 0.5
    if torch.any(valid_mano_mask) and mano_pose[valid_mano_mask].isnan().any().cpu().item():
        raise ValueError("MANO contains NaN")

    batch_out = {
        "__key__": sample_keys,
        "imgs_path": imgs_path,
        "handedness": handedness,
        "data_source": data_source,
        "source_split": source_split,
        "source_index": source_index,
        "intr_type": intr_type,
        "additional_desc": additional_desc,
        "flip": flip,
        "patches": patches,
        "patch_bbox": patch_bbox,
        "hand_bbox": hand_bbox,
        "joint_img": joint_img,
        "joint_hand_origin": joint_hand_origin,
        "joint_patch_origin": joint_patch_origin,
        "joint_patch_resized": joint_patch_resized,
        "joint_hand_bbox": joint_hand_origin,
        "joint_patch_bbox": joint_patch_origin,
        "joint_cam": joint_cam,
        "joint_rel": joint_rel,
        "joint_2d_valid": joint_2d_valid,
        "joint_3d_valid": joint_3d_valid,
        "joint_valid": joint_2d_valid,
        "mano_pose": mano_pose,
        "mano_shape": mano_shape,
        "has_mano": has_mano,
        "mano_valid": has_mano,
        "has_intr": has_intr,
        "timestamp": timestamp,
        "focal": focal,
        "princpt": princpt,
    }
    return batch_out, trans_2d_mat, correction_rot_mat
