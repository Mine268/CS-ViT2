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
    pixel_aug = None,
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
    # pixel_aug由调用方传入，不在此创建
    B, T = batch_origin["joint_cam"].shape[:2]
    trans_2d_mat = torch.eye(3, device=device).float()[None, None, :].expand(B, T, -1, -1)

    correction_rot_mat = None

    if not augmentation_flag and not perspective_normalization:
        # 数据规整
        # imgs_path, flip
        imgs_path: List[List[str]] = batch_origin["imgs_path"]
        flip: List[bool] = [v == "left" for v in batch_origin["handedness"]]

        # 计算patches对应的范围，以及patch图像分割
        # hand_bbox, patches_bbox, patches
        hand_bbox: torch.Tensor = batch_origin["hand_bbox"].to(device)  # [B,T,4]
        hand_bbox_center = (hand_bbox[..., :2] + hand_bbox[..., 2:]) * 0.5  # [B,T,2]
        width, height = torch.split(hand_bbox[..., 2:] - hand_bbox[..., :2], 1, dim=-1)  # [B,T]
        half_edge_len = torch.max(width, height) * patch_expanstion * 0.5  # [B,T]
        patch_bbox_k = torch.stack(  # [B,T,4,2]
            [
                hand_bbox_center - half_edge_len,
                torch.stack([
                    hand_bbox_center[..., 0] + half_edge_len[..., 0],
                    hand_bbox_center[..., 1] - half_edge_len[..., 0]
                ], dim=-1),
                hand_bbox_center + half_edge_len,
                torch.stack([
                    hand_bbox_center[..., 0] - half_edge_len[..., 0],
                    hand_bbox_center[..., 1] + half_edge_len[..., 0]
                ], dim=-1),
            ], dim=2
        )
        patch_bbox: torch.Tensor = torch.cat(
            [patch_bbox_k[:, :, 0], patch_bbox_k[:, :, 2]], dim=-1
        )  # [B,T,4]
        patches = []
        for bx, img_orig_tensor in enumerate(batch_origin["imgs"]):
            patch = KT.crop_and_resize(
                img_orig_tensor.to(device).float() / 255,
                patch_bbox_k[bx],
                patch_size,
                mode="bilinear"
            )
            patches.append(patch)
        patches: torch.Tensor = torch.stack(patches)

        # 处理关节点二维位置
        joint_img: torch.Tensor = batch_origin["joint_img"].to(device)  # [B,T,J,2]
        joint_patch_bbox: torch.Tensor = joint_img - patch_bbox_k[:, :, :1]  # [B,T,J,2]
        joint_hand_bbox: torch.Tensor = joint_img - hand_bbox[:, :, None, :2]  # [B,T,J,2]

        # 三维位置
        joint_cam: torch.Tensor = batch_origin["joint_cam"].to(device)  # [B,T,J,3]
        joint_rel: torch.Tensor = batch_origin["joint_rel"].to(device)  # [B,T,J,3]
        joint_valid: torch.Tensor = batch_origin["joint_valid"].to(device)  # [B,T,J]

        # MANO标注
        mano_pose: torch.Tensor = batch_origin["mano_pose"].to(device)  # [B,T,48]
        mano_shape: torch.Tensor = batch_origin["mano_shape"].to(device)  # [B,T,10]
        mano_valid: torch.Tensor = batch_origin["mano_valid"].to(device)  # [B,T,10]

        # timestamp
        timestamp: torch.Tensor = batch_origin["timestamp"].to(device)  # [B,T]

        # focal, princpt
        focal: torch.Tensor = batch_origin["focal"].to(device)  # [B,T,2]
        princpt: torch.Tensor = batch_origin["princpt"].to(device)  # [B,T,2]
    else:
        # focal, princpt
        focal: torch.Tensor = batch_origin["focal"].to(device)  # [B,T,2]
        princpt: torch.Tensor = batch_origin["princpt"].to(device)  # [B,T,2]

        # === perspective normalization（独立于 augmentation） ===
        if perspective_normalization:
            hand_bbox_orig: torch.Tensor = batch_origin["hand_bbox"].to(device)
            correction_rot_mat = compute_perspective_normalization_rotation(
                hand_bbox_orig, focal, princpt
            )  # [B,T,3,3]，基于原始内参

        # === 增强参数 ===
        if augmentation_flag:
            rad = torch.rand(B, 1, device=device).expand(-1, T) * 2 * torch.pi  # [B, T]
            # [B, T], [B, T]
            scale_z = (
                torch.rand(B, 1, device=device).expand(-1, T)
                * (scale_z_range[1] - scale_z_range[0])
                + scale_z_range[0]
            )
            scale_f = (
                torch.rand(B, 1, device=device).expand(-1, T)
                * (scale_f_range[1] - scale_f_range[0])
                + scale_f_range[0]
            )
            focal_new = focal * scale_f[:, :, None]
            princpt_noise = torch.randn(B, 1, 2, device=device).expand(-1, T, -1)
            princpt_noise = princpt_noise * torch.norm(princpt, dim=-1, keepdim=True) * 0.1111111
            princpt_new = princpt_noise + princpt
            persp_dir_rad = torch.rand(B, 1, device=device).expand(-1, T) * 2 * torch.pi
            persp_rot_rad = torch.rand(B, 1, device=device).expand(-1, T) * persp_rot_max
            persp_axis_angle = (
                torch.stack(
                    [
                        torch.cos(persp_dir_rad),
                        torch.sin(persp_dir_rad),
                        torch.zeros(B, T, device=device),
                    ],
                    dim=-1,
                )
                * persp_rot_rad[..., None]
            )
            # 归一化与透视增强互斥
            if perspective_normalization:
                persp_axis_angle = torch.zeros(B, T, 3, device=device)
        else:
            # 不做增强 → identity 参数（仅 perspective_normalization 走此路径）
            rad = torch.zeros(B, T, device=device)
            scale_z = torch.ones(B, T, device=device)
            focal_new = focal.clone()
            princpt_new = princpt.clone()
            persp_axis_angle = torch.zeros(B, T, 3, device=device)

        # === 组合变换矩阵 ===
        # [B,T,3,3]
        trans_3d_mat = get_trans_3d_mat(rad, scale_z, persp_axis_angle)
        trans_2d_mat = get_trans_2d_mat(
            rad, 1 / scale_z, focal, princpt, focal_new, princpt_new, persp_axis_angle
        )
        # 归一化旋转右乘
        if correction_rot_mat is not None:
            trans_3d_mat = trans_3d_mat @ correction_rot_mat
            old_intr, old_intr_inv = build_intrinsic_matrices(focal, princpt)
            correction_2d = old_intr @ correction_rot_mat @ old_intr_inv
            trans_2d_mat = trans_2d_mat @ correction_2d

        # 带数据增强的数据规整
        # imgs_path, flip
        imgs_path: List[List[str]] = batch_origin["imgs_path"]
        flip: List[bool] = [v == "left" for v in batch_origin["handedness"]]

        # 重新计算三维关节点
        # joint_cam, joint_rel, joint_valid
        joint_cam: torch.Tensor = batch_origin["joint_cam"].to(device)  # [B,T,J,3]
        joint_cam = torch.einsum("...jd,...nd->...jn", joint_cam, trans_3d_mat)
        joint_rel = joint_cam - joint_cam[:, :, :1]
        joint_valid: torch.Tensor = batch_origin["joint_valid"].to(device)  # [B,T,J]

        # 对MANO参数进行变换
        mano_pose: torch.Tensor = batch_origin["mano_pose"].to(device)  # [B,T,48]
        mano_shape: torch.Tensor = batch_origin["mano_shape"].to(device)  # [B,T,10]
        mano_valid: torch.Tensor = batch_origin["mano_valid"].to(device)  # [B,T]
        mano_pose_root = KC.axis_angle_to_rotation_matrix(
            mano_pose[:, :, :3].reshape(-1, 3)
        )  # [B*T,3,3]
        # R_z: Z轴旋转（aug_flag=False 时 rad=0，R_z=I）
        root_rot_mat = KC.axis_angle_to_rotation_matrix(
            (torch.Tensor([[[0, 0, 1]]]).to(device) * rad[:, :, None]).reshape(-1, 3)
        )  # [B*T,3,3]
        # R_corr: 透视归一化旋转（右乘，最先执行）
        if correction_rot_mat is not None:
            root_rot_mat = root_rot_mat @ correction_rot_mat.reshape(-1, 3, 3)
        # R_persp: 透视旋转增强（归一化启用时已被置零，双重保险跳过）
        if persp_rot_max > 0 and not perspective_normalization:
            persp_rot_mat = KC.axis_angle_to_rotation_matrix(
                persp_axis_angle.reshape(-1, 3)
            )  # [B*T,3,3]
            root_rot_mat = persp_rot_mat @ root_rot_mat  # R_persp @ R_z [@ R_corr]
        mano_pose_root = KC.rotation_matrix_to_axis_angle(
            root_rot_mat @ mano_pose_root
        ).reshape(B, T, 3)  # [B,T,3]
        mano_pose[:, :, :3] = mano_pose_root

        # timestamp
        timestamp: torch.Tensor = batch_origin["timestamp"].to(device)  # [B,T]

        # 对2D标注进行变换
        # 1. 首先变换joint_img，注意部分标注是无效的，需要滤掉
        joint_mask = (joint_valid < 0.5)[..., None].expand(-1, -1, -1, 2)  # [B,T,J]
        joint_img: torch.Tensor = batch_origin["joint_img"].to(device)  # [B,T,J,2]
        joint_img = apply_perspective_to_points(trans_2d_mat, joint_img)  # [B,T,J,2]
        # 2. 然后利用joint_img计算hand_bbox和patch_bbox
        xm, ym = torch.split(
            torch.min(joint_img.masked_fill(joint_mask, float("inf")), dim=-2).values,
            1,
            dim=-1,
        )  # [B,T]*2
        xM, yM = torch.split(
            torch.max(joint_img.masked_fill(joint_mask, -float("inf")), dim=-2).values,
            1,
            dim=-1,
        )  # [B,T]*2
        hand_bbox = torch.cat([xm, ym, xM, yM], dim=-1)  # [B,T,4]
        xc, yc = (xm + xM) * 0.5, (ym + yM) * 0.5
        half_edge_len = (  # [B,T]
            torch.max(hand_bbox[..., 2:] - hand_bbox[..., :2], dim=-1).values
            * 0.5
            * patch_expanstion
        )
        patch_bbox = torch.cat([  # [B,T,4]
            xc - half_edge_len[:, :, None], yc - half_edge_len[:, :, None],
            xc + half_edge_len[:, :, None], yc + half_edge_len[:, :, None],
        ], dim=-1)
        # 3. 利用新计算的hand_bbox和patch_bbox计算joint_hand_bbox和joint_patch_bbox
        joint_hand_bbox: torch.Tensor = joint_img - hand_bbox[:, :, None, :2]  # [B,T,J,2]
        joint_patch_bbox: torch.Tensor = joint_img - patch_bbox[:, :, None, :2]  # [B,T,J,2]
        # 4. 利用计算的patch_bbox进行采样
        patch_bbox_corner = torch.stack([  # [B,T,4,2]
            torch.stack([patch_bbox[:, :, 0], patch_bbox[:, :, 1]], dim=-1),
            torch.stack([patch_bbox[:, :, 2], patch_bbox[:, :, 1]], dim=-1),
            torch.stack([patch_bbox[:, :, 2], patch_bbox[:, :, 3]], dim=-1),
            torch.stack([patch_bbox[:, :, 0], patch_bbox[:, :, 3]], dim=-1),
        ], dim=2)
        patch_bbox_corner_orig = apply_perspective_to_points(
            trans_2d_mat.inverse(), patch_bbox_corner
        )
        patches = []
        for bx, img_orig_tensor in enumerate(batch_origin["imgs"]):
            patch = KT.crop_and_resize(
                img_orig_tensor.to(device).float() / 255,
                patch_bbox_corner_orig[bx],
                patch_size,
                mode="bilinear"
            )
            patches.append(patch)
        patches: torch.Tensor = torch.stack(patches)

        # 应用像素级增强（如果配置了）
        if pixel_aug is not None:
            patches = pixel_aug(patches)

        # 更新focal&princpt
        focal = focal_new
        princpt = princpt_new

    # 进行左右翻转
    for bx in range(len(imgs_path)):
        if flip[bx]:
            T, C, H, W = batch_origin["imgs"][bx].shape
            patch_bbox_w = patch_bbox[bx, :, 2] - patch_bbox[bx, :, 0]  # [T]
            hand_bbox_w =  hand_bbox[bx, :, 2] - hand_bbox[bx, :, 0]  # [T]

            patches[bx] = torch.flip(patches[bx], dims=[-1,])
            patch_bbox[bx, :, 0], patch_bbox[bx, :, 2] = (
                W - patch_bbox[bx, :, 2],
                W - patch_bbox[bx, :, 0],
            )
            hand_bbox[bx, :, 0], hand_bbox[bx, :, 2] = (
                W - hand_bbox[bx, :, 2],
                W - hand_bbox[bx, :, 0]
            )
            joint_img[bx, :, :, 0] = W - joint_img[bx, :, :, 0]
            joint_patch_bbox[bx, :, :, 0] = patch_bbox_w[:, None] - joint_patch_bbox[bx, :, :, 0]
            joint_hand_bbox[bx, :, :, 0] = hand_bbox_w[:, None] - joint_hand_bbox[bx, :, :, 0]
            joint_cam[bx, :, :, 0] *= -1
            joint_rel[bx, :, :, 0] *= -1

            mano_pose_bx = rearrange(mano_pose[bx], "t (j d) -> t j d", d=3)
            mano_pose_bx[:, :, 1:] *= -1
            mano_pose[bx] = rearrange(mano_pose_bx, "t j d -> t (j d)")

            princpt[bx, :, 0] = W - princpt[bx, :, 0]

    # 更新旋转表示
    if joint_rep_type == "3":
        pass
    elif joint_rep_type == "6d":
        B, T = mano_pose.shape[:2]
        mano_pose = rearrange(mano_pose, "b t (j d) -> (b t j) d", j=MANO_JOINT_COUNT)
        mano_pose = KC.axis_angle_to_rotation_matrix(mano_pose) # [N,3,3]
        mano_pose = rotation_matrix_to_rotation6d(mano_pose)
        mano_pose = rearrange(mano_pose, "(b t j) d -> b t (j d)", b=B, t=T)
    elif joint_rep_type == "quat":
        B, T = mano_pose.shape[:2]
        mano_pose = rearrange(mano_pose, "b t (j d) -> (b t j) d", j=MANO_JOINT_COUNT)
        mano_pose = KC.axis_angle_to_quaternion(mano_pose) # [N,4]
        mano_pose = rearrange(mano_pose, "(b t j) d -> b t (j d)", b=B, t=T)
    else:
        raise NotImplementedError(f"Unsupported rotation type={joint_rep_type}")

    if mano_pose[mano_valid > 0.5].isnan().any().cpu().item():
        raise ValueError("MANO contains NaN")

    return {
        "imgs_path": imgs_path,
        "flip": flip,
        "patches": patches,
        "patch_bbox": patch_bbox,
        "hand_bbox": hand_bbox,
        "joint_img": joint_img,
        "joint_patch_bbox": joint_patch_bbox,
        "joint_hand_bbox": joint_hand_bbox,
        "joint_cam": joint_cam,
        "joint_rel": joint_rel,
        "joint_valid": joint_valid,
        "mano_pose": mano_pose,
        "mano_shape": mano_shape,
        "mano_valid": mano_valid,
        "timestamp": timestamp,
        "focal": focal,
        "princpt": princpt
    }, trans_2d_mat, correction_rot_mat
