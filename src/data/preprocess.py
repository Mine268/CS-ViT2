from typing import *
import torch
from einops import rearrange
import kornia.geometry.transform as KT
import kornia.geometry.conversions as KC
import kornia.augmentation as KA

from ..utils.rot import *
from ..constant import *


class PixelLevelAugmentation(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.transforms = torch.nn.Sequential(
            KA.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
            KA.RandomGrayscale(p=0.2),
            KA.RandomPosterize(bits=3, p=0.2),
            KA.RandomSharpness(sharpness=0.5, p=0.3),
            KA.RandomEqualize(p=0.1),
            KA.RandomGaussianNoise(mean=0.0, std=0.05, p=0.2),
            KA.RandomMotionBlur(kernel_size=(3, 5), angle=35., direction=0.5, p=0.2),
            KA.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.2),
            KA.RandomErasing(scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0.0, p=0.3),
        )

    def forward(self, input_tensor):
        B, T, C, H, W = input_tensor.shape
        input_tensor_aug = self.transforms(
            input_tensor.reshape(B * T, C, H, W)
        ).reshape(B, T, C, H, W)
        input_tensor_aug = torch.clamp(input_tensor_aug, 0.0, 1.0)
        return input_tensor_aug

pixel_level_augemtation = PixelLevelAugmentation()

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
    device: torch.device
):
    """
    将wds的原始数据进行预处理和数据增强，最后送给模型

    Args:
        patch_size: 输出给模型的图像patch的大小
        patch_expansion: patch相较于正方形包围盒扩大的范围
        scale_z_range: 进行缩放/平移增强变换的系数范围
        scale_f_range: 进行内参增强变换的焦距乘数的范围
    """
    B, T = batch_origin["joint_cam"].shape[:2]
    trans_2d_mat = torch.eye(3, device=device).float()[None, None, :].expand(B, T, -1, -1)

    if not augmentation_flag:
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

        # 数据增强参数
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

        # 获得数据增强变换矩阵
        # [B,T,3,3]
        trans_3d_mat = get_trans_3d_mat(rad, scale_z, persp_axis_angle)
        trans_2d_mat = get_trans_2d_mat(
            rad, 1 / scale_z, focal, princpt, focal_new, princpt_new, persp_axis_angle
        )

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
        root_rot_mat = KC.axis_angle_to_rotation_matrix(
            (torch.Tensor([[[0, 0, 1]]]).to(device) * rad[:, :, None]).reshape(-1, 3)
        )  # [B*T,3,3]
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
        patches = pixel_level_augemtation(patches)

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
    }, trans_2d_mat
