import torch


def proj_points_3d(x3d: torch.Tensor, focal: torch.Tensor, princpt: torch.Tensor):
    """
    Project 3D points to 2D using a pinhole camera model.

    Args:
        x3d: [..., j, 3] - 3D coordinates (x, y, z)
        focal: [..., 2] - Focal length (fx, fy)
        princpt: [..., 2] - Principal point (cx, cy)

    Return:
        x2d: [..., j, 2] - Projected 2D coordinates
    """
    # 1. Unpack 3D coordinates
    # shape: [..., j, 2] and [..., j, 1]
    xy = x3d[..., :2]
    z = x3d[..., 2:3]

    # 2. Add singleton dimension to camera params for broadcasting
    # transforms [..., 2] -> [..., 1, 2] so it broadcasts against j points
    focal = focal.unsqueeze(-2)
    princpt = princpt.unsqueeze(-2)

    # 3. Perspective division (x/z, y/z)
    # Clamp z to avoid division by zero if necessary, usually safe in valid data
    z = torch.clamp(z, min=1e-8)
    xy_normalized = xy / z

    # 4. Apply intrinsics: (normalized * focal) + principal_point
    x2d = xy_normalized * focal + princpt

    return x2d


def patch_uv_to_image_uv(
    uv_patch: torch.Tensor,
    patch_bbox: torch.Tensor,
    patch_size,
) -> torch.Tensor:
    """
    将 resized patch 坐标系中的 2D 点还原到原图像素坐标系。

    Args:
        uv_patch: [..., 2]
        patch_bbox: [..., 4]，xyxy
        patch_size: (patch_h, patch_w)
    """
    patch_h, patch_w = patch_size
    patch_w = float(patch_w)
    patch_h = float(patch_h)

    patch_width = torch.clamp(patch_bbox[..., 2] - patch_bbox[..., 0], min=1e-8)
    patch_height = torch.clamp(patch_bbox[..., 3] - patch_bbox[..., 1], min=1e-8)

    uv_img = torch.empty_like(uv_patch)
    uv_img[..., 0] = patch_bbox[..., 0] + uv_patch[..., 0] * patch_width / patch_w
    uv_img[..., 1] = patch_bbox[..., 1] + uv_patch[..., 1] * patch_height / patch_h
    return uv_img


def image_uv_to_camera_q(
    uv_img: torch.Tensor,
    focal: torch.Tensor,
    princpt: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    将图像坐标系中的像素点转换为 pinhole camera 的未归一化射线方向 q。

    Args:
        uv_img: [..., 2]
        focal: [..., 2]
        princpt: [..., 2]
    """
    focal = torch.clamp(focal, min=eps)
    q = torch.empty(*uv_img.shape[:-1], 3, device=uv_img.device, dtype=uv_img.dtype)
    q[..., 0] = (uv_img[..., 0] - princpt[..., 0]) / focal[..., 0]
    q[..., 1] = (uv_img[..., 1] - princpt[..., 1]) / focal[..., 1]
    q[..., 2] = 1.0
    return q


def image_uv_to_camera_ray(
    uv_img: torch.Tensor,
    focal: torch.Tensor,
    princpt: torch.Tensor,
    eps: float = 1e-8,
):
    """
    将图像像素点转换为单位射线方向。

    Returns:
        q: [..., 3]，未归一化方向
        ray_unit: [..., 3]，单位方向
        q_norm: [...]，||q||
    """
    q = image_uv_to_camera_q(uv_img, focal, princpt, eps=eps)
    q_norm = torch.clamp(torch.linalg.norm(q, dim=-1), min=eps)
    ray_unit = q / q_norm[..., None]
    return q, ray_unit, q_norm


def backproject_uv_rho(
    uv_img: torch.Tensor,
    focal: torch.Tensor,
    princpt: torch.Tensor,
    rho: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    根据图像坐标和光心距离 rho 显式反投影到相机坐标系。

    Args:
        uv_img: [..., 2]
        focal: [..., 2]
        princpt: [..., 2]
        rho: [...] 或 [..., 1]
    """
    _, ray_unit, _ = image_uv_to_camera_ray(uv_img, focal, princpt, eps=eps)
    if rho.shape == uv_img.shape[:-1]:
        rho = rho[..., None]
    return ray_unit * rho
