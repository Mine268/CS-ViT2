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