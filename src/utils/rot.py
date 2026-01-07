import torch
import torch.nn.functional as F

def rotation6d_to_rotation_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    将 6D 旋转表示转换为 3x3 旋转矩阵。
    基于 Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019

    Args:
        d6: 输入张量，形状为 (*, 6)

    Returns:
        旋转矩阵，形状为 (*, 3, 3)
    """
    # 1. 取出两个 3D 向量 (a1, a2)
    a1 = d6[..., :3]
    a2 = d6[..., 3:]

    # 2. 对第一个向量进行归一化，得到第一列 b1
    b1 = F.normalize(a1, dim=-1)

    # 3. 对第二个向量进行施密特正交化（减去在 b1 上的投影分量），得到 b2
    # projection = (b1 . a2) * b1
    proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = a2 - proj

    # 4. 对 b2 进行归一化
    b2 = F.normalize(b2, dim=-1)

    # 5. 通过叉积计算第三列 b3 (右手定则)
    b3 = torch.cross(b1, b2, dim=-1)

    # 6. 堆叠成 3x3 矩阵 (最后一维是列向量)
    return torch.stack((b1, b2, b3), dim=-1)