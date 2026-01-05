import torch
import torch.nn as nn


def robust_masked_mean(loss: torch.Tensor, mask: torch.Tensor):
    """
    辅助函数：计算安全的加权平均值
    loss: [...] 任意维度
    mask: [...] 与 loss 广播兼容的掩码
    """
    # 确保 mask 和 loss 类型一致（防止 bool 类型报错）
    mask = mask.to(dtype=loss.dtype)

    # 1. 应用掩码
    loss_masked = loss * mask

    # 2. 计算总和
    total_loss = loss_masked.sum()

    total_valid = mask.sum()

    # 3. 安全归一化
    if total_valid > 1e-6:
        return total_loss / total_valid
    else:
        # 如果没有有效样本，返回 0，同时保留梯度图（乘以 0.0）
        return total_loss * 0.0

class Keypoint3DLoss(nn.Module):
    def __init__(self, loss_type: str = 'l1'):
        """
        3D keypoint loss module.
        """
        super(Keypoint3DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor):
        """
        pred, gt: [b,t,j,d]
        valid: [b,t,j]
        """
        # 1. 计算所有维度的元素级损失 [b,t,j,d]
        raw_loss = self.loss_fn(pred, gt)

        # 2. 计算二维距离
        raw_loss = torch.mean(raw_loss, dim=-1)

        # 3. 计算安全的全局平均
        return robust_masked_mean(raw_loss, valid)

class Axis3DLoss(nn.Module):
    def __init__(self, loss_type: str = 'l1'):
        """
        Axis angle / Rotation loss module.
        """
        super().__init__()
        # 保持默认 l1 以匹配原代码行为，但也支持配置
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor):
        """
        pred, gt: [b,t,n] (d=3 for axis angle, d=6 for rot6d)
        valid: [b,t]
        """
        # 1. 计算元素级损失 [b,t,n]
        raw_loss = self.loss_fn(pred, gt)

        # 2. 扩展 mask 维度 [b,t,j,1]
        valid_mask = valid.unsqueeze(-1)

        # 3. 计算安全的全局平均
        return robust_masked_mean(raw_loss, valid_mask)

class VertsLoss(nn.Module):
    def __init__(self, loss_type: str = 'l1'):
        """
        Mesh vertices loss module.
        """
        super(VertsLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor):
        """
        pred, gt: [b,t,v,3]
        valid: [b,t]
        """
        # 1. 计算元素级损失 [b,t,v,3]
        raw_loss = self.loss_fn(pred, gt)

        # 2. 按照原定义：先对顶点和坐标维求均值 -> [b,t]
        # 注意：这里不能简单使用 Global Mean，因为原定义赋予了每个 Frame 相同的权重
        # 无论该 Frame 的顶点数多少（虽然 SMPL 顶点数固定，但逻辑要一致）
        per_frame_loss = torch.mean(raw_loss, dim=(2, 3))

        # 3. 对 [b,t] 维度的 loss 应用 mask
        return robust_masked_mean(per_frame_loss, valid)

class ShapeLoss(nn.Module):
    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor):
        """
        pred: [b,t,10]
        gt: [b,t,10]
        valid: [b,t]
        """
        # 1. 计算元素级损失 [b,t,10]
        raw_loss = self.loss_fn(pred, gt)

        # 2. 按照原定义：先对 shape 参数维求均值 -> [b,t]
        per_frame_loss = torch.mean(raw_loss, dim=-1)

        # 3. 对 [b,t] 维度的 loss 应用 mask
        return robust_masked_mean(per_frame_loss, valid)