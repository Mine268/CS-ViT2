from typing import *

import torch
import torch.nn as nn

from ..utils.mano import *
from ..utils.proj import *


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
    def __init__(self, loss_type: str = 'l1', scale: float = 1.0):
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

        self.scale = scale

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor):
        """
        pred, gt: [b,t,j,d]
        valid: [b,t,j]
        """
        # 1. 计算所有维度的元素级损失 [b,t,j,d]
        pred = pred * self.scale
        gt = gt * self.scale
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
    def __init__(self, loss_type: str = 'l1', scale: float = 1.0):
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

        self.scale = scale

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor):
        """
        pred, gt: [b,t,v,3]
        valid: [b,t]
        """
        # 1. 计算元素级损失 [b,t,v,3]
        pred = pred * self.scale
        gt = gt * self.scale
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


class ParameterLoss(nn.Module):

    def __init__(self, loss_type: str = 'l2'):
        """
        MANO parameter loss module.
        """
        super(ParameterLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor):
        """
        pred, gt: [b,t,d]
        valid: [b,t]
        """
        # 1. 计算元素级损失 [b,t,d]
        raw_loss = self.loss_fn(pred, gt)

        per_frame_loss = torch.mean(raw_loss, dim=-1)

        # 3. 对 [b,t] 维度的 loss 应用 mask
        return robust_masked_mean(per_frame_loss, valid)


class BundleLoss(nn.Module):
    def __init__(
        self,
        rel: float,
        glo: float,
        proj: float,
        supervise_global: bool,
    ):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.l1 = nn.L1Loss(reduction="none")

        self.rel, self.glo, self.proj = rel, glo, proj
        self.supervise_global = supervise_global

    def forward(
        self,
        pose_pred: torch.Tensor,
        shape_pred: torch.Tensor,
        trans_pred: torch.Tensor,
        pose_gt: torch.Tensor,
        shape_gt: torch.Tensor,
        trans_gt: torch.Tensor,

        joint_rel_gt: torch.Tensor,
        joint_rel_pred: torch.Tensor,
        verts_rel_gt: torch.Tensor,
        verts_rel_pred: torch.Tensor,

        joint_img_gt: torch.Tensor,
        joint_img_pred: torch.Tensor,

        mano_valid: torch.Tensor,
        joint_valid: torch.Tensor,
        norm_valid: torch.Tensor,
    ):
        loss_theta = self.mse(pose_pred, pose_gt) # [b,t,d]
        loss_theta = torch.mean(loss_theta * mano_valid[..., None])
        loss_shape = self.mse(shape_pred, shape_gt) # [b,t,d]
        loss_shape = torch.mean(loss_shape * mano_valid[..., None])

        loss_verts = self.l1(verts_rel_pred, verts_rel_gt) # [b,t,v,d]
        loss_verts = torch.mean(loss_verts * mano_valid[..., None, None])

        if self.supervise_global:
            loss_joint_root = self.l1(trans_pred, trans_gt) # [b,t,d]
            loss_joint_root = torch.mean(
                loss_joint_root * joint_valid[:, :, :1] * norm_valid[:, :, None]
            )

            loss_joint_rel = self.l1(joint_rel_pred, joint_rel_gt) # [b,t,j,d]
            loss_joint_rel = torch.mean(loss_joint_rel * joint_valid[..., None])

            loss_joint_img = self.l1(joint_img_pred, joint_img_gt) # [b,t,j,2]
            loss_joint_img = torch.mean(
                loss_joint_img * joint_valid[..., None] * norm_valid[..., None, None]
            )

            loss_joint = (
                self.glo * loss_joint_root
                + self.proj * loss_joint_img
                + self.rel * loss_joint_rel
            )

            sub_state = {
                "loss_joint_root": loss_joint_root.detach(),
                "loss_joint_rel": loss_joint_rel.detach(),
                "loss_joint_img": loss_joint_img.detach(),
            }
        else:
            loss_joint_rel = self.l1(joint_rel_pred, joint_rel_gt) # [b,t,j,d]
            loss_joint = torch.mean(loss_joint_rel * joint_valid[..., None])

            sub_state = {}

        loss = loss_theta + loss_shape + loss_verts + loss_joint

        return (
            loss,
            {
                "loss_theta": loss_theta.detach(),
                "loss_shape": loss_shape.detach(),
                "loss_verts": loss_verts.detach(),
                "loss_joint": loss_joint.detach(),
            }
            | sub_state,
        )


class BundleLoss2(nn.Module):
    def __init__(
        self,
        lambda_theta: float,
        lambda_shape: float,
        lambda_trans: float,
        lambda_rel: float,
        lambda_img: float,
        supervise_global: bool,
        supervise_heatmap: bool,
        norm_by_hand: bool,
        norm_idx: List[int],
        hm_centers: Optional[Tuple[torch.Tensor]],
        hm_sigma: float,
    ):
        super().__init__()

        self.mse = nn.MSELoss(reduction="none")
        self.l1 = nn.L1Loss(reduction="none")

        self.lambda_theta = lambda_theta
        self.lambda_shape = lambda_shape
        self.lambda_trans = lambda_trans
        self.lambda_rel = lambda_rel
        self.lambda_img = lambda_img

        self.supervise_global = supervise_global
        self.supervise_heatmap = supervise_heatmap
        if supervise_heatmap:
            self.register_buffer("x_centers", hm_centers[0])
            self.register_buffer("y_centers", hm_centers[1])
            self.register_buffer("z_centers", hm_centers[2])
            self.hm_sigma = hm_sigma
        self.norm_by_hand = norm_by_hand
        self.norm_idx = norm_idx

        self.rmano_layer = RMANOLayer()

    def get_hand_norm_scale(self, j3d: torch.Tensor, valid: torch.Tensor):
        """
        Args:
            j3d: [...,j,3]
            valid: [...,j]
            return: [...], [...]
        """
        d = j3d[..., self.norm_idx[:-1], :] - j3d[..., self.norm_idx[1:], :]
        d = torch.sum(torch.sqrt(torch.sum(d ** 2, dim=-1)), dim=-1) # [...]
        flag = torch.all(valid[:, :, self.norm_idx] > 0.5, dim=-1).float()
        return d, flag

    def compute_hm_ce(
        self,
        pred_log_probs: torch.Tensor,
        gt_coords: torch.Tensor,
        grid_positions: torch.Tensor,
    ):
        """
        Args:
            pred_log_probs: Tensor [..., N] **必须已经经过 LogSoftmax 处理**。即输入已经是 log(probability)。
            gt_coords: Tensor [...] 真值坐标。
            grid_positions: Tensor [N] 网格坐标。
        Return:
            [...]
        """
        gt_coords.unsqueeze_(-1)

        grid_view_shape = [1] * gt_coords.dim()
        grid_view_shape[-1] = -1
        grid_positions = grid_positions.view(grid_view_shape)

        squared_diff = (gt_coords - grid_positions) ** 2

        target_unnormalized = torch.exp(-squared_diff / (2 * self.hm_sigma**2))
        target_probs = target_unnormalized / (target_unnormalized.sum(dim=-1, keepdim=True) + 1e-9)

        loss = -(target_probs * pred_log_probs).sum(dim=-1)

        return loss

    def forward(self, pose_pred, shape_pred, trans_pred, log_hm_pred, batch):
        """
        Args:
            xxx_pred: [b,t,48/10/3,n]
        """
        # fk to pose (只对最后一帧做 FK)
        with torch.no_grad():
            _, vert_rel_gt = self.rmano_layer(
                batch["mano_pose"][:, -1:], batch["mano_shape"][:, -1:]
            )
        # decouple shape and pose
        joint_rel_pred, vert_rel_pred = self.rmano_layer(
            pose_pred[:, -1:], shape_pred[:, -1:].detach()
        )
        if self.norm_by_hand:
            # [b,t]
            norm_scale_gt, norm_valid_gt = self.get_hand_norm_scale(
                batch["joint_cam"][:, -1:], batch["joint_valid"][:, -1:]
            )

        # get data
        pose_gt = batch["mano_pose"][:, -1:]
        shape_gt = batch["mano_shape"][:, -1:]
        mano_valid = batch["mano_valid"][:, -1:] # [b,t]
        joint_valid = batch["joint_valid"][:, -1:] # [b,t,j]

        # get trans gt data
        trans_gt = batch["joint_cam"][:, -1:, 0] # [b,t,3]
        if self.norm_by_hand:
            trans_gt = trans_gt / norm_scale_gt[..., None]

        # param loss
        loss_theta = self.l1(pose_pred, pose_gt) # [b,t,d]
        loss_theta = torch.mean(loss_theta * mano_valid[..., None])
        loss_shape = self.l1(shape_pred, shape_gt) # [b,t,d]
        loss_shape = torch.mean(loss_shape * mano_valid[..., None])
        if not self.supervise_heatmap: # supervise on trans
            loss_trans = self.l1(trans_pred, trans_gt) # [b,t,d]
            loss_trans = torch.mean(loss_trans * joint_valid[:, :, :1] * norm_valid_gt[..., None])
        else: # supervise on heatmap, using cross entropy
            loss_trans = (
                self.compute_hm_ce(log_hm_pred[0], trans_gt[..., 0], self.x_centers)
                + self.compute_hm_ce(log_hm_pred[1], trans_gt[..., 1], self.y_centers)
                + self.compute_hm_ce(log_hm_pred[2], trans_gt[..., 2], self.z_centers)
            )
            loss_trans = torch.mean(
                loss_trans * joint_valid[:, :, 0] * norm_valid_gt
            )

        # joint loss
        loss_joint_rel = self.l1(joint_rel_pred, batch["joint_rel"][:, -1:]) # [b,t,j,d]
        loss_joint_rel = torch.mean(loss_joint_rel * joint_valid[..., None])

        # 投影 loss（使用 scaled trans，避免修改原始 trans_pred）
        if self.norm_by_hand:
            trans_pred_scaled = trans_pred * norm_scale_gt[..., None]
        else:
            trans_pred_scaled = trans_pred

        joint_cam_gt = batch["joint_cam"][:, -1:]
        joint_cam_pred = joint_rel_pred + trans_pred_scaled[:, :, None, :]
        joint_img_gt = proj_points_3d(
            joint_cam_gt, batch["focal"][:, -1:], batch["princpt"][:, -1:]
        )
        joint_img_pred = proj_points_3d(
            joint_cam_pred, batch["focal"][:, -1:], batch["princpt"][:, -1:]
        )
        loss_joint_img = self.l1(joint_img_pred, joint_img_gt) # [b,1,j,2]
        loss_joint_img = torch.mean(
            loss_joint_img * joint_valid[..., None] * norm_valid_gt[..., None, None]
        )

        loss = (
            self.lambda_theta * loss_theta +
            self.lambda_shape * loss_shape +
            self.lambda_trans * loss_trans +
            self.lambda_rel * loss_joint_rel +
            self.lambda_img * loss_joint_img
        )

        loss_state = {
            "loss_theta": loss_theta.detach(),
            "loss_shape": loss_shape.detach(),
            "loss_trans": loss_trans.detach(),
            "loss_joint_rel": loss_joint_rel.detach(),
            "loss_joint_img": loss_joint_img.detach(),
        }

        fk_result = {
            "verts_cam_gt": vert_rel_gt + batch["joint_cam"][:, -1:, :1],
            "verts_rel_gt": vert_rel_gt,
            "joint_cam_pred": joint_cam_pred,
            "joint_rel_pred": joint_rel_pred,
            "verts_cam_pred": vert_rel_pred + trans_pred_scaled[..., None, :],
            "verts_rel_pred": vert_rel_pred,
            "norm_valid_gt": norm_valid_gt,
        }

        return loss, loss_state, fk_result
