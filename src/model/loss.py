from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.mano import *
from ..utils.proj import *
from .root_z import encode_delta_log_z_targets

# 防止 norm_scale 除零的 epsilon 值（单位: mm）
NORM_SCALE_EPSILON = 1e-6


class RobustL1Loss(nn.Module):
    """
    鲁棒L1损失函数

    在delta范围内使用L1 loss (线性)，超出delta后使用对数衰减，梯度逐渐变弱。

    数学表达式:
        loss(x) = |x|,                                  if |x| < delta
                  delta * (1 + log(1 + (|x|-delta)/delta)), if |x| >= delta

    梯度:
        grad(x) = sign(x),                              if |x| < delta
                  sign(x) * delta/(delta + |x| - delta), if |x| >= delta
                = sign(x) / (1 + (|x|-delta)/delta)

    特性:
    - 在delta内保持L1的线性特性，梯度为常数±1
    - 超过delta后，梯度逐渐衰减为0，避免异常值主导训练
    - 连续可导（在±delta处）

    Args:
        delta (float): 阈值，超过此值后启用鲁棒化。单位与loss相同（如像素）
        reduction (str): 'none' | 'mean' | 'sum'
    """
    def __init__(self, delta: float = 100.0, reduction: str = 'none'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [...] 预测值
            target: [...] 目标值

        Returns:
            loss: [...] or scalar，取决于reduction
        """
        abs_diff = torch.abs(pred - target)

        # 区域1: |x| < delta, 使用L1
        inside_mask = abs_diff < self.delta
        loss_l1 = abs_diff

        # 区域2: |x| >= delta, 使用对数衰减
        # 为了连续性，在delta处匹配
        # loss(delta) = delta
        # loss(delta + ε) = delta * (1 + log(1 + ε/delta))
        outside_diff = abs_diff - self.delta
        loss_log = self.delta * (1.0 + torch.log1p(outside_diff / self.delta))

        loss = torch.where(inside_mask, loss_l1, loss_log)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def robust_masked_mean(loss: torch.Tensor, mask: torch.Tensor):
    """
    辅助函数：计算安全的加权平均值
    loss: [...] 任意维度
    mask: [...] 与 loss 广播兼容的掩码
    """
    # 确保 mask 和 loss 类型一致（防止 bool 类型报错）
    mask = mask.to(dtype=loss.dtype)
    if mask.shape != loss.shape:
        mask = torch.broadcast_to(mask, loss.shape)

    # 1. 应用掩码
    loss_masked = loss * mask

    # 2. 计算总和
    total_loss = loss_masked.sum()

    total_valid = mask.sum()

    # 3. 安全归一化
    if total_valid.item() > 1e-6:
        return total_loss / total_valid
    else:
        # 如果没有有效样本，返回 0，同时保留梯度图（乘以 0.0）
        return total_loss * 0.0


def build_data_source_mask(
    data_sources: Optional[Sequence[str]],
    target_source: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    根据 batch 内 data_source 构造样本级掩码。

    Returns:
        [B, 1]，匹配样本级 / 帧级广播使用。
    """
    if data_sources is None:
        return torch.zeros((0, 1), device=device, dtype=dtype)

    mask = [str(source) == target_source for source in data_sources]
    return torch.tensor(mask, device=device, dtype=dtype).view(-1, 1)


def joint_img_to_patch_resized(
    joint_img: torch.Tensor,
    patch_bbox: torch.Tensor,
    patch_size: Tuple[int, int],
) -> torch.Tensor:
    """
    将原图坐标系 2D joints 映射到 resized patch 坐标系。

    Args:
        joint_img: [B, T, J, 2]
        patch_bbox: [B, T, 4]，xyxy
        patch_size: (patch_h, patch_w)
    """
    patch_h, patch_w = patch_size
    patch_width = torch.clamp(patch_bbox[..., 2] - patch_bbox[..., 0], min=1e-6)
    patch_height = torch.clamp(patch_bbox[..., 3] - patch_bbox[..., 1], min=1e-6)

    joint_patch_resized = torch.empty_like(joint_img)
    joint_patch_resized[..., 0] = (
        (joint_img[..., 0] - patch_bbox[..., 0, None]) * patch_w / patch_width[..., None]
    )
    joint_patch_resized[..., 1] = (
        (joint_img[..., 1] - patch_bbox[..., 1, None]) * patch_h / patch_height[..., None]
    )
    return joint_patch_resized


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

        has_mano: torch.Tensor,
        joint_3d_valid: torch.Tensor,
        joint_2d_valid: torch.Tensor,
        norm_valid: torch.Tensor,
    ):
        loss_theta = self.mse(pose_pred, pose_gt)
        loss_theta = robust_masked_mean(loss_theta, has_mano[..., None])
        loss_shape = self.mse(shape_pred, shape_gt)
        loss_shape = robust_masked_mean(loss_shape, has_mano[..., None])

        loss_verts = self.l1(verts_rel_pred, verts_rel_gt)
        loss_verts = robust_masked_mean(loss_verts, has_mano[..., None, None])

        if self.supervise_global:
            loss_joint_root = self.l1(trans_pred, trans_gt)
            loss_joint_root = robust_masked_mean(
                loss_joint_root,
                joint_3d_valid[:, :, :1] * norm_valid[:, :, None],
            )

            loss_joint_rel = self.l1(joint_rel_pred, joint_rel_gt)
            loss_joint_rel = robust_masked_mean(
                loss_joint_rel,
                joint_3d_valid[..., None],
            )

            loss_joint_img = self.l1(joint_img_pred, joint_img_gt)
            loss_joint_img = robust_masked_mean(
                loss_joint_img,
                joint_2d_valid[..., None] * norm_valid[..., None, None],
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
            loss_joint_rel = self.l1(joint_rel_pred, joint_rel_gt)
            loss_joint = robust_masked_mean(
                loss_joint_rel,
                joint_3d_valid[..., None],
            )
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
        lambda_coco_patch_2d: float,
        lambda_root_z_cls: float,
        lambda_root_z_res: float,
        supervise_global: bool,
        supervise_heatmap: bool,
        norm_by_hand: bool,
        norm_idx: List[int],
        hm_centers: Optional[Tuple[torch.Tensor]],
        hm_sigma: float,
        cam_head_type: str = "softargmax3d",
        root_z_num_bins: int = 8,
        root_z_d_min: float = -0.73,
        root_z_d_max: float = 0.74,
        root_z_min_valid_joints_2d: int = 0,
        root_z_min_hand_bbox_edge_px: float = 0.0,
        reproj_loss_type: str = "robust_l1",
        reproj_loss_delta: float = 84.0,
    ):
        super().__init__()

        self.mse = nn.MSELoss(reduction="none")
        self.l1 = nn.L1Loss(reduction="none")

        # 动态选择重投影loss类型
        self.reproj_loss_type = reproj_loss_type
        if reproj_loss_type == "l1":
            self.reproj_loss_fn = self.l1
        elif reproj_loss_type == "robust_l1":
            self.reproj_loss_fn = RobustL1Loss(delta=reproj_loss_delta, reduction='none')
        else:
            raise ValueError(f"Unsupported reproj_loss_type: {reproj_loss_type}. "
                           f"Supported types: ['l1', 'robust_l1']")

        self.lambda_theta = lambda_theta
        self.lambda_shape = lambda_shape
        self.lambda_trans = lambda_trans
        self.lambda_rel = lambda_rel
        self.lambda_img = lambda_img
        self.lambda_coco_patch_2d = lambda_coco_patch_2d
        self.lambda_root_z_cls = lambda_root_z_cls
        self.lambda_root_z_res = lambda_root_z_res

        self.supervise_global = supervise_global
        self.supervise_heatmap = supervise_heatmap
        self.cam_head_type = cam_head_type
        self.root_z_num_bins = root_z_num_bins
        self.root_z_d_min = float(root_z_d_min)
        self.root_z_d_max = float(root_z_d_max)
        self.root_z_min_valid_joints_2d = int(root_z_min_valid_joints_2d)
        self.root_z_min_hand_bbox_edge_px = float(root_z_min_hand_bbox_edge_px)
        if supervise_heatmap:
            self.register_buffer("x_centers", hm_centers[0])
            self.register_buffer("y_centers", hm_centers[1])
            if hm_centers[2] is not None:
                self.register_buffer("z_centers", hm_centers[2])
            self.hm_sigma = hm_sigma
        self.norm_by_hand = norm_by_hand
        self.norm_idx = norm_idx

        self.rmano_layer = RMANOLayer()

    def _zero_like(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.sum() * 0.0

    def _masked_mean_scalar(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(dtype=values.dtype)
        total_valid = mask.sum()
        if total_valid.item() <= 1e-6:
            return values.sum() * 0.0
        return (values * mask).sum() / total_valid

    def compute_root_z_frame_filter_mask(self, batch) -> torch.Tensor:
        """
        计算 root-z supervision 的额外 frame 级过滤掩码。

        这层过滤只用于 root-z 路径，不影响其它 loss：
        - 2D valid joint 数量过少的 frame 不参与 root-z supervision
        - hand bbox 最短边过小的 frame 不参与 root-z supervision
        """
        frame_mask = torch.ones_like(batch["has_intr"], dtype=torch.float32)

        if self.root_z_min_valid_joints_2d > 0:
            joint_2d_valid = batch.get("joint_2d_valid", batch["joint_valid"])
            valid_joint_count = torch.sum(joint_2d_valid > 0.5, dim=-1)
            frame_mask = frame_mask * (
                valid_joint_count >= self.root_z_min_valid_joints_2d
            ).float()

        if self.root_z_min_hand_bbox_edge_px > 0:
            hand_bbox = batch["hand_bbox"]
            bbox_min_edge = torch.minimum(
                hand_bbox[..., 2] - hand_bbox[..., 0],
                hand_bbox[..., 3] - hand_bbox[..., 1],
            )
            frame_mask = frame_mask * (
                bbox_min_edge >= self.root_z_min_hand_bbox_edge_px
            ).float()

        return frame_mask

    def get_hand_norm_scale(self, j3d: torch.Tensor, valid: torch.Tensor):
        """
        Args:
            j3d: [...,j,3]
            valid: [...,j]
            return: [...], [...]
        """
        d = j3d[..., self.norm_idx[:-1], :] - j3d[..., self.norm_idx[1:], :]
        d = torch.sum(torch.sqrt(torch.sum(d ** 2, dim=-1)), dim=-1) # [...]

        # 防止 norm_scale 过小（双重保护）
        d = torch.clamp(d, min=NORM_SCALE_EPSILON)

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

    def forward(self, pose_pred, shape_pred, trans_pred, cam_aux, batch):
        """
        Args:
            xxx_pred: [b,t,48/10/3,n]
        """
        with torch.no_grad():
            _, vert_rel_gt = self.rmano_layer(
                batch["mano_pose"], batch["mano_shape"]
            )
        joint_rel_pred, vert_rel_pred = self.rmano_layer(
            pose_pred, shape_pred.detach()
        )

        pose_gt = batch["mano_pose"]
        shape_gt = batch["mano_shape"]
        has_mano = batch["has_mano"]
        joint_2d_valid = batch["joint_2d_valid"]
        joint_3d_valid = batch["joint_3d_valid"]
        has_intr = batch["has_intr"]
        if batch.get("data_source") is None:
            coco_sample_mask = torch.zeros(
                (has_mano.shape[0], 1),
                device=pose_pred.device,
                dtype=pose_pred.dtype,
            )
        else:
            coco_sample_mask = build_data_source_mask(
                batch.get("data_source"),
                target_source="COCO-WholeBody",
                device=pose_pred.device,
                dtype=pose_pred.dtype,
            )

        if self.norm_by_hand and coco_sample_mask.numel() > 0 and torch.any(coco_sample_mask > 0.5):
            raise AssertionError(
                "COCO-WholeBody 2D patch loss does not support norm_by_hand=true yet. "
                "Please set MODEL.norm_by_hand=false."
            )

        if self.norm_by_hand:
            norm_scale_gt, norm_valid_gt = self.get_hand_norm_scale(
                batch["joint_cam"], batch["joint_3d_valid"]
            )
        else:
            norm_valid_gt = torch.ones_like(has_mano)

        trans_gt = batch["joint_cam"][:, :, 0]
        if self.norm_by_hand:
            trans_gt = trans_gt / (norm_scale_gt[..., None] + NORM_SCALE_EPSILON)

        loss_theta = self.l1(pose_pred, pose_gt)
        loss_theta = robust_masked_mean(loss_theta, has_mano[..., None])
        loss_shape = self.l1(shape_pred, shape_gt)
        loss_shape = robust_masked_mean(loss_shape, has_mano[..., None])
        root_valid_mask = joint_3d_valid[:, :, 0] * norm_valid_gt
        if self.cam_head_type == "softargmax3d":
            if not self.supervise_heatmap:
                loss_trans = self.l1(trans_pred, trans_gt)
                loss_trans = robust_masked_mean(
                    loss_trans,
                    joint_3d_valid[:, :, :1] * norm_valid_gt[..., None],
                )
            else:
                loss_trans = (
                    self.compute_hm_ce(cam_aux["log_hm_x"], trans_gt[..., 0], self.x_centers)
                    + self.compute_hm_ce(cam_aux["log_hm_y"], trans_gt[..., 1], self.y_centers)
                    + self.compute_hm_ce(cam_aux["log_hm_z"], trans_gt[..., 2], self.z_centers)
                )
                loss_trans = robust_masked_mean(
                    loss_trans,
                    root_valid_mask,
                )
            loss_trans_xy = loss_trans
            loss_root_z_cls = self._zero_like(trans_pred)
            loss_root_z_res = self._zero_like(trans_pred)
            root_z_bin_acc = self._zero_like(trans_pred)
            root_z_mae_mm = self._zero_like(trans_pred)
        elif self.cam_head_type == "xy_rootz_multibin":
            if self.norm_by_hand:
                raise NotImplementedError("xy_rootz_multibin does not support norm_by_hand=true")
            if not self.supervise_heatmap:
                loss_trans_xy = self.l1(trans_pred[..., :2], trans_gt[..., :2])
                loss_trans_xy = robust_masked_mean(
                    loss_trans_xy,
                    root_valid_mask[..., None],
                )
            else:
                loss_trans_xy = (
                    self.compute_hm_ce(cam_aux["log_hm_x"], trans_gt[..., 0], self.x_centers)
                    + self.compute_hm_ce(cam_aux["log_hm_y"], trans_gt[..., 1], self.y_centers)
                )
                loss_trans_xy = robust_masked_mean(loss_trans_xy, root_valid_mask)

            root_z_frame_filter = self.compute_root_z_frame_filter_mask(batch)
            root_z_valid = (
                root_valid_mask
                * (has_intr > 0.5).float()
                * (trans_gt[..., 2] > 0.0).float()
                * root_z_frame_filter
            )
            encoded_root_z = encode_delta_log_z_targets(
                root_z=trans_gt[..., 2],
                log_z_prior=cam_aux["log_z_prior"].squeeze(-1),
                d_min=self.root_z_d_min,
                d_max=self.root_z_d_max,
                num_bins=self.root_z_num_bins,
            )
            valid_bool = root_z_valid > 0.5
            if torch.any(valid_bool):
                loss_root_z_cls = F.cross_entropy(
                    cam_aux["z_cls_logits"][valid_bool],
                    encoded_root_z["bin_idx"][valid_bool],
                    reduction="mean",
                )
                pred_root_z_res = cam_aux["z_residuals"].gather(
                    dim=-1,
                    index=encoded_root_z["bin_idx"].unsqueeze(-1),
                ).squeeze(-1)
                loss_root_z_res = F.smooth_l1_loss(
                    pred_root_z_res[valid_bool],
                    encoded_root_z["residual"][valid_bool],
                    reduction="mean",
                    beta=0.1,
                )
                pred_bin = torch.argmax(cam_aux["z_cls_logits"], dim=-1)
                root_z_bin_acc = (pred_bin[valid_bool] == encoded_root_z["bin_idx"][valid_bool]).float().mean()
                root_z_mae_mm = torch.abs(trans_pred[..., 2] - trans_gt[..., 2])
                root_z_mae_mm = self._masked_mean_scalar(root_z_mae_mm, root_z_valid)
            else:
                loss_root_z_cls = self._zero_like(trans_pred)
                loss_root_z_res = self._zero_like(trans_pred)
                root_z_bin_acc = self._zero_like(trans_pred)
                root_z_mae_mm = self._zero_like(trans_pred)
            loss_trans = loss_trans_xy
        else:
            raise ValueError(f"Unsupported cam_head_type in loss: {self.cam_head_type}")

        loss_joint_rel = self.l1(joint_rel_pred, batch["joint_rel"])
        loss_joint_rel = robust_masked_mean(
            loss_joint_rel,
            joint_3d_valid[..., None],
        )

        if self.norm_by_hand:
            trans_pred_scaled = trans_pred * (norm_scale_gt[..., None] + NORM_SCALE_EPSILON)
        else:
            trans_pred_scaled = trans_pred

        joint_cam_gt = batch["joint_cam"]
        joint_cam_pred = joint_rel_pred + trans_pred_scaled[:, :, None, :]
        joint_img_gt = proj_points_3d(
            joint_cam_gt, batch["focal"], batch["princpt"]
        )
        joint_img_pred = proj_points_3d(
            joint_cam_pred, batch["focal"], batch["princpt"]
        )
        reproj_valid = joint_3d_valid * has_intr[..., None]
        loss_joint_img = self.reproj_loss_fn(joint_img_pred, joint_img_gt)
        loss_joint_img = robust_masked_mean(
            loss_joint_img,
            reproj_valid[..., None] * norm_valid_gt[..., None, None],
        )

        patch_size = tuple(batch["patches"].shape[-2:])
        joint_patch_pred = joint_img_to_patch_resized(
            joint_img_pred,
            batch["patch_bbox"],
            patch_size=patch_size,
        )
        coco_patch_valid = (
            joint_2d_valid
            * has_intr[..., None]
            * coco_sample_mask[:, :, None]
        )
        loss_coco_patch_2d = self.l1(
            joint_patch_pred,
            batch["joint_patch_resized"],
        )
        loss_coco_patch_2d = robust_masked_mean(
            loss_coco_patch_2d,
            coco_patch_valid[..., None],
        )

        loss = (
            self.lambda_theta * loss_theta
            + self.lambda_shape * loss_shape
            + self.lambda_trans * loss_trans_xy
            + self.lambda_root_z_cls * loss_root_z_cls
            + self.lambda_root_z_res * loss_root_z_res
            + self.lambda_rel * loss_joint_rel
            + self.lambda_img * loss_joint_img
            + self.lambda_coco_patch_2d * loss_coco_patch_2d
        )

        loss_state = {
            "loss_theta": loss_theta.detach(),
            "loss_shape": loss_shape.detach(),
            "loss_trans": loss_trans.detach(),
            "loss_trans_xy": loss_trans_xy.detach(),
            "loss_root_z_cls": loss_root_z_cls.detach(),
            "loss_root_z_res": loss_root_z_res.detach(),
            "root_z_bin_acc": root_z_bin_acc.detach(),
            "root_z_mae_mm": root_z_mae_mm.detach(),
            "root_z_valid_frac": root_z_valid.mean().detach() if self.cam_head_type == "xy_rootz_multibin" else self._zero_like(trans_pred).detach(),
            "loss_joint_rel": loss_joint_rel.detach(),
            "loss_joint_img": loss_joint_img.detach(),
            "loss_coco_patch_2d": loss_coco_patch_2d.detach(),
        }

        fk_result = {
            "verts_cam_gt": vert_rel_gt + batch["joint_cam"][:, :, :1],
            "verts_rel_gt": vert_rel_gt,
            "joint_cam_pred": joint_cam_pred,
            "joint_rel_pred": joint_rel_pred,
            "verts_cam_pred": vert_rel_pred + trans_pred_scaled[..., None, :],
            "verts_rel_pred": vert_rel_pred,
            "norm_valid_gt": norm_valid_gt,
        }

        return loss, loss_state, fk_result
