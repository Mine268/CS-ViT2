import torch


def compute_mpjpe_stats(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor = None,
):
    """
    计算 MPJPE 的累积统计量。

    Args:
        pred: 预测的关节坐标 [B, T, J, 3] 或 [..., J, 3]
        gt: 真实的关节坐标，形状同 pred
        mask: 关节级有效性掩码 [B, T, J] 或 [..., J]
    """
    error_per_joint = torch.norm(pred - gt, p=2, dim=-1)

    if mask is None:
        return error_per_joint.sum(), torch.tensor(error_per_joint.numel(), device=pred.device)

    mask_bool = mask > 0.5
    if mask_bool.any():
        return error_per_joint[mask_bool].sum(), mask_bool.sum()

    return torch.tensor(0.0, device=pred.device), torch.tensor(0.0, device=pred.device)


def compute_mpvpe_stats(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor = None,
):
    """
    计算 MPVPE 的累积统计量。

    Args:
        pred: 预测的顶点坐标 [B, T, V, 3]
        gt: 真实的顶点坐标 [B, T, V, 3]
        mask: 帧级有效性掩码 [B, T]
    """
    if pred is None or gt is None:
        zero = torch.tensor(0.0)
        return zero, zero

    error_per_vertex = torch.norm(pred - gt, p=2, dim=-1)
    error_per_vertex = torch.mean(error_per_vertex, dim=-1)

    if mask is None:
        return error_per_vertex.sum(), torch.tensor(error_per_vertex.numel(), device=pred.device)

    mask_bool = mask > 0.5
    if mask_bool.any():
        return (error_per_vertex * mask).sum(), mask_bool.sum()

    return torch.tensor(0.0, device=pred.device), torch.tensor(0.0, device=pred.device)


def compute_rte_stats(pred, gt, mask):
    """
    pred, gt: [b,t,3]
    mask: [b,t]
    """
    if pred is None or gt is None:
        zero = torch.tensor(0.0)
        return zero, zero

    error_per = torch.norm(pred - gt, p=2, dim=-1)
    if mask is None:
        return error_per.sum(), torch.tensor(error_per.numel(), device=pred.device)

    mask_bool = mask > 0.5
    if mask_bool.any():
        valid_errors = error_per[mask_bool]
        return valid_errors.sum(), torch.tensor(valid_errors.numel(), device=pred.device)

    return torch.tensor(0.0, device=pred.device), torch.tensor(0.0, device=pred.device)


def _safe_ratio(total_error: torch.Tensor, total_count: torch.Tensor) -> torch.Tensor:
    if total_count.item() <= 0:
        return torch.tensor(0.0, device=total_error.device)
    return total_error / total_count


class MetricMeter:
    def __init__(self, *args, **kwargs):
        pass

    @torch.no_grad()
    def __call__(
        self,
        joint_cam_gt,
        joint_rel_gt,
        verts_cam_gt,
        verts_rel_gt,
        joint_cam_pred,
        joint_rel_pred,
        verts_cam_pred,
        verts_rel_pred,
        has_mano,
        joint_3d_valid,
        norm_valid,
    ):
        cs_mpjpe_stats = compute_mpjpe_stats(
            joint_cam_pred,
            joint_cam_gt,
            joint_3d_valid * norm_valid[:, :, None],
        )
        rs_mpjpe_stats = compute_mpjpe_stats(
            joint_rel_pred,
            joint_rel_gt,
            joint_3d_valid * norm_valid[:, :, None],
        )
        cs_mpvpe_stats = compute_mpvpe_stats(
            verts_cam_pred,
            verts_cam_gt,
            has_mano * norm_valid,
        )
        rs_mpvpe_stats = compute_mpvpe_stats(
            verts_rel_pred,
            verts_rel_gt,
            has_mano * norm_valid,
        )
        rte_stats = compute_rte_stats(
            joint_cam_pred[:, :, 0],
            joint_cam_gt[:, :, 0],
            joint_3d_valid[:, :, 0] * norm_valid,
        )

        return {
            "micro_mpjpe": _safe_ratio(*cs_mpjpe_stats),
            "micro_mpjpe_rel": _safe_ratio(*rs_mpjpe_stats),
            "micro_mpvpe": _safe_ratio(*cs_mpvpe_stats),
            "micro_mpvpe_rel": _safe_ratio(*rs_mpvpe_stats),
            "micro_rte": _safe_ratio(*rte_stats),
        }


class StreamingMetricMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.accumulators = {
            "cs_mpjpe": [0.0, 0.0],
            "rs_mpjpe": [0.0, 0.0],
            "cs_mpvpe": [0.0, 0.0],
            "rs_mpvpe": [0.0, 0.0],
            "rte": [0.0, 0.0],
        }

    @torch.no_grad()
    def update(
        self,
        joint_cam_gt,
        joint_rel_gt,
        verts_cam_gt,
        verts_rel_gt,
        joint_cam_pred,
        joint_rel_pred,
        verts_cam_pred,
        verts_rel_pred,
        has_mano,
        joint_3d_valid,
        norm_valid,
    ):
        self._accumulate(
            "cs_mpjpe",
            compute_mpjpe_stats(
                joint_cam_pred,
                joint_cam_gt,
                joint_3d_valid * norm_valid[:, :, None],
            ),
        )
        self._accumulate(
            "rs_mpjpe",
            compute_mpjpe_stats(
                joint_rel_pred,
                joint_rel_gt,
                joint_3d_valid * norm_valid[:, :, None],
            ),
        )
        self._accumulate(
            "cs_mpvpe",
            compute_mpvpe_stats(
                verts_cam_pred,
                verts_cam_gt,
                has_mano * norm_valid,
            ),
        )
        self._accumulate(
            "rs_mpvpe",
            compute_mpvpe_stats(
                verts_rel_pred,
                verts_rel_gt,
                has_mano * norm_valid,
            ),
        )
        self._accumulate(
            "rte",
            compute_rte_stats(
                joint_cam_pred[:, :, 0],
                joint_cam_gt[:, :, 0],
                joint_3d_valid[:, :, 0] * norm_valid,
            ),
        )

    def _accumulate(self, key, stats_tuple):
        self.accumulators[key][0] += stats_tuple[0].item()
        self.accumulators[key][1] += stats_tuple[1].item()

    def compute(self):
        results = {}
        key_map = {
            "cs_mpjpe": "micro_mpjpe",
            "rs_mpjpe": "micro_mpjpe_rel",
            "cs_mpvpe": "micro_mpvpe",
            "rs_mpvpe": "micro_mpvpe_rel",
            "rte": "micro_rte",
        }

        for internal_key, output_key in key_map.items():
            total_error, total_count = self.accumulators[internal_key]
            if total_count > 0:
                results[output_key] = total_error / total_count
            else:
                results[output_key] = 0.0

        return results
