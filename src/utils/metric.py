import torch
import torch.nn as nn


def compute_mpjpe_stats(
    pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor = None
):
    """
    计算 MPJPE 的累积统计量。

    Args:
        pred: 预测的关节坐标 [B, T, J, 3] 或 [..., J, 3]
        gt: 真实的关节坐标，形状同 pred
        mask: 关节有效性掩码 [B, T, J] 或 [..., J]。如果为 None，则计算所有点。
              (对应原代码中的 joint_valid)

    Returns:
        tuple: (batch_total_error, batch_total_count)
               - batch_total_error: 当前 batch 所有有效关节的误差之和 (float)
               - batch_total_count: 当前 batch 有效关节的总数 (int)
    """
    # 1. 计算欧式距离 (L2 Norm)
    # [B, T, J, 3] -> [B, T, J]
    error_per_joint = torch.norm(pred - gt, p=2, dim=-1)

    # 2. 如果没有 mask，假定全部有效
    if mask is None:
        return error_per_joint.sum(), error_per_joint.numel()

    # 3. 应用 Mask
    # 确保 mask 是布尔类型
    mask_bool = mask > 0.5

    if mask_bool.any():
        # error_per_joint[mask_bool] 会展平为一维向量
        batch_total_error = error_per_joint[mask_bool].sum()
        batch_total_count = mask_bool.sum()
        return batch_total_error, batch_total_count
    else:
        # 如果当前 batch 没有有效数据
        return torch.tensor(0.0, device=pred.device), torch.tensor(
            0.0, device=pred.device
        )


def compute_mpvpe_stats(
    pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor = None
):
    """
    计算 MPVPE 的累积统计量。

    Args:
        pred: 预测的顶点坐标 [B, T, V, 3]
        gt: 真实的顶点坐标 [B, T, V, 3]
        mask: 手部/帧级有效性掩码 [B, T]。注意这里维度通常比 pred 少一维。
              (对应原代码中的 mano_valid)

    Returns:
        tuple: (batch_total_error, batch_total_count)
    """
    # 1. 基础检查
    if pred is None or gt is None:
        return torch.tensor(0.0), torch.tensor(0.0)

    # 2. 计算欧式距离
    # [B, T, V, 3] -> [B, T, V]
    error_per_vertex = torch.norm(pred - gt, p=2, dim=-1)

    # 3. 处理 Mask
    # 原逻辑中 metrics_accum[3] += verts_error[mask_v].numel()
    # 意味着 mask [B, T] 广播到了 [B, T, V]

    if mask is None:
        return error_per_vertex.sum(), error_per_vertex.numel()

    mask_bool = mask > 0.5  # [B, T]

    if mask_bool.any():
        # 利用 PyTorch 的高级索引:
        # error_per_vertex[mask_bool] 会选出 mask 为 True 的那些帧的所有顶点
        # 结果形状为 [N_valid_frames, V]
        valid_errors = error_per_vertex[mask_bool]

        batch_total_error = valid_errors.sum()
        batch_total_count = valid_errors.numel()  # 自动计算 N_valid_frames * V
        return batch_total_error, torch.tensor(batch_total_count, device=pred.device)
    else:
        return torch.tensor(0.0, device=pred.device), torch.tensor(
            0.0, device=pred.device
        )


def compute_rte_stats(pred, gt, mask):
    """
    pred, gt: [b,t,3]
    mask: [b,t]
    """
    if pred is None or gt is None:
        return torch.tensor(0.0), torch.tensor(0.0)

    error_per = torch.norm(pred - gt, p=2, dim=-1)  # [b,t]

    if mask is None:
        return error_per.sum(), error_per.numel()

    mask_bool = mask > 0.5  # [B, T]

    if mask_bool.any():
        valid_errors = error_per[mask_bool]

        batch_total_error = valid_errors.sum()
        batch_total_count = valid_errors.numel()
        return batch_total_error, torch.tensor(batch_total_count, device=pred.device)
    else:
        return torch.tensor(0.0, device=pred.device), torch.tensor(
            0.0, device=pred.device
        )


class MetricMeter:
    def __init__(self, *args, **kwargs):
        # super().__init__(*args, **kwargs)
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
        mano_valid,
        joint_valid,
        norm_valid,
    ):
        # cs-mpjpe
        cs_mpjpe = compute_mpjpe_stats(
            joint_cam_pred, joint_cam_gt, joint_valid * norm_valid[:, :, None]
        )
        cs_mpjpe = cs_mpjpe[0] / cs_mpjpe[1]

        # rs-mpjpe
        rs_mpjpe = compute_mpjpe_stats(
            joint_rel_pred, joint_rel_gt, joint_valid * norm_valid[:, :, None]
        )
        rs_mpjpe = rs_mpjpe[0] / rs_mpjpe[1]

        # cs-mpvpe
        cs_mpvpe = compute_mpvpe_stats(
            verts_cam_pred, verts_cam_gt, mano_valid * norm_valid
        )
        cs_mpvpe = cs_mpvpe[0] / cs_mpvpe[1]

        # rs-mpvpe
        rs_mpvpe = compute_mpvpe_stats(
            verts_rel_pred, verts_rel_gt, mano_valid * norm_valid
        )
        rs_mpvpe = rs_mpvpe[0] / rs_mpvpe[1]

        # cs-rte
        rte = compute_rte_stats(
            joint_cam_pred[:, :, 0],
            joint_cam_gt[:, :, 0],
            joint_valid[:, :, 0] * norm_valid,
        )
        rte = rte[0] / rte[1]

        return {
            "micro_mpjpe": cs_mpjpe,
            "micro_mpjpe_rel": rs_mpjpe,
            "micro_mpvpe": cs_mpvpe,
            "micro_mpvpe_rel": rs_mpvpe,
            "micro_rte": rte,
        }


class StreamingMetricMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有累加器"""
        # 存储结构: key -> [total_error (分子), total_count (分母)]
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
        mano_valid,
        joint_valid,
        norm_valid,
    ):
        """
        接收一个Batch的数据，计算统计量并累加到内部状态中。
        不会返回当前Batch的结果，只做存储。
        """

        # --- 1. CS-MPJPE ---
        stats_cs_mpjpe = compute_mpjpe_stats(
            joint_cam_pred, joint_cam_gt, joint_valid * norm_valid[:, :, None]
        )
        self._accumulate("cs_mpjpe", stats_cs_mpjpe)

        # --- 2. RS-MPJPE ---
        stats_rs_mpjpe = compute_mpjpe_stats(
            joint_rel_pred, joint_rel_gt, joint_valid * norm_valid[:, :, None]
        )
        self._accumulate("rs_mpjpe", stats_rs_mpjpe)

        # --- 3. CS-MPVPE ---
        stats_cs_mpvpe = compute_mpvpe_stats(
            verts_cam_pred, verts_cam_gt, mano_valid * norm_valid
        )
        self._accumulate("cs_mpvpe", stats_cs_mpvpe)

        # --- 4. RS-MPVPE ---
        stats_rs_mpvpe = compute_mpvpe_stats(
            verts_rel_pred, verts_rel_gt, mano_valid * norm_valid
        )
        self._accumulate("rs_mpvpe", stats_rs_mpvpe)

        # --- 5. CS-RTE ---
        stats_rte = compute_rte_stats(
            joint_cam_pred[:, :, 0],
            joint_cam_gt[:, :, 0],
            joint_valid[:, :, 0] * norm_valid,
        )
        self._accumulate("rte", stats_rte)

    def _accumulate(self, key, stats_tuple):
        """辅助函数：将Tensor转为Python float并累加，避免显存泄露"""
        # stats_tuple[0] 是总误差 (Sum of Errors)
        # stats_tuple[1] 是有效样本数 (Count)
        self.accumulators[key][0] += stats_tuple[0].item()
        self.accumulators[key][1] += stats_tuple[1].item()

    def compute(self):
        """
        计算并返回截至目前所有数据的平均指标。
        """
        results = {}

        # 映射内部key到输出key
        key_map = {
            "cs_mpjpe": "micro_mpjpe",
            "rs_mpjpe": "micro_mpjpe_rel",
            "cs_mpvpe": "micro_mpvpe",
            "rs_mpvpe": "micro_mpvpe_rel",
            "rte": "micro_rte",
        }

        for internal_key, output_key in key_map.items():
            total_error, total_count = self.accumulators[internal_key]
            # 防止除以0
            if total_count > 0:
                results[output_key] = total_error / total_count
            else:
                results[output_key] = 0.0

        return results
