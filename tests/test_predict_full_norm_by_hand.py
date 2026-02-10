"""
单元测试：验证 predict_full() 中 norm_by_hand 反归一化逻辑的修复

测试场景：
1. 有 GT 且 norm_valid=1（所有 norm_idx joints 都有效）
2. 有 GT 但 norm_valid=0（部分 norm_idx joints 无效）
3. 无 GT
"""

import torch
import pytest
from src.model.net import PoseNet
from accelerate import PartialState


@pytest.fixture
def model():
    # 初始化 Accelerate 状态（避免 logger 报错）
    try:
        PartialState()
    except RuntimeError:
        # 如果已经初始化过，忽略错误
        pass
    """创建测试模型（参考 test_src_model_net.py 的方式）"""
    net = PoseNet(
        stage="stage1",
        stage1_weight_path=None,
        #
        backbone_str="model/facebook/dinov2-base",
        img_size=224,
        img_mean=[0., 0., 0.],
        img_std=[1., 1., 1.],
        infusion_feats_lyr=[2, 6, 10],
        drop_cls=False,
        backbone_kwargs=None,
        #
        num_handec_layer=6,
        num_handec_head=12,
        ndim_handec_mlp=1024,
        ndim_handec_head=64,
        prob_handec_dropout=0.0,
        prob_handec_emb_dropout=0.0,
        handec_emb_dropout_type="drop",
        handec_norm="layer",
        ndim_handec_norm_cond_dim=-1,
        ndim_handec_ctx=768,
        handec_skip_token_embed=False,
        handec_mean_init=True,
        handec_denorm_output=False,
        handec_heatmap_resulotion=[512, 512, 1024],
        #
        pie_type="ca",
        num_pie_sample=16,
        pie_fusion="all",
        #
        num_temporal_head=12,
        num_temporal_layer=2,
        trope_scalar=20.,
        zero_linear=True,
        #
        joint_rep_type="3",
        #
        supervise_global=True,
        supervise_heatmap=False,
        lambda_theta=1.0,
        lambda_shape=1.0,
        lambda_trans=1.0,
        lambda_rel=1.0,
        lambda_img=1.0,
        hm_sigma=1.0,
        #
        freeze_backbone=False,
        norm_by_hand=True,  # 启用 norm_by_hand
    )
    net.eval()
    return net


def test_predict_full_with_valid_gt(model):
    """
    测试场景 1：有 GT 且 norm_valid=1
    期望：使用 GT 的 norm_scale
    """
    B = 2
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 准备测试数据
    batch = {
        "patches": torch.randn(B, 1, 3, 224, 224, device=device),
        "patch_bbox": torch.tensor([
            [[100, 100, 300, 300]],
            [[150, 150, 350, 350]]
        ], dtype=torch.float32, device=device),
        "focal": torch.tensor([
            [[1000, 1000]],
            [[1100, 1100]]
        ], dtype=torch.float32, device=device),
        "princpt": torch.tensor([
            [[512, 512]],
            [[512, 512]]
        ], dtype=torch.float32, device=device),
        "timestamp": torch.zeros(B, 1, device=device),
        "joint_cam": torch.randn(B, 1, 21, 3, device=device) * 100,  # GT joints
        "joint_valid": torch.ones(B, 1, 21, device=device),  # 所有 joints 都有效
    }

    # 推理
    with torch.no_grad():
        result = model.predict_full(
            img=batch["patches"],
            bbox=batch["patch_bbox"],
            focal=batch["focal"],
            princpt=batch["princpt"],
            timestamp=batch["timestamp"],
            joint_cam_gt=batch["joint_cam"],
            joint_valid_gt=batch["joint_valid"],
        )

    # 验证返回值
    assert result["trans_pred"].shape == (B, 1, 3)
    assert result["trans_pred_denorm"].shape == (B, 1, 3)
    assert result["norm_scale"].shape == (B, 1)
    assert result["norm_valid"].shape == (B, 1)

    # 验证 norm_valid 为 1（有效）
    assert torch.all(result["norm_valid"] == 1.0), \
        f"Expected norm_valid=1, got {result['norm_valid']}"

    # 验证反归一化公式
    expected_denorm = result["trans_pred"] * result["norm_scale"][:, :, None]
    assert torch.allclose(result["trans_pred_denorm"], expected_denorm, atol=1e-5), \
        "Denormalization formula mismatch"

    print("✓ 测试通过：有 GT 且 norm_valid=1")


def test_predict_full_with_invalid_gt(model):
    """
    测试场景 2：有 GT 但 norm_valid=0（部分 norm_idx joints 无效）
    期望：fallback 到使用预测的 norm_scale
    """
    B = 2
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 准备测试数据
    batch = {
        "patches": torch.randn(B, 1, 3, 224, 224, device=device),
        "patch_bbox": torch.tensor([
            [[100, 100, 300, 300]],
            [[150, 150, 350, 350]]
        ], dtype=torch.float32, device=device),
        "focal": torch.tensor([
            [[1000, 1000]],
            [[1100, 1100]]
        ], dtype=torch.float32, device=device),
        "princpt": torch.tensor([
            [[512, 512]],
            [[512, 512]]
        ], dtype=torch.float32, device=device),
        "timestamp": torch.zeros(B, 1, device=device),
        "joint_cam": torch.randn(B, 1, 21, 3, device=device) * 100,  # GT joints
        "joint_valid": torch.ones(B, 1, 21, device=device),  # 初始化为全部有效
    }

    # 手动设置部分 norm_idx joints 为无效
    # norm_idx = [0, 5, 9, 13, 17]（默认值）
    batch["joint_valid"][0, 0, 0] = 0  # 第 0 个样本的第 0 个 joint 无效
    batch["joint_valid"][1, 0, 5] = 0  # 第 1 个样本的第 5 个 joint 无效

    # 推理
    with torch.no_grad():
        result = model.predict_full(
            img=batch["patches"],
            bbox=batch["patch_bbox"],
            focal=batch["focal"],
            princpt=batch["princpt"],
            timestamp=batch["timestamp"],
            joint_cam_gt=batch["joint_cam"],
            joint_valid_gt=batch["joint_valid"],
        )

    # 验证返回值
    assert result["trans_pred"].shape == (B, 1, 3)
    assert result["trans_pred_denorm"].shape == (B, 1, 3)
    assert result["norm_scale"].shape == (B, 1)
    assert result["norm_valid"].shape == (B, 1)

    # 验证 norm_valid：因为 fallback 到 pred，pred 假设所有 joints 都有效
    # 所以最终 norm_valid = max(norm_valid_gt, norm_valid_pred) = max(0, 1) = 1
    assert torch.all(result["norm_valid"] == 1.0), \
        f"Expected norm_valid=1 (fallback to pred), got {result['norm_valid']}"

    # 验证反归一化公式
    expected_denorm = result["trans_pred"] * result["norm_scale"][:, :, None]
    assert torch.allclose(result["trans_pred_denorm"], expected_denorm, atol=1e-5), \
        "Denormalization formula mismatch"

    print("✓ 测试通过：有 GT 但 norm_valid=0（使用 pred scale）")


def test_predict_full_without_gt(model):
    """
    测试场景 3：无 GT
    期望：使用预测的 norm_scale
    """
    B = 2
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 准备测试数据
    batch = {
        "patches": torch.randn(B, 1, 3, 224, 224, device=device),
        "patch_bbox": torch.tensor([
            [[100, 100, 300, 300]],
            [[150, 150, 350, 350]]
        ], dtype=torch.float32, device=device),
        "focal": torch.tensor([
            [[1000, 1000]],
            [[1100, 1100]]
        ], dtype=torch.float32, device=device),
        "princpt": torch.tensor([
            [[512, 512]],
            [[512, 512]]
        ], dtype=torch.float32, device=device),
        "timestamp": torch.zeros(B, 1, device=device),
    }

    # 推理（无 GT）
    with torch.no_grad():
        result = model.predict_full(
            img=batch["patches"],
            bbox=batch["patch_bbox"],
            focal=batch["focal"],
            princpt=batch["princpt"],
            timestamp=batch["timestamp"],
            joint_cam_gt=None,       # 无 GT
            joint_valid_gt=None,     # 无 GT
        )

    # 验证返回值
    assert result["trans_pred"].shape == (B, 1, 3)
    assert result["trans_pred_denorm"].shape == (B, 1, 3)
    assert result["norm_scale"].shape == (B, 1)
    assert result["norm_valid"].shape == (B, 1)

    # 验证 norm_valid：pred 假设所有 joints 都有效
    assert torch.all(result["norm_valid"] == 1.0), \
        f"Expected norm_valid=1 (pred), got {result['norm_valid']}"

    # 验证反归一化公式
    expected_denorm = result["trans_pred"] * result["norm_scale"][:, :, None]
    assert torch.allclose(result["trans_pred_denorm"], expected_denorm, atol=1e-5), \
        "Denormalization formula mismatch"

    print("✓ 测试通过：无 GT（使用 pred scale）")


def test_predict_full_mixed_validity(model):
    """
    测试场景 4：批次中混合情况（部分样本 valid，部分 invalid）
    期望：逐样本选择 GT 或 pred scale
    """
    B = 4
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 准备测试数据
    batch = {
        "patches": torch.randn(B, 1, 3, 224, 224, device=device),
        "patch_bbox": torch.tensor([
            [[100, 100, 300, 300]],
            [[150, 150, 350, 350]],
            [[200, 200, 400, 400]],
            [[250, 250, 450, 450]],
        ], dtype=torch.float32, device=device),
        "focal": torch.ones(B, 1, 2, device=device) * 1000,
        "princpt": torch.ones(B, 1, 2, device=device) * 512,
        "timestamp": torch.zeros(B, 1, device=device),
        "joint_cam": torch.randn(B, 1, 21, 3, device=device) * 100,
        "joint_valid": torch.ones(B, 1, 21, device=device),
    }

    # 设置混合 validity
    # 样本 0, 1: norm_idx joints 全部有效（norm_valid_gt=1）
    # 样本 2, 3: 部分 norm_idx joints 无效（norm_valid_gt=0）
    batch["joint_valid"][2, 0, 0] = 0  # 样本 2
    batch["joint_valid"][3, 0, 5] = 0  # 样本 3

    # 推理
    with torch.no_grad():
        result = model.predict_full(
            img=batch["patches"],
            bbox=batch["patch_bbox"],
            focal=batch["focal"],
            princpt=batch["princpt"],
            timestamp=batch["timestamp"],
            joint_cam_gt=batch["joint_cam"],
            joint_valid_gt=batch["joint_valid"],
        )

    # 验证返回值
    assert result["trans_pred"].shape == (B, 1, 3)
    assert result["trans_pred_denorm"].shape == (B, 1, 3)
    assert result["norm_scale"].shape == (B, 1)
    assert result["norm_valid"].shape == (B, 1)

    # 验证 norm_valid：所有样本都应该有效（因为 fallback 到 pred）
    assert torch.all(result["norm_valid"] == 1.0), \
        f"Expected all norm_valid=1, got {result['norm_valid']}"

    # 验证反归一化公式
    expected_denorm = result["trans_pred"] * result["norm_scale"][:, :, None]
    assert torch.allclose(result["trans_pred_denorm"], expected_denorm, atol=1e-5), \
        "Denormalization formula mismatch"

    print("✓ 测试通过：批次混合情况（逐样本选择 scale）")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
