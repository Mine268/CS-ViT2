import os
import sys

from accelerate import PartialState
from omegaconf import OmegaConf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from script.train import setup_model
from src.model.root_z import (
    RHO_GEOM_DIM,
    compute_rho_prior_and_geom,
    compute_root_z_prior_and_geom,
    decode_delta_log_rho_predictions,
    encode_delta_log_rho_targets,
)
from src.model.module import MANOTransformerDecoderHead


def test_compute_rho_prior_and_geom_matches_bbox_center_ray_correction():
    hand_bbox = torch.tensor(
        [
            [100.0, 120.0, 180.0, 220.0],
            [50.0, 40.0, 90.0, 100.0],
        ],
        dtype=torch.float32,
    )
    focal = torch.tensor(
        [
            [1000.0, 1200.0],
            [800.0, 900.0],
        ],
        dtype=torch.float32,
    )
    princpt = torch.tensor(
        [
            [128.0, 128.0],
            [128.0, 128.0],
        ],
        dtype=torch.float32,
    )

    z_prior, _, root_z_geom = compute_root_z_prior_and_geom(
        hand_bbox=hand_bbox,
        focal=focal,
        princpt=princpt,
        prior_k=121.0,
    )
    rho_prior, log_rho_prior, rho_geom = compute_rho_prior_and_geom(
        hand_bbox=hand_bbox,
        focal=focal,
        princpt=princpt,
        prior_k=121.0,
    )

    q_ref_norm = torch.sqrt(root_z_geom[:, 1] ** 2 + root_z_geom[:, 2] ** 2 + 1.0)
    assert rho_prior.shape == (2,)
    assert log_rho_prior.shape == (2,)
    assert rho_geom.shape == (2, RHO_GEOM_DIM)
    assert torch.allclose(rho_prior, z_prior * q_ref_norm, atol=1e-5)
    assert torch.allclose(rho_geom[:, 0], log_rho_prior, atol=1e-5)


def test_encode_decode_delta_log_rho_roundtrip():
    rho = torch.tensor([1000.0, 1800.0, 720.0], dtype=torch.float32)
    log_rho_prior = torch.log(torch.tensor([900.0, 1500.0, 800.0], dtype=torch.float32))
    d_min = -0.71
    d_max = 0.75
    num_bins = 8

    encoded = encode_delta_log_rho_targets(
        rho=rho,
        log_rho_prior=log_rho_prior,
        d_min=d_min,
        d_max=d_max,
        num_bins=num_bins,
    )

    logits = torch.full((3, num_bins), -20.0, dtype=torch.float32)
    logits.scatter_(1, encoded["bin_idx"].unsqueeze(-1), 20.0)
    residuals = torch.zeros((3, num_bins), dtype=torch.float32)
    residuals.scatter_(1, encoded["bin_idx"].unsqueeze(-1), encoded["residual"].unsqueeze(-1))

    decoded = decode_delta_log_rho_predictions(
        rho_cls_logits=logits,
        rho_residuals=residuals,
        log_rho_prior=log_rho_prior,
        d_min=d_min,
        d_max=d_max,
    )

    assert torch.allclose(
        decoded["pred_delta_log_rho"],
        encoded["delta_log_rho_clamped"],
        atol=1e-5,
    )
    assert torch.allclose(
        decoded["pred_rho"],
        torch.exp(log_rho_prior + encoded["delta_log_rho_clamped"]),
        atol=1e-4,
    )


def test_mano_transformer_decoder_head_patch_uv_rho_multibin_outputs_expected_shapes():
    head = MANOTransformerDecoderHead(
        joint_rep_type="3",
        dim=32,
        depth=2,
        heads=4,
        mlp_dim=64,
        dim_head=8,
        dropout=0.0,
        emb_dropout=0.0,
        emb_dropout_type="drop",
        norm="layer",
        norm_cond_dim=-1,
        context_dim=32,
        skip_token_embedding=False,
        use_mean_init=False,
        denorm_output=False,
        norm_by_hand=False,
        heatmap_resolution=(16, 16, 32),
        patch_size=(224, 224),
        cam_head_type="patch_uv_rho_multibin",
        root_z_num_bins=8,
        root_z_d_min=-0.71,
        root_z_d_max=0.75,
        root_z_prior_k=121.0,
        root_z_geom_hidden_dim=16,
        root_z_dropout=0.0,
        root_z_use_data_source_embed=False,
    )

    x = torch.randn(2, 10, 32)
    patch_bbox = torch.tensor(
        [
            [40.0, 40.0, 184.0, 184.0],
            [50.0, 60.0, 210.0, 220.0],
        ],
        dtype=torch.float32,
    )
    hand_bbox = torch.tensor(
        [
            [70.0, 78.0, 154.0, 166.0],
            [90.0, 100.0, 170.0, 196.0],
        ],
        dtype=torch.float32,
    )
    focal = torch.tensor(
        [
            [1000.0, 1000.0],
            [900.0, 950.0],
        ],
        dtype=torch.float32,
    )
    princpt = torch.tensor(
        [
            [112.0, 112.0],
            [128.0, 128.0],
        ],
        dtype=torch.float32,
    )

    (pred_pose, pred_shape, pred_cam), cam_aux, token_out = head(
        x,
        patch_bbox=patch_bbox,
        hand_bbox=hand_bbox,
        focal=focal,
        princpt=princpt,
    )

    assert pred_pose.shape == (2, 48)
    assert pred_shape.shape == (2, 10)
    assert pred_cam.shape == (2, 3)
    assert token_out.shape == (2, 32)
    assert cam_aux["cam_head_type"] == "patch_uv_rho_multibin"
    assert cam_aux["pred_uv_patch"].shape == (2, 2)
    assert cam_aux["pred_uv_img"].shape == (2, 2)
    assert cam_aux["log_hm_uv_patch"].shape == (2, 16, 16)
    assert cam_aux["pred_rho"].shape == (2, 1)
    assert cam_aux["rho_cls_logits"].shape == (2, 8)
    assert cam_aux["rho_residuals"].shape == (2, 8)
    assert cam_aux["log_rho_prior"].shape == (2, 1)
    assert cam_aux["rho_geom_feat"].shape == (2, RHO_GEOM_DIM)


def _make_patch_uv_rho_cfg(config_path: str):
    cfg = OmegaConf.load(config_path)
    cfg.MODEL.backbone.backbone_str = "model/facebook/dinov2-base"
    cfg.MODEL.handec.context_dim = 768
    cfg.MODEL.handec.cam_head_type = "patch_uv_rho_multibin"
    cfg.MODEL.handec.root_z.dropout = 0.0
    cfg.MODEL.handec.root_z.use_data_source_embed = False
    cfg.MODEL.handec.root_z.prior_k = 121.0
    cfg.MODEL.handec.root_z.d_min = -0.71
    cfg.MODEL.handec.root_z.d_max = 0.75
    cfg.MODEL.norm_by_hand = False
    cfg.LOSS.supervise_heatmap = True
    cfg.LOSS.lambda_uv_patch = 1.0
    return cfg


def _compute_joint_patch_resized(joint_img, patch_bbox, patch_size):
    patch_h, patch_w = patch_size
    patch_width = torch.clamp(patch_bbox[..., 2] - patch_bbox[..., 0], min=1e-6)
    patch_height = torch.clamp(patch_bbox[..., 3] - patch_bbox[..., 1], min=1e-6)
    joint_patch = torch.empty_like(joint_img)
    joint_patch[..., 0] = (
        (joint_img[..., 0] - patch_bbox[..., 0, None]) * patch_w / patch_width[..., None]
    )
    joint_patch[..., 1] = (
        (joint_img[..., 1] - patch_bbox[..., 1, None]) * patch_h / patch_height[..., None]
    )
    return joint_patch


def _make_synthetic_training_batch(cfg, device: torch.device, batch_size: int):
    T = int(cfg.MODEL.num_frame)
    H = int(cfg.MODEL.img_size)
    W = int(cfg.MODEL.img_size)

    patches = torch.randn(batch_size, T, 3, H, W, device=device)

    patch_bbox = torch.tensor([40.0, 40.0, 184.0, 184.0], device=device).view(1, 1, 4).repeat(batch_size, T, 1)
    hand_bbox = torch.tensor([70.0, 78.0, 154.0, 166.0], device=device).view(1, 1, 4).repeat(batch_size, T, 1)
    focal = torch.tensor([1000.0, 1000.0], device=device).view(1, 1, 2).repeat(batch_size, T, 1)
    princpt = torch.tensor([112.0, 112.0], device=device).view(1, 1, 2).repeat(batch_size, T, 1)

    root = torch.tensor([0.0, 0.0, 1000.0], device=device).view(1, 1, 1, 3).repeat(batch_size, T, 1, 1)
    offsets = torch.zeros(batch_size, T, 21, 3, device=device)
    offsets[..., 1:, 0] = torch.linspace(-40.0, 40.0, steps=20, device=device).view(1, 1, 20)
    offsets[..., 1:, 1] = torch.linspace(-30.0, 30.0, steps=20, device=device).view(1, 1, 20)
    offsets[..., 1:, 2] = torch.linspace(5.0, 40.0, steps=20, device=device).view(1, 1, 20)
    joint_cam = root + offsets
    joint_rel = joint_cam - joint_cam[:, :, :1]

    joint_img = torch.zeros(batch_size, T, 21, 2, device=device)
    fx = focal[..., 0].unsqueeze(-1)
    fy = focal[..., 1].unsqueeze(-1)
    px = princpt[..., 0].unsqueeze(-1)
    py = princpt[..., 1].unsqueeze(-1)
    joint_img[..., 0] = joint_cam[..., 0] * fx / joint_cam[..., 2] + px
    joint_img[..., 1] = joint_cam[..., 1] * fy / joint_cam[..., 2] + py
    joint_patch_resized = _compute_joint_patch_resized(
        joint_img,
        patch_bbox,
        patch_size=(H, W),
    )

    batch = {
        "patches": patches,
        "patch_bbox": patch_bbox,
        "hand_bbox": hand_bbox,
        "focal": focal,
        "princpt": princpt,
        "timestamp": torch.arange(T, device=device, dtype=torch.float32).view(1, T).repeat(batch_size, 1),
        "imgs_path": [[f"synthetic_{b:02d}_{t:02d}.png" for t in range(T)] for b in range(batch_size)],
        "joint_cam": joint_cam,
        "joint_rel": joint_rel,
        "joint_img": joint_img,
        "joint_patch_resized": joint_patch_resized,
        "joint_2d_valid": torch.ones(batch_size, T, 21, device=device),
        "joint_3d_valid": torch.ones(batch_size, T, 21, device=device),
        "joint_valid": torch.ones(batch_size, T, 21, device=device),
        "mano_pose": torch.zeros(batch_size, T, 48, device=device),
        "mano_shape": torch.zeros(batch_size, T, 10, device=device),
        "has_mano": torch.ones(batch_size, T, device=device),
        "mano_valid": torch.ones(batch_size, T, device=device),
        "has_intr": torch.ones(batch_size, T, device=device),
        "data_source": ["InterHand2.6M"] * batch_size,
    }
    return batch


def _run_one_step_with_config(config_path: str):
    try:
        PartialState()
    except RuntimeError:
        pass

    cfg = _make_patch_uv_rho_cfg(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = setup_model(cfg).to(device)
    net.train()
    batch = _make_synthetic_training_batch(cfg, device=device, batch_size=1)

    optim = torch.optim.AdamW(net.parameters(), lr=1e-5)
    optim.zero_grad()
    output = net(batch)
    loss = output["loss"]
    assert torch.isfinite(loss).item()
    loss.backward()

    uv_head_grad = net.handec.deccam_uv.decuv.weight.grad
    rho_cls_grad = net.handec.decrho.cls_head.weight.grad
    rho_res_grad = net.handec.decrho.res_head.weight.grad
    assert uv_head_grad is not None
    assert rho_cls_grad is not None
    assert rho_res_grad is not None
    assert torch.isfinite(uv_head_grad).all()
    assert torch.isfinite(rho_cls_grad).all()
    assert torch.isfinite(rho_res_grad).all()

    state = output["state"]
    for key in ["loss_uv_patch", "loss_rho_cls", "loss_rho_res", "rho_bin_acc", "rho_mae_mm"]:
        assert key in state

    optim.step()


def test_posenet_one_step_stage1_yaml_patch_uv_rho_multibin():
    _run_one_step_with_config("config/stage1-dino_large_no_norm.yaml")
