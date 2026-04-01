import os
import sys
from pathlib import Path

from accelerate import PartialState
from accelerate import Accelerator
from omegaconf import OmegaConf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from script.train import setup_model, setup_optim, setup_scheduler
from src.model.module import MANOTransformerDecoderHead
from src.model.loss import BundleLoss2
from src.model.root_z import (
    ROOT_Z_GEOM_DIM,
    compute_root_z_prior_and_geom,
    decode_delta_log_z_predictions,
    encode_delta_log_z_targets,
)
from src.utils.vis import vis


def test_compute_root_z_prior_and_geom_shapes_and_positive_prior():
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

    z_prior, log_z_prior, geom = compute_root_z_prior_and_geom(
        hand_bbox=hand_bbox,
        focal=focal,
        princpt=princpt,
        prior_k=121.0,
    )

    assert z_prior.shape == (2,)
    assert log_z_prior.shape == (2,)
    assert geom.shape == (2, ROOT_Z_GEOM_DIM)
    assert torch.all(z_prior > 0.0)
    assert torch.allclose(geom[:, 0], log_z_prior)


def test_encode_decode_delta_log_z_roundtrip():
    root_z = torch.tensor([1000.0, 1600.0, 700.0], dtype=torch.float32)
    log_z_prior = torch.log(torch.tensor([900.0, 1000.0, 800.0], dtype=torch.float32))
    d_min = -0.73
    d_max = 0.74
    num_bins = 8

    encoded = encode_delta_log_z_targets(
        root_z=root_z,
        log_z_prior=log_z_prior,
        d_min=d_min,
        d_max=d_max,
        num_bins=num_bins,
    )

    logits = torch.full((3, num_bins), -20.0, dtype=torch.float32)
    logits.scatter_(1, encoded["bin_idx"].unsqueeze(-1), 20.0)
    residuals = torch.zeros((3, num_bins), dtype=torch.float32)
    residuals.scatter_(1, encoded["bin_idx"].unsqueeze(-1), encoded["residual"].unsqueeze(-1))

    decoded = decode_delta_log_z_predictions(
        z_cls_logits=logits,
        z_residuals=residuals,
        log_z_prior=log_z_prior,
        d_min=d_min,
        d_max=d_max,
    )

    assert torch.allclose(
        decoded["pred_delta_log_z"],
        encoded["delta_log_z_clamped"],
        atol=1e-5,
    )
    assert torch.allclose(
        decoded["pred_z"],
        torch.exp(log_z_prior + encoded["delta_log_z_clamped"]),
        atol=1e-4,
    )


def test_mano_transformer_decoder_head_xy_rootz_multibin_outputs_expected_shapes():
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
        cam_head_type="xy_rootz_multibin",
        root_z_num_bins=8,
        root_z_d_min=-0.73,
        root_z_d_max=0.74,
        root_z_prior_k=121.0,
        root_z_geom_hidden_dim=16,
        root_z_dropout=0.0,
        root_z_use_data_source_embed=False,
    )

    x = torch.randn(2, 10, 32)
    hand_bbox = torch.tensor(
        [
            [100.0, 100.0, 180.0, 220.0],
            [80.0, 90.0, 130.0, 170.0],
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
            [128.0, 128.0],
            [128.0, 128.0],
        ],
        dtype=torch.float32,
    )

    (pred_pose, pred_shape, pred_cam), cam_aux, token_out = head(
        x,
        hand_bbox=hand_bbox,
        focal=focal,
        princpt=princpt,
    )

    assert pred_pose.shape == (2, 48)
    assert pred_shape.shape == (2, 10)
    assert pred_cam.shape == (2, 3)
    assert token_out.shape == (2, 32)
    assert cam_aux["cam_head_type"] == "xy_rootz_multibin"
    assert cam_aux["z_cls_logits"].shape == (2, 8)
    assert cam_aux["z_residuals"].shape == (2, 8)
    assert cam_aux["log_z_prior"].shape == (2, 1)
    assert cam_aux["root_z_geom_feat"].shape == (2, ROOT_Z_GEOM_DIM)


def test_mano_transformer_decoder_head_softargmax3d_compatibility():
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
        cam_head_type="softargmax3d",
    )

    x = torch.randn(2, 10, 32)
    (pred_pose, pred_shape, pred_cam), cam_aux, token_out = head(x)

    assert pred_pose.shape == (2, 48)
    assert pred_shape.shape == (2, 10)
    assert pred_cam.shape == (2, 3)
    assert token_out.shape == (2, 32)
    assert cam_aux["cam_head_type"] == "softargmax3d"
    assert cam_aux["log_hm_x"].shape == (2, 16)
    assert cam_aux["log_hm_y"].shape == (2, 16)
    assert cam_aux["log_hm_z"].shape == (2, 32)


def _make_xy_rootz_cfg(config_path: str):
    cfg = OmegaConf.load(config_path)
    # 使用现有 yaml 作为基线，只把 backbone 缩到本地已有的 base 版以控制 smoke 时长。
    cfg.MODEL.backbone.backbone_str = "model/facebook/dinov2-base"
    cfg.MODEL.handec.context_dim = 768
    cfg.MODEL.handec.cam_head_type = "xy_rootz_multibin"
    cfg.MODEL.handec.root_z.dropout = 0.0
    cfg.MODEL.handec.root_z.use_data_source_embed = False
    cfg.MODEL.norm_by_hand = False
    cfg.LOSS.supervise_heatmap = True
    return cfg


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
        "joint_patch_resized": torch.zeros(batch_size, T, 21, 2, device=device),
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

    cfg = _make_xy_rootz_cfg(config_path)
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

    cls_weight_grad = net.handec.decz.cls_head.weight.grad
    res_weight_grad = net.handec.decz.res_head.weight.grad
    assert cls_weight_grad is not None
    assert res_weight_grad is not None
    assert torch.isfinite(cls_weight_grad).all()
    assert torch.isfinite(res_weight_grad).all()

    state = output["state"]
    assert "loss_root_z_cls" in state
    assert "loss_root_z_res" in state
    assert "root_z_bin_acc" in state
    assert "root_z_mae_mm" in state

    optim.step()


def test_posenet_one_step_stage1_yaml_xy_rootz_multibin():
    _run_one_step_with_config("config/stage1-dino_large_no_norm.yaml")


def test_posenet_one_step_stage2_yaml_xy_rootz_multibin():
    _run_one_step_with_config("config/stage2-dino_large_no_norm.yaml")


def _run_full_train_flow_smoke(config_path: str, tmp_path):
    try:
        PartialState()
    except RuntimeError:
        pass

    cfg = _make_xy_rootz_cfg(config_path)
    cfg.TRAIN.mixed_precision = "no"
    cfg.GENERAL.total_step = 1
    cfg.GENERAL.warmup_step = 0
    cfg.GENERAL.cosine_cycle = 0.5

    accelerator = Accelerator(mixed_precision=cfg.TRAIN.mixed_precision)
    device = accelerator.device

    net = setup_model(cfg)
    optim = setup_optim(cfg, net)
    scheduler = setup_scheduler(cfg, optim)
    net, optim, scheduler = accelerator.prepare(net, optim, scheduler)
    net.train()

    batch = _make_synthetic_training_batch(cfg, device=device, batch_size=1)
    trans_2d_mat = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(1, int(cfg.MODEL.num_frame), 1, 1)

    output = net(batch)
    loss = output["loss"]
    assert torch.isfinite(loss).item()

    accelerator.backward(loss)
    accelerator.clip_grad_norm_(net.parameters(), cfg.TRAIN.max_grad)
    optim.step()
    scheduler.step()
    optim.zero_grad()

    state = output["state"]
    # 模拟 train.py 的日志读取路径，确保关键状态项都可用且有限。
    log_keys = [
        "loss_theta",
        "loss_shape",
        "loss_trans",
        "loss_trans_xy",
        "loss_root_z_cls",
        "loss_root_z_res",
        "loss_joint_rel",
        "loss_joint_img",
        "root_z_bin_acc",
        "root_z_mae_mm",
        "micro_mpjpe",
        "micro_mpjpe_rel",
    ]
    for key in log_keys:
        assert key in state, f"Missing log key: {key}"
        assert torch.isfinite(state[key]).item(), f"Non-finite log value for {key}"

    vis_img = vis(batch, trans_2d_mat, output["result"], tx=0, bx=0)
    assert vis_img.ndim == 3
    assert vis_img.shape[0] > 0 and vis_img.shape[1] > 0

    save_dir = tmp_path / f"save_{Path(config_path).stem}"
    accelerator.save_state(str(save_dir))
    assert any(save_dir.iterdir()), f"No checkpoint artifacts found in {save_dir}"


def test_full_train_flow_stage1_yaml_xy_rootz_multibin(tmp_path):
    _run_full_train_flow_smoke("config/stage1-dino_large_no_norm.yaml", tmp_path)


def test_full_train_flow_stage2_yaml_xy_rootz_multibin(tmp_path):
    _run_full_train_flow_smoke("config/stage2-dino_large_no_norm.yaml", tmp_path)


def test_root_z_frame_filter_mask_respects_bbox_and_valid_thresholds():
    loss_fn = BundleLoss2(
        lambda_theta=1.0,
        lambda_shape=1.0,
        lambda_trans=1.0,
        lambda_rel=1.0,
        lambda_img=1.0,
        lambda_coco_patch_2d=0.0,
        lambda_root_z_cls=1.0,
        lambda_root_z_res=1.0,
        supervise_global=True,
        supervise_heatmap=True,
        norm_by_hand=False,
        norm_idx=[],
        hm_centers=(
            torch.linspace(-1.0, 1.0, 8),
            torch.linspace(-1.0, 1.0, 8),
            None,
        ),
        hm_sigma=1.0,
        cam_head_type="xy_rootz_multibin",
        root_z_num_bins=8,
        root_z_d_min=-0.73,
        root_z_d_max=0.74,
        root_z_min_valid_joints_2d=16,
        root_z_min_hand_bbox_edge_px=8.0,
        reproj_loss_type="l1",
        reproj_loss_delta=84.0,
    )

    batch = {
        "hand_bbox": torch.tensor(
            [
                [[10.0, 20.0, 30.0, 45.0]],   # keep
                [[10.0, 20.0, 15.0, 26.0]],   # reject by bbox
                [[10.0, 20.0, 30.0, 45.0]],   # reject by valid count
            ],
            dtype=torch.float32,
        ),
        "joint_2d_valid": torch.tensor(
            [
                [[1.0] * 21],
                [[1.0] * 21],
                [[1.0] * 8 + [0.0] * 13],
            ],
            dtype=torch.float32,
        ),
        "joint_valid": torch.tensor(
            [
                [[1.0] * 21],
                [[1.0] * 21],
                [[1.0] * 8 + [0.0] * 13],
            ],
            dtype=torch.float32,
        ),
        "has_intr": torch.ones(3, 1, dtype=torch.float32),
    }

    frame_mask = loss_fn.compute_root_z_frame_filter_mask(batch)
    expected = torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float32)
    assert torch.equal(frame_mask, expected)
