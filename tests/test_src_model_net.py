import torch
from accelerate import PartialState

PartialState()

from src.model.net import *


def _make_net(stage):
    return PoseNet(
        stage=stage,
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
        handec_mean_init=False,
        handec_denorm_output=False,
        handec_heatmap_resulotion=(64, 64, 64),
        #
        pie_type="dense",
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
        lambda_img=0.0,
        hm_sigma=2.5,
        reproj_loss_type="l1",
        reproj_loss_delta=10.0,
        #
        freeze_backbone=False,
        norm_by_hand=True,
    ).to("cuda:0")


def test_PoseNet1():
    """Stage 1: single frame, output [b,1,d]"""
    net = _make_net("stage1")

    img = torch.randn(4, 1, 3, 224, 224).to("cuda:0")
    bbox = torch.randn(4, 1, 4).to("cuda:0")
    focal = torch.randn(4, 1, 2).to("cuda:0")
    princpt = torch.randn(4, 1, 2).to("cuda:0")

    pose, shape, trans, _ = net.predict_mano_param(img, bbox, focal, princpt)

    assert tuple(pose.shape) == (4, 1, 48)
    assert tuple(shape.shape) == (4, 1, 10)
    assert tuple(trans.shape) == (4, 1, 3)

    joint_rel, verts_rel = net.mano_to_pose(pose, shape)

    assert tuple(joint_rel.shape) == (4, 1, 21, 3)
    assert tuple(verts_rel.shape) == (4, 1, 778, 3)


def test_PoseNet2():
    """Stage 2: temporal, output [b,t,d] for all frames"""
    net = _make_net("stage2")

    T = 7
    img = torch.randn(4, T, 3, 224, 224).to("cuda:0")
    bbox = torch.randn(4, T, 4).to("cuda:0")
    focal = torch.randn(4, T, 2).to("cuda:0")
    princpt = torch.randn(4, T, 2).to("cuda:0")
    timestamp = torch.arange(T).float().unsqueeze(0).expand(4, -1).to("cuda:0")

    pose, shape, trans, _ = net.predict_mano_param(img, bbox, focal, princpt, timestamp)

    # full-frame supervision: all T frames output
    assert tuple(pose.shape) == (4, T, 48), f"Expected (4,{T},48), got {pose.shape}"
    assert tuple(shape.shape) == (4, T, 10)
    assert tuple(trans.shape) == (4, T, 3)

    joint_rel, verts_rel = net.mano_to_pose(pose, shape)

    assert tuple(joint_rel.shape) == (4, T, 21, 3)
    assert tuple(verts_rel.shape) == (4, T, 778, 3)


if __name__ == "__main__":
    test_PoseNet1()
    print("test_PoseNet1 passed")
    test_PoseNet2()
    print("test_PoseNet2 passed")
