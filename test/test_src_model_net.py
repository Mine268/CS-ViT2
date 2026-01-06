import torch

from src.model.net import *


def test_PoseNet1():
    net = PoseNet(
        "stage1",
        #
        backbone_str="model/facebook/dinov2-base",
        img_size=224,
        img_mean=[0., 0., 0.],
        img_std=[1., 1., 1.],
        infusion_feats_lyr=[2, 6, 10],
        drop_cls=False,
        backbone_kwargs=None,
        #
        num_hf_layer=6,
        num_hf_head=12,
        ndim_hf_mlp=1024,
        ndim_hf_head=64,
        prob_hf_dropout=0.0,
        prob_hf_emb_dropout=0.0,
        hf_emb_dropout_type="drop",
        hf_norm="layer",
        ndim_hf_norm_cond_dim=-1,
        ndim_hf_ctx=768,
        hf_skip_token_embed=False,
        #
        num_pie_sample=16,
        pie_fusion="all",
        #
        num_temporal_head=12,
        num_temporal_layer=2,
        trope_scalar=20.,
        zero_linear=True,
        #
        detok_joint_type="3",
        #
        kps3d_loss_type="l2",
        verts_loss_type="l2",
    ).to("cuda:3")

    img = torch.randn(6, 1, 3, 224, 224).to("cuda:3")
    bbox = torch.randn(6, 1, 4).to("cuda:3")
    focal = torch.randn(6, 1, 2).to("cuda:3")
    princpt = torch.randn(6, 1, 2).to("cuda:3")

    pose, shape, trans = net.predict_mano_param(img, bbox, focal, princpt)

    assert tuple(pose.shape) == (6, 1, 48)
    assert tuple(shape.shape) == (6, 1, 10)
    assert tuple(trans.shape) == (6, 1, 3)

    joint_cam, verts_cam = net.mano_to_pose(pose, shape, trans)

    assert tuple(joint_cam.shape) == (6, 1, 21, 3)
    assert tuple(verts_cam.shape) == (6, 1, 778, 3)


def test_PoseNet2():
    net = PoseNet(
        "stage2",
        #
        backbone_str="model/facebook/dinov2-base",
        img_size=224,
        img_mean=[0., 0., 0.],
        img_std=[1., 1., 1.],
        infusion_feats_lyr=[2, 6, 10],
        drop_cls=False,
        backbone_kwargs=None,
        #
        num_hf_layer=6,
        num_hf_head=12,
        ndim_hf_mlp=1024,
        ndim_hf_head=64,
        prob_hf_dropout=0.0,
        prob_hf_emb_dropout=0.0,
        hf_emb_dropout_type="drop",
        hf_norm="layer",
        ndim_hf_norm_cond_dim=-1,
        ndim_hf_ctx=768,
        hf_skip_token_embed=False,
        #
        num_pie_sample=16,
        pie_fusion="all",
        #
        num_temporal_head=12,
        num_temporal_layer=2,
        trope_scalar=20.,
        zero_linear=True,
        #
        detok_joint_type="3",
        #
        kps3d_loss_type="l2",
        verts_loss_type="l2",
    ).to("cuda:3")

    img = torch.randn(6, 7, 3, 224, 224).to("cuda:3")
    bbox = torch.randn(6, 7, 4).to("cuda:3")
    focal = torch.randn(6, 7, 2).to("cuda:3")
    princpt = torch.randn(6, 7, 2).to("cuda:3")
    timestamp = torch.randn(6, 7).to("cuda:3")

    pose, shape, trans = net.predict_mano_param(img, bbox, focal, princpt, timestamp)

    assert tuple(pose.shape) == (6, 7, 48)
    assert tuple(shape.shape) == (6, 7, 10)
    assert tuple(trans.shape) == (6, 7, 3)

    joint_cam, verts_cam = net.mano_to_pose(pose, shape, trans)

    assert tuple(joint_cam.shape) == (6, 7, 21, 3)
    assert tuple(verts_cam.shape) == (6, 7, 778, 3)


if __name__ == "__main__":
    test_PoseNet1()
    test_PoseNet2()
