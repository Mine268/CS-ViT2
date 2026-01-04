import torch

from src.model.net import *


def test_PoseNet1():
    net = PoseNet(
        backbone_str="model/facebook/dinov2-base",
        img_size=224,
        infusion_feats_lyr=[2, 6, 10],
        drop_cls=False,
        #
        num_spatial_layer=6,
        num_spatial_head=12,
        ndim_spatial_mlp=1024,
        ndim_spatial_head=64,
        prob_spatial_dropout=0.0,
        prob_spatial_emb_dropout=0.0,
        spatial_emb_dropout_type="drop",
        spatial_norm="layer",
        ndim_spatial_norm_cond_dim=-1,
        ndim_spatial_ctx=768,
        spatial_skip_token_embed=False,
        #
        num_pie_sample=16,
        pie_fusion="all",
        #
        num_temporal_head=12,
        num_temporal_layer=2,
        trope_scalar=20.,
        zero_linear=True,
        #
        detok_joint_type="3"
    ).to("cuda:3")

    img = torch.randn(6, 3, 224, 224).to("cuda:3")
    bbox = torch.randn(6, 4).to("cuda:3")
    focal = torch.randn(6, 2).to("cuda:3")
    princpt = torch.randn(6, 2).to("cuda:3")

    pose, shape, trans = net.predict_pose(img, bbox, focal, princpt)

    assert tuple(pose.shape) == (6, 48)
    assert tuple(shape.shape) == (6, 10)
    assert tuple(trans.shape) == (6, 3)


def test_PoseNet2():
    net = PoseNet(
        backbone_str="model/facebook/dinov2-base",
        img_size=224,
        infusion_feats_lyr=[2, 6, 10],
        drop_cls=False,
        #
        num_spatial_layer=6,
        num_spatial_head=12,
        ndim_spatial_mlp=1024,
        ndim_spatial_head=64,
        prob_spatial_dropout=0.0,
        prob_spatial_emb_dropout=0.0,
        spatial_emb_dropout_type="drop",
        spatial_norm="layer",
        ndim_spatial_norm_cond_dim=-1,
        ndim_spatial_ctx=768,
        spatial_skip_token_embed=False,
        #
        num_pie_sample=16,
        pie_fusion="all",
        #
        num_temporal_head=12,
        num_temporal_layer=2,
        trope_scalar=20.,
        zero_linear=True,
        #
        detok_joint_type="3"
    ).to("cuda:3")

    img = torch.randn(6, 7, 3, 224, 224).to("cuda:3")
    bbox = torch.randn(6, 7, 4).to("cuda:3")
    focal = torch.randn(6, 7, 2).to("cuda:3")
    princpt = torch.randn(6, 7, 2).to("cuda:3")
    timestamp = torch.randn(6, 7).to("cuda:3")

    pose, shape, trans = net.predict_pose(img, bbox, focal, princpt, timestamp)

    assert tuple(pose.shape) == (6, 7, 48)
    assert tuple(shape.shape) == (6, 7, 10)
    assert tuple(trans.shape) == (6, 7, 3)


if __name__ == "__main__":
    # test_PoseNet1()
    test_PoseNet2()
