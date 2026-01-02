import torch

from src.model.net import *


def test_PoseNet():
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
        detok_joint_type="3"
    ).to("cuda:3")

    img = torch.randn(6, 3, 224, 224).to("cuda:3")

    pose, shape, trans = net.predict_pose(img)

    assert tuple(pose.shape) == (6, 48)
    assert tuple(shape.shape) == (6, 10)
    assert tuple(trans.shape) == (6, 3)


if __name__ == "__main__":
    test_PoseNet()