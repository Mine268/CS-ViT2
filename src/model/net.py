import torch.nn as nn

from .module import *
from .hamer_module import *


class PoseNet(nn.Module):
    def __init__(
        self,
        # DinoBackbone
        backbone_str: str,
        img_size: Optional[int],
        infusion_feats_lyr: List[int],
        drop_cls: bool,
        # TransformerDecoder
        num_spatial_layer: int,
        num_spatial_head: int,
        ndim_spatial_mlp: int,
        ndim_spatial_head: int,
        prob_spatial_dropout: float,
        prob_spatial_emb_dropout: float,
        spatial_emb_dropout_type: str,
        spatial_norm: str,
        ndim_spatial_norm_cond_dim: int,
        ndim_spatial_ctx: Optional[int],
        spatial_skip_token_embed: bool,
        # MANOPoseDetokenizer
        detok_joint_type: str,
    ):
        super(PoseNet, self).__init__()
        # Image encoder
        self.backbone = DinoBackbone(
            backbone_str=backbone_str,
            img_size=img_size,
            infusion_feats_lyr=infusion_feats_lyr,
        )
        self.drop_cls = drop_cls
        self.patch_size = self.backbone.get_patch_size()
        self.hidden_size = self.backbone.get_hidden_size()
        self.img_size = self.backbone.get_img_size()
        self.num_patch = self.backbone.get_num_patch()

        # TODO: add PIE

        # Spatial encoder
        self.query_tokens = nn.Parameter(
            data=torch.randn(1, 3, self.hidden_size),
            requires_grad=True,
        )
        self.spatial_encoder = TransformerDecoder(
            num_tokens=3,
            token_dim=self.hidden_size,
            dim=self.hidden_size,
            depth=num_spatial_layer,
            heads=num_spatial_head,
            mlp_dim=ndim_spatial_mlp,
            dim_head=ndim_spatial_head,
            dropout=prob_spatial_dropout,
            emb_dropout=prob_spatial_emb_dropout,
            emb_dropout_type=spatial_emb_dropout_type,
            norm=spatial_norm,
            norm_cond_dim=ndim_spatial_norm_cond_dim,
            context_dim=ndim_spatial_ctx,
            skip_token_embedding=spatial_skip_token_embed,
        )

        # MANO detokenizer
        self.mano_detokenizer = MANOPoseDetokenizer(
            dim=self.hidden_size,
            joint_rep_type=detok_joint_type
        )

    # TODO: add intri
    def predict_pose(self, img: torch.Tensor):
        batch_size = img.shape[0]

        # extract vision feature
        feats = self.backbone(img) # [b,l,d]
        if self.drop_cls:
            feats = feats[:, 1:]

        # extract hand feature
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        hand_feats = self.spatial_encoder(query_tokens, context=feats)

        # detokenize into pose
        # [b,d] each
        pose_token, shape_token, trans_token = torch.split(
            hand_feats, split_size_or_sections=1, dim=1
        )
        pose, shape, trans = self.mano_detokenizer(pose_token, shape_token, trans_token)

        return pose[:, 0], shape[:, 0], trans[:, 0]