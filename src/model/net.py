import enum
import itertools

import numpy as np
import torch.nn as nn
import smplx

from .loss import *
from .module import *
from .hamer_module import *


class PoseNet(nn.Module):
    class Stage(enum.Enum):
        STAGE1 = "stage1"
        STAGE2 = "stage2"

    def __init__(
        self,
        stage: Stage,
        # DinoBackbone
        backbone_str: str,
        img_size: Optional[int],
        img_mean: List[float],
        img_std: List[float],
        infusion_feats_lyr: List[int],
        drop_cls: bool,
        backbone_kwargs: Optional[Dict],
        # hand feat extractor
        num_hf_layer: int,
        num_hf_head: int,
        ndim_hf_mlp: int,
        ndim_hf_head: int,
        prob_hf_dropout: float,
        prob_hf_emb_dropout: float,
        hf_emb_dropout_type: str,
        hf_norm: str,
        ndim_hf_norm_cond_dim: int,
        ndim_hf_ctx: Optional[int],
        hf_skip_token_embed: bool,
        # PerspInfoEmbedderDense
        pie_type: str,
        num_pie_sample: int,
        pie_fusion: str,
        # TemporalEncoder
        num_temporal_head: int,
        num_temporal_layer: int,
        trope_scalar: float,
        zero_linear: bool,
        # MANOPoseDetokenizer
        detok_joint_type: str,
        # loss
        kps3d_loss_type: str,
        verts_loss_type: str,
    ):
        super(PoseNet, self).__init__()
        self.stage = stage
        # Image encoder
        backbone_kwargs = default(backbone_kwargs, {})
        self.backbone = ViTBackbone(
            backbone_str=backbone_str,
            img_size=img_size,
            infusion_feats_lyr=infusion_feats_lyr,
            backbone_kwargs=dict(backbone_kwargs),
        )
        self.register_buffer("img_mean", torch.Tensor(img_mean))
        self.register_buffer("img_std", torch.Tensor(img_std))
        self.drop_cls = drop_cls
        self.patch_size = self.backbone.get_patch_size()
        self.hidden_size = self.backbone.get_hidden_size()
        self.img_size = self.backbone.get_img_size()
        self.num_patch = self.backbone.get_num_patch()

        # Perspective Information Embedder
        if pie_type == "dense":
            self.persp_info_embedder = PerspInfoEmbedderDense(
                hidden_size=self.hidden_size,
                num_sample=num_pie_sample,
                pie_fusion=pie_fusion
            )
        elif pie_type == "ca":
            self.persp_info_embedder = PerspInfoEmbedderCrossAttn(
                hidden_size=self.hidden_size,
                num_sample=num_pie_sample,
                num_token=self.num_patch**2 + int(not self.drop_cls),
            )
        else:
            raise NotImplementedError(f"pie_type={pie_type} not implemented.")

        # MANO
        self.register_buffer(
            "J_regressor_mano",
            torch.from_numpy(np.load(MANO_J_REGRESSOR_PATH)).type(torch.float32)
        )
        self.rmano_layer = smplx.create(MANO_ROOT, "mano", is_rhand=True, use_pca=False)
        self.rmano_layer.requires_grad_(False)
        self.rmano_layer.eval()

        # Spatial encoder
        self.query_tokens = nn.Parameter(
            data=torch.randn(1, 3, self.hidden_size),
            requires_grad=True,
        )
        self.hand_feat_encoder = TransformerDecoder(
            num_tokens=3,
            token_dim=self.hidden_size,
            dim=self.hidden_size,
            depth=num_hf_layer,
            heads=num_hf_head,
            mlp_dim=ndim_hf_mlp,
            dim_head=ndim_hf_head,
            dropout=prob_hf_dropout,
            emb_dropout=prob_hf_emb_dropout,
            emb_dropout_type=hf_emb_dropout_type,
            norm=hf_norm,
            norm_cond_dim=ndim_hf_norm_cond_dim,
            context_dim=ndim_hf_ctx,
            skip_token_embedding=hf_skip_token_embed,
        )

        # Temporal encoder
        self.pose_temporal_encoder = TemporalEncoder(
            dim=self.hidden_size,
            num_head=num_temporal_head,
            num_layer=num_temporal_layer,
            dropout=prob_hf_dropout,
            trope_scalar=trope_scalar,
            zero_linear=zero_linear,
        )
        self.shape_temporal_encoder = TemporalEncoder(
            dim=self.hidden_size,
            num_head=num_temporal_head,
            num_layer=num_temporal_layer,
            dropout=prob_hf_dropout,
            trope_scalar=trope_scalar,
            zero_linear=zero_linear,
        )
        self.trans_temporal_encoder = TemporalEncoder(
            dim=self.hidden_size,
            num_head=num_temporal_head,
            num_layer=num_temporal_layer,
            dropout=prob_hf_dropout,
            trope_scalar=trope_scalar,
            zero_linear=zero_linear,
        )

        # MANO detokenizer
        self.mano_detokenizer = MANOPoseDetokenizer(
            dim=self.hidden_size,
            joint_rep_type=detok_joint_type
        )

        # Loss
        self.kps3d_loss = Keypoint3DLoss(kps3d_loss_type)
        self.verts_loss = VertsLoss(verts_loss_type)
        self.axis_loss = Axis3DLoss("l1")
        self.shape_loss = ShapeLoss("l1")

    def extract_hand_feature(
        self,
        img: torch.Tensor,
        bbox: torch.Tensor,
        focal: torch.Tensor,
        princpt: torch.Tensor,
    ):
        # extract vision feature
        feats = self.backbone(img) # [b,l,d]
        if self.drop_cls:
            feats = feats[:, 1:]

        # extract perspective feature
        feats = self.persp_info_embedder(
            feats=feats,
            bbox=bbox,
            focal=focal,
            princpt=princpt,
        )

        # extract hand feature
        query_tokens = self.query_tokens.expand(img.shape[0], -1, -1)
        hand_feats = self.hand_feat_encoder(query_tokens, context=feats) # [b,3,d] / [(bt),3,d]

        # split into 3 seperate feature
        # [b,d] / [(bt),d]
        pose_token, shape_token, trans_token = torch.split(
            hand_feats, split_size_or_sections=1, dim=-2
        )
        pose_token = pose_token.squeeze(dim=-2)
        shape_token = shape_token.squeeze(dim=-2)
        trans_token = trans_token.squeeze(dim=-2)

        return pose_token, shape_token, trans_token

    def predict_mano_param(
        self,
        img: torch.Tensor,
        bbox: torch.Tensor,
        focal: torch.Tensor,
        princpt: torch.Tensor,
        timestamp: Optional[torch.Tensor] = None
    ):
        """
        timestamp: None or [b,t]
        """
        assert (len(img.shape) == 5
            and len(bbox.shape) == 3
            and len(focal.shape) == 3
            and len(princpt.shape) == 3
        )

        num_frame = img.shape[1]

        img = (
            (img - self.img_mean[None, None, :, None, None]) /
            self.img_std[None, None, :, None, None]
        )

        img, bbox, focal, princpt = map(
            lambda t: eps.rearrange(t, "b t ... -> (b t) ..."),
            [img, bbox, focal, princpt]
        )

        # spatial encoding
        pose_token, shape_token, trans_token = self.extract_hand_feature(
            img=img,
            bbox=bbox,
            focal=focal,
            princpt=princpt,
        )

        # temporal decoding
        pose_token, shape_token, trans_token = map(
            lambda t: eps.rearrange(t, "(b t) d -> b t d", t=num_frame),
            [pose_token, shape_token, trans_token]
        )
        if self.stage == PoseNet.Stage.STAGE2:
            pose_token = pose_token + self.pose_temporal_encoder(pose_token, timestamp)
            shape_token = shape_token + self.shape_temporal_encoder(shape_token, timestamp)
            trans_token = trans_token + self.trans_temporal_encoder(trans_token, timestamp)

        # detokenize into pose
        # [b,d] [b,t,d]
        pose, shape, trans = self.mano_detokenizer(pose_token, shape_token, trans_token)

        return pose, shape, trans

    def mano_to_pose(self, pose, shape, trans):
        batch_size, _, _ = pose.shape
        njoint_hand = self.J_regressor_mano.shape[0]

        shape = eps.rearrange(shape, "b t d -> (b t) d")
        pose = eps.rearrange(pose, "b t d -> (b t) d")

        mano_output = self.rmano_layer(
            betas=shape,
            global_orient=pose[:, :3],
            hand_pose=pose[:, 3:],
            transl=torch.zeros(size=(pose.shape[0], 3), device=pose.device)
        )

        joints = torch.einsum(
            "nvd,jv->njd",
            mano_output.vertices, self.J_regressor_mano
        )

        # [B,T,V,3]
        verts_cam = rearrange(
            (mano_output.vertices - joints[:, :1]) * 1e3, # to mm
            "(b t) v d -> b t v d", b=batch_size
        ) + trans[:, :, None]
        # [B,T,J,3]
        joint_cam = rearrange(
            (joints - joints[:, :1]) * 1e3,
            "(b t) j d -> b t j d", b=batch_size, j=njoint_hand
        ) + trans[:, :, None]

        return joint_cam, verts_cam

    def forward(self, batch):
        # 1. forward
        pose_pred, shape_pred, trans_pred = self.predict_mano_param(
            img=batch["patches"],
            bbox=batch["patch_bbox"],
            focal=batch["focal"],
            princpt=batch["princpt"],
            timestamp=batch["timestamp"]
        )

        # 2. fk
        with torch.no_grad():
            _, verts_cam_gt = self.mano_to_pose(
                batch["mano_pose"], batch["mano_shape"], batch["joint_cam"][:, :, 0]
            )
        joint_cam_pred, verts_cam_pred = self.mano_to_pose(pose_pred, shape_pred, trans_pred)

        # 3. loss: 3d joint, 3d verts, 3d axis, shape
        loss_kps3d = self.kps3d_loss(joint_cam_pred, batch["joint_cam"], batch["joint_valid"])
        loss_verts = self.verts_loss(verts_cam_pred, verts_cam_gt, batch["mano_valid"])
        loss_axis = self.axis_loss(pose_pred, batch["mano_pose"], batch["mano_valid"])
        loss_shape = self.shape_loss(shape_pred, batch["mano_shape"], batch["mano_valid"])

        loss_state = {
            "loss": loss_kps3d + loss_verts + loss_axis + loss_shape,
            "state": {
                "loss_kps3d": loss_kps3d.detach(),
                "loss_verts": loss_verts.detach(),
                "loss_axis": loss_axis.detach(),
                "loss_shape": loss_shape.detach(),
            },
            "result": {
                "joint_cam_pred": joint_cam_pred.detach(),
                "verts_cam_pred": verts_cam_pred.detach(),
                "verts_cam_gt": verts_cam_gt.detach()
            }
        }

        return loss_state

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_regressor_params(self):
        return itertools.chain(
            self.persp_info_embedder.parameters(),
            self.hand_feat_encoder.parameters(),
            self.pose_temporal_encoder.parameters(),
            self.shape_temporal_encoder.parameters(),
            self.trans_temporal_encoder.parameters(),
            self.mano_detokenizer.parameters(),
            [self.query_tokens],
        )