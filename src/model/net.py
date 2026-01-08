import enum
import itertools

import numpy as np
import torch.nn as nn
import kornia
import smplx

from .loss import *
from .module import *
from .hamer_module import *
from ..utils.metric import *


class PoseNet(nn.Module):
    class Stage(enum.Enum):
        STAGE1 = "stage1"
        STAGE2 = "stage2"

    def __init__(
        self,
        stage: Stage,

        backbone_str: str,
        img_size: Optional[int],
        img_mean: List[float],
        img_std: List[float],
        infusion_feats_lyr: List[int],
        drop_cls: bool,
        backbone_kwargs: Optional[Dict],

        num_handec_layer: int,
        num_handec_head: int,
        ndim_handec_mlp: int,
        ndim_handec_head: int,
        prob_handec_dropout: float,
        prob_handec_emb_dropout: float,
        handec_emb_dropout_type: str,
        handec_norm: str,
        ndim_handec_norm_cond_dim: int,
        ndim_handec_ctx: Optional[int],
        handec_skip_token_embed: bool,
        handec_mean_init: bool,
        handec_denorm_output: bool,

        pie_type: str,
        num_pie_sample: int,
        pie_fusion: str,

        num_temporal_head: int,
        num_temporal_layer: int,
        trope_scalar: float,
        zero_linear: bool,

        joint_rep_type: str,

        kps3d_loss_type: str,
        verts_loss_type: str,
        param_loss_type: str,
        supervise_global: bool,
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

        # handec
        self.joint_rep_type = joint_rep_type
        self.handec = MANOTransformerDecoderHead(
            joint_rep_type=joint_rep_type,
            dim=self.hidden_size,
            depth=num_handec_layer,
            heads=num_handec_head,
            mlp_dim=ndim_handec_mlp,
            dim_head=ndim_handec_head,
            dropout=prob_handec_dropout,
            emb_dropout=prob_handec_emb_dropout,
            emb_dropout_type=handec_emb_dropout_type,
            norm=handec_norm,
            norm_cond_dim=ndim_handec_norm_cond_dim,
            context_dim=ndim_handec_ctx,
            skip_token_embedding=handec_skip_token_embed,
            use_mean_init=handec_mean_init,
            denorm_output=handec_denorm_output,
        )

        # Temporal encoder
        # self.pose_temporal_encoder = TemporalEncoder(
        #     dim=self.hidden_size,
        #     num_head=num_temporal_head,
        #     num_layer=num_temporal_layer,
        #     dropout=prob_hf_dropout,
        #     trope_scalar=trope_scalar,
        #     zero_linear=zero_linear,
        # )
        # self.shape_temporal_encoder = TemporalEncoder(
        #     dim=self.hidden_size,
        #     num_head=num_temporal_head,
        #     num_layer=num_temporal_layer,
        #     dropout=prob_hf_dropout,
        #     trope_scalar=trope_scalar,
        #     zero_linear=zero_linear,
        # )
        # self.trans_temporal_encoder = TemporalEncoder(
        #     dim=self.hidden_size,
        #     num_head=num_temporal_head,
        #     num_layer=num_temporal_layer,
        #     dropout=prob_hf_dropout,
        #     trope_scalar=trope_scalar,
        #     zero_linear=zero_linear,
        # )

        # MANO detokenizer
        # self.mano_detokenizer = MANOPoseDetokenizer(
        #     dim=self.hidden_size,
        #     joint_rep_type=detok_joint_type
        # )

        # Loss
        self.supervise_global = supervise_global
        self.loss_fn = BundleLoss(1.0, 1.0, supervise_global)
        # self.kps3d_loss = Keypoint3DLoss(kps3d_loss_type, 1e-3)
        # self.verts_loss = VertsLoss(verts_loss_type, 1e-3)
        # self.param_loss = ParameterLoss(param_loss_type)

    def decode_hand_param(
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

        # extract hand param
        # [b,d], [b,10], [b,3]
        pred_hand_pose, pred_shape, pred_trans = self.handec(feats)

        return pred_hand_pose, pred_shape, pred_trans

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
        pose, shape, trans = self.decode_hand_param(
            img=img,
            bbox=bbox,
            focal=focal,
            princpt=princpt,
        )

        # if self.joint_rep_type == "6d":
        #     pose = eps.rearrange(pose, "b (j d) -> (b j) d", j=MANO_JOINT_COUNT)
        #     pose = rotation6d_to_rotation_matrix(pose) # [b,3,3]
        #     pose = kornia.geometry.conversions.rotation_matrix_to_axis_angle(pose)
        #     pose = eps.rearrange(eps, "(b j) d -> b (j d)", j=MANO_JOINT_COUNT)
        # elif self.joint_rep_type == "3":
        #     pass
        # elif self.joint_rep_type == "quat":
        #     pose = eps.rearrange(pose, "b (j d) -> (b j) d", j=MANO_JOINT_COUNT)
        #     pose = kornia.geometry.conversions.quaternion_to_axis_angle(pose) # [b,4]
        #     pose = kornia.geometry.conversions.rotation_matrix_to_axis_angle(pose)
        #     pose = eps.rearrange(eps, "(b j) d -> b (j d)", j=MANO_JOINT_COUNT)
        # else:
        #     raise NotImplementedError

        # temporal decoding
        pose, shape, trans = map(
            lambda t: eps.rearrange(t, "(b t) d -> b t d", t=num_frame),
            [pose, shape, trans]
        )
        # if self.stage == PoseNet.Stage.STAGE2:
        #     pose_token = pose_token + self.pose_temporal_encoder(pose_token, timestamp)
        #     shape_token = shape_token + self.shape_temporal_encoder(shape_token, timestamp)
        #     trans_token = trans_token + self.trans_temporal_encoder(trans_token, timestamp)

        # detokenize into pose
        # [b,d] [b,t,d]
        # pose, shape, trans = self.mano_detokenizer(pose_token, shape_token, trans_token)

        return pose, shape, trans

    def mano_to_pose(self, pose, shape):
        batch_size, _, _ = pose.shape
        njoint_hand = self.J_regressor_mano.shape[0]

        shape = eps.rearrange(shape, "b t d -> (b t) d")
        pose = eps.rearrange(pose, "b t d -> (b t) d")

        if self.joint_rep_type == "6d":
            pose_aa = eps.rearrange(pose, "b (j d) -> (b j) d", j=MANO_JOINT_COUNT)
            pose_aa = rotation6d_to_rotation_matrix(pose_aa)
            pose_aa = kornia.geometry.conversions.rotation_matrix_to_axis_angle(pose_aa)
            pose_aa = eps.rearrange(pose_aa, "(b j) d -> b (j d)", j=MANO_JOINT_COUNT)
            pose = pose_aa
        elif self.joint_rep_type == "quat":
            pose_aa = eps.rearrange(pose, "b (j d) -> (b j) d", j=MANO_JOINT_COUNT)
            pose_aa = kornia.geometry.conversions.quaternion_to_axis_angle(pose_aa)
            pose_aa = eps.rearrange(pose_aa, "(b j) d -> b (j d)", j=MANO_JOINT_COUNT)
            pose = pose_aa
        elif self.joint_rep_type == "3":
            pass
        else:
            raise NotImplementedError(f"Unsupported rotation type={self.joint_rep_type}")

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
        joint_root_detach = joints[:, :1].detach()

        # [B,T,V,3]
        verts_rel = rearrange(
            (mano_output.vertices - joint_root_detach) * 1e3, # to mm
            "(b t) v d -> b t v d", b=batch_size
        )
        # [B,T,J,3]
        joint_rel = rearrange(
            (joints - joint_root_detach) * 1e3,
            "(b t) j d -> b t j d", b=batch_size, j=njoint_hand
        )

        return joint_rel, verts_rel

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
            _, verts_rel_gt = self.mano_to_pose(
                batch["mano_pose"],
                batch["mano_shape"],
                # batch["joint_cam"][:, :, 0],
            )
        joint_rel_pred, verts_rel_pred = self.mano_to_pose(
            pose_pred, shape_pred # , trans_pred
        )

        # 3. loss
        loss, loss_state = self.loss_fn(
            pose_pred,
            shape_pred,
            trans_pred,
            joint_rel_pred,
            verts_rel_pred,
            verts_rel_gt,
            batch,
        )

        # 4. micro metric
        with torch.no_grad():
            micro_mpjpe = compute_mpjpe_stats(
                joint_rel_pred, batch["joint_cam"], batch["joint_valid"]
            )
            micro_mpjpe = micro_mpjpe[0] / micro_mpjpe[1]

            micro_mpjpe_rel = compute_mpjpe_stats(
                joint_rel_pred - joint_rel_pred[:, :, :1],
                batch["joint_cam"] - batch["joint_cam"][:, :, :1],
                batch["joint_valid"]
            )
            micro_mpjpe_rel = micro_mpjpe_rel[0] / micro_mpjpe_rel[1]

            verts_cam_gt = verts_rel_gt + batch["joint_cam"][:, :, :1]
            micro_mpvpe = compute_mpvpe_stats(verts_rel_pred, verts_cam_gt, batch["mano_valid"])
            micro_mpvpe = micro_mpvpe[0] / micro_mpvpe[1]

        loss_state = {
            "loss": loss,
            "state": loss_state | {
                "micro_mpjpe": micro_mpjpe.detach(),
                "micro_mpjpe_rel": micro_mpjpe_rel.detach(),
                "micro_mpvpe": micro_mpvpe.detach(),
            },
            "result": {
                "joint_cam_pred": joint_rel_pred.detach(),
                "verts_cam_pred": verts_rel_pred.detach(),
                "verts_cam_gt": verts_cam_gt.detach(),
            }
        }

        return loss_state

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_regressor_params(self):
        return itertools.chain(
            self.persp_info_embedder.parameters(),
            self.handec.parameters(),
        )
