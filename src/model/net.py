from typing import *
import enum
import itertools

import numpy as np
import torch.nn as nn
import kornia
import smplx
from safetensors.torch import load_file

from .loss import *
from .module import *
from .hamer_module import *
from ..utils.metric import *
from ..utils.proj import *


class PoseNet(nn.Module):
    class Stage(enum.Enum):
        STAGE1 = "stage1"
        STAGE2 = "stage2"

    def __init__(
        self,
        stage: Stage,
        stage1_weight_path: Optional[str],

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
        handec_heatmap_resulotion: Union[int, Tuple[int]],

        pie_type: str,
        num_pie_sample: int,
        pie_fusion: str,

        num_temporal_head: int,
        num_temporal_layer: int,
        trope_scalar: float,
        zero_linear: bool,

        joint_rep_type: str,

        supervise_global: bool,
        supervise_heatmap: bool,
        lambda_theta: float,
        lambda_shape: float,
        lambda_trans: float,
        lambda_rel: float,
        lambda_img: float,
        hm_sigma: float,
        reproj_loss_type: str,
        reproj_loss_delta: float,

        freeze_backbone: bool,
        norm_by_hand: bool,
    ):
        super(PoseNet, self).__init__()

        self.stage = PoseNet.Stage(stage)

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
            norm_by_hand=norm_by_hand,
            heatmap_resolution=handec_heatmap_resulotion,
        )

        self.norm_by_hand = norm_by_hand
        if norm_by_hand:
            norm_stats = np.load(NORM_STAT_NPZ)
            norm_list = norm_stats["norm_list"].flatten().tolist()
            self.norm_idx = [HAND_JOINTS_ORDER.index(x) for x in norm_list]

        # temporal encoder
        self.temporal_refiner = TemporalEncoder(
            dim=self.hidden_size,
            num_head=num_temporal_head,
            num_layer=num_temporal_layer,
            dropout=prob_handec_dropout,
            trope_scalar=trope_scalar,
            zero_linear=zero_linear,
        )

        # Loss
        self.supervise_global = supervise_global
        self.supervise_heatmap = supervise_heatmap
        self.loss_fn = BundleLoss2(
            lambda_theta=lambda_theta,
            lambda_shape=lambda_shape,
            lambda_trans=lambda_trans,
            lambda_rel=lambda_rel,
            lambda_img=lambda_img,
            supervise_global=True,
            supervise_heatmap=supervise_heatmap,
            norm_by_hand=norm_by_hand,
            norm_idx=self.norm_idx if norm_by_hand else [],
            hm_centers=None if not supervise_heatmap else self.handec.get_centers(),
            hm_sigma=hm_sigma,
            reproj_loss_type=reproj_loss_type,
            reproj_loss_delta=reproj_loss_delta,
        )
        self.metric_meter = MetricMeter()

        # train
        self.freeze_backbone = freeze_backbone

        if self.stage == PoseNet.Stage.STAGE2 and stage1_weight_path is not None:
            self.load_pretrained(stage1_weight_path)

    def load_pretrained(self, path: str):
        """从 checkpoint 目录或文件加载预训练权重

        支持路径格式：checkpoint/date/run/checkpoints/checkpoint-9000

        Args:
            path: checkpoint 目录或权重文件路径

        Returns:
            None

        Note:
            使用 strict=False 允许部分加载（Stage 2 有新增的 temporal_refiner）
        """
        import os
        from accelerate.logging import get_logger

        logger = get_logger(__name__)

        # 1. 确定实际权重文件路径
        # 路径是目录，尝试加载标准 Accelerate checkpoint
        model_path = os.path.join(path, "model.safetensors")
        logger.info(f"Loading pretrained weights from directory: {model_path}")

        # 2. 加载权重
        state_dict = load_file(model_path)
        logger.info("Loaded weights using safetensors format")

        missing, unexpected = self.load_state_dict(state_dict, strict=False)

        # 3. 记录加载结果
        if len(missing) > 0:
            logger.warning(
                f"Missing keys when loading pretrained weights ({len(missing)} keys):\n"
                f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
            )
        if len(unexpected) > 0:
            logger.warning(
                f"Unexpected keys when loading pretrained weights ({len(unexpected)} keys):\n"
                f"{unexpected[:10]}{'...' if len(unexpected) > 10 else ''}"
            )

        # 4. 验证关键模块是否成功加载（Stage 2 特有逻辑）
        if self.stage == PoseNet.Stage.STAGE2:
            spatial_keys = [
                k
                for k in state_dict.keys()
                if any(
                    prefix in k
                    for prefix in ["backbone.", "persp_info_embedder.", "handec."]
                )
            ]
            loaded_spatial_keys = [k for k in spatial_keys if k not in missing]
            logger.info(
                f"Stage 2: Loaded {len(loaded_spatial_keys)}/{len(spatial_keys)} spatial module parameters"
            )

            # temporal_refiner 应该在 missing keys 中（预期行为）
            temporal_missing = [k for k in missing if "temporal_refiner" in k]
            if len(temporal_missing) > 0:
                logger.info(
                    f"Stage 2: temporal_refiner not in checkpoint (expected, will be randomly initialized)"
                )

        logger.info("Pretrained weights loaded successfully")

    def get_hand_norm_scale(self, j3d: torch.Tensor, valid: torch.Tensor):
        """
        Args:
            j3d: [...,j,3]
            valid: [...,j]
            return: [...], [...]
        """
        d = j3d[..., self.norm_idx[:-1], :] - j3d[..., self.norm_idx[1:], :]
        d = torch.sum(torch.sqrt(torch.sum(d ** 2, dim=-1)), dim=-1) # [...]

        # 防止 norm_scale 过小（双重保护）
        d = torch.clamp(d, min=NORM_SCALE_EPSILON)

        flag = torch.all(valid[:, :, self.norm_idx] > 0.5, dim=-1).float()
        return d, flag

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
        # [b,n], hm_x, hm_y, hm_z
        (pred_hand_pose, pred_shape, pred_trans), pred_log_heatmaps, pred_tokens = (
            self.handec(feats)
        )
        if self.supervise_heatmap:
            pred_trans.detach()

        return (pred_hand_pose, pred_shape, pred_trans), pred_log_heatmaps, pred_tokens

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

        if self.stage == PoseNet.Stage.STAGE1:
            (pose, shape, trans), log_heatmaps, _ = self.decode_hand_param(
                img=img,
                bbox=bbox,
                focal=focal,
                princpt=princpt,
            )
        elif self.stage == PoseNet.Stage.STAGE2:
            _, _, tokens_out = self.decode_hand_param( # repeat for readibility
                img=img,
                bbox=bbox,
                focal=focal,
                princpt=princpt,
            ) # [(b*t),d]
            # [b,t,d]
            tokens_out = eps.rearrange(tokens_out, "(b t) d -> b t d", t=num_frame)
            tokens_out = self.temporal_refiner(tokens_out, timestamp)

            (pose, shape, trans), log_heatmaps = self.handec.decode_token(
                eps.rearrange(tokens_out, "b t d -> (b t) d")
            )

        # reshape: Stage 1 输出 [b,1,d], Stage 2 输出 [b,t,d]
        out_frames = num_frame if self.stage == PoseNet.Stage.STAGE2 else 1
        pose, shape, trans = map(
            lambda t: eps.rearrange(t, "(b t) d -> b t d", t=out_frames),
            [pose, shape, trans]
        )
        log_heatmaps = tuple(
            map(
                lambda t: eps.rearrange(t, "(b t) d -> b t d", t=out_frames),
                log_heatmaps,
            )
        )

        return pose, shape, trans, log_heatmaps

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

    @torch.inference_mode()
    def predict_full(
        self,
        img: torch.Tensor,
        bbox: torch.Tensor,
        focal: torch.Tensor,
        princpt: torch.Tensor,
        timestamp: Optional[torch.Tensor] = None,
        joint_cam_gt: Optional[torch.Tensor] = None,
        joint_valid_gt: Optional[torch.Tensor] = None,
    ):
        """
        测试专用推理函数，返回完整的预测结果（包括原始 MANO 参数和 FK 结果）

        **重要**：正确处理 norm_by_hand 逻辑，返回真实相机坐标系下的 joints/verts

        Args:
            img: [B, T, 3, H, W] 图像 patches
            bbox: [B, T, 4] 手部 bounding box (xyxy格式)
            focal: [B, T, 2] 相机焦距 (fx, fy)
            princpt: [B, T, 2] 相机主点 (cx, cy)
            timestamp: [B, T] 时间戳（可选，Stage 2 需要）
            joint_cam_gt: [B, T, 21, 3] GT joints（可选，用于 norm_by_hand 反归一化）
            joint_valid_gt: [B, T, 21] GT joint valid（可选，用于 norm_by_hand 反归一化）

        Returns:
            dict: {
                "mano_pose_pred": [B, 1, 48],      # 最后一帧
                "mano_shape_pred": [B, 1, 10],
                "trans_pred": [B, 1, 3],           # 归一化的 trans（如果 norm_by_hand=true）
                "trans_pred_denorm": [B, 1, 3],    # 反归一化的 trans（真实相机坐标）
                "joint_cam_pred": [B, 1, 21, 3],   # 真实相机坐标系
                "vert_cam_pred": [B, 1, 778, 3],   # 真实相机坐标系
                "joint_rel_pred": [B, 1, 21, 3],   # 相对根关节坐标
                "vert_rel_pred": [B, 1, 778, 3],   # 相对根关节坐标
                "norm_scale": [B, 1],              # 手部大小（如果 norm_by_hand=true）
                "norm_valid": [B, 1],              # norm_scale 是否有效
            }
        """
        # 1. 获取原始 MANO 参数（复用 predict_mano_param）
        pose_pred, shape_pred, trans_pred, _ = self.predict_mano_param(
            img=img, bbox=bbox, focal=focal, princpt=princpt, timestamp=timestamp
        )

        # 2. FK — 推理只取最后一帧
        pose_pred = pose_pred[:, -1:]
        shape_pred = shape_pred[:, -1:]
        trans_pred = trans_pred[:, -1:]
        joint_rel_pred, vert_rel_pred = self.mano_to_pose(pose_pred, shape_pred)

        # 3. 处理 norm_by_hand 反归一化
        trans_for_fk = trans_pred  # [B, 1, 3]

        if not self.norm_by_hand:
            trans_pred_denorm = trans_for_fk
            norm_scale = torch.ones(
                (trans_for_fk.shape[0], 1), device=trans_for_fk.device
            )
            norm_valid = torch.ones(
                (trans_for_fk.shape[0], 1), device=trans_for_fk.device
            )
        else:
            norm_scale_gt, norm_valid_gt = self.get_hand_norm_scale(
                joint_cam_gt[:, -1:], joint_valid_gt[:, -1:]
            )
            norm_scale_pred, _ = self.get_hand_norm_scale(
                joint_rel_pred[:, -1:], torch.ones_like(joint_rel_pred[:, -1:])
            )
            norm_scale = ( # [b,1,1]
                norm_valid_gt[:, -1:] * norm_scale_gt[..., None] +
                (1 - norm_valid_gt[:, -1:]) * norm_scale_pred[..., None]
            )
            norm_valid = norm_valid_gt # [b,1]
            # [b,1,3]
            trans_pred_denorm = trans_for_fk * norm_scale

        # 4. 转换到相机坐标系（使用反归一化后的 trans）
        joint_cam_pred = joint_rel_pred + trans_pred_denorm[:, :, None, :]
        vert_cam_pred = vert_rel_pred + trans_pred_denorm[:, :, None, :]

        return {
            "mano_pose_pred": pose_pred,
            "mano_shape_pred": shape_pred,
            "trans_pred": trans_pred,  # 原始输出（可能归一化）
            "trans_pred_denorm": trans_pred_denorm,  # 反归一化后的 trans
            "joint_cam_pred": joint_cam_pred,  # 真实相机坐标
            "vert_cam_pred": vert_cam_pred,  # 真实相机坐标
            "joint_rel_pred": joint_rel_pred,
            "vert_rel_pred": vert_rel_pred,
            "norm_scale": norm_scale,
            "norm_valid": norm_valid,
        }

    def forward(self, batch):
        # 1. forward
        pose_pred, shape_pred, trans_pred, log_hm_pred = self.predict_mano_param(
            img=batch["patches"],
            bbox=batch["patch_bbox"],
            focal=batch["focal"],
            princpt=batch["princpt"],
            timestamp=batch["timestamp"]
        )

        # 2. loss, fk
        loss, loss_state, result = self.loss_fn(
            pose_pred, shape_pred, trans_pred, log_hm_pred, batch
        )

        # 3. micro metric
        metric_state = self.metric_meter(
            batch["joint_cam"][:, -1:],
            batch["joint_cam"][:, -1:] - batch["joint_cam"][:, -1:, :1],
            result["verts_cam_gt"][:, -1:],
            result["verts_rel_gt"][:, -1:],
            result["joint_cam_pred"][:, -1:],
            result["joint_rel_pred"][:, -1:],
            result["verts_cam_pred"][:, -1:],
            result["verts_rel_pred"][:, -1:],
            batch["mano_valid"][:, -1:],
            batch["joint_valid"][:, -1:],
            result["norm_valid_gt"][:, -1:],
        )

        loss_state = {
            "loss": loss,
            "state": loss_state | metric_state,
            "result": {
                "joint_cam_pred": result["joint_cam_pred"].detach(),
                "verts_cam_pred": result["verts_cam_pred"].detach(),
                "verts_cam_gt": result["verts_cam_gt"].detach(),
            }
            | ({"norm_idx": self.norm_idx} if self.norm_by_hand else {}),
        }

        return loss_state

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_regressor_params(self):
        return itertools.chain(
            self.persp_info_embedder.parameters(),
            self.handec.parameters(),
        )

    def get_optim_param_dict(self, lr: float, backbone_lr: Optional[float]):
        ret = []
        if self.stage == PoseNet.Stage.STAGE1:
            ret.append(
                {
                    "params": filter(
                        lambda p: p.requires_grad,
                        itertools.chain(
                            self.persp_info_embedder.parameters(),
                            self.handec.parameters(),
                        ),
                    ),
                    "lr": lr
                }
            )
            if backbone_lr is not None:
                ret.append(
                    {
                        "params": filter(
                            lambda p: p.requires_grad, self.backbone.parameters()
                        ),
                        "lr": backbone_lr,
                    }
                )
        elif self.stage == PoseNet.Stage.STAGE2:
            ret.append(
                {
                    "params": filter(
                        lambda p: p.requires_grad, self.temporal_refiner.parameters()
                    ),
                    "lr": lr,
                }
            )

        return ret

    def get_norm_idx(self):
        return self.norm_idx

    def set_dropout_rate(self, dropout_rate: float):
        """动态设置 HandDecoder 和 TemporalEncoder 中所有 Dropout 层的 dropout 率

        使用通用的模块遍历来修改所有 nn.Dropout 层，支持：
        - HandDecoder (TransformerCrossAttn)
        - TemporalEncoder (TRoPETransformerCrossAttn)

        用于实现渐进式dropout策略：训练早期使用低dropout率（或0），
        训练后期逐步提升dropout率以增强泛化性。

        Args:
            dropout_rate: 新的 dropout 率，范围 [0.0, 1.0]

        Raises:
            ValueError: 如果 dropout_rate 不在有效范围内

        Examples:
            >>> net.set_dropout_rate(0.0)  # 训练早期禁用dropout
            >>> net.set_dropout_rate(0.1)  # 训练后期启用dropout
        """
        # 校验输入范围
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(
                f"dropout_rate must be in range [0.0, 1.0], got {dropout_rate}"
            )

        # 修改 HandDecoder 中的所有 Dropout 层
        for module in self.handec.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate

        # 修改 TemporalEncoder 中的所有 Dropout 层（如果存在）
        if hasattr(self, "temporal_refiner") and self.temporal_refiner is not None:
            for module in self.temporal_refiner.modules():
                if isinstance(module, nn.Dropout):
                    module.p = dropout_rate

    @override
    def train(self, mode=True):
        if self.stage == PoseNet.Stage.STAGE1:
            super(PoseNet, self).train(mode)
            self.backbone.train(mode and not self.freeze_backbone)
        elif self.stage == PoseNet.Stage.STAGE2:
            super(PoseNet, self).train(mode)
            self.backbone.train(False)
            self.persp_info_embedder.train(False)
            self.handec.train(False)
            self.temporal_refiner.train(mode)

    @override
    def eval(self):
        self.train(False)
