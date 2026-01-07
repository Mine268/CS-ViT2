from typing import *

import einops as eps
import transformers
import torch
import torch.nn as nn
import kornia
import numpy as np
from accelerate.logging import get_logger

from ..constant import *
from ..utils.rot import *
from .hamer_module import Attention, PreNorm, FeedForward, TransformerDecoder, default


logger = get_logger(__name__)


class TRotionalPositionEmbedding(nn.Module):
    def __init__(self, dim: int, multi_head: bool = False):
        super(TRotionalPositionEmbedding, self).__init__()
        assert dim % 2 == 0

        self.dim = dim
        self.multi_head = multi_head

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, t: torch.Tensor = None):
        """
        Args:
            x (Tensor): [batch, seq, dim], [batch, head, seq, dim]
            t (Tensor): [batch, seq]
        """
        if t is None:
            raise ValueError("t must be provided for 'trope' mode")

        # Compute frequencies
        # [batch, seq, d_model//2]
        freqs = t.float().unsqueeze(-1) * self.inv_freq.unsqueeze(0)

        # Apply RoPE
        cos_vals = torch.cos(freqs)  # [batch, seq, d_model//2]
        sin_vals = torch.sin(freqs)
        if not self.multi_head:
            x_rotated = self._apply_rope(x, cos_vals, sin_vals)
        else:
            cos_vals = cos_vals.unsqueeze(1)
            sin_vals = sin_vals.unsqueeze(1)
            x_rotated = self._apply_rope(x, cos_vals, sin_vals)

        return x_rotated

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Applies RoPE to input tensor x using precomputed cos/sin values."""
        # Reshape x to [batch, seq, d_model//2, 2]
        x_reshaped = x.view(*x.shape[:-1], -1, 2)

        # Split into components
        x1, x2 = x_reshaped.unbind(dim=-1)

        # Apply rotation
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos

        # Recombine and flatten
        x_rotated = torch.stack([x1_rot, x2_rot], dim=-1)

        return x_rotated.flatten(start_dim=-2)


class PerspInfoEmbedderDense(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_sample: int,
        **kwargs,
    ):
        super(PerspInfoEmbedderDense, self).__init__()
        assert kwargs["pie_fusion"] in ["cls", "patch", "all"]

        self.hidden_size = hidden_size
        self.num_sample = num_sample
        self.pie_fusion = kwargs["pie_fusion"]

        in_dim = self.num_sample ** 2 * 2
        self.mlp = []
        for _ in range(3):
            self.mlp.extend([
                nn.Linear(in_dim, self.hidden_size, bias=True),
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU(),
            ])
            in_dim = self.hidden_size
        self.mlp = nn.Sequential(*self.mlp)

    def forward(
        self,
        feats: torch.Tensor,
        bbox: torch.Tensor,
        focal: torch.Tensor,
        princpt: torch.Tensor
    ):
        """
        feats: [b,n,d]
        bbox: [b,4] xyxy
        focal, princpt: [b,2]
        """
        grid = torch.linspace(
            1 / self.num_sample * 0.5,
            1 - 1 / self.num_sample * 0.5,
            self.num_sample,
            device=bbox.device,
        )  # [p]
        x_grid = (
            bbox[:, 0:1] + (bbox[:, 2:3] - bbox[:, 0:1]) * grid[None, :]
        )  # [b,p]
        y_grid = (
            bbox[:, 1:2] + (bbox[:, 3:4] - bbox[:, 1:2]) * grid[None, :]
        )  # [b,p]
        grid = torch.stack([
            x_grid[:, :, None].expand(-1, -1, grid.shape[0]),
            y_grid[:, None, :].expand(-1, grid.shape[0], -1),
        ], dim=-1)# [b,p,p,2]

        directions = (grid - princpt[:, None, None, :]) / focal[:, None, None, :]
        directions = torch.cat([directions, torch.ones_like(directions[..., :1])], dim=-1)
        directions = directions / torch.norm(directions, p="fro", dim=-1, keepdim=True)
        directions = directions[..., :2]  # [b,p,p,2] discard z value

        flatten = eps.rearrange(directions, "b p q d -> b (p q d)")
        persp_feat = self.mlp(flatten) # [b,d]

        if self.pie_fusion == "cls":
            feats[:, :0] = feats[:, :0] + persp_feat[:, None]
        elif self.pie_fusion == "patch":
            feats[:, 1:] = feats[:, 1:] + persp_feat[:, None]
        elif self.pie_fusion == "all":
            feats = feats + persp_feat[:, None]
        else:
            raise NotImplementedError(f"pie_fusion={self.pie_fusion} is not implemented.")

        return feats


class PerspInfoEmbedderCrossAttn(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_sample: int,
        num_token: int,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_sample = num_sample

        self.net = TransformerDecoder(
            num_tokens=num_token,
            token_dim=self.hidden_size,
            dim=self.hidden_size,
            depth=1,
            heads=8,
            mlp_dim=4*self.hidden_size,
            dim_head=64,
            dropout=0.0,
            emb_dropout=0.0,
            emb_dropout_type="drop",
            norm="layer",
            norm_cond_dim=-1,
            context_dim=2,
            skip_token_embedding=False
        )

    def forward(
        self,
        feats: torch.Tensor,
        bbox: torch.Tensor,
        focal: torch.Tensor,
        princpt: torch.Tensor
    ):
        grid = torch.linspace(
            1 / self.num_sample * 0.5,
            1 - 1 / self.num_sample * 0.5,
            self.num_sample,
            device=bbox.device,
        )  # [p]
        x_grid = (
            bbox[:, 0:1] + (bbox[:, 2:3] - bbox[:, 0:1]) * grid[None, :]
        )  # [b,p]
        y_grid = (
            bbox[:, 1:2] + (bbox[:, 3:4] - bbox[:, 1:2]) * grid[None, :]
        )  # [b,p]
        grid = torch.stack([
            x_grid[:, :, None].expand(-1, -1, grid.shape[0]),
            y_grid[:, None, :].expand(-1, grid.shape[0], -1),
        ], dim=-1)# [b,p,p,2]

        directions = (grid - princpt[:, None, None, :]) / focal[:, None, None, :]
        directions = torch.cat([directions, torch.ones_like(directions[..., :1])], dim=-1)
        directions = directions / torch.norm(directions, p="fro", dim=-1, keepdim=True)
        directions = directions[..., :2]  # [b,p,p,2] discard z value
        directions = eps.rearrange(directions, "b p q d -> b (p q) d") # [b,n,d]

        out = self.net(feats, context=directions)

        return out


class ViTBackbone(nn.Module):
    def __init__(
        self,
        backbone_str: str,
        img_size: Optional[int],
        infusion_feats_lyr: List[int],
        backbone_kwargs: Dict,
    ):
        super(ViTBackbone, self).__init__()

        self.backbone_str = backbone_str
        self.img_size = img_size
        self.infusion_feats_lyr = infusion_feats_lyr

        # read model config
        backbone_cfg = transformers.AutoConfig.from_pretrained(self.backbone_str)
        self.patch_size = backbone_cfg.patch_size
        self.hidden_size = backbone_cfg.hidden_size
        if self.img_size is None:
            self.img_size = backbone_cfg.image_size
            logger.info("No img_size provided for backbone. "
                f"Loading from pretrained config img_size={self.img_size}.")
        elif self.img_size != backbone_cfg.image_size:
            logger.warning(f"Provided img_size={self.img_size} is not consistent with "
                f"image_size={backbone_cfg.image_size} from config.json. "
                f"Using img_size={self.img_size}.")

        # post configuration
        assert (
            self.img_size % self.patch_size == 0
        ), f"img_size={self.img_size} and patch_size={self.patch_size} is not consistent."
        self.num_patch = self.img_size // self.patch_size

        # backbone
        self.backbone = transformers.AutoModel.from_pretrained(
            self.backbone_str,
            output_hidden_states=True,
            **backbone_kwargs,
        )

        # multi-layer fusion
        if self.infusion_feats_lyr is not None:
            self.proj_size = 64
            self.projection_cls = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.hidden_size, self.proj_size, bias=True),
                    nn.BatchNorm1d(self.proj_size),
                    nn.ReLU(),
                ) for _ in range(len(self.infusion_feats_lyr))
            ])
            self.projections_map = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.hidden_size, self.proj_size, kernel_size=1),
                    nn.BatchNorm2d(self.proj_size),
                    nn.ReLU(),
                ) for _ in range(len(self.infusion_feats_lyr))
            ])
            self.fusion_cls = nn.Sequential(
                nn.Linear(
                    self.proj_size * len(self.infusion_feats_lyr),
                    self.hidden_size,
                    bias=True
                ),
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU(),
            )
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(
                    self.proj_size * len(self.infusion_feats_lyr),
                    self.hidden_size,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(self.hidden_size),
                nn.ReLU()
            )

    def forward(self, x: torch.Tensor):
        assert x.shape[-1] == x.shape[-2], "Input tensor is not square."
        assert (
            x.shape[-1] == self.img_size
        ), f"Input tensor shape {x.shape} is not consistent with img_size={self.img_size}"

        backbone_output = self.backbone(x)

        if self.infusion_feats_lyr is not None:
            hidden_states = backbone_output.hidden_states
            hidden_states = [hidden_states[i] for i in self.infusion_feats_lyr]
            token_clss, token_patches = [], []
            for l in range(len(self.infusion_feats_lyr)):
                token_cls, token_patch = hidden_states[l][:, 0], hidden_states[l][:, 1:]
                token_patch = eps.rearrange(token_patch, "b (h w) c -> b c h w", h=self.num_patch)
                token_cls = self.projection_cls[l](token_cls)
                token_patch = self.projections_map[l](token_patch)
                token_clss.append(token_cls)
                token_patches.append(token_patch)
            token_clss = torch.cat(token_clss, dim=-1)
            token_patches = torch.cat(token_patches, dim=1)
            token_clss = self.fusion_cls(token_clss)
            token_patches = self.fusion_conv(token_patches)
            token_patches = eps.rearrange(token_patches, "b c h w -> b (h w) c")
            hidden_state = torch.cat([token_clss[:, None], token_patches], dim=1)
        else:
            hidden_state = backbone_output.last_hidden_state

        return hidden_state

    '''
    explicitly expose the member
    '''
    def get_patch_size(self):
        return self.patch_size

    def get_hidden_size(self):
        return self.hidden_size

    def get_img_size(self):
        return self.img_size

    def get_num_patch(self):
        return self.num_patch


class TRoPECrossAttention(nn.Module):

    def __init__(self, dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        context_dim = default(context_dim, dim)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.pe = TRotionalPositionEmbedding(dim_head, multi_head=bool(heads > 1))

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, tq, tk, context=None):
        """
        x: [b n d]
        context: [b m d]
        tq: [b n]
        tk: [b m]
        """
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q = self.to_q(x)
        q, k, v = map(lambda t: eps.rearrange(t, "b n (h d) -> b h n d", h=self.heads), [q, k, v])

        q = self.pe(q, tq)
        k = self.pe(k, tk)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = eps.rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class TRoPETransformerCrossAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            sa = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ca = TRoPECrossAttention(
                dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout
            )
            ff = FeedForward(dim, mlp_dim, dropout=dropout)
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, sa, norm=norm, norm_cond_dim=norm_cond_dim),
                        PreNorm(dim, ca, norm=norm, norm_cond_dim=norm_cond_dim),
                        PreNorm(dim, ff, norm=norm, norm_cond_dim=norm_cond_dim),
                    ]
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        tq: torch.Tensor,
        tk: torch.Tensor,
        *args,
        context=None,
        context_list=None,
    ):
        if context_list is None:
            context_list = [context] * len(self.layers)
        if len(context_list) != len(self.layers):
            raise ValueError(
                f"len(context_list) != len(self.layers) ({len(context_list)} != {len(self.layers)})"
            )

        for i, (self_attn, cross_attn, ff) in enumerate(self.layers):
            x = self_attn(x, *args) + x
            x = cross_attn(x, *args, tq=tq, tk=tk, context=context_list[i]) + x
            x = ff(x, *args) + x
        return x


# ref: hamer
class MANOTransformerDecoderHead(nn.Module):
    def __init__(
        self,
        joint_rep_type: str,
        # num_tokens: int,
        # token_dim: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        emb_dropout_type: str = 'drop',
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim: Optional[int] = None,
        skip_token_embedding: bool = False,
    ):
        super().__init__()
        assert joint_rep_type in JOINT_DIM_DICT

        self.joint_rep_type = joint_rep_type
        self.joint_dim = JOINT_DIM_DICT[joint_rep_type]
        npose = self.joint_dim * MANO_JOINT_COUNT

        self.transformer = TransformerDecoder(
            num_tokens=1,
            token_dim=(npose + MANO_SHAPE_DIM + 3),
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
            emb_dropout_type=emb_dropout_type,
            norm=norm,
            norm_cond_dim=norm_cond_dim,
            context_dim=context_dim,
            skip_token_embedding=skip_token_embedding
        )

        self.decpose = nn.Linear(dim, npose)
        self.decshape = nn.Linear(dim, MANO_SHAPE_DIM)
        self.deccam = nn.Linear(dim, 3)

        mean_params = np.load(MANO_MEAN_NPZ)
        # [96]
        init_hand_pose = torch.from_numpy(mean_params['pose'].astype(np.float32))
        if joint_rep_type == "6d":
            pass
        elif joint_rep_type == "3":
            init_hand_pose = rotation6d_to_rotation_matrix(init_hand_pose.reshape(-1, 6))
            init_hand_pose = kornia.geometry.conversions.rotation_matrix_to_axis_angle(
                init_hand_pose
            )
            # [1,48]
            init_hand_pose = torch.flatten(init_hand_pose).unsqueeze(0)
        elif joint_rep_type == "quat":
            init_hand_pose = rotation6d_to_rotation_matrix(init_hand_pose.reshape(-1, 6))
            init_hand_pose = kornia.geometry.conversions.rotation_matrix_to_quaternion(
                init_hand_pose
            )
            # [1,64]
            init_hand_pose = torch.flatten(init_hand_pose).unsqueeze(0)
        else:
            raise NotImplementedError
        # [1,10]
        init_betas = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)
        # [1,4]
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32)).unsqueeze(0)
        self.register_buffer('init_hand_pose', init_hand_pose)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]

        init_hand_pose = self.init_hand_pose.expand(batch_size, -1)
        init_betas = self.init_betas.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        # [b,1,npose+10+3]
        token = torch.cat([init_hand_pose, init_betas, init_cam], dim=1)[:, None, :]
        token_out = self.transformer(token, context=x)
        token_out = token_out.squeeze(1)

        pred_hand_pose = self.decpose(token_out) + init_hand_pose
        pred_betas = self.decshape(token_out) + init_betas
        pred_cam = self.deccam(token_out) + init_cam

        return pred_hand_pose, pred_betas, pred_cam


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_head: int,
        num_layer: int,
        dropout: float,
        trope_scalar: float = 20.0,
        zero_linear: bool = True
    ):
        super(TemporalEncoder, self).__init__()

        self.dim = dim
        self.trope_scalar = trope_scalar

        self.pe = TRotionalPositionEmbedding(self.dim)
        self.zero_linear = nn.Linear(self.dim, self.dim, bias=True)
        self.cross_attn = TRoPETransformerCrossAttn(
            dim=self.dim,
            depth=num_layer,
            heads=num_head,
            dim_head=self.dim // num_head,
            mlp_dim=self.dim * 2,
            dropout=dropout,
            norm="layer",
            norm_cond_dim=-1,
            context_dim=self.dim,
        )

        if zero_linear:
            nn.init.zeros_(self.zero_linear.weight)
            nn.init.zeros_(self.zero_linear.bias)

    def forward(
        self,
        token: torch.Tensor,
        timestamp: torch.Tensor
    ):
        """
        token: [b,t,d]
        timestamp: [b,t]
        """
        b, t, _ = token.shape

        x = token[:, -1:]
        ctx = token

        timestamp /= self.trope_scalar
        tq = timestamp[:, -1:]
        tk = timestamp

        y = self.cross_attn(x, tq, tk, context=ctx)
        y = self.zero_linear(y)

        return x + y


class MANOPoseDetokenizer(nn.Module):
    def __init__(self, dim: int, joint_rep_type: str):
        super(MANOPoseDetokenizer, self).__init__()
        assert joint_rep_type in JOINT_DIM_DICT
        joint_dim: int = JOINT_DIM_DICT[joint_rep_type]

        self.pose_linear = nn.Linear(dim, MANO_JOINT_COUNT * joint_dim, bias=True)
        self.shape_linear = nn.Linear(dim, MANO_SHAPE_DIM, bias=True)
        self.trans_linear = nn.Linear(dim, 3, bias=True)

    def forward(
        self,
        pose_token: torch.Tensor,
        shape_token: torch.Tensor,
        trans_token: torch.Tensor,
    ):
        pose = self.pose_linear(pose_token)
        shape = self.shape_linear(shape_token)
        trans = self.trans_linear(trans_token)

        return pose, shape, trans
