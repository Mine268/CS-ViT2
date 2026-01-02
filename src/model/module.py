from typing import *
import os.path as osp
import json

import transformers
import torch
import torch.nn as nn
import smplx
import einops as eps

from ..constant import *


class PerspInfoEmbedderDense(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_sample: int,
    ):
        super(PerspInfoEmbedderDense, self).__init__()

        self.hidden_size = hidden_size
        self.num_sample = num_sample

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
        bbox: torch.Tensor,
        focal: torch.Tensor,
        princpt: torch.Tensor
    ):
        """
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
        persp_feat = self.mlp(flatten)

        return persp_feat


class DinoBackbone(nn.Module):
    def __init__(
        self,
        backbone_str: str,
        img_size: Optional[int],
        infusion_feats_lyr: List[int],
    ):
        super(DinoBackbone, self).__init__()

        self.backbone_str = backbone_str
        self.img_size = img_size
        self.infusion_feats_lyr = infusion_feats_lyr

        # read model config
        backbone_cfg = transformers.AutoConfig.from_pretrained(self.backbone_str)
        self.patch_size = backbone_cfg.patch_size
        self.hidden_size = backbone_cfg.hidden_size
        if self.img_size is None:
            self.img_size = backbone_cfg.image_size
            print("No img_size provided for backbone. ", end="")
            print(f"Loading from pretrained config img_size={self.img_size}.", end="\n")
        elif self.img_size != backbone_cfg.image_size:
            print(f"Provided img_size={self.img_size} is not consistent with ", end="")
            print(f"image_size={backbone_cfg.image_size} from config.json. ", end="")
            print(f"Using img_size={self.img_size}", end="\n")

        # post configuration
        assert (
            self.img_size % self.patch_size == 0
        ), f"img_size={self.img_size} and patch_size={self.patch_size} is not consistent."
        self.num_patch = self.img_size // self.patch_size

        # backbone
        print("Loading model...")
        self.backbone = transformers.AutoModel.from_pretrained(
            self.backbone_str,
            output_hidden_states=True,
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


class TemporalEncoder(nn.Module):
    def __init__(self):
        super(TemporalEncoder, self).__init__()


class MANOPoseDetokenizer(nn.Module):
    joint_dim_dict = {"6d": 6, "3": 3, "quat": 4}

    def __init__(self, dim: int, joint_rep_type: str):
        super(MANOPoseDetokenizer, self).__init__()
        assert joint_rep_type in MANOPoseDetokenizer.joint_dim_dict
        joint_dim: int = MANOPoseDetokenizer.joint_dim_dict[joint_rep_type]

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
