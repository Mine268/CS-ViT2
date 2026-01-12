from typing import *
import random

import torch
import cv2
import numpy as np

from ..constant import *


def to_patch_coord(x2d, bbox):
    """
    x2d: [N,2]
    bbox: [4]
    """
    u = (x2d[:, 0] - bbox[None, 0]) / (bbox[None, 2] - bbox[None, 0])
    v = (x2d[:, 1] - bbox[None, 1]) / (bbox[None, 3] - bbox[None, 1])

    return torch.stack([u, v], dim=-1)


@torch.no_grad()
def vis(
    batch,
    trans_2d,
    result,
    tx: int,
    bx: Optional[int] = None,
):
    device = batch["joint_cam"].device
    batch_size = batch["joint_cam"].size(0)
    bx = bx if bx is not None else random.randint(0, batch_size - 1)

    img_path = batch["imgs_path"][bx][tx]
    joint_valid = batch["joint_valid"][bx, tx] # [j]
    joint_cam_pred = result["joint_cam_pred"][bx, tx] # [j,3]
    joint_img_gt = batch["joint_img"][bx, tx] # [j,2]

    focal = batch["focal"][bx, tx] # [2]
    princpt = batch["princpt"][bx, tx]

    # proj
    u = joint_cam_pred[:, 0] * focal[0] / joint_cam_pred[:, 2] + princpt[0]
    v = joint_cam_pred[:, 1] * focal[1] / joint_cam_pred[:, 2] + princpt[1]
    joint_img_pred = torch.stack([u, v], dim=-1)

    # img
    img = (255 * batch["patches"][bx, tx]).byte().cpu() # [c,h,w]
    img = torch.permute(img, dims=[1, 2, 0]).numpy().copy() # [h,w,c]
    length = img.shape[0]
    patch_bbox = batch["patch_bbox"][bx, tx] # [4]

    # new coord
    joint_img_pred = (
        (to_patch_coord(joint_img_pred, patch_bbox) * length)
        .cpu()
        .numpy()
        .astype(np.int32)
    )
    joint_img_gt = (
        (to_patch_coord(joint_img_gt, patch_bbox) * length)
        .cpu()
        .numpy()
        .astype(np.int32)
    )

    for conn in MANO_JOINTS_CONNECTION:
        pre, nex = conn

        img = cv2.line(
            img,
            joint_img_pred[pre],
            joint_img_pred[nex],
            color=(132, 132, 135),
            thickness=1
        )

        if not (joint_valid[pre] > 0.5 and joint_valid[nex] > 0.5):
            continue

        img = cv2.line(
            img,
            joint_img_gt[pre],
            joint_img_gt[nex],
            color=(132, 132, 135),
            thickness=1
        )

    for jx in range(HAND_JOINT_COUNT):
        img = cv2.circle(
            img,
            joint_img_pred[jx],
            radius=2,
            color=(0, 0, 255),
            thickness=-1
        )

        if joint_valid[jx] < 0.5:
            continue

        img = cv2.circle(
            img,
            joint_img_gt[jx],
            radius=2,
            color=(0, 255, 0),
            thickness=-1
        )

    return img
