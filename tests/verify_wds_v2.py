"""
Standalone visual verification tool for V2 WebDataset samples.
"""

import argparse
import copy
import glob
import json
import os
from typing import Dict, List

import cv2
import numpy as np
import torch
import kornia.geometry.transform as KT

try:
    import smplx
except ImportError:
    smplx = None

try:
    from .dataloader_v2 import get_dataloader
    from .preprocess_v2 import preprocess_batch
except ImportError:
    from dataloader_v2 import get_dataloader
    from preprocess_v2 import preprocess_batch


def _tensor_image_to_bgr(image: torch.Tensor) -> np.ndarray:
    if image.dtype != torch.uint8:
        image = torch.clamp(image * 255.0, 0.0, 255.0).to(torch.uint8)
    rgb = image.permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _draw_joints(
    image: np.ndarray,
    joints: np.ndarray,
    valid: np.ndarray,
    color=(255, 50, 50),
    radius: int = 3,
) -> np.ndarray:
    output = image.copy()
    for idx, joint in enumerate(joints):
        if valid is not None and float(valid[idx]) <= 0.5:
            continue
        cv2.circle(output, (int(joint[0]), int(joint[1])), radius, color, -1)
        cv2.putText(
            output,
            str(idx),
            (int(joint[0]), int(joint[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def _crop_bbox(image: torch.Tensor, bbox: torch.Tensor) -> np.ndarray:
    x1, y1, x2, y2 = bbox.float()
    width = max(1, int(torch.round(x2 - x1).item()))
    height = max(1, int(torch.round(y2 - y1).item()))

    corners = torch.tensor(
        [
            [x1.item(), y1.item()],
            [x2.item(), y1.item()],
            [x2.item(), y2.item()],
            [x1.item(), y2.item()],
        ],
        dtype=torch.float32,
        device=image.device,
    )[None]
    cropped = KT.crop_and_resize(
        image[None].float() / 255.0,
        corners,
        (height, width),
        mode="bilinear",
    )[0]
    return _tensor_image_to_bgr(cropped)


def _save_metadata(batch: Dict[str, List], output_dir: str, bx: int, tx: int) -> None:
    metadata = {
        "__key__": batch["__key__"][bx],
        "imgs_path": batch["imgs_path"][bx][tx],
        "handedness": batch["handedness"][bx],
        "data_source": batch["data_source"][bx],
        "source_split": batch["source_split"][bx],
        "intr_type": batch["intr_type"][bx],
        "source_index": batch["source_index"][bx][tx],
        "additional_desc": batch["additional_desc"][bx][tx],
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)


def verify_origin_sample(batch: Dict[str, List], output_dir: str, bx: int, tx: int) -> None:
    os.makedirs(output_dir, exist_ok=True)
    _save_metadata(batch, output_dir, bx, tx)

    img = _tensor_image_to_bgr(batch["imgs"][bx][tx])
    info_img = img.copy()
    cv2.putText(
        info_img,
        f"{batch['data_source'][bx]} | {batch['source_split'][bx]} | {batch['intr_type'][bx]}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        info_img,
        batch["handedness"][bx],
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(os.path.join(output_dir, "origin.png"), info_img)

    hand_bbox = batch["hand_bbox"][bx, tx].cpu().numpy()
    joint_img = batch["joint_img"][bx, tx].cpu().numpy()
    joint_2d_valid = batch["joint_2d_valid"][bx, tx].cpu().numpy()

    bbox_img = img.copy()
    cv2.rectangle(
        bbox_img,
        (int(hand_bbox[0]), int(hand_bbox[1])),
        (int(hand_bbox[2]), int(hand_bbox[3])),
        (0, 0, 255),
        2,
    )
    bbox_img = _draw_joints(bbox_img, joint_img, joint_2d_valid)
    cv2.imwrite(os.path.join(output_dir, "hand_bbox-joint_img.png"), bbox_img)

    hand_crop = _crop_bbox(batch["imgs"][bx][tx], batch["hand_bbox"][bx, tx].cpu())
    joint_hand_origin = batch.get("joint_hand_origin", batch["joint_hand_bbox"])[
        bx, tx
    ].cpu().numpy()
    hand_crop = _draw_joints(hand_crop, joint_hand_origin, joint_2d_valid)
    cv2.imwrite(os.path.join(output_dir, "joint_hand_origin.png"), hand_crop)

    has_intr = float(batch["has_intr"][bx, tx].item()) > 0.5
    has_3d = bool(torch.any(batch["joint_3d_valid"][bx, tx] > 0.5).item())
    if has_intr and has_3d:
        joint_cam = batch["joint_cam"][bx, tx].cpu()
        focal = batch["focal"][bx, tx].cpu()
        princpt = batch["princpt"][bx, tx].cpu()
        valid_mask = batch["joint_3d_valid"][bx, tx].cpu().numpy()
        reproj_u = focal[0] * joint_cam[:, 0] / joint_cam[:, 2] + princpt[0]
        reproj_v = focal[1] * joint_cam[:, 1] / joint_cam[:, 2] + princpt[1]
        reproj = torch.stack([reproj_u, reproj_v], dim=-1).numpy()
        reproj_img = _draw_joints(img, reproj, valid_mask)
        cv2.imwrite(os.path.join(output_dir, "joint_cam_reproj.png"), reproj_img)


def verify_processed_sample(
    batch: Dict[str, List],
    origin_batch: Dict[str, List],
    trans_2d_mat: torch.Tensor,
    output_dir: str,
    bx: int,
    tx: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    img = _tensor_image_to_bgr(origin_batch["imgs"][bx][tx])
    trans_np = trans_2d_mat[bx, tx].detach().cpu().numpy()
    img = cv2.warpPerspective(img, trans_np, (img.shape[1], img.shape[0]))
    if batch["flip"][bx]:
        img = img[:, ::-1].copy()
    cv2.imwrite(os.path.join(output_dir, "transformed_origin.png"), img)

    bbox_img = img.copy()
    patch_bbox = batch["patch_bbox"][bx, tx].cpu().numpy()
    hand_bbox = batch["hand_bbox"][bx, tx].cpu().numpy()
    joint_img = batch["joint_img"][bx, tx].cpu().numpy()
    joint_2d_valid = batch["joint_2d_valid"][bx, tx].cpu().numpy()
    cv2.rectangle(
        bbox_img,
        (int(patch_bbox[0]), int(patch_bbox[1])),
        (int(patch_bbox[2]), int(patch_bbox[3])),
        (0, 0, 255),
        2,
    )
    cv2.rectangle(
        bbox_img,
        (int(hand_bbox[0]), int(hand_bbox[1])),
        (int(hand_bbox[2]), int(hand_bbox[3])),
        (0, 255, 0),
        2,
    )
    bbox_img = _draw_joints(bbox_img, joint_img, joint_2d_valid)
    cv2.imwrite(os.path.join(output_dir, "bbox-joint_img.png"), bbox_img)

    patch_origin_crop = _crop_bbox(
        torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1),
        batch["patch_bbox"][bx, tx].cpu(),
    )
    joint_patch_origin = batch.get("joint_patch_origin", batch["joint_patch_bbox"])[
        bx, tx
    ].cpu().numpy()
    patch_origin_crop = _draw_joints(patch_origin_crop, joint_patch_origin, joint_2d_valid)
    cv2.imwrite(os.path.join(output_dir, "joint_patch_origin.png"), patch_origin_crop)

    patch = _tensor_image_to_bgr(batch["patches"][bx, tx].cpu())
    joint_patch_resized = batch["joint_patch_resized"][bx, tx].cpu().numpy()
    patch = _draw_joints(patch, joint_patch_resized, joint_2d_valid)
    cv2.imwrite(os.path.join(output_dir, "joint_patch_resized.png"), patch)

    hand_bbox_crop = _crop_bbox(
        torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1),
        batch["hand_bbox"][bx, tx].cpu(),
    )
    joint_hand_origin = batch.get("joint_hand_origin", batch["joint_hand_bbox"])[
        bx, tx
    ].cpu().numpy()
    hand_bbox_crop = _draw_joints(hand_bbox_crop, joint_hand_origin, joint_2d_valid)
    cv2.imwrite(os.path.join(output_dir, "joint_hand_origin.png"), hand_bbox_crop)

    has_intr = float(batch["has_intr"][bx, tx].item()) > 0.5
    has_3d = bool(torch.any(batch["joint_3d_valid"][bx, tx] > 0.5).item())
    if has_intr and has_3d:
        joint_cam = batch["joint_cam"][bx, tx].cpu()
        focal = batch["focal"][bx, tx].cpu()
        princpt = batch["princpt"][bx, tx].cpu()
        valid_mask = batch["joint_3d_valid"][bx, tx].cpu().numpy()
        reproj_u = focal[0] * joint_cam[:, 0] / joint_cam[:, 2] + princpt[0]
        reproj_v = focal[1] * joint_cam[:, 1] / joint_cam[:, 2] + princpt[1]
        reproj = torch.stack([reproj_u, reproj_v], dim=-1).numpy()
        reproj_img = _draw_joints(img, reproj, valid_mask)
        cv2.imwrite(os.path.join(output_dir, "joint_cam_reproj.png"), reproj_img)

    has_mano = float(batch["has_mano"][bx, tx].item()) > 0.5
    if has_intr and has_mano and smplx is not None:
        mano_layer = {
            "right": smplx.create("models", "mano", use_pca=False, is_rhand=True),
            "left": smplx.create("models", "mano", use_pca=False, is_rhand=False),
        }
        hand = batch["handedness"][bx]
        if batch["flip"][bx]:
            hand = "right" if hand == "left" else "left"
        mano_pose = batch["mano_pose"][bx, tx][None].cpu()
        mano_shape = batch["mano_shape"][bx, tx][None].cpu()
        focal = batch["focal"][bx, tx].cpu()
        princpt = batch["princpt"][bx, tx].cpu()
        with torch.inference_mode():
            mano_output = mano_layer[hand](
                betas=mano_shape,
                global_orient=mano_pose[:, :3],
                hand_pose=mano_pose[:, 3:],
                transl=torch.zeros((1, 3)),
            )
        mano_verts = mano_output.vertices[0] * 1e3
        joint_mano_cam = mano_output.joints[0] * 1e3
        joint_cam = batch["joint_cam"][bx, tx].cpu()
        mano_verts = mano_verts - joint_mano_cam[:1] + joint_cam[:1]
        reproj_u = focal[0] * mano_verts[:, 0] / mano_verts[:, 2] + princpt[0]
        reproj_v = focal[1] * mano_verts[:, 1] / mano_verts[:, 2] + princpt[1]
        mano_reproj = torch.stack([reproj_u, reproj_v], dim=-1).numpy()
        mano_img = img.copy()
        for vertex in mano_reproj:
            cv2.circle(mano_img, (int(vertex[0]), int(vertex[1])), 1, (255, 255, 0), -1)
        cv2.imwrite(os.path.join(output_dir, "mano.png"), mano_img)


def _resolve_urls(patterns: List[str]) -> List[str]:
    urls: List[str] = []
    for pattern in patterns:
        urls.extend(sorted(glob.glob(pattern)))
    if len(urls) == 0:
        raise FileNotFoundError(f"No tar files matched: {patterns}")
    return urls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual verification for V2 WebDataset")
    parser.add_argument("--urls", nargs="+", required=True, help="Tar path patterns")
    parser.add_argument("--output-dir", default="temp_verify_v2")
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shardshuffle", type=int, default=0)
    parser.add_argument("--shuffle-buffer", type=int, default=5000)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--augmentation", action="store_true")
    parser.add_argument("--patch-size", nargs=2, type=int, default=[256, 256])
    parser.add_argument("--patch-expansion", type=float, default=1.1)
    parser.add_argument("--scale-z-range", nargs=2, type=float, default=[0.9, 1.1])
    parser.add_argument("--scale-f-range", nargs=2, type=float, default=[0.8, 1.1])
    parser.add_argument("--persp-rot-max", type=float, default=float(torch.pi / 12))
    parser.add_argument("--default-data-source", default=None)
    parser.add_argument("--default-source-split", default="unknown")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    urls = _resolve_urls(args.urls)
    device = torch.device(args.device)

    loader = get_dataloader(
        urls,
        num_frames=args.num_frames,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shardshuffle=args.shardshuffle,
        shuffle_buffer=args.shuffle_buffer,
        default_data_source=args.default_data_source,
        default_source_split=args.default_source_split,
    )

    batch_origin = next(iter(loader))
    batch_idx = args.sample_index
    frame_idx = args.frame_index

    origin_dir = os.path.join(args.output_dir, "origin")
    verify_origin_sample(batch_origin, origin_dir, batch_idx, frame_idx)

    batch_for_preprocess = copy.deepcopy(batch_origin)
    batch_processed, trans_2d_mat = preprocess_batch(
        batch_for_preprocess,
        tuple(args.patch_size),
        args.patch_expansion,
        tuple(args.scale_z_range),
        tuple(args.scale_f_range),
        args.persp_rot_max,
        args.augmentation,
        device,
    )
    processed_dir = os.path.join(args.output_dir, "processed")
    verify_processed_sample(
        batch_processed, batch_origin, trans_2d_mat, processed_dir, batch_idx, frame_idx
    )

    print(f"Saved verification outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
