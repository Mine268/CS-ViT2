import os
import glob
import torch
import numpy as np
import copy
import kornia
import tqdm

from src.data.dataloader import get_dataloader
from src.data.preprocess import preprocess_batch


def verify_origin_data(batch, output_dir: str, bx: int = 0, tx: int = 0):
    import cv2
    import smplx

    os.makedirs(output_dir, exist_ok=True)

    # handedness
    img = cv2.cvtColor(batch["imgs"][bx][tx].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
    img1 = img.copy()
    cv2.putText(
        img1,
        batch["handedness"][bx],
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.imwrite(f"{output_dir}/origin.png", img1)

    # hand_bbox, hand_img
    img2 = img.copy()
    hand_bbox = batch["hand_bbox"][bx, tx]
    joint_img = batch["joint_img"][bx, tx]
    cv2.rectangle(
        img2,
        (int(hand_bbox[0]), int(hand_bbox[1])),
        (int(hand_bbox[2]), int(hand_bbox[3])),
        (0, 0, 255),
        3
    )
    for i, jnt in enumerate(joint_img):
        cv2.circle(
            img2,
            (int(jnt[0]), int(jnt[1])),
            3, (255, 50, 50), -1
        )
        cv2.putText(
            img2,
            str(i),
            (int(jnt[0]), int(jnt[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(f"{output_dir}/hand_bbox-joint_img.png", img2)

    # joint_hand_bbox
    hand_bbox = torch.round(batch["hand_bbox"][bx, tx:tx + 1])
    joint_hand_bbox = batch["joint_hand_bbox"][bx, tx]
    xm, ym, xM, yM = torch.split(hand_bbox, 1, dim=-1)
    hand_bbox_k = torch.stack(
        [
            torch.cat([xm, ym], dim=-1),
            torch.cat([xM, ym], dim=-1),
            torch.cat([xM, yM], dim=-1),
            torch.cat([xm, yM], dim=-1),
        ],
        dim=1
    )
    bbox_size = hand_bbox[0, 2:] - hand_bbox[0, :2]
    img_hand_bbox = kornia.geometry.transform.crop_and_resize(
        batch["imgs"][bx][tx:tx + 1].float() / 255,
        hand_bbox_k,
        (int(bbox_size[1]), int(bbox_size[0])),
    )
    img_hand_bbox = cv2.cvtColor(
        (img_hand_bbox[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )
    for i, jnt in enumerate(joint_hand_bbox):
        cv2.circle(
            img_hand_bbox,
            (int(jnt[0]), int(jnt[1])),
            3, (255, 50, 50), -1
        )
        cv2.putText(
            img_hand_bbox,
            str(i),
            (int(jnt[0]), int(jnt[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(f"{output_dir}/joint_hand_bbox.png", img_hand_bbox)

    # joint_cam
    joint_cam = batch["joint_cam"][bx, tx]
    focal = batch["focal"][bx, tx]
    princpt = batch["princpt"][bx, tx]
    joint_reproj_u = focal[0] * joint_cam[:, 0] / joint_cam[:, 2] + princpt[0]
    joint_reproj_v = focal[1] * joint_cam[:, 1] / joint_cam[:, 2] + princpt[1]
    joint_reproj = torch.stack([joint_reproj_u, joint_reproj_v], dim=-1)
    img3 = img.copy()
    for i, jnt in enumerate(joint_reproj):
        cv2.circle(
            img3,
            (int(jnt[0]), int(jnt[1])),
            3, (255, 50, 50), -1
        )
        cv2.putText(
            img3,
            str(i),
            (int(jnt[0]), int(jnt[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(f"{output_dir}/joint_cam.png", img3)

    # joint rel
    joint_cam = batch["joint_cam"][bx, tx]
    joint_rel = batch["joint_rel"][bx, tx]
    assert torch.allclose(
        joint_rel, joint_cam - joint_cam[:1], atol=1e-3
    ), "joint_rel not consistent with joint_cam"
    print("joint_rel is consistent with joint_cam")

    # joint valid
    joint_valid = batch["joint_valid"][bx, tx]
    print(f"joint_valid shape={joint_valid.shape}")
    print(f"joint_valid={joint_valid}")

    # mano_shape, mano_pose
    mano_layer = {
        "right": smplx.create("model/smplx_models", "mano", use_pca=False, is_rhand=True),
        "left": smplx.create("model/smplx_models", "mano", use_pca=False, is_rhand=False),
    }
    mano_pose = batch["mano_pose"][bx, tx][None, ...]  # [B,48]
    mano_shape = batch["mano_shape"][bx, tx][None, ...]  # [B,10]
    handedness = batch["handedness"][bx]
    focal = batch["focal"][bx, tx]
    princpt = batch["princpt"][bx, tx]
    img4 = img.copy()
    with torch.inference_mode():
        mano_output = mano_layer[handedness](
            betas=mano_shape,
            global_orient=mano_pose[:, :3],
            hand_pose=mano_pose[:, 3:],
            transl=torch.zeros((1, 3))
        )
    mano_verts = mano_output.vertices[0] * 1e3
    joint_mano_cam = mano_output.joints[0] * 1e3
    joint_cam = batch["joint_cam"][bx, tx]
    mano_verts = mano_verts - joint_mano_cam[:1] + joint_cam[:1]  # 使用gt根关节对齐mano
    mano_verts_reproj_u = focal[0] * mano_verts[:, 0] / mano_verts[:, 2] + princpt[0]
    mano_verts_reproj_v = focal[1] * mano_verts[:, 1] / mano_verts[:, 2] + princpt[1]
    mano_verts_reproj = torch.stack([mano_verts_reproj_u, mano_verts_reproj_v], dim=-1)
    for vt in mano_verts_reproj:
        cv2.circle(
            img4,
            (int(vt[0]), int(vt[1])),
            1, (255, 255, 0), -1
        )
    cv2.imwrite(f"{output_dir}/mano.png", img4)

import kornia
def crop_bbox_kornia(
    image: torch.Tensor,
    bbox: torch.Tensor,
    padding_mode: str = 'zeros',
    align_corners: bool = True
) -> torch.Tensor:
    """
    使用 Kornia 对图像按 bbox 裁剪，输出尺寸为 bbox 的整数宽高，越界部分填充为 0。

    Args:
        image (torch.Tensor): 输入图像，shape (C, H, W)
        bbox (torch.Tensor): 边界框，shape (4,)，格式 [x1, y1, x2, y2]
        padding_mode (str): 填充模式，'zeros', 'border', 'reflection'
        align_corners (bool): 插值对齐方式

    Returns:
        torch.Tensor: 裁剪结果，shape (C, h_out, w_out)，其中
                      h_out = max(1, round(y2 - y1))
                      w_out = max(1, round(x2 - x1))
    """
    if image.dim() != 3:
        raise ValueError(f"Expected image shape (C, H, W), got {image.shape}")
    if bbox.shape != (4,):
        raise ValueError(f"Expected bbox shape (4,), got {bbox.shape}")

    # 确保 bbox 是 float
    bbox = bbox.float()

    # 计算目标输出尺寸（取整，至少为1）
    x1, y1, x2, y2 = bbox
    w_out = max(1, int(torch.round(x2 - x1).item()))
    h_out = max(1, int(torch.round(y2 - y1).item()))

    # 构造源 box（原始 bbox 的四个角点）
    src_vertices = torch.tensor([
        [x1.item(), y1.item()],
        [x2.item(), y1.item()],
        [x2.item(), y2.item()],
        [x1.item(), y2.item()]
    ], device=image.device, dtype=image.dtype).unsqueeze(0)  # (1, 4, 2)

    # 添加 batch 维度
    image_batch = image.unsqueeze(0)  # (1, C, H, W)

    cropped = kornia.geometry.transform.crop_and_resize(
        image_batch,
        src_vertices,
        size=(h_out, w_out),          # 注意：size 是 (height, width)
    )  # 输出: (1, C, h_out, w_out)

    return cropped.squeeze(0)  # (C, h_out, w_out)


def verify_batch(
    batch,
    trans_2d_mat: torch.Tensor,
    output_dir: str,
    source_prefix: str,
    bx: int = 0,
    tx: int = 0,
    origin_batch = None,
):
    import cv2
    import smplx

    trans_2d_mat = trans_2d_mat[bx, tx].cpu().numpy()

    # origin image
    if origin_batch is None:
        print("\nimgs_path=" + f"{source_prefix}/" + batch["imgs_path"][bx][tx])
        img = cv2.imread(f"{source_prefix}/" + batch["imgs_path"][bx][tx])
    else:
        print("\nread image from origin batch")
        img = cv2.cvtColor(origin_batch["imgs"][bx][tx].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
    img = cv2.warpPerspective(img, trans_2d_mat, (img.shape[1], img.shape[0]))
    if batch["flip"][bx]:
        img = img[:, ::-1].copy()
        cv2.putText(
            img,
            "flipped",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
    cv2.imwrite(f"{output_dir}/origin.png", img)

    # patches
    patch = batch["patches"][bx, tx].cpu().permute(1, 2, 0).numpy() * 255
    patch = patch.astype(np.uint8)
    cv2.imwrite(f"{output_dir}/patch.png", patch)

    # hand/patch_bbox, joint_img
    img2 = img.copy()
    patch_bbox = batch["patch_bbox"][bx, tx].cpu().numpy()
    hand_bbox = batch["hand_bbox"][bx, tx].cpu().numpy()
    joint_img = batch["joint_img"][bx, tx].cpu().numpy()
    cv2.rectangle(
        img2,
        (int(patch_bbox[0]), int(patch_bbox[1])),
        (int(patch_bbox[2]), int(patch_bbox[3])),
        (0, 0, 255), 2
    )
    cv2.rectangle(
        img2,
        (int(hand_bbox[0]), int(hand_bbox[1])),
        (int(hand_bbox[2]), int(hand_bbox[3])),
        (0, 255, 0), 2
    )
    for i, jnt in enumerate(joint_img):
        cv2.circle(
            img2,
            (int(jnt[0]), int(jnt[1])),
            3, (255, 50, 50), -1
        )
        cv2.putText(
            img2,
            str(i),
            (int(jnt[0]), int(jnt[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(f"{output_dir}/bbox-joint_img.png", img2)

    # joint_patch_bbox, joint_hand_bbox
    img3 = torch.from_numpy(img.copy()).float().permute(2, 0, 1) / 255
    joint_patch_bbox = batch["joint_patch_bbox"][bx, tx].cpu()
    joint_hand_bbox = batch["joint_hand_bbox"][bx, tx].cpu()
    patch_bbox = batch["patch_bbox"][bx, tx].cpu()
    hand_bbox = batch["hand_bbox"][bx, tx].cpu()
    img_patch_bbox = crop_bbox_kornia(img3, patch_bbox)
    img_hand_bbox = crop_bbox_kornia(img3, hand_bbox)
    img3 = img3.permute(1, 2, 0).numpy()
    img_patch_bbox = (img_patch_bbox.permute(1, 2, 0).contiguous().numpy() * 255).astype(np.uint8)
    img_hand_bbox = (img_hand_bbox.permute(1, 2, 0).contiguous().numpy() * 255).astype(np.uint8)
    for i, _ in enumerate(joint_patch_bbox):
        cv2.circle(
            img_patch_bbox,
            (int(joint_patch_bbox[i, 0]), int(joint_patch_bbox[i, 1])),
            3, (255, 50, 50), -1
        )
        cv2.putText(
            img_patch_bbox,
            str(i),
            (int(joint_patch_bbox[i, 0]), int(joint_patch_bbox[i, 1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.circle(
            img_hand_bbox,
            (int(joint_hand_bbox[i, 0]), int(joint_hand_bbox[i, 1])),
            3, (255, 50, 50), -1
        )
        cv2.putText(
            img_hand_bbox,
            str(i),
            (int(joint_hand_bbox[i, 0]), int(joint_hand_bbox[i, 1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(f"{output_dir}/joint_patch_bbox.png", img_patch_bbox)
    cv2.imwrite(f"{output_dir}/joint_hand_bbox.png", img_hand_bbox)

    # joint_cam
    img4 = img.copy()
    joint_cam = batch["joint_cam"][bx, tx].cpu()
    focal = batch["focal"][bx, tx].cpu()
    princpt = batch["princpt"][bx, tx].cpu()
    joint_reproj_u = focal[0] * joint_cam[:, 0] / joint_cam[:, 2] + princpt[0]
    joint_reproj_v = focal[1] * joint_cam[:, 1] / joint_cam[:, 2] + princpt[1]
    joint_reproj = torch.stack([joint_reproj_u, joint_reproj_v], dim=-1)
    for i, jnt in enumerate(joint_reproj):
        cv2.circle(
            img4,
            (int(jnt[0]), int(jnt[1])),
            3, (255, 50, 50), -1
        )
        cv2.putText(
            img4,
            str(i),
            (int(jnt[0]), int(jnt[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(f"{output_dir}/joint_cam.png", img4)

    # joint_rel
    joint_cam = batch["joint_cam"][bx, tx].cpu()
    joint_rel = batch["joint_rel"][bx, tx].cpu()
    assert torch.allclose(
        joint_rel, joint_cam - joint_cam[:1]
    ), "joint_rel not consistent with joint_cam"
    print("joint_rel is consistent with joint_cam")

    # joint valid
    joint_valid = batch["joint_valid"][bx, tx].cpu()
    print(f"joint_valid shape={joint_valid.shape}")
    print(f"joint_valid={joint_valid}")

    # mano_shape, mano_pose
    mano_layer = {
        "right": smplx.create("model/smplx_models", "mano", use_pca=False, is_rhand=True),
        "left": smplx.create("model/smplx_models", "mano", use_pca=False, is_rhand=False),
    }
    mano_pose = batch["mano_pose"][bx, tx][None, ...].cpu()  # [B,48]
    mano_shape = batch["mano_shape"][bx, tx][None, ...].cpu()  # [B,10]
    flip = batch["flip"][bx]
    focal = batch["focal"][bx, tx].cpu()
    princpt = batch["princpt"][bx, tx].cpu()
    img5 = img.copy()
    with torch.inference_mode():
        mano_output = mano_layer["right"](
            betas=mano_shape,
            global_orient=mano_pose[:, :3],
            hand_pose=mano_pose[:, 3:],
            transl=torch.zeros((1, 3))
        )
    mano_verts = mano_output.vertices[0] * 1e3
    joint_mano_cam = mano_output.joints[0] * 1e3
    joint_cam = batch["joint_cam"][bx, tx].cpu()
    mano_verts = mano_verts - joint_mano_cam[:1] + joint_cam[:1]  # 使用gt根关节对齐mano
    mano_verts_reproj_u = focal[0] * mano_verts[:, 0] / mano_verts[:, 2] + princpt[0]
    mano_verts_reproj_v = focal[1] * mano_verts[:, 1] / mano_verts[:, 2] + princpt[1]
    mano_verts_reproj = torch.stack([mano_verts_reproj_u, mano_verts_reproj_v], dim=-1)
    for vt in mano_verts_reproj:
        cv2.circle(
            img5,
            (int(vt[0]), int(vt[1])),
            1, (255, 255, 0), -1
        )
    mano_valid = batch["mano_valid"][bx, tx].cpu()
    print(f"mano_valid={mano_valid}")
    cv2.imwrite(f"{output_dir}/mano.png", img5)


def test_dataloader1():
    train_urls = [
        "/mnt/qnap/data/datasets/webdatasets/InterHand2.6M/train/*",
        "/mnt/qnap/data/datasets/webdatasets/DexYCB/s1/train/*",
        "/mnt/qnap/data/datasets/webdatasets/HO3D_v3/train/*",
    ]
    train_sources = []
    for src in train_urls:
        matched_files = glob.glob(src)
        matched_files = sorted(matched_files)
        train_sources.extend(matched_files)
    train_loader = get_dataloader(
        url=train_sources,
        num_frames=1,
        stride=1,
        batch_size=16,
        num_workers=4,
        prefetcher_factor=2,
        infinite=True,
    )

    data_iter = iter(train_loader)
    for _ in tqdm.tqdm(range(10000), ncols=50):
        batch = next(data_iter)
        batch, _ = preprocess_batch(
            batch,
            [256, 256],
            1.1,
            [0.9, 1,1],
            [0.9, 1.2],
            0.1,
            "6d",
            True,
            torch.device("cuda:2")
        )

        assert not batch["mano_pose"].isnan().any().cpu().item()


def test_dataloader2():
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    loader = get_dataloader(
        glob.glob(
            # "/mnt/qnap/data/datasets/webdatasets/InterHand2.6M/train/*.tar",
            # "/mnt/qnap/data/datasets/webdatasets/DexYCB/s1/train/*.tar"
            # "/mnt/qnap/data/datasets/webdatasets/HO3D_v3/evaluation/*.tar"
            "/mnt/qnap/data/datasets/webdatasets/HOT3D/train/*",
        ),
        num_frames=1,
        stride=1,
        batch_size=32,
        num_workers=2,
        prefetcher_factor=2,
        infinite=True,
    )

    import os
    batch = None
    bx, tx = 10, 0
    x = 0
    for i, batch_ in enumerate(tqdm.tqdm(loader, ncols=70)):
        batch = copy.deepcopy(batch_)
        # 验证数据规整的正确性
        batch2, trans_2d_mat, _ = preprocess_batch(
            batch,
            [256, 256],
            1.1,
            [0.9, 1.1],
            [0.8, 1.1],
            torch.pi / 6,
            "3",
            True,
            torch.device("cuda:0"),
            None,
            True,
        )
        os.makedirs(f"tests/temp_processed_{i}", exist_ok=True)
        verify_batch(
            batch2,
            trans_2d_mat,
            f"tests/temp_processed_{i}",
            "/mnt/qnap/data/datasets/InterHand2.6M_5fps_batch1/images/train",
            bx,
            tx,
            batch,
        )
        if not torch.all(batch2["joint_valid"][10, 0] > 0.5).item():
            print(f">> False in joint_valid, #iter={i}: {batch2['joint_valid'][10, 0]}")
        x += 1
        if x > 10:
            break

    # 直接验证数据满足一致性
    verify_origin_data(batch_, "tests/temp_origin", bx, tx)


def test_dataloader3():
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    loader = get_dataloader(
        glob.glob(
            # "/mnt/qnap/data/datasets/webdatasets/InterHand2.6M/train/*",
            # "/mnt/qnap/data/datasets/webdatasets/DexYCB/s1/train/*",
            # "/mnt/qnap/data/datasets/webdatasets/HO3D_v3/train/*",
            "/mnt/qnap/data/datasets/webdatasets/HOT3D/train/*",
        ),
        num_frames=1,
        stride=1,
        batch_size=42*4,
        num_workers=2,
        prefetcher_factor=2,
        infinite=True,
    )

    for i, batch_ in enumerate(tqdm.tqdm(loader, ncols=70)):
        batch = copy.deepcopy(batch_)
        # 验证数据规整的正确性
        batch2, trans_2d_mat = preprocess_batch(
            batch,
            [256, 256],
            1.1,
            [0.9, 1.1],
            [0.8, 1.1],
            torch.pi / 12,
            "3",
            True,
            torch.device("cuda:0")
        )
        print(type(batch2))


if __name__ == "__main__":
    test_dataloader2()
    # test_dataloader3()
