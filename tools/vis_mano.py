"""
Visualize MANO hand model in 3 states:
1. Initial (zero pose, zero shape)
2. Shape from dataset sample
3. Pose from dataset sample

View: looking from +Y toward -Y (top-down, palm facing camera)

Outputs: tools/mano_vis.png
"""
import torch
import numpy as np
import smplx
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import webdataset as wds

# Chinese Song font (Noto Serif CJK SC)
SONG_FONT = font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")

MANO_ROOT = "model/smplx_models"

JOINT_LINKS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

J_REGRESSOR_PATH = "model/smplx_models/mano/sh_joint_regressor.npy"


def get_mano_output(pose, shape):
    """Run MANO forward pass and return joints + vertices in mm."""
    mano_layer = smplx.create(MANO_ROOT, "mano", is_rhand=True, use_pca=False)
    J_regressor = torch.from_numpy(np.load(J_REGRESSOR_PATH)).float()

    output = mano_layer(
        betas=shape,
        global_orient=pose[:, :3],
        hand_pose=pose[:, 3:],
        transl=torch.zeros(1, 3),
    )

    verts = output.vertices[0].detach().numpy() * 1000  # m -> mm
    faces = mano_layer.faces.astype(np.int64)

    joints = torch.einsum("vd,jv->jd", output.vertices[0], J_regressor)
    joints = joints.detach().numpy() * 1000  # m -> mm

    return joints, verts, faces


def load_sample_from_dataset():
    """Load a single sample's mano_pose and mano_shape from the dataset."""
    ds = wds.WebDataset(
        "/mnt/qnap/data/datasets/webdatasets/InterHand2.6M/val/000000.tar",
        shardshuffle=False,
    ).decode()
    for sample in ds:
        pose = sample.get("mano_pose.npy")
        shape = sample.get("mano_shape.npy")
        if pose is not None and shape is not None:
            # pose: [T, 48], shape: [T, 10] - take first frame
            if pose.ndim == 2:
                pose = pose[0]
                shape = shape[0]
            return pose, shape
    raise RuntimeError("No valid sample found")


def plot_hand(ax, joints, verts, faces, title):
    """Plot mesh (semi-transparent) + joints + skeleton.
    View: from +Y toward -Y. Plot X on horizontal, Z on vertical."""
    # Mesh
    mesh_faces = [verts[f] for f in faces]
    poly = Poly3DCollection(
        mesh_faces, alpha=0.15, facecolor="skyblue",
        edgecolor="gray", linewidth=0.1,
    )
    ax.add_collection3d(poly)

    # Joints
    ax.scatter(
        joints[:, 0], joints[:, 1], joints[:, 2],
        c="red", s=20, zorder=5, depthshade=False,
    )

    # Skeleton
    for i, j in JOINT_LINKS:
        ax.plot(
            [joints[i, 0], joints[j, 0]],
            [joints[i, 1], joints[j, 1]],
            [joints[i, 2], joints[j, 2]],
            c="darkred", linewidth=1.5, zorder=4,
        )

    # Axis limits (equal aspect)
    all_pts = verts
    center = all_pts.mean(axis=0)
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.2
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    ax.set_xlabel("X (mm)", fontsize=8)
    ax.set_ylabel("Y (mm)", fontsize=8)
    ax.set_zlabel("Z (mm)", fontsize=8)
    pass  # title set externally if needed
    # View from +Y looking toward -Y: elev=0, azim=0 shows X-Z plane
    # matplotlib 3d: elev=0 azim=270 gives +Y looking down
    ax.view_init(elev=0, azim=270)
    ax.tick_params(labelsize=6)


def main():
    # Load real data from dataset
    sample_pose, sample_shape = load_sample_from_dataset()
    print(f"Loaded sample pose shape: {sample_pose.shape}, sample shape shape: {sample_shape.shape}")

    pose_zero = torch.zeros(1, 48)
    shape_zero = torch.zeros(1, 10)

    # --- Case 1: Initial state ---
    j1, v1, f1 = get_mano_output(pose_zero, shape_zero)

    # --- Case 2: Dataset shape, zero pose ---
    shape_data = torch.from_numpy(sample_shape).float().unsqueeze(0)
    j2, v2, f2 = get_mano_output(pose_zero, shape_data)

    # --- Case 3: Dataset pose, zero shape ---
    pose_data = torch.from_numpy(sample_pose).float().unsqueeze(0)
    j3, v3, f3 = get_mano_output(pose_data, shape_zero)

    # --- Plot each separately ---
    cases = [
        (j1, v1, f1, "初始状态", "tools/mano_vis_initial.png"),
        (j2, v2, f2, "形状参数", "tools/mano_vis_shape.png"),
        (j3, v3, f3, "姿态参数", "tools/mano_vis_pose.png"),
    ]
    for joints, verts, faces, title, out_path in cases:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        plot_hand(ax, joints, verts, faces, title)
        plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
        plt.close()
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
