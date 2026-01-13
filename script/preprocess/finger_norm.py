"""
==================================================
Statistics Result
==================================================
[Scale Factor (Middle Finger Len)]
  Mean: 86.993889
  Std : 8.327513

[Normalized Root Position]
  Mean: [-0.27450442  0.0307872  10.432839  ]
  Std : [1.4501408 1.2677814 2.2520287]
  Covariance Matrix:
[[ 2.1029084  -0.22723609 -0.19446157]
 [-0.22723609  1.6072695  -0.40470588]
 [-0.19446157 -0.40470588  5.071633  ]]

[Raw Root Position]
  Mean: [-24.414896    2.2917128 910.08026  ]
  Std : [118.75197 102.83254 219.12888]
==================================================
"""
import os
import glob
import torch
import numpy as np
import tqdm
import random

from src.data.dataloader import get_dataloader
from src.data.preprocess import preprocess_batch
from src.constant import *

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class OnlineCovariance:
    def __init__(self, n_features, device='cpu'):
        self.n = 0
        self.mean = torch.zeros(n_features, device=device, dtype=torch.float64)
        self.M2 = torch.zeros(n_features, n_features, device=device, dtype=torch.float64)

    def update(self, x):
        x = x.double()
        batch_n = x.shape[0]
        if batch_n == 0: return

        batch_mean = x.mean(dim=0)
        batch_delta = x - batch_mean
        batch_M2 = batch_delta.T @ batch_delta

        delta = batch_mean - self.mean
        new_n = self.n + batch_n

        self.mean += delta * (batch_n / new_n)
        self.M2 += batch_M2 + (delta.unsqueeze(1) @ delta.unsqueeze(0)) * (self.n * batch_n / new_n)
        self.n = new_n

    def finalize(self):
        if self.n < 2: return self.mean.float(), torch.zeros_like(self.M2).float()
        cov = self.M2 / (self.n - 1)
        return self.mean.float(), cov.float()

# 关节索引定义
i1 = HAND_JOINTS_ORDER.index("Middle_1")
i2 = HAND_JOINTS_ORDER.index("Middle_2")
i3 = HAND_JOINTS_ORDER.index("Middle_3")
i4 = HAND_JOINTS_ORDER.index("Middle_4")
# 根关节索引通常是 0，根据你的描述 joint_cam[:,:,0] 指的是根关节
i_root = 0

def get_skel_length(j3d, i, j):
    """计算两点间欧式距离"""
    d = j3d[..., i, :] - j3d[..., j, :]
    d = torch.sqrt(torch.sum(d ** 2, dim=-1))
    return d

def get_middle_length(j3d):
    """计算中指三段骨骼长度之和"""
    return (
        get_skel_length(j3d, i1, i2) + get_skel_length(j3d, i2, i3) + get_skel_length(j3d, i3, i4)
    )

def main():
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
        num_workers=8,
        prefetcher_factor=2,
        infinite=True,
    )

    device = torch.device("cuda:2")
    max_steps = 30000

    # 1. 统计归一化系数 (中指长度)
    scale_stats_calculator = OnlineCovariance(n_features=1, device=device)
    # 2. 统计归一化后的 Root 位置
    norm_root_stats_calculator = OnlineCovariance(n_features=3, device=device)
    # 3. 统计原始 Root 位置
    raw_root_stats_calculator = OnlineCovariance(n_features=3, device=device)

    print("Start collecting statistics...")

    for ix, batch in tqdm.tqdm(enumerate(train_loader), total=max_steps, ncols=80):
        if ix >= max_steps:
            break

        batch, _ = preprocess_batch(
            batch,
            [256, 256],
            2,
            [1.0, 1.0],
            [1.0, 1.0],
            0.0,
            "3",
            False,
            device
        )

        # [b,t,j,3]
        joint_cam = batch["joint_cam"]
        # [b,t,j]
        joint_valid = batch["joint_valid"]

        # -------------------------------------------------------------
        # Step 1: 扁平化处理，方便掩码操作
        # -------------------------------------------------------------
        B, T, J, C = joint_cam.shape
        flat_joint_cam = joint_cam.view(-1, J, C) # [N, J, 3]
        flat_joint_valid = joint_valid.view(-1, J) # [N, J]

        # -------------------------------------------------------------
        # Step 2: 生成有效性掩码 (Requirement 1)
        # 必须同时满足: Root, Middle1, Middle2, Middle3, Middle4 的 valid >= 0.5
        # -------------------------------------------------------------
        # 检查的关键关节索引列表
        check_indices = [i_root, i1, i2, i3, i4]

        # 提取这些关节的 valid 值 [N, 5]
        target_validity = flat_joint_valid[:, check_indices]

        # 判断每一行是否所有关键点都有效
        # (target_validity >= 0.5) 返回 bool 矩阵，all(dim=1) 确保所有关键点都满足
        valid_mask = (target_validity >= 0.5).all(dim=1) # [N]

        # 如果当前 batch 没有有效数据，跳过
        if not valid_mask.any():
            continue

        # -------------------------------------------------------------
        # Step 3: 根据掩码过滤数据
        # -------------------------------------------------------------
        valid_joints = flat_joint_cam[valid_mask] # [M, J, 3], M <= N

        # -------------------------------------------------------------
        # Step 4: 计算统计量
        # -------------------------------------------------------------

        # A. 计算归一化系数 (中指长度和) (Requirement 2)
        # scale shape: [M]
        scale = get_middle_length(valid_joints)

        # B. 获取根关节位置 (Requirement 5 target)
        # raw_root shape: [M, 3]
        raw_root = valid_joints[:, i_root, :]

        # C. 计算归一化后的根关节位置 (Requirement 4)
        # 注意广播机制: [M, 3] / [M, 1]
        normalized_root = raw_root / scale.unsqueeze(-1)

        # -------------------------------------------------------------
        # Step 5: 更新统计器
        # -------------------------------------------------------------

        # 更新 Scale 统计 (输入需要是 [M, 1])
        scale_stats_calculator.update(scale.unsqueeze(-1))

        # 更新 Normalized Root 统计
        norm_root_stats_calculator.update(normalized_root)

        # 更新 Raw Root 统计 (Requirement 5)
        raw_root_stats_calculator.update(raw_root)

    # -------------------------------------------------------------
    # Step 6: 输出结果
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("Statistics Result")
    print("="*50)

    # 1. Scale Stats
    scale_mean, scale_cov = scale_stats_calculator.finalize()
    scale_std = torch.sqrt(scale_cov.diag())
    print(f"[Scale Factor (Middle Finger Len)]")
    print(f"  Mean: {scale_mean.item():.6f}")
    print(f"  Std : {scale_std.item():.6f}")

    # 2. Normalized Root Stats
    nr_mean, nr_cov = norm_root_stats_calculator.finalize()
    nr_std = torch.sqrt(nr_cov.diag())
    print(f"\n[Normalized Root Position]")
    print(f"  Mean: {nr_mean.cpu().numpy()}")
    print(f"  Std : {nr_std.cpu().numpy()}")
    print(f"  Covariance Matrix:\n{nr_cov.cpu().numpy()}")

    # 3. Raw Root Stats
    rr_mean, rr_cov = raw_root_stats_calculator.finalize()
    rr_std = torch.sqrt(rr_cov.diag())
    print(f"\n[Raw Root Position]")
    print(f"  Mean: {rr_mean.cpu().numpy()}")
    print(f"  Std : {rr_std.cpu().numpy()}")

    print("="*50)

if __name__ == "__main__":
    main()