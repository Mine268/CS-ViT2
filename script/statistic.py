import os
import glob
import torch
import numpy as np
import copy
import kornia
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

    joint_rep_type = "3"
    ndim = JOINT_DIM_DICT[joint_rep_type] * MANO_JOINT_COUNT + MANO_SHAPE_DIM + 3

    device = torch.device("cuda:2")

    # 1. 初始化统计器
    stats_calculator = OnlineCovariance(n_features=ndim, device=device)

    # ix = 0 # tqdm 会自动计数，不需要手动维护 ix
    max_steps = 10000 # 既然是流式，最好设一个上限，比如 1万个 batch (即16万数据)

    print("Start collecting statistics...")

    # 使用 tqdm 包装 loader
    for ix, batch in tqdm.tqdm(enumerate(train_loader), total=max_steps, ncols=80):

        # 预处理
        batch, _ = preprocess_batch(
            batch,
            [256, 256],
            1.1,
            [0.9, 1.1], # 注意你原来这里写成了 1,1
            [0.9, 1.2],
            0.1,
            "3",
            True,
            device
        )

        # [b,t,ndim]
        mano_param = torch.cat(
            [batch["mano_pose"], batch["mano_shape"], batch["joint_cam"][:, :, 0]],
            dim=-1
        )

        # 2. 修复 Valid Mask 处理逻辑
        # [b,t] boolean or float
        mano_valid = batch["mano_valid"]

        # Flatten
        mano_param = mano_param.reshape(-1, ndim)
        mano_valid = mano_valid.reshape(-1) # 修正: 之前你误写成了 mano_param.reshape(-1)

        # Filter valid samples
        # 假设 valid 是 0/1 或者是 bool，这里做一个阈值判断比较稳妥
        valid_mask = mano_valid > 0.5
        valid_data = mano_param[valid_mask]

        # 3. 更新统计量
        stats_calculator.update(valid_data)

        if ix >= max_steps:
            break

    # 4. 获取最终结果并保存
    final_mean, final_cov = stats_calculator.finalize()

    print(f"Statistics Computed on {stats_calculator.n} samples.")
    print("Mean shape:", final_mean.shape)
    print("Cov shape:", final_cov.shape)

    save_path = "mano_stats.pt"
    torch.save({"mean": final_mean, "cov": final_cov}, save_path)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()