"""
分析各数据集训练集的深度分布
对比 InterHand2.6M、DexYCB、HO3D 的根关节 Z 深度分布
"""
import os
import sys
import glob
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataloader import get_dataloader


def collect_depth_samples(tar_pattern, dataset_name, max_samples=3000, batch_size=32):
    """
    从数据集中采集根关节 Z 深度样本
    
    Args:
        tar_pattern: tar 文件路径模式
        dataset_name: 数据集名称
        max_samples: 最大采集样本数
        batch_size: batch 大小
    
    Returns:
        depths: Z 深度数组 (mm)
        focal_lengths: 焦距数组
    """
    # 获取 tar 文件列表
    tar_files = sorted(glob.glob(tar_pattern))
    if len(tar_files) == 0:
        print(f"Warning: No files found for {tar_pattern}")
        return np.array([]), np.array([])
    
    print(f"[{dataset_name}] Found {len(tar_files)} tar files")
    
    # 创建 dataloader
    loader = get_dataloader(
        url=tar_files,
        num_frames=1,  # 单帧
        stride=1,
        batch_size=batch_size,
        num_workers=2,
        prefetcher_factor=1,
        infinite=False,  # 单次遍历
        seed=42,
    )
    
    depths = []
    focal_x_list = []
    focal_y_list = []
    
    sample_count = 0
    for batch in tqdm(loader, desc=f"Loading {dataset_name}"):
        if len(batch) == 0:
            continue
            
        # joint_cam: [B, T, 21, 3], T=1
        joint_cam = batch["joint_cam"]  # [B, 1, 21, 3]
        focal = batch["focal"]  # [B, 1, 2]
        
        # 提取根关节 Z 深度 (索引 0)
        root_z = joint_cam[:, 0, 0, 2].numpy()  # [B]
        focal_x = focal[:, 0, 0].numpy()  # [B]
        focal_y = focal[:, 0, 1].numpy()  # [B]
        
        depths.extend(root_z.tolist())
        focal_x_list.extend(focal_x.tolist())
        focal_y_list.extend(focal_y.tolist())
        
        sample_count += len(root_z)
        if sample_count >= max_samples:
            break
    
    return np.array(depths), np.array(focal_x_list), np.array(focal_y_list)


def analyze_distribution(depths, focal_x, focal_y, dataset_name):
    """分析深度分布并打印统计信息"""
    print(f"\n{'='*60}")
    print(f"[{dataset_name}] 深度分布统计")
    print(f"{'='*60}")
    print(f"样本数: {len(depths)}")
    print(f"\n深度统计 (mm):")
    print(f"  均值: {np.mean(depths):.1f}")
    print(f"  中位数: {np.median(depths):.1f}")
    print(f"  标准差: {np.std(depths):.1f}")
    print(f"  最小值: {np.min(depths):.1f}")
    print(f"  最大值: {np.max(depths):.1f}")
    print(f"  P5: {np.percentile(depths, 5):.1f}")
    print(f"  P25: {np.percentile(depths, 25):.1f}")
    print(f"  P75: {np.percentile(depths, 75):.1f}")
    print(f"  P95: {np.percentile(depths, 95):.1f}")
    
    print(f"\n焦距统计 (fx, fy):")
    print(f"  fx 均值: {np.mean(focal_x):.1f}, 标准差: {np.std(focal_x):.1f}")
    print(f"  fy 均值: {np.mean(focal_y):.1f}, 标准差: {np.std(focal_y):.1f}")
    print(f"  fx 范围: [{np.min(focal_x):.1f}, {np.max(focal_x):.1f}]")
    
    # 分箱统计
    bins = [
        (0, 300, "<300mm (极近)"),
        (300, 400, "300-400mm (很近)"),
        (400, 500, "400-500mm (近)"),
        (500, 600, "500-600mm (中近)"),
        (600, 700, "600-700mm (中)"),
        (700, 800, "700-800mm (中远)"),
        (800, 1000, "800-1000mm (远)"),
        (1000, 9999, ">1000mm (极远)"),
    ]
    
    print(f"\n深度区间分布:")
    for z_min, z_max, label in bins:
        count = np.sum((depths >= z_min) & (depths < z_max))
        percentage = count / len(depths) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {label:20s}: {count:5d} ({percentage:5.1f}%) {bar}")
    
    return {
        "name": dataset_name,
        "depths": depths,
        "focal_x": focal_x,
        "focal_y": focal_y,
        "mean": np.mean(depths),
        "median": np.median(depths),
        "std": np.std(depths),
        "min": np.min(depths),
        "max": np.max(depths),
    }


def main():
    # 数据集配置
    datasets = {
        "InterHand2.6M": "/mnt/qnap/data/datasets/webdatasets/InterHand2.6M/train/*",
        "DexYCB": "/mnt/qnap/data/datasets/webdatasets/DexYCB/s1/train/*",
        "HO3D": "/mnt/qnap/data/datasets/webdatasets/HO3D_v3/train/*",
    }
    
    max_samples = 3000  # 每个数据集采样数
    results = {}
    
    print("="*60)
    print("数据集深度分布分析")
    print(f"每个数据集采样约 {max_samples} 个样本")
    print("="*60)
    
    for name, pattern in datasets.items():
        try:
            depths, focal_x, focal_y = collect_depth_samples(
                pattern, name, max_samples=max_samples
            )
            if len(depths) > 0:
                result = analyze_distribution(depths, focal_x, focal_y, name)
                results[name] = result
        except Exception as e:
            print(f"Error processing {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # 对比分析
    if len(results) >= 2:
        print(f"\n{'='*60}")
        print("数据集对比")
        print(f"{'='*60}")
        
        print(f"\n{'数据集':<15} {'均值':>8} {'中位数':>8} {'标准差':>8} {'最小值':>8} {'最大值':>8}")
        print("-" * 70)
        for name in ["InterHand2.6M", "DexYCB", "HO3D"]:
            if name in results:
                r = results[name]
                print(f"{name:<15} {r['mean']:>8.1f} {r['median']:>8.1f} {r['std']:>8.1f} {r['min']:>8.1f} {r['max']:>8.1f}")
        
        # 700-800mm 区间占比对比
        print(f"\n700-800mm (模型表现最好的区间) 样本占比:")
        for name in ["InterHand2.6M", "DexYCB", "HO3D"]:
            if name in results:
                depths = results[name]["depths"]
                count_700_800 = np.sum((depths >= 700) & (depths < 800))
                percentage = count_700_800 / len(depths) * 100
                print(f"  {name:<15}: {percentage:5.1f}% ({count_700_800} / {len(depths)})")
        
        # 与 HO3D test 的对比
        print(f"\n与 HO3D test 集的对比:")
        print(f"  HO3D test 平均深度: 568mm")
        print(f"  HO3D train 平均深度: {results.get('HO3D', {}).get('mean', 'N/A')}")
        
        # 分析可能的原因
        print(f"\n{'='*60}")
        print("分析与推论")
        print(f"{'='*60}")
        
        ih_mean = results.get("InterHand2.6M", {}).get("mean", 0)
        dex_mean = results.get("DexYCB", {}).get("mean", 0)
        ho_train_mean = results.get("HO3D", {}).get("mean", 0)
        
        print(f"""
1. 模型在 700-800mm 表现最好，可能是因为:
   - 训练数据在该区间有较高的分布密度
   - 或者该区间是训练数据深度分布的中心

2. HO3D test 平均深度 568mm，但模型在该区间误差大 (+180mm):
   - 如果训练数据在 500-600mm 区间较少，模型可能缺乏该范围的训练信号
   - 或者训练数据的 500-600mm 样本主要来自特定数据集，其深度来源不同

3. 需要进一步分析:
   - 各数据集中 700-800mm 区间的样本占比
   - 深度来源（GT 还是估计？）
""")


if __name__ == "__main__":
    main()
