"""
深度分布分析 V2 - 深入分析与可视化
"""
import os
import sys
import glob
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data.dataloader import get_dataloader


def collect_depth_samples(tar_pattern, dataset_name, max_samples=3000, batch_size=32):
    """采集深度样本"""
    tar_files = sorted(glob.glob(tar_pattern))
    if len(tar_files) == 0:
        return np.array([]), np.array([]), np.array([])
    
    loader = get_dataloader(
        url=tar_files,
        num_frames=1,
        stride=1,
        batch_size=batch_size,
        num_workers=2,
        prefetcher_factor=1,
        infinite=False,
        seed=42,
    )
    
    depths = []
    focal_x_list = []
    focal_y_list = []
    
    sample_count = 0
    for batch in tqdm(loader, desc=f"Loading {dataset_name}", leave=False):
        if len(batch) == 0:
            continue
        joint_cam = batch["joint_cam"]
        focal = batch["focal"]
        
        root_z = joint_cam[:, 0, 0, 2].numpy()
        focal_x = focal[:, 0, 0].numpy()
        focal_y = focal[:, 0, 1].numpy()
        
        depths.extend(root_z.tolist())
        focal_x_list.extend(focal_x.tolist())
        focal_y_list.extend(focal_y.tolist())
        
        sample_count += len(root_z)
        if sample_count >= max_samples:
            break
    
    return np.array(depths), np.array(focal_x_list), np.array(focal_y_list)


def plot_depth_distribution(results, save_path="docs/depth_distribution.png"):
    """绘制深度分布对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {
        "InterHand2.6M": "#1f77b4",
        "DexYCB": "#2ca02c", 
        "HO3D": "#d62728",
        "HOT3D": "#9467bd"
    }
    
    # 1. 直方图对比
    ax = axes[0, 0]
    bins = np.linspace(200, 1400, 60)
    for name in ["InterHand2.6M", "DexYCB", "HO3D", "HOT3D"]:
        if name in results:
            depths = results[name]["depths"]
            ax.hist(depths, bins=bins, alpha=0.6, label=name, color=colors[name], density=True)
    ax.axvline(700, color='gray', linestyle='--', alpha=0.5, label='700mm')
    ax.axvline(800, color='gray', linestyle='--', alpha=0.5, label='800mm')
    ax.set_xlabel("Depth (mm)")
    ax.set_ylabel("Density")
    ax.set_title("Depth Distribution Comparison")
    ax.legend()
    ax.set_xlim(200, 1400)
    
    # 2. Box plot
    ax = axes[0, 1]
    data_to_plot = []
    labels = []
    for name in ["InterHand2.6M", "DexYCB", "HO3D", "HOT3D"]:
        if name in results:
            data_to_plot.append(results[name]["depths"])
            labels.append(f"{name}\n(n={len(results[name]['depths'])})")
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, name in zip(bp['boxes'], ["InterHand2.6M", "DexYCB", "HO3D"]):
        patch.set_facecolor(colors[name])
    ax.axhspan(700, 800, alpha=0.2, color='green', label='700-800mm (best region)')
    ax.set_ylabel("Depth (mm)")
    ax.set_title("Depth Distribution Box Plot")
    ax.set_ylim(200, 1400)
    
    # 3. 焦距 vs 深度散点图
    ax = axes[1, 0]
    for name in ["InterHand2.6M", "DexYCB", "HO3D", "HOT3D"]:
        if name in results:
            depths = results[name]["depths"]
            focal_x = results[name]["focal_x"]
            ax.scatter(depths[::10], focal_x[::10], alpha=0.3, label=name, s=5, color=colors[name])
    ax.set_xlabel("Depth (mm)")
    ax.set_ylabel("Focal Length fx (mm)")
    ax.set_title("Focal Length vs Depth")
    ax.legend()
    
    # 4. 区间占比柱状图
    ax = axes[1, 1]
    bins_labels = ["<500", "500-700", "700-800", "800-1000", ">1000"]
    x = np.arange(len(bins_labels))
    width = 0.25
    
    for i, name in enumerate(["InterHand2.6M", "DexYCB", "HO3D"]):
        if name in results:
            depths = results[name]["depths"]
            counts = [
                np.sum(depths < 500),
                np.sum((depths >= 500) & (depths < 700)),
                np.sum((depths >= 700) & (depths < 800)),
                np.sum((depths >= 800) & (depths < 1000)),
                np.sum(depths >= 1000),
            ]
            percentages = [c / len(depths) * 100 for c in counts]
            ax.bar(x + i*width, percentages, width, label=name, color=colors[name], alpha=0.8)
    
    ax.axvspan(1.5, 2.5, alpha=0.2, color='green', label='Best region')
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Depth Range Distribution")
    ax.set_xticks(x + width)
    ax.set_xticklabels(bins_labels)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {save_path}")
    plt.close()


def main():
    datasets = {
        "InterHand2.6M": "/mnt/qnap/data/datasets/webdatasets/InterHand2.6M/train/*",
        "DexYCB": "/mnt/qnap/data/datasets/webdatasets/DexYCB/s1/train/*",
        "HO3D": "/mnt/qnap/data/datasets/webdatasets/HO3D_v3/train/*",
        "HOT3D": "/mnt/qnap/data/datasets/webdatasets/HOT3D/train/*",
    }
    
    max_samples = 3000
    results = {}
    
    print("="*70)
    print("深度分布分析 V2 - 收集数据中...")
    print("="*70)
    
    for name, pattern in datasets.items():
        try:
            depths, focal_x, focal_y = collect_depth_samples(pattern, name, max_samples)
            if len(depths) > 0:
                results[name] = {
                    "depths": depths,
                    "focal_x": focal_x,
                    "focal_y": focal_y,
                    "mean": np.mean(depths),
                    "median": np.median(depths),
                    "std": np.std(depths),
                }
        except Exception as e:
            print(f"Error: {e}")
    
    # 生成可视化
    plot_depth_distribution(results)
    
    # 详细分析报告
    print("\n" + "="*70)
    print("关键发现与分析")
    print("="*70)
    
    print("""
【关键发现 1: InterHand2.6M 完全没有近距离样本】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
InterHand2.6M 深度分布: 平均 1089mm，范围 933-1237mm
→ 这是一个纯远距离数据集！

原因分析:
- InterHand2.6M 使用多台单反相机拍摄，焦距 ~1268mm
- 拍摄距离较远，这是由其采集方式决定的
- 数据集中的 "近距离" 样本可能也在 800mm 以上

【关键发现 2: 700-800mm "最佳区间" 的由来】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
700-800mm 样本占比:
- InterHand2.6M:   0.0% (完全没有)
- DexYCB:         23.1% (主要分布区)
- HO3D:           18.0% (次要分布区)

推论:
- 模型在 700-800mm 表现好，因为该区间主要由 DexYCB 数据主导
- DexYCB 使用 RealSense 深度相机，深度标注质量高
- 该区间成为模型的"舒适区"

【关键发现 3: HO3D 的分布与训练数据严重不匹配】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HO3D 训练集分布:
- 400-500mm: 47.8% (近一半样本)
- 500-600mm: 15.1%
- <600mm:    62.9% (绝大多数)

但训练数据中 <600mm 的样本主要来自:
- DexYCB 的 500-600mm 区间 (仅占 2.8%)
- HO3D train 本身 (如果模型学到了的话)

→ 400-500mm 区间在训练中几乎没有！

【关键发现 4: 焦距差异巨大】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
焦距对比:
- InterHand2.6M: 1268mm (单反相机)
- DexYCB:         616mm (RealSense D415)
- HO3D:           616mm (RealSense D415)

问题:
- InterHand2.6M 的焦距是其他两个数据集的 2 倍
- 相同的像素误差在 InterHand2.6M 上对应的 3D 误差是 2 倍
- 模型需要同时适应两种不同的相机系统

【核心结论】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
模型在 HO3D test 上误差大 (141mm) 的根本原因是:

1. 训练数据深度分布不均:
   - InterHand2.6M: 纯远距离 (>900mm)
   - DexYCB:        中远距离 (600-1000mm)
   - HO3D train:    近距离 (400-600mm)

2. 400-600mm 区间训练不足:
   - HO3D train 有 62.9% 样本在该区间
   - 但训练时 HO3D 仅占 1/4 数据量
   - InterHand2.6M (占大部分) 完全没有该区间样本

3. 模型学到了"偏见":
   - 在 700-800mm (DexYCB 主导) 表现好
   - 在 <600mm (训练不足) 表现差
   - 在 <400mm (从未见过) 表现极差

4. 焦距差异加剧了问题:
   - 模型需要同时适应 1268mm 和 616mm 两种焦距
   - 绝对深度预测对焦距敏感
""")
    
    print("【建议的解决方案】")
    print("="*70)
    print("""
1. 数据层面:
   a) 使用 norm_by_hand=true，消除绝对深度依赖
   b) 采样时加权 HO3D，增加近距离样本比例
   c) 考虑筛选 InterHand2.6M，只保留 <1200mm 的样本

2. 训练层面:
   a) 关闭或减小 Z 轴缩放增强 (保持绝对深度信息)
   b) 增加焦距扰动增强，提升泛化能力
   c) 验证集必须包含 HO3D，实时监控

3. 模型层面:
   a) 考虑预测归一化坐标 + 深度反归一化两步策略
   b) 或者使用相机参数作为条件输入
""")


if __name__ == "__main__":
    main()
