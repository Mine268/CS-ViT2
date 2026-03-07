"""
可视化 trans 热力图监督信号
运行: python tools/visualize_trans_heatmap.py
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import hydra

# 加载配置
config_path = "config/stage1-dino_large_no_norm.yaml"
with hydra.initialize_config_dir(config_dir="/data_1/renkaiwen/CS-ViT2/config", version_base=None):
    cfg = hydra.compose(config_name="stage1-dino_large_no_norm")

# 从 norm_stats.npz 加载统计信息
norm_stats = np.load("model/smplx_models/mano/norm_stats.npz")
mean = norm_stats["mean"]  # [-24.4, 2.29, 910.08]
std = norm_stats["std"]    # [118.75, 102.83, 219.13]

# 计算 Range (mean ± 5*std)
x_range = [mean[0] - 5*std[0], mean[0] + 5*std[0]]  # [-618, 569]
y_range = [mean[1] - 5*std[1], mean[1] + 5*std[1]]  # [-512, 516]
z_range = [mean[2] - 5*std[2], mean[2] + 5*std[2]]  # [-186, 2006]

# 热力图分辨率
resolution = cfg.MODEL.handec.heatmap_resolution  # [512, 512, 1024]

# 创建网格中心点
x_centers = np.linspace(x_range[0], x_range[1], resolution[0])
y_centers = np.linspace(y_range[0], y_range[1], resolution[1])
z_centers = np.linspace(z_range[0], z_range[1], resolution[2])

# 模拟几个典型的 GT trans 值（从训练数据统计）
# 基于 mean 和 std 生成典型值
typical_trans_list = [
    {"name": "mean", "values": mean},
    {"name": "mean-2std", "values": mean - 2*std},
    {"name": "mean-std", "values": mean - std},
    {"name": "mean+std", "values": mean + std},
    {"name": "mean+2std", "values": mean + 2*std},
    {"name": "extreme_low", "values": mean - 3*std},  # 极端低值
    {"name": "extreme_high", "values": mean + 3*std},  # 极端高值
]

# heatmap_sigma
heatmap_sigma = cfg.LOSS.heatmap_sigma  # 0.006

print("=" * 70)
print("Trans 热力图可视化")
print("=" * 70)
print(f"heatmap_sigma: {heatmap_sigma}")
print(f"X range: [{x_range[0]:.2f}, {x_range[1]:.2f}] mm, resolution: {resolution[0]}")
print(f"Y range: [{y_range[0]:.2f}, {y_range[1]:.2f}] mm, resolution: {resolution[1]}")
print(f"Z range: [{z_range[0]:.2f}, {z_range[1]:.2f}] mm, resolution: {resolution[2]}")
print(f"\n典型 GT trans 值:")
for t in typical_trans_list:
    print(f"  {t['name']}: [{t['values'][0]:7.2f}, {t['values'][1]:7.2f}, {t['values'][2]:7.2f}] mm")

# 计算热力图目标分布的函数
def compute_heatmap_target(gt_coord, centers, sigma):
    """
    计算热力图目标分布（与 compute_hm_ce 中一致）
    """
    squared_diff = (gt_coord - centers) ** 2
    target = np.exp(-squared_diff / (2 * sigma ** 2))
    target = target / (target.sum() + 1e-9)
    return target

# 为每个典型值计算热力图
fig, axes = plt.subplots(len(typical_trans_list), 3, figsize=(18, 3*len(typical_trans_list)))
if len(typical_trans_list) == 1:
    axes = axes.reshape(1, -1)

for row_idx, trans_data in enumerate(typical_trans_list):
    gt_x, gt_y, gt_z = trans_data["values"]
    name = trans_data["name"]
    
    # 计算三轴的热力图
    heatmap_x = compute_heatmap_target(gt_x, x_centers, heatmap_sigma * (x_range[1] - x_range[0]))
    heatmap_y = compute_heatmap_target(gt_y, y_centers, heatmap_sigma * (y_range[1] - y_range[0]))
    heatmap_z = compute_heatmap_target(gt_z, z_centers, heatmap_sigma * (z_range[1] - z_range[0]))
    
    # 绘制 X 轴
    ax = axes[row_idx, 0]
    ax.plot(x_centers, heatmap_x, 'b-', linewidth=1.5)
    ax.axvline(gt_x, color='r', linestyle='--', label=f'GT: {gt_x:.1f}mm')
    peak_x = x_centers[np.argmax(heatmap_x)]
    ax.axvline(peak_x, color='g', linestyle=':', alpha=0.7, label=f'Peak: {peak_x:.1f}mm')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Probability')
    ax.set_title(f'{name} - X axis\nRange: [{x_range[0]:.0f}, {x_range[1]:.0f}] mm')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 绘制 Y 轴
    ax = axes[row_idx, 1]
    ax.plot(y_centers, heatmap_y, 'b-', linewidth=1.5)
    ax.axvline(gt_y, color='r', linestyle='--', label=f'GT: {gt_y:.1f}mm')
    peak_y = y_centers[np.argmax(heatmap_y)]
    ax.axvline(peak_y, color='g', linestyle=':', alpha=0.7, label=f'Peak: {peak_y:.1f}mm')
    ax.set_xlabel('Y (mm)')
    ax.set_ylabel('Probability')
    ax.set_title(f'{name} - Y axis\nRange: [{y_range[0]:.0f}, {y_range[1]:.0f}] mm')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 绘制 Z 轴
    ax = axes[row_idx, 2]
    ax.plot(z_centers, heatmap_z, 'b-', linewidth=1.5)
    ax.axvline(gt_z, color='r', linestyle='--', label=f'GT: {gt_z:.1f}mm')
    peak_z = z_centers[np.argmax(heatmap_z)]
    ax.axvline(peak_z, color='g', linestyle=':', alpha=0.7, label=f'Peak: {peak_z:.1f}mm')
    ax.set_xlabel('Z (mm)')
    ax.set_ylabel('Probability')
    ax.set_title(f'{name} - Z axis\nRange: [{z_range[0]:.0f}, {z_range[1]:.0f}] mm')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 打印统计信息
    print(f"\n{name}:")
    print(f"  X: GT={gt_x:.2f}, Peak={peak_x:.2f}, max_prob={heatmap_x.max():.6f}")
    print(f"  Y: GT={gt_y:.2f}, Peak={peak_y:.2f}, max_prob={heatmap_y.max():.6f}")
    print(f"  Z: GT={gt_z:.2f}, Peak={peak_z:.2f}, max_prob={heatmap_z.max():.6f}")

plt.tight_layout()
plt.savefig('checkpoint/trans_heatmap_visualization.png', dpi=150, bbox_inches='tight')
print(f"\n✓ 图像已保存: checkpoint/trans_heatmap_visualization.png")

# 同时绘制一个放大的版本，只看 Z 轴的中间区域
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 选择 mean 值进行放大显示
gt_x, gt_y, gt_z = mean

# 计算热力图
heatmap_x = compute_heatmap_target(gt_x, x_centers, heatmap_sigma * (x_range[1] - x_range[0]))
heatmap_y = compute_heatmap_target(gt_y, y_centers, heatmap_sigma * (y_range[1] - y_range[0]))
heatmap_z = compute_heatmap_target(gt_z, z_centers, heatmap_sigma * (z_range[1] - z_range[0]))

# X 轴放大
ax = axes[0]
zoom_range = 100  # 左右各 100mm
mask = (x_centers >= gt_x - zoom_range) & (x_centers <= gt_x + zoom_range)
ax.plot(x_centers[mask], heatmap_x[mask], 'b-', linewidth=2)
ax.axvline(gt_x, color='r', linestyle='--', linewidth=2, label=f'GT: {gt_x:.1f}mm')
ax.set_xlabel('X (mm)')
ax.set_ylabel('Probability')
ax.set_title(f'X axis (zoomed ±{zoom_range}mm)\nSigma={heatmap_sigma}, Res={resolution[0]}')
ax.legend()
ax.grid(True, alpha=0.3)

# Y 轴放大
ax = axes[1]
mask = (y_centers >= gt_y - zoom_range) & (y_centers <= gt_y + zoom_range)
ax.plot(y_centers[mask], heatmap_y[mask], 'b-', linewidth=2)
ax.axvline(gt_y, color='r', linestyle='--', linewidth=2, label=f'GT: {gt_y:.1f}mm')
ax.set_xlabel('Y (mm)')
ax.set_ylabel('Probability')
ax.set_title(f'Y axis (zoomed ±{zoom_range}mm)\nSigma={heatmap_sigma}, Res={resolution[1]}')
ax.legend()
ax.grid(True, alpha=0.3)

# Z 轴放大
ax = axes[2]
zoom_range = 200  # Z 轴范围大，左右各 200mm
mask = (z_centers >= gt_z - zoom_range) & (z_centers <= gt_z + zoom_range)
ax.plot(z_centers[mask], heatmap_z[mask], 'b-', linewidth=2)
ax.axvline(gt_z, color='r', linestyle='--', linewidth=2, label=f'GT: {gt_z:.1f}mm')
ax.set_xlabel('Z (mm)')
ax.set_ylabel('Probability')
ax.set_title(f'Z axis (zoomed ±{zoom_range}mm)\nSigma={heatmap_sigma}, Res={resolution[2]}')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('checkpoint/trans_heatmap_zoomed.png', dpi=150, bbox_inches='tight')
print(f"✓ 放大图已保存: checkpoint/trans_heatmap_zoomed.png")

# 计算并显示关键统计信息
print("\n" + "=" * 70)
print("热力图统计信息")
print("=" * 70)
for axis_name, gt, centers, heatmap, range_total in [
    ("X", gt_x, x_centers, heatmap_x, x_range[1] - x_range[0]),
    ("Y", gt_y, y_centers, heatmap_y, y_range[1] - y_range[0]),
    ("Z", gt_z, z_centers, heatmap_z, z_range[1] - z_range[0])
]:
    # 90% 质量范围
    cumsum = np.cumsum(heatmap)
    idx_05 = np.searchsorted(cumsum, 0.05)
    idx_95 = np.searchsorted(cumsum, 0.95)
    
    # 半高宽
    half_max = heatmap.max() / 2
    above_half = np.where(heatmap >= half_max)[0]
    if len(above_half) > 0:
        fwhm = centers[above_half[-1]] - centers[above_half[0]]
    else:
        fwhm = 0
    
    print(f"\n{axis_name} axis:")
    print(f"  GT value: {gt:.2f} mm")
    print(f"  Peak probability: {heatmap.max():.6f}")
    print(f"  90% mass range: [{centers[idx_05]:.2f}, {centers[idx_95]:.2f}] mm "
          f"(width: {centers[idx_95] - centers[idx_05]:.2f} mm)")
    print(f"  FWHM: {fwhm:.2f} mm")
    print(f"  Step size: {range_total / len(centers):.3f} mm/bin")

print("\n" + "=" * 70)
print("不同 sigma 值的对比")
print("=" * 70)
for sigma in [0.006, 0.009, 0.012, 0.015]:
    print(f"\nsigma = {sigma}:")
    for axis_name, gt, centers, range_total in [
        ("X", gt_x, x_centers, x_range[1] - x_range[0]),
        ("Y", gt_y, y_centers, y_range[1] - y_range[0]),
        ("Z", gt_z, z_centers, z_range[1] - z_range[0])
    ]:
        heatmap = compute_heatmap_target(gt, centers, sigma * range_total)
        cumsum = np.cumsum(heatmap)
        idx_05 = np.searchsorted(cumsum, 0.05)
        idx_95 = np.searchsorted(cumsum, 0.95)
        print(f"  {axis_name}: 90% mass width = {centers[idx_95] - centers[idx_05]:.2f} mm "
              f"({idx_95 - idx_05} bins)")
