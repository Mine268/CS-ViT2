#!/usr/bin/env python3
"""
分析训练log，提取loss_joint_img的统计信息和异常值
"""
import re
import sys
import numpy as np
from pathlib import Path

def parse_log_line(line):
    """
    解析单行训练日志
    返回 (step, lr, total_loss, loss_joint_img) 或 None
    """
    # 匹配格式: 270/100000 lr=1.0800e-05 dropout=0.000 total=420682720.0000 ...
    pattern = r'(\d+)/\d+\s+lr=([\d.e+-]+)\s+dropout=([\d.]+)\s+total=([\d.e+-]+).*?loss_joint_img=([\d.e+-]+)'
    match = re.search(pattern, line)

    if match:
        step = int(match.group(1))
        lr = float(match.group(2))
        total_loss = float(match.group(4))
        loss_joint_img = float(match.group(5))
        return step, lr, total_loss, loss_joint_img
    return None

def main():
    if len(sys.argv) > 1:
        log_file = Path(sys.argv[1])
    else:
        log_file = Path("checkpoint/2026-02-10/21-51-13_stage1-dino_large/log.txt")

    if not log_file.exists():
        print(f"错误: {log_file} 不存在")
        return

    print(f"分析日志文件: {log_file}\n")

    steps = []
    lrs = []
    total_losses = []
    img_losses = []

    with open(log_file, 'r') as f:
        for line in f:
            result = parse_log_line(line)
            if result:
                step, lr, total_loss, loss_joint_img = result
                steps.append(step)
                lrs.append(lr)
                total_losses.append(total_loss)
                img_losses.append(loss_joint_img)

    if not img_losses:
        print("未找到任何训练日志数据")
        return

    steps = np.array(steps)
    img_losses = np.array(img_losses)
    total_losses = np.array(total_losses)

    # 统计信息
    print("=" * 70)
    print("重投影误差 (loss_joint_img) 统计")
    print("=" * 70)
    print(f"总样本数: {len(img_losses)}")
    print(f"步数范围: {steps.min()} - {steps.max()}")
    print(f"\n基本统计:")
    print(f"  均值:     {img_losses.mean():.4f}")
    print(f"  中位数:   {np.median(img_losses):.4f}")
    print(f"  标准差:   {img_losses.std():.4f}")
    print(f"  最小值:   {img_losses.min():.4f} (step {steps[img_losses.argmin()]})")
    print(f"  最大值:   {img_losses.max():.4e} (step {steps[img_losses.argmax()]})")

    # 百分位数
    print(f"\n百分位数:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  {p}%:      {np.percentile(img_losses, p):.4f}")

    # 异常值检测 (超过均值+3*标准差)
    mean = img_losses.mean()
    std = img_losses.std()
    threshold_high = mean + 3 * std
    threshold_extreme = 1000.0  # 超过1000的绝对异常值

    outliers = img_losses > threshold_high
    extreme_outliers = img_losses > threshold_extreme

    print(f"\n异常值检测:")
    print(f"  阈值 (μ+3σ):        {threshold_high:.4f}")
    print(f"  异常值数量:         {outliers.sum()} ({100*outliers.sum()/len(img_losses):.2f}%)")
    print(f"  极端异常值 (>1000): {extreme_outliers.sum()}")

    if extreme_outliers.any():
        print(f"\n极端异常值详情:")
        extreme_indices = np.where(extreme_outliers)[0]
        for idx in extreme_indices[:10]:  # 最多显示10个
            print(f"  Step {steps[idx]:6d}: loss_joint_img={img_losses[idx]:.4e}, total_loss={total_losses[idx]:.4e}")

    # 时间序列分析（分段统计）
    print(f"\n分段统计 (每5000步):")
    print(f"  {'步数范围':<20} {'样本数':>8} {'均值':>10} {'中位数':>10} {'最大值':>12} {'异常数':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10} {'-'*12} {'-'*8}")

    segment_size = 5000
    max_step = steps.max()
    for start in range(0, max_step, segment_size):
        end = start + segment_size
        mask = (steps >= start) & (steps < end)

        if mask.any():
            seg_losses = img_losses[mask]
            seg_outliers = (seg_losses > threshold_high).sum()

            print(f"  {start:6d} - {end:6d}   {len(seg_losses):6d}   "
                  f"{seg_losses.mean():8.2f}   {np.median(seg_losses):8.2f}   "
                  f"{seg_losses.max():10.2e}   {seg_outliers:6d}")

    # 建议的损失函数阈值
    print(f"\n=" * 70)
    print("鲁棒损失函数建议")
    print("=" * 70)

    # 过滤极端异常后的统计
    clean_losses = img_losses[img_losses < threshold_extreme]
    if len(clean_losses) > 0:
        clean_mean = clean_losses.mean()
        clean_std = clean_losses.std()
        clean_p95 = np.percentile(clean_losses, 95)
        clean_p99 = np.percentile(clean_losses, 99)

        print(f"过滤极端异常后 (<{threshold_extreme}):")
        print(f"  均值:     {clean_mean:.4f}")
        print(f"  标准差:   {clean_std:.4f}")
        print(f"  95分位:   {clean_p95:.4f}")
        print(f"  99分位:   {clean_p99:.4f}")

        # 建议的delta值 (Huber loss的阈值)
        delta_suggestion = clean_p95  # 使用95分位作为阈值
        print(f"\n建议的 Huber Loss delta (像素误差):")
        print(f"  delta = {delta_suggestion:.2f} 像素")
        print(f"  含义: 误差 < {delta_suggestion:.2f}px 时使用L2 (二次), 超过后使用L1 (线性)")
        print(f"  优点: {95}% 的正常样本使用L2获得精确优化, {5}% 的异常样本使用L1避免梯度爆炸")

        # 更保守的建议
        delta_conservative = clean_mean + 2 * clean_std
        print(f"\n更保守的建议 (μ+2σ):")
        print(f"  delta = {delta_conservative:.2f} 像素")

    print(f"\n=" * 70)

if __name__ == "__main__":
    main()
