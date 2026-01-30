"""
测试数据增强缓存机制的性能
"""
import sys
from pathlib import Path
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.data.preprocess import get_or_create_augmentation, clear_augmentation_cache


def test_augmentation_cache_performance():
    """测试缓存机制的性能提升"""
    print("=" * 60)
    print("测试数据增强缓存性能")
    print("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 测试配置
    aug_config = {
        'color_jitter': {
            'enabled': True,
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.1,
            'hue': 0.0,
            'p': 0.5
        },
        'gaussian_noise': {
            'enabled': True,
            'mean': 0.0,
            'std': 0.03,
            'p': 0.5
        },
        'gaussian_blur': {
            'enabled': True,
            'kernel_size': [5, 5],
            'sigma': [0.3, 1.0],
            'p': 0.15
        },
    }

    # 测试1: 无缓存情况（模拟重复创建）
    print("\n[测试1] 无缓存 - 重复创建100次")
    clear_augmentation_cache()

    from src.data.preprocess import PixelLevelAugmentation

    start_time = time.time()
    for _ in range(100):
        aug = PixelLevelAugmentation(aug_config).to(device)
    no_cache_time = time.time() - start_time
    print(f"  耗时: {no_cache_time:.3f}秒")
    print(f"  平均每次: {no_cache_time/100*1000:.2f}毫秒")

    # 测试2: 使用缓存
    print("\n[测试2] 使用缓存 - 获取100次")
    clear_augmentation_cache()

    start_time = time.time()
    for _ in range(100):
        aug = get_or_create_augmentation(aug_config, device)
    cached_time = time.time() - start_time
    print(f"  耗时: {cached_time:.3f}秒")
    print(f"  平均每次: {cached_time/100*1000:.2f}毫秒")

    # 性能提升
    speedup = no_cache_time / cached_time
    print(f"\n性能提升: {speedup:.1f}x 倍速")
    print(f"时间节省: {(no_cache_time - cached_time):.3f}秒 ({(1 - cached_time/no_cache_time)*100:.1f}%)")

    # 测试3: 实际训练场景模拟
    print("\n" + "=" * 60)
    print("模拟实际训练场景")
    print("=" * 60)

    num_iterations = 1000  # 1000个训练步
    batch_size = 32

    print(f"\n训练设置: {num_iterations}步, batch_size={batch_size}")

    # 无缓存情况
    print("\n[场景1] 无缓存 - 每个batch创建新增强器")
    clear_augmentation_cache()

    start_time = time.time()
    for step in range(num_iterations):
        # 每个batch都创建新实例
        aug = PixelLevelAugmentation(aug_config).to(device)
        if step % 200 == 0:
            print(f"  Step {step}/{num_iterations}")
    no_cache_training_time = time.time() - start_time
    print(f"总耗时: {no_cache_training_time:.2f}秒")

    # 使用缓存
    print("\n[场景2] 使用缓存 - 复用同一个增强器")
    clear_augmentation_cache()

    start_time = time.time()
    for step in range(num_iterations):
        # 从缓存获取（第一次创建，后续直接返回）
        aug = get_or_create_augmentation(aug_config, device)
        if step % 200 == 0:
            print(f"  Step {step}/{num_iterations}")
    cached_training_time = time.time() - start_time
    print(f"总耗时: {cached_training_time:.2f}秒")

    # 训练场景的性能提升
    training_speedup = no_cache_training_time / cached_training_time
    time_saved = no_cache_training_time - cached_training_time
    print(f"\n训练场景性能提升: {training_speedup:.1f}x 倍速")
    print(f"节省时间: {time_saved:.2f}秒")
    print(f"1000步节省 {time_saved:.1f}秒 → 10万步预计节省 {time_saved*100/60:.1f}分钟")

    # 测试4: 验证缓存正确性
    print("\n" + "=" * 60)
    print("验证缓存正确性")
    print("=" * 60)

    clear_augmentation_cache()

    aug1 = get_or_create_augmentation(aug_config, device)
    aug2 = get_or_create_augmentation(aug_config, device)
    aug3 = get_or_create_augmentation(aug_config, device)

    print(f"\n三次调用获取的实例:")
    print(f"  aug1 id: {id(aug1)}")
    print(f"  aug2 id: {id(aug2)}")
    print(f"  aug3 id: {id(aug3)}")

    if aug1 is aug2 and aug2 is aug3:
        print("  ✅ 相同配置返回同一个实例（缓存生效）")
    else:
        print("  ❌ 返回了不同实例（缓存未生效）")

    # 测试不同配置
    aug_config2 = aug_config.copy()
    aug_config2['gaussian_blur']['enabled'] = False

    aug4 = get_or_create_augmentation(aug_config2, device)
    print(f"\n不同配置:")
    print(f"  aug4 id: {id(aug4)}")

    if aug4 is not aug1:
        print("  ✅ 不同配置返回不同实例")
    else:
        print("  ❌ 不同配置返回了相同实例")

    print("\n" + "=" * 60)
    print("✅ 所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_augmentation_cache_performance()
    print("\n**GG - 缓存机制测试完成!**")
