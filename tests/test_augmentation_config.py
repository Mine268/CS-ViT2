"""
测试数据增强配置系统
"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.data.preprocess import PixelLevelAugmentation


def test_augmentation_config():
    """测试不同配置下的增强器构建"""
    print("=" * 60)
    print("测试数据增强配置系统")
    print("=" * 60)

    # 测试1: 默认配置（None）
    print("\n[测试1] 默认配置")
    aug_default = PixelLevelAugmentation(None)
    print(f"  增强数量: {aug_default.num_transforms}")
    print(f"  增强列表: {aug_default.transforms}")
    assert aug_default.num_transforms == 2, "默认应该有2个增强"
    print("  ✅ 通过")

    # 测试2: 空配置（所有禁用）
    print("\n[测试2] 空配置（所有禁用）")
    aug_config_empty = {
        'color_jitter': {'enabled': False},
        'gaussian_noise': {'enabled': False},
        'gaussian_blur': {'enabled': False},
    }
    aug_empty = PixelLevelAugmentation(aug_config_empty)
    print(f"  增强数量: {aug_empty.num_transforms}")
    assert aug_empty.num_transforms == 1, "空配置应该只有Identity"
    print("  ✅ 通过")

    # 测试3: 只启用ColorJitter
    print("\n[测试3] 只启用ColorJitter")
    aug_config_cj = {
        'color_jitter': {
            'enabled': True,
            'brightness': 0.3,
            'contrast': 0.3,
            'saturation': 0.2,
            'hue': 0.05,
            'p': 0.6
        },
        'gaussian_noise': {'enabled': False},
    }
    aug_cj = PixelLevelAugmentation(aug_config_cj)
    print(f"  增强数量: {aug_cj.num_transforms}")
    assert aug_cj.num_transforms == 1, "应该只有1个ColorJitter"
    print("  ✅ 通过")

    # 测试4: 启用多个增强
    print("\n[测试4] 启用多个增强")
    aug_config_multi = {
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
    aug_multi = PixelLevelAugmentation(aug_config_multi)
    print(f"  增强数量: {aug_multi.num_transforms}")
    assert aug_multi.num_transforms == 3, "应该有3个增强"
    print("  ✅ 通过")

    # 测试5: 前向传播
    print("\n[测试5] 前向传播测试")
    batch_size = 2
    num_frames = 1
    height, width = 224, 224
    test_input = torch.rand(batch_size, num_frames, 3, height, width)

    for name, aug_model in [
        ("默认配置", aug_default),
        ("空配置", aug_empty),
        ("只ColorJitter", aug_cj),
        ("多个增强", aug_multi),
    ]:
        output = aug_model(test_input)
        assert output.shape == test_input.shape, f"{name}: 输出形状不匹配"
        assert output.min() >= 0.0 and output.max() <= 1.0, f"{name}: 输出值超出[0,1]"
        print(f"  ✅ {name}: 形状={output.shape}, 范围=[{output.min():.3f}, {output.max():.3f}]")

    print("\n" + "=" * 60)
    print("✅ 所有测试通过!")
    print("=" * 60)


def test_config_loading_from_yaml():
    """测试从YAML配置加载"""
    print("\n" + "=" * 60)
    print("测试从YAML配置加载增强器")
    print("=" * 60)

    from omegaconf import OmegaConf

    # 模拟YAML配置
    yaml_str = """
    augmentation:
      color_jitter:
        enabled: true
        brightness: 0.2
        contrast: 0.2
        saturation: 0.1
        hue: 0.0
        p: 0.5
      gaussian_noise:
        enabled: true
        mean: 0.0
        std: 0.03
        p: 0.5
      gaussian_blur:
        enabled: false
    """

    cfg = OmegaConf.create(yaml_str)
    aug_config = OmegaConf.to_container(cfg.augmentation, resolve=True)

    print(f"\n加载的配置:")
    print(OmegaConf.to_yaml(cfg.augmentation))

    aug = PixelLevelAugmentation(aug_config)
    print(f"\n构建的增强器:")
    print(f"  增强数量: {aug.num_transforms}")
    print(f"  ✅ 成功从YAML配置构建增强器")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_augmentation_config()
    test_config_loading_from_yaml()
    print("\n**GG - 数据增强配置系统测试完成!**")
