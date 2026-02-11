#!/usr/bin/env python3
"""
测试config中的reproj_loss_type配置是否正确读取
"""
import sys
from omegaconf import OmegaConf

print("=" * 70)
print("测试配置文件中的 reproj_loss 参数")
print("=" * 70)

# 测试 Stage1 配置
print("\n[1/2] 测试 stage1-dino_large.yaml...")
try:
    cfg_stage1 = OmegaConf.load("config/stage1-dino_large.yaml")

    reproj_type = cfg_stage1.LOSS.get("reproj_loss_type", "NOT_FOUND")
    reproj_delta = cfg_stage1.LOSS.get("reproj_loss_delta", "NOT_FOUND")

    print(f"   reproj_loss_type: {reproj_type}")
    print(f"   reproj_loss_delta: {reproj_delta}")

    assert reproj_type == "robust_l1", f"Expected 'robust_l1', got '{reproj_type}'"
    assert reproj_delta == 84.0, f"Expected 84.0, got {reproj_delta}"

    print(f"✅ Stage1 配置正确")
except Exception as e:
    print(f"❌ Stage1 配置错误: {e}")
    sys.exit(1)

# 测试 Stage2 配置
print("\n[2/2] 测试 stage2-dino_large.yaml...")
try:
    cfg_stage2 = OmegaConf.load("config/stage2-dino_large.yaml")

    reproj_type = cfg_stage2.LOSS.get("reproj_loss_type", "NOT_FOUND")
    reproj_delta = cfg_stage2.LOSS.get("reproj_loss_delta", "NOT_FOUND")

    print(f"   reproj_loss_type: {reproj_type}")
    print(f"   reproj_loss_delta: {reproj_delta}")

    assert reproj_type == "robust_l1", f"Expected 'robust_l1', got '{reproj_type}'"
    assert reproj_delta == 84.0, f"Expected 84.0, got {reproj_delta}"

    print(f"✅ Stage2 配置正确")
except Exception as e:
    print(f"❌ Stage2 配置错误: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("配置验证成功！")
print("=" * 70)
print("\n使用示例:")
print("  # 使用默认的 robust_l1:")
print("  python script/train.py --config-name=stage1-dino_large")
print()
print("  # 切换为标准 L1 loss:")
print("  python script/train.py --config-name=stage1-dino_large LOSS.reproj_loss_type=l1")
print()
print("  # 调整 delta 阈值:")
print("  python script/train.py --config-name=stage1-dino_large LOSS.reproj_loss_delta=110.0")
print()
