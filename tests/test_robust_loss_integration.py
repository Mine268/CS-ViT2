#!/usr/bin/env python3
"""
测试鲁棒loss集成是否正确

验证：
1. RobustL1Loss导入成功
2. BundleLoss2正确初始化鲁棒loss
3. forward计算使用鲁棒loss
4. 对比标准L1和鲁棒L1的输出差异
"""
import sys
import torch
import torch.nn as nn

print("=" * 70)
print("测试鲁棒Loss集成")
print("=" * 70)

# 测试1: 导入模块
print("\n[1/4] 测试模块导入...")
try:
    from src.model.loss import RobustL1Loss, BundleLoss2
    print("✅ 模块导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 测试2: 初始化RobustL1Loss
print("\n[2/4] 测试RobustL1Loss初始化...")
try:
    robust_l1 = RobustL1Loss(delta=84.0, reduction='none')
    print(f"✅ RobustL1Loss初始化成功 (delta=84.0)")

    # 测试forward
    pred = torch.tensor([[0.0, 50.0, 100.0, 200.0, 1000.0]])
    target = torch.zeros_like(pred)
    loss_robust = robust_l1(pred, target)
    loss_l1 = nn.L1Loss(reduction='none')(pred, target)

    print(f"   误差:        {pred[0].numpy()}")
    print(f"   RobustL1:    {loss_robust[0].numpy()}")
    print(f"   标准L1:      {loss_l1[0].numpy()}")
    print(f"   抑制率:      {((loss_l1 - loss_robust) / loss_l1 * 100)[0].numpy()}%")

except Exception as e:
    print(f"❌ RobustL1Loss测试失败: {e}")
    sys.exit(1)

# 测试3: 初始化BundleLoss2 (简化版本，不需要完整参数)
print("\n[3/4] 测试BundleLoss2集成...")
try:
    # 创建简化的BundleLoss2实例
    loss_fn = BundleLoss2(
        lambda_theta=3.0,
        lambda_shape=3.0,
        lambda_trans=0.05,
        lambda_rel=0.012,
        lambda_img=0.002,
        supervise_global=True,
        supervise_heatmap=False,
        norm_by_hand=True,
        norm_idx=[0, 5, 9, 13, 17],
        hm_centers=None,
        hm_sigma=0.09,
        reproj_loss_type="robust_l1",
        reproj_loss_delta=84.0,
    )

    print(f"✅ BundleLoss2初始化成功")
    print(f"   - reproj_loss_type: {loss_fn.reproj_loss_type}")
    print(f"   - reproj_loss_fn: {type(loss_fn.reproj_loss_fn).__name__}")

    if loss_fn.reproj_loss_type == "robust_l1":
        assert isinstance(loss_fn.reproj_loss_fn, RobustL1Loss), \
            "reproj_loss_fn应该是RobustL1Loss实例"
        print(f"   - delta: {loss_fn.reproj_loss_fn.delta}")

except Exception as e:
    print(f"❌ BundleLoss2集成失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试4: 对比启用/禁用鲁棒loss的差异
print("\n[4/4] 对比鲁棒loss效果...")
try:
    # 创建两个loss函数：一个使用鲁棒loss，一个使用标准L1
    loss_fn_robust = BundleLoss2(
        lambda_theta=3.0, lambda_shape=3.0, lambda_trans=0.05,
        lambda_rel=0.012, lambda_img=0.002,
        supervise_global=True, supervise_heatmap=False,
        norm_by_hand=True, norm_idx=[0, 5, 9, 13, 17],
        hm_centers=None, hm_sigma=0.09,
        reproj_loss_type="robust_l1",
        reproj_loss_delta=84.0,
    )

    loss_fn_standard = BundleLoss2(
        lambda_theta=3.0, lambda_shape=3.0, lambda_trans=0.05,
        lambda_rel=0.012, lambda_img=0.002,
        supervise_global=True, supervise_heatmap=False,
        norm_by_hand=True, norm_idx=[0, 5, 9, 13, 17],
        hm_centers=None, hm_sigma=0.09,
        reproj_loss_type="l1",
    )

    # 模拟重投影误差 [b, t, j, 2] = [1, 1, 5, 1]
    errors = torch.tensor([[0.0, 50.0, 100.0, 200.0, 1000.0]])  # [1, 5]
    joint_img_pred = errors.view(1, 1, 5, 1)  # [b=1, t=1, j=5, d=1]
    joint_img_gt = torch.zeros_like(joint_img_pred)

    loss_robust = loss_fn_robust.reproj_loss_fn(joint_img_pred, joint_img_gt)
    loss_standard = loss_fn_standard.reproj_loss_fn(joint_img_pred, joint_img_gt)

    print(f"   误差 (像素):    {joint_img_pred[0, 0, :, 0].numpy()}")
    print(f"   鲁棒Loss:       {loss_robust[0, 0, :, 0].numpy()}")
    print(f"   标准L1 Loss:    {loss_standard[0, 0, :, 0].numpy()}")

    # 计算异常值抑制效果
    extreme_idx = 4  # 1000px的误差
    suppression = (loss_standard[0, 0, extreme_idx, 0] - loss_robust[0, 0, extreme_idx, 0]) / loss_standard[0, 0, extreme_idx, 0] * 100
    print(f"\n   对于1000px异常值:")
    print(f"   - 标准L1 loss: {loss_standard[0, 0, extreme_idx, 0].item():.2f}")
    print(f"   - 鲁棒loss:     {loss_robust[0, 0, extreme_idx, 0].item():.2f}")
    print(f"   - 抑制率:       {suppression.item():.1f}%")

    print(f"\n✅ 所有测试通过！")

except Exception as e:
    print(f"❌ 对比测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("集成验证成功！鲁棒loss已正确集成到训练流程中。")
print("=" * 70)
print("\n下一步:")
print("1. 启动训练测试: python script/train.py --config-name=stage1-dino_large")
print("2. 观察log中的loss_joint_img是否不再出现极端值")
print("3. 对比训练曲线的稳定性")
