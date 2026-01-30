"""
测试渐进式Dropout实现
"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径
# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from src.utils.train_utils import get_progressive_dropout

def test_progressive_dropout_function():
    """测试dropout计算函数"""
    print("=" * 60)
    print("测试渐进式Dropout函数")
    print("=" * 60)

    total_steps = 100000
    warmup_steps = 10000
    target_dropout = 0.1

    test_steps = [0, 1000, 5000, 9999, 10000, 15000, 50000, 100000]

    for step in test_steps:
        dropout = get_progressive_dropout(step, total_steps, warmup_steps, target_dropout)
        expected = 0.0 if step < warmup_steps else target_dropout
        status = "✅" if dropout == expected else "❌"
        print(f"Step {step:6d}: dropout={dropout:.3f} (expected={expected:.3f}) {status}")

    print()


def test_dropout_update_in_model():
    """测试模型中dropout的动态更新"""
    print("=" * 60)
    print("测试模型Dropout动态更新")
    print("=" * 60)

    # 初始化accelerate state（避免logging错误）
    from accelerate import PartialState
    _ = PartialState()

    from src.model.net import PoseNet

    # 创建简单的测试模型
    net = PoseNet(
        stage="stage1",
        backbone_str="model/facebook/dinov2-large",
        img_size=224,
        img_mean=[0.485, 0.456, 0.406],
        img_std=[0.229, 0.224, 0.225],
        infusion_feats_lyr=None,
        drop_cls=False,
        backbone_kwargs=None,
        num_handec_layer=2,  # 减少层数以加快测试
        num_handec_head=8,
        ndim_handec_mlp=2048,
        ndim_handec_head=64,
        prob_handec_dropout=0.1,  # 初始dropout
        prob_handec_emb_dropout=0.0,
        handec_emb_dropout_type="drop",
        handec_norm="layer",
        ndim_handec_norm_cond_dim=-1,
        ndim_handec_ctx=1024,
        handec_skip_token_embed=False,
        handec_mean_init=True,
        handec_denorm_output=False,
        handec_heatmap_resulotion=[512, 512, 1024],
        pie_type="ca",
        num_pie_sample=8,
        pie_fusion="all",
        num_temporal_head=8,
        num_temporal_layer=1,
        trope_scalar=20.0,
        zero_linear=True,
        joint_rep_type="3",
        supervise_global=True,
        supervise_heatmap=True,
        lambda_theta=3.0,
        lambda_shape=3.0,
        lambda_trans=0.05,
        lambda_rel=0.012,
        lambda_img=0.002,
        hm_sigma=0.09,
        freeze_backbone=False,
        norm_by_hand=True,
    )

    print(f"模型创建成功")

    # 收集所有dropout层
    def collect_dropout_layers(module, name=""):
        dropouts = []
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Dropout):
                dropouts.append((full_name, child))
            else:
                dropouts.extend(collect_dropout_layers(child, full_name))
        return dropouts

    # 测试dropout更新逻辑
    print(f"\n使用PoseNet.set_dropout_rate()接口测试")

    # 设置新的dropout率
    new_dropout = 0.05
    print(f"尝试将dropout率从0.1更新到{new_dropout}")

    # 使用新接口更新dropout
    net.set_dropout_rate(new_dropout)

    # 验证更新成功
    print("\n验证所有dropout层已更新:")
    handec = net.handec
    if hasattr(handec, 'transformer') and hasattr(handec.transformer, 'transformer') and hasattr(handec.transformer.transformer, 'layers'):
        inner_transformer = handec.transformer.transformer
        print(f"HandDecoder有 {len(inner_transformer.layers)} 层")

        all_updated = True
        dropout_count = 0

        for layer_idx, layer_modules in enumerate(inner_transformer.layers):
            for module_idx, wrapped_module in enumerate(layer_modules):
                if hasattr(wrapped_module, 'fn'):
                    inner_module = wrapped_module.fn
                    # 检查Attention/CrossAttention中的dropout
                    if hasattr(inner_module, 'dropout') and isinstance(inner_module.dropout, nn.Dropout):
                        dropout_count += 1
                        if inner_module.dropout.p != new_dropout:
                            print(f"  ❌ Layer{layer_idx}.Module{module_idx}.dropout: {inner_module.dropout.p:.3f} != {new_dropout:.3f}")
                            all_updated = False
                        else:
                            print(f"  ✅ Layer{layer_idx}.Module{module_idx}.dropout: {inner_module.dropout.p:.3f}")
                    # 检查FeedForward中的dropout
                    if hasattr(inner_module, 'net'):
                        for sub_idx, sub_module in enumerate(inner_module.net):
                            if isinstance(sub_module, nn.Dropout):
                                dropout_count += 1
                                if sub_module.p != new_dropout:
                                    print(f"  ❌ Layer{layer_idx}.Module{module_idx}.net[{sub_idx}].dropout: {sub_module.p:.3f} != {new_dropout:.3f}")
                                    all_updated = False
                                else:
                                    print(f"  ✅ Layer{layer_idx}.Module{module_idx}.net[{sub_idx}].dropout: {sub_module.p:.3f}")

        print(f"\n总共检查了 {dropout_count} 个Dropout层")
        if all_updated:
            print("✅ 所有Dropout层已成功更新")
        else:
            print("❌ 部分Dropout层更新失败")
    else:
        print("❌ 模型结构不符合预期")

    print()


if __name__ == "__main__":
    # 测试dropout函数
    test_progressive_dropout_function()

    # 测试模型更新逻辑
    print("\n" + "=" * 60)
    print("正在加载模型进行测试...")
    print("=" * 60)
    try:
        test_dropout_update_in_model()
        print("\n✅ 所有测试通过!")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
