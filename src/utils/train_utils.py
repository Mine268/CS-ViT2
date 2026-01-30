"""
Training utility functions
"""
from typing import Optional


def get_progressive_dropout(
    step: int,
    total_steps: int,
    warmup_steps: int = 10000,
    target_dropout: float = 0.1
) -> float:
    """
    渐进式Dropout策略：训练早期禁用dropout，后期逐步启用

    训练早期(step < warmup_steps)时，模型需要稳定学习预训练特征，
    dropout会干扰这一过程。训练后期启用dropout可以提升泛化性。

    Args:
        step: 当前训练步数
        total_steps: 总训练步数
        warmup_steps: Dropout预热步数（在此之前dropout=0）
        target_dropout: 目标dropout率（预热后使用的dropout率）

    Returns:
        当前步数对应的dropout率

    Examples:
        >>> get_progressive_dropout(step=5000, total_steps=100000, warmup_steps=10000, target_dropout=0.1)
        0.0  # 早期无dropout
        >>> get_progressive_dropout(step=15000, total_steps=100000, warmup_steps=10000, target_dropout=0.1)
        0.1  # 后期使用目标dropout
    """
    if step < warmup_steps:
        return 0.0  # 早期无dropout
    else:
        return target_dropout  # 后期使用目标dropout
