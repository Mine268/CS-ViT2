"""
用简单的例子解释 RobustL1Loss 的 bug
"""

import torch

print("=" * 80)
print("RobustL1Loss Bug 详细解释")
print("=" * 80)

# 第一步：理解 RobustL1Loss 的设计
print("\n第一步：RobustL1Loss 的设计思想")
print("-" * 80)

print("""
RobustL1Loss 的目标：
  - 小误差（< delta）：像 L1 一样线性
  - 大误差（>= delta）：梯度衰减，不让异常值主导训练

实现方式：
  if abs(pred - target) < delta:
      loss = abs(pred - target)           # 区域1：L1
  else:
      loss = delta * (1 + log(...))       # 区域2：对数衰减
""")

delta = 84.0
print(f"当前配置：delta = {delta}")

# 第二步：torch.where 的问题
print("\n第二步：torch.where 的关键问题")
print("-" * 80)

print("""
代码看起来是这样：
    loss = torch.where(inside_mask, loss_l1, loss_log)

你以为：
    - 如果 inside_mask=True，只计算 loss_l1
    - 如果 inside_mask=False，只计算 loss_log

实际上：
    - loss_l1 和 loss_log **都会被计算**
    - 然后根据 mask 选择结果
    - 即使不选择 loss_log，它也被计算了！
""")

# 演示
print("\n演示 torch.where 的行为：")
x = torch.tensor([1.0, 2.0])
y = torch.tensor([3.0, 0.0])

print(f"x = {x}")
print(f"y = {y}")
print(f"x / y = {x / y}")  # [0.333, inf]

result = torch.where(torch.tensor([True, False]), x / y, torch.tensor([0.0, 0.0]))
print(f"where(True, x/y, 0) = {result}")
print("虽然第二个元素选择了 0，但 x/y 还是被计算了，产生了 inf")

# 第三步：问题出在哪里
print("\n第三步：问题出在 loss_log 的计算")
print("-" * 80)

print("""
loss_log 的计算：
    outside_diff = abs_diff - delta
    ratio = outside_diff / delta
    loss_log = delta * (1 + log1p(ratio))

关键问题：log1p(ratio)
""")

# log1p 的行为
print("\nlog1p(x) 的数学行为：")
print(f"  log1p(0.5) = {torch.log1p(torch.tensor(0.5)).item():.4f}  ← 正常")
print(f"  log1p(0.0) = {torch.log1p(torch.tensor(0.0)).item():.4f}  ← 正常")
print(f"  log1p(-0.5) = {torch.log1p(torch.tensor(-0.5)).item():.4f}  ← 正常")
print(f"  log1p(-0.99) = {torch.log1p(torch.tensor(-0.99)).item():.4f}  ← 很大的负数，但正常")
print(f"  log1p(-1.0) = {torch.log1p(torch.tensor(-1.0)).item()}  ← -inf！")
print(f"  log1p(-1.1) = {torch.log1p(torch.tensor(-1.1)).item()}  ← NaN！")

print("\n关键点：log1p(x) 要求 x > -1")
print("         如果 x = -1  → -inf")
print("         如果 x < -1  → NaN")

# 第四步：什么时候 ratio 会 ≈ -1
print("\n第四步：什么时候 ratio 会接近 -1？")
print("-" * 80)

print("""
ratio = (abs_diff - delta) / delta

假设 delta = 84：
""")

examples = [
    ("正常情况", 100.0, 0.0),
    ("刚好等于 delta", 84.0, 0.0),
    ("小于 delta", 50.0, 0.0),
    ("非常小的误差", 0.001, 0.0),
    ("几乎为零的误差", 1e-10, 0.0),
]

print(f"\n{'情况':<20} {'abs_diff':<15} {'ratio':<20} {'log1p(ratio)':<20}")
print("-" * 75)

for name, pred, target in examples:
    abs_diff = abs(pred - target)
    ratio = (abs_diff - delta) / delta

    if ratio > -1:
        log_val = torch.log1p(torch.tensor(ratio)).item()
        if abs(log_val) > 100:
            log_str = f"{log_val:.2e}"
        else:
            log_str = f"{log_val:.4f}"
    else:
        log_str = "NaN"

    print(f"{name:<20} {abs_diff:<15.2e} {ratio:<20.6f} {log_str:<20}")

print("\n观察：")
print("  - 当 abs_diff 非常小时，ratio 接近 -1")
print("  - 当 abs_diff ≈ 0 时，ratio = -84/84 = -1.0")
print("  - log1p(-1.0) = -inf")

# 第五步：为什么会触发 bug
print("\n第五步：为什么会触发 bug？")
print("-" * 80)

print("""
场景：预测值和目标值非常接近
    pred = 100.0
    target = 100.0000001  # 几乎相同
    abs_diff = 0.0000001  # 非常小

按理说应该走 L1 分支（因为 abs_diff < delta）:
    inside_mask = True
    应该选择 loss_l1 = abs_diff = 0.0000001

但是 torch.where 会计算 loss_log：
    outside_diff = 0.0000001 - 84 = -83.9999999
    ratio = -83.9999999 / 84 ≈ -1.0
    log1p(-1.0) = -inf  ← 产生了 -inf！

虽然最后选择了 loss_l1，但 -inf 可能污染结果
更糟的是，如果浮点误差导致 ratio 略 < -1：
    log1p(-1.00000001) = NaN  ← 直接 NaN！
""")

# 第六步：实际演示
print("\n第六步：实际演示 bug")
print("-" * 80)

class RobustL1Loss_Buggy:
    def __init__(self, delta=84.0):
        self.delta = delta

    def __call__(self, pred, target):
        abs_diff = torch.abs(pred - target)
        inside_mask = abs_diff < self.delta
        loss_l1 = abs_diff

        # Bug 所在：没有保护 ratio
        outside_diff = abs_diff - self.delta
        ratio = outside_diff / self.delta  # 可能 ≈ -1 或 < -1
        loss_log = self.delta * (1.0 + torch.log1p(ratio))  # log1p 可能产生 NaN

        loss = torch.where(inside_mask, loss_l1, loss_log)
        return loss

loss_fn = RobustL1Loss_Buggy(delta=84.0)

# 测试用例：非常接近的值
pred = torch.tensor([100.0, 100.0], dtype=torch.float32)
target = torch.tensor([100.0 + 1e-10, 100.0 + 1e-20], dtype=torch.float32)

print(f"\npred: {pred}")
print(f"target: {target}")

abs_diff = torch.abs(pred - target)
print(f"abs_diff: {abs_diff}")

loss = loss_fn(pred, target)
print(f"loss: {loss}")
print(f"是否有 NaN: {torch.isnan(loss).any().item()}")
print(f"是否有 Inf: {torch.isinf(loss).any().item()}")

# 查看中间计算
outside_diff = abs_diff - loss_fn.delta
ratio = outside_diff / loss_fn.delta
print(f"\n中间计算:")
print(f"  outside_diff: {outside_diff}")
print(f"  ratio: {ratio}")
print(f"  log1p(ratio): {torch.log1p(ratio)}")

# 第七步：修复方法
print("\n第七步：修复方法")
print("-" * 80)

print("""
核心思想：添加 epsilon 保护，确保 ratio > -1

修复代码：
    outside_diff = abs_diff - self.delta
    ratio = outside_diff / self.delta
    ratio = torch.clamp(ratio, min=-0.99)  # ← 关键：限制最小值
    loss_log = self.delta * (1.0 + torch.log1p(ratio))

为什么用 -0.99：
    - log1p(-0.99) ≈ -4.6，一个大的负数但不是 -inf
    - 保证 log1p 永远不会产生 NaN 或 -inf
    - -0.99 是安全的，因为 log1p(-0.99) 是有限值
""")

class RobustL1Loss_Fixed:
    def __init__(self, delta=84.0):
        self.delta = delta

    def __call__(self, pred, target):
        abs_diff = torch.abs(pred - target)
        inside_mask = abs_diff < self.delta
        loss_l1 = abs_diff

        # 修复：添加 clamp 保护
        outside_diff = abs_diff - self.delta
        ratio = outside_diff / self.delta
        ratio = torch.clamp(ratio, min=-0.99)  # epsilon 保护！
        loss_log = self.delta * (1.0 + torch.log1p(ratio))

        loss = torch.where(inside_mask, loss_l1, loss_log)
        return loss

loss_fn_fixed = RobustL1Loss_Fixed(delta=84.0)

print(f"\n修复后测试相同的输入:")
loss_fixed = loss_fn_fixed(pred, target)
print(f"pred: {pred}")
print(f"target: {target}")
print(f"loss (修复后): {loss_fixed}")
print(f"是否有 NaN: {torch.isnan(loss_fixed).any().item()}")
print(f"是否有 Inf: {torch.isinf(loss_fixed).any().item()}")

# 总结
print("\n" + "=" * 80)
print("总结")
print("=" * 80)

print("""
Bug 的完整链条：
    1. 预测值和目标值非常接近 (abs_diff ≈ 0)
    2. 虽然应该用 L1 分支，但 torch.where 会计算 loss_log
    3. loss_log 中：ratio = (abs_diff - delta) / delta ≈ -1
    4. log1p(ratio ≈ -1) = -inf 或 NaN
    5. 虽然最后选择 L1 分支，但 NaN/Inf 可能污染结果

为什么 float32 也会 NaN：
    - 这是数学问题，不是精度问题
    - log1p(x ≤ -1) 在任何精度下都有问题
    - 当 abs_diff ≈ 0 时，ratio 精确等于 -1.0

修复方法：
    ratio = torch.clamp(ratio, min=-0.99)

    确保 log1p 的输入永远 > -1，避免 NaN/Inf

这个修复：
    ✓ 解决 float32 和 bf16 下的 NaN
    ✓ 只改一行代码
    ✓ 不影响 RobustL1Loss 的功能
""")

print("\n" + "=" * 80)
print("GG")
print("=" * 80)
