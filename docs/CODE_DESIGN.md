# 渐进式Dropout实现 - 代码设计说明

## 📐 架构设计原则

### 关注点分离 (Separation of Concerns)
- **工具函数** (`src/utils/train_utils.py`): 纯计算逻辑，无模型依赖
- **模型接口** (`src/model/net.py`): 封装模型内部操作，提供统一API
- **训练脚本** (`script/stage1.py`): 编排训练流程，调用工具和模型接口

### 优势
1. **可测试性**: 工具函数独立，易于单元测试
2. **可维护性**: 模型内部细节封装在类方法中，外部无需关心实现
3. **可复用性**: `get_progressive_dropout()`可用于其他训练脚本
4. **可扩展性**: 未来支持其他渐进式策略（如学习率、数据增强）

---

## 🔧 核心组件

### 1. 工具函数: `get_progressive_dropout()`

**位置**: `src/utils/train_utils.py`

**职责**:
- 计算给定训练步数对应的dropout率
- 纯函数，无副作用
- 不依赖模型或训练状态

**接口**:
```python
def get_progressive_dropout(
    step: int,
    total_steps: int,
    warmup_steps: int = 10000,
    target_dropout: float = 0.1
) -> float
```

**设计要点**:
- 参数化配置（warmup_steps, target_dropout）
- 清晰的文档和示例
- 易于扩展为其他策略（如线性增长、余弦退火等）

---

### 2. 模型接口: `PoseNet.set_dropout_rate()`

**位置**: `src/model/net.py` - PoseNet类

**职责**:
- 封装dropout更新逻辑
- 隐藏模型内部结构（TransformerCrossAttn层级）
- 提供统一的外部调用接口

**接口**:
```python
def set_dropout_rate(self, dropout_rate: float):
    """动态设置HandDecoder中所有Dropout层的dropout率"""
    if not (0.0 <= dropout_rate <= 1.0):
        raise ValueError(...)
    # 更新所有dropout层
    ...
```

**设计要点**:
- **封装性**: 外部不需要知道`handec.transformer.layers`的结构
- **健壮性**: 参数验证，确保dropout_rate合法
- **完整性**: 遍历所有dropout层（Attention, CrossAttention, FeedForward）
- **DDP兼容**: 通过unwrap处理`net.module`

---

### 3. 训练编排: `script/stage1.py`

**职责**:
- 在训练循环中调用工具和模型接口
- 监控和记录dropout变化

**实现**:
```python
from src.utils.train_utils import get_progressive_dropout

while global_step < total_step:
    # 计算当前dropout率
    current_dropout = get_progressive_dropout(
        step=global_step,
        total_steps=total_step,
        warmup_steps=cfg.GENERAL.get("dropout_warmup_step", 10000),
        target_dropout=cfg.MODEL.handec.dropout
    )

    # 更新模型dropout
    unwrapped_net = net.module if hasattr(net, 'module') else net
    unwrapped_net.set_dropout_rate(current_dropout)

    # 训练步骤...
```

**设计要点**:
- **配置驱动**: 从config读取`dropout_warmup_step`和`target_dropout`
- **解耦**: 不直接操作模型内部结构
- **监控**: 记录dropout率到日志和AIM

---

## 🧪 测试策略

### 单元测试: `test_progressive_dropout.py`

**测试内容**:
1. **函数测试**: 验证`get_progressive_dropout()`在不同step的输出
2. **接口测试**: 验证`set_dropout_rate()`正确更新所有dropout层
3. **集成测试**: 加载完整模型，测试端到端流程

**测试用例**:
```python
# 测试边界条件
assert get_progressive_dropout(0, 100000, 10000, 0.1) == 0.0
assert get_progressive_dropout(9999, 100000, 10000, 0.1) == 0.0
assert get_progressive_dropout(10000, 100000, 10000, 0.1) == 0.1

# 测试模型更新
net.set_dropout_rate(0.05)
# 验证所有dropout层都变为0.05
```

---

## 📊 数据流

```
配置文件 (stage1-dino_large.yaml)
  ├─ GENERAL.dropout_warmup_step: 10000
  └─ MODEL.handec.dropout: 0.1
          ↓
训练脚本 (stage1.py)
  ├─ 读取配置
  ├─ 每个step调用: get_progressive_dropout(step, ...)
  │   └─ src/utils/train_utils.py
  │       └─ 返回: current_dropout
  └─ 调用: net.set_dropout_rate(current_dropout)
      └─ src/model/net.py
          └─ 更新handec中所有dropout层
```

---

## 🔄 未来扩展

### 1. 支持其他渐进式策略

**线性增长**:
```python
def get_progressive_dropout_linear(step, total_steps, warmup_steps, target_dropout):
    if step < warmup_steps:
        # 线性从0增长到target_dropout
        return (step / warmup_steps) * target_dropout
    else:
        return target_dropout
```

**余弦退火**:
```python
def get_progressive_dropout_cosine(step, total_steps, warmup_steps, target_dropout):
    if step < warmup_steps:
        # 余弦曲线增长
        progress = step / warmup_steps
        return target_dropout * (1 - math.cos(progress * math.pi)) / 2
    else:
        return target_dropout
```

### 2. 支持层级dropout

不同层使用不同dropout率:
```python
def set_dropout_rate_layerwise(self, dropout_rates: List[float]):
    """为每一层设置不同的dropout率"""
    for layer_idx, dropout_rate in enumerate(dropout_rates):
        # 更新特定层的dropout
        ...
```

### 3. 支持其他模块

扩展到TemporalEncoder等其他包含dropout的模块:
```python
def set_dropout_rate(self, dropout_rate: float, modules: List[str] = ["handec"]):
    """支持指定模块列表"""
    if "handec" in modules:
        # 更新handec
        ...
    if "temporal_encoder" in modules:
        # 更新temporal_encoder
        ...
```

---

## 🎯 最佳实践

### 1. 配置优先
- 所有超参数通过配置文件控制
- 提供合理的默认值

### 2. 接口稳定
- 模型类方法签名保持稳定
- 内部实现可以优化，不影响外部调用

### 3. 文档完善
- 每个函数/方法都有docstring
- 说明参数、返回值、示例

### 4. 测试覆盖
- 关键逻辑都有测试用例
- 边界条件测试
- 集成测试确保端到端正确

---

## 📝 代码审查清单

- [ ] `get_progressive_dropout()`是否为纯函数？
- [ ] `set_dropout_rate()`是否正确处理所有dropout层？
- [ ] 是否支持DDP（处理`net.module`）？
- [ ] 配置参数是否有默认值？
- [ ] 是否添加了日志和监控？
- [ ] 是否编写了测试用例？
- [ ] 文档是否完善？

---

**设计者**: Claude Code
**日期**: 2026-01-30
**版本**: 1.0

**GG**
