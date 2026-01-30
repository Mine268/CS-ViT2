# 训练优化改进总结

## 📋 实施日期
2026-01-30

## 🎯 优化目标
基于训练对比分析(checkpoint/28-01-2026 vs checkpoint/29-01-2026),解决以下核心问题:
1. **训练早期过拟合加剧** - Step 3000-6000的泛化gap增大
2. **验证波动增大** - 标准差从19.62mm增加到22.41mm
3. **Step 12000异常恶化** - MPJPE从82.61mm跳升到105.23mm

## ✅ 已实施改进

### 1. 渐进式Dropout策略 (高优先级)

**问题诊断:**
- 固定Dropout=0.1在训练早期阻碍特征学习
- 训练loss整体升高7.6%
- 训练初期(<6000步)过拟合加剧

**解决方案:**
```python
# src/utils/train_utils.py
def get_progressive_dropout(step, total_steps, warmup_steps=10000, target_dropout=0.1):
    """训练早期禁用dropout,后期逐步启用"""
    if step < warmup_steps:
        return 0.0  # 早期无dropout
    else:
        return target_dropout  # 后期使用目标dropout

# src/model/net.py - PoseNet类
def set_dropout_rate(self, dropout_rate: float):
    """动态设置HandDecoder中所有Dropout层的dropout率"""
    # 遍历更新所有transformer层的dropout
    ...
```

**实现细节:**
- Dropout预热步数: 10000步 (可通过`GENERAL.dropout_warmup_step`配置)
- 目标dropout率: 0.1 (从`MODEL.handec.dropout`读取)
- 封装在PoseNet类中，提供`set_dropout_rate()`接口
- 动态更新所有TransformerCrossAttn层中的Dropout模块:
  - Attention层的dropout
  - CrossAttention层的dropout
  - FeedForward层的2个dropout

**配置参数:**
```yaml
# config/stage1-dino_large.yaml
GENERAL:
  dropout_warmup_step: 10000  # 新增
```

**预期效果:**
- ✅ 训练loss降低
- ✅ 训练早期的泛化gap减小
- ✅ 验证波动降低

---

### 2. 验证集一致性保证 (高优先级)

**问题诊断:**
- Step 12000验证MPJPE异常恶化+22.62mm (+27.4%)
- WebDataset的shuffle可能导致不同训练run的验证batch不一致
- 无法准确对比不同配置的效果

**解决方案:**
```python
# src/data/dataloader.py:131-158
def get_dataloader(..., seed: Optional[int] = None):
    """添加seed参数固定验证集shuffle顺序"""
    dataset = (
        wds.WebDataset(...)
        .shuffle(20, rng=np.random.default_rng(seed) if seed is not None else None)
        .decode()
        .compose(...)
        .shuffle(200, rng=np.random.default_rng(seed) if seed is not None else None)
        ...
    )
```

**配置参数:**
```yaml
# config/stage1-dino_large.yaml
GENERAL:
  val_seed: 42  # 新增 - 固定验证集随机种子
```

**实施位置:**
```python
# script/stage1.py:78-87
val_loader = get_dataloader(
    ...,
    seed=cfg.GENERAL.get("val_seed", 42),  # 固定验证集seed
)
```

**预期效果:**
- ✅ 不同训练run使用完全相同的验证数据
- ✅ 验证指标可准确对比
- ✅ 消除验证集采样引起的异常波动

---

### 3. 监控增强

**新增AIM监控指标:**
- `dropout_rate`: 实时dropout率 (每10步记录)
- 便于观察dropout预热过程

**日志输出增强:**
```
# 原日志格式
3000/100000 lr=1.0000e-04 total=1.230 loss_theta=0.083 ...

# 新日志格式
3000/100000 lr=1.0000e-04 dropout=0.000 total=1.230 loss_theta=0.083 ...
                                   ↑ 新增dropout监控
```

---

## 🔍 修改文件清单

### 工具函数模块
1. **src/utils/train_utils.py** (新建)
   - 新增: `get_progressive_dropout()` 函数
   - 提供渐进式dropout计算逻辑

### 模型定义
2. **src/model/net.py**
   - 新增: `PoseNet.set_dropout_rate()` 方法
   - 封装dropout动态更新逻辑，提供统一接口

### 核心训练脚本
3. **script/stage1.py**
   - 新增: 导入`get_progressive_dropout`
   - 修改: `train()` 函数调用`net.set_dropout_rate()`更新dropout
   - 修改: 日志输出添加dropout监控
   - 修改: 验证集loader添加seed参数

### 数据加载模块
4. **src/data/dataloader.py**
   - 修改: `get_dataloader()` 添加seed参数
   - 两个shuffle调用都支持固定seed

### 配置文件
5. **config/stage1-dino_large.yaml**
   - 新增: `GENERAL.dropout_warmup_step: 10000`
   - 新增: `GENERAL.val_seed: 42`

### 测试脚本
6. **test_progressive_dropout.py** (新建)
   - 测试`get_progressive_dropout()`函数
   - 测试`PoseNet.set_dropout_rate()`方法

---

## 🧪 测试验证

### 测试脚本
创建了`test_progressive_dropout.py`:
1. 测试渐进式dropout函数逻辑
2. 测试模型中dropout层的动态更新
3. 验证所有TransformerCrossAttn层的dropout都能正确更新

### 运行测试
```bash
python test_progressive_dropout.py
```

**预期输出:**
- ✅ Dropout计算函数在warmup前返回0.0,之后返回目标值
- ✅ 能够正确收集并更新handec中的所有dropout层
- ✅ 更新后的dropout率验证通过

---

## 📊 预期训练改善

### 训练早期 (0-10000步)
**改善点:**
- Dropout=0.0 → 训练loss更低,优化更顺畅
- 过拟合gap减小 (预计从+57.5mm降至~+45mm)
- 验证MPJPE改善

**对比预测:**
```
Step    修改前Gap    预期新Gap    改善
3000     +57.5mm      ~+45mm     -12.5mm ✅
6000     +33.3mm      ~+25mm      -8.3mm ✅
```

### 训练中后期 (10000-100000步)
**改善点:**
- Dropout=0.1启用,提升泛化性
- 验证集一致性确保准确对比
- 波动减小,训练更稳定

**对比预测:**
```
Step    修改前MPJPE   预期新MPJPE   改善
12000    105.23mm      ~85mm       -20mm ✅
15000     71.07mm      ~68mm        -3mm ✅
```

### 整体指标
- **平均MPJPE**: 84.47mm → **<80mm** (目标改善5%+)
- **标准差**: 22.41mm → **<18mm** (目标降低20%+)
- **训练loss**: 1.230 → **~1.10** (目标降低10%+)

---

## 🚀 下一步行动

### 立即执行
1. ✅ 运行测试: `python test_progressive_dropout.py`
2. ⏳ 启动新训练:
   ```bash
   python script/stage1.py --config-name=stage1-dino_large
   ```
3. ⏳ 监控前15000步的指标变化

### 对比分析 (训练30000步后)
1. 绘制训练曲线对比:
   - 训练loss vs 验证loss
   - Dropout率变化曲线
   - 验证MPJPE趋势
2. 对比相同step的验证MPJPE:
   - Step 3000, 6000, 9000, 12000, 15000等
3. 统计验证波动(标准差)

### 可选后续优化 (Phase 2)
如果dropout优化效果显著,可以考虑:
1. 恢复GaussianBlur数据增强 (p=0.15)
2. 微调loss权重 (lambda_trans: 0.05→0.035)
3. 增加验证频率 (每1000步而非3000步)

---

## 📝 注意事项

### 向后兼容性
- ✅ 新参数都有默认值,旧配置文件可正常运行
- ✅ `dropout_warmup_step`默认10000
- ✅ `val_seed`默认42

### 恢复训练
如果从checkpoint恢复训练:
```bash
python script/stage1.py --config-name=stage1-dino_large \
    GENERAL.resume_path=checkpoint/xxx/checkpoints/checkpoint-30000
```
- ✅ Dropout率会根据恢复的step自动调整
- ✅ 验证集仍使用固定seed,确保一致性

### AIM监控
- 新增`dropout_rate`指标可在AIM dashboard中查看
- 建议创建新的实验组进行对比

---

## 🔬 理论依据

### Dropout预热的必要性
1. **DINOv2预训练特征鲁棒性强**: 已经过大规模自监督学习,不需要额外正则化
2. **Fine-tuning早期需要稳定**: Dropout干扰会阻碍backbone适应新任务
3. **后期泛化提升**: 数据拟合稳定后,dropout有助于防止过拟合

### 验证集固定seed的重要性
1. **WebDataset的shuffle机制**: 每次迭代数据顺序可能不同
2. **公平对比**: 不同配置必须在相同验证数据上测试
3. **调试友好**: 可复现的验证结果便于定位问题

---

## 📚 参考资料

- 代码设计文档: [CODE_DESIGN.md](CODE_DESIGN.md)
- 重构总结: [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)
- 快速开始: [QUICK_START.md](QUICK_START.md)
- 项目文档: [../CLAUDE.md](../CLAUDE.md)
- 测试脚本: [../tests/test_progressive_dropout.py](../tests/test_progressive_dropout.py)

---

**GG - 优化实施完成,准备开始新训练!**
