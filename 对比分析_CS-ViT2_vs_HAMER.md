# CS-ViT2 vs HAMER 项目对比分析

## 一、模型架构对比

### 1.1 Backbone差异

**CS-ViT2:**
- 使用 `transformers.AutoModel.from_pretrained()` 加载预训练ViT
- 支持多层级特征融合 (`infusion_feats_lyr`)
- 输出token序列 `[B, L, D]` (L = num_patches + 1)
- 可选择是否drop CLS token (`drop_cls`)

**HAMER:**
- 使用自定义ViT实现 (`hamer/models/backbones/vit.py`)
- 输出特征图 `[B, C, H, W]`，然后reshape为token序列
- 固定使用CLS token
- 图像输入尺寸为 `(256, 192)`，但实际使用 `[:,:,:,32:-32]` (裁剪为192x192)

**关键差异:**
```python
# CS-ViT2: 直接输出token序列
feats = self.backbone(img)  # [B, L, D]

# HAMER: 输出特征图后reshape
x = self.backbone(x[:,:,:,32:-32])  # [B, C, H, W]
x = einops.rearrange(x, 'b c h w -> b (h w) c')  # [B, L, D]
```

### 1.2 MANO Head差异

**CS-ViT2:**
- 使用 `MANOTransformerDecoderHead`
- 初始化方式：`use_mean_init=True` 时从mean params初始化，但代码中注释掉了残差连接
- 单次前向传播，无迭代优化
- 输出：`pred_hand_pose, pred_shape, pred_trans`

**HAMER:**
- 使用 `MANOTransformerDecoderHead` (类似结构)
- **关键差异：支持IEF迭代优化** (`IEF_ITERS`，默认1)
- 每次迭代都有残差连接：`pred = decoder_out + pred`
- 初始化方式：`TRANSFORMER_INPUT='mean_shape'` 时使用mean params作为初始token

**关键代码对比:**
```python
# CS-ViT2 (module.py:558-560)
pred_hand_pose = self.decpose(token_out) # + init_hand_pose  # 注释掉了残差
pred_betas = self.decshape(token_out) # + init_betas
pred_cam = self.deccam(token_out) # + init_cam

# HAMER (mano_head.py:89-91)
pred_hand_pose = self.decpose(token_out) + pred_hand_pose  # 有残差连接
pred_betas = self.decshape(token_out) + pred_betas
pred_cam = self.deccam(token_out) + pred_cam
```

### 1.3 透视信息嵌入 (Perspective Information Embedder)

**CS-ViT2:**
- 实现了两种方式：`PerspInfoEmbedderDense` 和 `PerspInfoEmbedderCrossAttn`
- 使用bbox、focal、princpt计算透视方向信息
- 融合方式：`pie_fusion` 可选 "cls", "patch", "all"

**HAMER:**
- **没有显式的透视信息嵌入模块**
- 透视信息可能通过数据增强间接学习

## 二、训练流程对比

### 2.1 优化器设置

**CS-ViT2:**
```python
optim = torch.optim.AdamW(
    params=[
        {"params": regressor_params, "lr": 1e-4},
        {"params": backbone_params, "lr": 1e-5},  # 更小的学习率
    ],
    weight_decay=1e-4,
)
```

**HAMER:**
```python
optimizer = torch.optim.AdamW(
    params=all_params,  # backbone和head统一学习率
    lr=cfg.TRAIN.LR,
    weight_decay=cfg.TRAIN.WEIGHT_DECAY,
)
```

**差异：** CS-ViT2对backbone使用更小的学习率，HAMER使用统一学习率

### 2.2 学习率调度

**CS-ViT2:**
- 使用 `get_cosine_schedule_with_warmup`
- warmup步数：5000
- cosine周期：0.5

**HAMER:**
- 使用PyTorch Lightning的默认调度器（需要查看具体配置）

### 2.3 梯度裁剪

**CS-ViT2:**
```python
accelerator.clip_grad_norm_(net.parameters(), cfg.TRAIN.max_grad)  # max_grad=1.0
```

**HAMER:**
```python
if cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
    torch.nn.utils.clip_grad_norm_(params, cfg.TRAIN.GRAD_CLIP_VAL)
```

## 三、预处理对比

### 3.1 数据增强策略

**CS-ViT2:**
- **3D空间增强：**
  - Z轴旋转 (`rad`)
  - Z轴缩放 (`scale_z_range: [0.7, 1.3]`)
  - 透视旋转 (`persp_rot_max: π/12`)
- **2D图像增强：**
  - 焦距缩放 (`scale_f_range: [0.8, 1.2]`)
  - 主点噪声
  - 像素级增强（ColorJitter, GaussianNoise, MotionBlur等）
- **左右手翻转**

**HAMER:**
- **2D图像增强：**
  - 缩放 (`SCALE_FACTOR`)
  - 旋转 (`ROT_FACTOR`)
  - 平移 (`TRANS_FACTOR`)
  - 颜色缩放 (`COLOR_SCALE`)
  - 左右手翻转
- **Extreme Cropping** (可选)
- **没有3D空间增强**

**关键差异：** CS-ViT2有更复杂的3D空间增强，HAMER更注重2D图像增强

### 3.2 图像归一化

**CS-ViT2:**
```python
img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]
```

**HAMER:**
```python
DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])  # 注意：乘以255
DEFAULT_STD = 255. * np.array([0.229, 0.224, 0.225])
```

**差异：** HAMER的mean/std是CS-ViT2的255倍（因为HAMER图像值域是[0,255]，CS-ViT2是[0,1]）

### 3.3 Bbox处理

**CS-ViT2:**
- 使用 `patch_expanstion=1.1` 扩展bbox
- 支持动态计算patch_bbox

**HAMER:**
- 使用 `rescale_factor=2.0` 扩展bbox
- 支持 `BBOX_SHAPE` 调整宽高比

## 四、后处理和损失函数对比

### 4.1 损失函数

**CS-ViT2:**
```python
loss_kps3d_cam = Keypoint3DLoss(joint_cam_pred, joint_cam_gt, joint_valid)
loss_kps3d_rel = Keypoint3DLoss(joint_cam_pred - joint_cam_pred[:, :, :1],
                                 joint_rel, joint_valid)
loss_verts_rel = VertsLoss(verts_cam_pred - joint_cam_pred[:, :, :1],
                          verts_cam_gt - joint_cam_gt[:, :, :1], mano_valid)
loss_param = ParameterLoss(concat_params_pred, concat_params_gt, mano_valid)

# 最终损失
if supervise_global:
    loss = loss_kps3d_cam + loss_kps3d_rel + loss_verts_rel + loss_param
else:
    loss = loss_kps3d_rel + loss_verts_rel  # 不监督全局位置
```

**HAMER:**
```python
loss_keypoints_2d = Keypoint2DLoss(pred_keypoints_2d, gt_keypoints_2d)
loss_keypoints_3d = Keypoint3DLoss(pred_keypoints_3d, gt_keypoints_3d, pelvis_id=0)
loss_mano_params = ParameterLoss(pred_mano_params, gt_mano_params, has_mano_params)

loss = (LOSS_WEIGHTS['KEYPOINTS_3D'] * loss_keypoints_3d +
        LOSS_WEIGHTS['KEYPOINTS_2D'] * loss_keypoints_2d +
        sum([loss_mano_params[k] * LOSS_WEIGHTS[k.upper()] for k in loss_mano_params]))
```

**关键差异：**
1. **CS-ViT2** 使用相对坐标损失（相对于根关节），HAMER使用绝对坐标
2. **CS-ViT2** 有顶点损失，HAMER没有
3. **HAMER** 有2D重投影损失，CS-ViT2没有
4. **CS-ViT2** 支持 `supervise_global` 开关

### 4.2 损失计算细节

**CS-ViT2的Keypoint3DLoss:**
```python
# 先对坐标维度求均值，再应用mask
raw_loss = self.loss_fn(pred, gt)  # [B, T, J, 3]
raw_loss = torch.mean(raw_loss, dim=-1)  # [B, T, J]
return robust_masked_mean(raw_loss, valid)
```

**HAMER的Keypoint3DLoss:**
```python
# 先减去根关节，再计算损失
pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, pelvis_id, :]
gt_keypoints_3d = gt_keypoints_3d - gt_keypoints_3d[:, pelvis_id, :]
loss = (conf * self.loss_fn(pred_keypoints_3d, gt_keypoints_3d)).sum()
```

## 五、关键差异总结

### 5.1 可能导致收敛问题的差异

1. **残差连接缺失** ⚠️
   - CS-ViT2的MANO head输出层没有残差连接（代码中注释掉了）
   - HAMER有残差连接，有助于训练稳定性

2. **初始化方式不同**
   - CS-ViT2: `use_mean_init=False` 时随机初始化
   - HAMER: 默认使用mean params初始化

3. **IEF迭代优化缺失**
   - CS-ViT2: 单次前向传播
   - HAMER: 支持多次迭代优化（`IEF_ITERS`）

4. **损失函数设计**
   - CS-ViT2: 主要使用相对坐标损失
   - HAMER: 使用绝对坐标 + 2D重投影损失

5. **Backbone学习率**
   - CS-ViT2: backbone_lr = 1e-5 (更小)
   - HAMER: 统一学习率

6. **数据增强强度**
   - CS-ViT2: 更强的3D空间增强
   - HAMER: 更注重2D图像增强

## 六、对齐建议

### 6.1 立即修复（高优先级）

1. **恢复残差连接**
```python
# 在 src/model/module.py 的 MANOTransformerDecoderHead.forward 中
pred_hand_pose = self.decpose(token_out) + init_hand_pose  # 取消注释
pred_betas = self.decshape(token_out) + init_betas
pred_cam = self.deccam(token_out) + init_cam
```

2. **启用mean初始化**
```yaml
# 在 config/default_stage1-handec-large.yaml 中
handec:
  use_mean_init: true  # 改为 true
```

3. **添加IEF迭代优化**
```python
# 在 MANOTransformerDecoderHead 中添加迭代循环
for i in range(self.cfg.MODEL.handec.get('ief_iters', 1)):
    token_out = self.transformer(token, context=x)
    pred_hand_pose = self.decpose(token_out) + pred_hand_pose
    pred_betas = self.decshape(token_out) + pred_betas
    pred_cam = self.deccam(token_out) + pred_cam
```

### 6.2 训练策略调整（中优先级）

4. **调整学习率策略**
```yaml
# 尝试统一学习率（如HAMER）
TRAIN:
  lr: 1e-4
  backbone_lr: 1e-4  # 与regressor相同
```

5. **添加2D重投影损失**
```python
# 在 net.py 的 forward 中添加
pred_keypoints_2d = perspective_projection(
    joint_cam_pred,
    translation=trans_pred,
    focal_length=focal / img_size
)
loss_kps2d = Keypoint2DLoss(pred_keypoints_2d, batch["joint_img"], batch["joint_valid"])
```

6. **调整损失权重**
```yaml
# 参考HAMER的损失权重设计
LOSS:
  kps3d_weight: 1.0
  kps2d_weight: 1.0  # 新增
  verts_weight: 1.0
  param_weight: 0.1  # 可能需要降低
```

### 6.3 数据增强调整（低优先级）

7. **简化数据增强**
   - 暂时减少3D空间增强的强度
   - 增加2D图像增强（如HAMER）

8. **调整bbox扩展比例**
```yaml
TRAIN:
  expansion_ratio: 2.0  # 从1.1改为2.0，对齐HAMER的rescale_factor
```

### 6.4 Backbone调整（可选）

9. **对齐backbone输入**
   - 考虑裁剪图像为192x192（如HAMER的 `[:,:,:,32:-32]`）
   - 或调整backbone配置以匹配HAMER的输出格式

## 七、实施优先级

**P0 (必须修复):**
1. 恢复残差连接
2. 启用mean初始化

**P1 (强烈建议):**
3. 添加IEF迭代优化
4. 调整学习率策略
5. 添加2D重投影损失

**P2 (可选优化):**
6. 调整损失权重
7. 简化数据增强
8. 调整bbox扩展比例

## 八、验证方法

实施每个修改后，建议：
1. 监控训练loss曲线，确保下降趋势
2. 对比验证集MPJPE/MPVPE指标
3. 检查梯度范数，确保训练稳定
4. 可视化预测结果，检查合理性
