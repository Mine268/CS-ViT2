# 透视矫正 (Center Correction) 设计方案

## 动机

当前网络需要输入手部裁切图、裁切框位置和内参来估计相机空间的手部姿态。网络同时承担两个任务：判断手在图像中的位置、估计手部姿态。

透视矫正的思路是：在预处理阶段将手部 bbox 中心通过透视旋转矫正到主点位置，使手部始终位于光轴上，网络只需专注于姿态估计。验证时将预测结果逆旋转回原始空间与GT对比。

通过 YAML 配置 `TRAIN.center_correction` 动态切换新旧方式，默认 `false`。

---

## 数学原理

给定 bbox 中心 `(cx_hand, cy_hand)`、内参 `focal=(fx,fy)`、主点 `princpt=(cx,cy)`:

```
ray = [(cx_hand - cx)/fx, (cy_hand - cy)/fy, 1]     // bbox中心的归一化相机方向
ray_norm = ray / ||ray||
axis = cross(ray_norm, [0,0,1]) = [ry, -rx, 0]       // 旋转轴（XY平面内）
angle = arccos(rz)                                     // 旋转角度
axis_angle = axis / sin(angle) * angle                 // 轴角表示
R_correction = rodrigues(axis_angle)                    // 3x3旋转矩阵
```

`R_correction` 满足 `R_correction @ ray_norm = [0, 0, 1]`，将 bbox 方向旋转到光轴。

### 变换组合顺序

点以行向量表示: `j_new = j_old @ M^T`

```
trans_3d_mat = R_aug_persp @ R_z_scale @ R_correction
```

执行顺序: 先矫正 -> 再Z旋转/缩放 -> 再增强透视旋转

实现方式：利用已有函数构建 base，再右乘矫正矩阵:

```python
trans_3d_base = get_trans_3d_mat(rad, scale_z, aug_axis_angle)   # = R_aug @ R_z_scale
trans_3d_mat  = trans_3d_base @ R_correction                      # 右乘

trans_2d_base = get_trans_2d_mat(...)                              # = K_new @ ... @ K_old_inv
correction_2d = K @ R_correction @ K_inv                           # 矫正在像素空间的等价homography
trans_2d_mat  = trans_2d_base @ correction_2d                      # 右乘
```

---

## 实现设计

### 1. `src/data/preprocess.py` — 核心修改

#### A. 新增 `compute_center_correction_rotation` 函数

位置: `apply_perspective_to_points` 之后、`preprocess_batch` 之前

```python
def compute_center_correction_rotation(hand_bbox, focal, princpt):
    """
    计算将bbox中心旋转到主点的透视矫正旋转矩阵。
    输入: hand_bbox [B,T,4], focal [B,T,2], princpt [B,T,2]
    输出: R_correction [B,T,3,3]
    """
    # 1. bbox中心归一化相机坐标
    cx_hand = (hand_bbox[..., 0] + hand_bbox[..., 2]) * 0.5
    cy_hand = (hand_bbox[..., 1] + hand_bbox[..., 3]) * 0.5
    rx = (cx_hand - princpt[..., 0]) / focal[..., 0]
    ry = (cy_hand - princpt[..., 1]) / focal[..., 1]
    rz = ones_like(rx)

    # 2. 归一化
    norm = sqrt(rx**2 + ry**2 + rz**2)
    rx, ry, rz = rx/norm, ry/norm, rz/norm

    # 3. 旋转轴和角度
    ax, ay = ry, -rx                                # cross(ray_norm, [0,0,1])
    sin_angle = sqrt(ax**2 + ay**2)
    angle = atan2(sin_angle, rz)

    # 4. 轴角表示 (数值稳定: 小角度时 angle/sin->1)
    safe_sin = clamp(sin_angle, min=1e-8)
    axis_angle = stack([ax/safe_sin*angle, ay/safe_sin*angle, zeros], dim=-1)
    axis_angle = masked_fill(axis_angle, sin_angle < 1e-7, 0.0)  # 退化处理

    # 5. 转为旋转矩阵
    R = KC.axis_angle_to_rotation_matrix(axis_angle.reshape(-1, 3))
    return R.reshape(*axis_angle.shape[:-1], 3, 3)
```

#### B. 修改 `preprocess_batch` 签名

```python
def preprocess_batch(
    ...,
    pixel_aug=None,
    center_correction: bool = False,    # 新增
) -> Tuple[dict, torch.Tensor, Optional[torch.Tensor]]:
    # 返回值变为 (batch, trans_2d_mat, correction_rot_mat)
```

#### C. 重构代码路径

将 `if not augmentation_flag:` 改为 `if not augmentation_flag and not center_correction:`:

```python
if not augmentation_flag and not center_correction:
    # === 原始非增强路径（完全不变） ===
    ...
    correction_rot_mat = None

else:
    # === 变换路径（增强 和/或 矫正） ===
    focal = batch_origin["focal"].to(device)
    princpt = batch_origin["princpt"].to(device)

    # 1. 设置增强参数
    if augmentation_flag:
        rad = ...; scale_z = ...; ...  # 现有随机参数生成代码
        persp_axis_angle = ...
    else:
        # identity 增强参数（仅矫正时使用）
        rad = zeros(B, T); scale_z = ones(B, T)
        focal_new = focal; princpt_new = princpt
        persp_axis_angle = zeros(B, T, 3)

    # 2. 计算矫正旋转
    correction_rot_mat = None
    if center_correction:
        hand_bbox_orig = batch_origin["hand_bbox"].to(device)
        correction_rot_mat = compute_center_correction_rotation(
            hand_bbox_orig, focal, princpt)  # [B,T,3,3]

    # 3. 构建变换矩阵（复用现有函数）
    trans_3d_mat = get_trans_3d_mat(rad, scale_z, persp_axis_angle)
    trans_2d_mat = get_trans_2d_mat(rad, 1/scale_z, focal, princpt,
                                     focal_new, princpt_new, persp_axis_angle)

    # 4. 右乘矫正
    if correction_rot_mat is not None:
        trans_3d_mat = trans_3d_mat @ correction_rot_mat
        # 构建 K 和 K_inv，计算像素空间矫正 homography
        old_intr, old_intr_inv = build_intrinsic_matrices(focal, princpt)
        correction_2d = old_intr @ correction_rot_mat @ old_intr_inv
        trans_2d_mat = trans_2d_mat @ correction_2d

    # 5. 后续变换逻辑（现有增强路径代码不变）
    #    - 应用 trans_3d_mat 到 joint_cam
    #    - 应用 MANO root 旋转（需包含矫正旋转）
    #    - 应用 trans_2d_mat 到 joint_img
    #    - 重新计算 hand_bbox, patch_bbox
    #    - 裁切 patch（trans_2d_mat.inverse() 回原图采样）
    #    - 像素增强
    #    - 更新 focal, princpt
```

#### D. MANO root 旋转需包含矫正

```python
# 完整旋转 = R_aug_persp @ R_z @ R_correction
root_rot = axis_angle_to_rotation_matrix([0, 0, rad])          # R_z
if correction_rot_mat is not None:
    root_rot = root_rot @ correction_rot_mat.reshape(-1, 3, 3)  # @ R_correction
if persp_rot_max > 0:
    R_aug = axis_angle_to_rotation_matrix(persp_axis_angle)
    root_rot = R_aug @ root_rot                                  # R_aug @
mano_root_new = root_rot @ mano_root_old
```

### 2. `script/stage1.py` — 调用方修改

#### `train()` 函数

```python
batch, trans_2d_mat, correction_rot_mat = preprocess_batch(
    ...,
    center_correction=cfg.TRAIN.get("center_correction", False),
)
# correction_rot_mat 训练时不使用
```

#### `val()` 函数

```python
batch, trans_2d_mat, correction_rot_mat = preprocess_batch(
    ...,
    center_correction=cfg.TRAIN.get("center_correction", False),
)

output = net(batch)

joint_cam_gt = batch["joint_cam"]
joint_cam_pred = output["result"]["joint_cam_pred"]
verts_cam_gt = output["result"]["verts_cam_gt"]
verts_cam_pred = output["result"]["verts_cam_pred"]

# 逆旋转到原始相机空间
if correction_rot_mat is not None:
    # 正变换: j_corr = j_orig @ R^T
    # 逆变换: j_orig = j_corr @ R
    joint_cam_pred = torch.einsum("...jd,...dn->...jn",
                                   joint_cam_pred, correction_rot_mat)
    joint_cam_gt = torch.einsum("...jd,...dn->...jn",
                                 joint_cam_gt, correction_rot_mat)
    verts_cam_pred = torch.einsum("...jd,...dn->...jn",
                                   verts_cam_pred, correction_rot_mat)
    verts_cam_gt = torch.einsum("...jd,...dn->...jn",
                                 verts_cam_gt, correction_rot_mat)

# 后续 metric 计算不变
```

旋转是等距变换，MPJPE = ||R(a-b)|| = ||a-b||，逆旋转不改变指标数值。但实现逆旋转保证了: (1) 结果在原始相机空间可用, (2) 可验证矫正正确性。

### 3. `config/stage1-dino_large.yaml` — 配置

```yaml
TRAIN:
  center_correction: false   # true=透视矫正, false=原始方式
```

---

## 验证方案

### 单元测试

| 测试 | 内容 |
|------|------|
| Identity | bbox中心==主点 -> R_correction = I |
| 方向对齐 | R_correction @ ray_norm = [0,0,1] |
| 往返 | 矫正->逆矫正 = 恒等 |
| 2D投影 | 矫正后3D点投影到主点附近 |
| MANO一致性 | 矫正后 MANO FK 与变换后 joint_cam 一致 |

### 集成测试

```bash
# 后向兼容
python script/stage1.py --config-name=stage1-dino_large \
    TRAIN.center_correction=false GENERAL.total_step=100

# 新方式
python script/stage1.py --config-name=stage1-dino_large \
    TRAIN.center_correction=true GENERAL.total_step=100
```

---

## 正确性保证

| 关注点 | 保证方式 |
|--------|----------|
| 2D homography 精确性 | `apply_perspective_to_points` 自动透视除法 |
| 数值稳定性（小角度） | `clamp(sin, min=1e-8)` + 零掩码 |
| MANO root 一致性 | 完整旋转 `R_aug @ R_z @ R_corr` 应用于 root |
| 后向兼容 | `center_correction=false` 时路径完全不变 |
| 3D/2D 同步 | 同一个 R_correction 通过 K/K_inv 转换到像素空间 |
| 变换顺序正确 | 矫正最先(右乘)，Z-rot/scale 居中，aug 最后(左乘) |
