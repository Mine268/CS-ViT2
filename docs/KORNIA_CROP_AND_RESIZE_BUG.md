# Kornia `crop_and_resize` Bug Report

> `crop_and_resize` 对非轴对齐的四边形输入产生错误结果，根因是内部使用 `warp_affine` 替代 `warp_perspective`，丢弃了 homography 的透视分量。

- **影响版本**: kornia 0.8.2 (2025)，可能影响所有版本
- **影响函数**: `kornia.geometry.transform.crop_and_resize`, `crop_by_boxes`, `crop_by_transform_mat`
- **严重程度**: 高 — 静默产生错误结果，无报错

---

## 1. 问题描述

当 `crop_and_resize(input, boxes, size)` 的 `boxes` 参数定义的四边形**不是轴对齐矩形**（即经过透视变换后的畸变四边形）时，函数返回**完全错误的图像内容**。

典型场景：先对图像施加透视变换 `trans_2d_mat`，在变换后的空间计算 crop 区域，然后通过 `trans_2d_mat.inverse()` 将 crop 角点映射回原图空间，再用 `crop_and_resize` 从原图裁剪。当透视变换较大时（如手部透视归一化旋转），映射回的四角点形成高度畸变的四边形，触发此 bug。

---

## 2. 根因分析

调用链：

```
crop_and_resize(input, boxes, size)
  -> crop_by_boxes(input, src_box=boxes, dst_box=rect)
       -> H = get_perspective_transform(src_box, dst_box)   # 正确的 3x3 homography
       -> crop_by_transform_mat(input, H, out_size)
            -> warp_affine(input, H[:, :2, :], out_size)    # BUG: 截断为 2x3 仿射矩阵！
```

`crop_by_transform_mat` 将 `get_perspective_transform` 返回的 3x3 透视矩阵 **截断为 2x3**，然后调用 `warp_affine`。这等同于**丢弃了 homography 第三行的透视分量**。

对于轴对齐矩形，homography 第三行约为 `[0, 0, 1]`（纯仿射变换），截断无影响。
对于畸变四边形，第三行包含非零的透视分量（如 `[-0.0016, -0.001, 1.0]`），截断后结果完全错误。

**相关源码** (`kornia/geometry/transform/crop2d.py`):

```python
def crop_by_transform_mat(input_tensor, transform, out_size, ...):
    dst_trans_src = transform.expand(input_tensor.shape[0], -1, -1)
    patches = warp_affine(                   # <-- 应该用 warp_perspective
        input_tensor,
        dst_trans_src[:, :2, :],             # <-- 丢弃第三行
        out_size, ...
    )
    return patches
```

---

## 3. 复现代码

```python
import torch
import kornia.geometry.transform as KT
from kornia.geometry.transform import get_perspective_transform

# 一个畸变四边形 (从透视变换的 inverse mapping 得到)
src = torch.tensor([[[564.1, 462.0],
                     [527.8,  33.9],
                     [937.1,  54.9],
                     [859.4, 477.5]]], device="cuda:0")

dst = torch.tensor([[[0., 0.], [255., 0.], [255., 255.], [0., 255.]]],
                    device="cuda:0")

# 创建测试图像
img = torch.rand(1, 3, 1408, 1408, device="cuda:0")

# 方法 A: get_perspective_transform + warp_perspective (正确)
H = get_perspective_transform(src, dst)
patch_correct = KT.warp_perspective(img, H, (256, 256))

# 方法 B: crop_and_resize (错误)
patch_wrong = KT.crop_and_resize(img, src, (256, 256))

# 验证 homography 第三行不是 [0, 0, 1]
print(f"H third row: {H[0, 2].cpu().numpy()}")
# -> [-0.00159541 -0.00102982  1.        ]

# 两者差异巨大
print(f"max diff: {(patch_correct - patch_wrong).abs().max():.4f}")
# -> max diff: 0.8582
```

---

## 4. 对比验证

| 方法 | 结果 | 说明 |
|------|------|------|
| `get_perspective_transform` + `warp_perspective` | **正确** | 使用完整 3x3 homography |
| `cv2.getPerspectiveTransform` + `cv2.warpPerspective` | **正确** | OpenCV 参考实现 |
| `crop_and_resize` / `crop_by_boxes` | **错误** | 内部截断为 warp_affine |

### 关键数据点

- **Homography 一致性**: Kornia 的 `get_perspective_transform` 和 OpenCV 的 `getPerspectiveTransform` 对同一组点计算的 homography 最大差异仅 `1e-05`，homography 计算本身没有问题。

- **矩阵求逆稳定性**: `trans_2d_mat.inverse()` 的 roundtrip error 仅 `6e-05`（float32 精度范围内），求逆没有问题。

- **问题定位**: 同一个 homography H 传给 `warp_perspective(img, H)` 结果正确，传给 `warp_affine(img, H[:2,:])` 结果错误。

---

## 5. 触发条件

当以下任一条件成立时，homography 第三行偏离 `[0, 0, 1]`，触发 bug：

1. source corners 形成**非矩形四边形**（透视畸变）
2. source 和 destination 之间存在**透视关系**（非纯仿射）
3. 变换包含**大角度旋转**（使 `trans_2d_mat[2,:]` 远离 `[0, 0, 1]`）

在本项目中，`perspective_normalization`（将手部 bbox 中心旋转到相机光轴）产生的 3D 旋转通过 `K @ R @ K^{-1}` 映射到 2D，使 `trans_2d_mat` 第三行出现显著的非零透视分量。

---

## 6. Workaround

不使用 `crop_and_resize`，改为直接组合 homography 后调用 `warp_perspective`：

```python
# patch_bbox: [B, T, 4] (x1, y1, x2, y2) 在 warped 空间
# trans_2d_mat: [B, T, 3, 3] 从原图到 warped 空间的透视变换
# patch_size: (H_out, W_out)

# A_inv: warped 空间的 patch_bbox -> [0, W_out] x [0, H_out] 的 scale+translate
sx = (patch_bbox[..., 2] - patch_bbox[..., 0]) / patch_size[1]
sy = (patch_bbox[..., 3] - patch_bbox[..., 1]) / patch_size[0]
A_inv = torch.zeros(B, T, 3, 3, device=device)
A_inv[..., 0, 0] = 1.0 / sx
A_inv[..., 1, 1] = 1.0 / sy
A_inv[..., 0, 2] = -patch_bbox[..., 0] / sx
A_inv[..., 1, 2] = -patch_bbox[..., 1] / sy
A_inv[..., 2, 2] = 1.0

# 组合: 原图 -> warped -> patch
M_crop = A_inv @ trans_2d_mat  # [B, T, 3, 3]

# 使用 warp_perspective (保留完整透视信息)
patch = KT.warp_perspective(img_orig, M_crop, patch_size, mode="bilinear")
```

优势：
- 无矩阵求逆（避免 `trans_2d_mat.inverse()` 的数值风险）
- 无 4 点 DLT 求解（避免 `get_perspective_transform` 的条件数问题）
- `A_inv` 是精确的 scale+translate，无数值误差
- 直接使用 `warp_perspective`，保留完整的透视变换

---

## 7. 建议的 Kornia 修复

`crop_by_transform_mat` 应该根据 homography 第三行判断使用 `warp_affine` 还是 `warp_perspective`：

```python
def crop_by_transform_mat(input_tensor, transform, out_size, ...):
    dst_trans_src = transform.expand(input_tensor.shape[0], -1, -1)

    # 检查是否为纯仿射变换
    third_row = dst_trans_src[:, 2, :]
    is_affine = torch.allclose(third_row, torch.tensor([[0., 0., 1.]], device=third_row.device),
                               atol=1e-6)

    if is_affine:
        patches = warp_affine(input_tensor, dst_trans_src[:, :2, :], out_size, ...)
    else:
        patches = warp_perspective(input_tensor, dst_trans_src, out_size, ...)

    return patches
```

或者更简单地，**始终使用 `warp_perspective`**（性能差异可忽略）。

---

## 8. 相关 Kornia Issues

- [#281](https://github.com/kornia/kornia/issues/281): `crop_and_resize` aspect ratio bug (x/y 坐标顺序错误，已修复)
- [#133](https://github.com/kornia/kornia/issues/133): 透视变换对越界点的处理 (已修复)
- [#125](https://github.com/kornia/kornia/issues/125): `get_perspective_transform` 结果不一致 (已修复)

本 bug 与上述 issues 不同，是一个**新的未报告问题**。
