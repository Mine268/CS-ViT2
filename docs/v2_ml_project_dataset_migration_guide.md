# V2 Dataset And Dataloader Migration Guide For Downstream ML Projects

## 1. Purpose

This document is for the downstream machine learning project team that currently uses the V1 dataset format and V1 dataloader/preprocess pipeline.

The goal is to let the team migrate to the new V2 dataset format and V2 dataloader with minimal ambiguity and minimal trial-and-error.

This guide covers:

- what changed from V1 to V2;
- the exact V2 storage schema;
- the exact V2 runtime interface;
- what the training code must change;
- compatibility behavior and transition strategy;
- usage examples;
- debugging and validation methods;
- common failure modes and how to detect them early.


## 2. Executive Summary

The V2 upgrade introduces three core changes:

1. supervision is split into explicit channels:
   `joint_2d_valid`, `joint_3d_valid`, `has_mano`, `has_intr`
2. data provenance is explicit:
   `data_source`, `source_split`, `source_index`
3. patch-space coordinates are no longer ambiguous:
   `joint_patch_origin` and `joint_patch_resized` are explicitly separated at runtime

The most important migration rule is:

- if the model consumes resized hand patches, the correct supervision target is `joint_patch_resized`
- do not use legacy `joint_patch_bbox` on resized patch tensors

The most important safety rule is:

- do not use `joint_valid` as a unified supervision mask anymore
- use `joint_2d_valid` for 2D logic
- use `joint_3d_valid` for 3D logic


## 3. Scope Of V2

The exporter side has been migrated for the following datasets:

- InterHand2.6M
- DexYCB
- HO3D train
- HO3D evaluation
- HOT3D train
- COCO-WholeBody train
- COCO-WholeBody val

Special note:

- HOT3D test is intentionally not exported in V2 because it does not provide usable hand pose annotations for this pipeline.


## 4. V1 To V2 Change Summary

### 4.1 What Stayed The Same

- Data is still stored as WebDataset tar shards.
- A single tar sample is still a hand clip sample.
- Most numeric fields still use the time-first layout `[T, ...]`.
- The downstream runtime still follows:
  `read tar -> decode -> dataloader -> preprocess -> model`

### 4.2 What Changed

- V1 `joint_valid` is no longer the primary validity signal.
- V1 `mano_valid` is no longer the primary MANO availability signal.
- V2 adds explicit source metadata.
- V2 supports `2D-only` samples such as COCO-WholeBody without pretending that they provide real 3D or real MANO labels.
- V2 runtime preprocessing explicitly distinguishes:
  - original image coordinates
  - bbox-local coordinates before resize
  - patch coordinates after resize

### 4.3 Why This Matters

In V1, training code could silently misuse:

- 2D-only labels as if they were 3D-capable
- placeholder MANO values as if they were GT
- patch-local origin coordinates as if they were resized patch-space coordinates

V2 removes that ambiguity.


## 5. Canonical V2 Storage Schema

This section describes the tar-side schema, meaning the keys physically stored in `.tar`.

### 5.1 Required Sample-Level Keys

| Key | Type After `wds.decode()` | Shape | Meaning |
| --- | --- | --- | --- |
| `__key__` | `str` | scalar | Unique sample key inside WebDataset |
| `imgs_path.json` | `list[str]` | length `T` | Relative image path per frame |
| `img_bytes.pickle` | `list[bytes]` | length `T` | Encoded image bytes per frame |
| `handedness.json` | `str` | scalar | `"right"` or `"left"` |
| `additional_desc.json` | `list[dict]` | length `T` | Auxiliary per-frame metadata |
| `data_source.json` | `str` | scalar | Dataset source name |
| `source_split.json` | `str` | scalar | Dataset split name |
| `source_index.json` | `list[dict]` | length `T` | Per-frame raw-dataset provenance |
| `intr_type.json` | `str` | scalar | `real`, `fixed_virtual`, `pseudo`, or `none` |

### 5.2 Required Numeric Keys

| Key | Dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `hand_bbox.npy` | `float32` | `[T, 4]` | Hand bbox in original image coordinates, format `[x1, y1, x2, y2]` |
| `joint_img.npy` | `float32` | `[T, 21, 2]` | 2D joints in original image coordinates |
| `joint_hand_bbox.npy` | `float32` | `[T, 21, 2]` | Legacy storage field for hand-local 2D joints |
| `joint_cam.npy` | `float32` | `[T, 21, 3]` | 3D camera-space joints in mm |
| `joint_rel.npy` | `float32` | `[T, 21, 3]` | Root-relative 3D joints in mm |
| `joint_2d_valid.npy` | `float32` | `[T, 21]` | 2D supervision mask |
| `joint_3d_valid.npy` | `float32` | `[T, 21]` | 3D supervision mask |
| `joint_valid.npy` | `float32` | `[T, 21]` | Legacy compatibility mask |
| `mano_pose.npy` | `float32` | `[T, 48]` | MANO pose |
| `mano_shape.npy` | `float32` | `[T, 10]` | MANO shape |
| `has_mano.npy` | `float32` | `[T]` | MANO availability mask |
| `mano_valid.npy` | `float32` | `[T]` | Legacy compatibility alias for MANO availability |
| `has_intr.npy` | `float32` | `[T]` | Intrinsics availability mask |
| `timestamp.npy` | `float32` | `[T]` | Timestamp in ms |
| `focal.npy` | `float32` | `[T, 2]` | `[fx, fy]` in original image coordinates |
| `princpt.npy` | `float32` | `[T, 2]` | `[cx, cy]` in original image coordinates |

### 5.3 Placeholder Rules

Some datasets do not provide all supervision types. V2 still writes a physically complete tensor layout.

Rules:

- if 3D supervision does not exist:
  `joint_cam` and `joint_rel` are placeholder arrays, and `joint_3d_valid` is zero
- if MANO supervision does not exist:
  `mano_pose` and `mano_shape` are placeholder arrays, and `has_mano` is zero
- if only virtual intrinsics exist:
  `has_intr = 1` and `intr_type = fixed_virtual`

Example:

- COCO-WholeBody is `2D-only`
- it has `joint_2d_valid = 1/0` from keypoint visibility
- it has `joint_3d_valid = 0`
- it has `has_mano = 0`
- it has `has_intr = 1`
- it has `intr_type = fixed_virtual`


## 6. Runtime Interface Used By The V2 Pipeline

This section is the most important part for the downstream ML project.

The tar-side keys above are normalized by:

- [schema_v2.py](/data_0/renkaiwen/webdatasetify-hand-datasets/src/schema_v2.py)
- [dataloader_v2.py](/data_0/renkaiwen/webdatasetify-hand-datasets/src/dataloader_v2.py)
- [preprocess_v2.py](/data_0/renkaiwen/webdatasetify-hand-datasets/src/preprocess_v2.py)

### 6.1 `get_dataloader(...)`

Source:
- [dataloader_v2.py](/data_0/renkaiwen/webdatasetify-hand-datasets/src/dataloader_v2.py)

Signature:

```python
get_dataloader(
    url,
    num_frames: int,
    stride: int,
    batch_size: int,
    num_workers: int,
    shardshuffle: int = 0,
    shuffle_buffer: int = 5000,
    default_data_source: Optional[str] = None,
    default_source_split: str = "unknown",
)
```

Behavior:

- reads WebDataset shards;
- decodes `.json`, `.pickle`, `.npy`;
- normalizes V1 and V2 tar samples into one runtime clip schema;
- slices clips into `num_frames` sub-clips with `stride`;
- decodes images into tensors;
- batches samples with a custom collate function.

### 6.2 Output Of `get_dataloader(...)`

Each yielded batch from `get_dataloader(...)` is a dictionary.

Important keys:

| Key | Runtime Type | Shape / Layout |
| --- | --- | --- |
| `__key__` | `list[str]` | length `B` |
| `imgs_path` | `list[list[str]]` | `[B][T]` |
| `imgs` | `list[torch.Tensor]` | each tensor is `[T, 3, H, W]`, image sizes may vary across samples |
| `handedness` | `list[str]` | length `B` |
| `data_source` | `list[str]` | length `B` |
| `source_split` | `list[str]` | length `B` |
| `source_index` | `list[list[dict]]` | `[B][T]` |
| `intr_type` | `list[str]` | length `B` |
| `additional_desc` | `list[list[dict]]` | `[B][T]` |
| numeric tensor fields | `torch.Tensor` | stacked to `[B, T, ...]` |

The numeric tensor fields include:

- `hand_bbox`
- `joint_img`
- `joint_hand_bbox`
- `joint_cam`
- `joint_rel`
- `joint_2d_valid`
- `joint_3d_valid`
- `joint_valid`
- `mano_pose`
- `mano_shape`
- `has_mano`
- `mano_valid`
- `has_intr`
- `timestamp`
- `focal`
- `princpt`

### 6.3 `preprocess_batch(...)`

Source:
- [preprocess_v2.py](/data_0/renkaiwen/webdatasetify-hand-datasets/src/preprocess_v2.py)

Signature:

```python
preprocess_batch(
    batch_origin,
    patch_size,
    patch_expanstion,
    scale_z_range,
    scale_f_range,
    persp_rot_max,
    augmentation_flag,
    device,
)
```

Returns:

```python
batch_out, trans_2d_mat
```

Where:

- `batch_out` is the model-facing batch
- `trans_2d_mat` is the per-sample per-frame image-space transform matrix

### 6.4 Output Of `preprocess_batch(...)`

The output dictionary includes:

| Key | Shape / Type | Meaning |
| --- | --- | --- |
| `__key__` | `list[str]` | Sample ids |
| `imgs_path` | `list[list[str]]` | Original frame paths |
| `handedness` | `list[str]` | Hand side |
| `data_source` | `list[str]` | Dataset source |
| `source_split` | `list[str]` | Source split |
| `source_index` | `list[list[dict]]` | Per-frame raw provenance |
| `intr_type` | `list[str]` | Intrinsics type |
| `additional_desc` | `list[list[dict]]` | Extra metadata |
| `flip` | `list[bool]` | Whether the sample is flipped to canonical orientation |
| `patches` | `[B, T, 3, H_patch, W_patch]` | Resized patch tensor fed to the model |
| `patch_bbox` | `[B, T, 4]` | Patch bbox in original image coordinates |
| `hand_bbox` | `[B, T, 4]` | Hand bbox in original image coordinates |
| `joint_img` | `[B, T, 21, 2]` | Original-image-space joints after augmentation |
| `joint_hand_origin` | `[B, T, 21, 2]` | Hand-local joints before resize |
| `joint_patch_origin` | `[B, T, 21, 2]` | Patch-local joints before resize |
| `joint_patch_resized` | `[B, T, 21, 2]` | Patch-local joints after scaling into `patches` space |
| `joint_hand_bbox` | `[B, T, 21, 2]` | Legacy alias of `joint_hand_origin` |
| `joint_patch_bbox` | `[B, T, 21, 2]` | Legacy alias of `joint_patch_origin` |
| `joint_cam` | `[B, T, 21, 3]` | 3D joints |
| `joint_rel` | `[B, T, 21, 3]` | Root-relative 3D joints |
| `joint_2d_valid` | `[B, T, 21]` | 2D supervision mask |
| `joint_3d_valid` | `[B, T, 21]` | 3D supervision mask |
| `joint_valid` | `[B, T, 21]` | Legacy alias of `joint_2d_valid` in V2 runtime |
| `mano_pose` | `[B, T, 48]` | MANO pose |
| `mano_shape` | `[B, T, 10]` | MANO shape |
| `has_mano` | `[B, T]` | MANO availability |
| `mano_valid` | `[B, T]` | Legacy alias of `has_mano` |
| `has_intr` | `[B, T]` | Intrinsics availability |
| `timestamp` | `[B, T]` | Timestamp |
| `focal` | `[B, T, 2]` | Focal |
| `princpt` | `[B, T, 2]` | Principal point |


## 7. Migration Checklist For The Downstream ML Project

### 7.1 Replace Imports

Old imports:

```python
from src.dataloader import get_dataloader
from src.preprocess import preprocess_batch
```

New imports:

```python
from src.dataloader_v2 import get_dataloader
from src.preprocess_v2 import preprocess_batch
```

### 7.2 Replace Supervision Masks

V1 code often assumes:

```python
joint_valid
mano_valid
```

V2 rules:

- `joint_2d_valid` is the 2D supervision mask
- `joint_3d_valid` is the 3D supervision mask
- `has_mano` is the MANO supervision mask
- `has_intr` is the intrinsics availability mask

Required migration:

- replace all 2D uses of `joint_valid` with `joint_2d_valid`
- replace all 3D uses of `joint_valid` with `joint_3d_valid`
- replace all uses of `mano_valid` with `has_mano`

### 7.3 Replace Patch Supervision Coordinates

If the model is trained on resized `patches`, then:

- do not use `joint_patch_bbox`
- do not use `joint_patch_origin`
- use `joint_patch_resized`

This is a critical migration rule.

### 7.4 Handle 2D-Only Samples Correctly

COCO-WholeBody samples intentionally have:

- `joint_3d_valid = 0`
- `has_mano = 0`
- `intr_type = fixed_virtual`

Required migration:

- 3D loss must be masked by `joint_3d_valid`
- MANO loss must be masked by `has_mano`
- any logic that treats `focal/princpt` as real calibration GT must check `intr_type == "real"`

### 7.5 Preserve Source Metadata

Do not discard:

- `data_source`
- `source_split`
- `source_index`

These fields are useful for:

- dataset-specific behavior;
- debugging;
- reproducing bad samples;
- filtered evaluation;
- provenance tracking.


## 8. Recommended Training-Side Adaptation

### 8.1 Minimum Safe Change Set

If the downstream model already works with V1 and you want the smallest possible migration:

1. switch imports to V2
2. gate losses with the new masks
3. switch patch supervision to `joint_patch_resized`
4. leave the rest of the model unchanged

### 8.2 Recommended Loss Gating

For 2D losses:

```python
mask_2d = batch["joint_2d_valid"] > 0.5
```

For 3D losses:

```python
mask_3d = batch["joint_3d_valid"] > 0.5
```

For MANO losses:

```python
mask_mano = batch["has_mano"] > 0.5
```

For intrinsics-dependent logic:

```python
mask_intr = batch["has_intr"] > 0.5
```

If the code needs real intrinsics only:

```python
real_intr = [
    (batch["intr_type"][i] == "real")
    for i in range(len(batch["intr_type"]))
]
```

### 8.3 Recommended Patch Supervision

If the model predicts 2D joints on `patches`:

```python
target_2d = batch["joint_patch_resized"]
mask_2d = batch["joint_2d_valid"]
```

If the model predicts 2D joints in original image coordinates:

```python
target_2d = batch["joint_img"]
mask_2d = batch["joint_2d_valid"]
```


## 9. V1 Compatibility Strategy

### 9.1 Good News

V2 dataloader can normalize both:

- new V2 tar files
- old V1 tar files

This is done in:
- [schema_v2.py](/data_0/renkaiwen/webdatasetify-hand-datasets/src/schema_v2.py)

### 9.2 Important Limitation

Compatibility only reconstructs what is inferable from V1.

That means:

- `joint_2d_valid` may fall back to legacy `joint_valid`
- `joint_3d_valid` may also fall back to legacy `joint_valid`
- `has_mano` may fall back to legacy `mano_valid`
- `data_source`, `source_split`, `source_index`, `intr_type` may be incomplete or inferred if the old tar did not contain them

For production training, regenerated V2 tar files are strongly recommended.

### 9.3 Recommended Transition Plan

Phase 1:
- keep the training loop structure
- switch to V2 loader/preprocess
- read both old and new tar files

Phase 2:
- regenerate V2 tar files for the full training set
- remove direct dependence on legacy semantics in training code

Phase 3:
- treat V1 compatibility only as fallback, not as the main path


## 10. Usage Examples

### 10.1 Minimal Read + Preprocess Example

```python
import glob
import torch

from src.dataloader_v2 import get_dataloader
from src.preprocess_v2 import preprocess_batch

urls = glob.glob("/mnt/qnap/data/datasets/webdatasets2/HO3D_v3/train/*.tar")

loader = get_dataloader(
    urls,
    num_frames=1,
    stride=1,
    batch_size=8,
    num_workers=4,
)

for batch_origin in loader:
    batch, trans_2d_mat = preprocess_batch(
        batch_origin,
        patch_size=(256, 256),
        patch_expanstion=1.1,
        scale_z_range=(0.9, 1.1),
        scale_f_range=(0.8, 1.1),
        persp_rot_max=float(torch.pi / 12),
        augmentation_flag=True,
        device=torch.device("cuda"),
    )

    patches = batch["patches"]
    target_2d = batch["joint_patch_resized"]
    mask_2d = batch["joint_2d_valid"]
    break
```

### 10.2 Mixed-Dataset Training Example

```python
urls = []
urls += glob.glob("/mnt/qnap/data/datasets/webdatasets2/InterHand2.6M/train/*.tar")
urls += glob.glob("/mnt/qnap/data/datasets/webdatasets2/DexYCB/s1/train/*.tar")
urls += glob.glob("/mnt/qnap/data/datasets/webdatasets2/HO3D_v3/train/*.tar")
urls += glob.glob("/mnt/qnap/data/datasets/webdatasets2/HOT3D/train/*.tar")
urls += glob.glob("/mnt/qnap/data/datasets/webdatasets2/COCO-WholeBody/train/*.tar")
```

Recommended practice:

- use `data_source` to log per-dataset batch composition
- mask 3D and MANO losses appropriately

### 10.3 Dataset-Aware Loss Example

```python
loss_2d = compute_2d_loss(pred_2d, batch["joint_patch_resized"], batch["joint_2d_valid"])

if torch.any(batch["joint_3d_valid"] > 0.5):
    loss_3d = compute_3d_loss(pred_3d, batch["joint_cam"], batch["joint_3d_valid"])
else:
    loss_3d = pred_3d.sum() * 0.0

if torch.any(batch["has_mano"] > 0.5):
    loss_mano = compute_mano_loss(pred_mano, batch["mano_pose"], batch["mano_shape"], batch["has_mano"])
else:
    loss_mano = pred_3d.sum() * 0.0
```


## 11. Common Migration Pitfalls

### Pitfall 1: Using `joint_valid` Everywhere

Symptom:

- 2D-only samples unexpectedly enter 3D loss
- placeholder 3D targets are treated as GT

Fix:

- split all supervision by the new masks

### Pitfall 2: Using `joint_patch_bbox` On Resized Patches

Symptom:

- patch-space supervision appears consistently shifted or scaled incorrectly

Fix:

- use `joint_patch_resized`

### Pitfall 3: Treating `intr_type=fixed_virtual` As Real Calibration

Symptom:

- camera-supervised losses behave inconsistently on COCO-WholeBody

Fix:

- use `has_intr` for geometric availability
- use `intr_type == "real"` when true real-calibration GT is required

### Pitfall 4: Assuming All Samples Have MANO

Symptom:

- crashes or meaningless loss on HO3D eval or COCO-WB

Fix:

- gate MANO losses with `has_mano`

### Pitfall 5: Dropping Source Metadata

Symptom:

- impossible to reproduce problematic samples

Fix:

- keep `data_source`, `source_split`, and `source_index` in logs or debug snapshots


## 12. Debugging Workflow

### 12.1 First-Level Checks

For any newly integrated dataset, check:

- `data_source`
- `source_split`
- `source_index`
- `intr_type`
- shapes of `joint_*`, `focal`, `princpt`
- whether `joint_2d_valid`, `joint_3d_valid`, and `has_mano` match expectations

### 12.2 Visual Verification

Use:
- [verify_wds_v2.py](/data_0/renkaiwen/webdatasetify-hand-datasets/src/verify_wds_v2.py)

Recommended images to inspect:

- `origin/hand_bbox-joint_img.png`
- `processed/bbox-joint_img.png`
- `processed/joint_patch_resized.png`
- `processed/joint_cam_reproj.png` for datasets with 3D
- `processed/mano.png` for datasets with MANO

### 12.3 Numeric Consistency Checks

For datasets with real 3D and intrinsics:

1. project `joint_cam` with `focal/princpt`
2. compare against `joint_img`
3. expected reprojection error should be near zero in the unaugmented path

This is the most reliable sanity check.

### 12.4 Sample Reproduction

Use `source_index` to go back to the raw dataset item.

Examples:

- InterHand2.6M:
  `capture_id`, `seq_name`, `cam_id`, `frame_idx`, `aid`
- DexYCB:
  `subject`, `capture`, `serial`, `frame_idx`
- HO3D:
  `seq_name`, `frame_name`
- HOT3D:
  `sequence_name`, `timestamp_ns`
- COCO-WB:
  `image_id`, `ann_id`


## 13. Validation Checklist Before Switching Production Training

The downstream ML project team should complete this checklist:

- V2 loader and preprocess imports are in place
- 2D, 3D, MANO, and intrinsics masks are handled separately
- patch supervision uses `joint_patch_resized`
- mixed-dataset training logs `data_source`
- sample-level debugging retains `source_index`
- `verify_wds_v2.py` has been run on every dataset used in production
- numeric reprojection checks have been run on every 3D dataset used in production
- at least one end-to-end training run has been completed without legacy-only assumptions


## 14. Recommended Rollout Strategy

1. Integrate V2 loader and preprocess into a feature branch.
2. Run one dataset at a time in the downstream training project.
3. Confirm loss gating and patch-space supervision are correct.
4. Enable mixed-dataset training only after all single-dataset checks pass.
5. Remove remaining V1-only assumptions after V2 tar regeneration is complete.


## 15. Final Recommendation

For a safe migration, the downstream team should treat this as an interface migration, not just a file-format migration.

The critical behavior changes are:

- explicit supervision gating;
- explicit provenance;
- explicit `origin` versus `resized` patch coordinates.

If these three points are handled correctly, the rest of the migration is mostly mechanical.
