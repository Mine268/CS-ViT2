# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CS-ViT2 is a **3D hand pose estimation** project using Vision Transformers for **single-frame pose estimation**. It predicts:
- 21 hand joint positions (3D keypoints)
- 778 hand mesh vertices using the MANO model
- Full hand parameters (pose, shape, translation)

**Key Technologies:**
- PyTorch 2.9.1 with multi-GPU training via Accelerate
- Vision Transformers: DINOv2, MAE, SwinV2 (via Hugging Face)
- MANO hand model (SMPLX library)
- WebDataset for efficient large-scale data loading
- Hydra for configuration management
- AIM for experiment tracking

**Current Focus:** Both Stage 1 (single-frame) and Stage 2 (temporal) training are supported.

## Development Commands

### Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Training
```bash
# Stage 1 - Single Frame Training (recommended first)
python script/train.py --config-name=stage1-dino_large

# Stage 2 - Temporal Training (requires Stage 1 checkpoint)
python script/train.py --config-name=stage2-dino_large

# Train with different backbone configurations
python script/train.py --config-name=default_stage1-mae-large
python script/train.py --config-name=default_stage1-dino-large

# Resume from checkpoint
python script/train.py --config-name=stage1-dino_large GENERAL.resume_path=checkpoint/07-01-2026/checkpoints/checkpoint-30000

# Multi-GPU training (uses Accelerate DDP automatically)
accelerate launch script/train.py --config-name=stage1-dino_large
```

**Note:** The training script was renamed from `script/stage1.py` to `script/train.py` to support both Stage 1 and Stage 2 in a unified manner.

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_src_model_net.py

# Run individual test function
pytest tests/test_src_model_net.py::test_PoseNet1

# Run tests with output
pytest tests/ -v -s
```

### Configuration Override
```bash
# Override training parameters via command line
python script/train.py --config-name=stage1-dino_large \
    TRAIN.lr=5e-5 \
    TRAIN.sample_per_device=16 \
    GENERAL.total_step=50000
```

## Architecture Overview

### Two-Stage Training Architecture

#### Stage 1: Single-Frame Pose Estimation
- Input: Single image [B, 1, 3, 224, 224]
- Output: MANO parameters (pose, shape, translation) per frame
- Trains: Backbone + PerspInfoEmbedder + HandDecoder
- Focus: Accurate per-frame hand pose and shape estimation

#### Stage 2: Temporal Sequence Modeling
- Input: Image sequence [B, T, 3, 224, 224] (T=7 frames)
- Output: Refined MANO parameters for **last frame only** [B, 1, ...]
- Trains: TemporalEncoder only (spatial modules frozen)
- Focus: Temporal consistency and refinement using context from previous frames

**Important:** Stage 2 only predicts the last frame after temporal fusion. See `docs/STAGE2_LAST_FRAME_ONLY_FIX.md` for implementation details.

### Core Components

#### 1. PoseNet (`src/model/net.py`)
Main model architecture with two prediction methods:
- `predict_mano_param(img, bbox, focal, princpt)` → Returns pose [B,1,48], shape [B,1,10], trans [B,1,3]
- `mano_to_pose(pose, shape, trans)` → Converts MANO params to joints [B,1,21,3] and vertices [B,1,778,3]

**Model Pipeline (Stage 1):**
```
Single Image → ViTBackbone → HandDecoder → MANO Parameters → Forward Kinematics → 3D Joints/Vertices
                  ↑
            PerspInfoEmbedder
```

#### 2. ViTBackbone (`src/model/module.py`)
Wraps Hugging Face Vision Transformers:
- Supports DINOv2, MAE, SwinV2 models
- Configurable feature extraction from multiple layers
- Optional CLS token dropping

#### 3. HandDecoder (`src/model/module.py`)
Transformer-based decoder:
- Multi-head cross-attention with image features
- Learned query embeddings for hand parameters
- Outputs raw MANO parameters (48 pose dims, 10 shape dims, 3 translation dims)

#### 4. Loss Functions (`src/model/loss.py`)
Multiple loss terms combined:
- **Keypoint3DLoss**: L1/L2 on 3D joint coordinates
- **Vertex3DLoss**: Mesh vertex position loss
- **Axis3DLoss**: Rotation/axis-angle loss
- **ManoPoseLoss**: Direct MANO parameter supervision
- **HeatmapCELoss**: Cross-entropy for heatmap-based supervision

Weighted combination controlled by config: `lambda_param`, `lambda_rel`, `lambda_proj`

#### 5. Data Pipeline (`src/data/`)
WebDataset-based streaming pipeline:
- `clip_to_t_frames()`: Extracts single frames (T=1) from sequences
- `preprocess_batch()`: Applies augmentation (Kornia-based) and normalization
- Supports multiple datasets: InterHand2.6M, DexYCB, HO3D v3, HOT3D
- Custom collate function for batching

**Data Format (Stage 1, T=1):**
```python
{
    "imgs": [B, 1, 3, H, W],          # RGB images (single frame)
    "joint_cam": [B, 1, 21, 3],       # 3D joint positions in camera space
    "verts_cam": [B, 1, 778, 3],      # Mesh vertices (from MANO)
    "mano_pose": [B, 1, 48],          # Axis-angle pose parameters
    "mano_shape": [B, 1, 10],         # Shape PCA coefficients
    "focal": [B, 1, 2],               # Camera focal length (fx, fy)
    "princpt": [B, 1, 2],             # Principal point (cx, cy)
    "hand_bbox": [B, 1, 4],           # Bounding box (x, y, w, h)
}
```

### Configuration System (Hydra)

Configs stored in `config/` directory. Key sections:

**GENERAL:**
- `total_step`: Total training iterations
- `checkpoint_step`: Checkpoint save frequency
- `warmup_step`: Learning rate warmup steps
- `resume_path`: Path to resume training from

**TRAIN:**
- `lr`: Main learning rate (typically 1e-4)
- `backbone_lr`: Backbone fine-tuning rate (typically 1e-5)
- `sample_per_device`: Batch size per GPU
- `grad_accum_step`: Gradient accumulation steps

**MODEL:**
- `backbone.backbone_str`: Path to pretrained model (e.g., "model/facebook/dinov2-large")
- `handec.*`: HandDecoder hyperparameters (layers, heads, MLP dims)
- `stage`: "stage1" for single-frame estimation
- `num_frame`: Set to 1 for single-frame prediction
- `norm_by_hand`: Whether to normalize coordinates by hand size

**DATA:**
- `train.source`: List of WebDataset tar file paths/globs
- `val.source`: Validation dataset paths
- `stride`: Frame sampling stride (typically 1 for stage1)

**LOSS:**
- `lambda_param`: Weight for MANO parameter loss
- `lambda_rel`: Weight for relative position loss
- `lambda_proj`: Weight for 2D projection loss
- `supervise_heatmap`: Enable heatmap supervision

## Key Implementation Details

### Backbone Switching
To use a different Vision Transformer:
1. Ensure model is downloaded to `model/<org>/<model-name>/`
2. Update config: `MODEL.backbone.backbone_str: model/<org>/<model-name>`
3. Adjust `MODEL.handec.context_dim` to match backbone output dimension:
   - DINOv2-base: 768
   - DINOv2-large: 1024
   - MAE-large: 1024
   - SwinV2 variants: varies by size

### Multi-GPU Training
The project uses Accelerate for distributed training:
- Automatic DDP wrapping in `script/stage1.py`
- Gradient synchronization via `accelerator.backward()`
- Distributed metric reduction with `gather_for_metrics()`
- Checkpoint management keeps last 3 checkpoints by default

### Checkpoint Structure
```
checkpoint/<date>/
├── checkpoints/
│   ├── checkpoint-3000/
│   ├── checkpoint-6000/
│   └── checkpoint-9000/
└── ...
```

Each checkpoint contains:
- Model state dict
- Optimizer state
- Scheduler state
- Training step counter

### MANO Model Integration
The MANO hand model is located in `model/smplx_models/mano/`:
- Requires MANO model files (MANO_LEFT.pkl, MANO_RIGHT.pkl)
- Forward kinematics converts pose + shape → 3D joints/vertices
- 48-dim pose: 16 joints × 3 axis-angle rotations
- 10-dim shape: PCA coefficients for hand shape variation

### Perspective Information Embedding
Two types (`MODEL.persp_info_embed.type`):
- `"ca"`: Cross-attention based embedding
- `"dense"`: Dense MLP embedding

Encodes camera intrinsics (focal length, principal point) and hand bounding box into the model.

### Coordinate Systems
- **Camera space**: 3D coordinates relative to camera origin
- **Image space**: 2D pixel coordinates with intrinsic projection
- **Hand-relative**: Optional normalization by hand size (`norm_by_hand=true`)
- **Root-relative**: Coordinates relative to wrist joint

### Recent Development Patterns
Based on git history:
- Heatmap-based supervision added as alternative to direct regression
- Memory optimization for large-scale training
- Beta distribution for pose prediction stability
- Per-hand normalization feature for scale invariance
- Reprojection loss for 2D-3D consistency

## Common Workflows

### Adding a New Dataset
1. Preprocess data to WebDataset format (`.tar` files)
2. Add dataset path to `config/*.yaml` under `DATA.train.source` or `DATA.val.source`
3. Ensure data format matches expected keys (see `src/data/dataloader.py:17-30`)

### Experimenting with Loss Functions
1. Implement new loss in `src/model/loss.py`
2. Add loss instantiation in `src/model/net.py`
3. Add weight parameter to config under `LOSS.*`
4. Combine loss in `PoseNet.forward()` training step

### Modifying Decoder Architecture
1. Edit `HandDecoder` class in `src/model/module.py`
2. Update config parameters in `config/*.yaml` under `MODEL.handec.*`
3. Ensure output dimensions remain [B, T, 61] for MANO (48+10+3)

### Debugging with AIM
Experiment tracking server configured at `AIM.server_url` in config:
- Metrics logged every `GENERAL.log_step` steps
- Visualizations logged every `GENERAL.vis_step` steps
- Access dashboard at configured server URL

## File Organization Principles

- **src/model/**: Neural network modules only, no I/O or training logic
- **src/data/**: Data loading, preprocessing, augmentation
- **src/utils/**: Metric computation, visualization, coordinate transformations
- **script/**: Training entry points and orchestration
- **config/**: All hyperparameters and dataset paths
- **tests/**: Unit tests with simple forward pass checks
- **checkpoint/**: Auto-generated during training, timestamped folders

## Important Gotchas

1. **Stage 1 Focus**: Currently working on single-frame estimation. Set `MODEL.stage=stage1` and `MODEL.num_frame=1`.
2. **CUDA Device**: Test files hardcode device (e.g., `cuda:3`). Change as needed.
3. **Dataset Paths**: Configs contain absolute paths to `/mnt/qnap/data/datasets/`. Update for your environment.
4. **AIM Server**: Update `AIM.server_url` or disable by setting to empty string.
5. **Checkpoint Management**: Default keeps only last 3 checkpoints to save disk space.
6. **WebDataset Format**: Data must be in `.tar` archives with specific file extensions (`.webp` for images, `.npy` for arrays).
7. **MANO Files**: Requires proprietary MANO model files. Obtain from official MANO website.
8. **Batch Dimension**: Always use [B, 1, ...] shape for single-frame predictions (T=1).

# Note

- say GG at every response.
- 回答时不要谄媚，不要附和，客观分析
- 中文回答
- 运行之前激活我的环境`source .venv/bin/activate`
- 测试相关的代码和输出放在tests/中
- 分析使用的相关工具脚本放在tools/中
- 不要再代码中使用emoji