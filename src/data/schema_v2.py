"""
V2 WebDataset schema helpers.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


LEGACY_NUMPY_KEYS = [
    "hand_bbox.npy",
    "joint_img.npy",
    "joint_hand_bbox.npy",
    "joint_cam.npy",
    "joint_rel.npy",
    "joint_valid.npy",
    "mano_pose.npy",
    "mano_shape.npy",
    "mano_valid.npy",
    "timestamp.npy",
    "focal.npy",
    "princpt.npy",
]

V2_NUMPY_KEYS = [
    "hand_bbox.npy",
    "joint_img.npy",
    "joint_hand_bbox.npy",
    "joint_cam.npy",
    "joint_rel.npy",
    "joint_2d_valid.npy",
    "joint_3d_valid.npy",
    "joint_valid.npy",
    "mano_pose.npy",
    "mano_shape.npy",
    "has_mano.npy",
    "mano_valid.npy",
    "has_intr.npy",
    "timestamp.npy",
    "focal.npy",
    "princpt.npy",
]

PER_FRAME_ARRAY_SPECS: Dict[str, Tuple[Tuple[int, ...], float]] = {
    "hand_bbox": ((4,), 0.0),
    "joint_img": ((21, 2), 0.0),
    "joint_hand_bbox": ((21, 2), 0.0),
    "joint_cam": ((21, 3), 0.0),
    "joint_rel": ((21, 3), 0.0),
    "joint_2d_valid": ((21,), 1.0),
    "joint_3d_valid": ((21,), 1.0),
    "joint_valid": ((21,), 1.0),
    "mano_pose": ((48,), 0.0),
    "mano_shape": ((10,), 0.0),
    "has_mano": ((), 0.0),
    "mano_valid": ((), 0.0),
    "has_intr": ((), 0.0),
    "timestamp": ((), 0.0),
    "focal": ((2,), 0.0),
    "princpt": ((2,), 0.0),
}

LIST_SLICE_KEYS = [
    "imgs_path",
    "imgs_bytes",
    "additional_desc",
    "source_index",
]

SCALAR_COPY_KEYS = [
    "handedness",
    "data_source",
    "source_split",
    "intr_type",
]


def _default_frame_array(
    num_frames: int,
    tail_shape: Tuple[int, ...],
    fill_value: float,
) -> np.ndarray:
    return np.full((num_frames, *tail_shape), fill_value, dtype=np.float32)


def _coerce_json_string(value: Any, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _coerce_json_list(value: Any, num_frames: int, default_factory) -> List[Any]:
    if value is None:
        return [default_factory(i) for i in range(num_frames)]
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        value = [value]
    if len(value) == num_frames:
        return value
    if len(value) == 1:
        return [value[0] for _ in range(num_frames)]
    raise ValueError(f"Expected list length {num_frames}, got {len(value)}")


def _coerce_image_bytes_list(value: Any, num_frames: int) -> List[bytes]:
    value_list = _coerce_json_list(value, num_frames, lambda _: b"")
    result: List[bytes] = []
    for idx, item in enumerate(value_list):
        if isinstance(item, bytes):
            result.append(item)
        elif isinstance(item, bytearray):
            result.append(bytes(item))
        elif isinstance(item, memoryview):
            result.append(item.tobytes())
        elif isinstance(item, np.ndarray):
            result.append(np.asarray(item, dtype=np.uint8).tobytes())
        else:
            raise TypeError(
                f"Unsupported image payload type at index {idx}: {type(item)!r}"
            )
    return result


def _coerce_frame_array(
    value: Any,
    num_frames: int,
    tail_shape: Tuple[int, ...],
    field_name: str,
    default_fill: Optional[float] = None,
) -> Optional[np.ndarray]:
    if value is None:
        if default_fill is None:
            return None
        return _default_frame_array(num_frames, tail_shape, default_fill)

    arr = np.asarray(value)
    if arr.ndim == len(tail_shape):
        if num_frames != 1:
            raise ValueError(f"{field_name} is missing the frame axis")
        arr = arr[None]
    if arr.ndim != len(tail_shape) + 1:
        raise ValueError(
            f"{field_name} expected ndim={len(tail_shape) + 1}, got {arr.ndim}"
        )
    if arr.shape[0] != num_frames:
        raise ValueError(
            f"{field_name} expected {num_frames} frames, got {arr.shape[0]}"
        )
    if tuple(arr.shape[1:]) != tail_shape:
        raise ValueError(
            f"{field_name} expected tail shape {tail_shape}, got {arr.shape[1:]}"
        )
    return arr.astype(np.float32, copy=False)


def infer_num_frames(decoded_sample: Dict[str, Any]) -> int:
    if "img_bytes.pickle" in decoded_sample:
        return len(decoded_sample["img_bytes.pickle"])
    if "imgs_path.json" in decoded_sample:
        return len(decoded_sample["imgs_path.json"])

    for key in LEGACY_NUMPY_KEYS + V2_NUMPY_KEYS:
        if key not in decoded_sample:
            continue
        arr = np.asarray(decoded_sample[key])
        if arr.ndim >= 1:
            return int(arr.shape[0])

    raise KeyError("Failed to infer the number of frames from the decoded sample")


def infer_data_source_from_key(sample_key: str) -> str:
    if sample_key.endswith("_ho3d"):
        return "HO3D_v3"
    if sample_key.endswith("_dexycb"):
        return "DexYCB"
    if sample_key.endswith("_hot3d"):
        return "HOT3D"

    key_parts = sample_key.split("_")
    if len(key_parts) >= 5 and key_parts[-2] in {"l", "r"}:
        return "InterHand2.6M"

    return "unknown"


def _normalize_source_index(
    value: Any,
    num_frames: int,
    sample_key: str,
    data_source: str,
    additional_desc: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if value is None:
        return [
            {
                "sample_key": sample_key,
                "data_source": data_source,
                "frame_idx_within_clip": idx,
                **(
                    additional_desc[idx]
                    if isinstance(additional_desc[idx], dict)
                    else {}
                ),
            }
            for idx in range(num_frames)
        ]

    if isinstance(value, dict):
        value = [value for _ in range(num_frames)]
    value_list = _coerce_json_list(
        value, num_frames, lambda idx: {"frame_idx_within_clip": idx}
    )

    result: List[Dict[str, Any]] = []
    for idx, item in enumerate(value_list):
        if not isinstance(item, dict):
            item = {"value": item}
        merged = dict(item)
        merged.setdefault("frame_idx_within_clip", idx)
        result.append(merged)
    return result


def normalize_decoded_clip_sample(
    decoded_sample: Dict[str, Any],
    default_data_source: Optional[str] = None,
    default_source_split: str = "unknown",
) -> Dict[str, Any]:
    num_frames = infer_num_frames(decoded_sample)
    sample_key = _coerce_json_string(decoded_sample.get("__key__"), "unknown")

    imgs_path = _coerce_json_list(
        decoded_sample.get("imgs_path.json"),
        num_frames,
        lambda idx: f"{sample_key}::{idx:04d}",
    )
    imgs_bytes = _coerce_image_bytes_list(
        decoded_sample.get("img_bytes.pickle"), num_frames
    )
    handedness = _coerce_json_string(decoded_sample.get("handedness.json"), "unknown")
    additional_desc = _coerce_json_list(
        decoded_sample.get("additional_desc.json"), num_frames, lambda _: {}
    )

    data_source = _coerce_json_string(
        decoded_sample.get("data_source.json"),
        default_data_source or infer_data_source_from_key(sample_key),
    )
    source_split = _coerce_json_string(
        decoded_sample.get("source_split.json"), default_source_split
    )

    joint_valid_legacy = _coerce_frame_array(
        decoded_sample.get("joint_valid.npy"),
        num_frames,
        PER_FRAME_ARRAY_SPECS["joint_valid"][0],
        "joint_valid.npy",
        default_fill=None,
    )
    if joint_valid_legacy is None:
        joint_valid_legacy = _default_frame_array(
            num_frames, PER_FRAME_ARRAY_SPECS["joint_valid"][0], 1.0
        )

    mano_valid_legacy = _coerce_frame_array(
        decoded_sample.get("mano_valid.npy"),
        num_frames,
        PER_FRAME_ARRAY_SPECS["mano_valid"][0],
        "mano_valid.npy",
        default_fill=None,
    )

    has_intr = _coerce_frame_array(
        decoded_sample.get("has_intr.npy"),
        num_frames,
        PER_FRAME_ARRAY_SPECS["has_intr"][0],
        "has_intr.npy",
        default_fill=1.0
        if "focal.npy" in decoded_sample and "princpt.npy" in decoded_sample
        else 0.0,
    )
    assert has_intr is not None

    intr_type = _coerce_json_string(
        decoded_sample.get("intr_type.json"),
        "real" if bool(np.all(has_intr > 0.5)) else "none",
    )

    source_index = _normalize_source_index(
        decoded_sample.get("source_index.json"),
        num_frames,
        sample_key,
        data_source,
        additional_desc,
    )

    hand_bbox = _coerce_frame_array(
        decoded_sample.get("hand_bbox.npy"),
        num_frames,
        PER_FRAME_ARRAY_SPECS["hand_bbox"][0],
        "hand_bbox.npy",
        default_fill=PER_FRAME_ARRAY_SPECS["hand_bbox"][1],
    )
    joint_img = _coerce_frame_array(
        decoded_sample.get("joint_img.npy"),
        num_frames,
        PER_FRAME_ARRAY_SPECS["joint_img"][0],
        "joint_img.npy",
        default_fill=PER_FRAME_ARRAY_SPECS["joint_img"][1],
    )
    joint_hand_bbox = _coerce_frame_array(
        decoded_sample.get("joint_hand_bbox.npy"),
        num_frames,
        PER_FRAME_ARRAY_SPECS["joint_hand_bbox"][0],
        "joint_hand_bbox.npy",
        default_fill=PER_FRAME_ARRAY_SPECS["joint_hand_bbox"][1],
    )
    joint_cam = _coerce_frame_array(
        decoded_sample.get("joint_cam.npy"),
        num_frames,
        PER_FRAME_ARRAY_SPECS["joint_cam"][0],
        "joint_cam.npy",
        default_fill=PER_FRAME_ARRAY_SPECS["joint_cam"][1],
    )
    joint_rel = _coerce_frame_array(
        decoded_sample.get("joint_rel.npy"),
        num_frames,
        PER_FRAME_ARRAY_SPECS["joint_rel"][0],
        "joint_rel.npy",
        default_fill=PER_FRAME_ARRAY_SPECS["joint_rel"][1],
    )
    joint_2d_valid = _coerce_frame_array(
        decoded_sample.get("joint_2d_valid.npy"),
        num_frames,
        PER_FRAME_ARRAY_SPECS["joint_2d_valid"][0],
        "joint_2d_valid.npy",
        default_fill=None,
    )
    if joint_2d_valid is None:
        joint_2d_valid = joint_valid_legacy.copy()

    joint_3d_valid = _coerce_frame_array(
        decoded_sample.get("joint_3d_valid.npy"),
        num_frames,
        PER_FRAME_ARRAY_SPECS["joint_3d_valid"][0],
        "joint_3d_valid.npy",
        default_fill=None,
    )
    if joint_3d_valid is None:
        joint_3d_valid = joint_valid_legacy.copy()

    mano_pose = _coerce_frame_array(
        decoded_sample.get("mano_pose.npy"),
        num_frames,
        PER_FRAME_ARRAY_SPECS["mano_pose"][0],
        "mano_pose.npy",
        default_fill=PER_FRAME_ARRAY_SPECS["mano_pose"][1],
    )
    mano_shape = _coerce_frame_array(
        decoded_sample.get("mano_shape.npy"),
        num_frames,
        PER_FRAME_ARRAY_SPECS["mano_shape"][0],
        "mano_shape.npy",
        default_fill=PER_FRAME_ARRAY_SPECS["mano_shape"][1],
    )
    has_mano = _coerce_frame_array(
        decoded_sample.get("has_mano.npy"),
        num_frames,
        PER_FRAME_ARRAY_SPECS["has_mano"][0],
        "has_mano.npy",
        default_fill=None,
    )
    if has_mano is None:
        if mano_valid_legacy is not None:
            has_mano = mano_valid_legacy.copy()
        else:
            has_mano = _default_frame_array(
                num_frames, PER_FRAME_ARRAY_SPECS["has_mano"][0], 0.0
            )

    if mano_valid_legacy is None:
        mano_valid_legacy = has_mano.copy()

    timestamp = _coerce_frame_array(
        decoded_sample.get("timestamp.npy"),
        num_frames,
        PER_FRAME_ARRAY_SPECS["timestamp"][0],
        "timestamp.npy",
        default_fill=PER_FRAME_ARRAY_SPECS["timestamp"][1],
    )
    focal = _coerce_frame_array(
        decoded_sample.get("focal.npy"),
        num_frames,
        PER_FRAME_ARRAY_SPECS["focal"][0],
        "focal.npy",
        default_fill=PER_FRAME_ARRAY_SPECS["focal"][1],
    )
    princpt = _coerce_frame_array(
        decoded_sample.get("princpt.npy"),
        num_frames,
        PER_FRAME_ARRAY_SPECS["princpt"][0],
        "princpt.npy",
        default_fill=PER_FRAME_ARRAY_SPECS["princpt"][1],
    )

    assert hand_bbox is not None
    assert joint_img is not None
    assert joint_hand_bbox is not None
    assert joint_cam is not None
    assert joint_rel is not None
    assert joint_2d_valid is not None
    assert joint_3d_valid is not None
    assert mano_pose is not None
    assert mano_shape is not None
    assert has_mano is not None
    assert mano_valid_legacy is not None
    assert timestamp is not None
    assert focal is not None
    assert princpt is not None

    return {
        "__key__": sample_key,
        "num_frames": num_frames,
        "imgs_path": imgs_path,
        "imgs_bytes": imgs_bytes,
        "handedness": handedness,
        "additional_desc": additional_desc,
        "data_source": data_source,
        "source_split": source_split,
        "source_index": source_index,
        "intr_type": intr_type,
        "hand_bbox": hand_bbox,
        "joint_img": joint_img,
        "joint_hand_bbox": joint_hand_bbox,
        "joint_cam": joint_cam,
        "joint_rel": joint_rel,
        "joint_2d_valid": joint_2d_valid,
        "joint_3d_valid": joint_3d_valid,
        "joint_valid": joint_valid_legacy,
        "mano_pose": mano_pose,
        "mano_shape": mano_shape,
        "has_mano": has_mano,
        "mano_valid": mano_valid_legacy,
        "has_intr": has_intr,
        "timestamp": timestamp,
        "focal": focal,
        "princpt": princpt,
    }


def slice_normalized_clip_sample(
    clip_sample: Dict[str, Any],
    start: int,
    end: int,
    slice_index: Optional[int] = None,
) -> Dict[str, Any]:
    sub_sample: Dict[str, Any] = {
        "__key__": clip_sample["__key__"]
        if slice_index is None
        else f"{clip_sample['__key__']}_{slice_index:04d}",
        "num_frames": end - start,
    }

    for key in LIST_SLICE_KEYS:
        sub_sample[key] = clip_sample[key][start:end]

    for key in SCALAR_COPY_KEYS:
        sub_sample[key] = clip_sample[key]

    for key in PER_FRAME_ARRAY_SPECS:
        sub_sample[key] = clip_sample[key][start:end]

    return sub_sample
