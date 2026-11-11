"""
Voxel-level tumour masks from the Duke 3D bounding-box annotations with MedSAM2.

The box [x_min, y_min, x_max, y_max] on the middle slice of the annotation ((start_slice+end_slice)//2)
is used as the prompt, and the SAM2 video predictor (NPZ interface) propagates forward and in reverse;
the two directions are merged. This center-outward strategy performed best on breast MRI in
"Towards Affordable Tumor Segmentation and Visualization for 3D Breast MRI Using SAM2"
(MICCAI 2025 DeepBreath workshop) and is also the mode of the MedSAM2 CT reference script
(medsam2_infer_3D_CT.py).

Main difference from the CT script: MRI has no absolute intensity scale, so intensities are clipped
at percentiles and min-max scaled to [0, 255] instead of using a fixed CT window.
"""
import os

import numpy as np
import pydicom
from skimage import measure

VIDEO_HEIGHT = VIDEO_WIDTH = None  # set at run time from the DICOM size
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


def load_dicom_series(series_dir):
    """Read all DICOM files of a series sorted by InstanceNumber; returns a (D,H,W) float array.
    spacing is returned in SimpleITK (x,y,z) order (DICOM PixelSpacing is [row spacing (y), column
    spacing (x)], the reverse of sitk), for SetSpacing when saving NIfTI."""
    files = [os.path.join(series_dir, f) for f in os.listdir(series_dir) if f.lower().endswith(".dcm")]
    datasets = [pydicom.dcmread(f) for f in files]
    datasets.sort(key=lambda d: int(getattr(d, "InstanceNumber", 0)))
    volume = np.stack([d.pixel_array.astype(np.float32) for d in datasets], axis=0)
    spacing_xyz = (1.0, 1.0, 1.0)
    if datasets and hasattr(datasets[0], "PixelSpacing"):
        ps = datasets[0].PixelSpacing  # [row_spacing(y), col_spacing(x)]
        thickness = float(getattr(datasets[0], "SliceThickness", 1.0))
        spacing_xyz = (float(ps[1]), float(ps[0]), thickness)
    return volume, spacing_xyz


def to_sitk_image(array, spacing_xyz):
    """A (D,H,W) array maps to sitk (z,y,x) order (standard GetImageFromArray convention)."""
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(array)
    img.SetSpacing(spacing_xyz)
    return img


def mri_normalize_to_uint8(volume, lower_pct=0.5, upper_pct=99.5):
    """Percentile clipping and min-max scaling to [0, 255] (MRI has no fixed window like CT)."""
    lo, hi = np.percentile(volume, [lower_pct, upper_pct])
    clipped = np.clip(volume, lo, hi)
    span = max(hi - lo, 1e-6)
    scaled = (clipped - lo) / span * 255.0
    return scaled.astype(np.uint8)


def resize_grayscale_to_rgb_and_resize(volume_uint8, image_size=512):
    """(D,H,W) uint8 grey -> (D,3,image_size,image_size) RGB, as in the MedSAM2 reference script."""
    from PIL import Image

    d, h, w = volume_uint8.shape
    out = np.zeros((d, 3, image_size, image_size), dtype=np.float32)
    for i in range(d):
        img = Image.fromarray(volume_uint8[i]).convert("RGB").resize((image_size, image_size))
        out[i] = np.array(img).transpose(2, 0, 1)
    return out


def bbox_row_col_to_xyxy(start_row, end_row, start_col, end_col):
    """Duke Annotation_Boxes use row=Y and column=X; convert to SAM2 [x_min, y_min, x_max, y_max]."""
    return np.array([start_col, start_row, end_col, end_row])


def segment_tumor_3d(predictor, volume_uint8, bbox_xyxy, key_slice_idx, image_size=512, device="cuda"):
    """
    Center-outward propagation: start at key_slice_idx, propagate in both directions and merge.
    Returns a (D,H,W) uint8 binary mask at the original resolution.
    """
    import torch

    d, h, w = volume_uint8.shape
    img_resized = resize_grayscale_to_rgb_and_resize(volume_uint8, image_size) / 255.0
    img_resized = torch.from_numpy(img_resized).to(device)
    mean = torch.tensor(IMG_MEAN, dtype=torch.float32)[:, None, None].to(device)
    std = torch.tensor(IMG_STD, dtype=torch.float32)[:, None, None].to(device)
    img_resized = (img_resized - mean) / std

    segs_3d = np.zeros((d, h, w), dtype=np.uint8)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = predictor.init_state(img_resized, h, w)
        predictor.add_new_points_or_box(
            inference_state=inference_state, frame_idx=key_slice_idx, obj_id=1, box=bbox_xyxy
        )
        for out_frame_idx, _out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            segs_3d[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
        predictor.reset_state(inference_state)

        predictor.add_new_points_or_box(
            inference_state=inference_state, frame_idx=key_slice_idx, obj_id=1, box=bbox_xyxy
        )
        for out_frame_idx, _out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
            segs_3d[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
        predictor.reset_state(inference_state)

    if segs_3d.max() > 0:
        labels = measure.label(segs_3d)
        counts = np.bincount(labels.flat)
        counts[0] = 0  # ignore background
        largest = np.argmax(counts)
        segs_3d = (labels == largest).astype(np.uint8)

    return segs_3d


def build_predictor(checkpoint, model_cfg):
    from sam2.build_sam import build_sam2_video_predictor_npz

    return build_sam2_video_predictor_npz(model_cfg, checkpoint)


def segment_case(predictor, series_dir, start_row, end_row, start_col, end_col, start_slice, end_slice):
    volume, spacing = load_dicom_series(series_dir)
    volume_u8 = mri_normalize_to_uint8(volume)
    bbox_xyxy = bbox_row_col_to_xyxy(start_row, end_row, start_col, end_col)
    key_slice_idx = (int(start_slice) + int(end_slice)) // 2
    mask = segment_tumor_3d(predictor, volume_u8, bbox_xyxy, key_slice_idx)
    return mask, volume, spacing
