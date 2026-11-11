"""
MedSAM2 segmentation of I-SPY cases along the acquired slices.
Input: output of prepare_ispy_fixed.py --stage native (native early post-contrast images (slice, row, column) + prompt_boxes.json).
Settings are identical to Duke: MRI_LiverLesion checkpoint, percentile normalisation, key-slice box prompt,
bidirectional propagation, largest connected component.
"""
import argparse
import json
import os

import numpy as np
import SimpleITK as sitk

from prism.paths import MEDSAM2_CFG, MEDSAM2_CKPT, RUN_DATA
from prism.segmentation import build_predictor, bbox_row_col_to_xyxy, mri_normalize_to_uint8, segment_tumor_3d

FIX = RUN_DATA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["ISPY1", "ISPY2"])
    a = ap.parse_args()
    nat = os.path.join(FIX, a.cohort, "native")
    out = os.path.join(FIX, a.cohort, "native_medsam2")
    os.makedirs(out, exist_ok=True)
    boxes = json.load(open(os.path.join(nat, "prompt_boxes.json")))
    predictor = build_predictor(
        checkpoint=MEDSAM2_CKPT,
        model_cfg=MEDSAM2_CFG,
    )
    for i, (pid, b) in enumerate(sorted(boxes.items())):
        dst = os.path.join(out, f"{pid}_mask.nii.gz")
        if os.path.exists(dst) or b is None:
            if b is None:
                print(f"[{pid}] SKIP: empty official mask", flush=True)
            continue
        vol = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(nat, f"{pid}_early.nii.gz")))
        box = bbox_row_col_to_xyxy(b["start_row"], b["end_row"], b["start_col"], b["end_col"])
        key = (b["start_slice"] + b["end_slice"]) // 2
        mask = segment_tumor_3d(predictor, mri_normalize_to_uint8(vol), box, key)
        sitk.WriteImage(sitk.GetImageFromArray(mask.astype(np.uint8)), dst)
        if (i + 1) % 50 == 0 or mask.sum() == 0:
            print(f"[{i + 1}/{len(boxes)}] {pid}: mask voxels {int(mask.sum())}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
