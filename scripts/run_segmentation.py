"""
MedSAM2 segmentation of all downloaded Duke cases; writes voxel-level masks for radiomics extraction.
Resumable: cases whose mask file already exists are skipped.
"""
import csv
import os
import sys
import time

import pandas as pd
import SimpleITK as sitk

from prism.paths import DATA_ROOT, MEDSAM2_CFG, MEDSAM2_CKPT
from prism.segmentation import build_predictor, segment_case, to_sitk_image

MANIFEST_PATH = os.path.join(DATA_ROOT, "duke_manifest.csv")
MASK_DIR = os.path.join(DATA_ROOT, "seg_masks")
IMAGE_DIR = os.path.join(DATA_ROOT, "seg_images")
RESULTS_CSV = os.path.join(DATA_ROOT, "seg_results.csv")

os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

manifest = pd.read_csv(MANIFEST_PATH)
downloaded = manifest[manifest["downloaded"]].reset_index(drop=True)
print(f"{len(downloaded)} cases available for segmentation", flush=True)

predictor = build_predictor(
    checkpoint=MEDSAM2_CKPT,
    model_cfg=MEDSAM2_CFG,
)

existing_results = {}
if os.path.exists(RESULTS_CSV):
    with open(RESULTS_CSV, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            existing_results[row["patient_id"]] = row

fieldnames = ["patient_id", "status", "mask_voxels", "n_slices_with_mask", "bbox_n_slices", "error"]
write_header = not os.path.exists(RESULTS_CSV)
out_f = open(RESULTS_CSV, "a", newline="", encoding="utf-8")
writer = csv.DictWriter(out_f, fieldnames=fieldnames)
if write_header:
    writer.writeheader()

n_done = n_skipped = n_failed = 0
t0 = time.time()
for i, row in downloaded.iterrows():
    pid = row["patient_id"]
    mask_path = os.path.join(MASK_DIR, f"{pid}_mask.nii.gz")
    image_path = os.path.join(IMAGE_DIR, f"{pid}_image.nii.gz")

    if pid in existing_results and existing_results[pid]["status"] in ("ok", "empty_mask") and os.path.exists(mask_path) and os.path.exists(image_path):
        n_skipped += 1
        continue

    try:
        mask, volume, spacing_xyz = segment_case(
            predictor, row["series_dir"],
            row["start_row"], row["end_row"], row["start_col"], row["end_col"],
            row["start_slice"], row["end_slice"],
        )
        n_voxels = int(mask.sum())
        n_slices_with_mask = int((mask.sum(axis=(1, 2)) > 0).sum())
        bbox_n_slices = int(row["end_slice"] - row["start_slice"] + 1)

        sitk.WriteImage(to_sitk_image(mask, spacing_xyz), mask_path)
        sitk.WriteImage(to_sitk_image(volume.astype("float32"), spacing_xyz), image_path)
        status = "ok" if n_voxels > 0 else "empty_mask"
        writer.writerow({
            "patient_id": pid, "status": status, "mask_voxels": n_voxels,
            "n_slices_with_mask": n_slices_with_mask, "bbox_n_slices": bbox_n_slices, "error": "",
        })
        n_done += 1
        if status == "empty_mask":
            print(f"[{pid}] WARNING: empty mask", flush=True)
    except Exception as e:  # noqa: BLE001
        writer.writerow({
            "patient_id": pid, "status": "failed", "mask_voxels": "", "n_slices_with_mask": "",
            "bbox_n_slices": "", "error": f"{type(e).__name__}: {e}",
        })
        n_failed += 1
        print(f"[{pid}] FAILED: {type(e).__name__}: {e}", flush=True)

    out_f.flush()

    if (n_done + n_failed) % 20 == 0 and (n_done + n_failed) > 0:
        elapsed = time.time() - t0
        rate = (n_done + n_failed) / elapsed
        print(f"progress: {i+1}/{len(downloaded)} done={n_done} failed={n_failed} skipped={n_skipped} "
              f"({rate:.2f} cases/s, elapsed {elapsed/60:.1f}min)", flush=True)

out_f.close()
print(f"\nFINISHED: done={n_done} failed={n_failed} skipped={n_skipped} total={len(downloaded)}", flush=True)
