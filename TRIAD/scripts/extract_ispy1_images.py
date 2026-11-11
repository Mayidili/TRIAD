"""
Extract, for each I-SPY1 patient, acq1 and the tumour mask from BreastDCEDL_spy1.zip without unpacking the
whole archive. acq1 is taken as the first post-contrast acquisition (matching Duke post_1); unlike I-SPY2, the
I-SPY1 metadata has no explicit phase-index column.

Usage:
  python3 extract_ispy1_images.py --zip /path/to/BreastDCEDL_spy1.zip --out-dir /path/to/extracted
"""
import argparse
import io
import os
import re
import zipfile

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "masks"), exist_ok=True)

    z = zipfile.ZipFile(args.zip)
    with z.open("BreastDCEDL_spy1/BreastDCEDL_spy1_metadata.csv") as f:
        meta = pd.read_csv(f)
    meta_pids = set(meta["pid"])

    names = z.namelist()
    dce_re = re.compile(r".*/(ISPY1_\d+)_spy1_vis1_acq(\d+)\.nii\.gz")
    mask_re = re.compile(r".*/(ISPY1_\d+)_spy1_vis1_mask\.nii\.gz")

    dce_by_patient = {}
    for n in names:
        m = dce_re.match(n)
        if m and int(m.group(2)) == 1:  # acq1 = first post-contrast
            dce_by_patient[m.group(1)] = n

    mask_by_patient = {}
    for n in names:
        m = mask_re.match(n)
        if m:
            mask_by_patient[m.group(1)] = n

    n_extracted = 0
    n_no_acq1 = 0
    n_no_mask = 0
    n_not_in_meta = 0
    for pid in sorted(set(dce_by_patient) | set(mask_by_patient)):
        if pid not in meta_pids:
            n_not_in_meta += 1
            continue
        if pid not in dce_by_patient:
            n_no_acq1 += 1
            continue
        if pid not in mask_by_patient:
            n_no_mask += 1
            continue
        with z.open(dce_by_patient[pid]) as src, open(
            os.path.join(args.out_dir, "images", f"{pid}_image.nii.gz"), "wb"
        ) as dst:
            dst.write(src.read())
        with z.open(mask_by_patient[pid]) as src, open(
            os.path.join(args.out_dir, "masks", f"{pid}_mask.nii.gz"), "wb"
        ) as dst:
            dst.write(src.read())
        n_extracted += 1

    print(f"Extracted {n_extracted} patients (image+mask)")
    print(f"Skipped: {n_no_acq1} no acq1 file, {n_no_mask} no mask file, {n_not_in_meta} not in metadata")


if __name__ == "__main__":
    main()
