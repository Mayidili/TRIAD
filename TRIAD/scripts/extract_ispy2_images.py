"""
Extract, from the 14 I-SPY2 DCE tar.gz archives, only the "post_early" phase listed in the BreastDCEDL metadata
for each patient (the phase used as the counterpart of Duke post_1), together with the masks, without unpacking
the whole archives.

Usage:
  python3 extract_ispy2_images.py --ispy2-dir /path/to/ISPY2 --out-dir /path/to/extracted
"""
import argparse
import os
import tarfile

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ispy2-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "masks"), exist_ok=True)

    meta = pd.read_csv(os.path.join(args.ispy2_dir, "BreastDCEDL_ISPY2_metadata.csv"))
    meta["post_early"] = meta["post_early"].astype(int)
    want_image = {}  # patient_id -> expected member basename
    for _, row in meta.iterrows():
        pid = row["pid"]
        want_image[pid] = f"{pid}_spy2_vis1_dce_aqc_{row['post_early']}.nii.gz"

    nopcr_parts = [f"BreastDCEDL_ISPY2_nopcr_dce_part{i}.tar.gz" for i in range(1, 9)]
    pcr_parts = [f"BreastDCEDL_ISPY2_pcr_dce_part{i}.tar.gz" for i in range(1, 7)]
    all_parts = nopcr_parts + pcr_parts

    found = {}
    for part in all_parts:
        path = os.path.join(args.ispy2_dir, part)
        if not os.path.exists(path):
            print(f"[skip] {part} not found")
            continue
        print(f"scanning {part} ...", flush=True)
        n_extracted_this_part = 0
        with tarfile.open(path, "r:gz") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                base = os.path.basename(member.name)
                # member.name looks like .../ACRIN-6698-102212/ACRIN-6698-102212_spy2_vis1_dce_aqc_2.nii.gz
                pid_dir = os.path.basename(os.path.dirname(member.name))
                if pid_dir in want_image and base == want_image[pid_dir]:
                    out_path = os.path.join(args.out_dir, "images", f"{pid_dir}_image.nii.gz")
                    with tf.extractfile(member) as src, open(out_path, "wb") as dst:
                        dst.write(src.read())
                    found[pid_dir] = out_path
                    n_extracted_this_part += 1
        print(f"  -> extracted {n_extracted_this_part} images from {part}, cumulative {len(found)}/{len(want_image)}",
              flush=True)

    missing = sorted(set(want_image.keys()) - set(found.keys()))
    print(f"\nTotal images extracted: {len(found)} / {len(want_image)}")
    if missing:
        print(f"Missing ({len(missing)}): {missing[:20]}{'...' if len(missing) > 20 else ''}")

    # masks: single small archive, flat layout, filename = "{pid}_{pid}_spy2_vis1_mask.nii.gz"
    mask_tar = os.path.join(args.ispy2_dir, "BreastDCEDL_ISPY2_mask.tar.gz")
    n_masks = 0
    with tarfile.open(mask_tar, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            base = os.path.basename(member.name)
            if not base.endswith("_spy2_vis1_mask.nii.gz"):
                continue
            prefix = base[: -len("_spy2_vis1_mask.nii.gz")]
            assert len(prefix) % 2 == 1 and prefix[: len(prefix) // 2] == prefix[len(prefix) // 2 + 1:], (
                f"mask file name prefix is not of the form '{{pid}}_{{pid}}': {base}"
            )
            pid = prefix[: len(prefix) // 2]
            out_path = os.path.join(args.out_dir, "masks", f"{pid}_mask.nii.gz")
            with tf.extractfile(member) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            n_masks += 1
    print(f"Total masks extracted: {n_masks}")


if __name__ == "__main__":
    main()
