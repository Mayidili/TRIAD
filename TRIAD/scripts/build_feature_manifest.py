"""
Join the successfully segmented Duke cases (status ok in seg_results.csv, with image and mask files) with the
HR labels of duke_manifest.csv, and write the manifest expected by feature_extraction.build_corpus()
(columns: patient_id, image_path, mask_path, label).

The label is HR status (ER+ or PR+); HER2 positivity (17.7%) is too rare for stable AUC estimates.
Cases with an empty MedSAM2 mask are excluded because PyRadiomics cannot compute shape/texture features on them.
"""
import argparse
import os

import pandas as pd


def build_feature_manifest(duke_manifest_csv, seg_results_csv, image_dir, mask_dir, out_csv, label_col="hr_label"):
    duke = pd.read_csv(duke_manifest_csv)
    seg = pd.read_csv(seg_results_csv)

    ok = seg[seg["status"] == "ok"][["patient_id"]]
    merged = ok.merge(duke[["patient_id", label_col]], on="patient_id", how="inner")

    rows = []
    n_missing_files = 0
    for _, row in merged.iterrows():
        pid = row["patient_id"]
        image_path = os.path.join(image_dir, f"{pid}_image.nii.gz")
        mask_path = os.path.join(mask_dir, f"{pid}_mask.nii.gz")
        if not (os.path.exists(image_path) and os.path.exists(mask_path)):
            n_missing_files += 1
            continue
        rows.append({
            "patient_id": pid,
            "image_path": image_path,
            "mask_path": mask_path,
            "label": int(row[label_col]),
        })

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_csv, index=False)
    print(f"feature manifest: {len(manifest)} cases written to {out_csv} ({n_missing_files} skipped with missing files)")
    print(f"label distribution: {manifest['label'].value_counts().to_dict()}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duke-manifest", required=True)
    parser.add_argument("--seg-results", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label-col", default="hr_label")
    args = parser.parse_args()
    build_feature_manifest(
        args.duke_manifest, args.seg_results, args.image_dir, args.mask_dir, args.out, args.label_col
    )
