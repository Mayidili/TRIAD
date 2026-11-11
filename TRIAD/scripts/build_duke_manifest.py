"""
Match the three official Duke-Breast-Cancer-MRI tables (Annotation_Boxes.xlsx with the tumour boxes,
Clinical_and_Other_Features.xlsx with HR/HER2 status, Breast-Cancer-MRI-filepath_filename-mapping.xlsx
with the series naming) to the DICOM directories downloaded with download_tcia.py, and write a manifest
CSV with one row per patient:

  patient_id, series_uid, series_dir, start_row, end_row, start_col, end_col,
  start_slice, end_slice, er, pr, her2, hr_label, her2_label_binary

The box coordinates refer to the post_1 (first post-contrast) series, so only the post_1
SeriesInstanceUID / local directory of each patient is kept.

Cases with HER2 = 2 (borderline) are excluded by default from the binary HER2 label (the usual handling
of IHC 2+ cases); --keep-her2-borderline keeps them.
"""
import argparse
import os
import re

import pandas as pd


def load_annotation_boxes(path):
    df = pd.read_excel(path)
    df = df.rename(columns={
        "Patient ID": "patient_id",
        "Start Row": "start_row", "End Row": "end_row",
        "Start Column": "start_col", "End Column": "end_col",
        "Start Slice": "start_slice", "End Slice": "end_slice",
    })
    return df


def load_clinical(path):
    df = pd.read_excel(path, header=[0, 1, 2])
    df.columns = [c[0] for c in df.columns]  # the top header level is not unique, so columns are taken by position below

    raw = pd.read_excel(path, header=[0, 1, 2])
    out = pd.DataFrame()
    out["patient_id"] = raw.iloc[:, 0]
    out["er"] = raw.iloc[:, 23]
    out["pr"] = raw.iloc[:, 24]
    out["her2"] = raw.iloc[:, 25]
    return out


def load_post1_series_dirs(mapping_path, downloaded_root):
    """
    Select the post_1 rows of filepath_mapping.xlsx, parse the SeriesInstanceUID from classic_path and
    check that the series exists in the download directory (downloaded_root/{patient_id}/{series_uid}).
    """
    df = pd.read_excel(mapping_path)
    is_post1 = df["original_path_and_filename"].astype(str).str.contains(r"/post_1/", regex=True)
    df = df[is_post1].copy()

    def parse_patient_and_series(classic_path):
        # e.g. Duke-Breast-Cancer-MRI/Breast_MRI_001/{StudyUID}/{SeriesUID}/1-001.dcm
        parts = str(classic_path).split("/")
        if len(parts) < 4:
            return None, None
        patient_id = parts[1]
        series_uid = parts[3]
        return patient_id, series_uid

    parsed = df["classic_path"].apply(parse_patient_and_series)
    df["patient_id"] = [p[0] for p in parsed]
    df["series_uid"] = [p[1] for p in parsed]

    per_patient = df.groupby("patient_id")["series_uid"].agg(lambda s: s.value_counts().idxmax())
    result = per_patient.reset_index()
    result["series_dir"] = result.apply(
        lambda r: os.path.join(downloaded_root, r["patient_id"], r["series_uid"]), axis=1
    )
    result["downloaded"] = result["series_dir"].apply(os.path.isdir)
    return result


def build(annotation_boxes_path, clinical_path, mapping_path, downloaded_root, out_csv,
          keep_her2_borderline=False):
    boxes = load_annotation_boxes(annotation_boxes_path)
    clinical = load_clinical(clinical_path)
    series = load_post1_series_dirs(mapping_path, downloaded_root)

    manifest = boxes.merge(clinical, on="patient_id", how="inner").merge(series, on="patient_id", how="inner")

    manifest["hr_label"] = ((manifest["er"] == 1) | (manifest["pr"] == 1)).astype(int)

    if not keep_her2_borderline:
        n_before = len(manifest)
        manifest = manifest[manifest["her2"].isin([0, 1])].copy()
        print(f"excluded HER2 borderline (2) cases: {n_before} -> {len(manifest)}")
    manifest["her2_label_binary"] = manifest["her2"].astype(int)

    n_downloaded = manifest["downloaded"].sum()
    print(f"manifest cases: {len(manifest)}, post_1 downloaded: {n_downloaded}, not downloaded: {len(manifest) - n_downloaded}")
    print(f"HR positive: {manifest['hr_label'].sum()} / {len(manifest)}")
    print(f"HER2 positive: {(manifest['her2_label_binary']==1).sum()} / {len(manifest)}")

    manifest.to_csv(out_csv, index=False)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-boxes", required=True)
    parser.add_argument("--clinical", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--downloaded-root", required=True, help="--out-dir of download_tcia.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--keep-her2-borderline", action="store_true")
    args = parser.parse_args()

    build(args.annotation_boxes, args.clinical, args.mapping, args.downloaded_root, args.out,
          args.keep_her2_borderline)
