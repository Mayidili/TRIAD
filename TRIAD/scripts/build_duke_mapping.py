"""Index of the Duke DICOM files: SOP instance UID -> patient, sequence, slice index and series UID.

Built from the collection's file path mapping table (filepath_mapping.xlsx on TCIA).

Usage:
  python scripts/build_duke_mapping.py --mapping data/refs/filepath_mapping.xlsx
"""
import argparse
import os

import pandas as pd

from prism.paths import INTERIM_DIR


def load_mapping(path):
    m = pd.read_excel(path, usecols=["sop_instance_UID", "original_path_and_filename", "classic_path"])
    parts = m.original_path_and_filename.str.split("/")
    return pd.DataFrame({"sop": m.sop_instance_UID, "pid": parts.str[1], "seq": parts.str[2],
                         "idx": parts.str[3].str.extract(r"_(\d+)\.dcm$")[0].astype(float),
                         "series_uid": m.classic_path.str.split("/").str[3]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--out", default=os.path.join(INTERIM_DIR, "duke_mapping.pkl"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df = load_mapping(a.mapping)
    df.to_pickle(a.out)
    print(f"{len(df)} files, {df.pid.nunique()} patients -> {a.out}")


if __name__ == "__main__":
    main()
