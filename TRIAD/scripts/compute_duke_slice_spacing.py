"""Slice thickness and actual slice spacing of the Duke first post-contrast series.

Some Duke series are reconstructed with overlapping slices, so the spacing between slices is smaller than
SliceThickness. The spacing is the median distance between adjacent ImagePositionPatient values along the
slice normal. Output columns: pid, thk, dz, ratio (= dz / thk).

Usage:
  python scripts/compute_duke_slice_spacing.py --n-proc 16
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pydicom

from prism.paths import DATA_ROOT, INTERIM_DIR, RUN_DATA

DICOM_ROOT = os.path.join(DATA_ROOT, "Duke-Breast-Cancer-MRI")


def spacing(args):
    pid, series_uid = args
    d = os.path.join(DICOM_ROOT, pid, series_uid)
    ds = [pydicom.dcmread(os.path.join(d, f), stop_before_pixels=True) for f in os.listdir(d) if f.endswith(".dcm")]
    iop = np.array(ds[0].ImageOrientationPatient, float)
    normal = np.cross(iop[:3], iop[3:])
    pos = np.sort([np.dot(normal, np.array(x.ImagePositionPatient, float)) for x in ds])
    thk = float(ds[0].SliceThickness)
    dz = float(np.median(np.diff(pos)))
    return pid, thk, dz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default=os.path.join(INTERIM_DIR, "duke_mapping.pkl"))
    ap.add_argument("--out", default=os.path.join(RUN_DATA, "duke_slice_spacing.csv"))
    ap.add_argument("--n-proc", type=int, default=8)
    a = ap.parse_args()
    m = pd.read_pickle(a.mapping)
    series = m[m.seq == "post_1"].drop_duplicates("pid")[["pid", "series_uid"]]
    with ProcessPoolExecutor(a.n_proc) as ex:
        rows = list(ex.map(spacing, series.itertuples(index=False, name=None)))
    df = pd.DataFrame(rows, columns=["pid", "thk", "dz"]).sort_values("pid")
    df["ratio"] = df.dz / df.thk
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} patients, {(np.abs(df.dz - df.thk) > 1e-3).sum()} with spacing != thickness -> {a.out}")


if __name__ == "__main__":
    main()
