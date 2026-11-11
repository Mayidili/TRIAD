"""
DCE kinetic features of the 922 Duke cases (definitions in prism/kinetic.py).

Phases: S0 = pre, S1 = post_1, SL = the patient's last post-contrast phase (highest of post_4/post_3/post_2).
Slice order: sorted by the index of the original file name in filepath_mapping (the order in which the
bounding boxes and MedSAM2 masks are defined).
Alignment check: post_1 read in this order must match the volume passed to MedSAM2 (seg_images) voxel for
voxel; otherwise the case is flagged post1_mismatch and its features are set to NaN (median-imputed within folds downstream).

Usage: python3 extract_kinetic_duke.py --out results/interim/duke_kinetic.npz --n-proc 32
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk

from prism.kinetic import FEATURE_NAMES, kinetic_features
from prism.paths import DATA_ROOT, INTERIM_DIR

DUKE_ROOT = os.path.join(DATA_ROOT, "Duke-Breast-Cancer-MRI")
SEG_IMAGES = os.path.join(DATA_ROOT, "seg_images")
SEG_MASKS = os.path.join(DATA_ROOT, "seg_masks")
POOL_NPZ = os.path.join(DATA_ROOT, "duke_feature_pool.npz")
MAPPING = os.path.join(INTERIM_DIR, "duke_mapping.pkl")

_MAP = None


def _init():
    global _MAP
    _MAP = pd.read_pickle(MAPPING)


def load_series(pid, seq, sub):
    s = sub[sub.seq == seq]
    suid = s.series_uid.value_counts().idxmax()
    d = os.path.join(DUKE_ROOT, pid, suid)
    idx = dict(zip(s.sop, s.idx))
    ds = [pydicom.dcmread(os.path.join(d, f)) for f in os.listdir(d) if f.endswith(".dcm")]
    order = np.argsort([idx[x.SOPInstanceUID] for x in ds])
    return np.stack([ds[i].pixel_array.astype(np.float32) for i in order])


def one(pid):
    try:
        sub = _MAP[_MAP.pid == pid]
        seqs = set(sub.seq)
        late = next((q for q in ["post_4", "post_3", "post_2"] if q in seqs), None)
        if late is None or "pre" not in seqs or "post_1" not in seqs:
            return pid, np.full(len(FEATURE_NAMES), np.nan), "missing_phase", late
        s1 = load_series(pid, "post_1", sub)
        img = sitk.ReadImage(os.path.join(SEG_IMAGES, f"{pid}_image.nii.gz"))
        if not np.array_equal(sitk.GetArrayFromImage(img).astype(np.float32), s1):
            return pid, np.full(len(FEATURE_NAMES), np.nan), "post1_mismatch", late
        s0 = load_series(pid, "pre", sub)
        sl = load_series(pid, late, sub)
        if not (s0.shape == s1.shape == sl.shape):
            return pid, np.full(len(FEATURE_NAMES), np.nan), f"shape_mismatch {s0.shape} {s1.shape} {sl.shape}", late
        mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(SEG_MASKS, f"{pid}_mask.nii.gz"))) > 0
        vox = float(np.prod(img.GetSpacing()))
        return pid, kinetic_features(s0, s1, sl, mask, vox), "ok", late
    except Exception as e:  # noqa: BLE001
        return pid, np.full(len(FEATURE_NAMES), np.nan), f"error {type(e).__name__}: {e}", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-proc", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    pids = list(np.load(POOL_NPZ)["patient_ids"])[: args.limit]
    with Pool(args.n_proc, initializer=_init) as pool:
        res = []
        for i, r in enumerate(pool.imap(one, pids, chunksize=2)):
            res.append(r)
            if r[2] != "ok":
                print(f"[{r[0]}] {r[2]}", flush=True)
            if (i + 1) % 50 == 0:
                print(f"progress {i + 1}/{len(pids)}", flush=True)
    X = np.stack([r[1] for r in res])
    status = np.array([r[2] for r in res])
    np.savez(args.out, X=X, feature_names=np.array(FEATURE_NAMES), patient_ids=np.array([r[0] for r in res]),
             status=status, late_phase=np.array([str(r[3]) for r in res]))
    print(pd.Series(status).str.split().str[0].value_counts())
    print(pd.Series([r[3] for r in res]).value_counts(dropna=False))


if __name__ == "__main__":
    main()
