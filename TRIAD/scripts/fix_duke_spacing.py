"""
Duke slice-spacing correction. run_segmentation.py takes SliceThickness as the z spacing; in 71/922 cases
the slice thickness is 2.2 mm but the actual slice spacing is 1.1 mm (overlapping reconstruction), so these
volumes would be stretched twofold in z.
Image and mask arrays are kept as they are (MedSAM2 segments slice by slice and does not use spacing); only the
header is rewritten with the actual spacing (median difference of consecutive ImagePositionPatient values,
read from duke_slice_spacing.csv) into <run>/data/duke_fixed/. Radiomic features of these cases are re-extracted,
and the two kinetic volume features are scaled by dz/thk.
Output: <run>/data/duke_radiomics_fixed.npz (same format as duke_feature_pool.npz) and duke_kinetic_fixed.npz.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import SimpleITK as sitk

from prism.feature_extraction import build_extractor
from prism.paths import DATA_ROOT, INTERIM_DIR, RUN_DATA

SRC = DATA_ROOT
FIX = RUN_DATA
V3 = INTERIM_DIR


def _one(args):
    pid, dz = args
    out = os.path.join(FIX, "duke_fixed")
    paths = []
    for kind in ("image", "mask"):
        im = sitk.ReadImage(f"{SRC}/seg_{kind}s/{pid}_{kind}.nii.gz")
        sx, sy, _ = im.GetSpacing()
        im.SetSpacing((sx, sy, float(dz)))
        p = os.path.join(out, f"{pid}_{kind}.nii.gz")
        sitk.WriteImage(im, p)
        paths.append(p)
    r = build_extractor().execute(*paths)
    return pid, {k: float(v) for k, v in r.items() if not k.startswith("diagnostics_")}


def main():
    os.makedirs(os.path.join(FIX, "duke_fixed"), exist_ok=True)
    sp = pd.read_csv(os.path.join(FIX, "duke_slice_spacing.csv")).set_index("pid")
    bad = sp[(sp.dz - sp.thk).abs() > 1e-3]
    print(f"{len(bad)} cases with slice spacing != slice thickness: {bad.groupby(['thk', 'dz']).size().to_dict()}")
    pool = np.load(f"{SRC}/duke_feature_pool.npz", allow_pickle=True)
    names = list(pool["feature_names"])
    X = pool["X_pool"].copy()
    idx = {p: i for i, p in enumerate(pool["patient_ids"])}
    with ProcessPoolExecutor(24) as ex:
        for pid, feats in ex.map(_one, list(zip(bad.index, bad.dz))):
            X[idx[pid]] = [feats[n] for n in names]
    np.savez(os.path.join(FIX, "duke_radiomics_fixed.npz"), X_pool=X, feature_names=pool["feature_names"],
             feature_to_filter_index=pool["feature_to_filter_index"], n_filters=pool["n_filters"],
             patient_ids=pool["patient_ids"], y=pool["y"], fixed_patients=np.array(bad.index))
    k = np.load(f"{V3}/duke_kinetic.npz", allow_pickle=True)
    K = k["X"].copy()
    kn = list(k["feature_names"])
    for pid in bad.index:
        f = bad.loc[pid, "dz"] / bad.loc[pid, "thk"]
        for n in ("kinetic_summary_EnhancingVolume", "kinetic_summary_WashoutVolume"):
            K[list(k["patient_ids"]).index(pid), kn.index(n)] *= f
    np.savez(os.path.join(FIX, "duke_kinetic_fixed.npz"), X=K, feature_names=k["feature_names"],
             patient_ids=k["patient_ids"], status=k["status"], late_phase=k["late_phase"])
    vol = names.index("original_shape_VoxelVolume")
    rows = [idx[p] for p in bad.index]
    print("VoxelVolume ratio fixed/old (median):", np.median(X[rows, vol] / pool["X_pool"][rows, vol]))


if __name__ == "__main__":
    main()
