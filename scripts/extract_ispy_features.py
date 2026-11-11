"""
Radiomic and kinetic features on the geometry-corrected I-SPY data (output of prepare_ispy_fixed.py --stage finalize).
Same extractor as Duke. Output: <run>/data/{ispy1,ispy2}_{radiomics,kinetic}_{medsam2,official}.npz
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import SimpleITK as sitk

from prism.feature_extraction import build_extractor
from prism.kinetic import FEATURE_NAMES as KIN_NAMES, kinetic_features
from prism.paths import DATA_ROOT, RUN_DATA

FIX = RUN_DATA
DUKE_POOL = os.path.join(DATA_ROOT, "duke_feature_pool.npz")


def _one(args):
    cohort, mk, pid = args
    d = os.path.join(FIX, cohort)
    mp = os.path.join(d, f"mask_{mk}", f"{pid}.nii.gz")
    out = {"pid": pid, "rad": None, "kin": None, "err": None}
    try:
        if not os.path.exists(mp):
            out["err"] = "no_mask"
            return out
        m = sitk.GetArrayFromImage(sitk.ReadImage(mp)) > 0
        if not m.any():
            out["err"] = "empty_mask"
            return out
        early = os.path.join(d, "early", f"{pid}.nii.gz")
        r = build_extractor().execute(early, mp)
        out["rad"] = {k: float(v) for k, v in r.items() if not k.startswith("diagnostics_")}
        im = sitk.ReadImage(early)
        s1 = sitk.GetArrayFromImage(im).astype(np.float64)
        s0 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(d, "pre", f"{pid}.nii.gz"))).astype(np.float64)
        sl = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(d, "late", f"{pid}.nii.gz"))).astype(np.float64)
        out["kin"] = kinetic_features(s0, s1, sl, m, float(np.prod(im.GetSpacing())))
    except Exception as e:  # noqa: BLE001
        out["err"] = f"{type(e).__name__}: {e}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["ISPY1", "ISPY2"])
    ap.add_argument("--mask", required=True, choices=["medsam2", "official"])
    ap.add_argument("--n-proc", type=int, default=16)
    a = ap.parse_args()
    names = list(np.load(DUKE_POOL)["feature_names"])
    pids = sorted(f[: -len(".nii.gz")] for f in os.listdir(os.path.join(FIX, a.cohort, "early")))
    with ProcessPoolExecutor(a.n_proc) as ex:
        res = list(ex.map(_one, [(a.cohort, a.mask, p) for p in pids], chunksize=2))
    R = np.full((len(res), len(names)), np.nan)
    K = np.full((len(res), len(KIN_NAMES)), np.nan)
    status = []
    for i, r in enumerate(res):
        if r["err"] is None:
            R[i] = [r["rad"][n] for n in names]
            K[i] = r["kin"]
            status.append("ok")
        else:
            status.append(r["err"])
            print(f"[{r['pid']}] {r['err']}", flush=True)
    ids = np.array([r["pid"] for r in res])
    tag = f"{a.cohort.lower()}_%s_{a.mask}.npz"
    np.savez(os.path.join(FIX, tag % "radiomics"), X=R, feature_names=np.array(names), patient_ids=ids, status=np.array(status))
    np.savez(os.path.join(FIX, tag % "kinetic"), X=K, feature_names=np.array(KIN_NAMES), patient_ids=ids, status=np.array(status))
    print(f"{a.cohort} {a.mask}: ok {status.count('ok')}/{len(status)}")


if __name__ == "__main__":
    main()
